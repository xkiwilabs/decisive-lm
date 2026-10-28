"""Label rows with a teacher on a vLLM server (docs/guide/12), resumably, then calibrate and score it against humans.

    uv run python scripts/label_teacher.py --teacher gemma4_26b_a4b \
        --data data/processed/helpsteer2.jsonl data/processed/civil_comments.jsonl:2000 --splits gold_calib gold_test

`PATH:N` samples N rows per split (seeded). Raw teacher distributions go to data/teacher/<stem>__<teacher>.jsonl, one
line per row, appended chunk by chunk; a restart skips finished rows. The request rate follows the teacher's schedule
(configs/teachers.yaml). When a file is complete and has gold_calib rows, temperatures are fitted on gold_calib
(the same calibration as the student, decisivelm.calibrate) and calibrated predictions are written to
data/predictions/teacher_<stem>__<teacher>.jsonl for scripts/eval_dataset.py.

`--local` runs the teacher's model here on the GPU instead of the server (decisivelm.synth.teacher.LocalTeacher),
one row at a time, and labels rows with images or audio too, which the server path leaves out: the media P0.

    uv run python scripts/label_teacher.py --teacher gemma4_e4b --local --data data/processed/koniq10k.jsonl
"""
import argparse
import collections
import json
import pathlib
import random
import time
from concurrent.futures import ThreadPoolExecutor

import transformers

from decisivelm.calibrate import QuestionLogits, calibration_json, fit_calibration, predictions, question_names, write_predictions
from decisivelm.data.dataset import load_records
from decisivelm.head import Calibration
from decisivelm.model import (backbone_profile, load_causal_lm, load_feature_extractor, load_image_processor,
                             load_tokenizer)
from decisivelm.preprocess import truncate_state
from decisivelm.render import as_text
from decisivelm.synth.label import prompt_template
from decisivelm.synth.teacher import (Gate, LocalTeacher, Schedule, VLLMTeacher, load_teacher_config, question_labels,
                                      server_waiting, teacher_logits)
from decisivelm.train import _target

REPO = pathlib.Path(__file__).resolve().parent.parent


def spec(s):
    path, _, n = s.partition(":")
    return pathlib.Path(path), int(n) if n else None


def label_file(path, n, splits, teacher, meta, out_path, chunk, groups_in_parallel, seed):
    rows, records = [], load_records(path)
    for s in splits:
        part = [r for r in records if r.split == s]
        if n is not None and n < len(part):
            part = [part[i] for i in sorted(random.Random(f"{seed}-{s}-{path.name}").sample(range(len(part)), n))]
        rows += part
    if not isinstance(teacher, LocalTeacher):
        rows = [r for r in rows if not r.media]   # the server path sends text only; --local labels media rows
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r.id not in done]
    print(f"{path.name}: {len(done)} rows done, {len(todo)} to go", flush=True)
    tok = teacher.tokenizer
    t0, r0 = time.perf_counter(), teacher.requests
    pool = ThreadPoolExecutor(max_workers=groups_in_parallel)
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        states = {r.id: truncate_state(as_text(r.state), tok, r.preprocess.max_state_tokens)[0] for r in part}
        groups = collections.defaultdict(list)   # rows asking the identical question share prompts and the KV cache
        for r in part:
            for qid, q in r.questions.items():   # a row with media is its own group: its prompts carry its media
                groups[(qid, q.model_dump_json(), r.id if r.media else None)].append(r)
        jobs = {key: pool.submit(question_labels, members[0].questions[key[0]], [states[r.id] for r in members],
                                 [r.id for r in members], teacher.bound(members[0].media) if key[2] else teacher,
                                 meta["max_logprobs"])
                for key, members in groups.items()}
        answers, infos = collections.defaultdict(dict), collections.defaultdict(dict)
        for key, job in jobs.items():
            for r, (ans, info) in zip(groups[key], job.result()):
                answers[r.id][key[0]], infos[r.id][key[0]] = ans, info
        with open(out_path, "a") as f:
            for r in part:
                f.write(json.dumps({"id": r.id, "split": r.split, "answers": answers[r.id], "info": infos[r.id],
                                    **meta}) + "\n")
        dt = time.perf_counter() - t0
        n_done = start + len(part)
        gate = getattr(teacher, "gate", None)
        print(f"{path.name} {n_done}/{len(todo)} rows  {n_done / dt * 3600:,.0f} rows/h  "
              f"{(teacher.requests - r0) / dt:.1f} req/s" + (f"  in flight ≤ {gate.limit()}  "
              f"({'day' if gate.schedule.is_day() else 'night'})" if gate else ""), flush=True)
    return rows


def calibrate_file(rows, out_path, pred_path):
    raw = {d["id"]: d for d in map(json.loads, open(out_path))}
    qs, offlabel = [], []
    for r in rows:
        d = raw.get(r.id)
        if d is None:
            return print(f"{out_path.name}: {r.id} not labelled yet; no calibration", flush=True)
        for qid, q in r.questions.items():
            ans, info = d["answers"].get(qid), d["info"].get(qid, {})
            if ans is None:
                continue
            offlabel.append(info.get("allowed_mass", 1.0) < 0.5)
            qs.append(QuestionLogits(r.id, r.split, qid, q.type, question_names(q), teacher_logits(q, ans),
                                     _target(q, r.targets.get(qid))))
    calib_qs = [q for q in qs if q.split == "gold_calib"]
    if not any(q.split == "gold_test" for q in qs):
        return print(f"{out_path.name}: no gold_test rows; no predictions", flush=True)
    # No gold_calib rows (a zero-shot benchmark): the teacher's raw distributions, uncalibrated.
    calib = fit_calibration(calib_qs) if calib_qs else Calibration()
    write_predictions(pred_path, predictions(qs, calib))
    print(f"{out_path.name}: {len(qs)} questions, off-label (allowed mass < .5) {sum(offlabel) / max(1, len(offlabel)):.2%}; "
          f"calibration {calibration_json(calib)}; wrote {pred_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="gemma4_26b_a4b")
    ap.add_argument("--data", nargs="+", type=spec, required=True)
    ap.add_argument("--splits", nargs="+", default=["gold_calib", "gold_test"])
    ap.add_argument("--chunk", type=int, default=64, help="rows per written chunk")
    ap.add_argument("--groups-in-parallel", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local", action="store_true", help="run the teacher's model here, media rows included")
    ap.add_argument("--media-root", type=pathlib.Path, default=REPO)
    a = ap.parse_args()

    cfg = load_teacher_config(a.teacher)
    meta = {"teacher": a.teacher, "teacher_model": cfg["model"], "teacher_revision": cfg["revision"],
            "max_logprobs": cfg["max_logprobs"],
            "prompt_hashes": {t: prompt_template(t)[1] for t in ("choice_v1", "choice_num_v1", "score_v1", "noul_v1")}}
    if a.local:
        backbone = cfg["tokenizer_backbone"]
        if backbone_profile(backbone)["revision"] != cfg["revision"]:
            raise SystemExit(f"{a.teacher}: the {backbone} profile's revision is not the teacher's")
        profile = backbone_profile(backbone)
        model, tok = load_causal_lm(backbone)
        audio = load_feature_extractor(backbone) if "audio" in profile.get("modalities", []) else None
        teacher = LocalTeacher(model, tok, load_image_processor(backbone), audio, profile["image_max_soft_tokens"],
                               a.media_root)
        meta |= {"backend": f"transformers {transformers.__version__}, local", "image_budget": teacher.image_budget}
        a.groups_in_parallel = 1
    else:
        gate = Gate(Schedule(**cfg["schedule"]), server_waiting(cfg["base_url"], cfg["model"]))
        teacher = VLLMTeacher(cfg, load_tokenizer(cfg["tokenizer_backbone"]), gate)
        meta["vllm_version"] = cfg["vllm_version"]
    out_dir = REPO / "data" / "teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, n in a.data:
        out_path = out_dir / f"{path.stem}__{a.teacher}.jsonl"
        rows = label_file(path, n, a.splits, teacher, meta, out_path, a.chunk, a.groups_in_parallel, a.seed)
        calibrate_file(rows, out_path, REPO / "data" / "predictions" / f"teacher_{path.stem}__{a.teacher}.jsonl")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
