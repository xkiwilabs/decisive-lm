"""Swap the human targets on train rows for calibrated teacher targets, to train a student on teacher labels alone.

    uv run python scripts/teacher_targets.py --data data/processed/civil_comments.jsonl --teacher gemma4_26b_a4b
    uv run python scripts/teacher_targets.py --data data/processed/synth_v1.jsonl --splits train val \
        --calib data/processed/helpsteer2.jsonl data/processed/goemotions.jsonl ... --noul-bias zero

Reads the raw teacher distributions from data/teacher/<stem>__<teacher>.jsonl (scripts/label_teacher.py), fits the
teacher's temperatures on gold_calib against the human labels (decisivelm.calibrate, as label_teacher does), and writes
data/processed/<stem>_teacher.jsonl: train rows the teacher labelled, with teacher targets only (calibrated
probabilities, docs/guide/12 §12.9), and every other split unchanged, so validation, calibration and the test stay human. A
question whose calls put < MIN_ALLOWED mass on allowed labels gets no target (masked in the loss). Train rows the
teacher did not label are left out; if there are any, data/processed/<stem>_human_matched.jsonl holds the same rows
with their human targets, for a like-for-like human run.

Rows with no human labels of their own (generated rows, `--splits train val`) are calibrated with `--calib`: the
teacher's temperatures are fitted on the pooled gold_calib rows of those datasets (each with a teacher file), per
(type, option-count bucket). `--noul-bias zero` keeps the Noul bias at 0 there, since a bias fitted on one dataset
mostly records its base rate.

`--blend TEACHER=W` mixes a second teacher in: each target is (1 − W) × this teacher's calibrated probabilities + W ×
the second's, each calibrated the same way on the same `--calib` rows (General v3 blends in the frozen E4B, whose
confidence tracked human agreement better than the general adapter's). A question the second teacher did not label
keeps the first teacher's target. Written to `--out` (default data/processed/<stem>_teacher.jsonl).
"""
import argparse
import json
import math
import pathlib
import random

import numpy as np

from decisivelm.calibrate import QuestionLogits, _soft_nll, calibration_json, fit_calibration, question_names
from decisivelm.data.dataset import load_records
from decisivelm.data.schema import QuestionTargets, TeacherTarget
from decisivelm.head import BUCKETS, Calibration, bucket, readout
from decisivelm.synth.label import MIN_ALLOWED
from decisivelm.synth.teacher import teacher_logits
from decisivelm.train import _target

REPO = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = {"choice": "choice_v1", "score": "score_v1", "noul": "noul_v1"}
PER_SOURCE = 500


def teacher_rows(path: pathlib.Path, teacher: str) -> dict:
    return {d["id"]: d for d in map(json.loads, open(REPO / "data" / "teacher" / f"{path.stem}__{teacher}.jsonl"))}


def calib_questions(records, raw) -> list[QuestionLogits]:
    """Teacher logits against the human targets on the gold_calib rows the teacher labelled."""
    return [QuestionLogits(r.id, r.split, qid, q.type, question_names(q), teacher_logits(q, raw[r.id]["answers"][qid]),
                           _target(q, r.targets.get(qid)))
            for r in records if r.split == "gold_calib" and r.id in raw
            for qid, q in r.questions.items() if raw[r.id]["answers"].get(qid) is not None]   # None: not labelled


def by_type(calib: Calibration, qtype: str) -> float:
    (t,) = {v for k, v in calib.temperatures.items() if k.split(":")[0] == qtype}
    return t


def balanced_calibration(by_source: dict[str, list[QuestionLogits]]) -> Calibration:
    """One temperature per question type (every option-count bucket shares it), fitted on an equal-sized seeded
    sample of each source's questions of that type, with the Noul bias at 0. For rows with no human labels of their
    own: no one dataset's difficulty or base rate decides it."""
    calib, grid = Calibration(), np.exp(np.linspace(math.log(0.25), math.log(20.0), 400))
    for qtype in ("noul", "choice", "score"):
        srcs = {k: [q for q in v if q.qtype == qtype] for k, v in by_source.items()}
        srcs = {k: v for k, v in srcs.items() if v}
        if not srcs:
            continue
        n = min(PER_SOURCE, *map(len, srcs.values()))
        qs = [q for k, v in srcs.items() for q in random.Random(f"{qtype}-{k}").sample(v, n)]
        if qtype == "noul":
            calib.temperatures["noul"] = fit_calibration(qs, noul_bias=False).temperatures["noul"]
        else:
            t = float(min(grid, key=lambda x: _soft_nll(qs, x)))
            calib.temperatures |= {f"{qtype}:{bucket(lo)}": t for lo, _ in BUCKETS}
        print(f"  {qtype}: {n} questions from each of {', '.join(srcs)}", flush=True)
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--teacher", default="gemma4_26b_a4b")
    ap.add_argument("--splits", nargs="+", default=["train"], help="splits whose targets become the teacher's")
    ap.add_argument("--calib", nargs="+", type=pathlib.Path, help="fit the teacher's calibration on these datasets' "
                                                                  "gold_calib rows (default: --data's own)")
    ap.add_argument("--noul-bias", choices=["fit", "zero"], default="fit")
    ap.add_argument("--balanced", action="store_true",
                    help="one temperature per question type, each --calib dataset weighted equally (up to "
                         f"{PER_SOURCE} questions per type each), Noul bias 0")
    ap.add_argument("--set", nargs="*", default=[], metavar="TYPE=T|TYPE",
                    help="override a type's temperature after fitting: a number, or another type's (score=choice)")
    ap.add_argument("--blend", metavar="TEACHER=W", help="mix in a second teacher's calibrated probabilities at weight W")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    records = load_records(a.data)
    calib_records = {path: load_records(path) for path in a.calib or [] if path != a.data}

    def fit(teacher: str):
        raw = teacher_rows(a.data, teacher)
        by_source = {}
        for path in a.calib or [a.data]:
            rows, labels = (records, raw) if path == a.data else (calib_records[path], teacher_rows(path, teacher))
            by_source[path.stem] = calib_questions(rows, labels)
        if a.balanced:
            calib = balanced_calibration(by_source)
        else:
            calib = fit_calibration([q for qs in by_source.values() for q in qs], noul_bias=a.noul_bias == "fit")
        for s in a.set:
            qtype, _, value = s.partition("=")
            t = by_type(calib, value) if value in ("noul", "choice", "score") else float(value)
            calib.temperatures = {k: v for k, v in calib.temperatures.items() if k.split(":")[0] != qtype}
            calib.temperatures |= {f"{qtype}:{bucket(lo)}": t for lo, _ in BUCKETS} if qtype != "noul" else {"noul": t}
        print(f"{teacher} calibration on {sum(map(len, by_source.values()))} gold_calib questions from "
              f"{', '.join(by_source)}: {calibration_json(calib)}", flush=True)
        return raw, calib

    raw, calib = fit(a.teacher)
    if a.blend:
        blend_teacher, _, w = a.blend.partition("=")
        w = float(w)
        raw2, calib2 = fit(blend_teacher)
    n_blended = 0

    out_rows, matched_rows, n_masked, n_questions = [], [], 0, 0
    for r in records:
        if r.split not in a.splits:
            out_rows.append(r)
            matched_rows.append(r)
            continue
        d = raw.get(r.id)
        if d is None:
            continue
        targets = {}
        for qid, q in r.questions.items():
            ans, info = d["answers"].get(qid), d["info"].get(qid, {})
            n_questions += 1
            if ans is None or info.get("allowed_mass", 1.0) < MIN_ALLOWED:
                n_masked += 1
                continue
            p = readout(q.type, teacher_logits(q, ans), calib)
            model = d["teacher_model"]
            if a.blend:
                d2 = raw2.get(r.id)
                ans2 = d2["answers"].get(qid) if d2 else None
                if ans2 is not None and d2["info"].get(qid, {}).get("allowed_mass", 1.0) >= MIN_ALLOWED:
                    p = (1 - w) * p + w * readout(q.type, teacher_logits(q, ans2), calib2)
                    model = f"{model}*{1 - w:g}+{d2['teacher_model']}*{w:g}"
                    n_blended += 1
            p = p.tolist()
            values = {"noul": p[0]} if q.type == "noul" else {"probabilities": dict(zip(question_names(q), p))}
            targets[qid] = QuestionTargets(teacher=TeacherTarget(
                method=info.get("method", "logprob") + ("+blend" if model != d["teacher_model"] else ""),
                n_orderings=info.get("n_orderings"), allowed_mass=info.get("allowed_mass"), teacher_model=model,
                teacher_revision=d["teacher_revision"], vllm_version=d["vllm_version"],
                prompt_hash=d["prompt_hashes"][TEMPLATE[q.type]],
                calibrated_T=calib.temperature(q.type, len(question_names(q))), **values))
        out_rows.append(r.model_copy(update={"targets": targets}))
        matched_rows.append(r)

    n_train = sum(r.split in a.splits for r in records)
    n_kept = sum(r.split in a.splits for r in out_rows)
    out = a.out or a.data.with_name(f"{a.data.stem}_teacher.jsonl")
    outs = [(out, out_rows)]
    if n_kept < n_train:
        outs.append((a.data.with_name(f"{a.data.stem}_human_matched.jsonl"), matched_rows))
    for path, rows in outs:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(r.model_dump_json(exclude_none=True) + "\n")
        print(f"wrote {path} ({len(rows)} rows)", flush=True)
    print(f"{'/'.join(a.splits)} rows: {n_kept} of {n_train} teacher-labelled; {n_masked} of {n_questions} questions masked "
          f"(allowed mass < {MIN_ALLOWED})", flush=True)
    if a.blend:
        print(f"{n_blended} questions blended with {blend_teacher} at weight {w:g}", flush=True)


if __name__ == "__main__":
    main()
