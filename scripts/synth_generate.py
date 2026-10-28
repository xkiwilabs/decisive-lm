"""Teacher-written question sets and states for the general adapter (docs/guide/13 §13.4), resumably.

    uv run python scripts/synth_generate.py --name synth_v1 --sets 2000 \
        --fanout data/processed/civil_comments.jsonl:500 data/processed/helpsteer2.jsonl:500

Two kinds of job, each one seeded spec (decisivelm.synth.generate):
- **synth**: the teacher writes a question set for a sampled domain and setting, then `--states` states for it with
  sampled profiles (easy, borderline, ambiguous, irrelevant, long, JSON). One row per state, all of them sharing the
  question set, which is the row's schema family. Splits by family: `--val-share` to val, the rest train.
- **fanout**: the teacher writes a question set about a real state from a seed's train or val rows; the row keeps
  the parent's split, licence and tier.

Raw outputs go to data/synth/<name>.raw.jsonl (one line per job, resumable). Rows that pass the filters go to
data/processed/<name>.jsonl with no targets, ready for scripts/label_teacher.py. Filters: questions that fail the
spec or point at other questions are dropped; exact duplicate question sets and states are dropped; questions and
states whose word n-grams overlap an evaluation or holdout row (any non-train row in data/processed, and every row
of a general-adapter holdout) are dropped and counted. Requests follow the teacher's schedule (configs/teachers.yaml).
"""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from decisivelm import CONFIGS
from decisivelm.data.dataset import load_records
from decisivelm.data.schema import Lineage, Preprocess, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION, as_text
from decisivelm.synth import generate as g
from decisivelm.synth.label import prompt_template
from decisivelm.synth.teacher import Gate, Schedule, VLLMTeacher, load_teacher_config, server_waiting

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = "synth_gemma4"
ATTEMPTS = 2
_WORD = re.compile(r"\w+")


def spec(s):
    path, _, n = s.partition(":")
    return pathlib.Path(path), int(n) if n else None


def shingles(text: str, n: int) -> set[int]:
    words = _WORD.findall(text.lower())
    return {hash(" ".join(words[i:i + n])) for i in range(max(0, len(words) - n + 1))}


class EvalIndex:
    """Word n-grams of every evaluation and holdout row's questions and states. A generated text overlaps when at
    least `share` of its n-grams occur in that index (texts shorter than n words never overlap)."""

    def __init__(self, processed: pathlib.Path, exclude: str, n_q=6, n_s=8):
        holdouts = yaml.safe_load((CONFIGS / "holdout.yaml").read_text())["holdouts"]
        held = {h["dataset"] for h in holdouts if "general" in h["adapters"] and not
                ({"source_ids", "family_ids", "languages"} & h.keys())}
        self.n_q, self.n_s, self.q, self.s = n_q, n_s, set(), set()
        for path in sorted(processed.glob("*.jsonl")):
            if path.stem.startswith(exclude):
                continue
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                if d.get("media") or (d["split"] == "train" and d["lineage"]["source_dataset"] not in held):
                    continue
                self.s |= shingles(as_text(d["state"]), n_s)
                for q in d["questions"].values():
                    self.q |= shingles(question_text(q), n_q)

    def overlaps(self, text: str, kind: str, share: float = 0.5) -> bool:
        grams = shingles(text, self.n_q if kind == "q" else self.n_s)
        index = self.q if kind == "q" else self.s
        return bool(grams) and sum(x in index for x in grams) / len(grams) >= share


def question_text(q: dict) -> str:
    return as_text(q["instructions"]) + " " + as_text(q.get("criteria") or "")


def next_hour(hour: int, now: datetime.datetime | None = None) -> datetime.datetime:
    """The next time the local clock reads hour:00."""
    now = now or datetime.datetime.now()
    t = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return t if t > now else t + datetime.timedelta(days=1)


def run_job(teacher, job: dict, n_states: int) -> dict:
    sp = g.sample_spec(job["id"], version=job["version"])
    seed = int(hashlib.sha256(job["id"].encode()).hexdigest()[:8], 16)
    msgs = (g.fanout_prompt(sp, job["state"], job["source"]) if job["kind"] == "fanout"
            else g.questions_prompt(sp))
    for attempt in range(ATTEMPTS):   # a sampled run sometimes degenerates into whitespace; resample
        raw_q, finish_q = teacher.chat_json(msgs, g.QUESTIONS_SCHEMA, seed + 2 * attempt, max_tokens=3072)
        if raw_q is not None:
            break
    out = {"id": job["id"], "kind": job["kind"], "spec": g.dataclasses.asdict(sp), "questions_raw": raw_q,
           "finish_questions": finish_q, **{k: job[k] for k in ("parent_id",) if k in job}}
    if raw_q is None or job["kind"] == "fanout":
        return out
    questions, _ = g.to_questions(raw_q, sp)
    if not questions:
        return out
    profiles = g.sample_profiles(job["id"], n_states, sp)
    for attempt in range(ATTEMPTS):
        raw_s, finish_s = teacher.chat_json(g.states_prompt(sp, questions, profiles), g.STATES_SCHEMA,
                                            seed + 1 + 2 * attempt, max_tokens=8192)
        if raw_s is not None:
            break
    return {**out, "profiles": profiles, "states_raw": raw_s, "finish_states": finish_s}


def build_rows(raw_path, parents, index, val_share, counts) -> list[Record]:
    rows, seen_q, seen_s = [], set(), set()
    for line in open(raw_path, encoding="utf-8"):
        d = json.loads(line)
        sp = g.Spec(**{**d["spec"], "questions": [g.QSpec(**q) for q in d["spec"]["questions"]]})
        if d["questions_raw"] is None:
            counts["job: unparsable questions"] += 1
            continue
        questions, why = g.to_questions(d["questions_raw"], sp)
        for reason in why:
            counts[f"question dropped: {re.sub(r'[0-9]+', 'N', reason)}"] += 1
        kept = {}
        for qid, q in questions.items():
            if index.overlaps(question_text(q.model_dump(exclude_none=True)), "q"):
                counts["question dropped: overlaps an eval/holdout question"] += 1
            else:
                kept[qid] = q
        key = json.dumps({k: q.model_dump(exclude_none=True) for k, q in kept.items()}, sort_keys=True)
        if not kept or key in seen_q:
            counts["job: no questions left" if not kept else "job: duplicate question set"] += 1
            continue
        seen_q.add(key)
        family = f"{raw_path.name.split('.')[0]}-{d['id']}"
        if d["kind"] == "fanout":
            p = parents[d["parent_id"]]
            rows.append(Record(id=f"{family}", lineage=Lineage(
                source_dataset=p.lineage.source_dataset, source_id=p.lineage.source_id, family_id=family,
                parent_id=p.id, multiplier="fanout"), license=p.license, tier=p.tier, modality=["text"],
                language=p.language, state=p.state, questions=kept, targets={}, preprocess=p.preprocess,
                split=p.split))
            counts["rows: fanout"] += 1
            continue
        states = (d.get("states_raw") or {}).get("states", [])
        if not states:
            counts["job: no states"] += 1
            continue
        split = "val" if random.Random(family).random() < val_share else "train"
        for i, (text, profile) in enumerate(zip(states, d["profiles"])):
            state = g.to_state(text, profile)
            flat = as_text(state)
            if not flat.strip() or flat in seen_s:
                counts["state dropped: empty or duplicate"] += 1
                continue
            if index.overlaps(flat, "s", share=0.3):
                counts["state dropped: overlaps an eval/holdout state"] += 1
                continue
            seen_s.add(flat)
            rows.append(Record(id=f"{family}-s{i}", lineage=Lineage(
                source_dataset=SOURCE, source_id=f"{d['id']}-s{i}", family_id=family, multiplier=f"synth:{profile}"),
                license="Apache-2.0", tier="R", modality=["text"], language="en", state=state, questions=kept,
                targets={}, preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS,
                                                  truncated=False), split=split))
            counts[f"rows: synth {split}"] += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synth_v1")
    ap.add_argument("--teacher", default="gemma4_26b_a4b")
    ap.add_argument("--sets", type=int, default=0, help="synth jobs (question set + states)")
    ap.add_argument("--states", type=int, default=6, help="states per synth question set")
    ap.add_argument("--fanout", nargs="*", type=spec, default=[], help="PATH:N real states from train/val rows")
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--stop-at", type=int, help="local hour after which no new job starts (e.g. 23, to hand the "
                                                 "server to an overnight labelling run); finished jobs are kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec-version", type=int, default=1, choices=sorted(g.VERSIONS))
    a = ap.parse_args()

    jobs, parents = [{"id": f"s{a.seed}-{i:06d}", "kind": "synth", "version": a.spec_version} for i in range(a.sets)], {}
    for path, n in a.fanout:
        recs = [r for r in load_records(path) if r.split in ("train", "val") and not r.media]
        if n is not None and n < len(recs):
            recs = random.Random(f"{a.seed}-{path.name}").sample(recs, n)
        for r in recs:
            parents[r.id] = r
            jobs.append({"id": f"f{a.seed}-{r.id}", "kind": "fanout", "parent_id": r.id, "state": as_text(r.state),
                         "source": r.lineage.source_dataset, "version": a.spec_version})

    random.Random(a.seed).shuffle(jobs)   # a run stopped early (--stop-at) still has both kinds of job
    raw_path = REPO / "data" / "synth" / f"{a.name}.raw.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(raw_path)} if raw_path.exists() else set()
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(done)} jobs done, {len(todo)} to go", flush=True)
    if todo and not a.build_only:
        cfg = load_teacher_config(a.teacher)
        gate = Gate(Schedule(**cfg["schedule"]), server_waiting(cfg["base_url"], cfg["model"]))
        teacher = VLLMTeacher(cfg, None, gate)
        meta = {"teacher": a.teacher, "teacher_model": cfg["model"], "teacher_revision": cfg["revision"],
                "vllm_version": cfg["vllm_version"], "prompt_hash": prompt_template(g.VERSIONS[a.spec_version]["prompt"])[1]}
        deadline = next_hour(a.stop_at) if a.stop_at is not None else None
        failed = []

        def job(j):
            if deadline and datetime.datetime.now() >= deadline:
                return None
            try:
                return run_job(teacher, j, a.states)
            except Exception as e:   # the server timed out past every retry: leave the job for the next run
                failed.append(f"{j['id']}: {e!r}")
                return None

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.workers) as pool, open(raw_path, "a", encoding="utf-8") as f:
            for k, fut in enumerate(as_completed([pool.submit(job, j) for j in todo]), 1):   # written as they finish
                if (out := fut.result()) is None:
                    continue
                f.write(json.dumps({**out, **meta}, ensure_ascii=False) + "\n")
                f.flush()
                if k % 25 == 0 or k == len(todo):
                    dt = time.perf_counter() - t0
                    print(f"{k}/{len(todo)} jobs  {k / dt * 3600:,.0f} jobs/h  {teacher.completion_tokens / dt:,.0f} "
                          f"generated tok/s  in flight ≤ {gate.limit()} "
                          f"({'day' if gate.schedule.is_day() else 'night'})", flush=True)

        if failed:
            print(f"{len(failed)} jobs failed and are left for the next run, e.g. {failed[:3]}", flush=True)
    index = EvalIndex(REPO / "data" / "processed", exclude="synth_")   # generated rows are not eval rows
    counts = collections.Counter()
    rows = build_rows(raw_path, parents, index, a.val_share, counts)
    out_path = REPO / "data" / "processed" / f"{a.name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    types = collections.Counter(q.type for r in rows for q in r.questions.values())
    print(json.dumps(dict(sorted(counts.items())), indent=1))
    print(f"wrote {out_path} ({len(rows)} rows, {sum(types.values())} questions: {dict(types)})", flush=True)


if __name__ == "__main__":
    main()
