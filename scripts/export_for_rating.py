"""Write records as items for the human-rating app (DecisiveLM-HV, format hv-items/1).

    uv run python scripts/export_for_rating.py --data data/processed/helpsteer2.jsonl --split gold_test --n 300 \
        --task-set helpsteer2_pilot --out data/rating/helpsteer2_pilot.items.jsonl
    uv run python scripts/export_for_rating.py --data data/processed/synth_v2_teacher.jsonl --split val --n 1500 \
        --task-set synth_v2_check --out data/rating/synth_v2_check.items.jsonl
    # training items (shown with the consensus answer, never counted):
    uv run python scripts/export_for_rating.py --data ... --split train --n 20 --role training --task-set ... --out ...
    # check items (unambiguous answers only: every rater agreed, by default):
    uv run python scripts/export_for_rating.py --data ... --split train --n 30 --role check --task-set ... --out ...
    # a data-check project on teacher-written rows (meta-questions about each item):
    uv run python scripts/export_for_rating.py --data data/processed/synth_v2_teacher.jsonl --split val --n 500 \
        --data-check --task-set synth_v2_check --out data/rating/synth_v2_check.items.jsonl

Items carry the state and questions exactly as the model sees them and nothing else a rater should not see: no human,
teacher or model answers. A training or check item's reference is the human consensus, never a teacher's answer.
`reference_source` (training and check items) and `meta_questions` (data-check items) are written by default;
--no-a1-fields leaves them out, for a rating app that predates them. `--task-set` must be the name of the task set the
file is uploaded into, and an item id uploaded once cannot come back with different content, so a new question
subset of the same rows needs its own task set. Private sources (not downloadable from a pinned public source) are refused unless
--allow-private, and the app refuses them too unless its deployment allows private data.
"""
import argparse, json, pathlib

from decisivelm.data.dataset import load_records, load_registry
from decisivelm.data.rating import data_class, sample, to_item, top_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--n", type=int, help="a random sample of this many rows (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", nargs="+", help="only these question ids (default: all of each row's)")
    ap.add_argument("--task-set", required=True)
    ap.add_argument("--role", choices=["rate", "training", "check"], default="rate")
    ap.add_argument("--check-min-share", type=float, default=1.0,
                    help="check items: keep rows whose human top answer has at least this share on every question")
    ap.add_argument("--data-check", action="store_true", help="add meta-questions about each item (data-check project)")
    ap.add_argument("--a1-fields", action=argparse.BooleanOptionalAction, default=True,
                    help="write meta_questions and reference_source (default; --no-a1-fields for an older app)")
    ap.add_argument("--content-warning", nargs="+", default=[], help="e.g. offensive_language self_harm")
    ap.add_argument("--allow-private", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    registry = load_registry()
    if a.data_check and not a.a1_fields:
        ap.error("--data-check writes meta_questions: drop --no-a1-fields")
    rows = [r for r in load_records(a.data, registry) if r.split in a.split]
    if a.questions:
        rows = [r for r in rows if set(a.questions) <= set(r.questions)]
    if a.role == "check":   # check items must have an unambiguous answer (docs/rating-formats.md)
        rows = [r for r in rows if all(top_share(r, q) >= a.check_min_share for q in a.questions or r.questions)]
    rows = sample(rows, a.n, a.seed)
    private = sorted({r.lineage.source_dataset for r in rows if data_class(r, registry) == "private"})
    if private and not a.allow_private:
        raise SystemExit(f"private sources {private}: pass --allow-private only for a deployment that allows them")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(to_item(r, registry, a.task_set, a.role, a.questions, a.content_warning,
                                       data_check=a.data_check, a1_fields=a.a1_fields), ensure_ascii=False) + "\n")
    n_q = sum(len(a.questions or r.questions) for r in rows)
    print(f"{len(rows):,} items, {n_q:,} questions -> {a.out}")


if __name__ == "__main__":
    main()
