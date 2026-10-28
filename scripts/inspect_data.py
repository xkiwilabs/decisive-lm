"""Summarise canonical rows (docs/guide/08 §8.9): per-type counts, option counts, Score levels, Noul targets, media, truncation.

    uv run python scripts/inspect_data.py data/processed/*.jsonl

Rows are loaded through the licence check, so a file that disagrees with
configs/datasets.yaml fails here first.
"""
import argparse, collections, json, pathlib

from decisivelm.data.dataset import load_records
from decisivelm.head import BUCKETS, bucket


def summarise(rows) -> dict:
    types, options, levels, noul, by_split = (collections.Counter() for _ in range(5))
    targets = collections.Counter()
    media = collections.Counter()
    for r in rows:
        by_split[r.split] += 1
        for m in r.media:
            media[m.type] += 1
        for qid, q in r.questions.items():
            types[q.type] += 1
            if q.type == "choice":
                options[bucket(len(q.criteria))] += 1
            elif q.type == "score":
                levels[len(q.criteria)] += 1
            t = r.targets.get(qid)
            targets["human" if t and t.human else "teacher" if t and t.teacher else "none"] += 1
            if q.type == "noul" and t and t.human:
                noul[min(int(t.human.noul * 10), 9) / 10] += 1
    n = len(rows)
    return {"rows": n, "rows_by_split": dict(by_split), "questions_by_type": dict(types),
            "choice_option_buckets": {b: options[b] for b in (bucket(lo) for lo, _ in BUCKETS) if options[b]},
            "score_levels": dict(sorted(levels.items())), "noul_human_histogram": dict(sorted(noul.items())),
            "targets": dict(targets), "media": dict(media),
            "rows_with_media": sum(bool(r.media) for r in rows),
            "truncated_rows": sum(r.preprocess.truncated for r in rows),
            "truncation_rate": round(sum(r.preprocess.truncated for r in rows) / n, 4) if n else 0.0,
            "tiers": dict(collections.Counter(r.tier for r in rows)),
            "languages": dict(collections.Counter(r.language for r in rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    for path in ap.parse_args().files:
        print(f"== {path}")
        print(json.dumps(summarise(load_records(path)), indent=1))


if __name__ == "__main__":
    main()
