"""Deterministic schema explosion over a processed seed (docs/guide/08 §8.10): subsets, drop-gold + `other`, Noul splits.

    uv run python scripts/explode.py data/processed/banking77.jsonl --question intent

Writes data/processed/<name>_explode.jsonl. Every split is exploded and derived rows keep their parent's
split, so gold rows produce gold-only variants for evaluation (docs/guide/08 §8.4); training mixes take `train` only.
"""
import argparse, collections, json, pathlib, random

from decisivelm.data.dataset import load_records
from decisivelm.synth.multipliers.schema_explosion import drop_gold_add_other, noul_splits, subset

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=pathlib.Path)
    ap.add_argument("--question", required=True)
    ap.add_argument("--subsets", type=int, default=2, help="subset variants per row")
    ap.add_argument("--max-k", type=int, default=60)
    ap.add_argument("--nouls", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = load_records(a.data)
    out_path = a.data.with_name(f"{a.data.stem}_explode.jsonl")
    counts = collections.Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            rng = random.Random(f"{a.seed}:{r.id}")
            n = len(r.questions[a.question].criteria)
            children = [subset(r, a.question, k=rng.randint(1, min(a.max_k, n - 1)), rng=rng) for _ in range(a.subsets)]
            children.append(drop_gold_add_other(r, a.question, k=rng.randint(1, min(a.max_k, n - 3)), rng=rng))
            children += noul_splits(r, a.question, n=a.nouls, rng=rng)
            for c in children:
                if c is not None:
                    f.write(c.model_dump_json(exclude_none=True) + "\n")
                    counts[(c.split, c.lineage.multiplier.split(":")[1])] += 1
    print(f"{len(rows)} parents -> {sum(counts.values())} rows in {out_path}")
    print(json.dumps({f"{s}/{op}": n for (s, op), n in sorted(counts.items())}, indent=1))


if __name__ == "__main__":
    main()
