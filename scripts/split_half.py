"""How predictable is per-item rater agreement? A comparison for review readiness on a multi-rater set.

    uv run python scripts/split_half.py --data data/processed/wiki_toxicity.jsonl [--split gold_test] [--splits 200]

Each item's votes are rebuilt from its human target (shares × number of raters) and split in two at random, many
times (200 by default, seed 0). Two figures, averaged over the splits:
- split-half reliability: Spearman ρ between half A's pair agreement and half B's, with the Spearman-Brown estimate
  for the full panel, 2r / (1 + r);
- the oracle: half A's top share used as a confidence, ρ with half B's pair agreement: how well half a panel's
  answers foretell the other half's agreement. It is a comparison, not a ceiling: a model is scored against the whole
  panel, whose agreement is measured more precisely than half of it.
Items need at least 4 raters (2 per half); pair agreement is as in docs/guide/11.
"""
import argparse
import pathlib

import numpy as np
from scipy.stats import spearmanr

from decisivelm.data.dataset import load_records


def votes(target, question) -> np.ndarray | None:
    """Each rater's answer as an index: a Score's level index, or a Choice option's position in its criteria."""
    h = target.human if target is not None else None
    if h is None or not h.n_raters or not h.probabilities:
        return None
    index = {k: i for i, k in enumerate(question.criteria)} if isinstance(question.criteria, dict) else None
    counts = {(index[k] if index else int(k)): round(p * h.n_raters) for k, p in h.probabilities.items()}
    return np.repeat(list(counts), list(counts.values()))


def agreement(v: np.ndarray, levels: int) -> float:
    c = np.bincount(v, minlength=levels)
    return float((c * (c - 1)).sum() / (len(v) * (len(v) - 1)))


def split_half(items: list[np.ndarray], levels: int, n_splits: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    items = [v for v in items if len(v) >= 4]
    rel, oracle = [], []
    for _ in range(n_splits):
        a, b, top = [], [], []
        for v in items:
            v = rng.permutation(v)
            h = len(v) // 2
            a.append(agreement(v[:h], levels))
            b.append(agreement(v[h:], levels))
            top.append(np.bincount(v[:h], minlength=levels).max() / h)
        rel.append(spearmanr(a, b).statistic)
        oracle.append(spearmanr(top, b).statistic)
    r = float(np.mean(rel))
    return {"items": len(items), "median_raters": float(np.median([len(v) for v in items])), "split_half": r,
            "spearman_brown": 2 * r / (1 + r), "oracle": float(np.mean(oracle)), "n_splits": n_splits, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="gold_test")
    ap.add_argument("--question", help="question id (default: each question with human votes)")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    by_q = {}
    for r in load_records(a.data):
        if r.split != a.split:
            continue
        for qid, q in r.questions.items():
            if (a.question and qid != a.question) or q.type == "noul" or (v := votes(r.targets.get(qid), q)) is None:
                continue
            by_q.setdefault(qid, (len(q.criteria), []))[1].append(v)
    for qid, (levels, items) in by_q.items():
        s = split_half(items, levels, a.splits, a.seed)
        print(f"{a.data.stem} {qid}: {s['items']} items, median {s['median_raters']:.0f} raters; split-half ρ "
              f"{s['split_half']:.3f} (Spearman-Brown {s['spearman_brown']:.3f}); half A's top share predicting half "
              f"B's agreement ρ {s['oracle']:.3f} ({s['n_splits']} random splits, seed {s['seed']})")


if __name__ == "__main__":
    main()
