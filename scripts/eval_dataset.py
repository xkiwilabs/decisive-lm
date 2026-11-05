"""The metrics of docs/guide/11 on gold_test for any dataset, every question type, with bootstrap CIs and paired differences.

    uv run python scripts/eval_dataset.py --data data/processed/banking77.jsonl \
        --predictions "P1 E4B=data/predictions/P1_banking77_e4b_bf16_final_q_z_seed0.jsonl" ... \
        [--reference "P1 E4B"] [--out results/banking77_gold_test.md]

Predictions are decisivelm.calibrate rows ("answers" per question; a single-question file may carry only the
top-level "probabilities"). Targets are the human ones in the data file.
- Choice: accuracy and macro-F1 against the human plurality (tied items left out), soft NLL / log n against the
  human shares, top-label ECE (15 bins), Brier, confidence AUROC (top p separating right from wrong); accuracy on
  items whose gold is a self-contained `other` option, where one exists.
- Score: MAE of the expected level against the raters' mean, exact level accuracy (untied), EMD between CDFs.
- Noul: AUROC and accuracy at .5 against human share ≥ .5, Brier, ECE, soft BCE against the share, and Spearman ρ
  between predicted P(yes) and the share.
- Review readiness (where rows record ≥ 3 raters; docs/guide/11 §11.3): ρ of top p with per-item pair agreement, and tie AUROC.
Primary metric per type (docs/guide/11 §11.2): Choice accuracy, Score MAE, Noul AUROC. Paired 95% CIs come from the
same item-level bootstrap resamples for every method.
"""
import argparse, json, math, pathlib

import numpy as np

from decisivelm.data.dataset import load_records
from decisivelm.eval import auroc, ece, review_curve, spearman

REPO = pathlib.Path(__file__).resolve().parent.parent
PRIMARY = {"choice": "accuracy", "score": "MAE", "noul": "AUROC"}


REVIEW = 0.2


def load_predictions(path, qid) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        a = r.get("answers", {}).get(qid)
        if a is None and "probabilities" in r:
            a = {"probabilities": r["probabilities"]}
        if a is not None:
            out[r["id"]] = a
    return out


def macro_f1(pred, gold, k) -> float:
    f = []
    for c in range(k):
        tp = ((pred == c) & (gold == c)).sum()
        fp, fn = ((pred == c) & (gold != c)).sum(), ((pred != c) & (gold == c)).sum()
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def unique_argmax(t: np.ndarray) -> np.ndarray:
    top = np.sort(t, 1)
    return np.where(top[:, -1] > top[:, -2] + 1e-9, t.argmax(1), -1)


def pair_agreement_from_shares(t: np.ndarray, n: np.ndarray) -> np.ndarray:
    counts = np.rint(t * n[:, None])
    return (counts * (counts - 1)).sum(1) / (n * (n - 1))


class Question:
    """Targets of one question over gold_test, and per-item metric contributions for one method."""

    def __init__(self, rows, qid, qtype, names, other):
        self.qid, self.qtype, self.names, self.other = qid, qtype, names, other
        self.ids = [r.id for r in rows]
        if qtype == "noul":
            self.t = np.array([r.targets[qid].human.noul for r in rows])
            self.n_raters = None
        else:
            keys = names
            self.t = np.array([[r.targets[qid].human.probabilities[k] for k in keys] for r in rows])
            self.n_raters = np.array([r.targets[qid].human.n_raters or 0 for r in rows])
            self.gold = unique_argmax(self.t)

    def probs(self, preds) -> np.ndarray:
        if self.qtype == "noul":
            return np.array([preds[i]["p"] for i in self.ids])
        return np.array([[preds[i]["probabilities"][k] for k in self.names] for i in self.ids])

    def metrics(self, p, idx) -> dict:
        if self.qtype == "noul":
            s, q = self.t[idx], np.clip(p[idx], 1e-7, 1 - 1e-7)
            y = s >= 0.5
            correct = (q >= 0.5) == y
            return {"AUROC": auroc(q, y) if 0 < y.sum() < len(y) else float("nan"),
                    "acc@.5": correct.mean(), "Brier": ((q - y) ** 2).mean(),
                    "ECE": ece(np.maximum(q, 1 - q), correct),
                    "soft BCE": float(-(s * np.log(q) + (1 - s) * np.log(1 - q)).mean()),
                    "ρ(p, share)": spearman(q, s)}
        t, q = self.t[idx], np.clip(p[idx], 1e-9, 1)
        n = t.shape[1]
        gold = self.gold[idx]
        un = gold >= 0
        pred = q.argmax(1)
        top = q.max(1)
        right = pred[un] == gold[un]
        m = {"soft NLL": float(-(t * np.log(q)).sum(1).mean() / math.log(n))}
        if self.qtype == "choice":
            m.update({"accuracy": right.mean(), "macro-F1": macro_f1(pred[un], gold[un], n),
                      "ECE": ece(top[un], right),
                      "Brier": float(((q[un] - np.eye(n)[gold[un]]) ** 2).sum(1).mean()),
                      "conf AUROC": auroc(top[un], right) if 0 < right.sum() < len(right) else float("nan")})
            if self.other is not None:
                o = gold[un] == self.other
                m["`other` accuracy"] = right[o].mean() if o.any() else float("nan")
        else:
            levels = np.arange(n)
            m.update({"MAE": float(np.abs(q @ levels - t @ levels).mean()), "exact": right.mean(),
                      "EMD": float(np.abs(np.cumsum(q, 1) - np.cumsum(t, 1)).sum(1).mean())})
        # Confidence-guided review (the manuscript's measure): a human checks the least-confident 20% of items.
        if 0 < right.sum() < len(right):
            caught, rest = review_curve(top, pred == gold, un, [REVIEW])
            m[f"errors in review {REVIEW:.0%}"], m[f"rest acc {REVIEW:.0%}"] = float(caught[0]), float(rest[0])
        if self.n_raters is not None and (self.n_raters[idx] >= 3).mean() > 0.5:
            r3 = self.n_raters[idx] >= 3
            agree = pair_agreement_from_shares(t[r3], self.n_raters[idx][r3])
            m["ρ(top p, agreement)"] = spearman(top[r3], agree)
            tie = (gold[r3] < 0)
            m["tie AUROC"] = auroc(-top[r3], tie) if 0 < tie.sum() < len(tie) else float("nan")
        return m


def fmt(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True, help="the rows whose human targets are scored against")
    ap.add_argument("--predictions", nargs="+", required=True, help="NAME=path.jsonl")
    ap.add_argument("--reference",
                    help="the NAME every other method is compared with (default: the first --predictions)")
    ap.add_argument("--split", default="gold_test", help="the split to score")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the 95%% CIs")
    ap.add_argument("--seed", type=int, default=0, help="seed for the bootstrap")
    ap.add_argument("--out", type=pathlib.Path, help="also write the report to this Markdown file")
    a = ap.parse_args()

    rows = [r for r in load_records(a.data) if r.split == a.split]
    methods = dict(p.split("=", 1) for p in a.predictions)
    # A method that answered a sample (e.g. a teacher labelled N rows per split): score every method on the rows
    # all of them answered, and say so.
    answered = [{json.loads(l)["id"] for l in open(path)} for path in methods.values()]
    common = [r for r in rows if all(r.id in ids for ids in answered)]
    note = "" if len(common) == len(rows) else f" (the {len(common)} of {len(rows)} that every method answered)"
    rows = common
    ref = a.reference or next(iter(methods))
    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, len(rows), len(rows)) for _ in range(a.n_boot)]
    lines = [f"# {a.data.stem}: {a.split}", "",
             f"{len(rows)} rows{note}. 95% CIs from {a.n_boot} item-level bootstrap resamples; Δ = method minus {ref}.", ""]
    summary = {m: [] for m in methods}
    def option_names(q):
        return list(q.criteria) if q.type == "choice" else [str(i) for i in range(len(q.criteria))] \
            if q.type == "score" else ["yes"]

    # Each question id, scored on the rows that ask it; rows whose options differ (a yes/no question beside four-option
    # ones under the same id) are scored as separate groups.
    first = {}
    for r in rows:
        for qid, q in r.questions.items():
            first.setdefault((qid, tuple(option_names(q))), q)
    for k, ((qid, names), q0) in enumerate(first.items()):
        names = list(names)
        asked = [r for r in rows if qid in r.questions and option_names(r.questions[qid]) == names]
        other = names.index("other") if q0.type == "choice" and "other" in names else None
        Q = Question(asked, qid, q0.type, names, other)
        P = {m: Q.probs(load_predictions(path, qid)) for m, path in methods.items()}
        full = np.arange(len(asked))
        g = np.random.default_rng([a.seed, k])
        q_boots = boots if len(asked) == len(rows) else [g.integers(0, len(asked), len(asked)) for _ in range(a.n_boot)]
        res = {m: Q.metrics(p, full) for m, p in P.items()}
        key = PRIMARY[q0.type]
        bs = {m: np.array([Q.metrics(p, b)[key] for b in q_boots]) for m, p in P.items()}
        cols = list(res[ref])
        kind = {"choice": f"choice, {len(names)} options", "score": f"score, {len(names)} levels", "noul": "noul"}[q0.type]
        if len(asked) < len(rows):
            kind += f"; asked on {len(asked)} rows"
        lines += [f"## {qid} ({kind})", "",
                  "| Method | " + " | ".join(cols) + f" | {key} 95% CI | Δ{key} vs ref | 95% CI |",
                  "|---|" + "---|" * (len(cols) + 3)]
        for m in methods:
            lo, hi = np.nanpercentile(bs[m], [2.5, 97.5])
            d = bs[m] - bs[ref]
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
            delta = "" if m == ref else f"{res[m][key] - res[ref][key]:+.3f} | [{dlo:+.3f}, {dhi:+.3f}]"
            lines.append(f"| {m} | " + " | ".join(fmt(res[m][c]) for c in cols) + f" | [{lo:.3f}, {hi:.3f}] | "
                         + (delta or "— | —") + " |")
            summary[m].append((q0.type, res[m][key]))
        lines.append("")
    if len(first) > 1:
        lines += ["## Mean of the primary metric over questions", "", "| Method | " +
                  " | ".join(sorted({t for t, _ in summary[ref]})) + " |", "|---|" + "---|" * len({t for t, _ in summary[ref]})]
        for m, v in summary.items():
            lines.append(f"| {m} | " + " | ".join(fmt(float(np.nanmean([x for t, x in v if t == ty])))
                                                  for ty in sorted({t for t, _ in v})) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
