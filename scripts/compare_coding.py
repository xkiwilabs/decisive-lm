"""Score methods on the first use case's gold_test (docs/guide/11): κ against each human coder, accuracy against the
coders' consensus, and for probabilistic methods NLL, ECE, Brier, confidence AUROC and review readiness: whether
confidence falls where the four coders disagreed, spreads widely enough to threshold, and recovers accuracy when
the least-confident items go to a human (Alhasan, Diaz Monfort & Richardson, under review). 95% CIs by item
bootstrap; differences are paired (same resamples).

    uv run python scripts/compare_coding.py --data data/processed/llm_response_coding.jsonl \
        --codes data/raw/llm_response_coding/full_coding_matrix.csv \
        --predictions P1=data/predictions/P1_llm_response_coding_e4b_seed0.jsonl P0=... \
        --baselines <study>/summary_outputs/full_coding_matrix.csv \
        --reference P1 --out results/coding_gold_test.md

`--baselines` reads only the prompted-Gemma columns of the source project's matrix (whitelist; docs/guide/16 §16.1).
"""
import argparse, itertools, json, pathlib

import numpy as np
import pandas as pd
import torch

from decisivelm.data.dataset import load_records
from decisivelm.eval import (agreement_profile, auroc, brier, ece, entropy_confidence, kappa, no_majority,
                             pair_agreement, review_curve, share_in_review, soft_nll, spearman,
                             within_category_spearman)

HUMANS = ["human001", "human002", "human003", "human004"]
BASELINES = {"Gemma4-EB4": "prompted E4B (reasoning, multi-pass)",
             "Gemma-4-26B-MoE": "prompted 26B A4B (reasoning, multi-pass)"}


def kappa_boot(a, b, idx, k):
    """Cohen's κ of int label arrays (-1 = missing) for every row of resample indices idx [R, N]."""
    A, B = a[idx], b[idx]
    m = (A >= 0) & (B >= 0)
    n = m.sum(1)
    po = ((A == B) & m).sum(1) / n
    cats = np.arange(k)
    pa = ((A[..., None] == cats) & m[..., None]).sum(1) / n[:, None]
    pb = ((B[..., None] == cats) & m[..., None]).sum(1) / n[:, None]
    pe = (pa * pb).sum(1)
    return (po - pe) / (1 - pe)


def ci(x, fmt=".3f"):
    return f"[{np.percentile(x, 2.5):{fmt}}, {np.percentile(x, 97.5):{fmt}}]"


SIGNALS = {"top p": lambda p: p.max(1), "1 − H/log n": entropy_confidence}


def review_readiness(probs, humans, consensus, untied, idx, reference, fractions, names):
    """Knowing which responses to check: per method and confidence signal, ρ with per-item human pair agreement
    (overall and within consensus categories), AUROC for finding the human ties, the signal's dynamic range across
    agreement levels, and confidence-guided review (least confident first)."""
    H = np.stack([humans[h] for h in HUMANS], 1)
    agree, tied = pair_agreement(H), no_majority(H)
    has = ~np.isnan(agree)
    f20 = 0.2 if 0.2 in fractions else fractions[-1]
    out = ["", "## Knowing which responses to check", "",
           f"Per-item human agreement = share of the coder pairs that agree (levels 0, .17, .33, .5, 1). "
           f"{tied.sum()} of {len(tied)} items have no unique human majority (ties). Signals: top p = probability "
           f"of the chosen option; 1 − H/log n = confidence from the spread of the whole distribution. "
           f"Within-category ρ: mean over consensus categories with ≥ 25 items (count in brackets).", "",
           "| Method | Signal | ρ with human agreement | 95% CI | within-category ρ | Tie AUROC | 95% CI | "
           "Mean at lowest → highest agreement | Range |", "|---|---|---|---|---|---|---|---|---|"]
    stats = {}
    for m, p in probs.items():
        for s, fn in SIGNALS.items():
            conf = fn(p)
            rho = spearman(conf[has], agree[has])
            rho_b = np.array([spearman(conf[i][has[i]], agree[i][has[i]]) for i in idx])
            within = within_category_spearman(conf[untied], agree[untied], consensus[untied])
            tie = auroc(-conf, tied)
            tie_b = np.array([auroc(-conf[i], tied[i]) for i in idx])
            means, spread = agreement_profile(conf[has], agree[has])
            stats[m, s] = {"rho": (rho, rho_b), "tie": (tie, tie_b), "conf": conf}
            out.append(f"| {m} | {s} | {rho:.3f} | {ci(rho_b)} | {np.mean(list(within.values())):.3f} "
                       f"({len(within)}) | {tie:.3f} | {ci(tie_b)} | {means[min(means)]:.3f} → "
                       f"{means[max(means)]:.3f} | {spread:.3f} |")

    out += ["", f"Confidence-guided review on top p: the least-confident share of all {len(tied)} items goes to a "
            "human (ties included in the ranking); errors caught and accuracy on the unreviewed rest are counted on "
            "the items with a human consensus.", "",
            "| Method | Unaided accuracy | " + " | ".join(f"Review {f:.0%}: errors caught / rest accuracy"
                                                       for f in fractions)
            + f" | Rest accuracy at {f20:.0%}, 95% CI | Ties in least-confident {f20:.0%} |",
            "|---|---|" + "---|" * (len(fractions) + 2)]
    for m, p in probs.items():
        conf, correct = p.max(1), p.argmax(1) == consensus
        caught, rest = review_curve(conf, correct, untied, [0.0, *fractions])
        rest_b = np.array([review_curve(conf[i], correct[i], untied[i], [f20])[1][0] for i in idx])
        stats[m, "top p"]["rest"] = (rest[1 + fractions.index(f20)], rest_b)
        out.append(f"| {m} | {rest[0]:.1%} | " + " | ".join(f"{c:.1%} / {r:.1%}" for c, r in zip(caught[1:], rest[1:]))
                   + f" | {ci(rest_b * 100, '.1f')} | {share_in_review(conf, tied, f20):.1%} |")

    if reference in probs:
        out += ["", f"Paired differences on top p, {reference} minus each:", "",
                f"| Comparison | Δρ | 95% CI | Δ tie AUROC | 95% CI | Δ rest accuracy at {f20:.0%} (points) | 95% CI |",
                "|---|---|---|---|---|---|---|"]
        r = stats[reference, "top p"]
        for m in probs:
            if m != reference:
                o = stats[m, "top p"]
                cells = []
                for key, scale, fmt in (("rho", 1, ".3f"), ("tie", 1, ".3f"), ("rest", 100, ".1f")):
                    cells += [f"{(r[key][0] - o[key][0]) * scale:+{fmt}}", ci((r[key][1] - o[key][1]) * scale, fmt)]
                out.append(f"| {m} | " + " | ".join(cells) + " |")

        conf = stats[reference, "top p"]["conf"]
        flagged = np.zeros(len(conf), bool)
        flagged[np.argsort(conf, kind="stable")[:int(round(f20 * len(conf)))]] = True
        out += ["", f"{reference} by consensus category (Box 1 step 5: which of the codebook's categories the "
                f"signal sends to review):", "",
                f"| Category | n | Human ties involving it | Mean top p | Flagged in least-confident {f20:.0%} | ρ within |",
                "|---|---|---|---|---|---|"]
        within = within_category_spearman(conf[untied], agree[untied], consensus[untied], min_n=10)
        for k, c in enumerate(names):
            sel = untied & (consensus == k)
            in_tie = tied & (H == k).any(1)
            rho = f"{within[k]:.2f}" if k in within else "—"
            out.append(f"| {c} | {sel.sum()} | {in_tie.sum()} | {conf[sel].mean():.3f} | {flagged[sel].mean():.1%} | {rho} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--codes", type=pathlib.Path, required=True)
    ap.add_argument("--predictions", nargs="*", default=[], help="NAME=path.jsonl")
    ap.add_argument("--baselines", type=pathlib.Path)
    ap.add_argument("--reference")
    ap.add_argument("--question", default="intent")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--review", type=float, nargs="*", default=[0.1, 0.2], help="review fractions to report")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    rows = [r for r in load_records(a.data) if r.split == "gold_test"]
    names = list(rows[0].questions[a.question].criteria)
    code = {c.lower(): i for i, c in enumerate(names)}
    ids = [r.id for r in rows]
    soft = np.array([[r.targets[a.question].human.probabilities[c] for c in names] for r in rows])

    def as_codes(values):
        return np.array([code[v.strip().lower()] if isinstance(v, str) else -1 for v in values])

    codes = pd.read_csv(a.codes, usecols=["SubjectID", "TrialNumber", *HUMANS])
    codes.index = "lrc-" + codes.SubjectID.astype(str) + "-" + codes.TrialNumber.astype(str)
    humans = {h: as_codes(codes.loc[ids, h]) for h in HUMANS}

    hard, probs = {}, {}
    for spec in a.predictions:
        name, path = spec.split("=", 1)
        wanted = set(ids)   # a general head's file also holds other datasets' rows, with several questions each
        preds = {d["id"]: d.get("probabilities") or d["answers"][a.question]["probabilities"]
                 for d in map(json.loads, open(path)) if d["split"] == "gold_test" and d["id"] in wanted}
        missing = [i for i in ids if i not in preds]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} gold_test items missing, e.g. {missing[:3]}")
        probs[name] = np.array([[preds[i][c] for c in names] for i in ids])
        hard[name] = probs[name].argmax(1)
    if a.baselines:
        base = pd.read_csv(a.baselines.expanduser(), usecols=["SubjectID", "TrialNumber", *BASELINES])
        base.index = "lrc-" + base.SubjectID.astype(str) + "-" + base.TrialNumber.astype(str)
        for col in BASELINES:
            hard[col] = as_codes(base.loc[ids, col])

    top = np.sort(soft, 1)
    untied = top[:, -1] > top[:, -2]
    consensus = soft.argmax(1)
    rng = np.random.default_rng(a.seed)
    N = len(ids)
    idx = rng.integers(0, N, (a.n_boot, N))
    ident = np.arange(N)[None]

    def mean_kappa(pred, index):
        return np.mean([kappa_boot(pred, humans[h], index, len(names)) for h in HUMANS], axis=0)

    for h in HUMANS:   # the vectorised κ must agree with the tested one
        assert abs(kappa_boot(hard[next(iter(hard))], humans[h], ident, len(names))[0]
                   - kappa(hard[next(iter(hard))][humans[h] >= 0], humans[h][humans[h] >= 0])) < 1e-9

    point = {m: float(mean_kappa(p, ident)[0]) for m, p in hard.items()}
    boot = {m: mean_kappa(p, idx) for m, p in hard.items()}
    pairs = list(itertools.combinations(HUMANS, 2))
    ceil_point = float(np.mean([kappa_boot(humans[x], humans[y], ident, len(names))[0] for x, y in pairs]))
    ceil_boot = np.mean([kappa_boot(humans[x], humans[y], idx, len(names)) for x, y in pairs], axis=0)

    lines = [f"gold_test: {N} items, {untied.sum()} with an untied human plurality. "
             f"95% CIs from {a.n_boot} item-level bootstrap resamples.", "",
             "| Method | κ vs humans (mean of 4) | 95% CI | Accuracy vs consensus |", "|---|---|---|---|"]
    for m in hard:
        acc = (hard[m][untied] == consensus[untied]).mean() * 100
        label = BASELINES.get(m, m)
        lines.append(f"| {label} | {point[m]:.3f} | {ci(boot[m])} | {acc:.1f}% |")
    lines.append(f"| human ceiling (mean pairwise κ) | {ceil_point:.3f} | {ci(ceil_boot)} | |")

    if a.reference:
        lines += ["", f"Paired differences in mean κ, {a.reference} minus each:", "",
                  "| Comparison | Δκ | 95% CI |", "|---|---|---|"]
        for m in hard:
            if m != a.reference:
                lines.append(f"| {BASELINES.get(m, m)} | {point[a.reference] - point[m]:+.3f} | "
                             f"{ci(boot[a.reference] - boot[m])} |")
        lines.append(f"| human ceiling | {point[a.reference] - ceil_point:+.3f} | {ci(boot[a.reference] - ceil_boot)} |")

    if probs:
        lines += ["", "Probabilistic methods (calibrated on gold_calib). Soft NLL against the coders' shares, "
                  "divided by log 8 (uniform = 1). ECE (15 bins), Brier and AUROC on untied items against the "
                  "consensus; AUROC = how well the top probability separates right from wrong answers.", "",
                  "| Method | Soft NLL | ECE | ECE 95% CI | Brier | Confidence AUROC |", "|---|---|---|---|---|---|"]
        for m, p in probs.items():
            conf, correct = p.max(1)[untied], (p.argmax(1) == consensus)[untied]
            u = np.flatnonzero(untied)
            ece_boot = [ece(p.max(1)[s], (p.argmax(1) == consensus)[s]) for s in u[rng.integers(0, len(u), (500, len(u)))]]
            nll = soft_nll(torch.tensor(np.log(np.clip(p, 1e-12, 1))), torch.tensor(soft))
            lines.append(f"| {m} | {nll:.3f} | {ece(conf, correct):.3f} | {ci(ece_boot)} | "
                         f"{brier(p[untied], consensus[untied]):.3f} | {auroc(conf, correct):.3f} |")

    if probs:
        lines += review_readiness(probs, humans, consensus, untied, idx, a.reference, a.review, names)

    lines += ["", "Recall by consensus category (untied items):", "",
              "| Category | n | " + " | ".join(BASELINES.get(m, m) for m in hard) + " |",
              "|---|---|" + "---|" * len(hard)]
    for k, c in enumerate(names):
        sel = untied & (consensus == k)
        lines.append(f"| {c} | {sel.sum()} | " + " | ".join(f"{(hard[m][sel] == k).mean() * 100:.1f}%" for m in hard) + " |")

    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
