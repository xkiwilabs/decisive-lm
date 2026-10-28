"""Metrics shared by every report (docs/guide/11). Gates use human-labelled splits only."""
import math

import numpy as np
import pandas as pd
import torch


def kappa(a, b) -> float:
    """Cohen's κ over the items both raters labelled."""
    a, b = pd.Series(list(a), dtype=object), pd.Series(list(b), dtype=object)
    m = (a.notna() & b.notna()).to_numpy()
    a, b = a[m].to_numpy(), b[m].to_numpy()
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return float((po - pe) / (1 - pe))


def ece(conf, correct, bins: int = 15) -> float:
    """Top-label expected calibration error, equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            out += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(out)


def brier(probs, labels) -> float:
    """Multi-class Brier score against hard labels (sum over classes, mean over items)."""
    probs = np.asarray(probs, float)
    onehot = np.eye(probs.shape[1])[np.asarray(labels)]
    return float(((probs - onehot) ** 2).sum(1).mean())


def soft_nll(logits: torch.Tensor, target: torch.Tensor) -> float | torch.Tensor:
    """Cross-entropy against soft targets, divided by log(n) so uniform predictions score 1."""
    loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean() / math.log(logits.shape[-1])
    return loss if loss.requires_grad else float(loss)


def fit_temperature(logits: torch.Tensor, target: torch.Tensor, lo: float = 0.05, hi: float = 20.0) -> float:
    """The T minimising soft NLL of softmax(logits / T), by grid search on a log scale (docs/guide/10 §10.3)."""
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 400))
    return float(min(grid, key=lambda t: float(soft_nll(logits / t, target))))


# Review readiness (docs/guide/11 §11.3): whether a coder's confidence says which of its answers a human should check.
# Definitions follow Alhasan, Diaz Monfort & Richardson, "Knowing which responses to check" (under review).

def pair_agreement(labels) -> np.ndarray:
    """Per item, the share of available rater pairs that gave the same label. labels: int [N, R], -1 = missing."""
    labels = np.asarray(labels)
    k = labels.max() + 1
    counts = np.stack([(labels == c).sum(1) for c in range(k)], 1)
    n = (labels >= 0).sum(1)
    pairs = n * (n - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pairs > 0, (counts * (counts - 1) / 2).sum(1) / pairs, np.nan)


def no_majority(labels) -> np.ndarray:
    """True where the raters' most frequent label is shared by two or more labels (a human tie)."""
    labels = np.asarray(labels)
    counts = np.sort(np.stack([(labels == c).sum(1) for c in range(labels.max() + 1)], 1), 1)
    return counts[:, -1] == counts[:, -2]


def spearman(x, y) -> float:
    """Rank correlation with average ranks for ties."""
    rx, ry = pd.Series(np.asarray(x, float)).rank(), pd.Series(np.asarray(y, float)).rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(score, positive) -> float:
    """P(score of a random positive > score of a random negative), ties counted half."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def entropy_confidence(probs) -> np.ndarray:
    """1 − H(p) / log n: a confidence read from the spread of the whole distribution, not its top value."""
    p = np.clip(np.asarray(probs, float), 1e-12, 1)
    return 1 + (p * np.log(p)).sum(1) / math.log(p.shape[1])


def agreement_profile(conf, agreement, min_n: int = 1) -> tuple[dict, float]:
    """Mean confidence at each level of human agreement, and its dynamic range (highest minus lowest mean).
    A review threshold needs the range: a signal running .986 → .998 ranks items but cannot carry one.
    Levels with fewer than min_n items are left out (items with hundreds of raters each have a level of their own)."""
    g = pd.Series(np.asarray(conf, float)).groupby(np.round(np.asarray(agreement, float), 6))
    s = g.mean()[g.size() >= min_n]
    return {float(k): float(v) for k, v in s.items()}, float(s.max() - s.min())


def within_category_spearman(conf, agreement, category, min_n: int = 25) -> dict:
    """ρ(confidence, human agreement) inside each consensus category with at least min_n items, so that a signal
    tracking only which categories are hard does not count as item-level calibration."""
    conf, agreement, category = np.asarray(conf, float), np.asarray(agreement, float), np.asarray(category)
    return {c: spearman(conf[category == c], agreement[category == c])
            for c in np.unique(category) if (category == c).sum() >= min_n}


def review_curve(conf, correct, scored, fractions) -> tuple[np.ndarray, np.ndarray]:
    """Send the least-confident fraction f of all items to a human. Returns, per f, the share of the coder's
    errors inside the reviewed set and its accuracy on the unreviewed rest. Items without a human consensus
    (`scored` False) are ranked and reviewed like any other but counted in neither."""
    conf, correct, scored = np.asarray(conf, float), np.asarray(correct, bool), np.asarray(scored, bool)
    order = np.argsort(conf, kind="stable")
    errors = scored & ~correct
    caught, remainder = [], []
    for f in fractions:
        reviewed = np.zeros(len(conf), bool)
        reviewed[order[:int(round(f * len(conf)))]] = True
        caught.append(errors[reviewed].sum() / errors.sum())
        rest = scored & ~reviewed
        remainder.append((correct & rest).sum() / rest.sum())
    return np.array(caught), np.array(remainder)


def share_in_review(conf, flag, f: float) -> float:
    """Share of the flagged items (e.g. human ties) that fall in the least-confident fraction f."""
    conf, flag = np.asarray(conf, float), np.asarray(flag, bool)
    reviewed = np.argsort(conf, kind="stable")[:int(round(f * len(conf)))]
    return float(flag[reviewed].sum() / flag.sum())
