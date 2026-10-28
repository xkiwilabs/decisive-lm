"""Confidence statistics over a probability vector (docs/guide/04 §4.8). 1.0 = all mass on one option."""
import math
from collections.abc import Sequence


def peak(p: Sequence[float]) -> float:
    """(n·p_max − 1)/(n − 1), clipped: TypeSafe's published Choice demo formula. The default."""
    n = len(p)
    return min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def entropy(p: Sequence[float]) -> float:
    h = -sum(x * math.log(x) for x in p if x > 0)
    return 1.0 - h / math.log(len(p))


def margin(p: Sequence[float]) -> float:
    top = sorted(p, reverse=True)
    return top[0] - top[1]


def ordinal(p: Sequence[float]) -> float:
    """1 − Var(level)/Var_max, for Score: mass on neighbouring levels costs less than mass at both ends."""
    n = len(p)
    mean = sum(i * x for i, x in enumerate(p))
    var = sum(x * (i - mean) ** 2 for i, x in enumerate(p))
    return 1.0 - var / ((n - 1) ** 2 / 4)


METHODS = {"peak": peak, "entropy": entropy, "margin": margin, "ordinal": ordinal}
