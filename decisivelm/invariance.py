"""Probability-level independence and order checks (docs/guide/11 §11.7).

The masks make a leaf independent of its siblings, of other questions and of other packed requests by
construction. Numerically, a different packing changes kernel shapes and summation order; on an MoE backbone
that noise can flip a token's top-k experts and grow from there (results/M0.md). These checks measure what a
user would see: how far the calibrated probabilities of one question move when only the packing changes.
"""
import numpy as np

from decisivelm.api_types import SystemOneRequest
from decisivelm.eval import spearman
from decisivelm.head import readout, task_key

VARIANTS = ("reversed_options", "after_another_request", "extra_question")
EXTRA = {"type": "noul", "instructions": "The text is written in English"}


def _plain(q) -> dict:
    return q.model_dump(exclude_none=True) if hasattr(q, "model_dump") else dict(q)


def variant_requests(request: dict, qid: str, name: str, other: dict) -> tuple[list[dict], int]:
    """(requests packed into one pass, index of the request under test). Only the packing differs from `request`
    alone: options in reverse order (Choice only), another request packed first, or one more question."""
    if name == "reversed_options":
        q = _plain(request["questions"][qid])
        if q["type"] != "choice":
            raise ValueError("only Choice options are unordered")
        q["criteria"] = dict(reversed(list(q["criteria"].items())))
        return [{**request, "questions": {**request["questions"], qid: q}}], 0
    if name == "after_another_request":
        return [other, request], 1
    if name == "extra_question":
        return [{**request, "questions": {**request["questions"], "_extra": EXTRA}}], 0
    raise ValueError(name)


def answer_probabilities(dlm, requests: list[dict], k: int, qid: str) -> dict[str, float]:
    """Calibrated probabilities of question `qid` of requests[k], with all `requests` packed into one pass; a Noul as
    {"yes": p, "no": 1 − p}."""
    reqs = [SystemOneRequest.model_validate(r) for r in requests]
    tree, logits = dlm.logits([dlm.tree_for(r) for r in reqs])
    mine = [i for i, r in enumerate(tree.request_of_question.tolist()) if r == k]
    q = reqs[k].questions[qid]
    names = list(q.criteria) if q.type == "choice" else None
    p = readout(q.type, logits[tree.leaf_question == mine[list(reqs[k].questions).index(qid)]], dlm.calibration,
                task_key(q), names)
    if q.type == "noul":
        return {"yes": float(p[0]), "no": 1 - float(p[0])}
    names = names or [str(i) for i in range(len(p))]
    return dict(zip(names, p.tolist()))


def compare(base: list[dict], variant: list[dict], review: float = 0.2) -> dict:
    """Per item, the largest |Δp| over options; answer flips; rank stability of the top probability; and the share
    of the least-confident `review` fraction (the items a human would check) that changes."""
    dp = np.array([max(abs(v[c] - b[c]) for c in b) for b, v in zip(base, variant)])
    top_b = np.array([max(b.values()) for b in base])
    top_v = np.array([max(v.values()) for v in variant])
    flips = sum(max(b, key=b.get) != max(v, key=v.get) for b, v in zip(base, variant))
    n = int(round(review * len(base)))
    reviewed_b, reviewed_v = set(np.argsort(top_b, kind="stable")[:n]), set(np.argsort(top_v, kind="stable")[:n])
    return {"n": len(base), "max_dp": float(dp.max()), "mean_dp": float(dp.mean()),
            "p99_dp": float(np.percentile(dp, 99)), "flips": int(flips),
            "top_p_rho": spearman(top_b, top_v) if len(base) > 2 else float("nan"),
            "review_churn": len(reviewed_b - reviewed_v) / n if n else 0.0}
