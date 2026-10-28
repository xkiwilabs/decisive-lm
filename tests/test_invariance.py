"""Probability-level independence and order checks (docs/guide/11 §11.7): how far answers move when only the packing
changes. Exact by construction on a dense backbone in fp32; on an MoE backbone, top-k routing turns float noise
into larger jumps (results/M0.md), so it is measured, not assumed."""
import pytest
import torch

from decisivelm.head import Scorer
from decisivelm.infer import DecisiveLM
from decisivelm.invariance import VARIANTS, answer_probabilities, compare, variant_requests
from tiny_gemma import tiny_decoder, tokenizer

DEPT = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems",
                     "sales": "Pricing or account questions"}}
REQ = {"state": "Our API started returning 500s 20 minutes ago; we can't process orders.", "questions": {"d": DEPT}}
OTHER = {"state": "Please send me last month's invoice again.", "questions": {"d": DEPT}}


@pytest.fixture(scope="module")
def dlm():
    backbone = tiny_decoder("dense", "sdpa", vocab=len(tokenizer()))
    torch.manual_seed(1)
    return DecisiveLM(backbone, tokenizer(), Scorer(backbone.config.hidden_size).eval())


def test_every_variant_leaves_the_answer_unchanged_on_a_dense_backbone_in_fp32(dlm):
    base = answer_probabilities(dlm, [REQ], 0, "d")
    for name in VARIANTS:
        requests, k = variant_requests(REQ, "d", name, OTHER)
        got = answer_probabilities(dlm, requests, k, "d")
        assert set(got) == set(base)
        assert max(abs(got[c] - base[c]) for c in base) < 1e-5, name


def test_variants_change_the_packing_not_the_question():
    (rev,), _ = variant_requests(REQ, "d", "reversed_options", OTHER)
    assert list(rev["questions"]["d"]["criteria"]) == ["sales", "technical", "billing"]
    pair, k = variant_requests(REQ, "d", "after_another_request", OTHER)
    assert pair == [OTHER, REQ] and k == 1
    (extra,), _ = variant_requests(REQ, "d", "extra_question", OTHER)
    assert len(extra["questions"]) == 2 and extra["questions"]["d"] == DEPT


BASE = [{"a": .9, "b": .1}, {"a": .6, "b": .4}, {"a": .2, "b": .8}, {"a": .55, "b": .45}, {"a": .95, "b": .05}]


def test_compare_reports_the_largest_shift_and_answer_flips():
    moved = [dict(p) for p in BASE]
    moved[1] = {"a": .4, "b": .6}
    out = compare(BASE, moved, review=0.2)
    assert out["max_dp"] == pytest.approx(0.2) and out["mean_dp"] == pytest.approx(0.04)
    assert out["flips"] == 1


def test_compare_reports_churn_in_the_review_set():
    moved = [dict(p) for p in BASE]
    moved[1] = {"a": .52, "b": .48}      # now the least confident, displacing item 3 from the reviewed 20%
    out = compare(BASE, moved, review=0.2)
    assert out["flips"] == 0 and out["review_churn"] == pytest.approx(1.0)
