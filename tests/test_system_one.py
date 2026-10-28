"""End-to-end on a tiny random Gemma 4 with a random scorer head, fp32 (docs/guide/11 §11.7).

Conformance, independence of questions, and order-invariance of options must
hold by construction, whatever the weights.
"""
import random

import pytest
import torch

from decisivelm.head import Scorer
from decisivelm.infer import DecisiveLM
from tiny_gemma import tiny_decoder, tokenizer

FP32_TOL = 1e-5   # docs/guide/11 §11.7

DEPT = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems",
                     "sales": "Pricing or account questions", "other": "Anything not about billing, bugs or pricing"}}
URGENT = {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}
ANGER = {"type": "score", "instructions": "How frustrated does the customer appear?",
         "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]}
STATE = "Our API started returning 500s 20 minutes ago; we can't process orders. This is the third outage this month."


@pytest.fixture(scope="module", params=["dense", "moe"])
def dlm(request):
    backbone = tiny_decoder(request.param, "sdpa", vocab=len(tokenizer()))
    torch.manual_seed(1)
    scorer = Scorer(backbone.config.hidden_size)
    with torch.no_grad():
        for p in scorer.parameters():
            p.normal_(0.0, 0.5)    # spread the logits so the checks below are not trivially uniform
    return DecisiveLM(backbone, tokenizer(), scorer)


def run(dlm, **questions):
    return dlm.system_one({"model": "test", "state": STATE, "questions": questions})


def test_response_conforms(dlm):
    resp = run(dlm, department=DEPT, is_urgent=URGENT, frustration=ANGER)
    a = resp["answers"]
    d = a["department"]
    assert list(d["probabilities"]) == list(DEPT["criteria"])
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert d["choice"] == max(d["probabilities"], key=d["probabilities"].get)
    assert 0.0 <= d["confidence"] <= 1.0
    f = a["frustration"]
    assert list(f["probabilities"]) == ["0", "1", "2"]
    assert f["legend"] == {"0": ANGER["criteria"][0], "1": ANGER["criteria"][1], "2": ANGER["criteria"][2]}
    assert 0.0 <= f["score"] <= 2.0
    assert f["score"] == pytest.approx(sum(int(k) * v for k, v in f["probabilities"].items()))
    u = a["is_urgent"]
    assert set(u) == {"type", "noul"} and 0.0 <= u["noul"] <= 1.0
    assert resp["usage"]["blocks"] == 4 + 1 + 3
    assert resp["usage"]["confidence_method"] == "peak"


def test_spread_scorer_gives_non_uniform_answers(dlm):
    """Guard: if the head were outputting constants, the invariance tests below would pass vacuously."""
    probs = run(dlm, department=DEPT)["answers"]["department"]["probabilities"]
    assert max(probs.values()) - min(probs.values()) > 0.05


def test_adding_and_removing_questions_does_not_change_answers(dlm):
    alone = run(dlm, department=DEPT)["answers"]["department"]["probabilities"]
    crowded = run(dlm, is_urgent=URGENT, department=DEPT, frustration=ANGER)["answers"]["department"]["probabilities"]
    for k in alone:
        assert abs(alone[k] - crowded[k]) <= FP32_TOL


def test_shuffling_choice_options_does_not_change_their_probabilities(dlm):
    base = run(dlm, department=DEPT)["answers"]["department"]["probabilities"]
    items = list(DEPT["criteria"].items())
    random.Random(0).shuffle(items)
    shuffled = run(dlm, department={**DEPT, "criteria": dict(items)})["answers"]["department"]["probabilities"]
    for k in base:
        assert abs(base[k] - shuffled[k]) <= FP32_TOL


def test_question_id_does_not_change_answers(dlm):
    a = run(dlm, department=DEPT)["answers"]["department"]["probabilities"]
    b = run(dlm, zz_renamed=DEPT)["answers"]["zz_renamed"]["probabilities"]
    for k in a:
        assert abs(a[k] - b[k]) <= FP32_TOL


def test_json_state_is_accepted(dlm):
    resp = dlm.system_one({"state": {"ticket": 17, "text": STATE}, "questions": {"u": URGENT}})
    assert 0.0 <= resp["answers"]["u"]["noul"] <= 1.0


def test_long_state_is_truncated_before_rendering(dlm):
    from decisivelm.preprocess import truncate_state
    long = " ".join(f"word{i}" for i in range(500))
    small = DecisiveLM(dlm.backbone, dlm.tokenizer, dlm.scorer, max_state_tokens=50)
    cut, truncated = truncate_state(long, dlm.tokenizer, 50)
    assert truncated
    tree = small.tree_for({"state": long, "questions": {"department": DEPT}})
    assert tree.prefix == small.tree_for({"state": cut, "questions": {"department": DEPT}}).prefix
