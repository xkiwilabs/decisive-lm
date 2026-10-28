"""Exchange with the human-rating app (decisivelm/data/rating.py)."""
import pytest

from decisivelm.api_types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from decisivelm.data.rating import (META, apply_ratings, meta_shares, pair_agreement, repeat_consistency, to_item,
                                    top_share)
from decisivelm.data.schema import (HumanTarget, Lineage, Preprocess, QuestionTargets, Record, TeacherTarget)

DEPT = ChoiceQuestion(type="choice", instructions="Which team?", criteria={"billing": "Payments", "technical": "Bugs"})
LEVEL = ScoreQuestion(type="score", instructions="How angry?", criteria=["calm", "cross", "furious"])
URGENT = NoulQuestion(type="noul", instructions="The message is urgent")
REGISTRY = {"toy": {"license": "CC0-1.0", "tier": "R", "fetch": [{}]}}


def record(source="toy"):
    teacher = TeacherTarget(probabilities={"billing": 0.3, "technical": 0.7}, method="logprobs", teacher_model="t")
    return Record(id="r0", lineage=Lineage(source_dataset=source, source_id="0", family_id="toy"), license="CC0-1.0",
                  tier="R", modality=["text"], language="en", state="The app crashes when I pay",
                  questions={"d": DEPT, "a": LEVEL, "u": URGENT},
                  targets={"d": QuestionTargets(human=HumanTarget(probabilities={"billing": 1.0, "technical": 0.0}),
                                                teacher=teacher)},
                  preprocess=Preprocess(render_version=1, max_state_tokens=8192, truncated=False), split="gold_test")


def resp(rater, value, kind="first"):
    return {"rater": rater, "value": value, "kind": kind}


def test_an_item_carries_the_request_but_no_answers():
    item = to_item(record(), REGISTRY, "pilot")
    assert item["state"] == "The app crashes when I pay" and set(item["questions"]) == {"d", "a", "u"}
    assert item["data_class"] == "public" and "targets" not in item and "reference" not in item
    assert to_item(record("synth_gemma4"), REGISTRY, "pilot")["data_class"] == "synthetic"
    assert to_item(record("study"), REGISTRY, "pilot")["data_class"] == "private"
    assert to_item(record(), REGISTRY, "pilot", role="training", questions=["d"])["reference"] == {"d": "billing"}


def test_ratings_become_shares_of_first_answers_and_keep_the_teacher():
    rating = {"format": "hv-ratings/1", "item_id": "r0", "questions": {
        "d": {"type": "choice", "responses": [resp("p1", "technical"), resp("p2", "technical"), resp("p3", "billing"),
                                             resp("p1", "billing", "repeat"), resp("p4", None),
                                             {**resp("p5", "billing"), "excluded": True}]},
        "a": {"type": "score", "responses": [resp("p1", 2), resp("p2", 1), resp("p3", 2)]},
        "u": {"type": "noul", "responses": [resp("p1", True), resp("p2", False)]}}}
    out, n = apply_ratings(record(), rating, min_raters=3)
    assert n == 2   # u has only two raters
    d = out.targets["d"]
    assert d.human.probabilities == pytest.approx({"billing": 1 / 3, "technical": 2 / 3}) and d.human.n_raters == 3
    assert d.teacher.probabilities["technical"] == 0.7
    assert out.targets["a"].human.probabilities == pytest.approx({"0": 0, "1": 1 / 3, "2": 2 / 3})
    assert "u" not in out.targets
    out, _ = apply_ratings(record(), rating, min_raters=2)
    assert out.targets["u"].human.noul == 0.5
    assert repeat_consistency([rating]) == {"p1": (0, 1)}


def test_an_answer_that_is_not_an_option_is_refused():
    rating = {"format": "hv-ratings/1", "item_id": "r0",
              "questions": {"d": {"type": "choice", "responses": [resp("p1", "sales")]}}}
    with pytest.raises(ValueError, match="not options"):
        apply_ratings(record(), rating, min_raters=1)


def test_pair_agreement():
    assert pair_agreement(["a", "a", "b"]) == pytest.approx(1 / 3) and pair_agreement(["a"]) is None


def test_over_quota_answers_count_and_unknown_fields_are_ignored():
    # A late first answer whose slot was taken (docs/rating-formats.md) is still an independent judgement.
    rating = {"format": "hv-ratings/1", "item_id": "r0", "exported_at": "2026-10-22", "questions": {
        "d": {"type": "choice", "responses": [{**resp(f"p{i}", "technical"), "over_quota": i == 3, "seconds": 9}
                                             for i in range(4)]}}}
    out, n = apply_ratings(record(), rating, min_raters=3)
    assert n == 1 and out.targets["d"].human.n_raters == 4


def test_a1_fields_check_items_and_data_check_meta_questions():
    r = record()
    assert "reference_source" not in to_item(r, REGISTRY, "pilot", role="check", questions=["d"])
    item = to_item(r, REGISTRY, "pilot", role="check", questions=["d"], a1_fields=True)
    assert item["reference"] == {"d": "billing"} and item["reference_source"] == "human_consensus"
    assert top_share(r, "d") == 1.0
    with pytest.raises(ValueError, match="A1"):
        to_item(r, REGISTRY, "pilot", data_check=True)
    with pytest.raises(ValueError, match="A1"):
        to_item(r, REGISTRY, "pilot", role="training", data_check=True, a1_fields=True)
    item = to_item(r, REGISTRY, "pilot", data_check=True, a1_fields=True)
    assert item["meta_questions"] == list(META) and set(item["questions"]) == {"d", "a", "u", *META}
    assert "reference" not in item
    noul_only = to_item(r, REGISTRY, "pilot", questions=["u"], data_check=True, a1_fields=True)
    assert "meta_distinct" not in noul_only["meta_questions"]   # a yes/no question has no options to tell apart


def test_meta_answers_never_become_targets():
    rating = {"format": "hv-ratings/1", "item_id": "r0", "meta_questions": ["meta_answerable"], "questions": {
        "d": {"type": "choice", "responses": [resp(f"p{i}", "technical") for i in range(3)]},
        "meta_answerable": {"type": "noul", "responses": [resp("p0", True), resp("p1", False), resp("p2", True),
                                                          resp("p0", False, "repeat")]}}}
    out, n = apply_ratings(record(), rating, min_raters=3)
    assert n == 1 and "meta_answerable" not in out.targets
    assert meta_shares(rating) == {"meta_answerable": (pytest.approx(2 / 3), 3)}


def test_training_and_check_lines_never_become_targets_and_checks_score_raters():
    from decisivelm.data.rating import check_results
    line = lambda role: {"format": "hv-ratings/1", "item_id": "r0", "role": role, "meta_questions": [],
                         "item_flags": [], "questions": {"d": {"type": "choice", "responses": [
                             resp("p_aaaaaa", "billing"), resp("p_bbbbbb", "technical"), resp("p_cccccc", "billing"),
                             {**resp("p_withdrawn_1", None), "unable_reason": "unreadable"}]}}}
    for role in ("training", "check"):
        out, n = apply_ratings(record(), line(role), min_raters=1)
        assert n == 0 and out.targets["d"].human.probabilities["billing"] == 1.0
    results = check_results([line("check"), line("training")], {"r0": record()})
    assert results == {"p_aaaaaa": (1, 1), "p_bbbbbb": (0, 1), "p_cccccc": (1, 1)}
