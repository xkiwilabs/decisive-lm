"""Request validation (docs/guide/04)."""
import pytest
from pydantic import ValidationError

from decisivelm.api_types import SystemOneRequest, lint_request


def request(**questions):
    return {"state": "Our API returns 500s.", "questions": questions}


def choice(n):
    return {"type": "choice", "instructions": "Which team?", "criteria": {f"opt{i}": f"Option {i}" for i in range(n)}}


def test_spec_example_request_parses():
    req = SystemOneRequest.model_validate({
        "model": "decisivelm-e4b-general",
        "state": "Our API started returning 500s 20 minutes ago; we can't process orders.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payment or subscription issues",
                                        "technical": "Bugs or integration problems",
                                        "sales": "Pricing or account questions"}},
            "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
            "frustration": {"type": "score", "instructions": "How frustrated does the customer appear?",
                            "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
        },
    })
    assert [q.type for q in req.questions.values()] == ["choice", "noul", "score"]


@pytest.mark.parametrize("n", [1, 256])
def test_choice_needs_2_to_255_options(n):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request(q=choice(n)))


@pytest.mark.parametrize("n", [2, 255])
def test_choice_option_count_limits_are_inclusive(n):
    SystemOneRequest.model_validate(request(q=choice(n)))


@pytest.mark.parametrize("levels", [1, 11])
def test_score_needs_2_to_10_levels(levels):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request(q={"type": "score", "instructions": "How bad?",
                                                   "criteria": [f"level {i}" for i in range(levels)]}))


def test_unknown_question_type_is_rejected():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request(q={"type": "rank", "instructions": "x", "criteria": ["a", "b"]}))


def test_request_needs_at_least_one_question():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request())


def test_at_most_four_images():
    body = request(q=choice(2))
    body["media"] = [{"type": "image", "path": f"{i}.png"} for i in range(5)]
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(body)


def test_media_needs_exactly_one_source():
    body = request(q=choice(2))
    body["media"] = [{"type": "image", "path": "a.png", "url": "https://example.com/a.png"}]
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(body)


@pytest.mark.parametrize("media", [{"type": "audio", "url": "https://example.com/a.wav"},
                                   {"type": "image", "url": "https://example.com/a.png"}])
def test_media_urls_are_refused_up_front(media):
    body = request(q=choice(2))
    body["media"] = [media]
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(body)


def test_state_may_be_a_json_object():
    body = request(q=choice(2))
    body["state"] = {"order": 17, "status": "late"}
    assert SystemOneRequest.model_validate(body).state == {"order": 17, "status": "late"}


def test_criteria_and_instructions_may_be_objects_or_arrays():
    q = {"type": "choice", "instructions": {"what": "Which team?", "examples": ["refund -> billing"]},
         "criteria": {"billing": ["payments", "refunds"], "technical": {"what": "bugs"}}}
    SystemOneRequest.model_validate(request(q=q))


def test_noul_criteria_accept_yes_no_and_true_false():
    yes_no = SystemOneRequest.model_validate(request(q={"type": "noul", "instructions": "Urgent?",
                                                        "criteria": {"yes": "needs action today", "no": "can wait"}}))
    true_false = SystemOneRequest.model_validate(request(q={"type": "noul", "instructions": "Urgent?",
                                                            "criteria": {"true": "needs action today", "false": "can wait"}}))
    assert yes_no.questions["q"].criteria == true_false.questions["q"].criteria


def test_bare_other_option_gets_a_warning():
    q = choice(2)
    q["criteria"]["other"] = "Other"
    assert any("other" in w for w in lint_request(SystemOneRequest.model_validate(request(q=q))))


def test_self_contained_other_option_passes_lint():
    q = choice(2)
    q["criteria"]["other"] = "The message is about something other than cards, transfers or top-ups"
    assert lint_request(SystemOneRequest.model_validate(request(q=q))) == []
