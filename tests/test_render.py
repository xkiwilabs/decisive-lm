"""Block text templates (docs/guide/06 §6.2). Leaves never see their index, siblings or question id."""
from decisivelm.api_types import SystemOneRequest
from decisivelm.render import as_text, leaf_texts, question_segment


def parse(q):
    return SystemOneRequest.model_validate({"state": "s", "questions": {"q_id_secret": q}}).questions["q_id_secret"]


CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
          "criteria": {"billing": "Payment issues", "technical": "Bugs", "sales": "Pricing"}}


def test_choice_leaf_holds_only_its_own_option():
    leaves = leaf_texts(parse(CHOICE))
    assert leaves[1] == "Option: technical — Bugs"
    assert "billing" not in leaves[1] and "sales" not in leaves[1]


def test_question_segment_carries_instructions():
    assert question_segment(parse(CHOICE)) == "Q: Which team should handle this?\n"


def test_score_leaf_has_no_index():
    leaves = leaf_texts(parse({"type": "score", "instructions": "How angry?", "criteria": ["Calm", "Annoyed", "Furious"]}))
    assert leaves == ["Level: Calm", "Level: Annoyed", "Level: Furious"]


def test_noul_has_empty_segment_and_statement_in_leaf():
    q = parse({"type": "noul", "instructions": "The message conveys urgency",
               "criteria": {"yes": "action needed today", "no": "can wait"}})
    assert question_segment(q) == ""
    assert leaf_texts(q) == ["Statement: The message conveys urgency. Yes means: action needed today. No means: can wait."]


def test_noul_without_criteria_omits_the_parts():
    assert leaf_texts(parse({"type": "noul", "instructions": "The customer is angry."})) == \
        ["Statement: The customer is angry."]


def test_question_id_never_rendered():
    q = parse(CHOICE)
    assert "q_id_secret" not in question_segment(q) + "".join(leaf_texts(q))


def test_objects_render_as_compact_json_in_insertion_order():
    assert as_text({"what": "Which team?", "examples": ["a", "b"]}) == '{"what":"Which team?","examples":["a","b"]}'
    assert as_text({"z": 1, "a": 2}) == '{"z":1,"a":2}'
