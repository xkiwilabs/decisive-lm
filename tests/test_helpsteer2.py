"""Score seed: HelpSteer2, five 0–4 attributes per response, soft targets from per-annotator ratings."""
import pandas as pd
import pytest

from decisivelm.data.converters.helpsteer2 import ATTRIBUTES, convert
from decisivelm.data.dataset import check_licence, load_registry

MULTI = "c#\n<extra_id_1>Assistant\nHello, what is your question?\n<extra_id_1>User\nlist of panels"


@pytest.fixture
def rows(tmp_path):
    def row(prompt, response, **s):
        return {"prompt": prompt, "response": response, **{a: s.get(a, 2) for a in ATTRIBUTES}}
    pd.DataFrame([row(MULTI, "Panel, FlowLayoutPanel, TableLayoutPanel", helpfulness=3),
                  row(MULTI, "I don't know", helpfulness=0),
                  row("what is 2+2", "4", correctness=4)]).to_json(tmp_path / "train.jsonl.gz", orient="records", lines=True)
    pd.DataFrame([row("capital of France?", "Paris")]).to_json(tmp_path / "validation.jsonl.gz", orient="records", lines=True)
    pd.DataFrame([{"prompt": MULTI, "response": "Panel, FlowLayoutPanel, TableLayoutPanel",
                   **{a: "[2, 2, 2]" for a in ATTRIBUTES}, "helpfulness": "[3, 3, 4]"}]) \
        .to_json(tmp_path / "disagreements.jsonl.gz", orient="records", lines=True)
    return convert(tmp_path)


def test_state_is_the_conversation_and_the_response(rows):
    r = next(r for r in rows if r.state["response"].startswith("Panel"))
    assert r.state["conversation"] == "User: c#\nAssistant: Hello, what is your question?\nUser: list of panels"


def test_five_score_questions_with_five_levels(rows):
    q = rows[0].questions
    assert list(q) == list(ATTRIBUTES)
    assert all(x.type == "score" and len(x.criteria) == 5 for x in q.values())


def test_targets_come_from_individual_annotators_where_available(rows):
    r = next(r for r in rows if r.state["response"].startswith("Panel"))
    t = r.targets["helpfulness"].human
    assert t.probabilities == pytest.approx({"0": 0, "1": 0, "2": 0, "3": 2 / 3, "4": 1 / 3}) and t.n_raters == 3


def test_otherwise_the_released_score_is_the_target(rows):
    r = next(r for r in rows if r.state["response"] == "4")
    assert r.targets["correctness"].human.probabilities["4"] == 1.0


def test_responses_to_one_prompt_share_a_split(rows):
    splits = {r.split for r in rows if r.state["conversation"].startswith("User: c#")}
    assert len(splits) == 1
    assert next(r for r in rows if r.state["response"] == "Paris").split in {"gold_calib", "gold_test"}


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows:
        check_licence(r, registry)
