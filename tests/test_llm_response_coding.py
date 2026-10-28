"""Converter for the first use case: free-text intent responses coded by four humans.

The fixtures are invented: the codebook, responses and codes only have the study files' structure.
"""
import json
import pathlib

import pytest
from pydantic import ValidationError

from decisivelm.data.converters.llm_response_coding import codebook_question, convert, split_subjects
from decisivelm.data.schema import Record

FIX = pathlib.Path(__file__).parent / "fixtures" / "llm_response_coding"
CATEGORIES = ["Refund", "Repair", "Upgrade", "Cancel", "Product-Question", "Account-Change", "Multiple", "Other"]


@pytest.fixture(scope="module")
def rows():
    return {r.lineage.source_id: r for r in convert(FIX)}


def test_state_is_the_transcription_only(rows):
    assert rows["2001-0"].state == "how do I turn on the night mode"


def test_no_non_whitelisted_column_leaks(rows):
    dumped = json.dumps([r.model_dump() for r in rows.values()])
    assert "CANARY-" not in dumped


def test_human_target_is_the_share_of_coders(rows):
    p = rows["2001-0"].targets["intent"].human.probabilities
    assert p["Product-Question"] == pytest.approx(0.75) and p["Account-Change"] == pytest.approx(0.25)
    assert list(p) == CATEGORIES   # the codebook's order
    assert rows["2001-0"].targets["intent"].human.n_raters == 4


def test_missing_coder_is_left_out_of_the_share(rows):
    p = rows["2002-0"].targets["intent"].human.probabilities
    assert p["Multiple"] == pytest.approx(2 / 3) and p["Cancel"] == pytest.approx(1 / 3)
    assert rows["2002-0"].targets["intent"].human.n_raters == 3


def test_precedence_note_moves_from_multiple_to_the_instructions():
    q = codebook_question(FIX / "coder_instructions_new.md")
    assert "first action wins" in q["instructions"]
    assert "first action wins" not in q["criteria"]["Multiple"]
    assert "Two or more requests" in q["criteria"]["Multiple"]


def test_generation_only_instructions_are_dropped():
    q = codebook_question(FIX / "coder_instructions_new.md")
    assert "Output only" not in q["instructions"]


def test_participants_never_span_splits():
    rows = convert(FIX)
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r.lineage.family_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_subject.values())


def test_split_is_deterministic_and_covers_every_subject():
    subjects = list(range(2001, 2061))
    a, b = split_subjects(subjects, seed=0), split_subjects(subjects, seed=0)
    assert a == b and set(a) == set(subjects)
    counts = {s: list(a.values()).count(s) for s in set(a.values())}
    assert counts == {"train": 36, "val": 6, "gold_calib": 7, "gold_test": 11}


def test_records_are_private_tier_until_a_licence_is_chosen(rows):
    r = rows["2001-0"]
    assert r.tier == "X" and r.lineage.source_dataset == "llm_response_coding"


def test_schema_rejects_target_keys_that_do_not_match_the_options(rows):
    bad = rows["2001-0"].model_dump()
    bad["targets"]["intent"]["human"]["probabilities"] = {"Refund": 1.0}
    with pytest.raises(ValidationError):
        Record.model_validate(bad)


def test_schema_rejects_targets_for_unknown_questions(rows):
    bad = rows["2001-0"].model_dump()
    bad["targets"]["nope"] = bad["targets"]["intent"]
    with pytest.raises(ValidationError):
        Record.model_validate(bad)


def test_states_over_budget_are_truncated_and_flagged():
    from decisivelm.preprocess import MARKER
    from tiny_gemma import tokenizer
    rows = {r.lineage.source_id: r for r in convert(FIX, tokenizer=tokenizer(), max_state_tokens=6)}
    long, short = rows["2002-0"], rows["2001-1"]
    assert long.preprocess.truncated and MARKER in long.state and long.preprocess.max_state_tokens == 6
    assert not short.preprocess.truncated and short.state == "refund my order"


def test_a_label_outside_the_codebook_is_refused(tmp_path):
    for f in FIX.iterdir():
        (tmp_path / f.name).write_text(f.read_text())
    m = tmp_path / "full_coding_matrix.csv"
    m.write_text(m.read_text().replace("refund,refund,refund,refund", "refund,refund,refund,drink"))
    with pytest.raises(ValueError, match="unknown labels"):
        convert(tmp_path)
