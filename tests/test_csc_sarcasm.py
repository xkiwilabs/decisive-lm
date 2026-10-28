"""Conversational Sarcasm Corpus: replies rated 1–6 for sarcasm by 4 or 6 other people."""
import csv

import pytest

from decisivelm.data.converters import csc_sarcasm
from decisivelm.data.dataset import check_licence, load_registry

COLUMNS = ["global_context_name", "context_type", "context_text", "response_text", "interlocutor_relationship",
           "global_speaker", "global_evaluator", "sarcasm_score_by_speaker", "sarcasm_score_by_evaluator"]


def _write(tmp_path, rows):
    with open(tmp_path / "data_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


@pytest.fixture
def source(tmp_path):
    scenario = "Steve said he would cook dinner. He ordered pizza."
    _write(tmp_path, [[101, "non_neutral", scenario, "Great cooking, chef.", "best_friend", 7, e, 6, s]
                      for e, s in zip(range(4), [6, 6, 5, 1])]
                     + [[101, "non_neutral", scenario, "Pizza is fine.", "colleague", 8, e, 1, 1] for e in range(4)])
    return tmp_path


def test_target_is_the_evaluators_shares_not_the_speakers_own_rating(source):
    rows = {r.lineage.source_id: r for r in csc_sarcasm.convert(source)}
    t = rows["101-7"].targets["sarcastic"].human
    assert t.probabilities == {"0": 0.25, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.25, "5": 0.5} and t.n_raters == 4
    assert rows["101-7"].state == ("Scenario: Steve said he would cook dinner. He ordered pizza.\n"
                                   "Steve is your best friend.\nReply to Steve: Great cooking, chef.")


def test_replies_to_one_scenario_share_a_family_and_split(source):
    a, b = csc_sarcasm.convert(source)
    assert a.lineage.family_id == b.lineage.family_id and a.split == b.split


def test_a_reply_whose_text_differs_between_evaluators_is_refused(tmp_path):
    _write(tmp_path, [[101, "neutral", "S.", "A.", "colleague", 7, 0, 1, 1],
                      [101, "neutral", "S.", "B.", "colleague", 7, 1, 1, 1]])
    with pytest.raises(ValueError, match="more than one text"):
        csc_sarcasm.convert(tmp_path)


def test_rows_match_the_registry(source):
    registry = load_registry()
    for r in csc_sarcasm.convert(source):
        check_licence(r, registry)
