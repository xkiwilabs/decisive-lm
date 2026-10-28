"""HelpSteer2-Preference: which of two responses is better, by how much, rated by several annotators."""
import gzip
import json

import pytest

from decisivelm.data.converters import helpsteer2, helpsteer2_preference as hp
from decisivelm.data.dataset import check_licence, load_registry


def _pair(split, prompt, ratings):
    return {"split": split, "prompt": prompt, "response_1": "one", "response_2": "two",
            "preference_strength": 0, "all_preferences_unprocessed": [{"strength": r} for r in ratings]}


@pytest.fixture
def rows(tmp_path):
    pairs = [_pair("train", "hi\n<extra_id_1>Assistant\nhello\n<extra_id_1>User\nhow are you", [-1, -1, 2]),
             _pair("val", "what is 2+2", [1, 1, 1, 3]), _pair("train", "lonely", [2])]
    with gzip.open(tmp_path / "preference.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    return {r.id: r for r in hp.convert(tmp_path)}


def test_target_is_the_shares_of_every_rating_and_single_ratings_are_dropped(rows):
    t = rows["helpsteer2_preference-train-00000"].targets["preference"].human
    assert t.probabilities == pytest.approx({"0": 0, "1": 0, "2": 2 / 3, "3": 0, "4": 1 / 3, "5": 0}) and t.n_raters == 3
    assert not any("00002" in i for i in rows)


def test_swapped_copy_reverses_the_responses_and_the_levels(rows):
    a, b = rows["helpsteer2_preference-train-00000"], rows["helpsteer2_preference-train-00000-swap"]
    assert (b.state["response_1"], b.state["response_2"]) == (a.state["response_2"], a.state["response_1"])
    pa, pb = a.targets["preference"].human.probabilities, b.targets["preference"].human.probabilities
    assert [pb[str(5 - i)] for i in range(6)] == [pa[str(i)] for i in range(6)]
    assert b.lineage.multiplier == "swap" and b.split == a.split and b.lineage.family_id == a.lineage.family_id


def test_split_and_family_follow_helpsteer2_by_prompt(rows):
    r = rows["helpsteer2_preference-val-00001"]
    assert r.split == helpsteer2._split("what is 2+2", "validation") and r.split in {"gold_calib", "gold_test"}
    assert r.lineage.family_id.startswith("helpsteer2-prompt-")
    assert rows["helpsteer2_preference-train-00000"].state["conversation"] == \
        "User: hi\nAssistant: hello\nUser: how are you"


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
