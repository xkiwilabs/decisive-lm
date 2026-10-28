"""Multi-rater reasoning seed: MultiNLI without fiction, label shares from the annotators."""
import json
import zipfile

import pytest

from decisivelm.data.converters import multinli as mn
from decisivelm.data.dataset import check_licence, load_registry


def _pair(pair_id, genre, labels, premise="The museum opens at nine.", hypothesis="The museum opens in the morning."):
    return {"pairID": pair_id, "promptID": pair_id[:-1], "genre": genre, "annotator_labels": labels,
            "gold_label": "-", "sentence1": premise, "sentence2": hypothesis}


@pytest.fixture
def rows(tmp_path):
    files = {"train": [_pair("1e", "travel", ["entailment"]), _pair("2c", "fiction", ["contradiction"])],
             "dev_matched": [_pair("3n", "slate", ["neutral", "entailment", "neutral", "contradiction", "entailment"])],
             "dev_mismatched": [_pair("4e", "letters", ["entailment"] * 4 + ["neutral"]),
                                _pair("4e", "letters", ["entailment"] * 5, hypothesis="It opens before ten."),
                                _pair("5n", "oup", ["", "", "", "", ""])]}
    files["train"].append(_pair("6c", "slate", ["contradiction"], premise=files["dev_matched"][0]["sentence1"]))
    files["dev_matched"][0]["sentence1"] = "Admission is free on Sundays."
    files["train"][-1]["sentence1"] = "Admission is free  on Sundays."   # same premise, other whitespace
    with zipfile.ZipFile(tmp_path / "multinli_1.0.zip", "w") as z:
        for split, pairs in files.items():
            z.writestr(f"multinli_1.0/{mn.FILES[split]}", "".join(json.dumps(p) + "\n" for p in pairs))
    return {r.lineage.source_id: r for r in mn.convert(tmp_path)}


def test_fiction_unlabelled_and_gold_premise_pairs_are_dropped_and_ids_are_unique(rows):
    assert set(rows) == {"train-1e", "dev_matched-3n", "dev_mismatched-4e", "dev_mismatched-4e-2"}


def test_targets_are_annotator_shares_and_ties_are_kept(rows):
    t = rows["dev_matched-3n"].targets["relation"].human
    assert t.probabilities == pytest.approx({"entailment": 0.4, "neutral": 0.4, "contradiction": 0.2})
    assert t.n_raters == 5
    single = rows["train-1e"].targets["relation"].human
    assert single.probabilities == {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0} and single.n_raters is None
    assert rows["dev_mismatched-4e"].state == {"premise": "The museum opens at nine.", "hypothesis": "The museum opens in the morning."}


def test_dev_matched_is_gold_and_the_rest_can_train(rows):
    assert rows["dev_matched-3n"].split in ("gold_calib", "gold_test")
    assert rows["train-1e"].split in ("train", "val") and rows["dev_mismatched-4e"].split in ("train", "val")


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
        assert r.tier == "R"


def test_gold_rows_ask_the_judgement_as_a_choice_a_noul_and_a_score(rows):
    gold = rows["dev_matched-3n"]
    assert list(gold.questions) == ["relation", "follows", "support"]
    assert gold.targets["follows"].human.noul == pytest.approx(0.4)
    assert gold.targets["support"].human.probabilities == pytest.approx({"0": 0.2, "1": 0.4, "2": 0.4})
    assert list(rows["train-1e"].questions) == ["relation"]
