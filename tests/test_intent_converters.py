"""Choice seeds: Banking77 and CLINC150 (docs/guide/14 §14.2)."""
import pathlib

import pandas as pd
import pytest

from decisivelm.data.converters.intents import OTHER, convert_banking77, convert_clinc150, item_split
from decisivelm.data.dataset import check_licence, load_registry

README = """---
dataset_info:
- config_name: plus
  features:
  - name: text
    dtype: string
  - name: intent
    dtype:
      class_label:
        names:
          '0': restaurant_reviews
          '1': oos
          '2': change_speed
---
"""


@pytest.fixture
def banking(tmp_path):
    pd.DataFrame({"text": ["I am still waiting on my card?", "Is there a fee for top up?", "My card was stolen"] * 4,
                  "category": ["card_arrival", "top_up_fee", "lost_or_stolen_card"] * 4}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"text": ["Card not here yet", "How much to top up?"], "category": ["card_arrival", "top_up_fee"]}) \
        .to_csv(tmp_path / "test.csv", index=False)
    return convert_banking77(tmp_path)


@pytest.fixture
def clinc(tmp_path):
    (tmp_path / "README.md").write_text(README)
    pd.DataFrame({"text": ["was the food good there", "what's the weather on mars", "talk faster"], "intent": [0, 1, 2]}) \
        .to_parquet(tmp_path / "train.parquet")
    pd.DataFrame({"text": ["speak slower please"], "intent": [2]}).to_parquet(tmp_path / "validation.parquet")
    pd.DataFrame({"text": ["is that place any good", "who won the 1902 cup"], "intent": [0, 1]}).to_parquet(tmp_path / "test.parquet")
    return convert_clinc150(tmp_path)


def test_banking_question_lists_every_intent_with_a_readable_description(banking):
    q = banking[0].questions["intent"]
    assert list(q.criteria) == ["card_arrival", "lost_or_stolen_card", "top_up_fee"]
    assert q.criteria["lost_or_stolen_card"] == "Lost or stolen card"


def test_banking_target_is_the_gold_intent(banking):
    r = next(r for r in banking if r.state == "My card was stolen")
    assert r.targets["intent"].human.probabilities == {"card_arrival": 0.0, "lost_or_stolen_card": 1.0, "top_up_fee": 0.0}


def test_source_test_rows_only_go_to_gold_splits(banking):
    by_source = {r.lineage.source_id.split("-")[0]: set() for r in banking}
    for r in banking:
        by_source[r.lineage.source_id.split("-")[0]].add(r.split)
    assert by_source["test"] <= {"gold_calib", "gold_test"}
    assert by_source["train"] <= {"train", "val"}


def test_split_is_a_stable_function_of_the_item():
    assert item_split("banking77", "train-00012", "train") == item_split("banking77", "train-00012", "train")
    shares = [item_split("x", f"test-{i}", "test") for i in range(2000)]
    assert 0.35 < shares.count("gold_calib") / 2000 < 0.45


def test_clinc_out_of_scope_becomes_a_self_contained_other(clinc):
    q = clinc[0].questions["intent"]
    assert list(q.criteria) == ["restaurant_reviews", "change_speed", OTHER]
    assert q.criteria[OTHER].startswith("Out of scope") and "listed" not in q.criteria[OTHER]
    mars = next(r for r in clinc if "mars" in r.state)
    assert mars.targets["intent"].human.probabilities[OTHER] == 1.0


def test_clinc_validation_rows_are_val(clinc):
    assert next(r for r in clinc if r.state == "speak slower please").split == "val"


def test_rows_match_the_registry(banking, clinc):
    registry = load_registry()
    for r in banking + clinc:
        check_licence(r, registry)
