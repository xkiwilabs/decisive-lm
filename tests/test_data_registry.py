"""Expected licences (configs/datasets.yaml) and enforced holdouts (configs/holdout.yaml): docs/guide/08 §8.5, §8.6."""
import pathlib

import pytest

from decisivelm.data.converters.llm_response_coding import convert
from decisivelm.data.dataset import check_licence, check_registry, load_registry, tier_of
from decisivelm.data.mix import check_mix, load_holdouts

FIX = pathlib.Path(__file__).parent / "fixtures" / "llm_response_coding"


@pytest.fixture(scope="module")
def rows():
    return convert(FIX)


@pytest.fixture(scope="module")
def train_rows(rows):
    return [r.model_copy(update={"split": "train"}) for r in rows]


def test_tier_follows_the_spdx_id():
    assert tier_of("CC-BY-4.0") == "R" and tier_of("Apache-2.0") == "R" and tier_of("CC0-1.0") == "R"
    assert tier_of("CC-BY-SA-4.0") == "S" and tier_of("ODbL-1.0") == "S"
    assert tier_of("CC-BY-NC-4.0") == "X" and tier_of("LicenseRef-pending") == "X"


def test_repo_registry_is_consistent():
    check_registry(load_registry())


def test_tier_above_the_licence_needs_an_override_reason():
    with pytest.raises(ValueError, match="override"):
        check_registry({"x": {"license": "CC-BY-SA-4.0", "tier": "R"}})
    check_registry({"x": {"license": "CC-BY-4.0", "tier": "X", "override": "third-party images inside"}})


def test_converted_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows:
        check_licence(r, registry)


def test_row_with_another_licence_or_dataset_is_refused(rows):
    registry = load_registry()
    with pytest.raises(ValueError, match="licence"):
        check_licence(rows[0].model_copy(update={"license": "CC-BY-4.0", "tier": "R"}), registry)
    with pytest.raises(ValueError, match="not in configs/datasets.yaml"):
        check_licence(rows[0], {})


def test_first_use_case_is_held_out_of_the_general_adapter_only(train_rows):
    holdouts = load_holdouts()
    with pytest.raises(ValueError, match="holdout"):
        check_mix(train_rows, holdouts, adapter="general")
    check_mix(train_rows, holdouts, adapter="user")


def test_selectors_narrow_a_holdout(train_rows):
    holdouts = [{"dataset": "llm_response_coding", "adapters": ["general"], "family_ids": ["lrc-subject-2001"]}]
    with pytest.raises(ValueError, match="lrc-2001-0"):
        check_mix(train_rows, holdouts, adapter="general")
    check_mix([r for r in train_rows if r.lineage.family_id != "lrc-subject-2001"], holdouts, adapter="general")


def test_massive_holds_out_ten_languages_and_keeps_the_rest(train_rows):
    [massive] = [h for h in load_holdouts() if h["dataset"] == "massive"]
    assert len(set(massive["languages"])) == 10 and "en-US" not in massive["languages"]
    row = train_rows[0].model_copy(update={"lineage": train_rows[0].lineage.model_copy(
        update={"source_dataset": "massive"})})
    with pytest.raises(ValueError, match="holdout"):
        check_mix([row.model_copy(update={"language": "sw-KE"})], [massive], adapter="general")
    check_mix([row.model_copy(update={"language": "tr-TR"})], [massive], adapter="general")


def test_tier_outside_the_mix_is_refused(train_rows):
    with pytest.raises(ValueError, match="tier"):
        check_mix(train_rows, [], adapter="user", tiers={"R"})


def test_only_train_rows_may_enter_a_training_mix(rows):
    gold = [r.model_copy(update={"split": "gold_test"}) for r in rows[:1]]
    with pytest.raises(ValueError, match="split"):
        check_mix(gold, [], adapter="user")
