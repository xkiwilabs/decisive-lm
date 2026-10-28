"""Noul seed with human soft labels: Civil Comments (docs/guide/14 §14.2)."""
import pandas as pd
import pytest

from decisivelm.data.converters.civil_comments import ATTRIBUTES, convert
from decisivelm.data.dataset import check_licence, load_registry


def frame(texts, toxicity):
    d = {"text": texts, "toxicity": toxicity}
    for a in ATTRIBUTES:
        d.setdefault(a, [0.0] * len(texts))
    return pd.DataFrame(d)


@pytest.fixture
def rows(tmp_path):
    frame([f"train comment {i}" for i in range(20)] + ["   "], [i / 20 for i in range(20)] + [0.0]) \
        .to_parquet(tmp_path / "train-0.parquet")
    frame(["more train"], [0.5]).to_parquet(tmp_path / "train-1.parquet")
    frame([f"val comment {i}" for i in range(5)], [0.1] * 5).to_parquet(tmp_path / "validation.parquet")
    frame([f"test comment {i}" for i in range(8)], [0.7] * 8).to_parquet(tmp_path / "test.parquet")
    return convert(tmp_path, n_train=10, n_val=3, n_eval=6)


def test_one_noul_per_attribute_with_the_share_of_raters_as_target(rows):
    r = next(r for r in rows if r.state == "train comment 7")
    assert list(r.questions) == list(ATTRIBUTES)
    assert all(q.type == "noul" for q in r.questions.values())
    assert r.targets["toxicity"].human.noul == pytest.approx(7 / 20)


def test_samples_are_uniform_and_sized(rows):
    by = {s: [r for r in rows if r.split == s] for s in ("train", "val", "gold_calib", "gold_test")}
    assert len(by["train"]) == 10 and len(by["val"]) == 3
    assert len(by["gold_calib"]) + len(by["gold_test"]) == 6
    assert all(r.state.startswith("test comment") for r in by["gold_calib"] + by["gold_test"])


def test_sampling_is_deterministic(tmp_path, rows):
    assert [r.id for r in rows] == [r.id for r in convert(tmp_path, n_train=10, n_val=3, n_eval=6)]


def test_blank_comments_are_skipped(rows):
    assert all(r.state.strip() for r in rows)


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows:
        check_licence(r, registry)
