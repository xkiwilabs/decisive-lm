import pandas as pd
import pytest

from decisivelm.data.converters import goemotions as ge


def _rater(cid, text, rater, *marked):
    return {"id": cid, "text": text, "rater_id": rater, **{e: int(e in marked) for e in ge.EMOTIONS}}


@pytest.fixture
def source(tmp_path):
    rows = [_rater("a1", "Thanks so much!", 1, "gratitude"), _rater("a1", "Thanks so much!", 2, "gratitude", "joy"),
            _rater("a1", "Thanks so much!", 3, "neutral"),
            _rater("b2", "Why would you do that?", 1, "confusion", "anger"), _rater("b2", "Why would you do that?", 2),
            _rater("c3", "ok", 1, "neutral"), _rater("c3", "ok", 2, "neutral")]
    pd.DataFrame(rows).to_parquet(tmp_path / "raw.parquet")
    pd.DataFrame({"id": ["b2"], "text": ["x"], "labels": [[6]]}).to_parquet(tmp_path / "simplified_validation.parquet")
    pd.DataFrame({"id": ["c3"], "text": ["x"], "labels": [[27]]}).to_parquet(tmp_path / "simplified_test.parquet")
    return tmp_path


def test_splits_follow_the_simplified_eval_ids(source):
    rows = {r.lineage.source_id: r for r in ge.convert(source, val_share=0.0)}
    assert rows["a1"].split == "train" and rows["b2"].split == "gold_calib" and rows["c3"].split == "gold_test"


def test_targets_are_rater_shares(source):
    rows = {r.lineage.source_id: r for r in ge.convert(source, nouls=len(ge.EMOTIONS), val_share=0.0)}
    a = rows["a1"]
    assert a.targets["gratitude"].human.noul == pytest.approx(2 / 3)
    assert a.targets["gratitude"].human.n_raters == 3
    # rater 2 marked two positive emotions (one group), rater 3 neutral
    assert a.targets["sentiment"].human.probabilities == pytest.approx(
        {"positive": 2 / 3, "negative": 0.0, "ambiguous": 0.0, "neutral": 1 / 3})
    # rater 1 split confusion (ambiguous) / anger (negative); rater 2 marked nothing and is left out
    assert rows["b2"].targets["sentiment"].human.probabilities == pytest.approx(
        {"positive": 0.0, "negative": 0.5, "ambiguous": 0.5, "neutral": 0.0})


def test_eval_rows_ask_every_emotion_train_rows_neutral_and_a_uniform_sample(source):
    rows = {r.lineage.source_id: r for r in ge.convert(source, nouls=6, val_share=0.0)}
    assert set(rows["b2"].questions) == set(ge.EMOTIONS) | {"sentiment"}
    asked = [q for q in rows["a1"].questions if q != "sentiment"]
    assert len(asked) == 6 and "neutral" in asked
    assert asked == [e for e in ge.EMOTIONS if e in asked]
    assert ge.convert(source, nouls=6, val_share=0.0)[0].questions.keys() == rows["a1"].questions.keys()
