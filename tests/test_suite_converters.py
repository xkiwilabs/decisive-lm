"""Converters for the frozen held-out suite (docs/RELEASE.md): HelpSteer3, Measuring Hate Speech, CUAD, MASSIVE."""
import gzip
import io
import json
import tarfile

import pandas as pd
import pytest

from decisivelm.data.converters import cuad, helpsteer3, massive, measuring_hate_speech as mhs
from decisivelm.data.mix import check_mix, load_holdouts


def _mhs_rater(cid, text, hate, **items):
    return {"comment_id": cid, "text": text, "annotator_id": 0, "annotator_gender": "x",
            **{k: items.get(k, 0) for k in mhs.SCORES}, "hatespeech": hate}


def test_mhs_targets_are_rater_shares_and_only_whitelisted_columns_are_read(tmp_path, monkeypatch):
    rows = [_mhs_rater(7, "a comment", 2, insult=4), _mhs_rater(7, "a comment", 1, insult=3),
            _mhs_rater(7, "a comment", 0, insult=3), _mhs_rater(9, "another", 0)]
    pd.DataFrame(rows).to_parquet(tmp_path / "train.parquet")
    monkeypatch.setattr(mhs, "EVAL_SHARE", 1.0)
    out = {r.lineage.source_id: r for r in mhs.convert(tmp_path)}
    a, b = out["00007"], out["00009"]
    assert a.targets["insult"].human.probabilities == pytest.approx({"0": 0, "1": 0, "2": 0, "3": 2 / 3, "4": 1 / 3})
    assert a.targets["hatespeech"].human.noul == pytest.approx(0.5)   # yes, unclear (half), no
    assert a.targets["insult"].human.n_raters == 3
    # only comments with >= 3 raters can be in the eval portion
    assert (a.split, a.lineage.family_id) == ("gold_test", "measuring_hate_speech-eval")
    assert b.split in ("train", "val") and b.lineage.family_id == "measuring_hate_speech-base"
    assert "gender" not in a.model_dump_json()


def test_mhs_eval_portion_is_held_out_of_training(tmp_path, monkeypatch):
    pd.DataFrame([_mhs_rater(7, "a comment", 0)] * 3).to_parquet(tmp_path / "train.parquet")
    monkeypatch.setattr(mhs, "EVAL_SHARE", 1.0)
    [row] = mhs.convert(tmp_path)
    with pytest.raises(ValueError, match="holdout"):
        check_mix([row.model_copy(update={"split": "train"})], load_holdouts(), adapter="general")


def _cuad_source(tmp_path, text, spans):
    qas = [{"id": f"DOC__{c}", "question": f'Highlight the parts related to "{c}". Details: Is there a {c}?',
            "answers": [{"text": text[a:b], "answer_start": a} for a, b in spans.get(c, [])]}
           for c in cuad.HELD_OUT + ["Parties"]]
    (tmp_path / "CUAD_v1.json").write_text(json.dumps(
        {"data": [{"title": "DOC", "paragraphs": [{"context": text, "qas": qas}]}]}))


def test_cuad_sections_are_cut_at_line_breaks_and_cover_the_text():
    text = ("x" * 30 + "\n") * 100
    cuts = cuad.sections(text, limit=400)
    assert cuts[0][0] == 0 and cuts[-1][1] == len(text)
    assert all(b == c for (_, b), (c, _) in zip(cuts, cuts[1:]))
    assert all(text[b - 1] == "\n" for _, b in cuts) and all(b - a <= 400 for a, b in cuts)


def test_cuad_label_is_inside_or_absent_and_a_cut_span_is_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(cuad, "SECTION_CHARS", 100)
    text = ("a" * 49 + "\n") * 4   # sections [0, 100) and [100, 200)
    _cuad_source(tmp_path, text, {"Non-Compete": [(10, 30)], "Audit Rights": [(90, 120)]})
    first, second = cuad.convert(tmp_path)
    assert first.targets["non_compete"].human.noul == 1.0 and second.targets["non_compete"].human.noul == 0.0
    assert "audit_rights" not in first.questions and "audit_rights" not in second.questions   # straddles the cut
    assert "parties" not in first.questions and len(first.questions) == len(cuad.HELD_OUT) - 1
    assert {first.split, second.split} == {"gold_test"}


def _massive_source(tmp_path, locales):
    with tarfile.open(tmp_path / massive.ARCHIVE, "w:gz") as tar:
        for loc in locales:
            lines = "".join(json.dumps({"id": str(i), "locale": loc, "partition": p, "intent": it, "utt": f"{loc} {i}"})
                            + "\n" for i, (p, it) in enumerate([("train", "alarm_set"), ("test", "alarm_set"),
                                                                ("test", "play_music")]))
            data = lines.encode()
            info = tarfile.TarInfo(f"1.1/data/{loc}.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_massive_takes_the_test_partition_of_the_held_out_locales(tmp_path):
    _massive_source(tmp_path, ["pl-PL", "ko-KR"])
    rows = massive.convert(tmp_path, locales=["pl-PL", "ko-KR"])
    assert len(rows) == 4 and {r.split for r in rows} == {"gold_test"}
    assert {r.language for r in rows} == {"pl-PL", "ko-KR"}
    # the same utterance id is one family across locales (MASSIVE is parallel)
    assert len({r.lineage.family_id for r in rows}) == 2
    assert rows[0].questions["intent"].criteria["alarm_set"] == "Alarm set"
    with pytest.raises(ValueError, match="holdout"):
        check_mix([rows[0].model_copy(update={"split": "train"})], load_holdouts(), adapter="general")


def test_massive_held_out_locales_come_from_the_holdout_config():
    assert massive.held_out_locales() == [h for h in load_holdouts() if h["dataset"] == "massive"][0]["languages"]


def _hs3_source(tmp_path, n):
    for split in ("train", "validation"):
        with gzip.open(tmp_path / f"preference_{split}.jsonl.gz", "wt", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({
                    "domain": "multilingual" if i % 2 else "code", "language": "polish" if i % 2 else "python",
                    "context": [{"role": "user", "content": f"q{i}"}], "response1": "r1", "response2": "r2",
                    "overall_preference": -1, "individual_preference": [{"score": -2}, {"score": -1}, {"score": 1}],
                }) + "\n")


def test_helpsteer3_targets_use_the_annotators_six_level_scale(tmp_path):
    _hs3_source(tmp_path, 3)
    rows = helpsteer3.convert(tmp_path, n_items=6)
    p = rows[0].targets["preference"].human.probabilities
    assert p == pytest.approx({"0": 0, "1": 1 / 3, "2": 1 / 3, "3": 1 / 3, "4": 0, "5": 0})
    assert len(rows[0].questions["preference"].criteria) == 6
    assert {r.language for r in rows} == {"pl", "en"}   # the code domain's `language` is a programming language
    assert rows[0].state["conversation"] == "User: q0" or rows[0].state["conversation"].startswith("User: q")


def test_helpsteer3_suite_sample_is_fixed(tmp_path):
    _hs3_source(tmp_path, 10)
    first = [r.id for r in helpsteer3.convert(tmp_path, n_items=7)]
    assert first == [r.id for r in helpsteer3.convert(tmp_path, n_items=7)] and len(first) == 7
