"""CREMA-D: perceived emotion in acted speech, from listeners who heard the voice without the video."""
import pytest

from decisivelm.data.converters import crema_d
from decisivelm.data.dataset import check_licence, load_registry

HEADER = '"","localid","pos","ans","ttr","queryType","numTries","clipNum","questNum","subType","clipName",' \
         '"sessionNums","respEmo","respLevel","dispEmo","dispVal","dispLevel"\n'


def _row(rater, query, clip, emo):
    return f'"1","{rater}",1,"{emo}_50","100",{query},0,1,1,4,"{clip}",1,"{emo}","50","S",50,"X"\n'


@pytest.fixture
def rows(tmp_path):
    clip, other = "1010_MTI_SAD_XX", "1011_DFA_ANG_HI"
    lines = [_row("a", 1, clip, "S"), _row("b", 1, clip, "S"), _row("c", 1, clip, "N"), _row("d", 1, clip, "F"),
             _row("e", 2, clip, "H"), _row("f", 3, clip, "H"),            # face-only and audio-visual: not used
             _row("a", 1, other, "A"), _row("b", 1, other, "A"),
             _row("a", 1, "1012_IEO_HAP_XX", "H"), _row("b", 1, "1012_IEO_HAP_XX", "H")]   # no WAV: left out
    (tmp_path / "finishedResponses.csv").write_text(HEADER + "".join(lines))
    (tmp_path / "AudioWAV").mkdir()
    for c in (clip, other):
        (tmp_path / "AudioWAV" / f"{c}.wav").write_bytes(b"RIFF")
    return {r.lineage.source_id: r for r in crema_d.convert(tmp_path)}


def test_target_is_the_voice_only_listeners_shares(rows):
    t = rows["1010_MTI_SAD_XX"].targets["emotion"].human
    assert t.n_raters == 4 and t.probabilities["sad"] == 0.5 and t.probabilities["neutral"] == 0.25
    assert t.probabilities["happy"] == 0 and sum(t.probabilities.values()) == pytest.approx(1)
    assert "1012_IEO_HAP_XX" not in rows


def test_state_is_the_sentence_and_the_media_is_the_clip(rows):
    r = rows["1010_MTI_SAD_XX"]
    assert r.state == 'An actor says: "Maybe tomorrow it will be cold."'
    assert r.media[0].type == "audio" and r.media[0].path.endswith("AudioWAV/1010_MTI_SAD_XX.wav")
    assert list(r.questions["emotion"].criteria) == ["anger", "disgust", "fear", "happy", "neutral", "sad"]


def test_split_is_by_actor(rows):
    assert rows["1010_MTI_SAD_XX"].lineage.family_id == "crema_d-actor-1010"
    assert rows["1010_MTI_SAD_XX"].split == crema_d._split("1010")


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
