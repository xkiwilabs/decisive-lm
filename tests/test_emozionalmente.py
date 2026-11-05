"""Emozionalmente: the emotion 5 listeners hear in acted Italian speech, from the voice alone."""
import zipfile

import pytest

from decisivelm.data.converters import emozionalmente
from decisivelm.data.dataset import check_licence, load_registry

R = "emozionalmente_dataset/"


@pytest.fixture
def source(tmp_path):
    with zipfile.ZipFile(tmp_path / emozionalmente.ARCHIVE, "w") as z:
        z.writestr(R + "metadata/samples.csv", "file_name,sentence,sentence_language,actor,emotion_expressed\n"
                   "a.wav,Il treno parte alle otto.,italian,7,sadness\n"
                   "b.wav,Il treno parte alle otto.,italian,8,joy\n"
                   "c.wav,Il treno parte alle otto.,italian,9,fear\n")
        z.writestr(R + "metadata/evaluations.csv", "file_name,evaluator,audio_quality,emotion_recognized\n"
                   + "".join(f"a.wav,{i},good,{e}\n" for i, e in enumerate(
                       ["sadness", "sadness", "sadness", "neutrality", "fear"]))
                   + "b.wav,1,good,joy\nb.wav,2,bad,joy\n"
                   + "c.wav,1,good,fear\nc.wav,2,good,fear\n")            # c.wav has no audio: left out
        z.writestr(R + "metadata/users.csv", "this file must never be read\n")
        z.writestr(R + "audio/a.wav", b"RIFFa")
        z.writestr(R + "audio/b.wav", b"RIFFb")
    return tmp_path


def test_targets_are_the_listeners_shares_and_clips_are_extracted(source):
    rows = {r.lineage.source_id: r for r in emozionalmente.convert(source)}
    assert set(rows) == {"a.wav", "b.wav"}
    t = rows["a.wav"].targets["emotion"].human
    assert t.n_raters == 5 and t.probabilities["sadness"] == 0.6 and t.probabilities["joy"] == 0
    assert sum(t.probabilities.values()) == pytest.approx(1)
    a = rows["a.wav"]
    assert a.state == 'An actor says, in Italian: "Il treno parte alle otto."' and a.language == "it"
    assert a.media[0].path.endswith("audio/a.wav") and (source / "audio" / "a.wav").read_bytes() == b"RIFFa"
    assert len(a.questions["emotion"].criteria) == 7


def test_users_file_is_never_opened(source, monkeypatch):
    opened = []
    real = zipfile.ZipFile.open
    monkeypatch.setattr(zipfile.ZipFile, "open",
                        lambda self, name, *a, **k: opened.append(name) or real(self, name, *a, **k))
    emozionalmente.convert(source)
    assert not any("users.csv" in n for n in opened)


def test_rows_match_the_registry(source):
    registry = load_registry()
    for r in emozionalmente.convert(source):
        check_licence(r, registry)
