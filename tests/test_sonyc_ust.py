"""SONYC-UST: which sounds three volunteers hear in a city street recording, from the sound alone."""
import io
import tarfile

import pytest

from decisivelm.data.converters import sonyc_ust
from decisivelm.data.dataset import check_licence, load_registry

FINE = ["1-1_small-sounding-engine", "1-X_engine-of-uncertain-size", "7-1_person-or-small-group-talking",
        "8-1_dog-barking-whining"]


def row(split, sensor, clip, annotator, marked):
    return ",".join([split, sensor, clip, str(annotator), "Manhattan", "1", "40.7", "-74.0"]
                    + ["1" if f in marked else "0" for f in FINE]) + "\n"


@pytest.fixture
def source(tmp_path):
    head = "split,sensor_id,audio_filename,annotator_id,borough,block,latitude,longitude," + ",".join(
        f"{f}_presence" for f in FINE) + "\n"
    (tmp_path / "annotations.csv").write_text(
        head
        + row("train", "1", "a.wav", 11, {"1-1_small-sounding-engine"})
        + row("train", "1", "a.wav", 12, {"1-X_engine-of-uncertain-size", "8-1_dog-barking-whining"})
        + row("train", "1", "a.wav", 13, set())
        + row("train", "1", "a.wav", -5, {"7-1_person-or-small-group-talking"})       # SONYC team: not used
        + row("train", "1", "a.wav", 0, {"7-1_person-or-small-group-talking"})        # agreed truth: not used
        + row("test", "2", "b.wav", 14, set()) + row("test", "2", "b.wav", 15, set())
        + row("validate", "3", "c.wav", 16, set()) + row("validate", "3", "c.wav", 17, set()))  # c.wav: no audio
    with tarfile.open(tmp_path / "audio-0.tar.gz", "w:gz") as t:
        for name, data in [("audio-0/a.wav", b"RIFFa"), ("audio-0/b.wav", b"RIFFb")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return tmp_path


def test_targets_are_the_volunteers_shares_per_coarse_class(source):
    rows = {r.lineage.source_id: r for r in sonyc_ust.convert(source)}
    assert set(rows) == {"a.wav", "b.wav"}
    a = rows["a.wav"]
    assert len(a.questions) == 8 and a.state == sonyc_ust.STATE
    assert a.targets["engine"].human.noul == pytest.approx(2 / 3) and a.targets["engine"].human.n_raters == 3
    assert a.targets["dog"].human.noul == pytest.approx(1 / 3) and a.targets["human_voice"].human.noul == 0
    assert a.media[0].path.endswith("audio/a.wav") and (source / "audio" / "a.wav").read_bytes() == b"RIFFa"
    assert "Manhattan" not in a.model_dump_json() and "40.7" not in a.model_dump_json()
    assert a.split == "train" and rows["b.wav"].split in ("gold_calib", "gold_test")


def test_rows_match_the_registry(source):
    registry = load_registry()
    for r in sonyc_ust.convert(source):
        check_licence(r, registry)
