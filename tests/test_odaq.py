"""ODAQ: how good a processed excerpt sounds against its original, from 26 expert listeners' MUSHRA scores."""
import zipfile

import pytest

from decisivelm.data.converters import odaq
from decisivelm.data.dataset import check_licence, load_registry

T = odaq.TEST
LICENCES = (";content;original file name;source;author;license;notes\n"
            "11_guitar.wav;guitar;x.wav;freesound;someone;https://creativecommons.org/licenses/by/4.0/;cut\n"
            "39_clapping.wav;claps;y.m4a;freesound;someone;http://creativecommons.org/publicdomain/zero/1.0/;cut\n"
            ";;;;;;\n"
            "AmateurOnPurpose.wav;music;n/a;Netflix own collection;Netflix;"
            "https://creativecommons.org/licenses/by-nc/4.0/;new\n")


@pytest.fixture
def source(tmp_path):
    rows = ["score,method,condition,process,item,subject"]
    for item in ("LP_11_guitar", "PE_39_clapping", "LP_AmateurOnPurpose"):
        rows += [f"{s},LP,LP3.5,LP35,{item},Subject {i}" for i, s in enumerate([10, 30, 35, 70, 100])]
        rows += [f"{s},LP,Ref,reference,{item},Subject {i}" for i, s in enumerate([100, 95, 90, 100, 85])]
    with zipfile.ZipFile(tmp_path / odaq.ARCHIVE, "w") as z:
        z.writestr("ODAQ/_detailed_license.csv", LICENCES)
        z.writestr(T + "ODAQ_results.csv", "\n".join(rows) + "\n")
        for item in ("LP_11_guitar", "PE_39_clapping", "LP_AmateurOnPurpose"):
            z.writestr(f"{T}{item}/reference.wav", b"RIFFref")
            z.writestr(f"{T}{item}/LP35.wav", b"RIFFlp")
    return tmp_path


def test_targets_are_the_listeners_bands_and_only_cc_by_or_cc0_excerpts_are_kept(source):
    rows = {r.lineage.source_id: r for r in odaq.convert(source)}
    assert set(rows) == {"LP_11_guitar/LP35", "LP_11_guitar/reference",
                         "PE_39_clapping/LP35", "PE_39_clapping/reference"}       # the CC BY-NC excerpt is left out
    t = rows["LP_11_guitar/LP35"].targets["quality"].human
    assert t.n_raters == 5 and t.probabilities == {"0": 0.2, "1": 0.4, "2": 0, "3": 0.2, "4": 0.2}
    assert rows["LP_11_guitar/reference"].targets["quality"].human.probabilities["4"] == 1
    r = rows["LP_11_guitar/LP35"]
    assert [m.path.split("listening_test/")[1] for m in r.media] == ["LP_11_guitar/reference.wav",
                                                                    "LP_11_guitar/LP35.wav"]
    assert (source / "listening_test" / "LP_11_guitar" / "LP35.wav").read_bytes() == b"RIFFlp"
    assert len(r.questions["quality"].criteria) == 5
    assert {x.split for x in rows.values()} <= {"gold_calib", "gold_test"}
    assert rows["LP_11_guitar/LP35"].split == rows["LP_11_guitar/reference"].split


def test_rows_match_the_registry(source):
    registry = load_registry()
    for r in odaq.convert(source):
        check_licence(r, registry)
