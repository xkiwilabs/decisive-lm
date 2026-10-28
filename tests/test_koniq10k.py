"""KonIQ-10k: photos rated for technical quality on five levels by about 107 raters each."""
import zipfile

import pytest

from decisivelm.data.converters import koniq10k
from decisivelm.data.dataset import check_licence, load_registry


@pytest.fixture
def rows(tmp_path):
    (tmp_path / "koniq10k_distributions_sets.csv").write_text(
        "image_name,c1,c2,c3,c4,c5,c_total,MOS,SD,set\n"
        "11.jpg,0.0,0.1,0.2,0.6,0.1,100,70,0.5,training\n"
        "22.jpg,0.5,0.5,0.0,0.0,0.0,98,20,0.4,test\n"
        "33.jpg,0.0,0.0,0.0,0.5,0.5,90,90,0.5,validation\n")
    (tmp_path / "1024x768").mkdir()
    for n in ("11", "22"):          # 33 has no image: left out
        (tmp_path / "1024x768" / f"{n}.jpg").write_bytes(b"\xff\xd8")
    return {r.lineage.source_id: r for r in koniq10k.convert(tmp_path)}


def test_target_is_the_raters_shares_on_five_levels(rows):
    t = rows["11"].targets["quality"].human
    assert t.probabilities == pytest.approx({"0": 0, "1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1}) and t.n_raters == 100
    assert rows["11"].questions["quality"].criteria == ["Bad", "Poor", "Fair", "Good", "Excellent"]
    assert "33" not in rows


def test_media_and_splits(rows):
    assert rows["11"].media[0].type == "image" and rows["11"].media[0].path.endswith("1024x768/11.jpg")
    assert rows["11"].split == "train" and rows["22"].split in {"gold_calib", "gold_test"}


def test_unpack_writes_only_expected_image_names(tmp_path):
    with zipfile.ZipFile(tmp_path / "1024x768_joined.zip", "w") as z:
        z.writestr("1024x768/5.jpg", b"ok")
        z.writestr("1024x768/../../evil.jpg", b"no")
        z.writestr("other/6.jpg", b"no")
    out = koniq10k.unpack(tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ["5.jpg"] and not (tmp_path.parent / "evil.jpg").exists()


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
