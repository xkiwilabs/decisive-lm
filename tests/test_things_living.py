"""THINGSplus: is the object "something that lives", 7 levels, about 37 raters per object."""
import zipfile

import pytest

from decisivelm.data.converters import things_living
from decisivelm.data.dataset import check_licence, load_registry

HEADER = "Word\tuniqueID\tmanmade\tprecious\tlives\theavy\tnatural\tmoves\tgrasp\thold\tbe.moved\tpleasant\tarousing\tsubject_Nr\twork_time\tage\tgender"


def line(word, uid, lives, subject):
    return '"' + "\t".join([word, uid, "4", "4", str(lives), "4", "4", "4", "4", "4", "4", "4", "4", str(subject), "50", "30", "x"]) + '"'


@pytest.fixture
def rows(tmp_path):
    lines = ['"' + HEADER + '"']
    lines += [line("apple", "apple", v, i) for i, v in enumerate([7, 6, 6, 5, 2, 1])]
    lines += [line("rock", "rock", 1, i) for i in range(4)]
    lines += [line("tree", "tree", 7, i) for i in range(4)]   # no photo: left out
    (tmp_path / "object-properties_trial-wise_thingsplus.tsv").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(tmp_path / "images_THINGSplus-CC0.zip", "w") as z:
        for uid in ("apple", "rock"):
            z.writestr(f"object_images_CC0/{uid}.jpg", b"\xff\xd8")
        z.writestr("object_images_CC0/../../evil.jpg", b"no")
    return {r.lineage.source_id: r for r in things_living.convert(tmp_path)}


def test_target_is_the_raters_shares_on_seven_levels(rows):
    t = rows["apple"].targets["lives"].human
    assert t.n_raters == 6 and t.probabilities["6"] == pytest.approx(1 / 6) and t.probabilities["5"] == pytest.approx(2 / 6)
    assert rows["rock"].targets["lives"].human.probabilities["0"] == 1
    assert len(rows["apple"].questions["lives"].criteria) == 7 and "tree" not in rows


def test_state_names_the_object_and_the_photo_is_attached(rows):
    r = rows["apple"]
    assert r.state == "The object: apple." and r.media[0].path.endswith("object_images_CC0/apple.jpg")
    assert r.split in {"train", "val", "gold_calib", "gold_test"}


def test_only_expected_image_names_are_unpacked(tmp_path, rows):
    assert sorted(p.name for p in (tmp_path / "object_images_CC0").iterdir()) == ["apple.jpg", "rock.jpg"]
    assert not (tmp_path.parent / "evil.jpg").exists()


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
