"""Humicroedit: edited headlines graded 0–3 for funniness by about 5 people."""
import zipfile

import pytest

from decisivelm.data.converters import humicroedit
from decisivelm.data.dataset import check_licence, load_registry

HEADER = "id,original,edit,grades,meanGrade\n"


@pytest.fixture
def rows(tmp_path):
    files = {"train": '1,"Trump <praises/> the plan , aides say",ignores,33210,1.8\n',
             "dev": "2,Senate <votes/> on budget,dances,10000,0.2\n",
             "test": "3,The <Latest/> : storm hits coast,Sneeziest,2211000000,0.6\n",
             "train_funlines": "4,Mayor <opens/> bridge,eats,3,3.0\n5,Mayor <opens/> bridge,sings,43210,2.0\n"}
    with zipfile.ZipFile(tmp_path / humicroedit.ARCHIVE, "w") as z:
        for name, body in files.items():
            z.writestr(f"semeval-2020-task-7-dataset/subtask-1/{name}.csv", HEADER + body)
    return {r.lineage.source_id: r for r in humicroedit.convert(tmp_path)}


def test_state_shows_the_original_and_the_edited_headline(rows):
    assert rows["1"].state == ("Original headline: Trump praises the plan , aides say\n"
                               "Edited headline: Trump ignores the plan , aides say")


def test_target_is_the_graders_shares_and_too_few_or_off_scale_grades_are_dropped(rows):
    t = rows["1"].targets["funny"].human
    assert t.probabilities == {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.4} and t.n_raters == 5
    assert rows["3"].targets["funny"].human.n_raters == 10
    assert "4" not in rows and "5" not in rows


def test_the_tasks_splits_are_kept_and_its_test_split_is_gold(rows):
    assert (rows["1"].split, rows["2"].split) == ("train", "val")
    assert rows["3"].split in {"gold_calib", "gold_test"}


def test_rows_match_the_registry(rows):
    registry = load_registry()
    for r in rows.values():
        check_licence(r, registry)
