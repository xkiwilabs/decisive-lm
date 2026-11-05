"""Photo question answering (VizWiz-VQA, TextVQA): ten crowd answers become a Choice with a fixed number of options."""
import json
import zipfile

import pytest

from decisivelm.data.converters import textvqa, vizwiz_vqa, vqa
from decisivelm.data.dataset import check_licence, load_registry

POOL = ["blue", "coca cola", "seven", "stop", "thursday", "tuesday"]


def test_options_are_the_most_given_answers_filled_up_to_four():
    question, p, n = vqa.choice(["Dakota.", "dakota", "Dakota digital", "dakota", "unanswerable"], POOL, "k")
    options = {question["criteria"][k]: p[k] for k in question["criteria"]}
    assert len(options) == vqa.OPTIONS and n == 4
    assert options["dakota"] == 0.75 and options["dakota digital"] == 0.25
    assert sum(options.values()) == pytest.approx(1) and "unanswerable" not in options
    assert all(o in POOL for o in options if not o.startswith("dakota"))


def test_option_count_never_depends_on_disagreement():
    unanimous = vqa.choice(["stop"] * 10, POOL, "k")[0]
    split = vqa.choice(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"], POOL, "k")
    assert len(unanimous["criteria"]) == len(split[0]["criteria"]) == vqa.OPTIONS and split[2] == 8


def test_yes_no_questions_have_two_options_and_scattered_answers_are_skipped():
    question, p, n = vqa.choice(["yes", "Yes", "no", "maybe"], POOL, "k", yes_no=True)
    assert sorted(question["criteria"].values()) == ["no", "yes"] and n == 3
    assert vqa.choice([str(i) for i in range(10)], POOL, "k") is None      # four options cover 4 of 10 answerers


@pytest.fixture
def textvqa_source(tmp_path):
    data = [{"question_id": 1, "image_id": "003a8ae2ef43b901", "question": "what brand is this camera?",
             "answers": ["dakota"] * 7 + ["dakota digital"] * 3},
            {"question_id": 2, "image_id": "../../etc/passwd", "question": "x?", "answers": ["a"] * 10},
            {"question_id": 3, "image_id": "00000000000000ff", "question": "no photo?", "answers": ["b"] * 10}]
    (tmp_path / textvqa.ANNOTATIONS).write_text(json.dumps({"data": data}))
    with zipfile.ZipFile(tmp_path / textvqa.ARCHIVE, "w") as z:
        z.writestr("train_images/003a8ae2ef43b901.jpg", b"JPEG")
    return tmp_path


def test_textvqa_rows(textvqa_source):
    rows = textvqa.convert(textvqa_source)
    assert [r.lineage.source_id for r in rows] == ["1"]           # a path-like id and a missing photo are skipped
    r = rows[0]
    assert r.state == "Question: what brand is this camera?" and r.split in ("gold_calib", "gold_test")
    assert (textvqa_source / "images" / "003a8ae2ef43b901.jpg").read_bytes() == b"JPEG"
    assert max(r.targets["answer"].human.probabilities.values()) == 0.7
    check_licence(r, load_registry())


@pytest.fixture
def vizwiz_source(tmp_path):
    def item(image, answers, answer_type="other"):
        return {"image": image, "question": "What is this?", "answer_type": answer_type,
                "answers": [{"answer": a, "answer_confidence": "yes"} for a in answers]}
    parts = {"train": [item("VizWiz_train_00000000.jpg", ["soup"] * 6 + ["unanswerable"] * 4),
                       item("VizWiz_train_00000001.jpg", ["unanswerable"] * 9 + ["blank screen"])],
             "val": [item("VizWiz_val_00000000.jpg", ["yes"] * 5 + ["no"] * 3 + ["unanswerable"] * 2, "yes/no")]}
    with zipfile.ZipFile(tmp_path / "Annotations.zip", "w") as z:
        for part, data in parts.items():
            z.writestr(f"{part}.json", json.dumps(data))
    for part, data in parts.items():
        with zipfile.ZipFile(tmp_path / f"{part}.zip", "w") as z:
            for d in data:
                z.writestr(f"{part}/{d['image']}", b"JPEG")
    return tmp_path


def test_vizwiz_rows(vizwiz_source):
    rows = {r.lineage.source_id: r for r in vizwiz_vqa.convert(vizwiz_source)}
    soup, blank = rows["VizWiz_train_00000000.jpg"], rows["VizWiz_train_00000001.jpg"]
    val = rows["VizWiz_val_00000000.jpg"]
    assert soup.targets["answerable"].human.noul == 0.6 and soup.targets["answer"].human.n_raters == 6
    assert set(blank.questions) == {"answerable"}             # one answer given: no answer Choice
    assert sorted(val.questions["answer"].criteria.values()) == ["no", "yes"]
    assert val.split in ("gold_calib", "gold_test") and soup.split in ("train", "val")
    assert (vizwiz_source / "images" / "VizWiz_val_00000000.jpg").read_bytes() == b"JPEG"
    for r in rows.values():
        check_licence(r, load_registry())
