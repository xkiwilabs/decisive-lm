"""scripts/build_dev_v1.py: one question per val row, never on a trained text or prompt, for the dev_v1 project."""
import importlib.util
import pathlib

from decisivelm.data.schema import Record

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_dev_v1", ROOT / "scripts" / "build_dev_v1.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)


def _row(sid, family="civil_comments-base", split="val", words=10):
    return Record.model_validate({
        "id": f"civil_comments-{sid}", "lineage": {"source_dataset": "civil_comments", "source_id": sid,
                                                   "family_id": family},
        "license": "CC0-1.0", "tier": "R", "modality": ["text"], "language": "en", "state": " ".join(["w"] * words),
        "questions": {"toxicity": {"type": "noul", "instructions": "The comment is toxic"},
                      "insult": {"type": "noul", "instructions": "The comment is insulting"}},
        "targets": {}, "preprocess": {"render_version": 1, "max_state_tokens": 1000, "truncated": False},
        "split": split})


def test_picks_short_untrained_val_rows_once_each_with_one_question():
    rows = [_row("a"), _row("b"), _row("c", words=500), _row("d", split="train"), _row("e")]
    ids = {("civil_comments", "b")}
    out = bd.pick(rows, "toxicity", 10, lambda r, q: True, ids, set(), {"civil_comments-e"}, seed=0)
    assert [r.lineage.source_id for r in out] == ["a"]
    r = out[0]
    assert r.id == "dev_v1-civil_comments-a" and list(r.questions) == ["toxicity"] and not r.targets
    assert r.split in {"gold_calib", "gold_test"}


def test_prompt_families_are_excluded_and_split_together():
    rows = [_row("a", family="helpsteer2-prompt-1"), _row("b", family="helpsteer2-prompt-2")]
    out = bd.pick(rows, "toxicity", 10, lambda r, q: True, set(), {"helpsteer2-prompt-1"}, set(), seed=0)
    assert [r.lineage.source_id for r in out] == ["b"]
    assert bd.group(out[0]) == "helpsteer2-prompt-2" and bd.group(_row("x")) == "civil_comments-x"


def test_short_limits_each_field():
    assert bd.short("a b c") and not bd.short(" ".join(["w"] * 151))
    assert bd.short({"conversation": "User: hi", "response": "ok"})
    assert not bd.short({"conversation": "User: hi\nAssistant: yo\nUser: more", "response": "ok"})
    assert not bd.short({"conversation": "User: hi", "response": " ".join(["w"] * 251)})
