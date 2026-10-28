"""scripts/build_two_sided.py: texts and two-sided Scores for the rating app's two_sided_v1 project."""
import importlib.util
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_two_sided", ROOT / "scripts" / "build_two_sided.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def _argument(i, url, title, words=60):
    return {"id": f"a{i}", "premises": [{"text": " ".join(["word"] * words), "stance": "PRO"}],
            "context": {"sourceId": f"s{i}", "discussionTitle": title, "sourceUrl": url}}


def test_every_scale_is_five_levels_with_a_neutral_middle():
    for q in [bt.for_against("Schools should start later"), *bt.ARG_QUESTIONS.values(), *bt.CHAT_QUESTIONS.values()]:
        assert q["type"] == "score" and len(q["criteria"]) == 5
        assert q["criteria"][2].startswith(("Neither", "Partly serious"))
    assert bt.for_against("X")["criteria"][0] == "Strongly against"


def test_arguments_take_an_equal_share_of_each_portal_one_per_debate_and_skip_short_ones(tmp_path):
    args = [_argument(0, "https://www.debate.org/debates/a/1/", "Cats are better than dogs"),
            _argument(1, "https://www.debate.org/debates/a/1/", "Cats are better than dogs"),
            _argument(2, "https://www.debate.org/debates/b/1/", "Short", words=10),
            _argument(3, "https://idebate.org/debatabase/c", "Ban homework")]
    args[1]["context"]["sourceId"] = "s0"   # the same debate as argument 0
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("args-me-1.0-cleaned.json", json.dumps({"arguments": args}))
    rows = bt.arguments(tmp_path / "a.zip", n=8, seed=0)
    assert len(rows) == 2 and {r.lineage.family_id for r in rows} == {"args_me-s0", "args_me-s3"}
    assert bt.portal(args[3]) == "idebate.org" and bt.portal(args[0]) == "www.debate.org"   # not a substring match
    r = next(r for r in rows if r.lineage.family_id == "args_me-s3")
    assert r.state.startswith("Debate: Ban homework\nArgument: word") and not r.targets
    assert r.questions["for_against"].instructions == 'Is the author for or against this: "Ban homework"?'
    assert r.split in {"gold_test", "gold_calib"}
