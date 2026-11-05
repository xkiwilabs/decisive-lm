"""scripts/eval_dataset.py: plurality labels, pair agreement from shares, per-type metrics."""
import importlib.util, pathlib

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "eval_dataset", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "eval_dataset.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_unique_argmax_marks_ties():
    t = np.array([[.5, .25, .25], [.5, .5, 0], [0, 1, 0]])
    assert ev.unique_argmax(t).tolist() == [0, -1, 1]


def test_pair_agreement_from_shares_counts_agreeing_pairs():
    # 3 of 4 raters agree: 3 agreeing pairs of 6; unanimous: 1; 2-2 split: 2 of 6
    t = np.array([[.75, .25], [1, 0], [.5, .5]])
    assert ev.pair_agreement_from_shares(t, np.array([4, 4, 4])) == pytest.approx([.5, 1, 1 / 3])


def _question(qtype, t, n_raters=None):
    q = ev.Question.__new__(ev.Question)
    q.qtype, q.other, q.t = qtype, None, np.array(t)
    q.n_raters = None if n_raters is None else np.array(n_raters)
    if qtype != "noul":
        q.gold = ev.unique_argmax(q.t)
    return q


def test_choice_metrics_leave_tied_items_out_of_accuracy():
    q = _question("choice", [[1, 0], [0, 1], [.5, .5]])
    m = q.metrics(np.array([[.9, .1], [.8, .2], [.1, .9]]), np.arange(3))
    assert m["accuracy"] == pytest.approx(.5)


def test_score_mae_is_expected_level_against_rater_mean():
    q = _question("score", [[0, 0, 1], [1, 0, 0]])
    m = q.metrics(np.array([[0, .5, .5], [1, 0, 0]]), np.arange(2))
    assert m["MAE"] == pytest.approx(.25) and m["exact"] == pytest.approx(.5)


def test_noul_labels_are_share_at_least_half():
    q = _question("noul", [.6, .5, .2, 0])
    m = q.metrics(np.array([.7, .6, .4, .1]), np.arange(4))
    assert m["AUROC"] == pytest.approx(1) and m["acc@.5"] == pytest.approx(1)


def test_a_question_on_some_rows_and_differing_option_sets_are_scored_as_groups(tmp_path, monkeypatch, capsys):
    from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
    yes_no = {"type": "choice", "instructions": "Answer?", "criteria": {"a": "yes", "b": "no"}}
    four = {"type": "choice", "instructions": "Answer?", "criteria": {"a": "red", "b": "blue", "c": "x", "d": "y"}}
    noul = {"type": "noul", "instructions": "Answerable"}
    rows, preds = [], []
    for i in range(12):
        qs, ts, ans = {"ok": noul}, {"ok": QuestionTargets(human=HumanTarget(noul=i % 2, n_raters=10))}, {}
        ans["ok"] = {"p": 0.8 if i % 2 else 0.2}
        if i % 3:                                 # 8 rows ask the answer: 4 with two options, 4 with four
            q = yes_no if i % 3 == 1 else four
            ts["answer"] = QuestionTargets(human=HumanTarget(probabilities={k: float(k == "a") for k in q["criteria"]},
                                                             n_raters=10))
            qs["answer"] = q
            ans["answer"] = {"probabilities": {k: 0.7 if k == "a" else 0.3 / (len(q["criteria"]) - 1)
                                               for k in q["criteria"]}}
        rows.append(Record(id=f"vizwiz_vqa-{i}", lineage=Lineage(source_dataset="vizwiz_vqa", source_id=str(i),
                                                                  family_id=str(i)),
                           license="CC-BY-4.0", tier="R", modality=["text"], language="en", state="Question: what?",
                           questions=qs, targets=ts, preprocess=Preprocess(render_version=1, max_state_tokens=8192,
                                                                           truncated=False), split="gold_test"))
        preds.append({"id": f"vizwiz_vqa-{i}", "answers": ans})
    data, pred = tmp_path / "vizwiz_vqa.jsonl", tmp_path / "p.jsonl"
    data.write_text("".join(r.model_dump_json(exclude_none=True) + "\n" for r in rows))
    pred.write_text("".join(__import__("json").dumps(p) + "\n" for p in preds))
    monkeypatch.setattr("sys.argv", ["eval_dataset.py", "--data", str(data), "--predictions", f"m={pred}",
                                     "--n-boot", "20"])
    ev.main()
    out = capsys.readouterr().out
    assert "## ok (noul)" in out
    assert "## answer (choice, 2 options; asked on 4 rows)" in out
    assert "## answer (choice, 4 options; asked on 4 rows)" in out
