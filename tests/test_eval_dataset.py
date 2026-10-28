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
