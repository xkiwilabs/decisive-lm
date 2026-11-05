"""Per-question calibration and predictions for every question type (docs/guide/10)."""
import math

import pytest
import torch

from decisivelm.calibrate import (QuestionLogits, fit_calibration, fit_new_task_scale, fit_task_calibration, predictions,
                                  question_logits, scaled)
from decisivelm.head import Calibration
from test_train import ROWS


def test_question_logits_cut_a_row_in_question_order():
    r = ROWS[0]                                   # questions: u (noul), d (choice, 3 options)
    qs = question_logits(r, torch.tensor([0.5, 1.0, 2.0, 3.0]))
    assert [(q.qid, q.qtype, q.names) for q in qs] == [("u", "noul", ["yes"]),
                                                        ("d", "choice", ["billing", "technical", "sales"])]
    assert qs[0].logits.tolist() == [0.5] and qs[1].logits.tolist() == [1.0, 2.0, 3.0]
    assert float(qs[0].target) == pytest.approx(0.2) and qs[1].target.tolist() == [1.0, 0.0, 0.0]


def test_question_logits_refuse_a_wrong_leaf_count():
    with pytest.raises(ValueError):
        question_logits(ROWS[0], torch.zeros(3))


def _choice(i, logits, target):
    return QuestionLogits(f"r{i}", "gold_calib", "d", "choice", ["a", "b", "c"], torch.tensor(logits),
                          torch.tensor(target))


def test_choice_temperature_undoes_a_sharpening_factor():
    torch.manual_seed(0)
    qs = []
    for i in range(400):
        true = torch.randn(3)
        target = torch.distributions.Categorical(logits=true).sample()
        qs.append(_choice(i, (3 * true).tolist(), torch.nn.functional.one_hot(target, 3).float().tolist()))
    assert fit_calibration(qs).temperatures["choice:3-5"] == pytest.approx(3, rel=0.25)


def test_all_easy_calibration_rows_stop_at_the_floor_and_warn():
    qs = [_choice(i, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]) for i in range(50)]   # always right: NLL falls as T -> 0
    with pytest.warns(UserWarning, match="grid edge"):
        t = fit_calibration(qs).temperatures["choice:3-5"]
    assert t == pytest.approx(0.25)


def test_noul_temperature_and_bias_recover_a_shifted_scaled_logit():
    torch.manual_seed(0)
    true = torch.randn(2000) * 2
    y = torch.bernoulli(torch.sigmoid(true))
    qs = [QuestionLogits(f"r{i}", "gold_calib", "u", "noul", ["yes"], torch.tensor([2 * t + 1.0]), y[i])
          for i, t in enumerate(true.tolist())]
    calib = fit_calibration(qs)
    assert calib.temperatures["noul"] == pytest.approx(2, rel=0.15)
    assert calib.b_noul == pytest.approx(-1, abs=0.3)


def test_predictions_carry_every_question_and_the_single_question_shortcut():
    qs = question_logits(ROWS[0], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    (row,) = predictions(qs, fit_calibration([]))
    assert row["answers"]["u"] == {"type": "noul", "p": pytest.approx(0.5)}
    assert sum(row["answers"]["d"]["probabilities"].values()) == pytest.approx(1)
    assert "probabilities" not in row                   # two questions: no top-level shortcut
    (single,) = predictions([_choice(0, [0.0, 0.0, math.log(2)], [0.0, 0.0, 1.0])], fit_calibration([]))
    assert single["probabilities"] == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})


def test_packed_logits_cut_per_request_match_each_request_alone():
    """train_p3's prediction path: leaves of request k in a packed batch are its questions' leaves, in order."""
    from test_train import dlm
    from decisivelm.train import make_items
    m = dlm()
    m.backbone.float().eval()
    items = make_items(m, ROWS)
    tree, logits = m.logits([it.tree for it in items])
    leaf_request = tree.request_of_question[tree.leaf_question]
    for k, (it, r) in enumerate(zip(items, ROWS)):
        packed = question_logits(r, logits[leaf_request == k])
        alone = question_logits(r, m.logits([it.tree])[1])
        for p, s in zip(packed, alone):
            assert p.qid == s.qid and torch.allclose(p.logits, s.logits, atol=1e-5)


def test_a_small_bucket_and_an_unseen_one_take_the_types_pooled_temperature():
    torch.manual_seed(0)
    big, small = [], []
    for i in range(400):
        true = torch.randn(3)
        target = torch.nn.functional.one_hot(torch.distributions.Categorical(logits=true).sample(), 3).float()
        big.append(_choice(i, (3 * true).tolist(), target.tolist()))
    for i in range(20):   # 2 options, perfectly separable: alone it would fit the grid floor
        small.append(QuestionLogits(f"s{i}", "gold_calib", "d", "choice", ["a", "b"], torch.tensor([2.0, 0.0]),
                                    torch.tensor([1.0, 0.0])))
    calib = fit_calibration(big + small)
    assert calib.temperatures["choice:2"] == calib.temperatures["choice:*"] > 0.25
    assert calib.temperatures["choice:3-5"] == pytest.approx(3, rel=0.25)       # big enough to keep its own
    assert calib.temperature("choice", 12) == calib.temperatures["choice:*"]    # no 6-15 questions at all
    assert fit_calibration(big + small, min_bucket=0).temperatures["choice:2"] == pytest.approx(0.25)


def test_new_task_scale_sharpens_an_underconfident_head_against_the_majority_answer():
    # The true answer's logit is 2 above the others, but the stored temperature (2) halves it: the scale should be
    # about 1/2 relative to the stored temperature. Rater shares are split 60/40 and Nouls are left alone.
    g = torch.Generator().manual_seed(0)
    qs = []
    for i in range(400):
        logits = torch.randn(3, generator=g) * 0.5
        right = int(torch.randint(3, (1,), generator=g))
        logits[right] += 2.0 * 2.0
        p = torch.softmax(logits / 2.0, -1)
        answer = int(torch.multinomial(torch.softmax(logits / 2.0 * 2, -1), 1, generator=g))   # calibrated at T 1
        target = torch.full((3,), 0.0)
        target[answer], target[(answer + 1) % 3] = 0.6, 0.4
        qs.append(QuestionLogits(f"r{i}", "gold_calib", "q", "choice", ["a", "b", "c"], logits, target))
    calib = Calibration(temperatures={"choice:3-5": 2.0, "noul": 1.3}, b_noul=0.1)
    s = fit_new_task_scale(qs, calib, "choice")
    assert 0.4 < s < 0.6
    with pytest.raises(ValueError, match="no untied score"):
        fit_new_task_scale(qs, calib, "score")
    new = scaled(calib, {"choice": s, "score": 0.9})
    assert new.temperatures["choice:3-5"] == pytest.approx(2.0 * s) and new.temperatures["noul"] == 1.3
    assert new.b_noul == 0.1


def _score(i, logits, target, key="score:k"):
    return QuestionLogits(f"r{i}", "gold_calib", "s", "score", [str(k) for k in range(len(logits))],
                          torch.tensor(logits), torch.tensor(target), key)


def test_task_offsets_recover_how_raters_use_the_scale():
    """Raters avoid the ends of the scale (true offsets); the head's logits do not know it."""
    torch.manual_seed(0)
    true = torch.tensor([-2.0, 0.0, 1.5, 1.0, -2.0])
    qs = [_score(i, (l := torch.randn(5)).tolist(), torch.softmax(l + true, -1).tolist()) for i in range(2000)]
    task = fit_task_calibration(qs, Calibration())["score:k"]
    got = torch.tensor([task["offsets"][str(k)] for k in range(5)])
    assert torch.allclose(got, true - true.mean(), atol=0.1) and task["scale"] == pytest.approx(1, abs=0.05)
    assert fit_task_calibration(qs[:50], Calibration()) == {}   # too few rows: none


def test_task_calibration_follows_the_question_not_its_option_order_or_id():
    from decisivelm.api_types import ChoiceQuestion
    from decisivelm.head import readout, task_key
    q = ChoiceQuestion.model_validate({"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B", "c": "C"}})
    shuffled = ChoiceQuestion.model_validate({"type": "choice", "instructions": "Which?", "criteria": {"c": "C", "a": "A", "b": "B"}})
    other = ChoiceQuestion.model_validate({"type": "choice", "instructions": "Which one?", "criteria": {"a": "A", "b": "B", "c": "C"}})
    assert task_key(q) == task_key(shuffled) != task_key(other)
    calib = Calibration(tasks={task_key(q): {"scale": 2.0, "offsets": {"a": 1.0, "b": 0.0, "c": -1.0}}})
    logits = torch.tensor([0.5, 1.0, -0.2])
    p = dict(zip("abc", readout("choice", logits, calib, task_key(q), list("abc")).tolist()))
    ps = dict(zip("cab", readout("choice", logits[[2, 0, 1]], calib, task_key(shuffled), list("cab")).tolist()))
    assert p == pytest.approx(ps)
    assert readout("choice", logits, calib, task_key(other), list("abc")).tolist() == \
        pytest.approx(torch.softmax(logits, -1).tolist())


@pytest.mark.parametrize("qtype", ["score", "noul"])
def test_invert_undoes_a_task_calibrated_readout(qtype):
    from decisivelm.calibrate import invert
    from decisivelm.head import readout
    names = ["yes"] if qtype == "noul" else ["0", "1", "2"]
    calib = Calibration(temperatures={"score:3-5": 1.5, "noul": 0.8}, b_noul=0.3,
                        tasks={"k": {"scale": 0.7, "offsets": dict(zip(names, [0.4, -0.1, 0.2]))}})
    logits = torch.tensor([0.9] if qtype == "noul" else [0.2, -0.6, 1.1])
    p = readout(qtype, logits, calib, "k", names)
    answer = {"type": "noul", "p": float(p[0])} if qtype == "noul" else {"type": qtype, "probabilities": dict(zip(names, p.tolist()))}
    _, back = invert(answer, calib, "k")
    if qtype == "noul":
        assert back.tolist() == pytest.approx(logits.tolist(), abs=1e-4)
    else:
        assert (back - logits).tolist() == pytest.approx([float(back[0] - logits[0])] * 3, abs=1e-4)


def test_too_few_calibration_questions_are_warned_about():
    qs = [_choice(i, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) for i in range(30)]
    with pytest.warns(UserWarning, match="only 30 choice questions") as caught:
        fit_calibration(qs)
    assert any("pooled temperature: choice:3-5 (30)" in str(w.message) for w in caught)
