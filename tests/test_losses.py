"""Losses (docs/guide/06 §6.7)."""
import math

import pytest
import torch

from decisivelm.losses import choice_loss, noul_loss, request_losses, score_loss


def test_choice_loss_is_one_for_uniform_logits():
    assert choice_loss(torch.zeros(5), torch.tensor([0.2, 0.3, 0.5, 0.0, 0.0])).item() == pytest.approx(1.0)


def test_choice_loss_smooths_hard_labels():
    target = torch.tensor([0.0, 1.0, 0.0, 0.0])
    sharp = torch.tensor([-20.0, 20.0, -20.0, -20.0])
    assert choice_loss(sharp, target).item() < 0.01
    assert choice_loss(sharp, target, smoothing=0.05).item() > 0.1   # smoothing penalises full certainty


def test_score_loss_adds_an_ordinal_term():
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    near = torch.log(torch.tensor([0.5, 0.5, 1e-6, 1e-6]))
    far = torch.log(torch.tensor([0.5, 1e-6, 1e-6, 0.5]))
    assert choice_loss(near, target).item() == pytest.approx(choice_loss(far, target).item(), abs=1e-4)
    assert score_loss(near, target).item() < score_loss(far, target).item()


def test_noul_loss_is_bce():
    assert noul_loss(torch.tensor([0.0]), torch.tensor(1.0)).item() == pytest.approx(math.log(2))


def test_request_losses_average_per_question_then_per_request():
    # request 0 has questions 0 and 1; request 1 has question 2
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    leaf_question = torch.tensor([0, 0, 1, 2, 2])
    request_of_question = torch.tensor([0, 0, 1])
    questions = [("choice", torch.tensor([0.5, 0.5])), ("noul", torch.tensor(1.0)), ("choice", None)]
    per_request = request_losses(logits, leaf_question, request_of_question, questions)
    assert per_request.tolist() == pytest.approx([(1.0 + math.log(2)) / 2])   # request 1 has no target: masked


def test_request_losses_of_a_batch_with_no_targets_is_empty():
    # A packed micro-batch can hold only rows whose questions all lack targets (a mix filtered by question type);
    # training skips it instead of failing on an empty stack.
    logits = torch.tensor([0.0, 0.0, 0.0])
    per_request = request_losses(logits, torch.tensor([0, 0, 1]), torch.tensor([0, 1]), [("choice", None), ("noul", None)])
    assert per_request.numel() == 0
