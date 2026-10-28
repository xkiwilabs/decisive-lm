"""Training losses (docs/guide/06 §6.7).

Choice and Score are divided by log(n) so a 255-way question doesn't swamp a
Noul. Score adds EMD² between predicted and target CDFs, since levels never see
each other and order is learned only through the loss. Questions without a
target are masked; the loss is averaged per question, then per request.
"""
import math

import torch
import torch.nn.functional as F

EMD_WEIGHT = 0.5


def choice_loss(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    n = logits.shape[-1]
    if smoothing:
        target = (1 - smoothing) * target + smoothing / n
    return -(target * torch.log_softmax(logits, -1)).sum(-1) / math.log(n)


def score_loss(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    cdf_p = torch.cumsum(torch.softmax(logits, -1), -1)
    cdf_t = torch.cumsum(target, -1)
    return choice_loss(logits, target, smoothing) + EMD_WEIGHT * ((cdf_p - cdf_t) ** 2).mean(-1)


def noul_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit.reshape(()), target.to(logit.dtype).reshape(()))


LOSSES = {"choice": choice_loss, "score": score_loss, "noul": noul_loss}


def request_losses(logits, leaf_question, request_of_question, questions) -> torch.Tensor:
    """Loss per request that has at least one target. `questions[i]` is (type, target tensor or None)."""
    by_request = {}
    for qi, (qtype, target) in enumerate(questions):
        if target is None:
            continue
        loss = LOSSES[qtype](logits[leaf_question == qi], target.to(logits.device))
        by_request.setdefault(int(request_of_question[qi]), []).append(loss)
    return torch.stack([torch.stack(v).mean() for _, v in sorted(by_request.items())])
