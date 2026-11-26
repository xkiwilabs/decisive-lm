"""Adapter training: LoRA + query rows + scorer on packed request trees (docs/guide/09 §9.5; P2 and P3).

Requests are packed into micro-batches of at most `max_tokens` packed tokens;
gradients accumulate over `rows_per_step` requests. AdamW with separate
learning rates for LoRA and for the head + query rows, cosine decay after a
linear warmup. Every `eval_every` optimiser steps three validation criteria are
measured (`val_metrics`), the best adapter under each is kept, and training stops
early when the `select` criterion has not improved for `patience` checks (docs/guide/09).
"""
import math
import random
import time
from dataclasses import dataclass

import torch

from decisivelm.api_types import ChoiceQuestion, NoulQuestion
from decisivelm.data.schema import Record
from decisivelm.losses import LOSSES, request_losses
from decisivelm.lora import lora_state
from decisivelm.tree import TreeRequest


@dataclass
class Item:
    id: str
    tree: TreeRequest
    questions: list[tuple[str, torch.Tensor | None]]   # (type, target) in request question order
    n_tokens: int


def _target(q, t) -> torch.Tensor | None:
    """Human target where present, else teacher (docs/guide/08 §8.2). Ordered like the question's options/levels."""
    src = (t.human or (t.teacher[0] if isinstance(t.teacher, list) else t.teacher)) if t else None
    if src is None:
        return None
    if isinstance(q, NoulQuestion):
        return torch.tensor(src.noul)
    keys = list(q.criteria) if isinstance(q, ChoiceQuestion) else [str(i) for i in range(len(q.criteria))]
    return torch.tensor([src.probabilities[k] for k in keys])


def _hard(target, key: str):
    """One-hot on the most likely answer; a tie goes to one of the tied answers, picked at random seeded by `key`.
    A Noul target becomes 1 above 0.5, 0 below, and a seeded coin flip at 0.5."""
    rng = random.Random(key)
    if target.noul is not None:
        v = target.noul
        return target.model_copy(update={"noul": float(rng.random() < 0.5) if abs(v - 0.5) < 1e-9 else float(v > 0.5)})
    p = target.probabilities
    top = max(p.values())
    pick = rng.choice([k for k, v in p.items() if top - v < 1e-9])
    return target.model_copy(update={"probabilities": {k: float(k == pick) for k in p}})


def harden(record: Record) -> Record:
    """The record with every human and teacher target replaced by its majority answer (`_hard`): the
    majority-label ablation of training on rater shares (train_p3.py --hard-targets)."""
    targets = {}
    for qid, t in record.targets.items():
        key = f"{record.id}/{qid}"
        teacher = t.teacher
        if isinstance(teacher, list):
            teacher = [_hard(x, f"{key}/teacher{i}") for i, x in enumerate(teacher)]
        elif teacher is not None:
            teacher = _hard(teacher, f"{key}/teacher")
        targets[qid] = t.model_copy(update={"human": t.human and _hard(t.human, f"{key}/human"), "teacher": teacher})
    return record.model_copy(update={"targets": targets})


def make_items(dlm, records: list[Record]) -> list[Item]:
    items = []
    for r in records:
        tree = dlm.tree_for({"state": r.state, "media": r.media, "questions": r.questions})
        for im in tree.images:
            im.drop_pixels()    # reloaded per micro-batch: every row's pixels at once would not fit in memory
        for clip in tree.audios:
            clip.drop_features()
        n = len(tree.prefix) + sum(len(q.segment) + sum(map(len, q.leaves)) for q in tree.questions)
        items.append(Item(r.id, tree, [(q.type, _target(q, r.targets.get(qid))) for qid, q in r.questions.items()], n))
    return items


def micro_batches(items: list[Item], max_tokens: int, rng: random.Random) -> list[list[Item]]:
    """Shuffle, then fill batches greedily up to max_tokens. An item over budget goes alone."""
    order = items[:]
    rng.shuffle(order)
    batches, cur, used = [], [], 0
    for it in order:
        if cur and used + it.n_tokens > max_tokens:
            batches.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += it.n_tokens
    return batches + ([cur] if cur else [])


def batch_losses(dlm, batch: list[Item], grad: bool) -> torch.Tensor:
    tree, q = dlm.query_states([it.tree for it in batch], grad=grad)
    with torch.set_grad_enabled(grad):
        logits = dlm.scorer(q.to(next(dlm.scorer.parameters()).dtype)).float()
        questions = [qt for it in batch for qt in it.questions]
        return request_losses(logits, tree.leaf_question.to(logits.device),
                              tree.request_of_question, questions)


@torch.no_grad()
def evaluate(dlm, items: list[Item], max_tokens: int) -> float:
    dlm.backbone.eval(); dlm.scorer.eval()
    losses = [batch_losses(dlm, b, grad=False) for b in micro_batches(items, max_tokens, random.Random(0))]
    return float(torch.cat(losses).mean())


TEMPERATURES = torch.exp(torch.linspace(math.log(0.25), math.log(4.0), 41)).tolist()
HIGHER_IS_BETTER = {"loss": False, "cal_loss": False, "acc": True}


@torch.no_grad()
def val_metrics(dlm, items: list[Item], max_tokens: int) -> dict[str, float]:
    """Validation criteria for model selection, all averaged per question, then per request:
    - loss: the training loss;
    - cal_loss: the same after one temperature per question type is fitted on these rows, so a model that is merely
      over- or under-confident is not penalised (calibration on gold_calib corrects that later anyway);
    - acc: argmax matches the target's unique argmax (Choice, Score; tied targets skipped), or P(yes) ≥ .5 matches
      target ≥ .5 (Noul)."""
    dlm.backbone.eval(); dlm.scorer.eval()
    qs = []     # (request index, type, logits, target)
    base = 0
    for b in micro_batches(items, max_tokens, random.Random(0)):
        tree, q = dlm.query_states([it.tree for it in b], grad=False)
        logits = dlm.scorer(q.to(next(dlm.scorer.parameters()).dtype)).float().cpu()
        lq = tree.leaf_question.cpu()
        for qi, (qtype, target) in enumerate(qt for it in b for qt in it.questions):
            if target is not None:
                qs.append((base + int(tree.request_of_question[qi]), qtype, logits[lq == qi], target.float()))
        base += len(b)

    def per_request(values):
        by = {}
        for (r, *_), v in zip(qs, values):
            by.setdefault(r, []).append(v)
        return sum(sum(v) / len(v) for v in by.values()) / len(by)

    raw = [float(LOSSES[t](x, y)) for _, t, x, y in qs]
    cal = list(raw)
    for qtype in {t for _, t, _, _ in qs}:
        idx = [i for i, (_, t, _, _) in enumerate(qs) if t == qtype]
        losses = {T: [float(LOSSES[qtype](qs[i][2] / T, qs[i][3])) for i in idx] for T in TEMPERATURES}
        best = min(losses.values(), key=sum)
        for i, v in zip(idx, best):
            cal[i] = v
    right = []
    for _, t, x, y in qs:
        if t == "noul":
            right.append(float((float(x.reshape(())) >= 0) == (float(y) >= 0.5)))
        else:
            top = torch.topk(y, 2).values
            right.append(float(x.argmax() == y.argmax()) if top[0] > top[1] + 1e-9 else None)
    scored = [(q, v) for q, v in zip(qs, right) if v is not None]
    acc = sum(v for _, v in scored) / len(scored) if scored else float("nan")
    return {"loss": per_request(raw), "cal_loss": per_request(cal), "acc": acc}


def train_adapter(dlm, train: list[Item], val: list[Item], *, epochs: int, rows_per_step: int, max_tokens: int,
                  lr_lora: float = 2e-4, lr_head: float = 1e-3, warmup_frac: float = 0.03, weight_decay: float = 0.01,
                  eval_every: int = 50, patience: int = 3, seed: int = 0, checkpointing: bool = True,
                  select: str = "loss", log=print) -> dict:
    """Returns the adapter and scorer that were best under `select`, the history, and `best`: for every criterion
    of `val_metrics`, its best value, the step it was reached at and that snapshot."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    backbone, scorer = dlm.backbone, dlm.scorer
    if checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = [p for n, p in backbone.named_parameters() if p.requires_grad and not n.endswith("query_rows.delta")]
    head = [p for n, p in backbone.named_parameters() if n.endswith("query_rows.delta")] + list(scorer.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": lr_lora}, {"params": head, "lr": lr_head}],
                            weight_decay=weight_decay)
    total = epochs * math.ceil(len(train) / rows_per_step)
    warmup = max(1, round(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    def snapshot():
        return {"adapter": lora_state(backbone), "scorer": {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}}

    def improved(m, new, old):
        return new > old + 1e-4 if HIGHER_IS_BETTER[m] else new < old - 1e-4

    def result():
        b = best[select]
        return {"adapter": b["state"]["adapter"], "scorer": b["state"]["scorer"], "history": history,
                "best_val": b["value"], "best_step": b["step"], "best": best}

    def fmt(m):
        return "  ".join(f"{k} {v:.4f}" for k, v in m.items())

    metrics = val_metrics(dlm, val, max_tokens)
    history = [{"step": 0, "train_loss": None, "val_loss": metrics["loss"], **{f"val_{k}": v for k, v in metrics.items()}}]
    log(f"step 0  val {fmt(metrics)}")
    state = snapshot()
    best = {m: {"value": v, "step": 0, "state": state} for m, v in metrics.items()}
    bad, step, t0 = 0, 0, time.perf_counter()
    window = []   # train losses since the last validation
    for epoch in range(epochs):
        order = train[:]
        rng.shuffle(order)
        for start in range(0, len(order), rows_per_step):
            backbone.train(); scorer.train()
            group = order[start:start + rows_per_step]
            for b in micro_batches(group, max_tokens, rng):
                losses = batch_losses(dlm, b, grad=True)
                if not losses.numel():
                    continue
                (losses.sum() / len(group)).backward()
                window.append(losses.detach().cpu())
            torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or step == total:
                metrics = val_metrics(dlm, val, max_tokens)
                train_loss = float(torch.cat(window).mean())
                window = []
                history.append({"step": step, "epoch": epoch, "train_loss": train_loss, "val_loss": metrics["loss"],
                                **{f"val_{k}": v for k, v in metrics.items()}})
                log(f"step {step}/{total}  train {train_loss:.4f}  val {fmt(metrics)}  "
                    f"{(time.perf_counter() - t0) / (step * rows_per_step):.2f} s/row")
                state = None
                for m, v in metrics.items():
                    if improved(m, v, best[m]["value"]):
                        state = state or snapshot()
                        best[m] = {"value": v, "step": step, "state": state}
                if best[select]["step"] == step:
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        log(f"early stop at step {step}; best val {select} {best[select]['value']:.4f} "
                            f"at step {best[select]['step']}")
                        return result()
    return result()
