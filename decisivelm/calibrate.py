"""Calibration and predictions for every question type (docs/guide/10).

A row's leaf logits come out of the packed tree in question order, each question's leaves in option or level order
(`render.leaf_texts`). `question_logits` cuts them back into questions; `fit_calibration` fits one temperature per
(type, option-count bucket) for Choice and Score and a temperature plus the shared bias `b_noul` for Noul, on
gold_calib; `predictions` applies them through `head.readout`, the same path serving uses.

Prediction rows: {"id", "split", "answers": {qid: {"type", "probabilities": {option or level: p}}}}, Noul answers as
{"type": "noul", "p": P(yes)}. A row with a single Choice or Score question also carries that question's
"probabilities" at the top level, the format compare_coding.py reads.
"""
import json
import math
import pathlib
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from decisivelm.data.schema import Record
from decisivelm.head import Calibration, bucket, readout, task_key
from decisivelm.train import _target


@dataclass
class QuestionLogits:
    id: str
    split: str
    qid: str
    qtype: str
    names: list[str]                  # option names, level indices as strings, or ["yes"] for Noul
    logits: torch.Tensor              # [n] (Noul: [1])
    target: torch.Tensor | None       # [n] shares, or a scalar P(yes) for Noul
    key: str | None = None            # head.task_key of the question, for task calibration


def question_names(q) -> list[str]:
    if q.type == "choice":
        return list(q.criteria)
    if q.type == "score":
        return [str(i) for i in range(len(q.criteria))]
    return ["yes"]


def question_logits(record: Record, leaf_logits: torch.Tensor) -> list[QuestionLogits]:
    """Cut one row's leaf logits (tree order) into its questions."""
    out, start = [], 0
    for qid, q in record.questions.items():
        names = question_names(q)
        out.append(QuestionLogits(record.id, record.split, qid, q.type, names,
                                  leaf_logits[start:start + len(names)].detach().float().cpu(),
                                  _target(q, record.targets.get(qid)), task_key(q)))
        start += len(names)
    if start != len(leaf_logits):
        raise ValueError(f"{record.id}: {len(leaf_logits)} leaf logits for {start} leaves")
    return out


def key(q: QuestionLogits) -> str:
    return "noul" if q.qtype == "noul" else f"{q.qtype}:{bucket(len(q.names))}"


def _soft_nll(group: list[QuestionLogits], t: float) -> float:
    return float(np.mean([-(q.target * torch.log_softmax(q.logits / t, -1)).sum() / math.log(len(q.names))
                          for q in group]))


MIN_BUCKET = 100   # calibration questions a Choice/Score bucket needs for a temperature of its own


def fit_calibration(qs: list[QuestionLogits], lo: float = 0.25, hi: float = 20.0, noul_bias: bool = True,
                    min_bucket: int = MIN_BUCKET) -> Calibration:
    """Temperatures per (type, bucket) by grid search on soft NLL; Noul temperature and bias by LBFGS on BCE.

    The grid starts at `lo`: a bucket whose calibration questions are all easy keeps lowering its NLL as T → 0, and
    an unbounded fit then makes every answer in that bucket near-certain on harder rows (General v0's 2- and 3–5-
    option temperatures hit the old floor, 0.05). A temperature at either end of the grid is reported.
    A Choice or Score bucket with fewer than `min_bucket` questions (General v3's 2-option bucket: 18, at the grid floor)
    takes the temperature fitted on all of that type's questions, as does a bucket with none. Pooling more than that
    hurt: with min_bucket 300 the 6–15-option bucket (135 exploded-intent questions) took the pooled 1.07, and General
    v2 ep2's ECE rose on the coding task (0.088 → 0.29), emotion and tweet_topic, all 6–15-option tasks its own 0.62
    had fitted well. `min_bucket=0` fits every bucket on its own.
    `noul_bias=False` keeps b_noul at 0, for calibration pooled across datasets whose base rates differ."""
    groups: dict[str, list[QuestionLogits]] = {}
    for q in qs:
        if q.target is not None:
            groups.setdefault(key(q), []).append(q)
    calib = Calibration()
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 400))
    def grid_fit(k, group):
        t = float(min(grid, key=lambda t: _soft_nll(group, t)))
        if t in (grid[0], grid[-1]):
            warnings.warn(f"calibration: {k} temperature at the grid edge ({t:.3g}) on {len(group)} questions; "
                          f"the calibration rows may be too easy or too few")
        return t

    for qtype in ("choice", "score"):   # the type's pooled temperature, for small buckets and unseen ones
        group = [q for k, g in groups.items() if k.startswith(qtype + ":") for q in g]
        if group:
            calib.temperatures[f"{qtype}:*"] = grid_fit(f"{qtype}:*", group)
    for qtype in ("choice", "score", "noul"):
        n = sum(len(g) for k, g in groups.items() if k.split(":")[0] == qtype)
        if 0 < n < MIN_BUCKET:
            warnings.warn(f"calibration: only {n} {qtype} questions to fit on; with fewer than {MIN_BUCKET} the "
                          f"temperature is noisy and the reported calibration error uncertain (docs/guide/10)")
    if pooled := {k: len(g) for k, g in groups.items() if k != "noul" and len(g) < min_bucket}:
        listed = ", ".join(f"{k} ({n})" for k, n in sorted(pooled.items()))
        warnings.warn(f"calibration: buckets with fewer than {min_bucket} questions take their type's pooled "
                      f"temperature: {listed}")
    for k, group in groups.items():
        if k != "noul" and len(group) < min_bucket:
            calib.temperatures[k] = calib.temperatures[f"{k.split(':')[0]}:*"]
            continue
        if k == "noul":
            x = torch.stack([q.logits.reshape(()) for q in group])
            y = torch.stack([q.target.reshape(()).float() for q in group])
            log_t, b = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=noul_bias)
            opt = torch.optim.LBFGS([log_t, b] if noul_bias else [log_t], max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits((x + b) / log_t.exp(), y)
                loss.backward()
                return loss
            opt.step(closure)
            calib.temperatures["noul"], calib.b_noul = float(log_t.detach().exp()), float(b.detach())
        else:
            calib.temperatures[k] = grid_fit(k, group)
    return calib


def fit_new_task_scale(qs: list[QuestionLogits], calib: Calibration, qtype: str, lo: float = 0.2,
                       hi: float = 2.0) -> float:
    """One factor for every `qtype` ("choice" or "score") temperature, fitted by NLL of the human majority answer (ties
    left out) on calibration rows of a task the head was not trained on (docs/RELEASE.md; guide chapter 10.6).

    Temperatures fitted by soft NLL on the mix's gold_calib make probabilities match rater shares on tasks the head has
    seen. Judged against the majority answer, as the release criteria are, a head can then be underconfident on new
    tasks: a 60/40 split is a correct majority answer at p 0.6, and the head may hedge more on tasks it has not seen.
    The size differs by type, so each type gets its own factor: General v3c needed 0.59 for Choice and 0.95 for Score
    on MultiNLI, which asks the same judgement both ways."""
    group = [q for q in qs if q.qtype == qtype and q.target is not None
             and int((q.target == q.target.max()).sum()) == 1]
    if not group:
        raise ValueError(f"no untied {qtype} calibration questions")
    gold = [int(q.target.argmax()) for q in group]
    temps = [calib.temperature(q.qtype, len(q.names)) for q in group]

    def nll(s):
        return float(np.mean([-torch.log_softmax(q.logits / (t * s), -1)[g] for q, g, t in zip(group, gold, temps)]))
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 200))
    s = float(min(grid, key=nll))
    if s in (grid[0], grid[-1]):
        warnings.warn(f"new-task {qtype} scale at the grid edge ({s:.3g}) on {len(group)} questions")
    return s


def scaled(calib: Calibration, scales: dict[str, float]) -> Calibration:
    """`calib` with every temperature of each type in `scales` ({"choice": s, "score": s}) multiplied by its factor."""
    return Calibration(temperatures={k: t * scales.get(k.split(":")[0], 1.0) for k, t in calib.temperatures.items()},
                       b_noul=calib.b_noul, tasks=calib.tasks)


def new_task_scales(qs: list[QuestionLogits], calib: Calibration) -> dict[str, float]:
    """A factor for Choice and for Score; Noul calibration is left as it is (General v3c's Noul ECE was 0.04 on
    MultiNLI unscaled)."""
    return {t: fit_new_task_scale(qs, calib, t) for t in ("choice", "score")}


MIN_TASK = 100     # calibration questions a question needs for a task calibration of its own
TASK_PRIOR = 1.0   # a weak Gaussian prior on the offsets: keeps a level no rater used finite


def fit_task_calibration(qs: list[QuestionLogits], calib: Calibration, min_questions: int = MIN_TASK) -> dict[str, dict]:
    """Task calibration (guide chapter 10): for each question asked on at least `min_questions` labelled calibration
    rows (grouped by `key`), a scale on its temperature and an offset per option or level (Noul: on the logit),
    fitted by soft NLL against the rater shares on top of `calib`.

    Temperatures fix how spread out the answers are; they cannot fix how raters use a scale. On Wikipedia Talk
    toxicity raters gave 2.6% of their votes to "very toxic" and 1.6% to "very healthy"; P0 and the general heads put
    9–21% there, and were most confident on exactly those items, where raters split. Their confidence then ran
    against rater agreement (General v4a ρ −0.09); with offsets fitted on 2,040 calibration rows it followed it
    (ρ +0.30, MAE 0.38 → 0.23). The offsets carry the raters' base rates for this question, so they belong to the
    task, never to the general head."""
    groups: dict[str, list[QuestionLogits]] = {}
    for q in qs:
        if q.key is not None and q.target is not None:
            groups.setdefault(q.key, []).append(q)
    out = {}
    for k, group in groups.items():
        if len(group) < min_questions:
            continue
        names = group[0].names if group[0].qtype != "choice" else sorted(group[0].names)
        noul = group[0].qtype == "noul"
        idx = [[q.names.index(n) for n in names] for q in group]   # every question in `names` order
        x = torch.stack([q.logits[i] / calib.temperature(q.qtype, len(q.names)) for q, i in zip(group, idx)])
        y = torch.stack([(q.target.reshape(1) if noul else q.target[i]).float() for q, i in zip(group, idx)])
        if noul:
            x = torch.stack([(q.logits + calib.b_noul) / calib.temperature("noul", 1) for q in group])
        log_s, b = torch.zeros(1, requires_grad=True), torch.zeros(len(names), requires_grad=True)
        opt = torch.optim.LBFGS([log_s, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = x / log_s.exp() + b
            nll = (F.binary_cross_entropy_with_logits(z, y) if noul
                   else -(y * torch.log_softmax(z, -1)).sum(-1).mean())
            loss = nll + TASK_PRIOR / (2 * len(group)) * (b ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        off = b.detach() - (0 if noul else b.detach().mean())   # softmax ignores a shared shift
        out[k] = {"scale": float(log_s.detach().exp()), "offsets": dict(zip(names, map(float, off))),
                  "questions": len(group)}
    return out


def predictions(qs: list[QuestionLogits], calib: Calibration) -> list[dict]:
    rows: dict[str, dict] = {}
    for q in qs:
        row = rows.setdefault(q.id, {"id": q.id, "split": q.split, "answers": {}})
        p = readout(q.qtype, q.logits, calib, q.key, q.names).tolist()
        row["answers"][q.qid] = ({"type": "noul", "p": p[0]} if q.qtype == "noul"
                                 else {"type": q.qtype, "probabilities": dict(zip(q.names, p))})
    for row in rows.values():
        if len(row["answers"]) == 1 and (a := next(iter(row["answers"].values())))["type"] != "noul":
            row["probabilities"] = a["probabilities"]
    return list(rows.values())


def invert(answer: dict, calib: Calibration, key: str | None = None) -> tuple[list[str], torch.Tensor]:
    """A prediction's raw logits back from its readout through `calib`, up to a per-question constant (which softmax
    ignores): T · log p for Choice and Score, T · logit(p) − b for Noul; with the question's task calibration (`key`),
    T·s · (log p − offset) and T·s · (logit(p) − offset) − b. Names as in `question_names`."""
    task = calib.tasks.get(key) if key is not None else None
    s, off = (task["scale"], task["offsets"]) if task else (1.0, {})
    if answer["type"] == "noul":
        p = min(max(answer["p"], 1e-12), 1 - 1e-12)
        t = calib.temperature("noul", 1) * s
        return ["yes"], torch.tensor([t * (math.log(p / (1 - p)) - off.get("yes", 0.0)) - calib.b_noul])
    names = list(answer["probabilities"])
    t = calib.temperature(answer["type"], len(names)) * s
    return names, torch.tensor([t * (math.log(max(answer["probabilities"][k], 1e-30)) - off.get(k, 0.0))
                                for k in names])


def write_predictions(path, rows: list[dict]):
    """One JSON row per line. Creates the folder: data/predictions/ is gitignored, so a fresh clone has none."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def calibration_json(calib: Calibration) -> dict:
    return {"temperatures": calib.temperatures, "b_noul": calib.b_noul} | ({"tasks": calib.tasks} if calib.tasks else {})
