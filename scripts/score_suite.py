"""Score a release candidate against the release criteria (docs/RELEASE.md) on the frozen held-out suite.

    uv run python scripts/score_suite.py --candidate general_v3b_e4b_r8_seed0 --calib-data data/mixes/general_v3b.jsonl
    uv run python scripts/score_suite.py --candidate ... --calib-data ... --sets helpsteer2 goemotions --out x.md
    uv run python scripts/score_suite.py --candidate ... --seeds-only     # criterion 4 alone (development)

Methods:
- the candidate: its bundle's predictions, data/predictions/<bundle>__<set>.jsonl (scripts/predict_bundle.py), else
  <bundle>.jsonl; "before calibration" is the same answers read out at temperature 1 (decisivelm.calibrate.invert);
- P0 (the frozen E4B prompted) and the 26B teacher, from data/teacher/<set>__<teacher>.jsonl (label_teacher.py).
  They are calibrated the way the candidate is (decisivelm.calibrate.fit_calibration) on the same human-labelled rows:
  the gold_calib rows of --calib-data (the candidate's mix), as each labelled them in
  data/teacher/<calib stem>__<teacher>.jsonl. When the candidate's bundle carries a new-task scale
  (scripts/recalibrate.py --new-task), each baseline gets one too, fitted on its own answers to the same held-out
  task's gold_calib rows (data/teacher/<task>__<teacher>.jsonl).
- the per-task step (criterion 2): on each set, 100 of its rows (chosen by a hash of their ids, never scored) give each
  method one Choice and one Score scale on its own answers, as `scripts/recalibrate.py --new-task` would on a user's
  100 labelled items; every criterion is scored on the remaining rows, calibration after that step and everything
  else before it. Nothing else is fitted on the suite.

A task is one set's questions of one type (Measuring Hate Speech has two: its nine Scores and its Noul). Per task,
candidate minus P0 with paired 95% CIs from item-level bootstrap resamples over rows:
1. review readiness, on Choice and Score tasks whose items mostly have ≥ 3 raters: ρ(top p, pair agreement) and tie
   AUROC on those items; within-category ρ (category: a question's consensus level, ≥ 25 items); dynamic range of top
   p over agreement levels with ≥ 25 items; the share of errors caught by reviewing the 20% least confident, and the
   accuracy gained by that review;
2. calibration, after the per-task step: top-label ECE (15 bins) against hard labels (the raters' unique plurality;
   Noul: share ≥ .5), Brier before any calibration and after the step; both are also reported without the step;
3. accuracy: the primary metric, the mean over the task's questions (Choice accuracy, Score MAE, Noul AUROC);
4. the seeds: on each seed's gold_test, against its reference (configs/release_seeds.yaml), the primary metric within
   a margin, ECE ≤ .10 or not significantly above the reference's, and review readiness not significantly below it.
Criteria 5–7 (contract, provenance, snapshot audit) have their own scripts, listed in the report.

A candidate is scored on the suite once: the report (default results/suite_<candidate>.md) must not exist yet.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib

import numpy as np
import torch
import yaml

from decisivelm.calibrate import (QuestionLogits, calibration_json, fit_calibration, fit_new_task_scale, invert,
                                  new_task_scales, question_names, scaled)
from decisivelm.data.dataset import load_records
from decisivelm.eval import agreement_profile, auroc, ece, review_curve, spearman, within_category_spearman
from decisivelm.head import Calibration, readout
from decisivelm.synth.teacher import teacher_logits
from decisivelm.train import _target

REPO = pathlib.Path(__file__).resolve().parent.parent
SUITE = ["helpsteer3", "measuring_hate_speech", "cuad", "massive"]
BASELINES = {"P0": "gemma4_e4b", "teacher": "gemma4_26b_a4b"}
PRIMARY = {"choice": "accuracy", "score": "MAE", "noul": "AUROC"}
REVIEW = 0.2
MIN_N = 25
ECE_0X, ECE_10 = 0.10, 0.07
DYNAMIC_RANGE = 0.2
TASK_CALIB_ROWS = 100   # the per-task step's labelled rows per set (docs/RELEASE.md criterion 2)
MAX_SIBLINGS = 50       # a family this small is one source item in several forms (a MASSIVE utterance in 10 languages)


def as_dict(answer: dict) -> dict[str, float]:
    return {"yes": answer["p"]} if answer["type"] == "noul" else answer["probabilities"]


def candidate_answers(bundle: str, stem: str, rows) -> dict:
    """{row id: {qid: (calibrated, before calibration)}}, each {name: p}."""
    meta = json.loads((REPO / "bundles" / bundle / "calibration.json").read_text())
    calib = Calibration(temperatures=meta["temperatures"], b_noul=meta["b_noul"])
    path = REPO / "data" / "predictions" / f"{bundle}__{stem}.jsonl"
    if not path.exists():
        path = path.with_name(f"{bundle}.jsonl")
    return read_predictions(path, rows, calib)


def read_predictions(path, rows, calib: Calibration | None = None) -> dict:
    """Prediction rows (decisivelm.calibrate format; a single-question row may carry only "probabilities")."""
    by_id, out = {r.id: r for r in rows}, {}
    for line in open(path):
        d = json.loads(line)
        r = by_id.get(d["id"])
        if r is None:
            continue
        answers = d.get("answers") or {next(iter(r.questions)): {"type": next(iter(r.questions.values())).type,
                                                                   "probabilities": d["probabilities"]}}
        out[r.id] = {}
        for qid, a in answers.items():
            raw = as_dict(a)
            if calib is not None:
                names, logits = invert(a, calib)
                raw = dict(zip(names, readout(a["type"], logits, Calibration()).tolist()))
            out[r.id][qid] = (as_dict(a), raw)
    return out


def teacher_calib_questions(teacher: str, path: pathlib.Path) -> list[QuestionLogits]:
    raw = {d["id"]: d for d in map(json.loads, open(REPO / "data" / "teacher" / f"{path.stem}__{teacher}.jsonl"))}
    return [QuestionLogits(r.id, r.split, qid, q.type, question_names(q), teacher_logits(q, ans),
                           _target(q, r.targets.get(qid)))
            for r in load_records(path) if r.split == "gold_calib" and r.id in raw
            for qid, q in r.questions.items() if (ans := raw[r.id]["answers"].get(qid)) is not None]


def new_task_of(bundle: str) -> pathlib.Path | None:
    """The held-out task a bundle's new-task scale was fitted on (data/processed/<stem>.jsonl), if it has one."""
    meta = json.loads((REPO / "bundles" / bundle / "calibration.json").read_text()).get("calibration") or {}
    if meta.get("method") != "new_task_scale":
        return None
    return REPO / "data" / "processed" / f"{meta['fitted_on'].split()[0]}.jsonl"


def baseline_calibration(teacher: str, calib_data: list[pathlib.Path],
                         new_task: pathlib.Path | None = None) -> tuple[Calibration, int]:
    qs = [q for path in calib_data for q in teacher_calib_questions(teacher, path)]
    calib = fit_calibration(qs)
    if new_task is not None:
        calib = scaled(calib, new_task_scales(teacher_calib_questions(teacher, new_task), calib))
    return calib, len(qs)


def baseline_answers(teacher: str, calib: Calibration, stem: str, rows) -> dict:
    path = REPO / "data" / "teacher" / f"{stem}__{teacher}.jsonl"
    by_id, out = {r.id: r for r in rows}, {}
    for d in map(json.loads, open(path)):
        r = by_id.get(d["id"])
        if r is None:
            continue
        out[r.id] = {}
        for qid, ans in d["answers"].items():
            if ans is None:
                continue
            q, names = r.questions[qid], question_names(r.questions[qid])
            logits = teacher_logits(q, ans)
            out[r.id][qid] = tuple(dict(zip(names, readout(q.type, logits, c).tolist())) for c in (calib, Calibration()))
    return out


def task_calib_split(rows) -> tuple[list, list]:
    """(the per-task calibration rows, the rows scored). The calibration rows are the 100 with the lowest hash of their
    id; rows sharing a small family with one of them (the same source item) are in neither, so no item is fitted in one
    form and scored in another. Large families (a set's whole eval portion) are not items and are ignored."""
    calib = sorted(rows, key=lambda r: hashlib.sha256(f"task-calib:{r.id}".encode()).hexdigest())[:TASK_CALIB_ROWS]
    size = collections.Counter(r.lineage.family_id for r in rows)
    ids = {r.id for r in calib}
    families = {r.lineage.family_id for r in calib if size[r.lineage.family_id] <= MAX_SIBLINGS}
    return calib, [r for r in rows if r.id not in ids and r.lineage.family_id not in families]


def per_task(answers: dict, rows, calib_rows) -> tuple[dict, dict[str, float]]:
    """`answers` after the per-task step, and its scales: one factor per type on the method's calibrated answers to
    `calib_rows`, fitted like a new-task scale (decisivelm.calibrate.fit_new_task_scale), and applied to every row as
    p ∝ p^(1/s), which is that factor on the temperature. Noul answers are left as they are, as the new-task scale
    leaves them; before-calibration answers too."""
    qs = [QuestionLogits(r.id, r.split, qid, q.type, question_names(q),
                         torch.log(torch.tensor([answers[r.id][qid][0][k] for k in question_names(q)]).clamp_min(1e-30)),
                         _target(q, r.targets.get(qid)))
          for r in calib_rows for qid, q in r.questions.items() if q.type != "noul"]
    scales = {t: fit_new_task_scale(qs, Calibration(), t) for t in ("choice", "score") if any(q.qtype == t for q in qs)}
    out = {}
    for r in rows:
        out[r.id] = {}
        for qid, (cal, raw) in answers[r.id].items():
            s = scales.get(r.questions[qid].type)
            if s is not None:
                names = list(cal)
                p = torch.softmax(torch.log(torch.tensor([cal[k] for k in names]).clamp_min(1e-30)) / s, -1)
                cal = dict(zip(names, p.tolist()))
            out[r.id][qid] = (cal, raw)
    return out, scales


def unique_argmax(t: np.ndarray) -> int:
    top = np.sort(t)
    return int(t.argmax()) if len(t) < 2 or top[-1] > top[-2] + 1e-9 else -1


class Task:
    """One set's questions of one type over the rows every method answered; per-unit (row, question) arrays."""

    def __init__(self, name: str, qtype: str, rows, answers: dict):
        self.name, self.qtype = name, qtype
        units = [(i, qid, q) for i, r in enumerate(rows) for qid, q in r.questions.items() if q.type == qtype]
        self.row, self.n_rows = np.array([i for i, _, _ in units]), len(rows)
        self.qids = sorted({qid for _, qid, _ in units})
        self.q = np.array([self.qids.index(qid) for _, qid, _ in units])
        human = [rows[i].targets[qid].human for i, qid, _ in units]
        self.m = {}
        if qtype == "noul":
            self.y = np.array([h.noul for h in human]) >= 0.5
            for m, ans in answers.items():
                cal, raw = zip(*(ans[rows[i].id][qid] for i, qid, _ in units))
                self.m[m] = {"p": np.array([c["yes"] for c in cal]), "raw": np.array([c["yes"] for c in raw])}
            self.review = False
            return
        t = [np.array([h.probabilities[k] for k in question_names(q)]) for (_, _, q), h in zip(units, human)]
        n = np.array([h.n_raters or 0 for h in human])
        self.gold = np.array([unique_argmax(x) for x in t])
        self.mean_level = np.array([x @ np.arange(len(x)) for x in t])
        counts = [np.rint(x * k) for x, k in zip(t, n)]
        self.agree = np.array([(c * (c - 1)).sum() / (k * (k - 1)) if k >= 2 else np.nan for c, k in zip(counts, n)])
        self.r3 = n >= 3
        self.category = np.array([f"{qid}={g}" for (_, qid, _), g in zip(units, self.gold)])
        self.review = self.r3.mean() > 0.5
        for m, ans in answers.items():
            vecs = [tuple(np.array([d[k] for k in question_names(q)]) for d in ans[rows[i].id][qid]) for i, qid, q in units]
            cal = [c for c, _ in vecs]
            onehot = lambda v, g: np.eye(len(v))[g] if g >= 0 else np.full(len(v), np.nan)
            self.m[m] = {"top": np.array([c.max() for c in cal]), "pred": np.array([c.argmax() for c in cal]),
                         "level": np.array([c @ np.arange(len(c)) for c in cal]),
                         "brier": np.array([((c - onehot(c, g)) ** 2).sum() for c, g in zip(cal, self.gold)]),
                         "brier raw": np.array([((r - onehot(r, g)) ** 2).sum() for (_, r), g in zip(vecs, self.gold)])}

    def per_question(self, values: np.ndarray, keep: np.ndarray, idx: np.ndarray) -> float:
        q = self.q[idx][keep]
        count = np.bincount(q, minlength=len(self.qids))
        total = np.bincount(q, weights=values[keep].astype(float), minlength=len(self.qids))
        return float(np.mean(total[count > 0] / count[count > 0]))

    def metrics(self, m: str, idx: np.ndarray) -> dict:
        a = self.m[m]
        if self.qtype == "noul":
            p, y = a["p"][idx], self.y[idx]
            correct = (p >= 0.5) == y
            q = self.q[idx]
            aurocs = [auroc(p[q == k], y[q == k]) for k in range(len(self.qids)) if 0 < y[q == k].sum() < (q == k).sum()]
            return {"AUROC": float(np.mean(aurocs)) if aurocs else math.nan, "accuracy": float(correct.mean()),
                    "ECE": ece(np.maximum(p, 1 - p), correct), "Brier": float(((p - y) ** 2).mean()),
                    "Brier raw": float(((a["raw"][idx] - y) ** 2).mean())}
        gold, top = self.gold[idx], a["top"][idx]
        scored = gold >= 0
        correct = a["pred"][idx] == gold
        out = {"accuracy": self.per_question(correct, scored, idx),
               "ECE": ece(top[scored], correct[scored]),
               "Brier": float(a["brier"][idx][scored].mean()), "Brier raw": float(a["brier raw"][idx][scored].mean())}
        if self.qtype == "score":
            out["MAE"] = self.per_question(np.abs(a["level"][idx] - self.mean_level[idx]), np.ones(len(idx), bool), idx)
        if self.review:
            r3 = self.r3[idx]
            out["ρ(top p, agreement)"] = spearman(top[r3], self.agree[idx][r3])
            tie = gold[r3] < 0
            out["tie AUROC"] = auroc(-top[r3], tie) if 0 < tie.sum() < len(tie) else math.nan
            caught, _ = review_curve(top, correct, scored, [REVIEW])
            out["errors caught in review"] = float(caught[0])
            out["accuracy gained by review"] = float(caught[0] * (scored & ~correct).sum() / scored.sum())
        return out

    def point_extras(self, m: str) -> dict:
        """Within-category ρ and dynamic range (no CI; the criteria set a floor on the point value)."""
        if not self.review:
            return {}
        top, r3 = self.m[m]["top"], self.r3 & (self.gold >= 0)
        within = within_category_spearman(top[r3], self.agree[r3], self.category[r3], MIN_N)
        _, spread = agreement_profile(top[self.r3], self.agree[self.r3], MIN_N)
        return {"lowest within-category ρ": min(within.values()) if within else math.nan,
                "categories (≥ 25 items)": len(within), "dynamic range": spread}

    def boot_indices(self, rng, n_boot: int) -> list[np.ndarray]:
        """Unit indices of each resample of rows (units are stored grouped by row)."""
        counts = np.bincount(self.row, minlength=self.n_rows)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        out = []
        for _ in range(n_boot):
            r = rng.integers(0, self.n_rows, self.n_rows)
            c = counts[r]
            out.append(np.repeat(offsets[r], c) + np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
        return out


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def score_task(task: Task, n_boot: int, seed: int, base: str = "P0") -> dict:
    """Point values, each method's CIs, and CIs of candidate minus `base`."""
    point = {m: task.metrics(m, np.arange(len(task.row))) | task.point_extras(m) for m in task.m}
    boots = task.boot_indices(np.random.default_rng(seed), n_boot)
    bs = {m: [task.metrics(m, b) for b in boots] for m in task.m}
    out = {"point": point, "ci": {}, "diff": {},
           "calibration gain": ci(np.array([b["Brier"] - b["Brier raw"] for b in bs["candidate"]]))}
    for k in bs["candidate"][0]:
        for m in task.m:
            out["ci"][m, k] = ci(np.array([b[k] for b in bs[m]]))
        out["diff"][k] = ci(np.array([c[k] - p[k] for c, p in zip(bs["candidate"], bs[base])]))
    return out


def criteria(task: Task, s: dict, s_task: dict) -> list[tuple[str, str, bool, bool]]:
    """(criterion, what, passes 0.x, passes 1.0) for one task: candidate against P0; calibration from `s_task`, the
    scores after the per-task step."""
    c, d = s["point"]["candidate"], s["diff"]
    lower_better = task.qtype == "score"
    not_worse = lambda k, low=False: d[k][0] <= 0 if low else d[k][1] >= 0
    better = lambda k, low=False: d[k][1] < 0 if low else d[k][0] > 0
    out = []
    if task.review:
        for k in ("ρ(top p, agreement)", "tie AUROC"):
            if not math.isnan(c[k]):
                out.append(("1 review", f"{k}: Δ CI [{d[k][0]:+.3f}, {d[k][1]:+.3f}]", not_worse(k), better(k)))
        # 0.x: the floor, or at least P0's (point values; docs/RELEASE.md, 2026-10-24)
        p0 = s["point"]["P0"]
        k = "lowest within-category ρ"
        ok = c[k] > 0
        out.append(("1 review", f"{k} {c[k]:.3f} (P0 {p0[k]:.3f}) over {c['categories (≥ 25 items)']} categories",
                    ok or c[k] >= p0[k], ok))
        k = "dynamic range"
        ok = c[k] >= DYNAMIC_RANGE
        out.append(("1 review", f"{k} {c[k]:.3f} (P0 {p0[k]:.3f})", ok or c[k] >= p0[k], ok))
        k = "errors caught in review"
        out.append(("1 review", f"{k}: Δ CI [{d[k][0]:+.3f}, {d[k][1]:+.3f}]", not_worse(k), better(k)))
        lo, hi = s["ci"]["candidate", "accuracy gained by review"]
        out.append(("1 review", f"accuracy gained by review: CI [{lo:+.3f}, {hi:+.3f}]", lo > 0, lo > 0))
    ct = s_task["point"]["candidate"]
    lo, hi = s_task["ci"]["candidate", "ECE"]
    out.append(("2 calibration", f"ECE after the per-task step {ct['ECE']:.3f}, CI [{lo:.3f}, {hi:.3f}] "
                                 f"(without it {c['ECE']:.3f})", ct["ECE"] <= ECE_0X, hi < ECE_10))
    lo, hi = s_task["calibration gain"]   # Brier after minus before calibration: not significantly worse
    out.append(("2 calibration", f"Brier after the per-task step {ct['Brier']:.4f} vs {ct['Brier raw']:.4f} before "
                                 f"calibration, Δ CI [{lo:+.4f}, {hi:+.4f}]", lo <= 0, lo <= 0))
    k = PRIMARY[task.qtype]
    out.append(("3 accuracy", f"{k}: Δ CI [{d[k][0]:+.3f}, {d[k][1]:+.3f}]", not_worse(k, lower_better),
                better(k, lower_better)))   # 1.0's "better on most tasks" is checked across tasks
    return out


def common_rows(rows, answers: dict):
    """The rows every method answered, each with an answer to every question."""
    keep = [r for r in rows if all(r.id in a for a in answers.values())]
    for m, a in answers.items():
        for r in keep:
            missing = set(r.questions) - set(a[r.id])
            if missing:
                raise SystemExit(f"{m} did not answer {sorted(missing)} of {r.id}")
    return keep


def fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def seed_criteria(task: Task, s: dict, margin: float) -> list[tuple[str, str, bool]]:
    """(criterion, what, passes) for one seed task: candidate against its reference (criterion 4)."""
    c, r, d = s["point"]["candidate"], s["point"]["reference"], s["diff"]
    k = PRIMARY[task.qtype]
    delta = c[k] - r[k]
    ok = delta <= margin if task.qtype == "score" else delta >= -margin
    out = [("accuracy", f"{k} {c[k]:.3f} vs {r[k]:.3f}: Δ {delta:+.3f} [{d[k][0]:+.3f}, {d[k][1]:+.3f}], "
                        f"margin {margin}", ok),
           ("calibration", f"ECE {c['ECE']:.3f} vs {r['ECE']:.3f}: Δ CI [{d['ECE'][0]:+.3f}, {d['ECE'][1]:+.3f}]",
            c["ECE"] <= ECE_0X or d["ECE"][0] <= 0)]
    if task.review:
        for k in ("ρ(top p, agreement)", "tie AUROC", "errors caught in review"):
            if not math.isnan(c[k]):
                out.append(("review", f"{k} {c[k]:.3f} vs {r[k]:.3f}: Δ CI [{d[k][0]:+.3f}, {d[k][1]:+.3f}]",
                            d[k][1] >= 0))
    return out


def within_noise(task: Task, s: dict) -> bool:
    """The aspirational target: the primary metric's Δ CI reaches 0 or better."""
    lo, hi = s["diff"][PRIMARY[task.qtype]]
    return lo <= 0 if task.qtype == "score" else hi >= 0


def seed_checks(bundle: str, n_boot: int, seed: int) -> tuple[list[str], list[bool]]:
    cfg = yaml.safe_load((REPO / "configs" / "release_seeds.yaml").read_text())
    lines = ["## Criterion 4: close to the specialists on the seeds' own gold_test", "",
             "Candidate against the reference in `configs/release_seeds.yaml`, paired 95% CIs from "
             f"{n_boot:,} row resamples. Accuracy margins: {cfg['margins']}; ECE ≤ {ECE_0X} or not significantly above "
             "the reference's; review readiness not significantly below the reference. *Within noise* (the target, not a gate): the primary metric's Δ CI "
             "reaches 0.", "",
             "| Seed | Type | rows | Criterion | Value | Passes |", "|---|---|---|---|---|---|"]
    passes, noise = [], []
    for name, sd in cfg["seeds"].items():
        rows = [r for r in load_records(REPO / sd["data"]) if r.split == "gold_test"]
        answers = {"candidate": read_predictions(REPO / "data" / "predictions" / f"{bundle}.jsonl", rows),
                   "reference": read_predictions(REPO / sd["reference"], rows)}
        rows = common_rows(rows, answers)
        for qtype in sorted({q.type for r in rows for q in r.questions.values()}):
            task = Task(name, qtype, rows, answers)
            s = score_task(task, n_boot, seed, base="reference")
            noise.append(within_noise(task, s))
            for crit, what, ok in seed_criteria(task, s, cfg["margins"][qtype]):
                passes.append(ok)
                lines.append(f"| {name} | {qtype} | {len(rows):,} | {crit} | {what} | {'yes' if ok else '**no**'} |")
    lines += ["", f"Within noise of the reference on the primary metric: {sum(noise)} of {len(noise)} seed tasks.", ""]
    return lines, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="bundle name under bundles/")
    ap.add_argument("--calib-data", nargs="+", type=pathlib.Path,
                    help="the candidate's mix: the baselines are calibrated on its gold_calib rows")
    ap.add_argument("--sets", nargs="+", default=SUITE)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--seeds-only", action="store_true",
                    help="criterion 4 alone, printed (and written to --out if given); a development check")
    a = ap.parse_args()
    if a.seeds_only:
        text = "\n".join([f"# {a.candidate}: criterion 4", ""] + seed_checks(a.candidate, a.n_boot, a.seed)[0])
        print(text)
        if a.out:
            a.out.write_text(text)
        return
    if not a.calib_data:
        ap.error("--calib-data is required")
    out = a.out or REPO / "results" / f"suite_{a.candidate}.md"
    if set(a.sets) & set(SUITE) and out.exists():
        raise SystemExit(f"{out} exists: a candidate is scored on the suite once (docs/RELEASE.md)")

    new_task = new_task_of(a.candidate)
    calibs = {m: baseline_calibration(t, a.calib_data, new_task) for m, t in BASELINES.items()}
    scale = f", Choice and Score temperatures scaled on {new_task.stem} gold_calib (new-task scale)" if new_task else ""
    lines = [f"# {a.candidate} against the release criteria", "",
             f"Sets: {', '.join(a.sets)}. Candidate minus P0, paired 95% CIs from {a.n_boot:,} item-level bootstrap "
             f"resamples (rows). Baselines calibrated on the gold_calib rows of {', '.join(p.name for p in a.calib_data)}"
             f"{scale}:"]
    lines += [f"- {m} ({BASELINES[m]}), {n:,} questions: `{json.dumps(calibration_json(c))}`" for m, (c, n) in calibs.items()]
    lines.append("")
    verdicts, tables, primary_better = [], [], []
    for stem in a.sets:
        rows = [r for r in load_records(REPO / "data" / "processed" / f"{stem}.jsonl") if r.split == "gold_test"]
        answers = {"candidate": candidate_answers(a.candidate, stem, rows)}
        answers |= {m: baseline_answers(BASELINES[m], c, stem, rows) for m, (c, _) in calibs.items()}
        n_all, rows = len(rows), common_rows(rows, answers)
        calib_rows, rows = task_calib_split(rows)
        task_answers, scales = {}, {}
        for m, ans in answers.items():
            task_answers[m], scales[m] = per_task(ans, rows, calib_rows)
        for qtype in ("choice", "score", "noul"):
            if not any(q.type == qtype for r in rows for q in r.questions.values()):
                continue
            task = Task(f"{stem} {qtype}", qtype, rows, answers)
            s = score_task(task, a.n_boot, a.seed)
            s_task = score_task(Task(task.name, qtype, rows, task_answers), a.n_boot, a.seed)
            print(f"{task.name}: {len(rows):,} rows, {len(task.row):,} questions", flush=True)
            verdicts += [(task.name, *v) for v in criteria(task, s, s_task)]
            k = PRIMARY[qtype]
            primary_better.append(s["diff"][k][1] < 0 if qtype == "score" else s["diff"][k][0] > 0)
            keys = list(s["point"]["candidate"])
            tables += [f"### {task.name}", "",
                       f"{len(rows):,} of {n_all:,} rows (those every method answered, less the per-task step's "
                       f"{len(calib_rows)} and their siblings), {len(task.row):,} questions"
                       f"{'; review readiness on items with ≥ 3 raters' if task.review else ''}. Per-task scales: "
                       + "; ".join(f"{m} " + (", ".join(f"{t} {v:.3f}" for t, v in scales[m].items()) or "none")
                                   for m in scales) + ".", "",
                       "| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |", "|---|---|---|---|---|"]
            for key in keys:
                d = s["diff"].get(key)
                diff = (f"{s['point']['candidate'][key] - s['point']['P0'][key]:+.3f} [{d[0]:+.3f}, {d[1]:+.3f}]"
                        if d else "")
                tables.append(f"| {key} | " + " | ".join(fmt(s["point"][m][key]) for m in ("candidate", "P0", "teacher"))
                              + f" | {diff} |")
            for key in ("ECE", "Brier"):
                d, pt = s_task["diff"][key], s_task["point"]
                tables.append(f"| {key} after the per-task step | " + " | ".join(fmt(pt[m][key]) for m in pt)
                              + f" | {pt['candidate'][key] - pt['P0'][key]:+.3f} [{d[0]:+.3f}, {d[1]:+.3f}] |")
            tables.append("")
    seed_lines, seed_passes = seed_checks(a.candidate, a.n_boot, a.seed)
    lines += ["## Criteria 1–3 on each task", "", "| Task | Criterion | Value | 0.x | 1.0 |", "|---|---|---|---|---|"]
    lines += [f"| {t} | {c} | {w} | {'pass' if p else '**fail**'} | {'pass' if q else 'fail'} |"
              for t, c, w, p, q in verdicts]
    most = sum(primary_better) > len(primary_better) / 2
    pass_0x = all(v[3] for v in verdicts) and all(seed_passes)
    pass_10 = all(v[4] for v in verdicts) and all(seed_passes) and most
    lines += ["", f"1.0 also needs the primary metric better than P0 on most tasks: {sum(primary_better)} of "
                  f"{len(primary_better)} ({'pass' if most else 'fail'}).", ""]
    lines += seed_lines
    lines += ["## Criteria 5–7", "", "Checked by their own scripts on the released bundle and snapshot:",
              "- 5 contract: `scripts/check_bundle.py` (conformance) and `scripts/check_invariance.py` (independence, order) "
              "on the real backbone;",
              "- 6 provenance: the mix manifest (`data/mixes/<mix>.manifest.json`, R tier only), the model-origin rule, the "
              "clean-room canary and scanner (`uv run pytest`);",
              "- 7 snapshot audit: the release export's checks.", "",
              f"**Criteria 1–4: {'pass' if pass_0x else 'fail'} for 0.x, {'pass' if pass_10 else 'fail'} for 1.0.**", ""]
    lines += ["## Per-task metrics", ""] + tables
    out.write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
