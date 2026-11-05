"""scripts/score_suite.py: the release criteria on one task (docs/RELEASE.md)."""
import importlib.util
import pathlib
from types import SimpleNamespace as NS

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("score_suite", ROOT / "scripts" / "score_suite.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def score_rows(n=300):
    """Rows with one 3-level Score rated by 3 raters: all agree, 2–1, or a three-way tie."""
    rows = []
    for i in range(n):
        kind, k = i % 3, (i // 3) % 3
        votes = [[k, k, k], [k, k, (k + 1) % 3], [0, 1, 2]][kind]
        shares = np.bincount(votes, minlength=3) / 3
        q = NS(type="score", criteria=["low", "mid", "high"])
        human = NS(probabilities={str(k): float(shares[k]) for k in range(3)}, n_raters=3, noul=None)
        rows.append(NS(id=f"r{i}", questions={"q": q}, targets={"q": NS(human=human)}))
    return rows


def answers_tracking_agreement(rows, sharp):
    """Confident (and right) where raters agree, unsure where they split; `sharp` scales how much."""
    out = {}
    for r in rows:
        t = np.array([r.targets["q"].human.probabilities[str(k)] for k in range(3)])
        p = (1 - sharp) / 3 + sharp * t
        d = {str(k): float(p[k]) for k in range(3)}
        out[r.id] = {"q": (d, d)}
    return out


def test_boot_indices_keep_each_rows_units_together():
    rows = [NS(id=f"r{i}", questions={f"q{j}": NS(type="noul", criteria={}) for j in range(i % 3 + 1)},
               targets={f"q{j}": NS(human=NS(noul=float(j % 2))) for j in range(3)}) for i in range(10)]
    answers = {"candidate": {r.id: {q: ({"yes": 0.5}, {"yes": 0.5}) for q in r.questions} for r in rows}}
    task = ss.Task("t", "noul", rows, answers)
    for idx in task.boot_indices(np.random.default_rng(0), 5):
        rows_in = task.row[idx]
        counts = np.bincount(task.row, minlength=10)
        assert all((rows_in == k).sum() % counts[k] == 0 for k in set(rows_in.tolist()))


def random_answers(rows, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for r in rows:
        p = rng.dirichlet(np.ones(3))
        d = {str(k): float(p[k]) for k in range(3)}
        out[r.id] = {"q": (d, d)}
    return out


def test_a_candidate_tracking_human_agreement_passes_review_readiness_against_an_uninformed_p0():
    rows = score_rows()
    answers = {"candidate": answers_tracking_agreement(rows, 0.9), "P0": random_answers(rows),
               "teacher": answers_tracking_agreement(rows, 0.5)}
    task = ss.Task("t", "score", rows, answers)
    assert task.review
    s = ss.score_task(task, n_boot=50, seed=0)
    c = s["point"]["candidate"]
    assert c["ρ(top p, agreement)"] > 0.8 and c["tie AUROC"] > 0.9 and c["dynamic range"] > 0.5
    rho = next(v for v in ss.criteria(task, s, s) if v[1].startswith("ρ"))
    assert rho[2] and rho[3]   # not worse (0.x) and better (1.0) than P0


def test_noul_auroc_is_the_mean_over_questions():
    rows = [NS(id=f"r{i}", questions={"a": NS(type="noul", criteria={}), "b": NS(type="noul", criteria={})},
               targets={"a": NS(human=NS(noul=float(i % 2))), "b": NS(human=NS(noul=float(i % 2)))}) for i in range(40)]
    good = {r.id: {"a": ({"yes": 0.9 if i % 2 else 0.1},) * 2, "b": ({"yes": 0.1 if i % 2 else 0.9},) * 2}
            for i, r in enumerate(rows)}
    task = ss.Task("t", "noul", rows, {"candidate": good})
    assert task.metrics("candidate", np.arange(len(task.row)))["AUROC"] == 0.5   # 1.0 on a, 0.0 on b


def test_seed_criteria_hold_review_readiness_to_the_reference_and_accuracy_to_a_margin():
    rows = score_rows()
    answers = {"candidate": answers_tracking_agreement(rows, 0.9), "reference": answers_tracking_agreement(rows, 0.9)}
    task = ss.Task("t", "score", rows, answers)
    s = ss.score_task(task, n_boot=50, seed=0, base="reference")
    assert all(ok for _, _, ok in ss.seed_criteria(task, s, margin=0.05)[:1])   # identical: accuracy passes
    assert ss.within_noise(task, s)
    answers["candidate"] = random_answers(rows)   # same level of noise, but confidence no longer tracks agreement
    task = ss.Task("t", "score", rows, answers)
    s = ss.score_task(task, n_boot=50, seed=0, base="reference")
    review = [ok for crit, _, ok in ss.seed_criteria(task, s, margin=0.05) if crit == "review"]
    assert review and not any(review)


def test_floors_pass_at_0x_when_p0_is_no_better_and_brier_tolerates_noise():
    # Both flat: neither reaches the dynamic-range floor, and the candidate is no worse than P0 (2026-10-24 change).
    rows = score_rows()
    answers = {m: answers_tracking_agreement(rows, 0.01) for m in ("candidate", "P0", "teacher")}
    task = ss.Task("t", "score", rows, answers)
    s = ss.score_task(task, n_boot=50, seed=0)
    verdicts = {v[1].split(" ")[0] + v[1].split(" ")[1]: v for v in ss.criteria(task, s, s)}
    dyn = verdicts["dynamicrange"]
    assert s["point"]["candidate"]["dynamic range"] < ss.DYNAMIC_RANGE and dyn[2] and not dyn[3]
    brier = next(v for v in ss.criteria(task, s, s) if v[1].startswith("Brier"))
    assert brier[2]   # calibrated = raw here: Δ is 0, not significantly worse


def test_the_per_task_split_keeps_an_items_other_forms_out_of_scoring():
    rows = [NS(id=f"r{i}", lineage=NS(family_id=f"utt-{i // 10}")) for i in range(2000)]   # one item in 10 forms
    calib, scored = ss.task_calib_split(rows)
    assert len(calib) == ss.TASK_CALIB_ROWS and calib == ss.task_calib_split(list(reversed(rows)))[0]
    fitted = {r.lineage.family_id for r in calib}
    assert scored and not fitted & {r.lineage.family_id for r in scored}
    whole_set = [NS(id=r.id, lineage=NS(family_id="eval")) for r in rows]   # a set-wide family is not an item
    assert len(ss.task_calib_split(whole_set)[1]) == len(rows) - ss.TASK_CALIB_ROWS


def test_the_per_task_step_softens_an_overconfident_method_and_leaves_its_raw_answers():
    rows = [NS(split="gold_test", **vars(r)) for r in score_rows(600)]
    answers = {}
    for i, r in enumerate(rows):   # 0.96 on one level, right only half the time
        k = (i // 3) % 3
        top = k if i % 2 else (k + 1) % 3
        d = {str(j): 0.96 if j == top else 0.02 for j in range(3)}
        answers[r.id] = {"q": (d, {str(j): 1 / 3 for j in range(3)})}
    after, scales, forms = ss.per_task(answers, rows, rows[:100])
    assert forms == {}
    assert set(scales) == {"score"} and scales["score"] > 1
    before = ss.Task("t", "score", rows, {"candidate": answers}).metrics("candidate", np.arange(len(rows)))
    task = ss.Task("t", "score", rows, {"candidate": after})
    m = task.metrics("candidate", np.arange(len(rows)))
    assert m["ECE"] < before["ECE"] and m["Brier raw"] == before["Brier raw"]
    assert (task.m["candidate"]["pred"] == ss.Task("t", "score", rows, {"candidate": answers}).m["candidate"]["pred"]).all()


def test_p0_is_the_candidates_own_backbone_prompted(monkeypatch):
    for backbone, p0 in [("e4b", "gemma4_e4b"), ("26b-a4b", "gemma4_26b_a4b")]:
        monkeypatch.setattr(ss, "bundle_backbone", lambda bundle, b=backbone: b)
        assert ss.baselines("any") == {"P0": p0, "teacher": "gemma4_26b_a4b"}


def overconfident_score_answers(rows, raw_level):
    """0.96 on one level, right half the time; before calibration all mass on `raw_level` (None: uniform)."""
    answers = {}
    for i, r in enumerate(rows):
        k = (i // 3) % 3
        top = k if i % 2 else (k + 1) % 3
        d = {str(j): 0.96 if j == top else 0.02 for j in range(3)}
        raw = {str(j): 1 / 3 for j in range(3)} if raw_level is None else {str(j): float(j == raw_level) for j in range(3)}
        answers[r.id] = {"q": (d, raw)}
    return answers


def test_the_guard_keeps_the_scale_where_it_helps():
    rows = [NS(split="gold_test", **vars(r)) for r in score_rows(600)]
    answers = overconfident_score_answers(rows, raw_level=0)   # raw answers are worse still: certain and often wrong
    plain, plain_scales, _ = ss.per_task(answers, rows, rows[:100])
    after, scales, forms = ss.per_task(answers, rows, rows[:100], guard=True)
    assert forms == {"score": "scaled"} and scales == plain_scales and after == plain


def test_the_guard_falls_back_when_the_step_would_make_brier_worse():
    rows = [NS(split="gold_test", **vars(r)) for r in score_rows(600)]
    answers = overconfident_score_answers(rows, raw_level=None)   # uniform raw answers beat 0.96-on-a-coin-flip
    after, scales, forms = ss.per_task(answers, rows, rows[:100], guard=True)
    assert forms == {"score": "raw"} and scales == {}
    assert all(after[r.id]["q"][0] == answers[r.id]["q"][1] for r in rows)
