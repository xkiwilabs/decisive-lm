"""Release criterion 8 (docs/RELEASE.md): is the general head a better starting point for fine-tunes than the base?

    uv run python scripts/score_c8.py --general general_v3c_e4b_r8_seed0_nt [--out results/c8_<general>.md]

Reads the predictions of the fine-tune runs (data/predictions/c8_{task}_{base|general}_n{N}_seed{S}.jsonl)
and scores them on gold_test against the human targets, with scripts/eval_dataset.py's metrics. For each task and
training size, each arm's value is the mean over its seeds; Δ = general minus base, with a 95% CI from item-level
bootstrap resamples shared by every run (so it reflects item sampling, not seed variation; the per-seed values are
listed so the seed spread can be read too). "Worse" means the CI lies wholly on the wrong side of 0; "better", wholly
on the right side.
"""
import argparse, pathlib, re

import numpy as np

from decisivelm.data.dataset import load_records

import eval_dataset as ev

REPO = pathlib.Path(__file__).resolve().parent.parent
TASKS = {"coding": "data/processed/llm_response_coding.jsonl", "multinli": "data/c8/multinli.jsonl"}
# metric -> True when higher is better; the primary metric first, then review readiness (docs/RELEASE.md criterion 1), then calibration
METRICS = {"accuracy": True, "ρ(top p, agreement)": True, "tie AUROC": True, f"errors in review {ev.REVIEW:.0%}": True,
           "ECE": False, "soft NLL": False}
REVIEW = ("ρ(top p, agreement)", "tie AUROC", f"errors in review {ev.REVIEW:.0%}")


def verdict(lo, hi, higher):
    if (hi < 0) if higher else (lo > 0):
        return "worse"
    if (lo > 0) if higher else (hi < 0):
        return "better"
    return "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", required=True, help="the general bundle the c8 runs started from (for the report)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    pred_dir = REPO / "data/predictions"
    runs = {}
    for p in pred_dir.glob("c8_*.jsonl"):
        if m := re.fullmatch(r"c8_(\w+?)_(base|general)_n(\d+)_seed(\d+)", p.stem):
            runs.setdefault((m[1], int(m[3])), {}).setdefault(m[2], {})[int(m[4])] = p
    lines = [f"# Criterion 8: fine-tuning from {a.general} vs from the base", "",
             "Adapters fine-tuned on N labelled training items (scripts/train_p3.py --n-train; 3 seeds per arm and size), "
             "from the general head (--init-bundle) or from the base model, scored on gold_test against the human "
             "targets. Arm value = mean over seeds; Δ = general − base; 95% CI from "
             f"{a.n_boot} item-level bootstrap resamples shared by every run. Primary metric: accuracy against the "
             "human plurality (ties left out). Review readiness: ρ of top p with per-item pair agreement, tie AUROC, "
             f"and the share of errors among the least-confident {ev.REVIEW:.0%}.", ""]
    summary = []
    rng = np.random.default_rng(a.seed)
    for task, data in TASKS.items():
        rows = [r for r in load_records(REPO / data) if r.split == "gold_test"]
        qid, q0 = next(iter(rows[0].questions.items()))
        assert q0.type == "choice", "criterion 8 tasks are single Choice questions"
        Q = ev.Question(rows, qid, "choice", list(q0.criteria), None)
        boots = [rng.integers(0, len(rows), len(rows)) for _ in range(a.n_boot)]
        lines += [f"## {task} ({len(rows):,} gold_test items, {len(q0.criteria)} options)", ""]
        for n in sorted(k[1] for k in runs if k[0] == task):
            arms = runs[task, n]
            seeds = sorted(set(arms.get("base", {})) & set(arms.get("general", {})))
            P = {(arm, s): Q.probs(ev.load_predictions(arms[arm][s], qid)) for arm in ("base", "general") for s in seeds}
            point = {k: Q.metrics(p, np.arange(len(rows))) for k, p in P.items()}
            boot = {k: [Q.metrics(p, b) for b in boots] for k, p in P.items()}
            lines += [f"### {n} training items ({len(seeds)} seeds)", "",
                      "| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |",
                      "|---|---|---|---|---|---|---|---|"]
            for metric, higher in METRICS.items():
                v = {arm: [point[arm, s][metric] for s in seeds] for arm in ("base", "general")}
                d = np.array([np.mean([boot["general", s][i][metric] for s in seeds])
                              - np.mean([boot["base", s][i][metric] for s in seeds]) for i in range(a.n_boot)])
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                diff = np.mean(v["general"]) - np.mean(v["base"])
                call = verdict(lo, hi, higher)
                lines.append(f"| {metric} | {', '.join(ev.fmt(x) for x in v['base'])} | "
                             f"{', '.join(ev.fmt(x) for x in v['general'])} | {np.mean(v['base']):.3f} | "
                             f"{np.mean(v['general']):.3f} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {call} |")
                summary.append((task, n, metric, call))
            lines.append("")
    gated = [m for m in METRICS if m == "accuracy" or m in REVIEW]
    worse = [(t, n, m) for t, n, m, c in summary if c == "worse" and m in gated]
    better100 = [(t, m) for t, n, m, c in summary if c == "better" and n == 100 and m in gated]
    lines += ["## Against criterion 8", "",
              "- Not significantly worse at either size, on accuracy or review readiness: "
              + ("**yes**" if not worse else "**no**: " + "; ".join(f"{t} n={n} {m}" for t, n, m in worse)) + ".",
              "- Better at 100 items on at least one task: "
              + ("**yes**: " + "; ".join(f"{t} {m}" for t, m in better100) if better100 else "**no**") + ".", ""]
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
