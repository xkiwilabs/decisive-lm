# 26B head minus E4B head on Measuring Hate Speech Score, paired: score_suite.py's own rows, metrics and row bootstrap
# (1,000 resamples, seed 0), on the gold_test rows that both heads and both prompted baselines answered, less the
# per-task step's rows. A comparison of written predictions, not a new suite scoring (no criteria are applied).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from score_suite import (REPO, Task, baseline_answers, baseline_calibration, baselines, candidate_answers, common_rows,
                         load_records, score_task, task_calib_split)
mix = [REPO / "data/mixes/general_v5c.jsonl"]
heads = {"candidate": "general_v5c_26b-a4b_r8_seed0", "E4B": "general_v5c_e4b_r8_seed0"}
rows = [r for r in load_records(REPO / "data/processed/measuring_hate_speech.jsonl") if r.split == "gold_test"]
answers = {k: candidate_answers(b, "measuring_hate_speech", rows) for k, b in heads.items()}
both = {}
for k, b in heads.items():
    for m, t in baselines(b).items():
        both[t] = baseline_answers(t, baseline_calibration(t, mix, None)[0], "measuring_hate_speech", rows)
rows = common_rows(rows, answers | {f"base {t}": a for t, a in both.items()})
_, rows = task_calib_split(rows)
task = Task("measuring_hate_speech score", "score", rows, answers)
s = score_task(task, 1000, 0, base="E4B")
print(f"{len(rows):,} rows, {len(task.row):,} questions")
for k in ("ρ(top p, agreement)", "tie AUROC", "errors caught in review", "MAE"):
    c, e, d = s["point"]["candidate"][k], s["point"]["E4B"][k], s["diff"][k]
    print(f"{k}: 26B {c:.3f}  E4B {e:.3f}  Δ {c - e:+.3f} [{d[0]:+.3f}, {d[1]:+.3f}]")
