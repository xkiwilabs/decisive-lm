# general_v5c_26b-a4b_r8_seed0 against the release criteria

Sets: helpsteer3, measuring_hate_speech, cuad, massive. Candidate minus P0, paired 95% CIs from 1,000 item-level bootstrap resamples (rows). Baselines calibrated on the gold_calib rows of general_v5c.jsonl:
- P0 (gemma4_26b_a4b), 80,996 questions: `{"temperatures": {"choice:*": 2.862873158208299, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.8014187812805176, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.776196001310275, "choice:2": 2.862873158208299, "choice:6-15": 2.4819708042242072}, "b_noul": -9.985109329223633}`
- teacher (gemma4_26b_a4b), 80,996 questions: `{"temperatures": {"choice:*": 2.862873158208299, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.8014187812805176, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.776196001310275, "choice:2": 2.862873158208299, "choice:6-15": 2.4819708042242072}, "b_noul": -9.985109329223633}`

## Criteria 1–3 on each task

| Task | Criterion | Value | 0.x | 1.0 |
|---|---|---|---|---|
| massive choice | 2 calibration | ECE after the per-task step 0.031, CI [0.027, 0.036] (without it 0.099) | pass | pass |
| massive choice | 2 calibration | Brier after the per-task step 0.4448 vs 0.4799 before calibration, Δ CI [-0.0376, -0.0326] | pass | pass |
| massive choice | 3 accuracy | accuracy: Δ CI [-0.055, -0.046] | **fail** | fail |

1.0 also needs the primary metric better than P0 on most tasks: 0 of 1 (fail).

## Criterion 4: close to the specialists on the seeds' own gold_test

Candidate against the reference in `configs/release_seeds.yaml`, paired 95% CIs from 1,000 row resamples. Accuracy margins: {'choice': 0.03, 'noul': 0.03, 'score': 0.05}; ECE ≤ 0.1 or not significantly above the reference's; review readiness not significantly below the reference. *Within noise* (the target, not a gate): the primary metric's Δ CI reaches 0.

| Seed | Type | rows | Criterion | Value | Passes |
|---|---|---|---|---|---|
| banking77 | choice | 1,819 | accuracy | accuracy 0.899 vs 0.902: Δ -0.003 [-0.016, +0.013], margin 0.03 | yes |
| banking77 | choice | 1,819 | calibration | ECE 0.025 vs 0.016: Δ CI [-0.006, +0.020] | yes |
| clinc150 | choice | 3,345 | accuracy | accuracy 0.923 vs 0.920: Δ +0.004 [-0.005, +0.012], margin 0.03 | yes |
| clinc150 | choice | 3,345 | calibration | ECE 0.021 vs 0.014: Δ CI [-0.004, +0.017] | yes |
| helpsteer2 | score | 658 | accuracy | MAE 0.455 vs 0.468: Δ -0.013 [-0.023, -0.002], margin 0.05 | yes |
| helpsteer2 | score | 658 | calibration | ECE 0.102 vs 0.105: Δ CI [-0.015, +0.006] | yes |
| helpsteer2 | score | 658 | review | ρ(top p, agreement) 0.265 vs 0.263: Δ CI [-0.023, +0.028] | yes |
| helpsteer2 | score | 658 | review | tie AUROC 0.617 vs 0.613: Δ CI [-0.017, +0.025] | yes |
| helpsteer2 | score | 658 | review | errors caught in review 0.324 vs 0.330: Δ CI [-0.039, +0.020] | yes |
| civil_comments | noul | 5,974 | accuracy | AUROC 0.973 vs 0.982: Δ -0.009 [-0.014, -0.005], margin 0.03 | yes |
| civil_comments | noul | 5,974 | calibration | ECE 0.017 vs 0.018: Δ CI [-0.002, +0.001] | yes |
| goemotions | choice | 1,500 | accuracy | accuracy 0.732 vs 0.760: Δ -0.028 [-0.044, -0.011], margin 0.03 | yes |
| goemotions | choice | 1,500 | calibration | ECE 0.082 vs 0.108: Δ CI [-0.044, -0.011] | yes |
| goemotions | choice | 1,500 | review | ρ(top p, agreement) 0.420 vs 0.420: Δ CI [-0.024, +0.026] | yes |
| goemotions | choice | 1,500 | review | tie AUROC 0.687 vs 0.675: Δ CI [-0.022, +0.047] | yes |
| goemotions | choice | 1,500 | review | errors caught in review 0.369 vs 0.408: Δ CI [-0.080, +0.016] | yes |
| goemotions | noul | 1,500 | accuracy | AUROC 0.946 vs 0.952: Δ -0.006 [-0.011, -0.001], margin 0.03 | yes |
| goemotions | noul | 1,500 | calibration | ECE 0.012 vs 0.014: Δ CI [-0.002, -0.000] | yes |
| ledgar | choice | 1,000 | accuracy | accuracy 0.830 vs 0.821: Δ +0.009 [-0.008, +0.026], margin 0.03 | yes |
| ledgar | choice | 1,000 | calibration | ECE 0.075 vs 0.032: Δ CI [+0.013, +0.060] | yes |

Within noise of the reference on the primary metric: 4 of 7 seed tasks.

## Criteria 5–7

Checked by their own scripts on the released bundle and snapshot:
- 5 contract: `scripts/check_bundle.py` (conformance) and `scripts/check_invariance.py` (independence, order) on the real backbone;
- 6 provenance: the mix manifest (`data/mixes/<mix>.manifest.json`, R tier only), the model-origin rule, the clean-room canary and scanner (`uv run pytest`);
- 7 snapshot audit: the release export's checks.

**Criteria 1–4: fail for 0.x, fail for 1.0.**

## Per-task metrics

### massive choice

28,769 of 29,739 rows (those every method answered, less the per-task step's 100 and their siblings), 28,769 questions. Per-task scales: candidate choice 1.430; P0 choice 0.976; teacher choice 0.976.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.680 | 0.731 | 0.731 | -0.051 [-0.055, -0.046] |
| ECE | 0.099 | 0.041 | 0.041 | +0.058 [+0.051, +0.064] |
| Brier | 0.459 | 0.391 | 0.391 | +0.068 [+0.063, +0.073] |
| Brier raw | 0.480 | 0.419 | 0.419 | +0.061 [+0.055, +0.067] |
| ECE after the per-task step | 0.031 | 0.033 | 0.033 | -0.002 [-0.008, +0.004] |
| Brier after the per-task step | 0.445 | 0.390 | 0.390 | +0.055 [+0.050, +0.059] |
