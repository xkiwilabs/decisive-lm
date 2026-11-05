# general_v5c_e4b_r8_seed0 against the release criteria

Sets: helpsteer3, measuring_hate_speech, cuad, massive. Candidate minus P0, paired 95% CIs from 1,000 item-level bootstrap resamples (rows). Baselines calibrated on the gold_calib rows of general_v5c.jsonl:
- P0 (gemma4_e4b), 80,996 questions: `{"temperatures": {"choice:*": 2.298312742369242, "score:*": 6.817174091832383, "choice:64-255": 1.4812322573882764, "score:3-5": 6.817174091832383, "noul": 5.243362903594971, "choice:16-63": 1.6898920014441763, "choice:3-5": 6.175577027545438, "choice:2": 2.298312742369242, "choice:6-15": 1.7852835191771719}, "b_noul": -10.294469833374023}`
- teacher (gemma4_26b_a4b), 80,996 questions: `{"temperatures": {"choice:*": 2.862873158208299, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.8014187812805176, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.776196001310275, "choice:2": 2.862873158208299, "choice:6-15": 2.4819708042242072}, "b_noul": -9.985109329223633}`

## Criteria 1–3 on each task

| Task | Criterion | Value | 0.x | 1.0 |
|---|---|---|---|---|
| helpsteer3 score | 1 review | ρ(top p, agreement): Δ CI [-0.010, +0.048] | pass | fail |
| helpsteer3 score | 1 review | tie AUROC: Δ CI [-0.038, +0.019] | pass | fail |
| helpsteer3 score | 1 review | lowest within-category ρ -0.047 (P0 -0.107) over 6 categories | pass | fail |
| helpsteer3 score | 1 review | dynamic range 0.013 (P0 0.005) | pass | fail |
| helpsteer3 score | 1 review | errors caught in review: Δ CI [-0.003, +0.016] | pass | fail |
| helpsteer3 score | 1 review | accuracy gained by review: CI [+0.150, +0.161] | pass | pass |
| helpsteer3 score | 2 calibration | ECE after the per-task step 0.027, CI [0.019, 0.040] (without it 0.037) | pass | pass |
| helpsteer3 score | 2 calibration | Brier after the per-task step 0.7823 vs 0.7829 before calibration, Δ CI [-0.0009, -0.0003] | pass | pass |
| helpsteer3 score | 3 accuracy | MAE: Δ CI [-0.287, -0.250] | pass | pass |
| measuring_hate_speech score | 1 review | ρ(top p, agreement): Δ CI [-0.138, -0.110] | **fail** | fail |
| measuring_hate_speech score | 1 review | tie AUROC: Δ CI [-0.055, -0.037] | **fail** | fail |
| measuring_hate_speech score | 1 review | lowest within-category ρ -0.223 (P0 -0.311) over 44 categories | pass | fail |
| measuring_hate_speech score | 1 review | dynamic range 0.092 (P0 0.083) | pass | fail |
| measuring_hate_speech score | 1 review | errors caught in review: Δ CI [-0.000, +0.013] | pass | fail |
| measuring_hate_speech score | 1 review | accuracy gained by review: CI [+0.138, +0.144] | pass | pass |
| measuring_hate_speech score | 2 calibration | ECE after the per-task step 0.046, CI [0.040, 0.053] (without it 0.058) | pass | pass |
| measuring_hate_speech score | 2 calibration | Brier after the per-task step 0.6715 vs 0.6752 before calibration, Δ CI [-0.0044, -0.0032] | pass | pass |
| measuring_hate_speech score | 3 accuracy | MAE: Δ CI [-0.011, +0.006] | pass | fail |
| measuring_hate_speech noul | 2 calibration | ECE after the per-task step 0.047, CI [0.038, 0.062] (without it 0.047) | pass | pass |
| measuring_hate_speech noul | 2 calibration | Brier after the per-task step 0.1536 vs 0.1469 before calibration, Δ CI [+0.0055, +0.0077] | **fail** | fail |
| measuring_hate_speech noul | 3 accuracy | AUROC: Δ CI [-0.013, +0.007] | pass | fail |
| cuad noul | 2 calibration | ECE after the per-task step 0.024, CI [0.022, 0.026] (without it 0.024) | pass | pass |
| cuad noul | 2 calibration | Brier after the per-task step 0.0228 vs 0.0232 before calibration, Δ CI [-0.0006, -0.0002] | pass | pass |
| cuad noul | 3 accuracy | AUROC: Δ CI [-0.010, +0.014] | pass | fail |
| massive choice | 2 calibration | ECE after the per-task step 0.018, CI [0.017, 0.025] (without it 0.094) | pass | pass |
| massive choice | 2 calibration | Brier after the per-task step 0.4473 vs 0.4921 before calibration, Δ CI [-0.0474, -0.0423] | pass | pass |
| massive choice | 3 accuracy | accuracy: Δ CI [+0.010, +0.019] | pass | pass |

1.0 also needs the primary metric better than P0 on most tasks: 2 of 5 (fail).

## Criterion 4: close to the specialists on the seeds' own gold_test

Candidate against the reference in `configs/release_seeds.yaml`, paired 95% CIs from 1,000 row resamples. Accuracy margins: {'choice': 0.03, 'noul': 0.03, 'score': 0.05}; ECE ≤ 0.1 or not significantly above the reference's; review readiness not significantly below the reference. *Within noise* (the target, not a gate): the primary metric's Δ CI reaches 0.

| Seed | Type | rows | Criterion | Value | Passes |
|---|---|---|---|---|---|
| banking77 | choice | 1,819 | accuracy | accuracy 0.909 vs 0.902: Δ +0.008 [-0.005, +0.021], margin 0.03 | yes |
| banking77 | choice | 1,819 | calibration | ECE 0.017 vs 0.016: Δ CI [-0.012, +0.012] | yes |
| clinc150 | choice | 3,345 | accuracy | accuracy 0.943 vs 0.920: Δ +0.024 [+0.016, +0.032], margin 0.03 | yes |
| clinc150 | choice | 3,345 | calibration | ECE 0.017 vs 0.014: Δ CI [-0.009, +0.011] | yes |
| helpsteer2 | score | 658 | accuracy | MAE 0.466 vs 0.468: Δ -0.002 [-0.012, +0.008], margin 0.05 | yes |
| helpsteer2 | score | 658 | calibration | ECE 0.123 vs 0.105: Δ CI [+0.005, +0.028] | **no** |
| helpsteer2 | score | 658 | review | ρ(top p, agreement) 0.271 vs 0.263: Δ CI [-0.016, +0.032] | yes |
| helpsteer2 | score | 658 | review | tie AUROC 0.614 vs 0.613: Δ CI [-0.018, +0.022] | yes |
| helpsteer2 | score | 658 | review | errors caught in review 0.327 vs 0.330: Δ CI [-0.030, +0.023] | yes |
| civil_comments | noul | 5,974 | accuracy | AUROC 0.979 vs 0.982: Δ -0.003 [-0.005, -0.001], margin 0.03 | yes |
| civil_comments | noul | 5,974 | calibration | ECE 0.022 vs 0.018: Δ CI [+0.003, +0.005] | yes |
| goemotions | choice | 1,500 | accuracy | accuracy 0.743 vs 0.760: Δ -0.017 [-0.034, +0.000], margin 0.03 | yes |
| goemotions | choice | 1,500 | calibration | ECE 0.100 vs 0.108: Δ CI [-0.025, +0.008] | yes |
| goemotions | choice | 1,500 | review | ρ(top p, agreement) 0.408 vs 0.420: Δ CI [-0.041, +0.015] | yes |
| goemotions | choice | 1,500 | review | tie AUROC 0.689 vs 0.675: Δ CI [-0.023, +0.049] | yes |
| goemotions | choice | 1,500 | review | errors caught in review 0.384 vs 0.408: Δ CI [-0.064, +0.026] | yes |
| goemotions | noul | 1,500 | accuracy | AUROC 0.951 vs 0.952: Δ -0.002 [-0.004, +0.001], margin 0.03 | yes |
| goemotions | noul | 1,500 | calibration | ECE 0.011 vs 0.014: Δ CI [-0.003, -0.002] | yes |
| ledgar | choice | 1,000 | accuracy | accuracy 0.836 vs 0.821: Δ +0.015 [-0.002, +0.030], margin 0.03 | yes |
| ledgar | choice | 1,000 | calibration | ECE 0.092 vs 0.032: Δ CI [+0.026, +0.077] | yes |

Within noise of the reference on the primary metric: 6 of 7 seed tasks.

## Criteria 5–7

Checked by their own scripts on the released bundle and snapshot:
- 5 contract: `scripts/check_bundle.py` (conformance) and `scripts/check_invariance.py` (independence, order) on the real backbone;
- 6 provenance: the mix manifest (`data/mixes/<mix>.manifest.json`, R tier only), the model-origin rule, the clean-room canary and scanner (`uv run pytest`);
- 7 snapshot audit: the release export's checks.

**Criteria 1–4: fail for 0.x, fail for 1.0.**

## Per-task metrics

### helpsteer3 score

5,866 of 5,966 rows (those every method answered, less the per-task step's 100 and their siblings), 5,866 questions; review readiness on items with ≥ 3 raters. Per-task scales: candidate score 1.121; P0 score 2.000; teacher score 1.463.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.286 | 0.197 | 0.293 | +0.089 [+0.077, +0.101] |
| ECE | 0.037 | 0.038 | 0.016 | -0.001 [-0.012, +0.013] |
| Brier | 0.783 | 0.833 | 0.807 | -0.049 [-0.053, -0.045] |
| Brier raw | 0.783 | 0.999 | 1.050 | -0.216 [-0.225, -0.207] |
| MAE | 1.023 | 1.292 | 1.158 | -0.269 [-0.287, -0.250] |
| ρ(top p, agreement) | 0.061 | 0.043 | 0.088 | +0.018 [-0.010, +0.048] |
| tie AUROC | 0.507 | 0.518 | 0.511 | -0.011 [-0.038, +0.019] |
| errors caught in review | 0.217 | 0.210 | 0.217 | +0.007 [-0.003, +0.016] |
| accuracy gained by review | 0.155 | 0.169 | 0.153 | -0.014 [-0.021, -0.007] |
| lowest within-category ρ | -0.047 | -0.107 | -0.011 |  |
| categories (≥ 25 items) | 6.000 | 6.000 | 6.000 |  |
| dynamic range | 0.013 | 0.005 | 0.013 |  |
| ECE after the per-task step | 0.027 | 0.038 | 0.052 | -0.011 [-0.022, +0.006] |
| Brier after the per-task step | 0.782 | 0.830 | 0.808 | -0.048 [-0.052, -0.043] |

### measuring_hate_speech score

3,284 of 3,384 rows (those every method answered, less the per-task step's 100 and their siblings), 29,556 questions; review readiness on items with ≥ 3 raters. Per-task scales: candidate score 1.188; P0 score 0.811; teacher score 0.739.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.417 | 0.447 | 0.501 | -0.031 [-0.038, -0.024] |
| ECE | 0.058 | 0.033 | 0.075 | +0.025 [+0.015, +0.034] |
| Brier | 0.677 | 0.662 | 0.653 | +0.015 [+0.011, +0.020] |
| Brier raw | 0.675 | 0.930 | 0.845 | -0.255 [-0.268, -0.243] |
| MAE | 0.653 | 0.657 | 0.703 | -0.004 [-0.011, +0.006] |
| ρ(top p, agreement) | 0.078 | 0.201 | 0.162 | -0.124 [-0.138, -0.110] |
| tie AUROC | 0.533 | 0.578 | 0.562 | -0.046 [-0.055, -0.037] |
| errors caught in review | 0.245 | 0.238 | 0.267 | +0.007 [-0.000, +0.013] |
| accuracy gained by review | 0.141 | 0.131 | 0.132 | +0.010 [+0.006, +0.014] |
| lowest within-category ρ | -0.223 | -0.311 | -0.175 |  |
| categories (≥ 25 items) | 44.000 | 44.000 | 44.000 |  |
| dynamic range | 0.092 | 0.083 | 0.043 |  |
| ECE after the per-task step | 0.046 | 0.057 | 0.033 | -0.011 [-0.019, -0.002] |
| Brier after the per-task step | 0.671 | 0.663 | 0.644 | +0.008 [+0.003, +0.013] |

### measuring_hate_speech noul

3,284 of 3,384 rows (those every method answered, less the per-task step's 100 and their siblings), 3,284 questions. Per-task scales: candidate score 1.188; P0 score 0.811; teacher score 0.739.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| AUROC | 0.852 | 0.855 | 0.845 | -0.003 [-0.013, +0.007] |
| accuracy | 0.785 | 0.683 | 0.694 | +0.102 [+0.090, +0.115] |
| ECE | 0.047 | 0.195 | 0.197 | -0.149 [-0.160, -0.132] |
| Brier | 0.154 | 0.209 | 0.214 | -0.056 [-0.062, -0.050] |
| Brier raw | 0.147 | 0.188 | 0.206 | -0.042 [-0.052, -0.031] |
| ECE after the per-task step | 0.047 | 0.195 | 0.197 | -0.149 [-0.160, -0.132] |
| Brier after the per-task step | 0.154 | 0.209 | 0.214 | -0.056 [-0.062, -0.050] |

### cuad noul

2,453 of 2,553 rows (those every method answered, less the per-task step's 100 and their siblings), 24,530 questions. Per-task scales: candidate none; P0 none; teacher none.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| AUROC | 0.947 | 0.945 | 0.954 | +0.002 [-0.010, +0.014] |
| accuracy | 0.970 | 0.955 | 0.966 | +0.016 [+0.013, +0.018] |
| ECE | 0.024 | 0.019 | 0.012 | +0.005 [+0.002, +0.007] |
| Brier | 0.023 | 0.030 | 0.025 | -0.007 [-0.008, -0.006] |
| Brier raw | 0.023 | 0.029 | 0.035 | -0.006 [-0.007, -0.004] |
| ECE after the per-task step | 0.024 | 0.019 | 0.012 | +0.005 [+0.002, +0.007] |
| Brier after the per-task step | 0.023 | 0.030 | 0.025 | -0.007 [-0.008, -0.006] |

### massive choice

28,769 of 29,739 rows (those every method answered, less the per-task step's 100 and their siblings), 28,769 questions. Per-task scales: candidate choice 1.288; P0 choice 0.999; teacher choice 0.976.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.672 | 0.658 | 0.731 | +0.014 [+0.010, +0.019] |
| ECE | 0.094 | 0.035 | 0.041 | +0.059 [+0.052, +0.065] |
| Brier | 0.460 | 0.476 | 0.391 | -0.016 [-0.021, -0.011] |
| Brier raw | 0.492 | 0.501 | 0.419 | -0.009 [-0.015, -0.003] |
| ECE after the per-task step | 0.018 | 0.035 | 0.033 | -0.016 [-0.021, -0.009] |
| Brier after the per-task step | 0.447 | 0.476 | 0.390 | -0.029 [-0.033, -0.025] |
