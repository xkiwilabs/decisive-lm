# general_v3d_e4b_r8_seed0 against the release criteria

Sets: helpsteer3, measuring_hate_speech, cuad, massive. Candidate minus P0, paired 95% CIs from 1,000 item-level bootstrap resamples (rows). Baselines calibrated on the gold_calib rows of general_v3c.jsonl:
- P0 (gemma4_e4b), 79,198 questions: `{"temperatures": {"choice:*": 2.175508975704365, "score:*": 6.817174091832383, "choice:64-255": 1.4812322573882764, "score:3-5": 6.817174091832383, "noul": 4.709537506103516, "choice:16-63": 1.6898920014441763, "choice:3-5": 6.10812468787312, "choice:2": 2.175508975704365, "choice:6-15": 1.7852835191771719}, "b_noul": -10.05241870880127}`
- teacher (gemma4_26b_a4b), 79,198 questions: `{"temperatures": {"choice:*": 2.622073817778682, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.734797477722168, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.126784998845338, "choice:2": 2.622073817778682, "choice:6-15": 2.4819708042242072}, "b_noul": -10.610552787780762}`

## Criteria 1–3 on each task

| Task | Criterion | Value | 0.x | 1.0 |
|---|---|---|---|---|
| helpsteer3 score | 1 review | ρ(top p, agreement): Δ CI [-0.007, +0.050] | pass | fail |
| helpsteer3 score | 1 review | tie AUROC: Δ CI [-0.036, +0.020] | pass | fail |
| helpsteer3 score | 1 review | lowest within-category ρ -0.048 (P0 -0.107) over 6 categories | pass | fail |
| helpsteer3 score | 1 review | dynamic range 0.013 (P0 0.005) | pass | fail |
| helpsteer3 score | 1 review | errors caught in review: Δ CI [-0.003, +0.016] | pass | fail |
| helpsteer3 score | 1 review | accuracy gained by review: CI [+0.146, +0.157] | pass | pass |
| helpsteer3 score | 2 calibration | ECE after the per-task step 0.029, CI [0.021, 0.042] (without it 0.029) | pass | pass |
| helpsteer3 score | 2 calibration | Brier after the per-task step 0.7753 vs 0.7754 before calibration, Δ CI [-0.0004, +0.0001] | pass | pass |
| helpsteer3 score | 3 accuracy | MAE: Δ CI [-0.299, -0.263] | pass | pass |
| measuring_hate_speech score | 1 review | ρ(top p, agreement): Δ CI [-0.072, -0.047] | **fail** | fail |
| measuring_hate_speech score | 1 review | tie AUROC: Δ CI [-0.031, -0.014] | **fail** | fail |
| measuring_hate_speech score | 1 review | lowest within-category ρ -0.139 (P0 -0.311) over 44 categories | pass | fail |
| measuring_hate_speech score | 1 review | dynamic range 0.092 (P0 0.083) | pass | fail |
| measuring_hate_speech score | 1 review | errors caught in review: Δ CI [-0.020, -0.007] | **fail** | fail |
| measuring_hate_speech score | 1 review | accuracy gained by review: CI [+0.139, +0.145] | pass | pass |
| measuring_hate_speech score | 2 calibration | ECE after the per-task step 0.080, CI [0.074, 0.087] (without it 0.092) | pass | fail |
| measuring_hate_speech score | 2 calibration | Brier after the per-task step 0.6690 vs 0.6707 before calibration, Δ CI [-0.0020, -0.0014] | pass | pass |
| measuring_hate_speech score | 3 accuracy | MAE: Δ CI [-0.044, -0.029] | pass | pass |
| measuring_hate_speech noul | 2 calibration | ECE after the per-task step 0.061, CI [0.052, 0.077] (without it 0.061) | pass | fail |
| measuring_hate_speech noul | 2 calibration | Brier after the per-task step 0.1586 vs 0.1658 before calibration, Δ CI [-0.0080, -0.0065] | pass | pass |
| measuring_hate_speech noul | 3 accuracy | AUROC: Δ CI [-0.000, +0.019] | pass | fail |
| cuad noul | 2 calibration | ECE after the per-task step 0.015, CI [0.014, 0.018] (without it 0.015) | pass | pass |
| cuad noul | 2 calibration | Brier after the per-task step 0.0230 vs 0.0230 before calibration, Δ CI [-0.0001, +0.0001] | pass | pass |
| cuad noul | 3 accuracy | AUROC: Δ CI [-0.005, +0.017] | pass | fail |
| massive choice | 2 calibration | ECE after the per-task step 0.023, CI [0.020, 0.029] (without it 0.115) | pass | pass |
| massive choice | 2 calibration | Brier after the per-task step 0.4413 vs 0.4788 before calibration, Δ CI [-0.0401, -0.0351] | pass | pass |
| massive choice | 3 accuracy | accuracy: Δ CI [+0.014, +0.023] | pass | pass |

1.0 also needs the primary metric better than P0 on most tasks: 3 of 5 (pass).

## Criterion 4: close to the specialists on the seeds' own gold_test

Candidate against the reference in `configs/release_seeds.yaml`, paired 95% CIs from 1,000 row resamples. Accuracy margins: {'choice': 0.03, 'noul': 0.03, 'score': 0.05}; ECE ≤ 0.1 or not significantly above the reference's; review readiness not significantly below the reference. *Within noise* (the target, not a gate): the primary metric's Δ CI reaches 0.

| Seed | Type | rows | Criterion | Value | Passes |
|---|---|---|---|---|---|
| banking77 | choice | 1,819 | accuracy | accuracy 0.892 vs 0.902: Δ -0.010 [-0.024, +0.004], margin 0.03 | yes |
| banking77 | choice | 1,819 | calibration | ECE 0.025 vs 0.016: Δ CI [-0.007, +0.020] | yes |
| clinc150 | choice | 3,345 | accuracy | accuracy 0.947 vs 0.920: Δ +0.028 [+0.020, +0.036], margin 0.03 | yes |
| clinc150 | choice | 3,345 | calibration | ECE 0.013 vs 0.014: Δ CI [-0.011, +0.007] | yes |
| helpsteer2 | score | 658 | accuracy | MAE 0.478 vs 0.468: Δ +0.010 [+0.001, +0.020], margin 0.05 | yes |
| helpsteer2 | score | 658 | calibration | ECE 0.101 vs 0.105: Δ CI [-0.014, +0.005] | yes |
| helpsteer2 | score | 658 | review | ρ(top p, agreement) 0.261 vs 0.263: Δ CI [-0.026, +0.021] | yes |
| helpsteer2 | score | 658 | review | tie AUROC 0.621 vs 0.613: Δ CI [-0.011, +0.028] | yes |
| helpsteer2 | score | 658 | review | errors caught in review 0.338 vs 0.330: Δ CI [-0.020, +0.032] | yes |
| civil_comments | noul | 5,974 | accuracy | AUROC 0.978 vs 0.982: Δ -0.004 [-0.007, -0.002], margin 0.03 | yes |
| civil_comments | noul | 5,974 | calibration | ECE 0.020 vs 0.018: Δ CI [+0.001, +0.003] | yes |
| goemotions | choice | 1,500 | accuracy | accuracy 0.744 vs 0.760: Δ -0.016 [-0.031, -0.001], margin 0.03 | yes |
| goemotions | choice | 1,500 | calibration | ECE 0.099 vs 0.108: Δ CI [-0.026, +0.004] | yes |
| goemotions | choice | 1,500 | review | ρ(top p, agreement) 0.424 vs 0.420: Δ CI [-0.017, +0.025] | yes |
| goemotions | choice | 1,500 | review | tie AUROC 0.690 vs 0.675: Δ CI [-0.012, +0.043] | yes |
| goemotions | choice | 1,500 | review | errors caught in review 0.386 vs 0.408: Δ CI [-0.063, +0.024] | yes |
| goemotions | noul | 1,500 | accuracy | AUROC 0.948 vs 0.952: Δ -0.004 [-0.007, -0.001], margin 0.03 | yes |
| goemotions | noul | 1,500 | calibration | ECE 0.012 vs 0.014: Δ CI [-0.002, -0.000] | yes |
| ledgar | choice | 1,000 | accuracy | accuracy 0.806 vs 0.821: Δ -0.015 [-0.032, +0.000], margin 0.03 | yes |
| ledgar | choice | 1,000 | calibration | ECE 0.043 vs 0.032: Δ CI [-0.011, +0.034] | yes |

Within noise of the reference on the primary metric: 3 of 7 seed tasks.

## Criteria 5–7

Checked by their own scripts on the released bundle and snapshot:
- 5 contract: `scripts/check_bundle.py` (conformance) and `scripts/check_invariance.py` (independence, order) on the real backbone;
- 6 provenance: the mix manifest (`data/mixes/<mix>.manifest.json`, R tier only), the model-origin rule, the clean-room canary and scanner (`uv run pytest`);
- 7 snapshot audit: `scripts/export_release.py`.

**Criteria 1–4: fail for 0.x, fail for 1.0.**

## Per-task metrics

### helpsteer3 score

5,866 of 5,966 rows (those every method answered, less the per-task step's 100 and their siblings), 5,866 questions; review readiness on items with ≥ 3 raters. Per-task scales: candidate score 1.096; P0 score 2.000; teacher score 1.463.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.302 | 0.197 | 0.293 | +0.105 [+0.094, +0.118] |
| ECE | 0.029 | 0.038 | 0.016 | -0.009 [-0.020, +0.008] |
| Brier | 0.776 | 0.833 | 0.807 | -0.057 [-0.061, -0.052] |
| Brier raw | 0.775 | 0.999 | 1.050 | -0.223 [-0.232, -0.213] |
| MAE | 1.010 | 1.292 | 1.158 | -0.282 [-0.299, -0.263] |
| ρ(top p, agreement) | 0.063 | 0.043 | 0.088 | +0.021 [-0.007, +0.050] |
| tie AUROC | 0.508 | 0.518 | 0.511 | -0.009 [-0.036, +0.020] |
| errors caught in review | 0.217 | 0.210 | 0.217 | +0.006 [-0.003, +0.016] |
| accuracy gained by review | 0.151 | 0.169 | 0.153 | -0.018 [-0.025, -0.010] |
| lowest within-category ρ | -0.048 | -0.107 | -0.011 |  |
| categories (≥ 25 items) | 6.000 | 6.000 | 6.000 |  |
| dynamic range | 0.013 | 0.005 | 0.013 |  |
| ECE after the per-task step | 0.029 | 0.038 | 0.052 | -0.009 [-0.022, +0.008] |
| Brier after the per-task step | 0.775 | 0.830 | 0.808 | -0.055 [-0.059, -0.050] |

### measuring_hate_speech score

3,284 of 3,384 rows (those every method answered, less the per-task step's 100 and their siblings), 29,556 questions; review readiness on items with ≥ 3 raters. Per-task scales: candidate score 1.109; P0 score 0.811; teacher score 0.739.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.367 | 0.447 | 0.501 | -0.081 [-0.089, -0.072] |
| ECE | 0.092 | 0.033 | 0.075 | +0.059 [+0.047, +0.066] |
| Brier | 0.672 | 0.662 | 0.653 | +0.010 [+0.006, +0.015] |
| Brier raw | 0.671 | 0.930 | 0.845 | -0.259 [-0.272, -0.247] |
| MAE | 0.620 | 0.657 | 0.703 | -0.037 [-0.044, -0.029] |
| ρ(top p, agreement) | 0.142 | 0.201 | 0.162 | -0.059 [-0.072, -0.047] |
| tie AUROC | 0.556 | 0.578 | 0.562 | -0.022 [-0.031, -0.014] |
| errors caught in review | 0.225 | 0.238 | 0.267 | -0.013 [-0.020, -0.007] |
| accuracy gained by review | 0.142 | 0.131 | 0.132 | +0.011 [+0.007, +0.015] |
| lowest within-category ρ | -0.139 | -0.311 | -0.175 |  |
| categories (≥ 25 items) | 44.000 | 44.000 | 44.000 |  |
| dynamic range | 0.092 | 0.083 | 0.043 |  |
| ECE after the per-task step | 0.080 | 0.057 | 0.033 | +0.024 [+0.014, +0.032] |
| Brier after the per-task step | 0.669 | 0.663 | 0.644 | +0.006 [+0.001, +0.011] |

### measuring_hate_speech noul

3,284 of 3,384 rows (those every method answered, less the per-task step's 100 and their siblings), 3,284 questions. Per-task scales: candidate score 1.109; P0 score 0.811; teacher score 0.739.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| AUROC | 0.865 | 0.855 | 0.845 | +0.010 [-0.000, +0.019] |
| accuracy | 0.772 | 0.683 | 0.688 | +0.089 [+0.078, +0.100] |
| ECE | 0.061 | 0.207 | 0.216 | -0.146 [-0.155, -0.131] |
| Brier | 0.159 | 0.215 | 0.224 | -0.056 [-0.061, -0.051] |
| Brier raw | 0.166 | 0.188 | 0.206 | -0.023 [-0.035, -0.010] |
| ECE after the per-task step | 0.061 | 0.207 | 0.216 | -0.146 [-0.155, -0.131] |
| Brier after the per-task step | 0.159 | 0.215 | 0.224 | -0.056 [-0.061, -0.051] |

### cuad noul

2,453 of 2,553 rows (those every method answered, less the per-task step's 100 and their siblings), 24,530 questions. Per-task scales: candidate none; P0 none; teacher none.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| AUROC | 0.951 | 0.945 | 0.954 | +0.006 [-0.005, +0.017] |
| accuracy | 0.971 | 0.955 | 0.965 | +0.016 [+0.013, +0.018] |
| ECE | 0.015 | 0.021 | 0.015 | -0.006 [-0.009, -0.003] |
| Brier | 0.023 | 0.030 | 0.026 | -0.007 [-0.009, -0.006] |
| Brier raw | 0.023 | 0.029 | 0.035 | -0.006 [-0.007, -0.005] |
| ECE after the per-task step | 0.015 | 0.021 | 0.015 | -0.006 [-0.009, -0.003] |
| Brier after the per-task step | 0.023 | 0.030 | 0.026 | -0.007 [-0.009, -0.006] |

### massive choice

28,769 of 29,739 rows (those every method answered, less the per-task step's 100 and their siblings), 28,769 questions. Per-task scales: candidate choice 1.498; P0 choice 0.999; teacher choice 0.976.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.676 | 0.658 | 0.731 | +0.019 [+0.014, +0.023] |
| ECE | 0.115 | 0.035 | 0.041 | +0.080 [+0.072, +0.086] |
| Brier | 0.462 | 0.476 | 0.391 | -0.014 [-0.019, -0.009] |
| Brier raw | 0.479 | 0.501 | 0.419 | -0.023 [-0.029, -0.017] |
| ECE after the per-task step | 0.023 | 0.035 | 0.033 | -0.012 [-0.017, -0.006] |
| Brier after the per-task step | 0.441 | 0.476 | 0.390 | -0.035 [-0.039, -0.031] |
