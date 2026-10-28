# general_v3c_e4b_r8_seed0_nt against the release criteria

Sets: helpsteer3, measuring_hate_speech, cuad, massive. Candidate minus P0, paired 95% CIs from 1,000 item-level bootstrap resamples (rows). Baselines calibrated on the gold_calib rows of general_v3c.jsonl, Choice and Score temperatures scaled on multinli gold_calib (new-task scale):
- P0 (gemma4_e4b), 79,198 questions: `{"temperatures": {"choice:*": 0.9019319874885592, "score:*": 2.698465018988863, "choice:64-255": 0.614095721395875, "score:3-5": 2.698465018988863, "noul": 4.709537506103516, "choice:16-63": 0.7006027869915291, "choice:3-5": 2.5323329395952743, "choice:2": 0.9019319874885592, "choice:6-15": 0.7401506179310061}, "b_noul": -10.05241870880127}`
- teacher (gemma4_26b_a4b), 79,198 questions: `{"temperatures": {"choice:*": 1.112520636178803, "score:*": 4.424206675803203, "choice:64-255": 0.7914964178453141, "score:3-5": 4.424206675803203, "noul": 3.734797477722168, "choice:16-63": 0.9435465676154067, "choice:3-5": 3.4481165082770744, "choice:2": 1.112520636178803, "choice:6-15": 1.0530762785435033}, "b_noul": -10.610552787780762}`

## Criteria 1–3 on each task

| Task | Criterion | Value | 0.x | 1.0 |
|---|---|---|---|---|
| helpsteer3 score | 1 review | ρ(top p, agreement): Δ CI [-0.005, +0.047] | pass | fail |
| helpsteer3 score | 1 review | tie AUROC: Δ CI [-0.046, +0.013] | pass | fail |
| helpsteer3 score | 1 review | lowest within-category ρ -0.057 over 6 categories | **fail** | fail |
| helpsteer3 score | 1 review | dynamic range 0.014 | **fail** | fail |
| helpsteer3 score | 1 review | errors caught in review: Δ CI [+0.004, +0.021] | pass | pass |
| helpsteer3 score | 1 review | accuracy gained by review: CI [+0.158, +0.168] | pass | pass |
| helpsteer3 score | 2 calibration | ECE 0.041, CI [0.032, 0.054] | pass | pass |
| helpsteer3 score | 2 calibration | Brier 0.7822 vs 0.7820 before calibration | **fail** | fail |
| helpsteer3 score | 3 accuracy | MAE: Δ CI [-0.254, -0.223] | pass | pass |
| measuring_hate_speech score | 1 review | ρ(top p, agreement): Δ CI [-0.084, -0.059] | **fail** | fail |
| measuring_hate_speech score | 1 review | tie AUROC: Δ CI [-0.035, -0.017] | **fail** | fail |
| measuring_hate_speech score | 1 review | lowest within-category ρ -0.151 over 44 categories | **fail** | fail |
| measuring_hate_speech score | 1 review | dynamic range 0.100 | **fail** | fail |
| measuring_hate_speech score | 1 review | errors caught in review: Δ CI [-0.008, +0.005] | pass | fail |
| measuring_hate_speech score | 1 review | accuracy gained by review: CI [+0.129, +0.135] | pass | pass |
| measuring_hate_speech score | 2 calibration | ECE 0.058, CI [0.053, 0.065] | pass | pass |
| measuring_hate_speech score | 2 calibration | Brier 0.6686 vs 0.6670 before calibration | **fail** | fail |
| measuring_hate_speech score | 3 accuracy | MAE: Δ CI [-0.063, -0.044] | pass | pass |
| measuring_hate_speech noul | 2 calibration | ECE 0.033, CI [0.026, 0.048] | pass | pass |
| measuring_hate_speech noul | 2 calibration | Brier 0.1519 vs 0.1528 before calibration | pass | pass |
| measuring_hate_speech noul | 3 accuracy | AUROC: Δ CI [-0.010, +0.010] | pass | fail |
| cuad noul | 2 calibration | ECE 0.013, CI [0.011, 0.015] | pass | pass |
| cuad noul | 2 calibration | Brier 0.0237 vs 0.0236 before calibration | **fail** | fail |
| cuad noul | 3 accuracy | AUROC: Δ CI [-0.017, +0.006] | pass | fail |
| massive choice | 2 calibration | ECE 0.221, CI [0.217, 0.226] | **fail** | fail |
| massive choice | 2 calibration | Brier 0.5403 vs 0.4997 before calibration | **fail** | fail |
| massive choice | 3 accuracy | accuracy: Δ CI [+0.004, +0.014] | pass | pass |

1.0 also needs the primary metric better than P0 on most tasks: 3 of 5 (pass).

## Criterion 4: close to the specialists on the seeds' own gold_test

Candidate against the reference in `configs/release_seeds.yaml`, paired 95% CIs from 1,000 row resamples. Accuracy margins: {'choice': 0.03, 'noul': 0.03, 'score': 0.05}; ECE ≤ 0.1 or not significantly above the reference's; review readiness not significantly below the reference. *Within noise* (the target, not a gate): the primary metric's Δ CI reaches 0.

| Seed | Type | rows | Criterion | Value | Passes |
|---|---|---|---|---|---|
| banking77 | choice | 1,819 | accuracy | accuracy 0.915 vs 0.902: Δ +0.013 [-0.000, +0.027], margin 0.03 | yes |
| banking77 | choice | 1,819 | calibration | ECE 0.040 vs 0.016: Δ CI [+0.008, +0.033] | yes |
| clinc150 | choice | 3,345 | accuracy | accuracy 0.948 vs 0.920: Δ +0.028 [+0.019, +0.036], margin 0.03 | yes |
| clinc150 | choice | 3,345 | calibration | ECE 0.032 vs 0.014: Δ CI [+0.006, +0.026] | yes |
| helpsteer2 | score | 658 | accuracy | MAE 0.464 vs 0.468: Δ -0.003 [-0.014, +0.007], margin 0.05 | yes |
| helpsteer2 | score | 658 | calibration | ECE 0.101 vs 0.105: Δ CI [-0.013, +0.006] | yes |
| helpsteer2 | score | 658 | review | ρ(top p, agreement) 0.268 vs 0.263: Δ CI [-0.018, +0.029] | yes |
| helpsteer2 | score | 658 | review | tie AUROC 0.626 vs 0.613: Δ CI [-0.006, +0.033] | yes |
| helpsteer2 | score | 658 | review | errors caught in review 0.338 vs 0.330: Δ CI [-0.021, +0.032] | yes |
| civil_comments | noul | 5,974 | accuracy | AUROC 0.980 vs 0.982: Δ -0.003 [-0.005, -0.000], margin 0.03 | yes |
| civil_comments | noul | 5,974 | calibration | ECE 0.016 vs 0.018: Δ CI [-0.003, -0.001] | yes |
| goemotions | choice | 1,500 | accuracy | accuracy 0.739 vs 0.760: Δ -0.021 [-0.035, -0.007], margin 0.03 | yes |
| goemotions | choice | 1,500 | calibration | ECE 0.033 vs 0.108: Δ CI [-0.096, -0.037] | yes |
| goemotions | choice | 1,500 | review | ρ(top p, agreement) 0.414 vs 0.420: Δ CI [-0.026, +0.015] | yes |
| goemotions | choice | 1,500 | review | tie AUROC 0.684 vs 0.675: Δ CI [-0.021, +0.037] | yes |
| goemotions | choice | 1,500 | review | errors caught in review 0.386 vs 0.408: Δ CI [-0.062, +0.024] | yes |
| goemotions | noul | 1,500 | accuracy | AUROC 0.951 vs 0.952: Δ -0.001 [-0.003, +0.002], margin 0.03 | yes |
| goemotions | noul | 1,500 | calibration | ECE 0.015 vs 0.014: Δ CI [+0.001, +0.002] | yes |
| ledgar | choice | 1,000 | accuracy | accuracy 0.838 vs 0.821: Δ +0.017 [+0.002, +0.031], margin 0.03 | yes |
| ledgar | choice | 1,000 | calibration | ECE 0.051 vs 0.032: Δ CI [-0.008, +0.040] | yes |

Within noise of the reference on the primary metric: 5 of 7 seed tasks.

## Criteria 5–7

Checked by their own scripts on the released bundle and snapshot:
- 5 contract: `scripts/check_bundle.py` (conformance) and `scripts/check_invariance.py` (independence, order) on the real backbone;
- 6 provenance: the mix manifest (`data/mixes/<mix>.manifest.json`, R tier only), the model-origin rule, the clean-room canary and scanner (`uv run pytest`);
- 7 snapshot audit: `scripts/export_release.py`.

**Criteria 1–4: fail for 0.x, fail for 1.0.**

## Per-task metrics

### helpsteer3 score

5,966 of 5,966 rows (those every method answered), 5,966 questions; review readiness on items with ≥ 3 raters.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.268 | 0.197 | 0.293 | +0.071 [+0.058, +0.083] |
| ECE | 0.041 | 0.108 | 0.099 | -0.067 [-0.077, -0.052] |
| Brier | 0.782 | 0.868 | 0.841 | -0.086 [-0.091, -0.081] |
| Brier raw | 0.782 | 0.999 | 1.051 | -0.217 [-0.226, -0.208] |
| MAE | 1.020 | 1.259 | 1.077 | -0.239 [-0.254, -0.223] |
| ρ(top p, agreement) | 0.056 | 0.036 | 0.083 | +0.020 [-0.005, +0.047] |
| tie AUROC | 0.496 | 0.512 | 0.516 | -0.016 [-0.046, +0.013] |
| errors caught in review | 0.223 | 0.211 | 0.217 | +0.012 [+0.004, +0.021] |
| accuracy gained by review | 0.163 | 0.170 | 0.153 | -0.006 [-0.013, +0.001] |
| lowest within-category ρ | -0.057 | -0.101 | -0.009 |  |
| categories (≥ 25 items) | 6.000 | 6.000 | 6.000 |  |
| dynamic range | 0.014 | 0.012 | 0.025 |  |

### measuring_hate_speech score

3,384 of 3,384 rows (those every method answered), 30,456 questions; review readiness on items with ≥ 3 raters.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.436 | 0.448 | 0.502 | -0.012 [-0.019, -0.005] |
| ECE | 0.058 | 0.252 | 0.139 | -0.193 [-0.202, -0.184] |
| Brier | 0.669 | 0.760 | 0.675 | -0.091 [-0.099, -0.083] |
| Brier raw | 0.667 | 0.930 | 0.844 | -0.263 [-0.274, -0.250] |
| MAE | 0.638 | 0.692 | 0.584 | -0.054 [-0.063, -0.044] |
| ρ(top p, agreement) | 0.092 | 0.163 | 0.161 | -0.071 [-0.084, -0.059] |
| tie AUROC | 0.537 | 0.562 | 0.560 | -0.025 [-0.035, -0.017] |
| errors caught in review | 0.238 | 0.240 | 0.270 | -0.001 [-0.008, +0.005] |
| accuracy gained by review | 0.132 | 0.132 | 0.134 | +0.000 [-0.004, +0.004] |
| lowest within-category ρ | -0.151 | -0.287 | -0.160 |  |
| categories (≥ 25 items) | 44.000 | 44.000 | 44.000 |  |
| dynamic range | 0.100 | 0.138 | 0.082 |  |

### measuring_hate_speech noul

3,384 of 3,384 rows (those every method answered), 3,384 questions.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| AUROC | 0.854 | 0.854 | 0.845 | +0.000 [-0.010, +0.010] |
| accuracy | 0.777 | 0.681 | 0.686 | +0.097 [+0.083, +0.109] |
| ECE | 0.033 | 0.209 | 0.218 | -0.177 [-0.186, -0.159] |
| Brier | 0.152 | 0.216 | 0.226 | -0.065 [-0.071, -0.058] |
| Brier raw | 0.153 | 0.189 | 0.206 | -0.036 [-0.047, -0.026] |

### cuad noul

2,553 of 2,553 rows (those every method answered), 25,530 questions.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| AUROC | 0.942 | 0.947 | 0.955 | -0.005 [-0.017, +0.006] |
| accuracy | 0.971 | 0.955 | 0.965 | +0.016 [+0.013, +0.018] |
| ECE | 0.013 | 0.021 | 0.015 | -0.008 [-0.011, -0.005] |
| Brier | 0.024 | 0.030 | 0.026 | -0.007 [-0.008, -0.005] |
| Brier raw | 0.024 | 0.029 | 0.036 | -0.005 [-0.006, -0.004] |

### massive choice

29,739 of 29,739 rows (those every method answered), 29,739 questions.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.666 | 0.657 | 0.729 | +0.009 [+0.004, +0.014] |
| ECE | 0.221 | 0.196 | 0.147 | +0.025 [+0.020, +0.030] |
| Brier | 0.540 | 0.536 | 0.424 | +0.005 [-0.002, +0.012] |
| Brier raw | 0.500 | 0.502 | 0.421 | -0.003 [-0.009, +0.003] |
