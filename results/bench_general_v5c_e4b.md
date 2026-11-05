# general_v5c_e4b_r8_seed0 on the seven public benchmarks, scored the way the suite is

Development report, not a release gate: `scripts/score_suite.py --sets bench_*`, for the per-task step's ECE (100 rows) on each benchmark, as the model card reports it. The criteria's pass/fail columns are printed by the script but apply only to the frozen suite. The same run for v3d gives at most 0.091 after the step, matching its model card.

Sets: bench_agnews, bench_sst2, bench_banking77, bench_emotion, bench_tweet_topic, bench_fin_topic, bench_daily_dialog. Candidate minus P0, paired 95% CIs from 1,000 item-level bootstrap resamples (rows). Baselines calibrated on the gold_calib rows of general_v5c.jsonl:
- P0 (gemma4_e4b), 80,996 questions: `{"temperatures": {"choice:*": 2.298312742369242, "score:*": 6.817174091832383, "choice:64-255": 1.4812322573882764, "score:3-5": 6.817174091832383, "noul": 5.243362903594971, "choice:16-63": 1.6898920014441763, "choice:3-5": 6.175577027545438, "choice:2": 2.298312742369242, "choice:6-15": 1.7852835191771719}, "b_noul": -10.294469833374023}`
- teacher (gemma4_26b_a4b), 80,996 questions: `{"temperatures": {"choice:*": 2.862873158208299, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.8014187812805176, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.776196001310275, "choice:2": 2.862873158208299, "choice:6-15": 2.4819708042242072}, "b_noul": -9.985109329223633}`

## Criteria 1–3 on each task

| Task | Criterion | Value | 0.x | 1.0 |
|---|---|---|---|---|
| bench_agnews choice | 2 calibration | ECE after the per-task step 0.058, CI [0.042, 0.099] (without it 0.059) | pass | fail |
| bench_agnews choice | 2 calibration | Brier after the per-task step 0.2456 vs 0.2461 before calibration, Δ CI [-0.0050, +0.0042] | pass | pass |
| bench_agnews choice | 3 accuracy | accuracy: Δ CI [+0.017, +0.085] | pass | pass |
| bench_sst2 choice | 2 calibration | ECE after the per-task step 0.025, CI [0.019, 0.054] (without it 0.114) | pass | pass |
| bench_sst2 choice | 2 calibration | Brier after the per-task step 0.1197 vs 0.1387 before calibration, Δ CI [-0.0315, -0.0054] | pass | pass |
| bench_sst2 choice | 3 accuracy | accuracy: Δ CI [-0.028, +0.018] | pass | fail |
| bench_banking77 choice | 2 calibration | ECE after the per-task step 0.058, CI [0.042, 0.093] (without it 0.277) | pass | fail |
| bench_banking77 choice | 2 calibration | Brier after the per-task step 0.2240 vs 0.2570 before calibration, Δ CI [-0.0505, -0.0159] | pass | pass |
| bench_banking77 choice | 3 accuracy | accuracy: Δ CI [+0.078, +0.160] | pass | pass |
| bench_emotion choice | 2 calibration | ECE after the per-task step 0.048, CI [0.039, 0.075] (without it 0.090) | pass | fail |
| bench_emotion choice | 2 calibration | Brier after the per-task step 0.5532 vs 0.5585 before calibration, Δ CI [-0.0086, -0.0022] | pass | pass |
| bench_emotion choice | 3 accuracy | accuracy: Δ CI [+0.007, +0.043] | pass | pass |
| bench_tweet_topic choice | 2 calibration | ECE after the per-task step 0.040, CI [0.032, 0.063] (without it 0.151) | pass | pass |
| bench_tweet_topic choice | 2 calibration | Brier after the per-task step 0.3297 vs 0.3559 before calibration, Δ CI [-0.0347, -0.0173] | pass | pass |
| bench_tweet_topic choice | 3 accuracy | accuracy: Δ CI [-0.043, -0.002] | **fail** | fail |
| bench_fin_topic choice | 2 calibration | ECE after the per-task step 0.059, CI [0.050, 0.076] (without it 0.263) | pass | fail |
| bench_fin_topic choice | 2 calibration | Brier after the per-task step 0.5471 vs 0.5660 before calibration, Δ CI [-0.0256, -0.0127] | pass | pass |
| bench_fin_topic choice | 3 accuracy | accuracy: Δ CI [-0.030, -0.004] | **fail** | fail |
| bench_daily_dialog choice | 2 calibration | ECE after the per-task step 0.083, CI [0.076, 0.095] (without it 0.136) | pass | fail |
| bench_daily_dialog choice | 2 calibration | Brier after the per-task step 0.5270 vs 0.5355 before calibration, Δ CI [-0.0128, -0.0037] | pass | pass |
| bench_daily_dialog choice | 3 accuracy | accuracy: Δ CI [-0.045, -0.027] | **fail** | fail |

1.0 also needs the primary metric better than P0 on most tasks: 3 of 7 (fail).

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

### bench_agnews choice

400 of 500 rows (those every method answered, less the per-task step's 100 and their siblings), 400 questions. Per-task scales: candidate choice 0.774; P0 choice 0.465; teacher choice 0.439.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.853 | 0.802 | 0.853 | +0.050 [+0.017, +0.085] |
| ECE | 0.059 | 0.232 | 0.270 | -0.173 [-0.201, -0.117] |
| Brier | 0.248 | 0.371 | 0.335 | -0.124 [-0.149, -0.097] |
| Brier raw | 0.246 | 0.331 | 0.255 | -0.085 [-0.128, -0.043] |
| ECE after the per-task step | 0.058 | 0.072 | 0.040 | -0.014 [-0.049, +0.024] |
| Brier after the per-task step | 0.246 | 0.294 | 0.235 | -0.048 [-0.080, -0.017] |

### bench_sst2 choice

400 of 500 rows (those every method answered, less the per-task step's 100 and their siblings), 400 questions. Per-task scales: candidate choice 0.365; P0 choice 0.802; teacher choice 1.034.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.920 | 0.925 | 0.958 | -0.005 [-0.028, +0.018] |
| ECE | 0.114 | 0.024 | 0.017 | +0.090 [+0.049, +0.115] |
| Brier | 0.149 | 0.102 | 0.072 | +0.047 [+0.023, +0.070] |
| Brier raw | 0.139 | 0.113 | 0.072 | +0.026 [-0.005, +0.054] |
| ECE after the per-task step | 0.025 | 0.036 | 0.016 | -0.011 [-0.027, +0.013] |
| Brier after the per-task step | 0.120 | 0.104 | 0.072 | +0.015 [-0.011, +0.041] |

### bench_banking77 choice

400 of 500 rows (those every method answered, less the per-task step's 100 and their siblings), 400 questions. Per-task scales: candidate choice 0.434; P0 choice 1.274; teacher choice 1.202.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.845 | 0.728 | 0.767 | +0.117 [+0.078, +0.160] |
| ECE | 0.277 | 0.067 | 0.070 | +0.210 [+0.146, +0.239] |
| Brier | 0.323 | 0.411 | 0.352 | -0.088 [-0.130, -0.051] |
| Brier raw | 0.257 | 0.428 | 0.376 | -0.171 [-0.219, -0.126] |
| ECE after the per-task step | 0.058 | 0.065 | 0.075 | -0.007 [-0.064, +0.021] |
| Brier after the per-task step | 0.224 | 0.413 | 0.350 | -0.189 [-0.233, -0.146] |

### bench_emotion choice

1,900 of 2,000 rows (those every method answered, less the per-task step's 100 and their siblings), 1,900 questions. Per-task scales: candidate choice 0.793; P0 choice 1.823; teacher choice 1.447.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.604 | 0.579 | 0.598 | +0.025 [+0.007, +0.043] |
| ECE | 0.090 | 0.228 | 0.135 | -0.137 [-0.173, -0.102] |
| Brier | 0.561 | 0.641 | 0.589 | -0.080 [-0.099, -0.059] |
| Brier raw | 0.559 | 0.716 | 0.593 | -0.158 [-0.182, -0.132] |
| ECE after the per-task step | 0.048 | 0.043 | 0.155 | +0.005 [-0.016, +0.032] |
| Brier after the per-task step | 0.553 | 0.571 | 0.601 | -0.018 [-0.030, -0.006] |

### bench_tweet_topic choice

1,593 of 1,693 rows (those every method answered, less the per-task step's 100 and their siblings), 1,593 questions. Per-task scales: candidate choice 0.499; P0 choice 0.890; teacher choice 0.900.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.756 | 0.779 | 0.829 | -0.023 [-0.043, -0.002] |
| ECE | 0.151 | 0.031 | 0.066 | +0.120 [+0.084, +0.133] |
| Brier | 0.361 | 0.322 | 0.271 | +0.039 [+0.021, +0.056] |
| Brier raw | 0.356 | 0.341 | 0.273 | +0.014 [-0.011, +0.037] |
| ECE after the per-task step | 0.040 | 0.043 | 0.068 | -0.004 [-0.020, +0.016] |
| Brier after the per-task step | 0.330 | 0.323 | 0.268 | +0.007 [-0.015, +0.026] |

### bench_fin_topic choice

4,017 of 4,117 rows (those every method answered, less the per-task step's 100 and their siblings), 4,017 questions. Per-task scales: candidate choice 0.444; P0 choice 1.175; teacher choice 1.413.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.586 | 0.605 | 0.648 | -0.019 [-0.030, -0.004] |
| ECE | 0.263 | 0.092 | 0.141 | +0.171 [+0.144, +0.192] |
| Brier | 0.634 | 0.555 | 0.534 | +0.080 [+0.064, +0.093] |
| Brier raw | 0.566 | 0.617 | 0.587 | -0.051 [-0.068, -0.034] |
| ECE after the per-task step | 0.059 | 0.042 | 0.035 | +0.017 [+0.002, +0.030] |
| Brier after the per-task step | 0.547 | 0.544 | 0.508 | +0.003 [-0.010, +0.013] |

### bench_daily_dialog choice

7,640 of 7,740 rows (those every method answered, less the per-task step's 100 and their siblings), 7,640 questions. Per-task scales: candidate choice 0.587; P0 choice 1.188; teacher choice 0.900.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.645 | 0.680 | 0.664 | -0.035 [-0.045, -0.027] |
| ECE | 0.136 | 0.125 | 0.106 | +0.011 [-0.006, +0.026] |
| Brier | 0.540 | 0.477 | 0.488 | +0.063 [+0.052, +0.074] |
| Brier raw | 0.536 | 0.530 | 0.505 | +0.006 [-0.007, +0.020] |
| ECE after the per-task step | 0.083 | 0.077 | 0.097 | +0.006 [-0.002, +0.017] |
| Brier after the per-task step | 0.527 | 0.464 | 0.486 | +0.063 [+0.053, +0.074] |
