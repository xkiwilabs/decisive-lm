# general_v5c_26b-a4b_r8_seed0 on the seven public benchmarks, scored the way the suite is

Development report, not a release gate: `scripts/score_suite.py --sets bench_*` (as for `results/bench_general_v5c_e4b.md`). P0 is Gemma 4 26B A4B prompted, which is also the teacher. The criteria's pass/fail columns are printed by the script but apply only to the frozen suite.


Sets: bench_agnews, bench_sst2, bench_banking77, bench_emotion, bench_tweet_topic, bench_fin_topic, bench_daily_dialog. Candidate minus P0, paired 95% CIs from 1,000 item-level bootstrap resamples (rows). Baselines calibrated on the gold_calib rows of general_v5c.jsonl:
- P0 (gemma4_26b_a4b), 80,996 questions: `{"temperatures": {"choice:*": 2.862873158208299, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.8014187812805176, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.776196001310275, "choice:2": 2.862873158208299, "choice:6-15": 2.4819708042242072}, "b_noul": -9.985109329223633}`
- teacher (gemma4_26b_a4b), 80,996 questions: `{"temperatures": {"choice:*": 2.862873158208299, "score:*": 8.971098230482383, "choice:64-255": 1.865459360130255, "score:3-5": 8.971098230482383, "noul": 3.8014187812805176, "choice:16-63": 2.2238227951411003, "choice:3-5": 8.776196001310275, "choice:2": 2.862873158208299, "choice:6-15": 2.4819708042242072}, "b_noul": -9.985109329223633}`

## Criteria 1–3 on each task

| Task | Criterion | Value | 0.x | 1.0 |
|---|---|---|---|---|
| bench_agnews choice | 2 calibration | ECE after the per-task step 0.037, CI [0.031, 0.080] (without it 0.056) | pass | fail |
| bench_agnews choice | 2 calibration | Brier after the per-task step 0.2329 vs 0.2454 before calibration, Δ CI [-0.0218, -0.0024] | pass | pass |
| bench_agnews choice | 3 accuracy | accuracy: Δ CI [-0.022, +0.015] | pass | fail |
| bench_sst2 choice | 2 calibration | ECE after the per-task step 0.047, CI [0.031, 0.072] (without it 0.083) | pass | fail |
| bench_sst2 choice | 2 calibration | Brier after the per-task step 0.1069 vs 0.1120 before calibration, Δ CI [-0.0194, +0.0093] | pass | pass |
| bench_sst2 choice | 3 accuracy | accuracy: Δ CI [-0.055, -0.008] | **fail** | fail |
| bench_banking77 choice | 2 calibration | ECE after the per-task step 0.047, CI [0.040, 0.087] (without it 0.298) | pass | fail |
| bench_banking77 choice | 2 calibration | Brier after the per-task step 0.2599 vs 0.2989 before calibration, Δ CI [-0.0582, -0.0192] | pass | pass |
| bench_banking77 choice | 3 accuracy | accuracy: Δ CI [+0.013, +0.090] | pass | pass |
| bench_emotion choice | 2 calibration | ECE after the per-task step 0.076, CI [0.060, 0.098] (without it 0.148) | pass | fail |
| bench_emotion choice | 2 calibration | Brier after the per-task step 0.5567 vs 0.5714 before calibration, Δ CI [-0.0194, -0.0098] | pass | pass |
| bench_emotion choice | 3 accuracy | accuracy: Δ CI [-0.013, +0.025] | pass | fail |
| bench_tweet_topic choice | 2 calibration | ECE after the per-task step 0.047, CI [0.037, 0.067] (without it 0.216) | pass | pass |
| bench_tweet_topic choice | 2 calibration | Brier after the per-task step 0.2703 vs 0.3188 before calibration, Δ CI [-0.0599, -0.0372] | pass | pass |
| bench_tweet_topic choice | 3 accuracy | accuracy: Δ CI [-0.038, -0.002] | **fail** | fail |
| bench_fin_topic choice | 2 calibration | ECE after the per-task step 0.085, CI [0.076, 0.100] (without it 0.239) | pass | fail |
| bench_fin_topic choice | 2 calibration | Brier after the per-task step 0.5404 vs 0.5597 before calibration, Δ CI [-0.0247, -0.0138] | pass | pass |
| bench_fin_topic choice | 3 accuracy | accuracy: Δ CI [-0.036, -0.012] | **fail** | fail |
| bench_daily_dialog choice | 2 calibration | ECE after the per-task step 0.086, CI [0.080, 0.098] (without it 0.151) | pass | fail |
| bench_daily_dialog choice | 2 calibration | Brier after the per-task step 0.5341 vs 0.5403 before calibration, Δ CI [-0.0113, -0.0007] | pass | pass |
| bench_daily_dialog choice | 3 accuracy | accuracy: Δ CI [-0.045, -0.026] | **fail** | fail |

1.0 also needs the primary metric better than P0 on most tasks: 1 of 7 (fail).

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

### bench_agnews choice

400 of 500 rows (those every method answered, less the per-task step's 100 and their siblings), 400 questions. Per-task scales: candidate choice 0.757; P0 choice 0.439; teacher choice 0.439.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.850 | 0.853 | 0.853 | -0.003 [-0.022, +0.015] |
| ECE | 0.056 | 0.270 | 0.270 | -0.214 [-0.228, -0.177] |
| Brier | 0.237 | 0.335 | 0.335 | -0.098 [-0.115, -0.079] |
| Brier raw | 0.245 | 0.255 | 0.255 | -0.010 [-0.041, +0.021] |
| ECE after the per-task step | 0.037 | 0.040 | 0.040 | -0.003 [-0.021, +0.033] |
| Brier after the per-task step | 0.233 | 0.235 | 0.235 | -0.002 [-0.020, +0.017] |

### bench_sst2 choice

400 of 500 rows (those every method answered, less the per-task step's 100 and their siblings), 400 questions. Per-task scales: candidate choice 0.325; P0 choice 1.034; teacher choice 1.034.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.927 | 0.958 | 0.958 | -0.030 [-0.055, -0.008] |
| ECE | 0.083 | 0.017 | 0.017 | +0.066 [+0.041, +0.093] |
| Brier | 0.115 | 0.072 | 0.072 | +0.043 [+0.021, +0.065] |
| Brier raw | 0.112 | 0.072 | 0.072 | +0.040 [+0.016, +0.065] |
| ECE after the per-task step | 0.047 | 0.016 | 0.016 | +0.030 [+0.010, +0.047] |
| Brier after the per-task step | 0.107 | 0.072 | 0.072 | +0.035 [+0.009, +0.064] |

### bench_banking77 choice

400 of 500 rows (those every method answered, less the per-task step's 100 and their siblings), 400 questions. Per-task scales: candidate choice 0.455; P0 choice 1.202; teacher choice 1.202.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.818 | 0.767 | 0.767 | +0.050 [+0.013, +0.090] |
| ECE | 0.298 | 0.070 | 0.070 | +0.228 [+0.168, +0.262] |
| Brier | 0.368 | 0.352 | 0.352 | +0.016 [-0.027, +0.057] |
| Brier raw | 0.299 | 0.376 | 0.376 | -0.077 [-0.128, -0.031] |
| ECE after the per-task step | 0.047 | 0.075 | 0.075 | -0.028 [-0.066, +0.014] |
| Brier after the per-task step | 0.260 | 0.350 | 0.350 | -0.090 [-0.134, -0.050] |

### bench_emotion choice

1,900 of 2,000 rows (those every method answered, less the per-task step's 100 and their siblings), 1,900 questions. Per-task scales: candidate choice 0.674; P0 choice 1.447; teacher choice 1.447.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.605 | 0.598 | 0.598 | +0.007 [-0.013, +0.025] |
| ECE | 0.148 | 0.135 | 0.135 | +0.013 [-0.016, +0.040] |
| Brier | 0.580 | 0.589 | 0.589 | -0.009 [-0.022, +0.004] |
| Brier raw | 0.571 | 0.593 | 0.593 | -0.022 [-0.040, -0.003] |
| ECE after the per-task step | 0.076 | 0.155 | 0.155 | -0.080 [-0.103, -0.054] |
| Brier after the per-task step | 0.557 | 0.601 | 0.601 | -0.045 [-0.058, -0.031] |

### bench_tweet_topic choice

1,593 of 1,693 rows (those every method answered, less the per-task step's 100 and their siblings), 1,593 questions. Per-task scales: candidate choice 0.349; P0 choice 0.900; teacher choice 0.900.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.809 | 0.829 | 0.829 | -0.020 [-0.038, -0.002] |
| ECE | 0.216 | 0.066 | 0.066 | +0.150 [+0.124, +0.169] |
| Brier | 0.336 | 0.271 | 0.271 | +0.065 [+0.050, +0.080] |
| Brier raw | 0.319 | 0.273 | 0.273 | +0.046 [+0.026, +0.066] |
| ECE after the per-task step | 0.047 | 0.068 | 0.068 | -0.020 [-0.038, +0.003] |
| Brier after the per-task step | 0.270 | 0.268 | 0.268 | +0.002 [-0.015, +0.020] |

### bench_fin_topic choice

4,017 of 4,117 rows (those every method answered, less the per-task step's 100 and their siblings), 4,017 questions. Per-task scales: candidate choice 0.547; P0 choice 1.413; teacher choice 1.413.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.624 | 0.648 | 0.648 | -0.024 [-0.036, -0.012] |
| ECE | 0.239 | 0.141 | 0.141 | +0.098 [+0.071, +0.122] |
| Brier | 0.601 | 0.534 | 0.534 | +0.067 [+0.051, +0.082] |
| Brier raw | 0.560 | 0.587 | 0.587 | -0.027 [-0.044, -0.009] |
| ECE after the per-task step | 0.085 | 0.035 | 0.035 | +0.050 [+0.032, +0.069] |
| Brier after the per-task step | 0.540 | 0.508 | 0.508 | +0.032 [+0.022, +0.042] |

### bench_daily_dialog choice

7,640 of 7,740 rows (those every method answered, less the per-task step's 100 and their siblings), 7,640 questions. Per-task scales: candidate choice 0.460; P0 choice 0.900; teacher choice 0.900.

| Metric | Candidate | P0 | Teacher | Δ candidate − P0 [95% CI] |
|---|---|---|---|---|
| accuracy | 0.629 | 0.664 | 0.664 | -0.036 [-0.045, -0.026] |
| ECE | 0.151 | 0.106 | 0.106 | +0.046 [+0.032, +0.059] |
| Brier | 0.550 | 0.488 | 0.488 | +0.062 [+0.053, +0.071] |
| Brier raw | 0.540 | 0.505 | 0.505 | +0.035 [+0.024, +0.047] |
| ECE after the per-task step | 0.086 | 0.097 | 0.097 | -0.011 [-0.022, +0.003] |
| Brier after the per-task step | 0.534 | 0.486 | 0.486 | +0.048 [+0.038, +0.059] |
