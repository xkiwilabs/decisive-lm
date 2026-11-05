# General v5a: the first 1.0 candidate on its image and audio sets, and its text results against v4a

General v5a (`general_v5a_e4b_r8_seed0`, E4B, rank 8) is v4a's text mix plus VizWiz-VQA (10,000 photos), SONYC-UST
(8,000 street recordings) and all of Emozionalmente (`data/queues/general_v5a.sh`). Early stopping ended training at
step 7,500 of 31,178; the bundle scored here is the one selected by validation calibration loss, at step 6,000.
Scored with `scripts/eval_dataset.py` on gold_test, 95% CIs from 1,000 item-level bootstrap resamples.

Image and audio sets: v5a against the zero-shot starting points of `results/media_v1_zero_shot_v3d.md` (the prompted
base model P0 and the text-only head v3d, each with task calibration fitted on the set's gold_calib rows). v5a is
scored without a task step: it trained on these sets' train rows. The frozen 1.0 suites (TextVQA, ODAQ) are not touched.

| Set | Measure | P0+task | v3d+task | v5a | Δ v5a − P0+task [95% CI] |
|---|---|---|---|---|---|
| Emozionalmente (7 emotions, 1,435 clips) | accuracy | 0.199 | 0.200 | 0.265 | +0.066 [+0.038, +0.092] |
| Emozionalmente | ρ(top p, agreement) | +0.047 | −0.001 | +0.023 | |
| SONYC-UST (8 sounds, 412 clips) | mean AUROC over the 8 | 0.585 | 0.667 | 0.862 | each of the 8 higher; 6 significantly |
| VizWiz-VQA (2,567 photos) | answerable AUROC | 0.788 | 0.828 | 0.922 | +0.135 [+0.119, +0.151] |
| VizWiz-VQA, 4-option answers (1,953) | accuracy | 0.432 | 0.434 | 0.699 | +0.267 [+0.236, +0.298] |
| VizWiz-VQA, 4-option answers | ρ(top p, agreement) | +0.292 | +0.332 | +0.599 | |
| VizWiz-VQA, 4-option answers | tie AUROC | 0.588 | 0.574 | 0.695 | |
| VizWiz-VQA, yes/no answers (127) | accuracy | 0.776 | 0.732 | 0.784 | +0.009 [−0.080, +0.092] |

Text: v5a against v4a (v4a's mix without the image and audio rows), both raw, no task step. Δ = v5a − v4a on the
primary metric (accuracy, Noul AUROC; for Score, MAE, where lower is better).

| Set | Question | Primary | v4a | v5a | Δ [95% CI] | |
|---|---|---|---|---|---|---|
| wiki_toxicity (dev, held out) | toxicity, 5 levels | MAE | 0.381 | 0.487 | +0.106 [+0.099, +0.113] | worse |
| humicroedit (dev, held out) | funny, 4 levels | MAE | 0.492 | 0.486 | −0.005 [−0.008, −0.002] | better |
| csc_sarcasm (dev, held out) | sarcastic, 6 levels | MAE | 1.062 | 0.999 | −0.063 [−0.079, −0.047] | better |
| MultiNLI (dev) | relation, 3 options | accuracy | 0.811 | 0.846 | +0.035 [+0.026, +0.043] | better |
| MultiNLI | follows | AUROC | 0.965 | 0.964 | −0.001 [−0.002, +0.001] | same |
| MultiNLI | support, 3 levels | MAE | 0.307 | 0.269 | −0.038 [−0.042, −0.035] | better |
| AG News | label, 4 | accuracy | 0.842 | 0.842 | +0.000 [−0.016, +0.016] | same |
| SST-2 | label, 2 | accuracy | 0.932 | 0.924 | −0.008 [−0.024, +0.006] | same |
| emotion (tweets) | emotion, 6 | accuracy | 0.603 | 0.570 | −0.033 [−0.046, −0.020] | worse |
| TweetTopic | topic, 6 | accuracy | 0.785 | 0.788 | +0.003 [−0.007, +0.014] | same |
| financial topics | topic, 20 | accuracy | 0.576 | 0.591 | +0.015 [+0.008, +0.023] | better |
| DailyDialog | emotion, 7 | accuracy | 0.684 | 0.601 | −0.084 [−0.090, −0.077] | worse |
| Banking77 (trained on) | label, 77 | accuracy | 0.814 | 0.830 | +0.016 [−0.006, +0.042] | same |

What it says:
- **Training on multi-rater audio and photos worked.** City sounds rose from near chance to a mean AUROC of 0.86,
  every class above both starting points (human voice 0.44 → 0.92). The photo answer Choice went from 43% to 70%
  right with four options, and its confidence now follows how far the answerers agreed (ρ 0.60, up from 0.29–0.33).
  That is above one half of the answerers predicting the other half (0.43), which is a comparison, not a ceiling.
- **Speech emotion is still weak.** 27% right with 7 options, up from 20%, but its confidence does not follow
  listener agreement (ρ 0.02). The listeners themselves agree little (split-half ρ 0.26).
- **Text mostly held, with three losses.** Better than v4a on sarcasm, humour, MultiNLI and financial topics; worse
  on wiki_toxicity (the known Score middle-level failure, now larger) and on the two emotion-from-text benchmarks
  (DailyDialog −8 points, tweet emotion −3). The two emotion losses may come from Emozionalmente's emotion labels;
  that is untested.

Per-question tables follow.

## emozionalmente: gold_test

1435 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### emotion (choice, 7 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.990 | 0.199 | 0.125 | 0.014 | 0.852 | 0.562 | 0.210 | 0.208 | 0.047 | 0.477 | [0.175, 0.222] | — | — |
| v3d+task | 0.988 | 0.200 | 0.126 | 0.015 | 0.852 | 0.532 | 0.208 | 0.206 | -0.001 | 0.466 | [0.176, 0.223] | +0.002 | [-0.018, +0.022] |
| v5a | 0.943 | 0.265 | 0.193 | 0.037 | 0.813 | 0.627 | 0.226 | 0.286 | 0.023 | 0.494 | [0.239, 0.290] | +0.066 | [+0.038, +0.092] |

## sonyc_ust: gold_test

412 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### engine (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.537 | 0.752 | 0.187 | 0.050 | 0.589 | 0.102 | [0.479, 0.595] | — | — |
| v3d+task | 0.579 | 0.757 | 0.185 | 0.091 | 0.584 | 0.173 | [0.513, 0.648] | +0.042 | [-0.042, +0.135] |
| v5a | 0.756 | 0.762 | 0.160 | 0.075 | 0.553 | 0.489 | [0.700, 0.806] | +0.219 | [+0.156, +0.287] |

### machinery_impact (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.658 | 0.981 | 0.022 | 0.054 | 0.215 | 0.214 | [0.455, 0.833] | — | — |
| v3d+task | 0.742 | 0.981 | 0.023 | 0.061 | 0.222 | 0.036 | [0.589, 0.887] | +0.085 | [-0.107, +0.274] |
| v5a | 0.837 | 0.981 | 0.018 | 0.005 | 0.219 | 0.309 | [0.753, 0.922] | +0.179 | [-0.012, +0.390] |

### non_machinery_impact (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.470 | 0.971 | 0.029 | 0.025 | 0.204 | 0.004 | [0.296, 0.640] | — | — |
| v3d+task | 0.578 | 0.971 | 0.030 | 0.037 | 0.203 | 0.112 | [0.397, 0.759] | +0.108 | [-0.042, +0.263] |
| v5a | 0.774 | 0.971 | 0.026 | 0.011 | 0.199 | 0.249 | [0.585, 0.940] | +0.304 | [+0.187, +0.434] |

### powered_saw (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.912 | 0.998 | 0.003 | 0.019 | 0.059 | 0.154 | [0.883, 0.939] | — | — |
| v3d+task | 0.599 | 0.998 | 0.004 | 0.025 | 0.067 | -0.029 | [0.549, 0.647] | -0.314 | [-0.374, -0.261] |
| v5a | 0.971 | 0.998 | 0.002 | 0.008 | 0.056 | 0.111 | [0.954, 0.985] | +0.058 | [+0.025, +0.090] |

### alert_signal (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.504 | 0.876 | 0.109 | 0.033 | 0.420 | 0.031 | [0.421, 0.589] | — | — |
| v3d+task | 0.555 | 0.876 | 0.110 | 0.042 | 0.419 | 0.132 | [0.466, 0.643] | +0.050 | [-0.062, +0.171] |
| v5a | 0.803 | 0.886 | 0.094 | 0.052 | 0.419 | 0.324 | [0.728, 0.866] | +0.298 | [+0.189, +0.407] |

### music (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.578 | 0.959 | 0.039 | 0.011 | 0.215 | 0.022 | [0.434, 0.726] | — | — |
| v3d+task | 0.712 | 0.956 | 0.039 | 0.015 | 0.209 | 0.113 | [0.546, 0.847] | +0.134 | [-0.093, +0.366] |
| v5a | 0.919 | 0.959 | 0.036 | 0.021 | 0.196 | 0.308 | [0.861, 0.966] | +0.341 | [+0.200, +0.483] |

### human_voice (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.435 | 0.648 | 0.239 | 0.105 | 0.680 | -0.150 | [0.399, 0.474] | — | — |
| v3d+task | 0.714 | 0.682 | 0.206 | 0.066 | 0.607 | 0.421 | [0.662, 0.763] | +0.278 | [+0.209, +0.344] |
| v5a | 0.924 | 0.799 | 0.145 | 0.067 | 0.487 | 0.786 | [0.896, 0.949] | +0.489 | [+0.439, +0.537] |

### dog (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.587 | 0.966 | 0.033 | 0.025 | 0.217 | 0.148 | [0.425, 0.748] | — | — |
| v3d+task | 0.855 | 0.966 | 0.033 | 0.011 | 0.190 | 0.266 | [0.716, 0.957] | +0.267 | [+0.072, +0.473] |
| v5a | 0.913 | 0.968 | 0.024 | 0.016 | 0.150 | 0.351 | [0.834, 0.974] | +0.326 | [+0.152, +0.499] |

### Mean of the primary metric over questions

| Method | noul |
|---|---|
| P0+task | 0.585 |
| v3d+task | 0.667 |
| v5a | 0.862 |

## vizwiz_vqa: gold_test

2567 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### answerable (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.788 | 0.781 | 0.162 | 0.064 | 0.549 | 0.539 | [0.768, 0.807] | — | — |
| v3d+task | 0.828 | 0.802 | 0.145 | 0.058 | 0.522 | 0.601 | [0.808, 0.847] | +0.041 | [+0.027, +0.053] |
| v5a | 0.922 | 0.830 | 0.112 | 0.045 | 0.440 | 0.809 | [0.912, 0.933] | +0.135 | [+0.119, +0.151] |

### answer (choice, 4 options; asked on 1953 rows)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.948 | 0.432 | 0.432 | 0.061 | 0.678 | 0.638 | 0.222 | 0.461 | 0.292 | 0.588 | [0.406, 0.457] | — | — |
| v3d+task | 0.960 | 0.433 | 0.434 | 0.087 | 0.640 | 0.690 | 0.238 | 0.465 | 0.332 | 0.574 | [0.407, 0.458] | +0.001 | [-0.025, +0.027] |
| v5a | 0.832 | 0.699 | 0.699 | 0.186 | 0.472 | 0.735 | 0.285 | 0.744 | 0.599 | 0.695 | [0.677, 0.723] | +0.267 | [+0.236, +0.298] |

### answer (choice, 2 options; asked on 127 rows)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0+task | 0.879 | 0.776 | 0.776 | 0.196 | 0.400 | 0.713 | 0.308 | 0.804 | 0.296 | 0.590 | [0.692, 0.853] | — | — |
| v3d+task | 0.844 | 0.733 | 0.732 | 0.059 | 0.361 | 0.685 | 0.290 | 0.766 | 0.095 | 0.542 | [0.644, 0.815] | -0.043 | [-0.136, +0.046] |
| v5a | 0.698 | 0.784 | 0.784 | 0.093 | 0.301 | 0.730 | 0.320 | 0.813 | 0.368 | 0.587 | [0.709, 0.857] | +0.009 | [-0.080, +0.092] |

### Mean of the primary metric over questions

| Method | choice | noul |
|---|---|---|
| P0+task | 0.604 | 0.788 |
| v3d+task | 0.583 | 0.828 |
| v5a | 0.742 | 0.922 |

## wiki_toxicity: gold_test

3060 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### toxicity (score, 5 levels)

| Method | soft NLL | MAE | exact | EMD | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | MAE 95% CI | ΔMAE vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.788 | 0.381 | 0.425 | 0.590 | 0.213 | 0.432 | -0.087 | 0.492 | [0.372, 0.390] | — | — |
| v5a | 0.829 | 0.487 | 0.427 | 0.620 | 0.199 | 0.429 | -0.114 | 0.534 | [0.476, 0.499] | +0.106 | [+0.099, +0.113] |

## humicroedit: gold_test

1869 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### funny (score, 4 levels)

| Method | soft NLL | MAE | exact | EMD | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | MAE 95% CI | ΔMAE vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.917 | 0.492 | 0.521 | 0.571 | 0.288 | 0.574 | 0.183 | 0.533 | [0.476, 0.506] | — | — |
| v5a | 0.918 | 0.486 | 0.529 | 0.572 | 0.275 | 0.580 | 0.201 | 0.541 | [0.472, 0.500] | -0.005 | [-0.008, -0.002] |

## csc_sarcasm: gold_test

752 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### sarcastic (score, 6 levels)

| Method | soft NLL | MAE | exact | EMD | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | MAE 95% CI | ΔMAE vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 1.006 | 1.062 | 0.268 | 1.185 | 0.224 | 0.274 | -0.104 | 0.475 | [1.012, 1.109] | — | — |
| v5a | 0.986 | 0.999 | 0.320 | 1.125 | 0.244 | 0.356 | 0.030 | 0.529 | [0.951, 1.043] | -0.063 | [-0.079, -0.047] |

## multinli: gold_test

4823 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### relation (choice, 3 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.571 | 0.811 | 0.810 | 0.136 | 0.297 | 0.801 | 0.476 | 0.877 | 0.315 | 0.727 | [0.800, 0.823] | — | — |
| v5a | 0.527 | 0.846 | 0.843 | 0.139 | 0.257 | 0.810 | 0.523 | 0.909 | 0.375 | 0.761 | [0.836, 0.856] | +0.035 | [+0.026, +0.043] |

### follows (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.965 | 0.899 | 0.083 | 0.090 | 0.346 | 0.818 | [0.960, 0.969] | — | — |
| v5a | 0.964 | 0.900 | 0.075 | 0.068 | 0.328 | 0.815 | [0.960, 0.969] | -0.001 | [-0.002, +0.001] |

### support (score, 3 levels)

| Method | soft NLL | MAE | exact | EMD | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | MAE 95% CI | ΔMAE vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.594 | 0.307 | 0.784 | 0.343 | 0.457 | 0.854 | 0.272 | 0.697 | [0.300, 0.315] | — | — |
| v5a | 0.554 | 0.269 | 0.812 | 0.311 | 0.453 | 0.872 | 0.299 | 0.706 | [0.262, 0.275] | -0.038 | [-0.042, -0.035] |

### Mean of the primary metric over questions

| Method | choice | noul | score |
|---|---|---|---|
| v4a | 0.811 | 0.965 | 0.307 |
| v5a | 0.846 | 0.964 | 0.269 |

## bench_agnews: gold_test

500 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### label (choice, 4 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.352 | 0.842 | 0.842 | 0.045 | 0.251 | 0.734 | 0.481 | 0.897 | [0.808, 0.874] | — | — |
| v5a | 0.361 | 0.842 | 0.842 | 0.050 | 0.256 | 0.711 | 0.418 | 0.885 | [0.808, 0.874] | +0.000 | [-0.016, +0.016] |

## bench_sst2: gold_test

500 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### label (choice, 2 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.361 | 0.932 | 0.932 | 0.099 | 0.135 | 0.843 | 0.735 | 0.978 | [0.908, 0.954] | — | — |
| v5a | 0.381 | 0.924 | 0.924 | 0.093 | 0.143 | 0.819 | 0.632 | 0.965 | [0.900, 0.946] | -0.008 | [-0.024, +0.006] |

## bench_emotion: gold_test

2000 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### emotion (choice, 6 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.991 | 0.603 | 0.528 | 0.241 | 0.635 | 0.684 | 0.301 | 0.652 | [0.582, 0.623] | — | — |
| v5a | 0.687 | 0.570 | 0.500 | 0.067 | 0.597 | 0.651 | 0.281 | 0.614 | [0.549, 0.592] | -0.033 | [-0.046, -0.020] |

## bench_tweet_topic: gold_test

1693 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### topic (choice, 6 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.379 | 0.785 | 0.684 | 0.112 | 0.326 | 0.811 | 0.459 | 0.855 | [0.767, 0.804] | — | — |
| v5a | 0.398 | 0.788 | 0.685 | 0.197 | 0.352 | 0.783 | 0.443 | 0.852 | [0.771, 0.807] | +0.003 | [-0.007, +0.014] |

## bench_fin_topic: gold_test

4117 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### topic (choice, 20 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.430 | 0.576 | 0.585 | 0.065 | 0.565 | 0.743 | 0.335 | 0.648 | [0.561, 0.590] | — | — |
| v5a | 0.456 | 0.591 | 0.575 | 0.168 | 0.581 | 0.734 | 0.327 | 0.656 | [0.577, 0.606] | +0.015 | [+0.008, +0.023] |

## bench_daily_dialog: gold_test

7740 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### emotion (choice, 7 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.770 | 0.684 | 0.346 | 0.210 | 0.524 | 0.694 | 0.350 | 0.744 | [0.674, 0.695] | — | — |
| v5a | 0.652 | 0.601 | 0.313 | 0.137 | 0.595 | 0.603 | 0.283 | 0.642 | [0.590, 0.611] | -0.084 | [-0.090, -0.077] |

## bench_banking77: gold_test

500 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus v4a.

### label (choice, 77 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4a | 0.202 | 0.814 | 0.794 | 0.229 | 0.337 | 0.847 | 0.570 | 0.900 | [0.780, 0.846] | — | — |
| v5a | 0.196 | 0.830 | 0.817 | 0.232 | 0.324 | 0.836 | 0.541 | 0.902 | [0.794, 0.860] | +0.016 | [-0.006, +0.042] |
