# Zero-shot starting points on the 1.0 image and audio training sets

The prompted base model (P0: Gemma 4 E4B run locally with each row's image or audio, `scripts/label_teacher.py
--local`) and the general head v3d (trained on text only), on the gold_test rows of the three sets the first 1.0
candidate trains on, before any image or audio training. Each also with task calibration fitted on the set's
gold_calib rows (`scripts/task_calibrate.py`). Scored with `scripts/eval_dataset.py`, P0+task the reference; 95% CIs
from 1,000 item-level bootstrap resamples. The frozen 1.0 suites (TextVQA, ODAQ) are not touched here.

Summary:

| Set | Measure | P0+task | v3d+task | Δ [95% CI] |
|---|---|---|---|---|
| Emozionalmente (7 emotions, 1,435 clips) | accuracy | 0.199 | 0.200 | +0.002 [−0.018, +0.022] |
| Emozionalmente | ρ(top p, agreement) | +0.047 | −0.001 | |
| SONYC-UST (8 sounds, 412 clips) | mean AUROC over the 8 | 0.585 | 0.667 | |
| VizWiz-VQA (2,567 photos) | answerable AUROC | 0.788 | 0.828 | +0.041 [+0.027, +0.053] |
| VizWiz-VQA, 4-option answers (1,953) | accuracy | 0.432 | 0.433 | +0.001 [−0.025, +0.027] |
| VizWiz-VQA, 4-option answers | ρ(top p, agreement) | +0.292 | +0.332 | |

Raters, half against half (`scripts/split_half.py`), for comparison, not a ceiling: Emozionalmente ρ 0.255 (Spearman-
Brown 0.407); VizWiz answers ρ 0.431 (0.602).

What it says:
- **Audio is close to chance for both.** Emotion from an Italian voice: 20% right with 7 options (the human plurality
  label); P0's own confidence barely follows how far the five listeners agreed. On city sounds, P0's AUROC is near
  0.5 for most classes and inverted for human voice (0.256): it gives clips without a voice the higher P(yes). v3d
  is better on voice (0.714) and dog (0.861) but worse on powered saw (0.599 against P0's 0.912). The CREMA-D example
  showed the same pattern: P0 0.55, trained on its rows 0.75 (`results/media_crema_d_gold_test.md`). Training on
  multi-rater audio is what the 1.0 candidate has to add.
- **Photos start better.** Whether a blind user's photo can answer their question: AUROC 0.79 (P0) and 0.83 (v3d),
  a significant +0.04 for the text-trained head. The answer itself: 43% match the answerers' most given answer with
  four options, where the answerers' own answers often differ only in wording. Confidence follows their agreement
  (ρ 0.29–0.33), less well than one half of them predicts the other (0.43).
- **Task calibration can say "ignore the model".** On two SONYC sounds the fitted scale ran to 3.7 × 10⁴ and
  1.1 × 10⁵: v3d's logits carried no signal there, so the task step answers each sound's base rate. That is the
  right fit for an uninformative score, but a scale that large should be read as a warning, not a calibration.

Per-question tables follow.

## emozionalmente: gold_test

1435 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### emotion (choice, 7 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.989 | 0.204 | 0.124 | 0.012 | 0.851 | 0.530 | 0.207 | 0.214 | 0.126 | 0.528 | [0.182, 0.227] | +0.005 | [-0.018, +0.030] |
| P0+task | 0.990 | 0.199 | 0.125 | 0.014 | 0.852 | 0.562 | 0.210 | 0.208 | 0.047 | 0.477 | [0.175, 0.222] | — | — |
| v3d | 1.475 | 0.180 | 0.062 | 0.565 | 1.241 | 0.549 | 0.209 | 0.192 | 0.088 | 0.488 | [0.159, 0.204] | -0.019 | [-0.042, +0.005] |
| v3d+task | 0.988 | 0.200 | 0.126 | 0.015 | 0.852 | 0.532 | 0.208 | 0.206 | -0.001 | 0.466 | [0.176, 0.223] | +0.002 | [-0.018, +0.022] |

## sonyc_ust: gold_test

412 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### engine (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.537 | 0.752 | 0.200 | 0.121 | 0.673 | 0.102 | [0.479, 0.595] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.537 | 0.752 | 0.187 | 0.050 | 0.589 | 0.102 | [0.479, 0.595] | — | — |
| v3d | 0.579 | 0.248 | 0.358 | 0.421 | 0.898 | 0.173 | [0.513, 0.648] | +0.042 | [-0.042, +0.135] |
| v3d+task | 0.579 | 0.757 | 0.185 | 0.091 | 0.584 | 0.173 | [0.513, 0.648] | +0.042 | [-0.042, +0.135] |

### machinery_impact (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.658 | 0.981 | 0.023 | 0.062 | 0.217 | 0.214 | [0.455, 0.833] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.658 | 0.981 | 0.022 | 0.054 | 0.215 | 0.214 | [0.455, 0.833] | — | — |
| v3d | 0.742 | 0.981 | 0.208 | 0.435 | 0.615 | 0.036 | [0.589, 0.887] | +0.085 | [-0.107, +0.274] |
| v3d+task | 0.742 | 0.981 | 0.023 | 0.061 | 0.222 | 0.036 | [0.589, 0.887] | +0.085 | [-0.107, +0.274] |

### non_machinery_impact (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.470 | 0.971 | 0.034 | 0.073 | 0.220 | 0.004 | [0.296, 0.640] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.470 | 0.971 | 0.029 | 0.025 | 0.204 | 0.004 | [0.296, 0.640] | — | — |
| v3d | 0.584 | 0.029 | 0.332 | 0.551 | 0.851 | 0.115 | [0.402, 0.767] | +0.114 | [-0.029, +0.272] |
| v3d+task | 0.578 | 0.971 | 0.030 | 0.037 | 0.203 | 0.112 | [0.397, 0.759] | +0.108 | [-0.042, +0.263] |

### powered_saw (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.912 | 0.998 | 0.004 | 0.044 | 0.077 | 0.154 | [0.883, 0.939] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.912 | 0.998 | 0.003 | 0.019 | 0.059 | 0.154 | [0.883, 0.939] | — | — |
| v3d | 0.599 | 0.993 | 0.106 | 0.314 | 0.398 | -0.029 | [0.549, 0.647] | -0.314 | [-0.374, -0.261] |
| v3d+task | 0.599 | 0.998 | 0.004 | 0.025 | 0.067 | -0.029 | [0.549, 0.647] | -0.314 | [-0.374, -0.261] |

### alert_signal (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.504 | 0.876 | 0.109 | 0.044 | 0.421 | 0.031 | [0.421, 0.589] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.504 | 0.876 | 0.109 | 0.033 | 0.420 | 0.031 | [0.421, 0.589] | — | — |
| v3d | 0.555 | 0.150 | 0.269 | 0.375 | 0.728 | 0.132 | [0.466, 0.643] | +0.050 | [-0.062, +0.171] |
| v3d+task | 0.555 | 0.876 | 0.110 | 0.042 | 0.419 | 0.132 | [0.466, 0.643] | +0.050 | [-0.062, +0.171] |

### music (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.578 | 0.959 | 0.041 | 0.044 | 0.221 | 0.022 | [0.434, 0.726] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.578 | 0.959 | 0.039 | 0.011 | 0.215 | 0.022 | [0.434, 0.726] | — | — |
| v3d | 0.712 | 0.956 | 0.168 | 0.357 | 0.532 | 0.113 | [0.546, 0.847] | +0.134 | [-0.093, +0.366] |
| v3d+task | 0.712 | 0.956 | 0.039 | 0.015 | 0.209 | 0.113 | [0.546, 0.847] | +0.134 | [-0.093, +0.366] |

### human_voice (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.256 | 0.648 | 0.260 | 0.179 | 0.736 | -0.432 | [0.209, 0.308] | -0.180 | [-0.221, -0.139] |
| P0+task | 0.435 | 0.648 | 0.239 | 0.105 | 0.680 | -0.150 | [0.399, 0.474] | — | — |
| v3d | 0.714 | 0.648 | 0.220 | 0.131 | 0.632 | 0.421 | [0.662, 0.763] | +0.278 | [+0.209, +0.344] |
| v3d+task | 0.714 | 0.682 | 0.206 | 0.066 | 0.607 | 0.421 | [0.662, 0.763] | +0.278 | [+0.209, +0.344] |

### dog (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.587 | 0.966 | 0.033 | 0.008 | 0.179 | 0.148 | [0.425, 0.748] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.587 | 0.966 | 0.033 | 0.025 | 0.217 | 0.148 | [0.425, 0.748] | — | — |
| v3d | 0.861 | 0.044 | 0.330 | 0.539 | 0.853 | 0.269 | [0.730, 0.958] | +0.274 | [+0.082, +0.474] |
| v3d+task | 0.855 | 0.966 | 0.033 | 0.011 | 0.190 | 0.266 | [0.716, 0.957] | +0.267 | [+0.072, +0.473] |

### Mean of the primary metric over questions

| Method | noul |
|---|---|
| P0 | 0.563 |
| P0+task | 0.585 |
| v3d | 0.668 |
| v3d+task | 0.667 |

## vizwiz_vqa: gold_test

2567 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### answerable (noul)

| Method | AUROC | acc@.5 | Brier | ECE | soft BCE | ρ(p, share) | AUROC 95% CI | ΔAUROC vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.788 | 0.781 | 0.162 | 0.064 | 0.549 | 0.539 | [0.768, 0.807] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.788 | 0.781 | 0.162 | 0.064 | 0.549 | 0.539 | [0.768, 0.807] | — | — |
| v3d | 0.828 | 0.803 | 0.147 | 0.068 | 0.522 | 0.601 | [0.808, 0.847] | +0.041 | [+0.027, +0.053] |
| v3d+task | 0.828 | 0.802 | 0.145 | 0.058 | 0.522 | 0.601 | [0.808, 0.847] | +0.041 | [+0.027, +0.053] |

### answer (choice, 4 options; asked on 1953 rows)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.948 | 0.432 | 0.432 | 0.061 | 0.678 | 0.638 | 0.222 | 0.461 | 0.292 | 0.588 | [0.406, 0.457] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.948 | 0.432 | 0.432 | 0.061 | 0.678 | 0.638 | 0.222 | 0.461 | 0.292 | 0.588 | [0.406, 0.457] | — | — |
| v3d | 0.960 | 0.433 | 0.434 | 0.087 | 0.640 | 0.690 | 0.238 | 0.465 | 0.332 | 0.574 | [0.407, 0.458] | +0.001 | [-0.025, +0.027] |
| v3d+task | 0.960 | 0.433 | 0.434 | 0.087 | 0.640 | 0.690 | 0.238 | 0.465 | 0.332 | 0.574 | [0.407, 0.458] | +0.001 | [-0.025, +0.027] |

### answer (choice, 2 options; asked on 127 rows)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.879 | 0.776 | 0.776 | 0.196 | 0.400 | 0.713 | 0.308 | 0.804 | 0.296 | 0.590 | [0.692, 0.853] | +0.000 | [+0.000, +0.000] |
| P0+task | 0.879 | 0.776 | 0.776 | 0.196 | 0.400 | 0.713 | 0.308 | 0.804 | 0.296 | 0.590 | [0.692, 0.853] | — | — |
| v3d | 0.844 | 0.733 | 0.732 | 0.059 | 0.361 | 0.685 | 0.290 | 0.766 | 0.095 | 0.542 | [0.644, 0.815] | -0.043 | [-0.136, +0.046] |
| v3d+task | 0.844 | 0.733 | 0.732 | 0.059 | 0.361 | 0.685 | 0.290 | 0.766 | 0.095 | 0.542 | [0.644, 0.815] | -0.043 | [-0.136, +0.046] |

### Mean of the primary metric over questions

| Method | choice | noul |
|---|---|---|
| P0 | 0.604 | 0.788 |
| P0+task | 0.604 | 0.788 |
| v3d | 0.583 | 0.828 |
| v3d+task | 0.583 | 0.828 |
