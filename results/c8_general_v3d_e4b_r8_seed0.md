# Criterion 8: fine-tuning from general_v3d_e4b_r8_seed0 vs from the base

Adapters fine-tuned on N labelled training items (scripts/train_p3.py --n-train; 3 seeds per arm and size), from the general head (--init-bundle) or from the base model, scored on gold_test against the human targets. Arm value = mean over seeds; Δ = general − base; 95% CI from 1000 item-level bootstrap resamples shared by every run. Primary metric: accuracy against the human plurality (ties left out). Review readiness: ρ of top p with per-item pair agreement, tie AUROC, and the share of errors among the least-confident 20%.

## coding (792 gold_test items, 8 options)

### 100 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.854, 0.872, 0.872 | 0.889, 0.922, 0.922 | 0.866 | 0.911 | +0.045 | [+0.029, +0.060] | better |
| ρ(top p, agreement) | 0.440, 0.446, 0.533 | 0.511, 0.565, 0.544 | 0.473 | 0.540 | +0.067 | [+0.039, +0.095] | better |
| tie AUROC | 0.732, 0.723, 0.777 | 0.723, 0.735, 0.748 | 0.744 | 0.735 | -0.008 | [-0.032, +0.015] | n.s. |
| errors in review 20% | 0.705, 0.696, 0.598 | 0.600, 0.839, 0.643 | 0.666 | 0.694 | +0.028 | [-0.043, +0.114] | n.s. |
| ECE | 0.082, 0.079, 0.057 | 0.054, 0.079, 0.064 | 0.072 | 0.066 | -0.007 | [-0.022, +0.005] | n.s. |
| soft NLL | 0.335, 0.340, 0.312 | 0.309, 0.245, 0.291 | 0.329 | 0.282 | -0.047 | [-0.059, -0.035] | better |

### 300 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.914, 0.936, 0.907 | 0.914, 0.946, 0.907 | 0.919 | 0.922 | +0.003 | [-0.009, +0.016] | n.s. |
| ρ(top p, agreement) | 0.592, 0.579, 0.555 | 0.658, 0.614, 0.533 | 0.575 | 0.602 | +0.026 | [+0.003, +0.051] | better |
| tie AUROC | 0.813, 0.825, 0.779 | 0.866, 0.834, 0.730 | 0.806 | 0.810 | +0.004 | [-0.022, +0.034] | n.s. |
| errors in review 20% | 0.645, 0.783, 0.582 | 0.919, 0.769, 0.716 | 0.670 | 0.802 | +0.132 | [+0.034, +0.213] | better |
| ECE | 0.060, 0.051, 0.080 | 0.049, 0.051, 0.060 | 0.064 | 0.053 | -0.011 | [-0.022, +0.001] | n.s. |
| soft NLL | 0.277, 0.256, 0.297 | 0.220, 0.228, 0.278 | 0.277 | 0.242 | -0.034 | [-0.043, -0.026] | better |

## multinli (4,823 gold_test items, 3 options)

### 100 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.856, 0.828, 0.845 | 0.870, 0.854, 0.865 | 0.843 | 0.863 | +0.020 | [+0.013, +0.026] | better |
| ρ(top p, agreement) | 0.354, 0.329, 0.372 | 0.402, 0.391, 0.422 | 0.351 | 0.405 | +0.054 | [+0.040, +0.067] | better |
| tie AUROC | 0.757, 0.735, 0.715 | 0.767, 0.774, 0.776 | 0.736 | 0.772 | +0.037 | [+0.011, +0.063] | better |
| errors in review 20% | 0.523, 0.516, 0.499 | 0.568, 0.567, 0.592 | 0.513 | 0.576 | +0.063 | [+0.037, +0.089] | better |
| ECE | 0.085, 0.052, 0.073 | 0.070, 0.047, 0.058 | 0.070 | 0.059 | -0.012 | [-0.018, -0.006] | better |
| soft NLL | 0.519, 0.537, 0.518 | 0.483, 0.491, 0.477 | 0.525 | 0.484 | -0.041 | [-0.046, -0.036] | better |

### 300 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.857, 0.848, 0.857 | 0.868, 0.874, 0.865 | 0.854 | 0.869 | +0.015 | [+0.009, +0.021] | better |
| ρ(top p, agreement) | 0.397, 0.358, 0.377 | 0.421, 0.431, 0.401 | 0.377 | 0.418 | +0.041 | [+0.026, +0.054] | better |
| tie AUROC | 0.760, 0.752, 0.750 | 0.782, 0.791, 0.756 | 0.754 | 0.776 | +0.022 | [-0.003, +0.047] | n.s. |
| errors in review 20% | 0.551, 0.510, 0.564 | 0.595, 0.614, 0.560 | 0.541 | 0.590 | +0.048 | [+0.021, +0.072] | better |
| ECE | 0.070, 0.059, 0.063 | 0.060, 0.053, 0.061 | 0.064 | 0.058 | -0.006 | [-0.012, -0.000] | better |
| soft NLL | 0.496, 0.514, 0.496 | 0.475, 0.471, 0.483 | 0.502 | 0.477 | -0.026 | [-0.031, -0.021] | better |

## Against criterion 8

- Not significantly worse at either size, on accuracy or review readiness: **yes**.
- Better at 100 items on at least one task: **yes**: coding accuracy; coding ρ(top p, agreement); multinli accuracy; multinli ρ(top p, agreement); multinli tie AUROC; multinli errors in review 20%.
