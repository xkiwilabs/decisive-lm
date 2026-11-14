# Criterion 8: fine-tuning from general_v5c_e4b_r8_seed0 vs from the base

Adapters fine-tuned on N labelled training items (scripts/train_p3.py --n-train; 3 seeds per arm and size), from the general head (--init-bundle) or from the base model, scored on gold_test against the human targets. Arm value = mean over seeds; Δ = general − base; 95% CI from 1000 item-level bootstrap resamples shared by every run. Primary metric: accuracy against the human plurality (ties left out). Review readiness: ρ of top p with per-item pair agreement, tie AUROC, and the share of errors among the least-confident 20%.

## coding (792 gold_test items, 8 options)

### 100 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.882, 0.884, 0.864 | 0.905, 0.930, 0.922 | 0.877 | 0.919 | +0.043 | [+0.029, +0.058] | better |
| ρ(top p, agreement) | 0.536, 0.465, 0.438 | 0.580, 0.607, 0.565 | 0.479 | 0.584 | +0.105 | [+0.079, +0.130] | better |
| tie AUROC | 0.763, 0.755, 0.763 | 0.755, 0.771, 0.763 | 0.760 | 0.763 | +0.003 | [-0.029, +0.031] | n.s. |
| errors in review 20% | 0.647, 0.783, 0.684 | 0.632, 0.860, 0.696 | 0.705 | 0.730 | +0.025 | [-0.077, +0.093] | n.s. |
| ECE | 0.060, 0.068, 0.060 | 0.052, 0.068, 0.062 | 0.062 | 0.061 | -0.002 | [-0.017, +0.011] | n.s. |
| soft NLL | 0.311, 0.311, 0.360 | 0.289, 0.246, 0.281 | 0.327 | 0.272 | -0.055 | [-0.068, -0.043] | better |

### 300 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.935, 0.932, 0.866 | 0.943, 0.948, 0.950 | 0.911 | 0.947 | +0.036 | [+0.024, +0.049] | better |
| ρ(top p, agreement) | 0.630, 0.554, 0.556 | 0.666, 0.647, 0.662 | 0.580 | 0.658 | +0.078 | [+0.053, +0.104] | better |
| tie AUROC | 0.831, 0.812, 0.795 | 0.836, 0.860, 0.860 | 0.813 | 0.852 | +0.039 | [+0.015, +0.065] | better |
| errors in review 20% | 0.851, 0.857, 0.688 | 0.829, 0.838, 0.833 | 0.799 | 0.833 | +0.035 | [-0.060, +0.137] | n.s. |
| ECE | 0.058, 0.048, 0.058 | 0.053, 0.055, 0.051 | 0.055 | 0.053 | -0.002 | [-0.015, +0.008] | n.s. |
| soft NLL | 0.239, 0.274, 0.296 | 0.226, 0.219, 0.223 | 0.270 | 0.223 | -0.047 | [-0.057, -0.038] | better |

## multinli (4,823 gold_test items, 3 options)

### 100 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.856, 0.828, 0.845 | 0.868, 0.850, 0.866 | 0.843 | 0.861 | +0.018 | [+0.011, +0.025] | better |
| ρ(top p, agreement) | 0.354, 0.329, 0.372 | 0.403, 0.385, 0.409 | 0.351 | 0.399 | +0.048 | [+0.032, +0.062] | better |
| tie AUROC | 0.757, 0.735, 0.715 | 0.756, 0.750, 0.754 | 0.736 | 0.754 | +0.018 | [-0.009, +0.046] | n.s. |
| errors in review 20% | 0.523, 0.516, 0.499 | 0.570, 0.522, 0.545 | 0.513 | 0.546 | +0.033 | [+0.008, +0.058] | better |
| ECE | 0.085, 0.052, 0.073 | 0.067, 0.053, 0.063 | 0.070 | 0.061 | -0.009 | [-0.016, -0.003] | better |
| soft NLL | 0.519, 0.537, 0.518 | 0.483, 0.497, 0.484 | 0.525 | 0.488 | -0.037 | [-0.042, -0.031] | better |

### 300 training items (3 seeds)

| Metric | base (per seed) | general (per seed) | base | general | Δ | 95% CI | |
|---|---|---|---|---|---|---|---|
| accuracy | 0.857, 0.848, 0.857 | 0.868, 0.873, 0.878 | 0.854 | 0.873 | +0.019 | [+0.012, +0.026] | better |
| ρ(top p, agreement) | 0.397, 0.358, 0.377 | 0.415, 0.433, 0.440 | 0.377 | 0.430 | +0.052 | [+0.038, +0.065] | better |
| tie AUROC | 0.760, 0.752, 0.750 | 0.770, 0.794, 0.778 | 0.754 | 0.781 | +0.027 | [+0.000, +0.053] | better |
| errors in review 20% | 0.551, 0.510, 0.564 | 0.564, 0.601, 0.558 | 0.541 | 0.574 | +0.033 | [+0.005, +0.060] | better |
| ECE | 0.070, 0.059, 0.063 | 0.059, 0.053, 0.065 | 0.064 | 0.059 | -0.005 | [-0.011, +0.001] | n.s. |
| soft NLL | 0.496, 0.514, 0.496 | 0.479, 0.469, 0.470 | 0.502 | 0.473 | -0.030 | [-0.035, -0.024] | better |

## Against criterion 8

- Not significantly worse at either size, on accuracy or review readiness: **yes**.
- Better at 100 items on at least one task: **yes**: coding accuracy; coding ρ(top p, agreement); multinli accuracy; multinli ρ(top p, agreement); multinli errors in review 20%.
