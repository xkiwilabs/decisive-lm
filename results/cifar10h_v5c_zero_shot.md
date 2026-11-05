# CIFAR-10H zero-shot: E4B v5c head against P0

Held out of every mix, never trained on; scored once on the 5,993 gold_test photos, task calibration fitted on the 4,007 gold_calib photos (`scripts/taskcal_report.py`; predictions `data/queues` by hand, predict_bundle bf16, 2026-11-03). v3d's run: `results/cifar10h_v3d_zero_shot.md`.

## cifar10h

5993 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus P0+task.

### label (choice, 10 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.245 | 0.875 | 0.875 | 0.056 | 0.193 | 0.894 | 0.739 | 0.959 | 0.323 | 0.882 | [0.867, 0.883] | -0.014 | [-0.020, -0.008] |
| P0+task | 0.220 | 0.889 | 0.889 | 0.048 | 0.166 | 0.911 | 0.772 | 0.968 | 0.362 | 0.911 | [0.881, 0.897] | — | — |
| general_v5c_e4b_r8_seed0 | 0.388 | 0.909 | 0.910 | 0.439 | 0.370 | 0.894 | 0.761 | 0.973 | 0.408 | 0.944 | [0.901, 0.916] | +0.020 | [+0.013, +0.027] |
| general_v5c_e4b_r8_seed0+task | 0.160 | 0.927 | 0.927 | 0.046 | 0.114 | 0.929 | 0.856 | 0.987 | 0.440 | 0.960 | [0.921, 0.934] | +0.038 | [+0.032, +0.045] |

**Raters, half against half** (`scripts/split_half.py`): cifar10h label: 5993 items, median 51 raters; split-half ρ 0.537 (Spearman-Brown 0.698); half A's top share predicting half B's agreement ρ 0.537 (200 random splits, seed 0)

**Fitted on fewer ratings**: 5 random draws (seeds 0-4) of the gold_calib rows, scored on all of gold_test.

| Fitted on | Method | accuracy, mean [range] | ρ(top p, agreement), mean [range] |
|---|---|---|---|
| 100 gold_calib rows | P0+task | 0.886 [0.876, 0.893] | +0.346 [+0.336, +0.358] |
| 100 gold_calib rows | general_v5c_e4b_r8_seed0+task | 0.922 [0.916, 0.925] | +0.429 [+0.421, +0.437] |
| 300 gold_calib rows | P0+task | 0.890 [0.886, 0.892] | +0.344 [+0.338, +0.352] |
| 300 gold_calib rows | general_v5c_e4b_r8_seed0+task | 0.925 [0.923, 0.927] | +0.429 [+0.423, +0.435] |

