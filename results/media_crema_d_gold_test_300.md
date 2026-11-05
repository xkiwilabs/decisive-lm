# crema_d: gold_test

1476 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus base 300.

## emotion (choice, 6 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base 300 | 0.872 | 0.577 | 0.122 | 0.150 | 0.650 | 0.589 | 0.243 | 0.602 | 0.112 | 0.559 | [0.551, 0.601] | — | — |
| general 300 | 0.840 | 0.613 | 0.208 | 0.170 | 0.606 | 0.639 | 0.290 | 0.657 | 0.154 | 0.564 | [0.587, 0.636] | +0.035 | [+0.022, +0.049] |
