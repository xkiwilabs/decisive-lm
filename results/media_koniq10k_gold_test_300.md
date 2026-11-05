# koniq10k: gold_test

1214 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus base 300.

## quality (score, 5 levels)

| Method | soft NLL | MAE | exact | EMD | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | MAE 95% CI | ΔMAE vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base 300 | 0.608 | 0.228 | 0.728 | 0.249 | 0.293 | 0.760 | 0.085 | 0.455 | [0.217, 0.238] | — | — |
| general 300 | 0.601 | 0.220 | 0.732 | 0.238 | 0.322 | 0.773 | 0.046 | 0.613 | [0.210, 0.230] | -0.008 | [-0.014, -0.001] |
