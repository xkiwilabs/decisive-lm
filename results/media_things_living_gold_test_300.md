# things_living: gold_test

356 rows. 95% CIs from 1000 item-level bootstrap resamples; Δ = method minus base 300.

## lives (score, 7 levels)

| Method | soft NLL | MAE | exact | EMD | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | MAE 95% CI | ΔMAE vs ref | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base 300 | 0.423 | 0.239 | 0.960 | 0.271 | 1.000 | 1.000 | 0.502 | 0.961 | [0.208, 0.271] | — | — |
| general 300 | 0.415 | 0.184 | 0.957 | 0.222 | 1.000 | 1.000 | 0.638 | 0.963 | [0.162, 0.211] | -0.055 | [-0.078, -0.034] |
