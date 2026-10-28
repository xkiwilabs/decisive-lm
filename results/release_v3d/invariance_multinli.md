general_v3d_e4b_r8_seed0 on google/gemma-4-E4B-it@ee0ef602, 200 choice, 200 noul, 200 score questions from gold_test (seed 0), dtype torch.bfloat16. A Noul's answer is yes or no; options are reversed for Choice questions only.

| Type | Variant | max \|Δp\| | mean | p99 | answer flips | top-p rank ρ | review-set churn (20%) |
|---|---|---|---|---|---|---|---|
| choice | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| choice | reversed_options | 5.94e-03 | 1.63e-03 | 5.17e-03 | 0 / 200 | 0.9998 | 2.5% |
| choice | after_another_request | 7.56e-03 | 2.51e-03 | 6.81e-03 | 0 / 200 | 0.9996 | 2.5% |
| choice | extra_question | 1.12e-02 | 1.50e-03 | 7.91e-03 | 0 / 200 | 0.9996 | 0.0% |
| noul | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| noul | after_another_request | 1.65e-02 | 2.18e-03 | 9.33e-03 | 0 / 200 | 0.9995 | 2.5% |
| noul | extra_question | 8.64e-03 | 1.15e-03 | 6.79e-03 | 0 / 200 | 0.9997 | 2.5% |
| score | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| score | after_another_request | 1.86e-02 | 2.65e-03 | 1.11e-02 | 1 / 200 | 0.9997 | 0.0% |
| score | extra_question | 1.79e-02 | 1.94e-03 | 1.30e-02 | 1 / 200 | 0.9997 | 2.5% |
