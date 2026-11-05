general_v5c_26b-a4b_r8_seed0 on google/gemma-4-26B-A4B-it@4d7ae498, 200 choice, 200 noul, 200 score questions from gold_test (seed 0), dtype torch.float32. A Noul's answer is yes or no; options are reversed for Choice questions only.

| Type | Variant | max \|Δp\| | mean | p99 | answer flips | top-p rank ρ | review-set churn (20%) |
|---|---|---|---|---|---|---|---|
| choice | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| choice | reversed_options | 1.13e-03 | 6.37e-06 | 1.71e-06 | 0 / 200 | 1.0000 | 0.0% |
| choice | after_another_request | 1.18e-04 | 1.54e-06 | 4.95e-05 | 0 / 200 | 1.0000 | 0.0% |
| choice | extra_question | 1.18e-04 | 2.06e-06 | 8.94e-05 | 0 / 200 | 1.0000 | 0.0% |
| noul | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| noul | after_another_request | 7.75e-07 | 1.56e-07 | 5.36e-07 | 0 / 200 | 1.0000 | 0.0% |
| noul | extra_question | 7.15e-07 | 1.36e-07 | 5.97e-07 | 0 / 200 | 1.0000 | 0.0% |
| score | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| score | after_another_request | 4.43e-03 | 3.69e-05 | 2.37e-04 | 0 / 200 | 1.0000 | 0.0% |
| score | extra_question | 4.43e-03 | 4.55e-05 | 5.83e-04 | 0 / 200 | 1.0000 | 0.0% |
