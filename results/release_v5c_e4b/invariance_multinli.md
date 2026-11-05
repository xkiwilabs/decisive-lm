general_v5c_e4b_r8_seed0 on google/gemma-4-E4B-it@ee0ef602, 200 choice, 200 noul, 200 score questions from gold_test (seed 0), dtype torch.bfloat16. A Noul's answer is yes or no; options are reversed for Choice questions only.

| Type | Variant | max \|Δp\| | mean | p99 | answer flips | top-p rank ρ | review-set churn (20%) |
|---|---|---|---|---|---|---|---|
| choice | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| choice | reversed_options | 7.61e-03 | 2.01e-03 | 6.41e-03 | 0 / 200 | 0.9998 | 0.0% |
| choice | after_another_request | 1.35e-02 | 2.72e-03 | 9.02e-03 | 2 / 200 | 0.9997 | 2.5% |
| choice | extra_question | 9.74e-03 | 1.77e-03 | 7.19e-03 | 1 / 200 | 0.9998 | 0.0% |
| noul | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| noul | after_another_request | 1.47e-02 | 1.94e-03 | 9.11e-03 | 0 / 200 | 0.9997 | 5.0% |
| noul | extra_question | 2.90e-02 | 1.14e-03 | 5.70e-03 | 0 / 200 | 0.9998 | 0.0% |
| score | repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| score | after_another_request | 1.53e-02 | 2.65e-03 | 1.19e-02 | 1 / 200 | 0.9997 | 0.0% |
| score | extra_question | 1.12e-02 | 1.69e-03 | 7.74e-03 | 0 / 200 | 0.9998 | 0.0% |
