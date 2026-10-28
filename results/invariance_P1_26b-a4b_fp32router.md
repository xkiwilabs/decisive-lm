P1_llm_response_coding_26b-a4b_seed0 on google/gemma-4-26B-A4B-it@4d7ae498, 200 Choice questions from gold_test (seed 0), dtype torch.bfloat16, 30 routers in fp32.

| Variant | max \|Δp\| | mean | p99 | answer flips | top-p rank ρ | review-set churn (20%) |
|---|---|---|---|---|---|---|
| repeat | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 200 | 1.0000 | 0.0% |
| reversed_options | 6.99e-01 | 1.69e-01 | 5.64e-01 | 44 / 200 | 0.7152 | 37.5% |
| after_another_request | 7.65e-01 | 2.27e-01 | 7.33e-01 | 47 / 200 | 0.5449 | 62.5% |
| extra_question | 7.72e-01 | 1.78e-01 | 6.52e-01 | 44 / 200 | 0.6865 | 45.0% |
