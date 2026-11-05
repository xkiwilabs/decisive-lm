# Criterion 5 (contract), 26B A4B v5c head (2026-11-05)

`data/queues/release_v5c_26b.sh` and `data/queues/release_v5c_26b_rest.sh`: the same checks as the E4B head's (`results/release_v5c_e4b/criterion5.md`), on the real backbone in float32.

**Conformance: 100% on every suite gold_test item.** That is 2,553 CUAD, 5,966 HelpSteer3, 3,384 Measuring Hate Speech and 29,739 MASSIVE rows, through the serving path (`check_bundle_*.txt`).

**Independence and order** (`invariance_multinli.md`, `invariance_wiki_toxicity.md` and `results/invariance_general_v5c_26b-a4b_r8_seed0.md` on Banking77, 200 questions per type):
- No answer flips and no review-set churn in any variant. The E4B head had at most 2 of 200 flips.
- max |Δp| is at most 0.011, and the 99th percentile is at most 6e-4. The E4B head's max was 0.029.
- Repeating a request gives identical answers.

**Agreement with the written predictions.** `check_bundle.py` also compares the serving path, one request at a time, with `predict_bundle`'s packed predictions. Its float32 tolerance is 0.001, and every set prints FAILS against it:

| Set | Questions | Answers that changed | max \|Δp\| | Largest written margin among the changes |
|---|---|---|---|---|
| CUAD | 25,530 | 0 | 0.010 | |
| HelpSteer3 | 5,966 | 18 (0.30%) | 0.041 | 0.026 |
| Measuring Hate Speech | 33,840 | 17 (0.05%) | 0.056 | 0.075 |
| MASSIVE | 29,739 | 1 | 0.052 | 0.046 |

- **The tolerance was set for a dense model.** In float32 the dense E4B matches to about 1e-5 across packings (`results/M0.md`). The 26B A4B picks 8 of 128 experts per token, so a near-tie in one token's routing can switch experts and move the hidden state by a jump rather than by rounding. M0 measured this directly: with free routing, fp32 packed-versus-alone differences were 1.5e-2 (leaf) and 9.9e-2 (query) against 1e-3, first flipping at layer 15 (`results/M0.md`); the mask check passes only with the expert selection forced.
- Nearly all of the changed answers were near-ties in the written prediction: 34 of 36 had margins of 0.026 or less (12 of them 0.002 or less, which the script does not list). The other two had margins of 0.046 (MASSIVE) and 0.075 (MHS).
- The E4B head's bf16 figures for comparison: 206 MASSIVE answers changed, max |Δp| 0.119.
- This comparison is not part of criterion 5 as written in docs/RELEASE.md (conformance, independence, order).

**Criterion 5: pass**, by the same reading as the E4B head and v3d. On the serving path's own repeatability, the 26B head is closer than the E4B head: fewer answers change, and by less.
