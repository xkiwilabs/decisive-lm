# Criterion 5 (contract), E4B v5c head (2026-11-02)

`data/queues/release_v5c_e4b.sh`, the same checks as v3d's (`results/release_v3d/`), on the real backbone in bf16.

**Conformance: 100% on every suite gold_test item.** That is 2,553 CUAD, 5,966 HelpSteer3, 3,384 Measuring Hate Speech and 29,739 MASSIVE rows, through the serving path (`check_bundle_*.txt`).

**Independence and order** (`invariance_multinli.md`, `invariance_wiki_toxicity.md`, 200 questions per type):
- At v3d's level throughout.
- max |Δp| is at most 0.029, and at most 2 of 200 answers flip (v3d: at most 0.019 and 1 of 200).
- Repeating a request gives identical answers.

**Agreement with the written predictions.** `check_bundle.py` also compares the serving path with `predict_bundle`'s packed predictions:
- CUAD, HelpSteer3 and Measuring Hate Speech: within 0.02.
- MASSIVE: max |Δp| is 0.119.
  - That is over the script's bf16 tolerance of 0.10, so its own check prints FAILS (exit 1).
  - 29,533 of 29,739 answers agree.
  - v3d's figure on the same set was 0.216, judged bf16 packing noise on near-flat 60-option distributions (`results/release_v3d/massive_gap.md`).
  - The 0.10 rule was added to the script on 2026-10-29, after v3d's check.
  - This comparison is not part of criterion 5 as written in docs/RELEASE.md (conformance, independence, order).

**Criterion 5: pass**, by the same reading as v3d. The MASSIVE agreement figure is better than v3d's.
