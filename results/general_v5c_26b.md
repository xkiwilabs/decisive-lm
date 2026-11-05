# General v5c, 26B A4B head: release checks against the prompted 26B (P0) and the E4B v5c head

`general_v5c_26b-a4b_r8_seed0` is the second first-release candidate. It is trained on the same mix as the released E4B head (`general_v5c_e4b_r8_seed0`, `results/general_v5c_e4b.md`), with the same recipe, on Gemma 4 26B A4B (`google/gemma-4-26B-A4B-it@4d7ae498`). It uses rank-8 LoRA, bf16 weights and **float32 computation** in training, checks and serving (RELEASE.md, 2026-10-30: in bf16 the expert routing changes 19–22% of answers when only the packing changes). For this head, P0 is the 26B prompted, so P0 and the teacher are the same model.

**Training** (`scripts/train_p3.py`, on one RTX PRO 6000 with nothing else on it):
- 215,433 training rows, 2 epochs at most (26,930 steps of 16 rows). Validation every 500 steps on 1,500 val rows (the E4B head's rows), patience 3.
- Early stopping ended training at step 16,500. The bundle is the one selected by validation calibration loss, at step 15,000 (0.1982; best accuracy 92.5% at step 15,500).
- Training took 101 h (6,072 min) at 1.38 s/row, from 31 October 06:39 to 4 November 12:04. Calibration and predictions for the three bundles (by calibration loss, loss and accuracy) took until 18:13.
- Memory: 66.0 GiB (67,611 MiB) on the card in training (nvidia-smi, 4,096-token micro-batches); 55–60 GiB allocated by PyTorch in prediction.

**How it was scored.**
- The frozen suite once: `scripts/score_suite.py`, `results/suite_general_v5c_26b-a4b_r8_seed0.md`.
- Development sets and benchmarks: `scripts/eval_dataset.py` (`results/release_v5c_26b/dev_<set>.md`) and `scripts/score_suite.py --sets bench_*` (`results/bench_general_v5c_26b.md`), on gold_test with 95% CIs from 1,000 item-level bootstrap resamples. Every head is raw, with no per-task step.
- One slip in the run: three suite sets were started before the GPU they needed was free. They ran out of memory and the suite was scored on MASSIVE alone (kept as `results/release_v5c_26b/suite_partial_massive_only.md`). The three sets were then predicted, and all four were scored together. No set was scored twice, and MASSIVE's numbers are identical in both files.

## The frozen suite: fails five checks of 0.x

| Task | Check | 26B head | P0 (26B prompted) | Δ [95% CI] | 0.x |
|---|---|---|---|---|---|
| MASSIVE | 3 accuracy | 0.680 | 0.731 | −0.051 [−0.055, −0.046] | **fail** |
| MHS Score | 1 errors caught in a 20% review | 0.234 | 0.267 | −0.033 [−0.040, −0.025] | **fail** |
| MHS Score | 1 lowest within-category ρ (44 categories) | −0.205 | −0.175 | | **fail** |
| MHS Score | 2 Brier, after the per-task step vs before | 0.6403 | 0.6399 (before) | +0.0004 [+0.0002, +0.0006] | **fail** |
| HelpSteer3 Score | 1 lowest within-category ρ (6 categories) | −0.038 | −0.011 | | **fail** |

Everything else passes for 0.x, including:
- **MHS Score review readiness:** ρ(top p, agreement) 0.177 against 0.162, Δ +0.015 [+0.001, +0.029]; tie AUROC 0.579 against 0.562.
- **HelpSteer3 Score:** MAE 0.899 against 1.158.
- **MHS Noul:** AUROC 0.882 against 0.845.
- **Calibration:** ECE after the per-task step 0.018–0.040 on every task.

**Criterion 4** (close to the E4B specialists on their own gold_test): every check passes, review readiness included (HelpSteer2 and GoEmotions). The primary metric is within noise of the specialist on 4 of 7 seed tasks.

For 1.0, the primary metric beats P0 on 3 of 5 tasks.

### Against the E4B head

The E4B head failed three checks (RELEASE.md, 2026-11-03), against a weaker P0: the E4B prompted. The two heads side by side, from the two suite files:

| Task | Measure | 26B head | E4B head | P0 26B | P0 E4B |
|---|---|---|---|---|---|
| HelpSteer3 Score | MAE | **0.899** | 1.023 | 1.158 | 1.292 |
| HelpSteer3 Score | ρ(top p, agreement) | 0.065 | 0.061 | 0.088 | 0.043 |
| MHS Score | MAE | **0.571** | 0.653 | 0.703 | 0.657 |
| MHS Score | ρ(top p, agreement) | **0.177** | 0.078 | 0.162 | 0.201 |
| MHS Score | tie AUROC | **0.579** | 0.533 | 0.562 | 0.578 |
| MHS Score | errors caught in a 20% review | 0.234 | 0.245 | 0.267 | 0.238 |
| MHS Noul | AUROC | **0.882** | 0.852 | 0.845 | 0.855 |
| CUAD Noul | AUROC | **0.961** | 0.947 | 0.954 | 0.945 |
| MASSIVE | accuracy | 0.680 | 0.672 | **0.731** | 0.658 |

- These are not paired comparisons (each file scores the rows every one of its methods answered), so there are no CIs.
- **The 26B head improves on the E4B head on every suite task's primary metric.** The largest gain is review readiness on MHS Score, where the E4B head failed: ρ 0.177 against 0.078.
- **It fails more checks because its bar is higher.** The prompted 26B is much more accurate than the prompted E4B on MASSIVE (0.731 against 0.658). It also catches more errors in an MHS review (0.267 against 0.238).

## Development sets and benchmarks

Primary metric, plus review readiness where items have several raters:

| Set | Measure | 26B head | P0 26B | E4B head | Δ 26B head − E4B head [95% CI] |
|---|---|---|---|---|---|
| VizWiz answerable | AUROC | **0.938** | 0.802 | 0.933 | +0.005 [−0.001, +0.011] |
| VizWiz answer, 4 options | accuracy | **0.731** | 0.485 | 0.711 | +0.020 [−0.001, +0.042] |
| VizWiz answer, 4 options | ρ(top p, agreement) | **0.676** | 0.367 | 0.633 | |
| MultiNLI relation | accuracy | 0.845 | **0.856** | 0.810 | +0.035 [+0.026, +0.045] |
| MultiNLI relation | ρ(top p, agreement) | **0.365** | 0.321 | 0.323 | |
| MultiNLI support, 3 levels | MAE | 0.239 | **0.223** | 0.333 | −0.094 [−0.102, −0.087] |
| MultiNLI support | ρ(top p, agreement) | **0.347** | 0.223 | 0.200 | |
| Sarcasm corpus, 6 levels | MAE | **0.830** | 0.832 | 1.070 | −0.240 [−0.286, −0.194] |
| Sarcasm corpus | ρ(top p, agreement) | **0.231** | 0.179 | 0.028 | |
| Wikipedia toxicity, 5 levels | MAE | 0.526 | **0.290** | 0.462 | +0.064 [+0.054, +0.075] |
| Wikipedia toxicity | ρ(top p, agreement) | +0.002 | −0.162 | −0.010 | |
| Humicroedit, 4 levels | MAE | 0.845 | 0.640 | **0.567** | +0.278 [+0.263, +0.294] |
| Humicroedit | ρ(top p, agreement) | −0.032 | +0.061 | **+0.113** | |

- **Review readiness: ρ is above P0's on four of the five sets** (no CIs computed for the ρ differences). The exception is Humicroedit.
- **Wikipedia toxicity:** the E4B head's review-readiness deficit against its P0 is gone. Against the prompted 26B, whose ρ is −0.162, the 26B head is at +0.002. The E4B head was −0.010 against the E4B prompted's +0.057.
- **Humicroedit is the 26B head's weak set.** It rates the edits as funnier than the raters do: a mean predicted level of 1.75 against the raters' 0.96 (E4B head 1.35). Its ranking of the items is about the same as the E4B head's (correlation with the raters' mean 0.30 against 0.33). It is worse than both prompted models on MAE and on review readiness.

**The seven benchmarks** (`results/bench_general_v5c_26b.md`), accuracy against the prompted 26B:

| | AG News | SST-2 | Banking77 | emotion | TweetTopic | financial topics | DailyDialog |
|---|---|---|---|---|---|---|---|
| 26B head | 0.850 | 0.927 | 0.818 | 0.605 | 0.809 | 0.624 | 0.629 |
| P0 26B | 0.853 | 0.958 | 0.767 | 0.598 | 0.829 | 0.648 | 0.664 |
| E4B head | 0.853 | 0.920 | 0.845 | 0.604 | 0.756 | 0.586 | 0.645 |

- The 26B head is level with P0 on 2 of 7 benchmarks, better on Banking77, and worse on 4 (by 2–4 points). ECE after the per-task step is 0.04–0.09.
- Against the E4B head (unpaired) it is clearly better on 2 (TweetTopic +5 points, financial topics +4) and worse on 2 (Banking77 −3, DailyDialog −2).

## Criterion 5: contract

- **Invariance** (`results/invariance_general_v5c_26b-a4b_r8_seed0.md`, `results/release_v5c_26b/invariance_{multinli,wiki_toxicity}.md`; 200 questions each in float32): no answer flips and no review-set churn under repeats, reversed options, another request in the same batch or an extra question. The largest probability change is 1.05e-02 (one Wikipedia toxicity item, after another request), and the 99th percentile is 6e-04 or less.
- **Conformance** (`scripts/check_bundle.py`, `results/release_v5c_26b/check_bundle_<set>.txt`): 100% on all four suite sets.
- **Agreement with the written predictions** (one request at a time against the packed prediction run, at the float32 tolerance of 0.001): fails on every set, as the E4B head's MASSIVE check did in bf16. 36 answers changed: 18 of 5,966 on HelpSteer3, 17 of 33,840 on MHS, 1 of 29,739 on MASSIVE and none on CUAD. The largest change is 0.056, and 34 of the 36 were near-ties (written margin ≤ 0.026). The most likely cause is MoE routing (`results/release_v5c_26b/criterion5.md`). The comparison is not part of criterion 5 as written.
- **Criterion 5: pass**, by the reading used for the E4B head and v3d.

## Review readiness against P0, with intervals

ρ(top p, pair agreement) on gold_test items with two or more raters, paired bootstrap of 1,000 item resamples, seed 0
(`scripts/paper_v5c_26b/drho.py`):

| Set (items) | P0 26B ρ | 26B head ρ | Δ [95% CI] |
|---|---|---|---|
| Wikipedia toxicity (3,060) | −0.162 | +0.002 | **+0.164 [+0.118, +0.210]** |
| Humicroedit (1,869) | +0.061 | −0.032 | **−0.092 [−0.149, −0.030]** |
| Sarcasm corpus (752) | +0.179 | +0.231 | +0.052 [−0.024, +0.122] |
| Politeness (987) | −0.005 | +0.071 | +0.076 [−0.007, +0.157] |
| MultiNLI relation (4,823) | +0.321 | +0.365 | **+0.044 [+0.022, +0.065]** |
| MultiNLI support (4,823) | +0.223 | +0.347 | **+0.124 [+0.096, +0.152]** |

## Serving

Through `decisivelm.serve` on one RTX PRO 6000 Blackwell Max-Q with nothing else on it, at the server's 12,288-token
limit (`scripts/paper_v5c_26b/serve_measure.py`, 2026-11-05): 55.8 GiB (57,179 MiB) after loading and 67.0 GiB (68,593 MiB) at the peak (nvidia-smi), so `DECISIVELM_MEMORY_GB=72` (GiB); 473 ms p50 (p95 475 ms) over HTTP for a 1,185-token request with 20
blocks, and 5.7–5.8 s for requests of about 12,000 tokens; audio refused with 400.
