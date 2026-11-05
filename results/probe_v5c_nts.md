# Probe: General v5c without the teacher's Score labels (for 0.2)

**Question.** v5c's Score confidence does not follow rater disagreement: wiki_toxicity ρ is −0.010 for seed 0 and −0.175 for seed 1, against +0.057 for the prompted base model P0. This is the dev proxy for the suite failure on MHS Score review readiness. Is the cause that 87% of v5c's Score targets come from the teacher, and the teacher avoids neutral middles?

**Bar, set before the run** (user, 2026-10-31). The probe's wiki_toxicity ρ must clearly beat both v5c seeds, and ideally P0, before it means anything.

**Answer: no.** Removing the teacher's Score labels left wiki_toxicity review readiness where it was, at ρ −0.003. It also cost 7–8 points on DailyDialog and 3–5 on MultiNLI relation. The teacher's avoided middles are not, on their own, the cause. Nothing changes for the first release.

## The run

`general_v5c_nts_e4b_r8_seed0` is v5c's mix with every teacher-labelled Score question removed from train and val (`scripts/drop_teacher_score.py`):
- 130,505 Score questions removed in train, about 87% of v5c's Score targets.
- Rows left with no targeted question were dropped: 1,524 train and 57 val.
- The remaining Score targets are human, mostly HelpSteer2's 60,000.
- gold_calib and gold_test are unchanged.

**Recipe:** the same as v5c (E4B, rank 8, 2 epochs at most, seed 0, `data/queues/general_v5c_nts.sh`).
- The first attempt crashed at step ~3,700 on a micro-batch whose rows had no targets. The fix is commit 45abf38, and the run was restarted 2026-11-01 18:39.
- Early stopping ended training at step 14,500, after 824 min at 0.21 s/row. The bundle scored is the one selected by validation calibration loss, at step 13,000.
- Its validation numbers cannot be compared with v5c's, because its val set also lost the teacher's Score questions.

**How it was scored.** `scripts/eval_dataset.py` on gold_test, 95% CIs from 1,000 item-level bootstrap resamples. Δ is the probe minus the named head. Every head is raw, with no per-task step. None of these sets is in the frozen suite.

**The yardstick is seed noise.** The CIs cover the sampling of test items only. The gap between v5c seeds 0 and 1 (`results/general_v5c_e4b.md`) is the yardstick for a single run.

## wiki_toxicity (3,060 items; the question the probe asked)

| | Probe | v5c seed 0 | v5c seed 1 | v3d | P0 | 26B teacher |
|---|---|---|---|---|---|---|
| ρ(top p, agreement) | −0.003 | −0.010 | −0.175 | −0.125 | **+0.057** | −0.162 |
| tie AUROC | 0.481 | 0.531 | 0.500 | 0.487 | 0.539 | 0.520 |
| MAE | 0.453 | 0.462 | 0.456 | 0.482 | 0.355 | **0.290** |
| Δ MAE, probe − head [95% CI] | | −0.009 [−0.019, +0.001] | −0.003 [−0.012, +0.006] | −0.029 [−0.035, −0.023] | +0.098 [+0.087, +0.110] | +0.162 [+0.149, +0.176] |
| Middle level picked where the raters' plurality is neutral (1,573 items) | 21% | 36% | 20% | 19% | 34% | 11% |
| Mean P(middle), all items (raters' mean share 0.495) | 0.218 | 0.240 | 0.185 | 0.194 | 0.217 | 0.174 |

- **ρ is at seed 0's level, not clearly better, and still below P0's.** Tie AUROC is the lowest of the six heads. The probe fails the bar.
- **The middle level does not come back.** The probe picks "neither" on 21% of the neutral-plurality items, inside the range of the two seeds (20–36%).
- **Picking the middle more often does not by itself fix ρ.** Seed 0 picks it as often as P0 does (36% against 34%) and still has a ρ of about zero.
- **The 26B teacher does avoid the middle most** (11%). That is what prompted the probe, but taking its labels out changed neither measure.
- The 60,000 human HelpSteer2 Score labels that remain do not teach this behaviour either.

## The other two multi-rater dev sets

| Set | Measure | Probe | v5c seed 0 | v5c seed 1 | v3d | P0 | 26B teacher |
|---|---|---|---|---|---|---|---|
| csc_sarcasm (6 levels) | MAE | 0.989 | 1.070 | 1.130 | 1.131 | 1.065 | **0.832** |
| csc_sarcasm | ρ(top p, agreement) | +0.117 | +0.028 | +0.008 | +0.007 | −0.006 | **+0.179** |
| humicroedit (4 levels) | MAE | 0.514 | 0.567 | **0.507** | 0.566 | 0.733 | 0.640 |
| humicroedit | ρ(top p, agreement) | +0.166 | +0.113 | **+0.196** | +0.124 | −0.115 | +0.061 |

- **Sarcasm improved by more than the seed gap.** The MAE gain is 0.08 [0.06, 0.10] over seed 0 and 0.14 [0.11, 0.17] over seed 1, against a seed-to-seed gap of 0.06. ρ rose from about 0.02 to 0.12.
- It is one seed. The probe is now ahead of P0 on sarcasm, but still behind the 26B teacher.
- **Humour is between the two seeds.**

## Everything else

| Set | Measure | Probe | v5c seed 0 | v5c seed 1 | P0 | Δ probe − s0 [95% CI] |
|---|---|---|---|---|---|---|
| VizWiz answerable | AUROC | 0.930 | 0.933 | 0.930 | 0.788 | −0.002 [−0.006, +0.002] |
| VizWiz answer, 4 options | accuracy | 0.690 | 0.711 | 0.689 | 0.432 | −0.020 [−0.039, −0.001] |
| VizWiz answer, 4 options | ρ(top p, agreement) | +0.636 | +0.633 | +0.619 | +0.292 | |
| MultiNLI relation | accuracy | 0.779 | 0.810 | 0.825 | 0.772 | **−0.031 [−0.040, −0.021]** |
| MultiNLI relation | ρ(top p, agreement) | +0.283 | +0.323 | +0.343 | +0.261 | |
| MultiNLI follows | AUROC | 0.959 | 0.963 | 0.966 | 0.936 | −0.004 [−0.006, −0.002] |
| MultiNLI support, 3 levels | MAE | 0.340 | 0.333 | 0.298 | 0.311 | +0.007 [+0.003, +0.011] |
| MultiNLI support | ρ(top p, agreement) | +0.178 | +0.200 | +0.221 | +0.169 | |
| AG News | accuracy | 0.840 | 0.850 | 0.824 | 0.810 | −0.010 [−0.022, −0.000] |
| SST-2 | accuracy | 0.922 | 0.924 | 0.920 | 0.934 | −0.002 [−0.014, +0.012] |
| emotion (tweets) | accuracy | 0.606 | 0.605 | 0.610 | 0.581 | +0.001 [−0.008, +0.011] |
| TweetTopic | accuracy | 0.754 | 0.757 | 0.781 | 0.779 | −0.003 [−0.014, +0.008] |
| financial topics | accuracy | 0.586 | 0.588 | 0.595 | 0.605 | −0.002 [−0.010, +0.006] |
| DailyDialog | accuracy | **0.571** | 0.645 | 0.655 | 0.680 | **−0.074 [−0.081, −0.067]** |
| Banking77 (trained on) | accuracy | 0.860 | 0.856 | 0.840 | 0.720 | +0.004 [−0.016, +0.024] |

**Two costs are larger than the seed gap.**

*DailyDialog (emotion in conversation, 7 options).*
- 57.1% against 64.5% and 65.5% for the seeds, and 68.0% for P0. That is 7–8 points, where the seed gap is 1 point.
- The likely source is the 2,029 teacher Score questions removed from GoEmotions rows, the mix's other emotion data. This is not tested.

*MultiNLI relation.*
- 77.9% against 81.0% and 82.5%, where the seed gap is 1.5 points.
- The probe's ρ on relation is also the lowest of the three.

Photos and the remaining benchmarks are within seed noise.

## What it says

- **The teacher's avoided middles do not explain the Score review-readiness failure.** Removing 87% of the Score targets, all of them teacher-labelled, left wiki_toxicity ρ and middle-level picks where they were. Training on the human Score labels alone does not produce confidence that follows disagreement.
- **The first release is unchanged.** Both heads train on v5c. If v5c fails MHS Score review readiness on the frozen suite, the release decision is the same one taken for v3d.
- **For 0.2:**
  - The two-sided human ratings (`two_sided_v1`) remain the main route to this failure.
  - The sarcasm gain is worth a second seed before anything is built on it.
  - Dropping teacher Score data wholesale is not an option, because of the DailyDialog and MultiNLI costs.
