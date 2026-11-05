# General v5c, E4B head: development results against v3d and v5a

General v5c (`general_v5c_e4b_r8_seed0`, E4B, rank 8) is the first-release training mix. It is v3d's mix plus 10,000 VizWiz-VQA photos. It has no audio, no MHS train and no HelpSteer2-Preference (RELEASE.md change log, 2026-10-30).

**Training.** 215,433 training rows, 2 epochs at most (26,930 steps of 16 rows), on one RTX PRO 6000 shared with other work.
- Early stopping ended training at step 14,000. The bundle scored here is the one selected by validation calibration loss, at step 12,500.
- Training took about 15 h: 14,000 steps of 16 rows at the logged 0.24 s/row, validation passes included. The whole job, from 08:10 to 01:24 with loading, calibration and the three bundles' predictions, took 17 h. (An earlier version of this line said "about 00:10, roughly 16 h", an estimate.)

**The two comparisons:**
- *v3d*: the 0.x candidate. v5c is v3d's mix plus the photos.
- *v5a*: the earlier 1.0 dev candidate, `results/media_v1_general_v5a.md`. It is v4a's mix (v3d's plus MHS train and HelpSteer2-Preference) plus VizWiz, SONYC-UST and Emozionalmente.

**How it was scored.** `scripts/eval_dataset.py` on gold_test, 95% CIs from 1,000 item-level bootstrap resamples. Every head is raw, with no per-task step, unless marked "+task". None of these sets is in the frozen suite, which has not been touched.

## Photos (VizWiz-VQA, trained on)

P0+task and v3d+task are the zero-shot starting points from `results/media_v1_zero_shot_v3d.md`, each with task calibration fitted on the set's gold_calib rows.

| Question | Measure | P0+task | v3d+task | v5a | v5c | Δ v5c − v5a [95% CI] |
|---|---|---|---|---|---|---|
| answerable (2,567 photos) | AUROC | 0.788 | 0.828 | 0.922 | **0.933** | +0.010 [+0.005, +0.015] |
| answerable | ECE | 0.064 | 0.058 | 0.045 | 0.084 | |
| answer, 4 options (1,953) | accuracy | 0.432 | 0.434 | 0.699 | **0.711** | +0.012 [−0.009, +0.030] |
| answer, 4 options | ρ(top p, agreement) | +0.292 | +0.332 | +0.599 | **+0.633** | |
| answer, 4 options | tie AUROC | 0.588 | 0.574 | 0.695 | 0.701 | |
| answer, 4 options | ECE | 0.061 | 0.087 | 0.186 | 0.217 | |
| answer, yes/no (127) | accuracy | 0.776 | 0.732 | 0.784 | 0.819 | +0.034 [−0.041, +0.106] |

## Text, against v3d and v5a

Δ is v5c minus the named head on the primary measure: accuracy, Noul AUROC, or, for Score, MAE, where lower is better.

| Set | Question | Primary | v3d | v5a | v5c | Δ vs v3d [95% CI] | Δ vs v5a [95% CI] |
|---|---|---|---|---|---|---|---|
| wiki_toxicity (dev, held out) | toxicity, 5 levels | MAE | 0.482 | 0.487 | 0.462 | −0.021 [−0.027, −0.014] | −0.026 [−0.033, −0.019] |
| humicroedit (dev, held out) | funny, 4 levels | MAE | 0.566 | 0.486 | 0.567 | +0.001 [−0.003, +0.005] | +0.080 [+0.071, +0.089] |
| csc_sarcasm (dev, held out) | sarcastic, 6 levels | MAE | 1.131 | 0.999 | 1.070 | −0.061 [−0.074, −0.048] | +0.071 [+0.053, +0.089] |
| MultiNLI (dev) | relation, 3 options | accuracy | 0.818 | 0.846 | 0.810 | −0.008 [−0.015, −0.002] | −0.036 [−0.044, −0.029] |
| MultiNLI | follows | AUROC | 0.966 | 0.964 | 0.963 | −0.003 [−0.004, −0.001] | −0.001 [−0.003, +0.001] |
| MultiNLI | support, 3 levels | MAE | 0.305 | 0.269 | 0.333 | +0.028 [+0.024, +0.031] | +0.065 [+0.060, +0.070] |
| AG News | label, 4 | accuracy | 0.836 | 0.842 | 0.850 | +0.014 [+0.002, +0.026] | +0.008 [−0.006, +0.022] |
| SST-2 | label, 2 | accuracy | 0.918 | 0.924 | 0.924 | +0.006 [−0.008, +0.020] | +0.000 [−0.018, +0.018] |
| emotion (tweets) | emotion, 6 | accuracy | 0.609 | 0.570 | 0.605 | −0.004 [−0.015, +0.007] | +0.035 [+0.021, +0.048] |
| TweetTopic | topic, 6 | accuracy | 0.750 | 0.788 | 0.757 | +0.006 [−0.003, +0.017] | −0.031 [−0.044, −0.019] |
| financial topics | topic, 20 | accuracy | 0.590 | 0.591 | 0.588 | −0.002 [−0.009, +0.006] | −0.003 [−0.010, +0.004] |
| DailyDialog | emotion, 7 | accuracy | 0.614 | 0.601 | **0.645** | +0.032 [+0.025, +0.038] | +0.045 [+0.037, +0.051] |
| Banking77 (trained on) | label, 77 | accuracy | 0.830 | 0.830 | **0.856** | +0.026 [+0.004, +0.048] | +0.026 [+0.002, +0.052] |

### Review readiness on the multi-rater dev sets, with the prompted base model P0

ρ is between the head's top probability and per-item rater pair agreement. Above 0 means its confidence is lower where raters disagree.

| Set | P0 MAE | v5c MAE | P0 ρ | v3d ρ | v5a ρ | v5c ρ |
|---|---|---|---|---|---|---|
| wiki_toxicity | 0.355 | 0.462 | +0.057 | −0.125 | −0.114 | −0.010 |
| humicroedit | 0.733 | 0.567 | −0.115 | +0.124 | +0.201 | +0.113 |
| csc_sarcasm | 1.065 | 1.070 | −0.006 | +0.007 | +0.030 | +0.028 |
| MultiNLI relation | (acc 0.772) | (acc 0.810) | +0.261 | +0.340 | +0.375 | +0.323 |
| MultiNLI support | 0.311 | 0.333 | +0.169 | +0.228 | +0.299 | +0.200 |

## Seed 1: how much of this is training noise

A second E4B v5c run (`general_v5c_e4b_r8_seed1`) used the same mix and recipe, with only the seed changed. It stopped early at step 10,500; the bundle selected by validation calibration loss is from step 9,000. Seed 0 stopped at 14,000, selected at 12,500.

The bootstrap CIs elsewhere in this file cover the sampling of test items only. The gap between two seeds adds the randomness of training itself.

| Set | Measure | P0 | v3d | v5a | v5c seed 0 | v5c seed 1 | Δ s1 − s0 [95% CI] |
|---|---|---|---|---|---|---|---|
| VizWiz answerable | AUROC | 0.788 | 0.828* | 0.922 | 0.933 | 0.930 | −0.003 [−0.006, +0.001] |
| VizWiz answer, 4 options | accuracy | 0.432 | 0.434* | 0.699 | 0.711 | 0.689 | −0.022 [−0.041, −0.005] |
| VizWiz answer, 4 options | ρ(top p, agreement) | +0.292 | +0.332* | +0.599 | +0.633 | +0.619 | |
| wiki_toxicity | MAE | 0.355 | 0.482 | 0.487 | 0.462 | 0.456 | −0.005 [−0.010, −0.000] |
| wiki_toxicity | ρ(top p, agreement) | +0.057 | −0.125 | −0.114 | −0.010 | **−0.175** | |
| humicroedit | MAE | 0.733 | 0.566 | 0.486 | 0.567 | 0.507 | −0.059 [−0.065, −0.053] |
| humicroedit | ρ(top p, agreement) | −0.115 | +0.124 | +0.201 | +0.113 | +0.196 | |
| csc_sarcasm | MAE | 1.065 | 1.131 | 0.999 | 1.070 | 1.130 | +0.061 [+0.047, +0.073] |
| MultiNLI relation | accuracy | 0.772 | 0.818 | 0.846 | 0.810 | 0.825 | +0.015 [+0.009, +0.022] |
| MultiNLI support | MAE | 0.311 | 0.305 | 0.269 | 0.333 | 0.298 | −0.035 [−0.039, −0.032] |
| AG News | accuracy | 0.810 | 0.836 | 0.842 | 0.850 | 0.824 | −0.026 [−0.042, −0.012] |
| Banking77 | accuracy | 0.720 | 0.830 | 0.830 | 0.856 | 0.840 | −0.016 [−0.034, +0.000] |
| DailyDialog | accuracy | **0.680** | 0.614 | 0.601 | 0.645 | 0.655 | +0.009 [+0.004, +0.015] |
| emotion (tweets) | accuracy | 0.581 | 0.609 | 0.570 | 0.605 | 0.610 | +0.005 [−0.004, +0.014] |
| TweetTopic | accuracy | 0.779 | 0.750 | 0.788 | 0.757 | 0.781 | +0.024 [+0.011, +0.036] |
| DailyDialog | ECE | 0.225 | 0.125 | 0.137 | 0.137 | 0.287 | |
| emotion (tweets) | ECE | 0.319 | 0.105 | 0.067 | 0.093 | 0.241 | |
| TweetTopic | ECE | 0.114 | 0.148 | 0.197 | 0.151 | 0.305 | |

\*v3d with the per-task step fitted on VizWiz's gold_calib rows (it never trained on photos); P0's VizWiz column also has the step.

**On text, two seeds differ about as much as two mixes do.**
- humicroedit's MAE moves by 0.06 between seeds, against 0.08 between v5c and v5a.
- sarcasm moves by 0.06 between seeds.
- MultiNLI support moves by 0.035 between seeds, so the claim above that v5c was worse than v3d does not hold for seed 1.
- AG News and TweetTopic accuracy move by 2–3 points between seeds.
- Raw calibration varies even more: tweet emotion ECE is 0.09 for one seed and 0.24 for the other.

The photo results are stable across seeds.

## What it says

- **Photos are at least as good as v5a without the audio.** Answerability is slightly better: AUROC 0.933 against 0.922, a difference of +0.010 [+0.005, +0.015].
  - The four-option answer is 71% right against 70%, a difference within noise.
  - Its confidence follows answerer agreement a little more closely (ρ 0.63 against 0.60).
  - Raw calibration is worse (ECE 0.22 against 0.19 on the answer). The release bar judges calibration after the 100-row per-task step, so this is not yet a failure.
- **The two emotion-from-text losses of v5a are gone.** DailyDialog is the best of the three heads at 64.5%: +3.2 points over v3d, +4.5 over v5a. Tweet emotion is back to v3d's level.
  - This fits the guess that v5a's emotion losses came from Emozionalmente, the speech-emotion set.
  - It does not prove it, because v5c also lacks v4a's two text sources.
- **Correction after seed 1:** the "improved" in the next point is seed noise. Seed 1's ρ on wiki_toxicity is −0.175, worse than v3d's. Read every text difference between single runs in this file against the seed-to-seed gaps above. Also, P0 beats every head on DailyDialog (68% against 65%).
- **wiki_toxicity improved but still fails against P0.** It is the dev proxy for v3d's suite failure, MHS Score review readiness.
  - Its MAE (0.462) is better than v3d's and v5a's, but well behind P0's 0.355.
  - Its ρ is about zero (−0.010), where P0 is +0.057.
  - The Score middle-level problem is smaller than in v3d (−0.125), but still there. v5c may fail the same suite criterion v3d failed.
- **Humour, sarcasm and the MultiNLI levels sit at or below v3d, and below v5a.** v5a's advantage there came with v4a's text sources (MHS train, HelpSteer2-Preference). Those are deferred under the suite-task rule.
  - MultiNLI's support Score is slightly worse than v3d's (MAE +0.028) and than P0's (+0.022). Its ρ is still above P0's (0.20 against 0.17).
- **Elsewhere, text held or rose.** AG News +1.4 points and Banking77 +2.6 over v3d; the rest are within noise.
  - Raw calibration is worse on two benchmarks: financial topics ECE 0.265 (v3d 0.188) and Banking77 0.292 (v3d 0.251).
