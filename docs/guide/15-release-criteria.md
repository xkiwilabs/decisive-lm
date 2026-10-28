# 15. Release criteria

A general head is released only when it passes a fixed set of criteria, written down before any candidate was scored,
so the bar cannot move to fit a result. The question the criteria ask is simple: **is someone who uses the general
head better off than if they had prompted the base model it is built on?** This chapter explains the criteria and the
held-out data they are scored on. The authoritative text, with its change log, is [docs/RELEASE.md](../RELEASE.md).

## 15.1 Two bars: 0.x and 1.0

- **0.x (preview)**: the general head is *never worse* than the frozen backbone prompted (P0, chapter 12) on the
  frozen held-out suite, and it is calibrated well enough to set a review threshold on.
- **1.0**: the general head *beats* P0 on review readiness and passes the stricter calibration gate on the same suite.

The comparison is always with P0, because P0 is what a user would get from Gemma 4 E4B with no DecisiveLM at all. A
head that does not beat P0 adds a training step without adding value. The 26B teacher and task-specific fine-tunes
are reported beside it for context.

**Very good everywhere; excellent through fine-tuning.** The general head is meant to be very good at every task:
better than prompting the base model, close to a task-specific fine-tune on the tasks it was trained on, and as good
as that fine-tune at saying which answers to check. Excellence on one task is what fine-tuning is for (chapter 9). A
fine-tune spends all of its capacity on one task's labels and conventions, while the general head shares one adapter
across every task. The criteria gate on *very good*. *Exceptional* is a target that every model card reports but that
does not gate a release:

- within noise of the task-specific fine-tune on every training dataset's own test set;
- on the frozen suite, accuracy not significantly below the 26B teacher, and review readiness at least as good.

## 15.2 The frozen held-out suite

Gates are scored on four public, human-labelled datasets that no training, selection, tuning or calibration decision
ever sees. They are held out of every mix in `configs/holdout.yaml`, so the mix builder refuses any mix that contains
them. Each was chosen to test something different:

| Set | What it tests | Labels |
|---|---|---|
| **HelpSteer3**, preference (a fixed random 6,000 response pairs) | LLM judging beyond the HelpSteer2 seed: which of two responses is better, and by how much (one six-level Score), across general, STEM, code and multilingual conversations | Up to 3 annotators per pair |
| **Measuring Hate Speech**, evaluation portion (20% of the comments with ≥ 3 raters, by comment id) | Calibration against real human disagreement: nine survey items as Scores (0–4) and hate speech as a Noul | ≥ 3 raters per comment |
| **CUAD**, the 10 held-out clause types (Cap on Liability, Audit Rights, Change of Control, Non-Compete, Uncapped Liability, Covenant Not to Sue, Liquidated Damages, Non-Disparagement, Most Favored Nation, Source Code Escrow) | Unseen questions in a known domain: Nouls on contract sections of ≤ 10,000 characters | One expert label each |
| **MASSIVE**, the test split of 10 held-out languages (Polish, Urdu, Finnish, Georgian, Amharic, Swahili, Tamil, Korean, Vietnamese, Tagalog) | Zero-shot language transfer: one Choice over 60 intents | One label each |

The ten MASSIVE languages are spread over:

- six scripts, including Georgian, Ge'ez, Tamil and Hangul, which appear nowhere in training;
- resource levels from high to low;
- about half with a close relative left in training, and half without.

The criteria that need several raters per item (review readiness) apply to HelpSteer3 and to Measuring Hate Speech's
Scores. The others apply to every task.

**No overlap with trainable data.** `scripts/overlap_audit.py` compares the suite's text with every trainable row in
both directions, using word 8-grams. Overlapping suite rows are left out, and their ids are kept in
`configs/suite_exclusions.txt`; most are CUAD sections that share text with LEDGAR provisions. The generator also
filters every new synthetic row against the suite. The audit is rerun before a candidate is scored.

**Development sets are not gates.** The coding study and the public development benchmarks (chapter 14) were used to
choose between candidate heads, so they no longer test a candidate fairly. They are reported on the model card as
development results, labelled as such.

## 15.3 The criteria

A **task** is one suite set's questions of one type, so the suite has five tasks: HelpSteer3's Score, Measuring Hate
Speech's Scores, Measuring Hate Speech's Noul, CUAD's Nouls and MASSIVE's Choice. On each task the candidate is
compared with P0 using paired, item-level bootstrap 95% confidence intervals (1,000 resamples of the rows). All seven
criteria must hold.

**1. Review readiness** (tasks whose items have ≥ 3 raters). Does the head's confidence say which answers a person
should check? The terms are defined in chapter 11.

| Check | 0.x | 1.0 |
|---|---|---|
| ρ (Spearman rank correlation) between top probability and per-item human agreement | not significantly below P0 | above P0 |
| Tie AUROC: low confidence finds items with no unique human majority | not significantly below P0 | above P0 |
| Within-category ρ, in every consensus category with ≥ 25 items | > 0 | > 0 |
| Dynamic range of top probability across agreement levels | ≥ 0.2 | ≥ 0.2 |
| Errors caught by reviewing the 20% least confident answers | not significantly fewer than P0 | more than P0 |
| Accuracy gained by that review | above zero (CI excludes 0) | above zero |

"Not significantly below" means the confidence interval of the difference does not lie wholly below zero.
Within-category ρ matters because a signal that only knows which categories are hard is not item-level calibration.
Dynamic range matters because a confidence that runs from .986 to .998 can rank items but cannot carry a review
threshold.

**2. Calibration.** Does a stated 80% mean right about 80% of the time?

- Top-label ECE (expected calibration error, 15 bins) against hard human labels: ≤ 0.10 on every task (0.x); the
  upper bound of its bootstrap interval below 0.07 (1.0).
- Brier score no worse than before calibration.

**3. Accuracy.** The primary metric per type (Choice accuracy, Score mean absolute error, Noul AUROC) must be:

- 0.x: not significantly worse than P0 on any task;
- 1.0: better on most tasks and worse on none.

**4. Close to the specialists on the seeds.** On each seed dataset's own `gold_test`, the head is compared with a
reference: the previous release, or for the first release the specialist trained on that seed alone
(`configs/release_seeds.yaml`). With paired bootstrap intervals:

- accuracy: the primary metric is at most 0.03 below the reference (Choice accuracy, Noul AUROC), or at most 0.05
  levels above it (Score mean absolute error);
- calibration: top-label ECE ≤ 0.10, or not significantly above the reference's;
- review readiness, on seed tasks whose items have ≥ 3 raters: ρ, tie AUROC and the errors caught by a 20% review are
  not significantly below the reference.

The margin allows a small accuracy cost for sharing one adapter across tasks. It is about two standard errors on
these test sets, so a head that passes is close to the specialist, not merely not far from it. Calibration and review
readiness get no such allowance.

**5. Contract.**

- 100% conformance on every evaluation item: probabilities cover exactly the request's options and sum to 1, scores
  stay in range, and so on (chapter 4).
- The independence and order tests pass on the released bundle on the real backbone, at the tolerances recorded in
  the bundle (`scripts/check_bundle.py`, `scripts/check_invariance.py`).

**6. Provenance.**

- The release mix is R tier only, and its manifest is published.
- The teacher and the backbone pass the model-origin rule.
- The clean-room canary and scanner pass.
- No output of a third-party commercial decision service was used in any step (chapter 16).

**7. Snapshot audit.** The exported public tree passes the release audit: allowlisted paths
only, the test suite green, and no secrets, private paths, private datasets or restricted-tier results.

**8. A better starting point for fine-tunes** (gates 1.0; reported for 0.x). On two development tasks outside the
training mix, adapters are fine-tuned on 100 and on 300 labelled items, three seeds each, once from the general head
and once from the base model. Those started from the general head must be no worse at either size, on the primary
metric and on review readiness where it can be measured, and better at 100 items on at least one task. This is the
claim that the general head serves people who need excellence on their own task: they get there with fewer labels.

## 15.4 Scoring a candidate once

```bash
uv run python scripts/score_suite.py --candidate <bundle name> --calib-data data/mixes/<mix>.jsonl
```

`scripts/score_suite.py` scores criteria 1–4 and lists the commands for 5–7. With `--seeds-only` it scores criterion
4 alone, which is how development mixes are compared. Before it runs, three sets of
predictions must exist:

- the candidate's on each suite set (`scripts/predict_bundle.py`);
- P0's and the teacher's on each suite set (`scripts/label_teacher.py`, with `--teacher gemma4_e4b` for P0);
- P0's and the teacher's on the candidate mix's `gold_calib` rows.

P0 and the teacher are calibrated exactly the way the candidate is, on the same human-labelled `gold_calib` rows of the
candidate's mix, and then by the same per-task step on 100 rows of each suite set
([chapter 10.6](10-calibration.md)). Nothing else is fitted on the suite. A weaker calibration for the baselines would make the bar easier to
pass.

Other details of how it reads the criteria:

- **Hard labels.** A hard label is the raters' unique plurality: the single answer chosen most often. For a Noul it is
  a human share ≥ 0.5.
- **Consensus categories.** A consensus category is one question's consensus level.
- **Dynamic range.** It is taken over agreement levels with ≥ 25 items.
- **Several questions.** A metric over a task's several questions is the mean of the per-question values.

The report goes to `results/suite_<candidate>.md`, and the script refuses to overwrite an existing report. Each
candidate is scored on the suite **once**, after it is chosen. A candidate that fails is not tuned on the result and
rescored.

To check the scorer itself on development data, without touching the suite, pass other sets and an output path:

```bash
uv run python scripts/score_suite.py --candidate <bundle> --calib-data data/mixes/<mix>.jsonl \
    --sets bench_emotion bench_tweet_topic --n-boot 200 --out /tmp/dev_score.md
```

## 15.5 What every release publishes

- **A model card** ([docs/model-card.md](../model-card.md)): every number for the suite, the development sets and the
  seeds, beside P0, the 26B teacher, the specialist fine-tunes and, where they exist, published aggregate figures of
  other systems. Each criterion is marked pass or fail, *including where the model loses*.
- **Records of the build**: the mix manifest, the bundle's base revision, pinned library versions and calibration
  temperatures.
- **A shared tag** (`vX.Y`) for the public repository snapshot and the Hugging Face model revision.
- **The release's data** on Hugging Face (chapter 14).

Next: [16. Provenance, licences and responsible use](16-provenance-licences-and-responsible-use.md)
