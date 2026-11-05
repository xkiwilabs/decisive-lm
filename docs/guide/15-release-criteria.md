# 15. Release criteria

A general head is judged for release against a fixed set of criteria. They were first written down on 2026-10-17,
before any candidate was scored, and the version in force for the first release was fixed before its heads were
scored. Criteria 1 and 2 were revised after an earlier candidate (General v3c) failed on the held-out suite. The first
release's E4B head fails three checks, criterion 1 on one task and criteria 2 and 4 by small margins, and was released
by a decision made after that result was seen. The first release also carries a 26B A4B head, which is judged against
the prompted 26B A4B (a stronger bar, section 15.1). It fails five checks, listed in section 15.3, and was released as
a preview by a decision made after its result was seen. Its criterion 8 has not yet been measured. Each change is
recorded with its date and reason, and later releases are held to the criteria as written. The criteria test **whether
someone who uses the general head is better off than if they had prompted the base model it is built on**. This
chapter explains the criteria and the held-out data they are scored on. The authoritative text, with its change log,
is [docs/RELEASE.md](../RELEASE.md).

## 15.1 Two bars: 0.x and 1.0

- **0.x (preview)**: the general head is *never worse* than P0 on the frozen held-out suite (15.2), and it is
  calibrated well enough to set a review threshold on. P0 is the base model, Gemma 4 E4B with its weights unchanged
  (the *frozen* backbone), prompted to answer the same questions (chapter 12.8). For a head on Gemma 4 26B A4B, P0 is
  26B A4B prompted the same way, which is then also the teacher. That bar is higher: on the ten-language intents the
  prompted 26B A4B is right 73.1% of the time, against 65.8% for the prompted E4B.
- **1.0**: the general head *beats* P0 on review readiness (whether its confidence says which answers a person should
  check, chapter 11.3) and passes the stricter calibration gate on the same suite.

The comparison is always with P0, because P0 is what a user would get from Gemma 4 E4B with no DecisiveLM at all. A 0.x
preview need only be no worse than P0. A 1.0 release must beat P0 on review readiness, because review readiness is what
the head is for, and a user who needs only P0's can prompt Gemma 4 E4B with no head. The 26B teacher (Gemma 4 26B A4B,
the larger open model that wrote and labelled part of the training data, chapter 12) and task-specific fine-tunes are
reported beside it for context.

**What the general head is for.** The general head is meant to be very good at every task: better than prompting the
base model, close to a task-specific fine-tune on the tasks it was trained on, and as good as that fine-tune at saying
which answers to check. For excellence on one task, fine-tune on that task (chapter 9). A fine-tune spends all of its
capacity on one task's labels and conventions, while the general head shares one adapter across every task. The
criteria gate on *very good*. *Exceptional* is a higher target for the general head. Every DecisiveLM model card
reports it, but it does not gate a release. It means:

- within noise of the task-specific fine-tune (the confidence interval of the difference includes 0) on every
  training dataset's own test set;
- on the frozen suite, accuracy not significantly below the 26B teacher, and review readiness at least as good.

## 15.2 The frozen held-out suite

Gates are scored on four public, human-labelled datasets, fixed on 2026-10-20 before any candidate was scored (hence
*frozen*). Their rows are never trained on, and no mix, stopping rule or prompt is chosen by looking at them. One fit
is made on suite rows, the per-task calibration step of criterion 2, and on each set it uses the 100 rows with the
lowest hash of their ids. Those rows, and any rows that share a source item with them (a MASSIVE utterance in its other
languages), are left out of every score (15.4). The suite results of General v3c and v3d motivated later changes,
which were then chosen on development data, and the change log in [docs/RELEASE.md](../RELEASE.md) records each one.

The four sets are held out of every mix in `configs/holdout.yaml`, so the mix builder refuses any mix that contains
them. Each was chosen to test something different:

| Set | What it tests | Labels |
|---|---|---|
| **HelpSteer3**, preference (a fixed random 6,000 response pairs) | LLM judging beyond HelpSteer2, one of the six human-labelled training datasets (the *seeds*, chapter 13.2): which of two responses is better, and by how much (one six-level Score), across general, STEM, code and multilingual conversations | Up to 3 annotators per pair |
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
`configs/suite_exclusions.txt`. Most are CUAD sections that share text with LEDGAR, a training dataset of contract
provisions from US SEC filings (chapter 14.2). The generator also filters every new synthetic row against the suite.
The audit is rerun before a candidate is scored.

**Development sets are not gates.** The coding study and the public development benchmarks (chapter 14) were used to
choose between candidate heads, so they no longer test a candidate fairly. They are reported on the model card as
development results, labelled as such.

## 15.3 The criteria

A **task** is one suite set's questions of one type, so the suite has five tasks: HelpSteer3's Score, Measuring Hate
Speech's Scores, Measuring Hate Speech's Noul, CUAD's Nouls and MASSIVE's Choice. On each task the candidate is
compared with P0 using paired, item-level bootstrap 95% confidence intervals (1,000 resamples of the rows). Criteria 1–7
must all hold for a release, and criterion 8 must also hold for 1.0.

**1. Review readiness** (tasks whose items have ≥ 3 raters). Does the head's confidence say which answers a person
should check? The terms are defined below the table and in chapter 11.3.

| Check | 0.x | 1.0 |
|---|---|---|
| ρ (Spearman rank correlation) between top probability and per-item human agreement | not significantly below P0 | above P0 |
| Tie AUROC: low confidence finds items with no unique human majority | not significantly below P0 | above P0 |
| Within-category ρ, in every consensus category with ≥ 25 items | > 0, or at least P0's | > 0 |
| Dynamic range of top probability across agreement levels | ≥ 0.2, or at least P0's | ≥ 0.2 |
| Errors caught by reviewing the 20% least confident answers | not significantly fewer than P0 | more than P0 |
| Accuracy gained by that review | above zero (CI excludes 0) | above zero |

The terms, with the value a confidence unrelated to human agreement would give:

- **ρ**: the rank correlation between the top probability (the probability of the chosen answer) and per-item
  agreement (the share of rater pairs who gave the same answer). Chance is 0, and 1 is perfect.
- **Tie AUROC** (area under the ROC curve): the probability that an item on which raters tie, with no single
  most-chosen answer, gets lower confidence than an item on which they do not. Chance is 0.5, and 1 is perfect.
- **Consensus category**: the items of one question whose raters' single most-chosen answer is the same.
- **Dynamic range**: the highest minus the lowest mean top probability across levels of per-item agreement, counting
  only levels with ≥ 25 items (items rated by hundreds of people each have a level of their own).
- **Errors caught**: the share of the head's errors among its 20% least confident answers. A review of a random 20%
  catches 20% on average.

"Not significantly below" means the confidence interval of the difference does not lie wholly below zero.
Within-category ρ matters because a signal that only knows which categories are hard is not item-level calibration.
Dynamic range matters because a confidence that runs from .986 to .998 can rank items but cannot carry a review
threshold.

**2. Calibration.** Does a stated 80% mean right about 80% of the time?

Calibration is judged after the per-task step (15.2, 15.4), the step a user takes on about 100 labelled items of a
new task ([chapter 10.6](10-calibration.md)).

- Top-label ECE against hard human labels: ≤ 0.10 on every task (0.x); the upper bound of its bootstrap interval
  below 0.07 (1.0). ECE (expected calibration error, chapter 10.7) is the gap between the probability of the chosen
  answer and how often that answer is right, averaged over 15 bins of that probability weighted by their size, and 0
  is perfect. A hard label is the raters' unique plurality (the single answer chosen most often), or for a Noul a
  human share ≥ 0.5.
- Brier score after the step not significantly worse than before any calibration. The Brier score is the squared
  distance between the predicted probabilities and the hard label, summed over options (chapter 10.7).

**3. Accuracy.** The primary metric per type is Choice accuracy, Score MAE (mean absolute error, the mean distance in
levels between the expected level and the raters' mean level, so lower is better) or Noul AUROC (how well `p(yes)`
separates items people call yes from items they call no, where 0.5 is chance). It must be:

- 0.x: not significantly worse than P0 on any task;
- 1.0: better on most tasks and worse on none.

**4. Close to the specialists on the seeds.** On each seed dataset's own `gold_test`, the head is compared with a
reference: the previous release, or for the first release the specialist trained on that seed alone
(`configs/release_seeds.yaml`). A head on 26B A4B is compared with the same E4B specialists, so the bar is the same for
both backbones. With paired bootstrap intervals:

- accuracy: the primary metric is at most 0.03 below the reference (Choice accuracy, Noul AUROC), or at most 0.05
  levels above it (Score mean absolute error);
- calibration: top-label ECE ≤ 0.10, or not significantly above the reference's;
- review readiness, on seed tasks whose items have ≥ 3 raters: ρ, tie AUROC and the errors caught by a 20% review are
  not significantly below the reference.

The accuracy margin is applied to the point difference. It allows a small accuracy cost for sharing one adapter
across tasks, and it is about two standard errors on these test sets. Calibration and review readiness get no such
allowance.

**5. Contract.**

- 100% conformance on every evaluation item: probabilities cover exactly the request's options and sum to 1, scores
  stay in range, and so on (chapter 4).
- The independence and order tests pass on the released bundle on the real backbone, at the tolerances recorded in
  the bundle (`scripts/check_bundle.py`, `scripts/check_invariance.py`).

**6. Provenance.**

- The release mix is R tier only, meaning every source's licence allows commercial use and redistribution with no
  share-alike terms ([chapter 14.1](14-datasets.md)). Its manifest (sources, licences, row counts and SHA-256) is
  published.
- The teacher and the backbone pass the model-origin rule, which bars models developed by a Chinese organisation or
  derived from one ([chapter 16.2](16-provenance-licences-and-responsible-use.md)).
- The clean-room canary and scanner pass. They check that no answer from TypeSafe's Jev System One API has entered
  the data (chapter 16.1).
- No output of TypeSafe's Jev System One API, whose public request format DecisiveLM follows, was used in any step
  (chapter 16.1).

**7. Snapshot audit.** The public repository snapshot, the set of files copied from the private development repository
for a release, passes the release audit: allowlisted paths only, the test suite green, and no secrets, private paths,
private datasets or restricted-tier results.

**8. A better starting point for fine-tunes** (gates 1.0; reported for 0.x). On two development tasks outside the
training mix, adapters are fine-tuned on 100 and on 300 labelled items, three random seeds each, once from the general
head and once from the base model. Those started from the general head must be no worse at either size, on the primary
metric and on review readiness where it can be measured, and better at 100 items on at least one task. This tests the
claim that people who need excellence on their own task get there with fewer labels by starting from the general head.

**The 26B A4B head's exception.** Scored once against the prompted 26B A4B, the 26B A4B head fails five checks: the
accuracy on MASSIVE (criterion 3: 0.680 against 0.731), errors caught in a 20% review and the lowest within-category ρ
on Measuring Hate Speech's Scores, the lowest within-category ρ on HelpSteer3 (criterion 1), and Brier after the
per-task step on Measuring Hate Speech's Scores (criterion 2: 0.6403 against 0.6399 before). It passes criteria 4 to 7.
Its criterion 8 is scheduled for 7 to 8 November 2026. It was released as a preview, with the failures stated on its
model card, by a decision made after its result was seen, and the change log in [docs/RELEASE.md](../RELEASE.md)
records it. It is compared with the same E4B specialists in criterion 4 and is better than the E4B head on the main
measure of every suite task (model card of the 26B A4B head).

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
candidate's mix, and then by the same per-task step ([chapter 10.6](10-calibration.md)). The step fits one factor on
each method's Choice temperatures and one on its Score temperatures, using the method's own answers to the rows set
aside in 15.2, and leaves Noul answers as they are. Those rows are not scored. Calibration is scored after the step
and every other criterion before it, since the step does not change which answer is chosen. Nothing else is fitted on
the suite. A weaker calibration for the baselines would make the bar easier to pass.

A metric over a task's several questions is the mean of the per-question values.

The report goes to `results/suite_<candidate>.md`, and the script refuses to overwrite an existing report. Each
candidate is scored on the suite **once**, after it is chosen. A candidate that fails is not tuned on the result and
rescored.

To check the scorer itself on development data, without touching the suite, pass other sets and an output path:

```bash
uv run python scripts/score_suite.py --candidate <bundle> --calib-data data/mixes/<mix>.jsonl \
    --sets bench_emotion bench_tweet_topic --n-boot 200 --out /tmp/dev_score.md
```

## 15.5 What every release publishes

- **A model card** for each head ([E4B](../model-card.md), [26B A4B](../model-card-26b.md)): every number for the
  suite, the development sets and the seeds, beside P0, the 26B teacher, the specialist fine-tunes and, where they
  exist, published aggregate figures of other systems. Each criterion is marked pass or fail, *including where the
  model loses*.
- **Records of the build**: the mix manifest, the bundle's base revision, pinned library versions and calibration
  temperatures.
- **A shared tag** (`vX.Y`) for the public repository snapshot and the Hugging Face model revision.
- **The release's data** on Hugging Face (chapter 14).

Next: [16. Provenance, licences and responsible use](16-provenance-licences-and-responsible-use.md)
