# 10. Calibration

A model is **calibrated** when its stated confidence matches how often it is right. Of all the answers it gives with
probability 0.8, about 80% should agree with the human label. Calibration is what makes a confidence threshold mean
something, for example "send everything below 0.7 to a person". This chapter explains how DecisiveLM calibrates its
answers, where the calibration is stored, and how it is measured.

## 10.1 What is calibrated

The head gives every block one raw score, a **logit** ([chapter 6](06-how-it-works.md)). Calibration rescales those
logits before they become probabilities:

| Type | Readout |
|---|---|
| Choice | `p = softmax(logits / T)` over the question's options |
| Score | `p = softmax(logits / T)` over the levels; `score = Σ level × p(level)` |
| Noul | `p(yes) = sigmoid((logit + b_noul) / T)` |

`T` is a **temperature**:

- T > 1 spreads probability out, making the model less confident.
- T < 1 sharpens it.
- T = 1 leaves the head's output unchanged.

`b_noul` is a single bias shared by all Noul questions. It shifts how readily the model says yes.

Temperature scaling changes how confident the model is, not which answer it prefers: the top option of a Choice or
Score question is the same before and after. Calibration therefore cannot repair a model that ranks options badly. It
corrects confidence that is systematically too high or too low.

The `confidence` field of a Choice or Score answer is computed from the calibrated probabilities (by default
`(n·p_max − 1)/(n − 1)`, [chapter 4](04-requests-and-responses.md)), so it is calibrated too.

## 10.2 One temperature per question type and size

A question with 2 options and a question with 100 options need different temperatures, so temperatures are fitted
per **(question type, option-count bucket)**:

| Bucket | Options or levels |
|---|---|
| `2` | 2 |
| `3-5` | 3–5 |
| `6-15` | 6–15 |
| `16-63` | 16–63 |
| `64-255` | 64–255 |

Choice questions use every bucket. Score questions have 2–10 levels, so they use `2`, `3-5` and `6-15`. Noul has one
temperature of its own. The keys look like `choice:6-15`, `score:3-5` and `noul`.

Two rules handle buckets without enough data:

- **Small buckets.** A Choice or Score bucket with fewer than 100 calibration questions takes its type's pooled
  temperature, fitted on all of that type's questions together. A temperature fitted on a handful of easy questions
  tends to run to an extreme and make every answer in that bucket near-certain.
- **Unseen buckets.** The pooled temperature is stored as `choice:*` and `score:*`. A request whose option count falls
  in a bucket the calibration data never had uses it. With no calibration at all, the temperature is 1.

## 10.3 How the fit works

Calibration is fitted on the `gold_calib` split **only**, against **human** labels
(`decisivelm.calibrate.fit_calibration`):

- **Choice and Score.** For each bucket, the temperature is chosen from 400 values spaced evenly on a log scale between
  0.25 and 20. The chosen value minimises the soft negative log-likelihood against the raters' shares. The NLL is
  divided by `log n`, so a uniform guess scores 1 at every size. A temperature at either end of the range is reported
  with a warning, because it usually means the calibration rows are too easy or too few.
- **Noul.** The temperature and `b_noul` are fitted together by minimising binary cross-entropy against the raters'
  yes-shares.

**Why `gold_calib` and human labels.** The model is calibrated to *people*: a temperature fitted on the teacher's
labels would calibrate the student to the teacher instead. A temperature fitted on the test split would make the test
look better than it is. `gold_calib` and `gold_test` are therefore separate, fixed, human-labelled splits
([chapter 8](08-data-format.md)).

**Why soft targets.** The fit uses the full share of raters, not just the majority. An item where raters split 3–2
should get about 60% confidence, not 100%. Fitting on shares rewards a model that is unsure exactly where people are.

`scripts/train_p1.py` and `scripts/train_p3.py` fit calibration automatically after training and write calibrated
predictions. Fit it yourself on any set of question logits:

```python
from decisivelm.calibrate import fit_calibration, predictions, question_logits

qs = []   # QuestionLogits for every gold_calib question: question_logits(record, leaf_logits) per row
calib = fit_calibration(qs)                     # Calibration(temperatures={...}, b_noul=...)
rows = predictions(test_qs, calib)              # calibrated answers in the predictions format
```

The P0 baseline is calibrated more simply: `scripts/run_p0.py` fits a single temperature for its one question on
`gold_calib`.

## 10.4 Where calibration is stored

Calibration lives in the bundle's `calibration.json`, beside the training metadata:

```json
{
  "temperatures": {"choice:*": 1.2, "score:*": 0.9, "choice:2": 1.2, "choice:3-5": 1.0,
                   "choice:6-15": 1.4, "score:3-5": 0.9, "noul": 1.1},
  "b_noul": 0.05,
  "backbone": {"model_id": "google/gemma-4-E4B-it", "revision": "…"},
  "...": "training metadata"
}
```

`attach_bundle` reads `temperatures` and `b_noul` into a `decisivelm.head.Calibration`, and every answer the bundle
gives goes through `decisivelm.head.readout` with it. Training-time predictions and served answers therefore use the
same calibration.

## 10.5 Refitting without rerunning the model

Each stored prediction is the head's logits read out through the stored calibration, and the readout can be inverted
exactly:

- Choice and Score: `logit = T · log p`, up to a constant per question, which softmax ignores.
- Noul: `logit = T · logit(p) − b`.

`scripts/recalibrate.py` uses this to refit a run's calibration in seconds, with no GPU:

```bash
uv run python scripts/recalibrate.py --name P3_my_task_e4b_r8_seed0 [--suffix _recal]
```

It recovers the logits from `data/predictions/<name>.jsonl`, refits on the `gold_calib` rows of the data file the
bundle was trained on, and writes:

- `data/predictions/<name>_recal.jsonl`;
- `<name>_recal__<dataset>.jsonl` for every other prediction file of the run (for example from `predict_bundle.py`);
- `data/predictions/<name>_recal.calibration.json`, holding the new temperatures and bias.

The bundle itself is not changed. To serve with the refitted calibration, copy `temperatures` and `b_noul` from the
new file into the bundle's `calibration.json`. `decisivelm.calibrate.invert` is the inverse readout, if you need it
elsewhere.

## 10.6 The general head on a new task: calibrate on about 100 items

The temperatures above are fitted by soft NLL against rater shares, on the `gold_calib` rows of the tasks the head was
trained on. On a task the head has not seen, that is not enough:

- **Shares are not the majority.** Calibration is judged against the human majority answer (section 10.7). When
  raters split 60/40, a head that matches the shares says 0.6, yet its answer is the majority answer. On items like
  these a share-matching head looks underconfident.
- **The head hedges more on tasks it has not seen.** The released head (General v3d) is underconfident zero-shot on
  most new tasks: its ECE is 0.05–0.25 on seven public benchmarks it never trained on, and 0.26 on the coding study.
- **But not on every task.** On the release suite's 60 intents in ten languages it was not trained on, it and the
  base model are *over*confident.

**So, on a task of your own, calibrate on about 100 labelled items before you set a review threshold.** Label about
100 items, put them in `gold_calib`, predict them with the bundle, and fit two numbers on them: one factor on all the
Choice temperatures and one on all the Score temperatures (Noul calibration is unchanged).

```bash
uv run python scripts/predict_bundle.py --bundle bundles/<head> --data data/processed/<your stem>.jsonl \
    --splits gold_calib gold_test
uv run python scripts/recalibrate.py --name <head> --new-task <your stem> --suffix _mine
```

`recalibrate.py` reads the bundle's predictions on your task (`data/predictions/<head>__<your stem>.jsonl`), fits the
two factors by NLL of the human majority answer (ties left out), and writes a new bundle `bundles/<head>_mine/`. Its
`calibration.json` records the factors, the task and the temperatures it started from. With 100 items this brings the
released head's ECE to 0.09 or below on all seven public benchmarks and to 0.08 or below on every task of the release
suite. The release criteria measure calibration after this step (docs/RELEASE.md, criterion 2).

**Why not one correction for every new task?** An earlier candidate (General v3c) shipped with one: the same two
factors, fitted once on a human-labelled task held out of the mix (MultiNLI) and applied everywhere. It brought every
public benchmark's ECE to 0.09 or below. It was dropped for two reasons. Errors on new tasks go both ways, so one
factor cannot fix both: the English-fitted factor made the overconfident multilingual task worse, and on SIB-200 (the
same sentences in 195 languages) the head was underconfident outside English. And it cost calibration on the tasks
the head was trained on: with it, v3d's LEDGAR ECE was 0.108 against its fine-tune's 0.032; without it, 0.043.
MultiNLI stays held out of the general mix (`configs/holdout.yaml`) as a development check of how the head behaves on
a task it has not seen.

### Task calibration: how your raters use the scale

Temperatures set how spread out the answers are. They cannot fix *where* the probability goes. Raters have habits on
a scale that a model cannot know in advance: on Wikipedia Talk toxicity (about 10 raters a comment, five levels from
"very toxic" to "very healthy"), raters gave 2.6% of their votes to "very toxic" and 1.6% to "very healthy". The base
model and the general heads put 9–21% there, and were most confident on exactly those comments, where raters split.
Their confidence then ran *against* rater agreement.

Task calibration fits, for one question, a scale on its temperature and an offset per option or level (for a Noul, on
its logit), on that question's labelled `gold_calib` rows (`decisivelm.calibrate.fit_task_calibration`; a question
needs at least 100). The calibration is keyed by the question's wording and options, never its id, and Choice
options are sorted first, so reordering them finds the same calibration. On Wikipedia toxicity's 2,040 calibration rows:

| Method (gold_test, 3,060 comments) | MAE | ρ(top p, agreement) |
|---|---|---|
| P0, temperatures only | 0.355 | +0.06 |
| General v4a, temperatures only | 0.381 | −0.09 |
| P0 + task calibration | 0.277 | +0.29 |
| General v4a + task calibration | **0.225** | **+0.30** |

For comparison, half of each comment's raters predict the other half's agreement at ρ 0.17. Use task calibration on a
task of your own when you have a few hundred labelled items, and give every method you compare the same step:

```bash
uv run python scripts/task_calibrate.py --data data/processed/<your stem>.jsonl \
    --predictions data/predictions/<head>__<your stem>.jsonl data/predictions/teacher_<your stem>__gemma4_e4b.jsonl
```

The offsets carry your raters' base rates for this question. They belong to the task, never to the general head.

## 10.7 Measuring calibration

Calibration is measured on `gold_test`, against **hard** human labels:

- Choice and Score: the raters' unique plurality. Items where raters tie are left out.
- Noul: yes when the share is at least 0.5.

| Metric | Meaning | Good |
|---|---|---|
| **ECE** (expected calibration error) | Answers are grouped into 15 equal-width bins by top probability. In each bin, the gap between mean confidence and accuracy is measured, and the gaps are averaged weighted by bin size. | Low; 0 is perfect |
| **Brier score** | Mean squared distance between the predicted distribution and the one-hot label, summed over options. It rewards being both right and appropriately confident. | Low |
| **Soft NLL** | Cross-entropy against the raters' shares, divided by `log n` (uniform = 1). | Low |

ECE on a few hundred items is noisy, so reports give a bootstrap confidence interval for it. Calibration should also
lower the Brier score: if Brier is worse after calibration than before, the calibration rows do not resemble the test
rows. [Chapter 11](11-evaluation.md) covers every metric in the reports.

## 10.8 Limits

- **A model cannot be better calibrated than its targets.** If the training targets are overconfident, for instance
  majority votes or an uncalibrated teacher, the head learns sharp logits. A temperature can soften them, but it
  cannot restore the item-by-item differences in human agreement the targets never carried. This is why training uses
  rater shares, and why teacher labels are themselves calibrated on human data before they become targets
  ([chapter 12](12-teacher-labelling.md)).
- **Calibration transfers imperfectly.** A temperature fitted on one kind of data may not hold on another. The general
  head's temperatures are fitted on a mix of human-labelled tasks and scaled for new tasks on one held-out task
  (section 10.6), whose Score questions are a 3-level view of a Choice; Score tasks with more levels may differ.
  For a task of your own,
  measure ECE on your own `gold_test`, or fine-tune and calibrate on your own labels ([chapter 9](09-fine-tuning-for-your-task.md)).
- **One temperature per bucket is a global correction.** It cannot make the model more confident on easy items and
  less confident on hard ones. That item-level behaviour has to come from training; it is what the review-readiness
  metrics measure.

Next: [11. Evaluation](11-evaluation.md)
