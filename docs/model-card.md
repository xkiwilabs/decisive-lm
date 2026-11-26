---
license: apache-2.0
base_model: google/gemma-4-E4B-it
library_name: decisivelm
language:
  - en
pipeline_tag: text-classification
tags:
  - decision-model
  - classification
  - calibration
  - zero-shot-classification
  - lora
  - gemma4
---

# DecisiveLM E4B general head 0.1 (preview)

A general-purpose decision head for Google's Gemma 4 E4B (a small model with about 4B effective parameters, the "E"
standing for effective). The head is about 20M trained parameters (LoRA adapters, which are small trained corrections
added to the model's layers, plus embeddings for three reserved tokens and a scorer network, see Summary) used with
the base model, whose other weights are not retrained. Given a state (text, JSON or a photo with text) and a set of
typed questions, it answers every question in one forward pass (one run of the model over the input, with no text
generated), with a probability distribution over the options you supply. These probabilities are calibrated on the
tasks the head was trained on. On a new task they are usually underconfident until calibrated on about 100 labelled
items of that task (below). The head answers questions it was not trained on, without examples. On the held-out suite
below (unseen contract clause types, comparisons of chatbot responses, hate speech ratings, and intents in ten
languages it was not trained on), it is never significantly less accurate than the base model prompted with the same
questions, and it is more accurate on two of the five tasks (the five are the four sets above, with Measuring Hate
Speech asked twice, as nine Scores and as one Noul). Its confidence is built to say which answers a person should
check. On two-sided social rating scales it has not been trained on, it does this worse than the prompted base model
(below).

- **Code**: [github.com/xkiwilabs/decisive-lm](https://github.com/xkiwilabs/decisive-lm) (Apache-2.0)
- **Documentation**: [the guide](https://github.com/xkiwilabs/decisive-lm/tree/main/docs/guide), starting with the
  [quickstart](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/03-quickstart.md)
- **Base model**: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) at revision
  `ee0ef6023621cff504d758262d4e04895a5af4a2`, which this head requires

> **Known limitations of this preview.** It fails three checks of the
> [release criteria](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), and was released with them
> stated (Evaluation, below). They are counted per task: the first fails on two measures of one task (ρ and tie
> AUROC), counted once. Counted per measure, as the 26B head's card counts, it is four:
>
> 1. **Two-sided social rating scales it has not been trained on.** On scales such as "strongly respectful … neither …
>    strongly disrespectful", its confidence follows how extreme the text is rather than how much people agree. On
>    Measuring Hate Speech's survey items it is a worse guide to which answers to check than the prompted base model.
>    This is the main limitation. On such a task, use task calibration (a scale and an offset for each level, fitted
>    on about 100 of your own ratings, guide §10.6) or fine-tune on them. On a development set this made its
>    confidence follow rater agreement more closely than the prompted base model's under the same step.
> 2. **Yes/no calibration on that survey.** The head's general calibration makes its yes/no answers there slightly
>    worse (Brier score, the mean squared error of its probabilities against the human answer where lower is better,
>    0.154 against 0.147 uncalibrated). It is still better calibrated there than the prompted base model (0.209).
> 3. **Calibration on HelpSteer2**, one of its own training sets. Its calibration error is 0.123, slightly but
>    significantly above the task-specific fine-tune's 0.105. Its accuracy and review readiness there match the
>    fine-tune's.
>
> See [Limitations](#limitations-and-biases).
>
> **Calibrate before you set a review threshold.** Zero-shot, the head is usually underconfident on a new task.
> Fitting the new-task factor (one factor on the Choice temperatures and one on the Score temperatures) on about 100
> labelled items of your task brought its expected calibration error to 0.05 or below on every task of the held-out
> suite ([guide §10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)).

## Summary

| | |
|---|---|
| Question types | Choice (2–255 unordered options), Score (2–10 ordered levels), Noul (probability a statement is true) |
| Input | text or JSON state up to 8,192 tokens (longer states are shortened head and tail), optionally with a photo |
| Output | a probability per option or level, the chosen option or expected level, a confidence; for Noul one probability |
| Independence | each question, and each option, is scored on its own, so by design adding or removing questions or reordering options does not change other answers. On the released head such changes move no probability by more than 0.03 (rounding in bf16, 16-bit floating point, criterion 5 below) |
| Calibration | temperatures per question type and number of options (a temperature divides the scores before the softmax, so above 1 it spreads probability out and below 1 it sharpens it), and a Noul bias, fitted on human-labelled questions of the training mix; on a new task, calibrate on about 100 of your own labelled items |
| Size | 17.4M trained parameters inside the backbone (LoRA adapters added to its projection layers, and the embedding rows of the three query tokens) and a 2.6M-parameter scorer. The backbone's other weights are frozen (not trained) |

## Intended use

The head is for decisions that a person would otherwise make item by item, where most items are easy and a few need a
second look. You ask your questions, it answers every item, and its confidence says which answers to check.

### Use cases

Each row names what was measured on that kind of task and what it was compared against: the prompted E4B without
the head (P0), a task-specific fine-tune, or human coders. "Trained on" means the dataset's training portion is in
the head's training mix, and the figure is on its held-out test portion. Every other set is zero-shot (no training
examples of it).

| Use case | For example | What was measured |
|---|---|---|
| Routing and intent detection | a Choice "Which team should handle this message?" over billing, technical and sales, or over a list of up to 255 intents | Banking77, 77 banking intents (trained on): 90.9% right, against 90.2% for a fine-tune on Banking77 alone. CLINC150, 150 intents (trained on): 94.3% against 92.0%. Intents in ten languages it was not trained on (MASSIVE): 67.2% against P0's 65.8% |
| Coding open-ended answers, as a first coder | a Choice of eight codes for a free-text survey answer, with the least confident answers sent to a human coder | a coding study (792 answers, four human coders), zero-shot: agreement with the coders κ 0.675, against 0.739 between the coders themselves and 0.649 for P0. Its least confident 20% held 60.8% of its errors, against 42.2% for P0's least confident 20%. Fine-tuned on 100 coded answers: 91.9% right, against 87.7% for the same fine-tune started from the base model |
| Moderation and yes/no screening | a Noul "The comment contains a personal attack" | Civil Comments (trained on): mean AUROC 0.979, against 0.982 for a fine-tune on it alone. The hate-speech survey's yes/no question (not trained on): AUROC 0.852, against P0's 0.855 |
| Contract and document review | a Noul "This clause caps the supplier's liability", or a Choice over 100 provision types | CUAD, ten clause types it was not trained on: AUROC 0.947, against P0's 0.945. LEDGAR, 100 provision types (trained on): 83.6% right, against 82.1% for a fine-tune on it alone |
| Rating against a rubric | a Score "How helpful is this reply?" on five levels | HelpSteer2 (trained on): mean MAE 0.466, against 0.468 for a fine-tune on it alone. HelpSteer3, comparing two chatbot replies (not trained on): MAE 1.023, against P0's 1.292. On HelpSteer3 neither model's confidence says which answers to check, so review a random sample there |
| Topic, sentiment and emotion tagging | a Choice over topics, or a Choice of emotion | seven public benchmarks, zero-shot except Banking77: more accurate than P0 on three, level on one, less accurate on three. GoEmotions (trained on): 74.3% right, against 76.0% for a fine-tune on it alone |
| Whether one text supports another | a Choice "Does the second sentence follow from the first?" over follows, contradicts and neither | MultiNLI, five annotators per pair (not trained on): ρ between its confidence and annotator agreement 0.323, against P0's 0.261. Its least confident 20% held 47.2% of its errors, against 39.2% for P0 |
| Questions about photos | a Noul "The photo shows enough to answer the question", or a Choice between a few answers | VizWiz-VQA, photos taken by blind and low-vision people (trained on; held-out photos): AUROC 0.933 against P0's 0.788, and 71.1% right on a four-option answer against 43.2%. CIFAR-10H, objects in small photos (not trained on): 90.9% right against P0's 87.5%, but calibrate first (its raw ECE is 0.44) |
| A starting point for your own fine-tune | 100 to 300 of your own labelled items (guide chapter 9) | fine-tunes on 100 items started from this head beat the same fine-tunes started from the base model: 91.9% against 87.7% on the coding study, and 86.1% against 84.3% on MultiNLI, and their confidence tracks rater agreement more closely |

It is fast: about 0.10 s for a request with four questions of five options each, against 0.6–0.8 s for P0
(Hardware, below).

Not a good fit without your own ratings: two-sided social rating scales ("strongly respectful … neither … strongly
disrespectful") that it has not been trained on. There its confidence is a worse guide to which answers to check than
P0's (limitation 1, above). Use task calibration or a fine-tune on about 100–300 of your own ratings.

### How the confidence is meant to be used

1. Label about 100 items of your task yourself.
2. Calibrate the head on them ([guide
   §10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)).
3. Check its accuracy on items you labelled but did not calibrate on, and choose how many answers people will review.
4. Send its least confident answers to people, and spot-check a random few of the rest.

Guide [chapter 7](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/07-using-the-general-head.md) walks
through these steps.

## Out-of-scope use

- Unreviewed decisions that affect people: clinical, credit, hiring, housing, legal, policing or benefits decisions.
  Such uses need your own labelled evaluation, checks of errors across the groups affected, and a person in the loop.
- Free-text generation, explanation or summarisation.
- Choice questions with more than 255 options.
- Audio: this head was not trained on audio, and has not been evaluated on it.
- Images beyond photo questions without calibration on your task, because it was trained on one photo task (VizWiz-VQA).
  Other kinds of image are zero-shot, and its probabilities on them need task calibration (below).
- Questions whose answer is not in the state.

## How to use

```bash
git clone https://github.com/xkiwilabs/decisive-lm.git && cd decisive-lm && uv sync
uv run hf download xkiwilabs/decisivelm-e4b-general --local-dir bundles/decisivelm-e4b-general
```

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/decisivelm-e4b-general")
response = dlm.system_one({
    "state": "Our API started returning 500s 20 minutes ago and we can't process any orders.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this message?",
                       "criteria": {"billing": "Payment, invoice or subscription problems",
                                    "technical": "Bugs, outages or integration problems",
                                    "sales": "Pricing, plans or account questions"}},
        "is_urgent": {"type": "noul", "instructions": "The message reports a problem that needs action within the hour"},
    },
})
print(response["answers"])
```

The head directory holds `adapter.pt` (LoRA weights and query-token embeddings), `scorer.pt` (the scorer network) and
`calibration.json` (temperatures, Noul bias, backbone revision and training settings).

### Hardware

| | Running the head | Fine-tuning it on your task |
|---|---|---|
| Hardware | Linux or macOS, with an NVIDIA GPU (CUDA) or Apple silicon (MPS) | the same |
| Memory | a 24 GB GPU is enough. Predicting the coding study's 792 responses peaked at 16.0 GiB with 2,048 tokens per forward pass and at 17.8 GiB with 8,192. One request at the server's default limit peaks at about 18.6 GiB (a state is cut at 8,192 tokens, and the whole request may be up to 12,288) | a 24 GB GPU is enough. A 300-item fine-tune peaked at 21.2 GiB with the default micro-batches of 8,192 tokens, at 18.5 GiB with 4,096 and at 16.7 GiB with 2,048, with gradient checkpointing (on by default). It took 15 to 25 minutes on one RTX PRO 6000 that was shared with other work |
| Time | 0.10 s (102 ms p50) for a request of about 1,240 tokens with 20 options (four questions of five options each), batch 1, bf16, on one RTX PRO 6000 Blackwell Max-Q. The prompted E4B, which reads each question once per option order, took 613–803 ms for the same request on vLLM (`results/p0_latency.md`) | 15 to 25 minutes for a 300-item fine-tune (Memory, left) |
| Disk | about 15 GB for Gemma 4 E4B and about 1 GB for the Python environment | the same, plus your data |

The peaks are per process, as `nvidia-smi` reports them, so they include the CUDA context. A 16 GB GPU is not enough
at these settings. The bf16 weights take about 15 GiB, so a smaller micro-batch saves only a few GiB.

The CPU runs the test suite, not the model. On a Mac with large unified memory, fine-tune (`scripts/train_p3.py`,
guide chapter 9) with `--no-checkpointing --max-tokens 2048`. A training row then takes about 0.14 s for a short
message and about 2 s for a longer row, such as a free-text answer coded into one of eight categories. The full
requirements are in guide [chapter
2](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/02-installation.md), and settings for a GPU shared
with other work are in section 9.5 of [chapter
9](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/09-fine-tuning-for-your-task.md).

## Training data

The training mix is release tier only: every source carries a licence that allows training and redistribution of the
result (guide chapter 16). The *teacher* is Gemma 4 26B A4B, a larger open model that writes question sets and states
and labels rows that have no human answers (guide chapter 12). The mix is General v5c: 215,433 training requests
(995,083 questions: 235,660 Choice, 190,571 Score, 568,852 Noul) of four kinds:

| Kind | Sources | Labels |
|---|---|---|
| Human-labelled public datasets | Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), HelpSteer2 (CC BY 4.0), Civil Comments (CC0 1.0), GoEmotions (Apache-2.0), LEDGAR (CC BY 4.0) | the share of human raters who chose each label |
| Photos with questions | VizWiz-VQA (CC BY 4.0): 10,000 photos taken by blind and low-vision people with a question each, asked whether the photo can answer it and, where it can, which of the offered answers (usually four) is right | the share of the ten answerers who gave each answer |
| Rewritten questions on the same states | subsets of the intent sets' options, options removed with a self-contained "other", intents asked as yes/no statements | carried over from the human labels |
| Teacher-written question sets and states | question sets in many settings (support, finance, moderation, documents, qualitative coding of free-text answers and more), with states written to be easy, borderline, ambiguous, irrelevant or open-ended, and plainly worded Scores with a neutral middle; and new question sets about real states of the public datasets | Gemma 4 26B A4B's label probabilities, averaged over option orderings and calibrated on human labels |

Each release publishes its mix manifest: every source, its licence, the row counts per split and the sha256 of the mix.
The public datasets are linked rather than re-hosted. The teacher-written rows and their labels are published under CC
BY 4.0 in the companion dataset repository on Hugging Face (`decisive-lm-data`, tagged with the release version, guide
§14.7). A row whose state is a public dataset's item is published without that state, which is rebuilt from its source
by id. The 1,442 such rows whose questions quote that item are not published.
Details:
[guide chapter 13](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/13-how-the-general-head-was-trained.md)
and [chapter 14](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/14-datasets.md).

Splits are assigned per source item and question family before any rewriting, so no test item or its rewrites reach
training. The frozen held-out suite (below), four test sets fixed before any candidate was scored, is listed in
`configs/holdout.yaml`, and its text was checked for overlap with every trainable row.

## Training procedure

| Setting | Value |
|---|---|
| Backbone | Gemma 4 E4B, frozen, bf16 |
| Adapter | LoRA rank 8, α 16, dropout 0.05, on the decoder's attention (q, k, v, o) and MLP (gate, up, down) projections only |
| Query tokens | three reserved token ids (Choice, Score, Noul), their embedding rows trained |
| Scorer | per-dimension standardisation (fitted on 2,000 training rows) → LayerNorm → Linear(2560, 1024) → GELU → Linear(1024, 1) |
| Readout | softmax over a question's options or levels (Choice, Score); sigmoid for Noul |
| Losses | soft cross-entropy against label shares divided by log n (Choice); the same plus 0.5 × EMD² (EMD is the earth mover's distance) between the predicted and target cumulative distributions (Score); binary cross-entropy against the share (Noul) |
| Optimiser | learning rate 3e-5 for LoRA, 1e-4 for the head and query rows; 10% warm-up; 16 rows per step; packed micro-batches of up to 4,096 tokens with gradient checkpointing |
| Selection | validation every 500 steps on 1,500 held-back rows; the checkpoint with the lowest calibrated validation loss is kept, and training stops after three checks without improvement (this release: step 12,500, stopped at 14,000 of at most 26,930, about 15 hours on one RTX PRO 6000) |
| Calibration | one temperature per question type and option-count bucket (2, 3–5, 6–15, 16–63, 64–255), with buckets of fewer than 100 questions sharing their type's temperature, and a Noul temperature and bias, fitted on the human-labelled `gold_calib` rows of the mix. No global correction for new tasks: a scale fitted on one held-out task made the head overconfident on its training tasks and could not fix errors that go both ways on new ones, so users calibrate on their own task instead (guide §10.6) |

The exact settings of each release, including the number of epochs and the selected step, are recorded in its
`calibration.json`.

## Evaluation

The head is judged first on **review readiness**, whether its confidence says which answers to check, then on
calibration and accuracy. It is always compared with Gemma 4 E4B without the head, prompted to answer the same questions
(P0), and with the Gemma 4 26B A4B teacher. The
[release criteria](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md) were first written down before
any candidate was scored, and those in force for this head were fixed before it was scored. Criteria 1 and 2 were
revised after an earlier candidate's suite result was seen, and this head was released by a decision made after its own
result was seen (below). RELEASE.md records each change with its date and reason.

The measures, defined in guide chapters
[10](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md) and
[11](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/11-evaluation.md):

- **ρ(confidence, agreement)**: the Spearman rank correlation between the head's top probability and per-item human
  agreement (the share of rater pairs who gave the same answer). Chance is 0.
- **Tie AUROC**: the probability that an item on which raters tie (no single most-chosen answer) gets lower confidence
  than an item on which they do not. Chance is 0.5, and 1 is perfect.
- **Errors caught in a 20% review**: the share of the head's errors among its 20% least confident answers. A review of
  a random 20% catches 20% on average.
- **ECE** (expected calibration error): the gap between the probability of the chosen answer and how often it matches
  the raters' most-chosen answer, averaged over 15 bins of that probability. 0 is perfect.
- **MAE** (mean absolute error, for Scores): the mean distance, in scale levels, between the head's expected level and
  the raters' mean level. Lower is better.
- **AUROC** (for Nouls): how well the probability of yes separates items people call yes from items they call no.
  Chance is 0.5.
- **κ** (Cohen's kappa): agreement with a human coder, corrected for agreement by chance. Chance is 0, and 1 is
  perfect.

### Frozen held-out suite

Four public, human-labelled sets, fixed before any candidate was scored and never trained on. No mix, stopping rule or
prompt is chosen by looking at them. The one fit made on them is the new-task factor, the step a user takes on a new
task (the release criteria's per-task step). It fits one factor on the Choice temperatures and one on the Score
temperatures, for each method (the head, P0 and the teacher alike), on the 100 rows of each set with the lowest hash
of their ids. Those rows, and any rows that share a source item with them, are left out of every score below. Each
candidate is scored on the suite once. ECE is shown zero-shot and after that step. Full report:
`results/suite_general_v5c_e4b_r8_seed0.md`.

| Task | What it tests | ρ(confidence, agreement) (P0) | tie AUROC (P0) | errors caught in a 20% review (P0) | ECE zero-shot / after 100 rows | primary metric | P0 primary metric |
|---|---|---|---|---|---|---|---|
| HelpSteer3 preference (Score, 6 levels) | judging chatbot responses beyond the training set | 0.061 (0.043) | 0.507 (0.518) | 21.7% (21.0%) | 0.037 / 0.027 | **1.023** (MAE) | 1.292 |
| Measuring Hate Speech (9 Scores) | calibration against real rater disagreement | **0.078 (0.201)** | **0.533 (0.578)** | 24.5% (23.8%) | 0.058 / 0.046 | 0.653 (MAE) | 0.657 |
| Measuring Hate Speech (Noul) | the same, as a yes/no judgement | | | | 0.047 / 0.047 | 0.852 (AUROC) | 0.855 |
| CUAD, 10 held-out clause types (Noul) | unseen questions in a known domain | | | | 0.024 / 0.024 | 0.947 (AUROC) | 0.945 |
| MASSIVE, 10 held-out languages (Choice, 60 intents) | zero-shot language transfer | | | | 0.094 / 0.018 | **0.672** (accuracy) | 0.658 |

Bold: significantly different from P0 (paired 95% bootstrap).
- **Measuring Hate Speech's Scores.** P0 is significantly better on ρ and tie AUROC, so this head fails criterion 1
  there. The share of errors caught in a review is not significantly different.
- **HelpSteer3.** The head's and P0's review measures are all close to chance: ρ 0.061 and 0.043 (chance 0), tie
  AUROC 0.507 and 0.518 (chance 0.5), and 21.7% and 21.0% of errors caught (chance 20%). Neither confidence is a
  useful guide to which HelpSteer3 answers to check. Criterion 1 passes there only because it asks that the head be
  no worse than P0.

Release criteria (pass or fail, with paired bootstrap confidence intervals against P0):

| Criterion | Result |
|---|---|
| 1. Review readiness not worse than P0 | **fails on Measuring Hate Speech's Scores** (ρ and tie AUROC). Passes on HelpSteer3, where the head and P0 are both close to chance (above) |
| 2. Calibration after the per-task step (the new-task factor): ECE ≤ 0.10 on every task, and Brier not significantly worse than before any calibration | ECE passes on every task (0.018–0.047). **Brier fails on Measuring Hate Speech's Noul**: 0.154 calibrated against 0.147 uncalibrated, Δ [+0.0055, +0.0077]. It stays below P0's 0.209 |
| 3. Accuracy not worse than P0 | passes; better on 2 of 5 tasks, level on the rest |
| 4. Close to the task-specific fine-tunes on the training datasets' own test sets | **fails on one of 20 checks**: HelpSteer2's ECE is 0.123 against the fine-tune's 0.105, Δ [+0.005, +0.028]. Its accuracy and review readiness there pass, and every accuracy check passes (table below) |
| 5. Contract: conformance, independence and order | passes: conformance 100% on every suite item (41,642 requests). Reversing options, packing another request first or adding a question moves no probability by more than 0.03 (bf16 rounding); an identical request reproduces exactly |
| 6. Provenance | passes: release-tier mix with its manifest; Gemma 4 teacher and backbone; clean-room checks pass, which check that no answer from TypeSafe's Jev System One API, whose public request format DecisiveLM follows, has entered the data ([guide §16.1](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/16-provenance-licences-and-responsible-use.md)) |
| 7. Snapshot audit | passes: allowlisted paths only, the test suite green, no secrets, private paths or private data |
| 8. A better starting point for fine-tunes (1.0; reported for 0.x) | passes: fine-tunes on 100 or 300 items of the coding study and MultiNLI, started from this head, are never significantly worse than ones started from the base model. At 100 items they are better on accuracy and ρ on both tasks (coding: 91.9% against 87.7%, ρ 0.584 against 0.479) and on errors caught in review on MultiNLI |

This head is released as a preview although it fails three checks. That decision was made after its suite result
was seen, and the criteria's change log records it
([RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), 2026-11-03).

### Training datasets' own test sets

| Dataset | Metric | Task-specific fine-tune | This head |
|---|---|---|---|
| Banking77 | accuracy | 0.902 | 0.909 |
| CLINC150 | accuracy | 0.920 | 0.943 |
| HelpSteer2 | mean MAE / ECE | 0.468 / 0.105 | 0.466 / **0.123** |
| Civil Comments | mean AUROC | 0.982 | 0.979 |
| GoEmotions | accuracy / mean AUROC | 0.760 / 0.952 | 0.743 / 0.951 |
| LEDGAR | accuracy | 0.821 | 0.836 |

Every accuracy difference is inside the pre-registered margin (3 points of accuracy or AUROC, 0.05 levels of MAE).
On the multi-rater sets (HelpSteer2, GoEmotions) the head's review readiness is not significantly different from the
fine-tune's. Bold: the one failed check, HelpSteer2's calibration.

### Development sets

These sets were used to choose between candidate heads, so they are reported as development results, not tests.
P0 is calibrated the same way as the head.

| Set | What it shows | This head | P0 |
|---|---|---|---|
| VizWiz-VQA: photos by blind and low-vision people, 10 answerers each (trained on; scored on held-out photos) | photo questions | can the photo answer the question: AUROC 0.933; answer, 4 options: 71.1% right, ρ 0.633 | AUROC 0.788; 43.2%, ρ 0.292 (with task calibration) |
| Coding study: 792 free-text answers, 8 codes, 4 human coders (Alhasan, Diaz Monfort & Richardson, under review; data not distributed) | a rule-heavy coding task, zero-shot | κ with coders 0.675 (coders with each other 0.739); errors caught in a 20% review 60.8%; ρ 0.208; ECE 0.398 zero-shot | κ 0.649; errors caught 42.2%; ρ 0.386 |
| Seven public benchmarks (AG News, Banking77, DailyDialog, emotion, financial topics, SST-2, tweet topics) | classification, zero-shot except Banking77, whose training portion is in the mix | more accurate than P0 on 3, level on 1, less on 3; ECE ≤ 0.09 after 100 rows (0.06–0.28 zero-shot) | ECE ≤ 0.08 after 100 rows |
| MultiNLI (5 annotators per pair; held out) | review readiness on a task outside the mix | ρ 0.323 (Choice), 0.200 (Score); errors caught 47.2% (Choice) | ρ 0.261, 0.169; errors caught 39.2% |
| Wikipedia toxicity (about 10 raters per comment; held out) | review readiness on an unseen two-sided social scale | ρ −0.010; tie AUROC 0.531; MAE 0.462 | ρ +0.057; tie AUROC 0.539; MAE 0.355 |

**How much of this is training noise.** Two more training runs with only the seed changed (seeds 1 and 2) differ
from this one by about as much as two different mixes do on the text development sets. For example, on Humicroedit (a
humour-rating set held out of training) the MAE moves by up to 0.06 between seeds, against 0.08 between the v5c and
v5a mixes, and on the sarcasm corpus it moves by up to 0.06. The other two runs against this one:
- On Wikipedia toxicity, their ρ is −0.175 and −0.142, against this head's −0.010.
- On DailyDialog (emotion in dialogue, one of the seven benchmarks), seed 2 is 7.4 points less accurate (57.1%
  against 64.5%), and seed 1 is 1 point more accurate.
- Their photo results are close to this head's: answerable AUROC 0.930 and 0.930 against 0.933, and four-option
  accuracy 68.9% and 71.9% against 71.1%.
- On the coding study, seed 1's ρ is 0.372 against this head's 0.208, and the errors it catches in review are 48.3%
  against 60.8%. Seed 2 has not been scored there.

Across the three runs, ρ on the text development sets ranges by up to 0.17, accuracy by up to 3 points (8 on
DailyDialog) and Score MAE by up to 0.06.

Single differences of that size between heads should not be read as real (`results/general_v5c_e4b.md`).

**Wikipedia toxicity.** The last row of the table is limitation 1, reproduced on development data. It was added
because of an earlier candidate's suite result. Three later changes to the training mix, each meant to fix the
failure, did not improve it on toxicity:
- politeness ratings;
- humour and sarcasm ratings;
- the same mix without the teacher's Score labels.

They are recorded in the [RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md) change log
and in `results/probe_v5c_nts.md`.

**Task calibration** fits a scale and an offset per level on the task's own ratings (guide §10.6). It helps here
because raters rarely use the ends of this scale, which the head could not know zero-shot. Fitted on toxicity's
2,040 calibration rows, with P0 given the same step, the head's results against P0's are:
- ρ +0.349 against +0.293;
- MAE 0.229 against 0.277;
- tie AUROC 0.513 against 0.503;
- errors caught by a review of its least confident 20%: 26.7% against 32.6%, so P0 still does better on this measure.

Fitted on a random 100 or 300 of those rows instead, ρ and MAE are nearly the same (ρ +0.337 and +0.348). On
humour and sarcasm, which the head was never trained on, the same step takes ρ from +0.11 to +0.19 and from +0.03 to
+0.18, against P0's +0.10 and +0.16 under the same step (`results/taskcal_general_v5c_e4b.md`).

### Zero-shot on other images

CIFAR-10H (Peterson et al. 2019; CC BY-NC-SA 4.0): the CIFAR-10 test photos (Krizhevsky 2009; 32 × 32 pixels), each
labelled by about 51 people, one Choice over ten classes. It is held out of every mix and was scored once, on the
5,993 test photos. Task calibration is fitted on the other 4,007 (`results/cifar10h_v5c_zero_shot.md`).

| | Accuracy | ρ with agreement | Tie AUROC | ECE |
|---|---|---|---|---|
| This head, zero-shot | 0.909 | 0.408 | 0.944 | 0.439 |
| This head + task calibration | 0.927 | 0.440 | 0.960 | 0.046 |
| P0 (prompted with the photo) | 0.875 | 0.323 | 0.882 | 0.056 |
| P0 + task calibration | 0.889 | 0.362 | 0.911 | 0.048 |

The head was trained on one photo task, not on object classes. It is more accurate than P0 on these photos, and its
confidence follows rater agreement more closely than P0's, but its raw probabilities are badly scaled on them (ECE
0.44). Calibrate on about 100 labelled images of your task before relying on its confidence. Fitted on 100 photos, the
mean of five draws gives accuracy 0.922 and ρ 0.429. ECE was not recorded for the 100-photo fits. Fitted on all 4,007
calibration photos, it is 0.046 after task calibration, down from 0.439 raw (table above).

## Limitations and biases

- **Review readiness on unseen two-sided scales.** On social rating scales with a neutral middle that the head has not
  been trained on, its confidence tracks how extreme the text is, not how much people agree. On Measuring Hate
  Speech's survey items and on Wikipedia toxicity it is a worse guide to which answers to check than the prompted base
  model, and it chooses the neutral middle less often than people do. With 100–300 of the task's own ratings, task
  calibration (guide §10.6) makes its confidence follow rater agreement on development data, with answers more
  accurate than P0's under the same step (though P0's least confident 20% still holds more of its errors). A fine-tune
  on them is the other remedy. Use one of them for such a task. The cause is not known, but two causes are ruled out.
  One is the teacher's Score labels, because training without them did not fix it. The other is avoiding the middle
  level, because where the raters' most common answer is the neutral middle, the head picks it about as often as the
  prompted base model (36% against 34% on Wikipedia toxicity; the teacher 11%). Two-sided human ratings (Roadmap, 0.2)
  are the planned route.
- **Two smaller calibration failures.** On Measuring Hate Speech's yes/no question, the head's general calibration
  leaves its Brier score slightly worse than uncalibrated (0.154 against 0.147), though better than the prompted base
  model's. On HelpSteer2 its calibration error (0.123) is slightly above the task-specific fine-tune's (0.105). Both
  are criteria it fails. Task calibration on your own items may help here too, but it has not been tested on these two
  cases.
- **Training noise.** Three training runs of the same recipe that differ only in their random seed differ on text
  development sets about as much as two different training mixes do: by up to 0.06 in Score MAE, up to 3 points of
  accuracy (8 on one benchmark, DailyDialog), and up to 0.17 in ρ (`results/general_v5c_e4b.md`, the seeds
  section). Small differences between this head and
  others are not reliable.
- **Calibration on a new task.** Confidence is well calibrated on the tasks the head was trained on. On a new task it
  is usually underconfident zero-shot, and on some (the suite's 60-intent task in languages it was not trained on)
  overconfident. Calibrate on about 100 labelled items of your task before setting a review threshold (guide §10.6).
  The release criteria measure calibration after the new-task factor, which they call the per-task step.
- **One pass, no reasoning.** The head reads each option once with the state. Long instructions with precedence rules
  ("code B unless C is also mentioned") are answered less well than by a model prompted to reason step by step. A
  fine-tune on labelled examples is the route for such tasks (guide chapter 9, section 9.6).
- **Inherited judgement.** Part of the training labels come from Gemma 4 26B A4B. The head can share that model's
  systematic errors and biases, and the human-labelled datasets carry the views of their raters.
- **Language.** Training is mostly English. Other languages are answered through the multilingual backbone and
  measured on the held-out suite.
- **Self-contained options.** Each option is judged without seeing the others, so an "other" option needs a
  description of what it covers ("about something other than payments, outages or pricing").
- **One image task, no audio.** Of images, this head is trained only on VizWiz-VQA's photo questions. On another
  image set (CIFAR-10H, above) its raw probabilities were badly scaled until calibrated on the task. It is not trained
  or evaluated on audio. The toolkit fine-tunes on both, and the guide's image and audio examples (§9.9) start from
  a general head.

## Roadmap

This preview was released with three failed checks (Evaluation, above). Later versions are held to the release
criteria as written ([RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md)), so the dates
below are estimates and move if a version does not pass.

| Version | Estimated | Main change |
|---|---|---|
| 0.1 | Early November 2026 | This preview, the first public release: English text, JSON and photo questions, as two heads on Gemma 4 E4B and 26B A4B trained on the same mix |
| 0.2 | Mid to late November 2026 | Review readiness on two-sided scales: 600 public texts rated by five people each through our open rating app (agree–disagree, sure–unsure, formal–informal, serious–joking, convincing–unconvincing), first as a check set, then as training data. HelpSteer2-Preference (the judgement the suite's HelpSteer3 task asks) and the training portion of Measuring Hate Speech, held back until the first release, join the mix. The suite's rows stay held out, and from 0.2 those two suite tasks are reported as tasks the head has seen. Also human-labelled data in languages other than English |
| 1.0 | By the end of November 2026 | Images and audio (photographs, including questions about them and the text in them; speech emotion, everyday sounds and audio quality) with multi-rater labels and frozen image and audio suites; better than the prompted base model on review readiness across the suite, the stricter calibration gate, and a better starting point for fine-tunes at 100 items |

## Licence

The code and this head are released under Apache-2.0. The head is used with Gemma 4 E4B, released by Google DeepMind
under Apache-2.0. See its model card for its terms. Training datasets keep their own licences (above), and attribution
for CC BY sources is in
[guide chapter 14](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/14-datasets.md).

## Citation

```bibtex
@misc{decisivelm,
  title  = {DecisiveLM: an open, calibrated System One decision model on Gemma 4},
  author = {Richardson, Michael J.},
  year   = {2026},
  note   = {https://github.com/xkiwilabs/decisive-lm}
}
```

The same entry is in the repository's `README.md` and `CITATION.cff`. When the companion paper is on arXiv, cite it
instead (the entry will be updated here).
