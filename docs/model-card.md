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

<!-- DRAFT 2026-11-03: E4B sections rewritten for General v5c. Version 0.1 (owner, 2026-11-04). The 26B A4B head
also ships in 0.1 (owner, 2026-11-05, RELEASE.md); it gets its own card. -->

# DecisiveLM E4B general head 0.1 (preview)

A general-purpose decision head for Google's Gemma 4 E4B. The head is about 20M trained parameters (LoRA adapters,
embeddings for three reserved tokens and a scorer network, see Summary) used with the base model, whose other weights
are not retrained. Given a state (text, JSON or a photo with text) and a set of typed questions, it answers every
question in one forward pass (one run of the model over the input, with no text generated), with a probability
distribution over the options you supply. These probabilities are calibrated on the tasks the head was trained on.
On a new task they are usually underconfident until calibrated on about 100 labelled items of that task (below). The
head answers questions it was not trained on, without examples. On the held-out suite below (unseen contract clause
types, comparisons of chatbot responses, hate speech ratings, and intents in ten languages it was not trained on), it
is never significantly less accurate than the base model prompted with the same questions, and it is more accurate on
two of the five tasks. Its confidence is built to say which answers a person should check. On two-sided social rating
scales it has not been trained on, it does this worse than the prompted base model (below).

- **Code**: [github.com/xkiwilabs/decisive-lm](https://github.com/xkiwilabs/decisive-lm) (Apache-2.0)
- **Documentation**: [the guide](https://github.com/xkiwilabs/decisive-lm/tree/main/docs/guide), starting with the
  [quickstart](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/03-quickstart.md)
- **Base model**: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) at revision
  `ee0ef6023621cff504d758262d4e04895a5af4a2`, which this head requires

> **Known limitations of this preview.** It fails three checks of the
> [release criteria](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), and was released with them
> stated (Evaluation, below):
>
> 1. **Two-sided social rating scales it has not been trained on.** On scales such as "strongly respectful … neither …
>    strongly disrespectful", its confidence follows how extreme the text is rather than how much people agree. On
>    Measuring Hate Speech's survey items it is a worse guide to which answers to check than the prompted base model.
>    This is the main limitation. On such a task, calibrate each level on about 100 of your own ratings (task
>    calibration, guide §10.6) or fine-tune on them; on a development set this made its confidence follow rater
>    agreement more closely than the prompted base model's under the same step.
> 2. **Yes/no calibration on that survey.** The head's general calibration makes its yes/no answers there slightly
>    worse (Brier score 0.154 against 0.147 uncalibrated). It is still better calibrated there than the prompted base
>    model (0.209).
> 3. **Calibration on HelpSteer2**, one of its own training sets. Its calibration error is 0.123, slightly but
>    significantly above the task-specific fine-tune's 0.105. Its accuracy and review readiness there match the
>    fine-tune's.
>
> See [Limitations](#limitations-and-biases).
>
> **Calibrate before you set a review threshold.** Zero-shot, the head is usually underconfident on a new task.
> Fitting two numbers on about 100 labelled items of your task brought its expected calibration error to 0.05 or
> below on every task of the held-out suite
> ([guide §10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)).

## Summary

| | |
|---|---|
| Question types | Choice (2–255 unordered options), Score (2–10 ordered levels), Noul (probability a statement is true) |
| Input | text or JSON state up to 8,192 tokens (longer states are shortened head and tail), optionally with a photo |
| Output | a probability per option or level, the chosen option or expected level, a confidence; for Noul one probability |
| Independence | each question, and each option, is scored on its own, so by design adding or removing questions or reordering options does not change other answers. On the released head such changes move no probability by more than 0.03 (bf16 rounding, criterion 5 below) |
| Calibration | temperatures per question type and number of options, and a Noul bias, fitted on human-labelled questions of the training mix; on a new task, calibrate on about 100 of your own labelled items |
| Size | 17.4M trained parameters inside the backbone (LoRA adapters added to its projection layers, and the embedding rows of the three query tokens) and a 2.6M-parameter scorer. The backbone's other weights are frozen (not trained) |

## Intended use

- Routing, triage and tagging of messages, tickets and documents.
- Rubric scoring and quality judgements with a review step for uncertain items.
- Coding qualitative data, such as survey answers and open-ended responses, as a first coder whose confidence says
  which items a human coder should see.
- Screening and extraction expressed as yes/no statements (Nouls).
- Measuring a new decision task zero-shot before deciding whether to train a task-specific fine-tune.
- Questions about photos: whether a photo can answer a question, and which of a few answers it supports (the one
  image task it was trained on, VizWiz-VQA, below).

Check accuracy and the review threshold on a labelled sample of your own task before relying on the head
([guide chapter 7](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/07-using-the-general-head.md)).

## Out-of-scope use

- Unreviewed decisions that affect people: clinical, credit, hiring, housing, legal, policing or benefits decisions.
  Such uses need your own labelled evaluation, checks of errors across the groups affected, and a person in the loop.
- Free-text generation, explanation or summarisation.
- Choice questions with more than 255 options.
- Audio: this head was not trained on audio, and has not been evaluated on it.
- Images beyond photo questions without calibration on your task: it was trained on one photo task (VizWiz-VQA).
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
| Memory | a 24 GB GPU is enough. Predicting the coding study's 792 responses peaked at 16.0 GiB with 2,048 tokens per forward pass and at 17.8 GiB with 8,192. One request at the server's default limit of 12,288 tokens peaks at about 18.6 GiB | a 24 GB GPU is enough. A 300-item fine-tune peaked at 21.2 GiB with the default micro-batches of 8,192 tokens, at 18.5 GiB with 4,096 and at 16.7 GiB with 2,048, with gradient checkpointing (on by default). It took 15 to 25 minutes on one RTX PRO 6000 that was shared with other work |
| Disk | about 15 GB for Gemma 4 E4B and about 1 GB for the Python environment | the same, plus your data |

The peaks are per process, as `nvidia-smi` reports them, so they include the CUDA context. A 16 GB GPU is not enough
at these settings. The bf16 weights take about 15 GiB, so a smaller micro-batch saves only a few GiB.

The CPU runs the test suite, not the model. On a Mac with large unified memory, fine-tune with
`--no-checkpointing --max-tokens 2048`. A training row then takes about 0.14 s for a short message and about 2 s for a
longer row, such as a free-text answer coded into one of eight categories. The full requirements are in guide
[chapter 2](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/02-installation.md), and settings for a GPU
shared with other work are in section 9.5 of
[chapter 9](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/09-fine-tuning-for-your-task.md).

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
The public datasets are linked rather than re-hosted, and the teacher-written rows and their labels are published in the
companion dataset repository on Hugging Face (`decisive-lm-data`, tagged with the release version, guide §14.7).
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
| Losses | soft cross-entropy against label shares divided by log n (Choice); the same plus 0.5 × EMD² between the predicted and target cumulative distributions (Score); binary cross-entropy against the share (Noul) |
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

Four public, human-labelled sets, fixed before any candidate was scored and never trained on. No mix, stopping rule
or prompt is chosen by looking at them. The one fit made on them is the per-task calibration step, the step a user
takes on a new task. It fits two numbers per method (one factor on its Choice temperatures and one on its Score
temperatures) on the 100 rows of each set with the lowest hash of their ids. Those rows, and any rows that
share a source item with them, are left out of every score below. Each candidate is scored on the suite once. ECE is
shown zero-shot and after that step. Full report: `results/suite_general_v5c_e4b_r8_seed0.md`.

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
| 2. Calibration after the per-task step: ECE ≤ 0.10 on every task, and Brier not significantly worse than before any calibration | ECE passes on every task (0.018–0.047). **Brier fails on Measuring Hate Speech's Noul**: 0.154 calibrated against 0.147 uncalibrated, Δ [+0.0055, +0.0077]. It stays below P0's 0.209 |
| 3. Accuracy not worse than P0 | passes; better on 2 of 5 tasks, level on the rest |
| 4. Close to the task-specific fine-tunes on the training datasets' own test sets | **fails on one of 20 checks**: HelpSteer2's ECE is 0.123 against the fine-tune's 0.105, Δ [+0.005, +0.028]. Its accuracy and review readiness there pass, and every accuracy check passes (table below) |
| 5. Contract: conformance, independence and order | passes: conformance 100% on every suite item (41,642 requests). Reversing options, packing another request first or adding a question moves no probability by more than 0.03 (bf16 rounding); an identical request reproduces exactly |
| 6. Provenance | passes: release-tier mix with its manifest; Gemma 4 teacher and backbone; clean-room checks pass, which check that no answer from TypeSafe's Jev System One API, whose public request format DecisiveLM follows, has entered the data ([guide §16.1](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/16-provenance-licences-and-responsible-use.md)) |
| 7. Snapshot audit | passes: allowlisted paths only, the test suite green, no secrets, private paths or private data |
| 8. A better starting point for fine-tunes (1.0; reported for 0.x) | passes: fine-tunes on 100 or 300 items of the coding study and MultiNLI, started from this head, are never significantly worse than ones started from the base model. At 100 items they are better on accuracy, ρ and errors caught in review on both tasks (coding: 91.9% against 86.6%) |

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
| Coding study: 792 free-text answers, 8 codes, 4 human coders (Alhasan, Diaz Monfort & Richardson, under review; data not distributed) | a rule-heavy coding task, zero-shot | κ with coders 0.676 (coders with each other 0.739); errors caught in a 20% review 60.6%; ρ 0.206; ECE 0.396 zero-shot | κ 0.649; errors caught 41.9%; ρ 0.385 |
| Seven public benchmarks (AG News, Banking77, DailyDialog, emotion, financial topics, SST-2, tweet topics) | zero-shot classification | more accurate than P0 on 3, level on 1, less on 3; ECE ≤ 0.09 after 100 rows (0.06–0.28 zero-shot) | ECE ≤ 0.08 after 100 rows |
| MultiNLI (5 annotators per pair; held out) | review readiness on a task outside the mix | ρ 0.323 (Choice), 0.200 (Score); errors caught 47.2% (Choice) | ρ 0.261, 0.169; errors caught 39.2% |
| Wikipedia toxicity (about 10 raters per comment; held out) | review readiness on an unseen two-sided social scale | ρ −0.010; tie AUROC 0.531; MAE 0.462 | ρ +0.057; tie AUROC 0.539; MAE 0.355 |

**How much of this is training noise.** A second training run with only the seed changed differs from this one by
about as much as two different mixes do on the text development sets.
- On the coding study, its ρ is 0.373 against this head's 0.206, and the errors it catches in review are 49.2%
  against 60.6%.
- On Wikipedia toxicity, its ρ is −0.175 against −0.010.
- Its κ on the coding study (0.684) and its photo results are close to this head's.

Single differences of that size between heads should not be read as real (`results/general_v5c_e4b.md`).

**Wikipedia toxicity.** The last row of the table is limitation 1, reproduced on development data. It was added
because of an earlier candidate's suite result, which is why it is reported here. Three later attempts to fix the
failure from the training mix did not improve it on toxicity:
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
5,993 test photos; task calibration is fitted on the other 4,007 (`results/cifar10h_v5c_zero_shot.md`).

| | Accuracy | ρ with agreement | Tie AUROC | ECE |
|---|---|---|---|---|
| This head, zero-shot | 0.909 | 0.408 | 0.944 | 0.439 |
| This head + task calibration | 0.927 | 0.440 | 0.960 | 0.046 |
| P0 (prompted with the photo) | 0.875 | 0.323 | 0.882 | 0.056 |
| P0 + task calibration | 0.889 | 0.362 | 0.911 | 0.048 |

The head was trained on one photo task, not on object classes. It is more accurate than P0 on these photos, and its
confidence follows where people disagree more closely, but its raw probabilities are badly scaled on them (ECE 0.44).
Calibrate on about 100 labelled images of your task before relying on its confidence. Fitted on 100 photos, the mean
of five draws gives accuracy 0.922 and ρ 0.429.

## Limitations and biases

- **Review readiness on unseen two-sided scales.** On social rating scales with a neutral middle that the head has
  not been trained on, its confidence tracks how extreme the text is, not how much people agree. On Measuring Hate
  Speech's survey items and on Wikipedia toxicity it is a worse guide to which answers to check than the prompted base
  model, and it chooses the neutral middle less often than people do. With 100–300 of the task's own ratings, task
  calibration (guide §10.6) makes its confidence follow rater agreement on development data, with answers more
  accurate than P0's under the same step (though P0's least confident 20% still holds more of its errors). A
  fine-tune on them is the other remedy. Use one of them for such a task. The cause is not known. Training without
  the teacher's Score labels did not fix it, and where the raters' most common answer is the neutral middle, the head
  picks it about as often as the prompted base model (36% against 34% on Wikipedia toxicity; the teacher 11%).
  Two-sided human ratings (Roadmap, 0.2) are the planned route.
- **Two smaller calibration failures.** On Measuring Hate Speech's yes/no question, the head's general calibration
  leaves its Brier score slightly worse than uncalibrated (0.154 against 0.147), though better than the prompted base
  model's. On HelpSteer2 its calibration error (0.123) is slightly above the task-specific fine-tune's (0.105).
  Both are criteria it fails. Task calibration on your own items is the remedy here too.
- **Training noise.** Two training runs of the same recipe that differ only in their random seed differ on text
  development sets about as much as two different training mixes do: by 0.06 in Score MAE, 2–3 points of accuracy,
  and up to 0.17 in ρ (Development sets, above). Small differences between this head and others are not reliable.
- **Calibration on a new task.** Confidence is well calibrated on the tasks the head was trained on. On a new task it
  is usually underconfident zero-shot, and on some (the suite's 60-intent task in languages it was not trained on)
  overconfident. Calibrate on about 100 labelled items of your task before setting a review threshold (guide §10.6).
  The release criteria measure calibration after that step.
- **One pass, no reasoning.** The head reads each option once with the state. Long instructions with precedence rules
  ("code B unless C is also mentioned") are answered less well than by a model prompted to reason step by step. A
  fine-tune on labelled examples closes much of that gap (guide chapter 9, section 9.6).
- **Inherited judgement.** Part of the training labels come from Gemma 4 26B A4B. The head can share that model's
  blind spots and biases, and the human-labelled datasets carry the views of their raters.
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
