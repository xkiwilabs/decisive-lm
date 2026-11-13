---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
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

# DecisiveLM 26B A4B general head 0.1 (preview)

A general-purpose decision head for Google's Gemma 4 26B A4B, a mixture-of-experts model that uses about 4B of its 26B
parameters for each token. The head is about 12M trained parameters (LoRA adapters, embeddings for three reserved
tokens and a scorer network, see Summary) used with the base model, whose other weights are not retrained. Given a
state (text, JSON or a photo with text) and a set of typed questions, it answers every question in one forward pass
(one run of the model over the input, with no text generated), with a probability distribution over the options you
supply. It is trained on the same data as the E4B head ([its
card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md)), on the same rows of General v5c, and
scored on the same test items, so the two can be compared item by item.

**How it compares.** It is compared with Gemma 4 26B A4B without the head, prompted to answer the same questions
(P0), which is also the model that labelled part of its training data.
- **Against the E4B head**, it is better on the main measure of every held-out suite task. The comparison is unpaired,
  and on MASSIVE (0.680 against 0.672) and CUAD (0.961 against 0.947) the gap is within the spread between two
  training runs of the E4B head. On the hate-speech survey its confidence ranks answers by rater agreement much better
  (ρ 0.177 against 0.078, where ρ is Spearman's rank correlation between its top probability and per-item human pair
  agreement, 0 is chance and 1 is a perfect ranking), though a review of its least confident 20% catches slightly
  fewer of its errors.
- **Against P0**, its confidence ranks the hate-speech survey's ratings by rater agreement better, but a review of its
  least confident 20% catches fewer of its errors (figures in the box below). It is the better guide on four of five
  held-out development sets, and its rating errors are smaller on the suite's two rating tasks.
- **It is less accurate than P0** on intents in ten languages it was not trained on, on four of seven public
  benchmarks, and on two rating sets (Wikipedia toxicity and Humicroedit). The figures are in the box below and in
  Evaluation.
- **It is slower and larger than the E4B head**: about 0.47 s per request against 0.10 s, on a GPU with about 72 GiB free (Hardware, below).

If you have a 24 GB GPU or need speed, use the E4B head.

- **Code**: [github.com/xkiwilabs/decisive-lm](https://github.com/xkiwilabs/decisive-lm) (Apache-2.0)
- **Documentation**: [the guide](https://github.com/xkiwilabs/decisive-lm/tree/main/docs/guide), starting with the
  [quickstart](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/03-quickstart.md)
- **Base model**: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) at revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, which this head requires

> **Known limitations of this preview.** It fails five checks of the [release
> criteria](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), and was released with them stated
> (Evaluation, below). Items 1–3 compare it with P0, a stronger model than the E4B head's P0; item 4 compares its
> calibrated answers with its uncalibrated ones. Item 2 holds two of the five checks, which are counted per measure:
>
> 1. **Intents in languages it was not trained on** (MASSIVE): 68.0% right against 73.1%.
> 2. **Which hate-speech survey answers to check.** Its confidence ranks answers by rater agreement better than P0's
>    (ρ 0.177 against 0.162), but a review of its least confident 20% catches fewer of its errors (23.4% against
>    26.7%). In the worst of 44 groups of that survey (a group is a category, the consensus level of a question), its
>    confidence runs against rater agreement more than P0's does (ρ −0.205 against −0.175).
> 3. **The worst group of HelpSteer3**, a chatbot-response comparison task: ρ −0.038 against −0.011, both close to
>    chance.
> 4. **Calibrating on 100 items of the hate-speech survey's ratings** (the new-task factor) leaves their Brier score
>    (the mean squared error of the probabilities against the human answer, where lower is better) slightly worse than
>    before (0.6403 against 0.6399).
>
> On Humicroedit, a humour-rating set held out of training, it rates edited headlines as funnier than people do, and is
> worse than P0 there on both accuracy and review readiness. See [Limitations](#limitations-and-biases).
>
> **Calibrate before you set a review threshold.** Fitting the new-task factor (one factor on the Choice temperatures
> and one on the Score temperatures) on about 100 labelled items of your task brought its expected calibration error
> to 0.04 or below on every task of the held-out suite ([guide
> §10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)).

## Summary

| | |
|---|---|
| Question types | Choice (2–255 unordered options), Score (2–10 ordered levels), Noul (probability a statement is true) |
| Input | text or JSON state up to 8,192 tokens (longer states are shortened head and tail), optionally with a photo. No audio: the 26B A4B backbone has no audio input |
| Output | a probability per option or level, the chosen option or expected level, a confidence; for Noul one probability |
| Independence | each question, and each option, is scored on its own, so by design adding or removing questions or reordering options does not change other answers. On the released head such changes move no probability by more than 0.011, and an identical request reproduces exactly (criterion 5 below) |
| Computation | bf16 weights with float32 computation, always. In bf16 computation the backbone's expert routing changes 19–22% of answers when only the packing of a request changes, so the head is trained, checked and served in float32. Loading the head does this on its own |
| Calibration | temperatures per question type and number of options, and a Noul bias, fitted on human-labelled questions of the training mix; on a new task, calibrate on about 100 of your own labelled items |
| Size | 9.3M trained parameters inside the backbone (LoRA adapters added to its attention and shared MLP projections, not to its 128 experts, and the embedding rows of the three query tokens) and a 2.9M-parameter scorer. The backbone's other weights are frozen (not trained) |

## Intended use

As for the E4B head ([its card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#intended-use)):
decisions that a person would otherwise make item by item, where its confidence says which answers to check. Choose
this head over the E4B head when a GPU with about 72 GiB free is available, and when ratings, or knowing which ratings
to check, matter more than speed.

### Use cases

Each row names what was measured on that kind of task and what it was compared against: the prompted 26B A4B without
the head (P0, a stronger model than the E4B head's P0), the E4B head, or a task-specific fine-tune of E4B. "Trained
on" means the dataset's training portion is in the head's training mix, and the figure is on its held-out test
portion. Every other set is zero-shot (no training examples of it).

| Use case | For example | What was measured |
|---|---|---|
| Rating text on a scale, such as survey items | a Score "How respectful is this comment?" on five levels | Measuring Hate Speech's nine survey items (not trained on): ratings closer to the raters' than P0's (MAE 0.571 against 0.703) and than the E4B head's (0.653). Its confidence ranks the items by rater agreement better than P0's (ρ 0.177 against 0.162), but its least confident 20% holds fewer of its errors (23.4% against 26.7%) |
| Rating against a rubric | a Score "How helpful is this reply?" on five levels | HelpSteer2 (trained on): mean MAE 0.455 and ECE 0.102, against 0.468 and 0.105 for an E4B fine-tune on it alone. HelpSteer3, comparing two chatbot replies (not trained on): MAE 0.899, against P0's 1.158 and the E4B head's 1.023. On HelpSteer3 neither model's confidence says which answers to check, so review a random sample there |
| Moderation and yes/no screening | a Noul "The comment contains a personal attack" | the hate-speech survey's yes/no question (not trained on): AUROC 0.882, against P0's 0.845. Civil Comments (trained on): mean AUROC 0.973, against 0.982 for an E4B fine-tune on it alone |
| Contract and document review | a Noul "This clause caps the supplier's liability", or a Choice over 100 provision types | CUAD, ten clause types it was not trained on: AUROC 0.961, against P0's 0.954. LEDGAR, 100 provision types (trained on): 83.0% right, against 82.1% for an E4B fine-tune on it alone |
| Whether one text supports another | a Choice "Does the second sentence follow from the first?" over follows, contradicts and neither | MultiNLI, five annotators per pair (not trained on): ρ between its confidence and annotator agreement 0.365, against P0's 0.321, with 84.5% right against P0's 85.6% |
| Questions about photos | a Noul "The photo shows enough to answer the question", or a Choice between a few answers | VizWiz-VQA, photos taken by blind and low-vision people (trained on; held-out photos): AUROC 0.938 against P0's 0.802, and 73.1% right on a four-option answer against 48.5% |
| Routing and intent detection, in English | a Choice "Which team should handle this message?" | Banking77 (trained on): 89.9% right, against 90.2% for an E4B fine-tune on it alone. CLINC150 (trained on): 92.3% against 92.0%. On seven public benchmarks P0 is more accurate on four, by 2–4 points |

Not a good fit:
- **intents in languages other than English**, where it is 5 points less accurate than P0 (68.0% against 73.1% on
  MASSIVE);
- **humour ratings**, where it rates headlines as funnier than people do and its confidence is no guide (Limitations);
- **work that needs speed or a smaller GPU**, where the E4B head is the choice.

It has not yet been scored on coding open-ended answers, or as a starting point for fine-tunes (criterion 8, below).

Use the confidence as for the E4B head: label about 100 items, calibrate on them ([guide
§10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)), check accuracy on other
labelled items, then send the least confident answers to people ([guide chapter
7](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/07-using-the-general-head.md)).

## Out-of-scope use

- Unreviewed decisions that affect people: clinical, credit, hiring, housing, legal, policing or benefits decisions.
  Such uses need your own labelled evaluation, checks of errors across the groups affected, and a person in the loop.
- Free-text generation, explanation or summarisation.
- Choice questions with more than 255 options.
- Audio. The server refuses it with a 400 error.
- Images beyond photo questions without calibration on your task, because it was trained on one photo task
  (VizWiz-VQA) and has not been scored on any other image set.
- Questions whose answer is not in the state.
- Work that needs answers in about 0.1 s, or a GPU with less than about 72 GiB free. Use the E4B head for these.

## How to use

```bash
git clone https://github.com/xkiwilabs/decisive-lm.git && cd decisive-lm && uv sync
uv run hf download xkiwilabs/decisivelm-26b-a4b-general --local-dir bundles/decisivelm-26b-a4b-general
```

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/decisivelm-26b-a4b-general")   # float32 computation, set by the bundle
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
`calibration.json` (temperatures, Noul bias, backbone revision, the float32 setting and training settings). The server
(`python -m decisivelm.serve`) needs `DECISIVELM_MEMORY_GB=72`, the environment variable that caps the server's GPU
memory, in GiB (guide §4.10). One server holds heads of one backbone only.

### Hardware

| | Running the head | Fine-tuning it on your task |
|---|---|---|
| Hardware | Linux with an NVIDIA GPU (CUDA), or a Mac with Apple silicon (MPS) and 96 GB or more of unified memory. On an M5 Max with 128 GB it held 56.0 GiB after loading and peaked at 58.6 GiB over 20 requests of up to 1,599 tokens, about 1.9 s per request (median, one at a time), and its answers matched the CUDA predictions within 0.002 with no answer changed (60 questions) | the same; fine-tune memory on a Mac not measured |
| Memory | a GPU with about 72 GiB free. Through the server, 55.8 GiB after loading and 67.0 GiB at the peak, on requests at its default limits (a state is cut at 8,192 tokens, and a whole request may be up to 12,288), with or without a photo | not yet measured for a user fine-tune (criterion 8, below). Training this head on the general mix peaked at 66.0 GiB with micro-batches of 4,096 tokens and gradient checkpointing |
| Time | 0.47 s for a request of about 1,200 tokens with 20 options (four questions of five options each), batch 1, on one RTX PRO 6000 Blackwell Max-Q; 5.7–5.8 s for requests of about 12,000 tokens. The prompted 26B A4B, which reads each question once per option order, took 577–1,109 ms for the same request on vLLM (`results/p0_latency.md`) | 1.38 s per training row on the general mix |
| Disk | about 49 GB for Gemma 4 26B A4B and about 1 GB for the Python environment | the same, plus your data |

The peaks are per process, as `nvidia-smi` reports them, so they include the CUDA context. Float32 computation is the
main cost. The bare backbone in bf16, with no head, took 0.14 s on the same request, and the E4B head takes 0.10 s.

## Training data

The same mix as the E4B head, General v5c: 215,433 training requests (995,083 questions) from human-labelled public
datasets, 10,000 VizWiz-VQA photos with questions, rewritten questions on the same states, and teacher-written
question sets and states labelled by Gemma 4 26B A4B. Every source is in the release tier, whose licences allow
redistribution (guide chapter 14). The sources, licences and labels are listed on [the E4B head's
card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#training-data), and in [guide chapter
13](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/13-how-the-general-head-was-trained.md) and [chapter
14](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/14-datasets.md). The teacher is the same model as
this head's backbone.

Splits are assigned per source item and question family before any rewriting, so no test item or its rewrites reach
training. The frozen held-out suite (below) is listed in `configs/holdout.yaml`, and its text was checked for overlap
with every trainable row.

## Training procedure

| Setting | Value |
|---|---|
| Backbone | Gemma 4 26B A4B, frozen, bf16 weights with float32 computation |
| Adapter | LoRA rank 8, α 16, dropout 0.05, on the decoder's attention (q, k, v, o) and shared MLP (gate, up, down) projections only |
| Query tokens | three reserved token ids (Choice, Score, Noul), their embedding rows trained |
| Scorer | per-dimension standardisation (fitted on 2,000 training rows) → LayerNorm → Linear(2816, 1024) → GELU → Linear(1024, 1) |
| Readout and losses | as for the E4B head ([its card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#training-procedure)) |
| Optimiser | learning rate 3e-5 for LoRA, 1e-4 for the head and query rows; 10% warm-up; 16 rows per step; packed micro-batches of up to 4,096 tokens with gradient checkpointing |
| Selection | validation every 500 steps on the E4B head's 1,500 held-back rows; the checkpoint with the lowest calibrated validation loss is kept, and training stops after three checks without improvement (this release: step 15,000, stopped at 16,500 of at most 26,930, about 101 hours on one RTX PRO 6000 Blackwell Max-Q) |
| Calibration | as for the E4B head: one temperature per question type and option-count bucket, and a Noul temperature and bias, fitted on the mix's human-labelled `gold_calib` rows |

## Evaluation

The head is judged first on **review readiness**, whether its confidence says which answers to check, then on
calibration and accuracy. **P0 here is Gemma 4 26B A4B prompted**, the head's own backbone, and so also the teacher.
RELEASE.md's rule is that a user should be "better off than if they had prompted the base model it is built on", and
it was written down before either head was trained (change log, 2026-10-30). It makes this head's P0 a stronger model
than the E4B head's, whose P0 is the prompted E4B. The criteria in force for this head were fixed before it was
scored. It was released by a decision made after its result was seen (below). The measures are defined on [the E4B
head's card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#evaluation) and in guide chapters
[10](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md) and
[11](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/11-evaluation.md).

### Frozen held-out suite

The same four sets and the same new-task factor as for the E4B head, scored once. Full report:
`results/suite_general_v5c_26b-a4b_r8_seed0.md`.

| Task | What it tests | ρ(confidence, agreement) (P0) | tie AUROC (P0) | errors caught in a 20% review (P0) | ECE zero-shot / after 100 rows | primary metric | P0 primary metric |
|---|---|---|---|---|---|---|---|
| HelpSteer3 preference (Score, 6 levels) | judging chatbot responses beyond the training set | 0.065 (0.088) | 0.488 (0.511) | 22.2% (21.7%) | 0.044 / 0.022 | **0.899** (MAE) | 1.158 |
| Measuring Hate Speech (9 Scores) | calibration against real rater disagreement | **0.177 (0.162)** | **0.579 (0.562)** | **23.4% (26.7%)** | 0.029 / 0.026 | **0.571** (MAE) | 0.703 |
| Measuring Hate Speech (Noul) | the same, as a yes/no judgement | | | | 0.040 / 0.040 | **0.882** (AUROC) | 0.845 |
| CUAD, 10 held-out clause types (Noul) | unseen questions in a known domain | | | | 0.018 / 0.018 | 0.961 (AUROC) | 0.954 |
| MASSIVE, 10 held-out languages (Choice, 60 intents) | zero-shot language transfer | | | | 0.099 / 0.031 | **0.680** (accuracy) | 0.731 |

Bold: significantly different from P0 (paired 95% bootstrap). On HelpSteer3 the head's and P0's review measures are
all close to chance, as for the E4B head.

Release criteria (pass or fail, with paired bootstrap confidence intervals against P0):

| Criterion | Result |
|---|---|
| 1. Review readiness not worse than P0 | **fails three checks**: on Measuring Hate Speech's Scores, errors caught in a 20% review, Δ [−0.040, −0.025], and the lowest within-category ρ (−0.205 against −0.175, 44 categories); on HelpSteer3, the lowest within-category ρ (−0.038 against −0.011, 6 categories). ρ and tie AUROC pass on both tasks, and on Measuring Hate Speech they are better than P0's (ρ Δ [+0.001, +0.029]) |
| 2. Calibration after the per-task step (the new-task factor): ECE ≤ 0.10 on every task, and Brier not significantly worse than before any calibration | ECE passes on every task (0.018–0.040). **Brier fails on Measuring Hate Speech's Scores**: 0.6403 calibrated against 0.6399 uncalibrated, Δ [+0.0002, +0.0006] |
| 3. Accuracy not worse than P0 | **fails on MASSIVE**: 0.680 against 0.731, Δ [−0.055, −0.046]. Better on 3 of 5 tasks, level on CUAD |
| 4. Close to the task-specific fine-tunes on the training datasets' own test sets | passes all 20 checks, review readiness included (table below) |
| 5. Contract: conformance, independence and order | passes: conformance 100% on every suite item (41,642 requests). On 200 requests from each of two development sets, reversing options, packing another request first or adding a question changed no answer and moved no probability by more than 0.011, and an identical request reproduced exactly. Across the suite, answering each request alone rather than packed changed 36 of about 95,000 answers, 34 of them near-ties, by at most 0.056 (Limitations) |
| 6. Provenance | passes: the same release-tier mix and manifest as the E4B head; Gemma 4 teacher and backbone; clean-room checks pass |
| 7. Snapshot audit | passes: the same repository snapshot as the E4B head |
| 8. A better starting point for fine-tunes (1.0; reported for 0.x) | not yet measured for 0.1. It needs a GPU with about 72 GiB free for two days, and is planned for a later 0.x release, with the E4B head's recipe (coding study and MultiNLI, 100 and 300 items, three seeds) |

This head is released as a preview although it fails five checks. That decision was made after its suite result was
seen, and the criteria's change log records it
([RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), 2026-11-05).

### Against the E4B head

The two heads on the suite, each from its own suite report (not paired, so no intervals). This head is better on every
row:

| Task | Measure | This head | E4B head |
|---|---|---|---|
| HelpSteer3 | MAE | 0.899 | 1.023 |
| Measuring Hate Speech, Scores | MAE | 0.571 | 0.653 |
| Measuring Hate Speech, Scores | ρ(confidence, agreement) | 0.177 | 0.078 |
| Measuring Hate Speech, Scores | tie AUROC | 0.579 | 0.533 |
| Measuring Hate Speech, Noul | AUROC | 0.882 | 0.852 |
| CUAD | AUROC | 0.961 | 0.947 |
| MASSIVE | accuracy | 0.680 | 0.672 |

On Measuring Hate Speech's Scores, matched on the same 29,556 questions, its ρ is higher by +0.099 [+0.089, +0.110]
and its tie AUROC by +0.047 [+0.038, +0.054]. The share of errors caught in review is slightly lower (Δ −0.011
[−0.018, −0.004]). It fails more checks than the E4B head because its P0 is stronger. The prompted 26B is far more
accurate on MASSIVE than the prompted E4B (0.731 against 0.658).

### Training datasets' own test sets

The reference is the same E4B task-specific fine-tunes as for the E4B head.

| Dataset | Metric | Task-specific fine-tune | This head |
|---|---|---|---|
| Banking77 | accuracy | 0.902 | 0.899 |
| CLINC150 | accuracy | 0.920 | 0.923 |
| HelpSteer2 | mean MAE / ECE | 0.468 / 0.105 | 0.455 / 0.102 |
| Civil Comments | mean AUROC | 0.982 | 0.973 |
| GoEmotions | accuracy / mean AUROC | 0.760 / 0.952 | 0.732 / 0.946 |
| LEDGAR | accuracy | 0.821 | 0.830 |

Every accuracy difference is inside the pre-registered margin (3 points of accuracy or AUROC, 0.05 levels of MAE). On
the multi-rater sets (HelpSteer2, GoEmotions) its review readiness is not significantly different from the
fine-tune's.

### Development sets

These sets were used to choose between candidate heads, so they are reported as development results, not tests.
P0 is calibrated the same way as the head.

| Set | What it shows | This head | P0 |
|---|---|---|---|
| VizWiz-VQA: photos by blind and low-vision people, 10 answerers each (trained on; scored on held-out photos) | photo questions | can the photo answer the question: AUROC 0.938; answer, 4 options: 73.1% right, ρ 0.676 | AUROC 0.802; 48.5%, ρ 0.367 |
| Seven public benchmarks (AG News, Banking77, DailyDialog, emotion, financial topics, SST-2, tweet topics) | classification, zero-shot except Banking77, whose training portion is in the mix | more accurate than P0 on 1 (Banking77, 81.8% against 76.7%), level on 2, less on 4 by 2–4 points; ECE 0.04–0.09 after 100 rows (0.06–0.30 zero-shot) | ECE 0.02–0.16 after 100 rows |
| MultiNLI (5 annotators per pair; held out) | review readiness on a task outside the mix | ρ 0.365 (Choice), 0.347 (Score); errors caught 49.1% (Choice); accuracy 84.5% | ρ 0.321, 0.223; errors caught 50.2%; accuracy 85.6% |
| Wikipedia toxicity (about 10 raters per comment; held out) | review readiness on an unseen two-sided social scale | ρ +0.002; tie AUROC 0.509; MAE 0.526 | ρ −0.162; tie AUROC 0.520; MAE 0.290 |
| Sarcasm corpus (6 levels; held out) | the same, on sarcasm | ρ 0.231; MAE 0.830 | ρ 0.179; MAE 0.832 |
| Humicroedit (4 levels; held out) | the same, on humour | ρ −0.032; MAE 0.845 | ρ +0.061; MAE 0.640 |

- **Review readiness.** Its ρ is above P0's on four of the five held-out sets: the three in the table other than
  Humicroedit, and Politeness (ρ +0.071 against −0.005). The paired intervals (`results/general_v5c_26b.md`) are
  significant on Wikipedia toxicity and both MultiNLI questions. On Humicroedit it is significantly worse.
- **Wikipedia toxicity.** The E4B head's limitation there (its confidence a worse guide than its P0's) does not carry
  over. This head's confidence is not a worse guide than its P0's there, and its own ρ is about 0 (chance), against
  the prompted 26B's −0.162. Its ratings are further from the raters' than P0's (MAE 0.526 against 0.290), and it
  still puts about 25% of its probability on the two extreme levels (7.2% on "very toxic" and 17.3% on "very
  healthy"), against 4% of the raters' votes.
- **Humicroedit.** It rates edited headlines as funnier than the raters do (a mean predicted level of 1.75 against the
  raters' 0.96, the E4B head 1.35). Its ranking of the headlines is about the E4B head's (correlation with the raters'
  mean 0.30 against 0.33).
- **The coding study** used on the E4B head's card has not been scored with this head.

## Limitations and biases

- **Accuracy against the prompted 26B.** It is less accurate than P0 on intents in ten languages it was not trained on
  (5 points), on four of seven public benchmarks (2–4 points), and in rating level on Wikipedia toxicity and
  Humicroedit. On the suite's two rating tasks its errors are smaller than P0's. Check accuracy on your own task.
- **Review on the hate-speech survey.** Its confidence ranks answers by rater agreement better than P0's, but a review
  of its least confident 20% finds fewer of its errors, and in the survey's worst group its confidence runs against
  agreement. If you set a review threshold on such a task, calibrate on your own ratings first (guide §10.6).
- **Humour.** On Humicroedit its ratings are too high and its confidence is no guide to agreement. Neither task
  calibration nor a fine-tune has been tested on humour for this head. Both are the routes to try, on your own
  ratings.
- **One training run.** This head was trained once. Three E4B runs that differ only in their seed differ on the text
  development sets by up to 0.17 in ρ and up to 3 points of accuracy, 8 on one benchmark ([the E4B head's
  card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#limitations-and-biases)), so small
  differences between this head and others are not reliable. A second 26B run is planned when more GPUs are available.
- **Near-tie answers depend on packing.** The mixture-of-experts backbone picks 8 of its 128 experts per token. Even
  in float32, a request answered on its own and the same request packed with others (answered in the same forward
  pass) can differ slightly. Against the packed suite predictions, 36 of about 95,000 answers changed when each
  request was answered alone, 34 of them near-ties (top two options within 0.026), with no probability moving by more
  than 0.056 (`results/release_v5c_26b/criterion5.md`).
- **Speed and size.** 0.47 s per request and a GPU with about 72 GiB free (Hardware, above), or about 1.9 s per
  request on a Mac with 96 GB or more of unified memory.
- **Calibration on a new task.** Calibrate on about 100 labelled items of your task before setting a review threshold
  (guide §10.6). On MASSIVE's 60 intents it was overconfident zero-shot (ECE 0.099, 0.031 after the new-task factor).
- **One pass, no reasoning; inherited judgement; language; self-contained options.** As for the E4B head ([its
  card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#limitations-and-biases)). Here the
  teacher and the backbone are the same model, so the head can share that model's systematic errors and biases more
  directly.
- **One image task, no audio.** Of images, it is trained only on VizWiz-VQA's photo questions, and it has not been
  scored on another image set. The backbone takes no audio.

## Roadmap

The same as for the E4B head ([its
card](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/model-card.md#roadmap)): 0.2 in mid to late November
2026 adds two-sided human ratings and the two same-task sources held back until the first release
(HelpSteer2-Preference and the training portion of Measuring Hate Speech), and 1.0 by the end of November 2026 adds
images and audio with multi-rater labels and the stricter gates. Criterion 8 for this head is measured in a later 0.x
release.

## Licence

The code and this head are released under Apache-2.0. The head is used with Gemma 4 26B A4B, released by Google
DeepMind under Apache-2.0. See its model card for its terms. Training datasets keep their own licences, and
attribution for CC BY sources is in
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
