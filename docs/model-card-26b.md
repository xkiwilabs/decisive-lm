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

<!-- DRAFT 2026-11-05: the 26B A4B head of 0.1 (owner, 2026-11-05, RELEASE.md). Criterion 8 is run on the weekend of
7–8 November and filled in then; criteria 6 and 7 are rechecked on the release snapshot in the dry run. -->

# DecisiveLM 26B A4B general head 0.1 (preview)

A general-purpose decision head for Google's Gemma 4 26B A4B, a mixture-of-experts model that uses about 4B of its 26B
parameters for each token. The head is about 12M trained parameters (LoRA adapters, embeddings for three reserved
tokens and a scorer network, see Summary) used with the base model, whose other weights are not retrained. Given a
state (text, JSON or a photo with text) and a set of typed questions, it answers every question in one forward pass
(one run of the model over the input, with no text generated), with a probability distribution over the options you
supply. It is trained on the same data as the E4B head ([its card](model-card.md)), so the two can be compared row for
row.

**How it compares.** It is compared with Gemma 4 26B A4B without the head, prompted to answer the same questions
(P0), which is also the model that labelled part of its training data.
- **Against the E4B head**, it is better on the main measure of every held-out suite task. Its confidence is a much
  better guide to which hate-speech survey answers to check (ρ 0.177 against 0.078).
- **Against P0**, its confidence is the better guide to which answers to check on the hate-speech survey's ratings and
  on four of five held-out development sets. Its rating errors are smaller on the suite's two rating tasks.
- **It is less accurate than P0** on intents in ten languages it was not trained on (68.0% against 73.1%), on four of
  seven public benchmarks (by 2–4 points), and on two rating sets (Wikipedia toxicity and Humicroedit).
- **It is slower and larger than the E4B head**: about 0.47 s per request against 0.10 s, on a GPU with about 72 GB
  free (Hardware, below).

If you have a 24 GB GPU or need speed, use the E4B head.

- **Code**: [github.com/xkiwilabs/decisive-lm](https://github.com/xkiwilabs/decisive-lm) (Apache-2.0)
- **Documentation**: [the guide](https://github.com/xkiwilabs/decisive-lm/tree/main/docs/guide), starting with the
  [quickstart](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/03-quickstart.md)
- **Base model**: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) at revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, which this head requires

> **Known limitations of this preview.** It fails five checks of the
> [release criteria](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), and was released with them
> stated (Evaluation, below). Each is a comparison with P0, a stronger model than the E4B head's P0:
>
> 1. **Intents in languages it was not trained on** (MASSIVE): 68.0% right against 73.1%.
> 2. **Which hate-speech survey answers to check.** Its confidence ranks answers by rater agreement better than P0's,
>    but a review of its least confident 20% catches fewer of its errors (23.4% against 26.7%). In the worst of 44
>    groups of that survey, its confidence runs against rater agreement more than P0's does (ρ −0.205 against −0.175).
> 3. **The worst group of HelpSteer3**, a chatbot-response comparison task: ρ −0.038 against −0.011, both close to
>    chance.
> 4. **Calibrating on 100 items of the hate-speech survey's ratings** leaves their Brier score very slightly worse
>    than before (0.6403 against 0.6399).
>
> On Humicroedit, a humour-rating set held out of training, it rates edited headlines as funnier than people do, and is
> worse than P0 there on both accuracy and review readiness. See [Limitations](#limitations-and-biases).
>
> **Calibrate before you set a review threshold.** Fitting two numbers on about 100 labelled items of your task brought
> its expected calibration error to 0.04 or below on every task of the held-out suite
> ([guide §10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)).

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

As for the E4B head ([its card](model-card.md#intended-use)): routing, triage and tagging; rubric scoring with a review
step; coding qualitative data as a first coder whose confidence says which items a human should see; screening as
yes/no statements; measuring a new task zero-shot; and questions about photos. Choose this head over the E4B head when
a GPU with about 72 GB free is available and the extra review readiness matters more than speed. Check accuracy and
the review threshold on a labelled sample of your own task before relying on the head
([guide chapter 7](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/07-using-the-general-head.md)).

## Out-of-scope use

- Unreviewed decisions that affect people: clinical, credit, hiring, housing, legal, policing or benefits decisions.
  Such uses need your own labelled evaluation, checks of errors across the groups affected, and a person in the loop.
- Free-text generation, explanation or summarisation.
- Choice questions with more than 255 options.
- Audio. The server refuses it with a 400 error.
- Images beyond photo questions without calibration on your task: it was trained on one photo task (VizWiz-VQA), and
  has not been scored on any other image set.
- Questions whose answer is not in the state.
- Work that needs answers in about 0.1 s, or a GPU with less than about 72 GB free: use the E4B head.

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
`calibration.json` (temperatures, Noul bias, backbone revision, the float32 setting and training settings). The
server (`python -m decisivelm.serve`, guide §4.10) needs `DECISIVELM_MEMORY_GB=72`; one server holds heads of one
backbone only.

### Hardware

| | Running the head | Fine-tuning it on your task |
|---|---|---|
| Hardware | Linux with an NVIDIA GPU (CUDA). It has not been run on Apple silicon | the same |
| Memory | a GPU with about 72 GB free. Through the server, 57.2 GB after loading and 68.6 GB at the peak, on requests at its default limit of 12,288 tokens, with or without a photo | not yet measured for a user fine-tune (measured with criterion 8, below). Training this head on the general mix peaked at 67.6 GB with micro-batches of 4,096 tokens and gradient checkpointing |
| Time | 0.47 s for a request of about 1,200 tokens with 20 options (four questions of five), batch 1, on one RTX PRO 6000 Blackwell Max-Q; 5.7–5.8 s for requests of about 12,000 tokens | 1.38 s per training row on the general mix |
| Disk | about 49 GB for Gemma 4 26B A4B and about 1 GB for the Python environment | the same, plus your data |

The peaks are per process, as `nvidia-smi` reports them, so they include the CUDA context. Float32 computation is the
main cost: the bare backbone in bf16, with no head, took 0.14 s on the same request, and the E4B head takes 0.10 s.

## Training data

The same mix as the E4B head, General v5c: 215,433 training requests (995,083 questions) from human-labelled public
datasets, 10,000 VizWiz-VQA photos with questions, rewritten questions on the same states, and teacher-written
question sets and states labelled by Gemma 4 26B A4B. Every source is release tier. The sources, licences and labels
are listed on [the E4B head's card](model-card.md#training-data), and in
[guide chapter 13](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/13-how-the-general-head-was-trained.md)
and [chapter 14](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/14-datasets.md). The teacher is the
same model as this head's backbone.

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
| Readout and losses | as for the E4B head ([its card](model-card.md#training-procedure)) |
| Optimiser | learning rate 3e-5 for LoRA, 1e-4 for the head and query rows; 10% warm-up; 16 rows per step; packed micro-batches of up to 4,096 tokens with gradient checkpointing |
| Selection | validation every 500 steps on the E4B head's 1,500 held-back rows; the checkpoint with the lowest calibrated validation loss is kept, and training stops after three checks without improvement (this release: step 15,000, stopped at 16,500 of at most 26,930, about 101 hours on one RTX PRO 6000 Blackwell Max-Q) |
| Calibration | as for the E4B head: one temperature per question type and option-count bucket, and a Noul temperature and bias, fitted on the mix's human-labelled `gold_calib` rows |

## Evaluation

The head is judged first on **review readiness**, whether its confidence says which answers to check, then on
calibration and accuracy. **P0 here is Gemma 4 26B A4B prompted**, the head's own backbone ("better off than if
they had prompted the base model it is built on"), and so also the teacher. This rule was written down before either
head was trained (RELEASE.md, 2026-10-30). It makes this head's bar higher than the E4B head's, whose P0 is the
prompted E4B. The criteria in force for this head were fixed before it was scored. It was released by a decision
made after its result was seen (below). The measures are defined on [the E4B head's card](model-card.md#evaluation)
and in guide chapters [10](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md) and
[11](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/11-evaluation.md).

### Frozen held-out suite

The same four sets and the same per-task step as for the E4B head, scored once. Full report:
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
| 2. Calibration after the per-task step: ECE ≤ 0.10 on every task, and Brier not significantly worse than before any calibration | ECE passes on every task (0.018–0.040). **Brier fails on Measuring Hate Speech's Scores**: 0.6403 calibrated against 0.6399 uncalibrated, Δ [+0.0002, +0.0006] |
| 3. Accuracy not worse than P0 | **fails on MASSIVE**: 0.680 against 0.731, Δ [−0.055, −0.046]. Better on 3 of 5 tasks, level on CUAD |
| 4. Close to the task-specific fine-tunes on the training datasets' own test sets | passes all 20 checks, review readiness included (table below) |
| 5. Contract: conformance, independence and order | passes: conformance 100% on every suite item (41,642 requests). Reversing options, packing another request first or adding a question changes no answer and moves no probability by more than 0.011; an identical request reproduces exactly |
| 6. Provenance | passes: the same release-tier mix and manifest as the E4B head; Gemma 4 teacher and backbone; clean-room checks pass |
| 7. Snapshot audit | passes: the same repository snapshot as the E4B head |
| 8. A better starting point for fine-tunes (1.0; reported for 0.x) | not yet measured; scheduled for 7–8 November 2026, with the E4B head's recipe (coding study and MultiNLI, 100 and 300 items, three seeds) |

This head is released as a preview although it fails five checks. That decision was made after its suite result was
seen, and the criteria's change log records it
([RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), 2026-11-05).

### Against the E4B head

The two heads on the suite, each from its own suite report (not paired, so no intervals; this head is higher on
every row):

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
[−0.018, −0.004]). It fails more checks than the E4B head because its P0 is stronger: the prompted 26B is far more
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
| Seven public benchmarks (AG News, Banking77, DailyDialog, emotion, financial topics, SST-2, tweet topics) | zero-shot classification | more accurate than P0 on 1 (Banking77, 81.8% against 76.7%), level on 2, less on 4 by 2–4 points; ECE 0.04–0.09 after 100 rows (0.06–0.30 zero-shot) | ECE 0.02–0.16 after 100 rows |
| MultiNLI (5 annotators per pair; held out) | review readiness on a task outside the mix | ρ 0.365 (Choice), 0.347 (Score); errors caught 49.1% (Choice); accuracy 84.5% | ρ 0.321, 0.223; errors caught 50.2%; accuracy 85.6% |
| Wikipedia toxicity (about 10 raters per comment; held out) | review readiness on an unseen two-sided social scale | ρ +0.002; tie AUROC 0.509; MAE 0.526 | ρ −0.162; tie AUROC 0.520; MAE 0.290 |
| Sarcasm corpus (6 levels; held out) | the same, on sarcasm | ρ 0.231; MAE 0.830 | ρ 0.179; MAE 0.832 |
| Humicroedit (4 levels; held out) | the same, on humour | ρ −0.032; MAE 0.845 | ρ +0.061; MAE 0.640 |

- **Review readiness.** Its ρ is above P0's on four of the five held-out sets: the four in the table other than
  Humicroedit, and Politeness (ρ +0.071 against −0.005). The paired intervals (`results/general_v5c_26b.md`) are
  significant on Wikipedia toxicity and both MultiNLI questions. On Humicroedit it is significantly worse.
- **Wikipedia toxicity.** The E4B head's limitation there (its confidence a worse guide than its P0's) does not carry
  over: this head's ρ is about 0, against the prompted 26B's −0.162. Its ratings are further from the raters' than
  P0's (MAE 0.526 against 0.290), and it still puts far more probability on the two extreme levels than raters do.
- **Humicroedit.** It rates edited headlines as funnier than the raters do (a mean predicted level of 1.75 against the
  raters' 0.96, the E4B head 1.35). Its ranking of the headlines is about the E4B head's.
- **The coding study** used on the E4B head's card has not been scored with this head.

## Limitations and biases

- **Accuracy against the prompted 26B.** It is less accurate than P0 on intents in ten languages it was not trained on
  (5 points), on four of seven public benchmarks (2–4 points), and in rating level on Wikipedia toxicity and
  Humicroedit. On the suite's two rating tasks its errors are smaller than P0's. Check accuracy on your own task.
- **Review on the hate-speech survey.** Its confidence ranks answers by rater agreement better than P0's, but a review
  of its least confident 20% finds fewer of its errors, and in the survey's worst group its confidence runs against
  agreement. If you set a review threshold on such a task, calibrate on your own ratings first (guide §10.6).
- **Humour.** On Humicroedit its ratings are too high and its confidence is no guide to agreement. A fine-tune or task
  calibration on your own ratings is the remedy, as for the E4B head on two-sided scales (its card).
- **One training run.** This head was trained once. Two E4B runs that differ only in their seed differ on the text
  development sets by up to 0.17 in ρ and 2–3 points of accuracy ([the E4B head's card](model-card.md#limitations-and-biases)),
  so small differences between this head and others are not reliable. A second 26B run is planned when more GPUs
  are available.
- **Near-tie answers depend on packing.** The mixture-of-experts backbone picks 8 of its 128 experts per token. Even
  in float32, a request answered on its own and the same request packed with others can differ slightly. Against the
  packed suite predictions, 36 of about 95,000 answers changed when each request was answered alone, 34 of them
  near-ties (top two options within 0.026), with no probability moving by more than 0.056
  (`results/release_v5c_26b/criterion5.md`).
- **Speed and size.** 0.47 s per request and a GPU with about 72 GB free (Hardware, above). It has not been run on
  Apple silicon.
- **Calibration on a new task.** Calibrate on about 100 labelled items of your task before setting a review threshold
  (guide §10.6). On MASSIVE's 60 intents it was overconfident zero-shot (ECE 0.099, 0.031 after the step).
- **One pass, no reasoning; inherited judgement; language; self-contained options.** As for the E4B head (its card).
  Here the teacher and the backbone are the same model, so the head can share that model's blind spots more
  directly.
- **One image task, no audio.** Of images, it is trained only on VizWiz-VQA's photo questions, and it has not been
  scored on another image set. The backbone takes no audio.

## Roadmap

The same as for the E4B head ([its card](model-card.md#roadmap)): 0.2 in mid to late November 2026 adds two-sided human
ratings and the two same-task sources held back until the first release; 1.0 by the end of November 2026 adds images
and audio with multi-rater labels and the stricter gates. Criterion 8 for this head is measured on 7–8 November 2026.

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
