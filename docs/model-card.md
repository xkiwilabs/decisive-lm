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

# DecisiveLM E4B general head (0.1 preview)

A general-purpose decision head for Google's Gemma 4 E4B. Given a state (text or JSON) and a set of typed questions,
it answers every question in one forward pass with a calibrated probability distribution over the options you supply.
It answers new questions in any domain without examples. Its confidence is built to say which answers a person should
check.

- **Code**: [github.com/xkiwilabs/decisive-lm](https://github.com/xkiwilabs/decisive-lm) (Apache-2.0)
- **Documentation**: [the guide](https://github.com/xkiwilabs/decisive-lm/tree/main/docs/guide), starting with the
  [quickstart](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/03-quickstart.md)
- **Base model**: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) at revision
  `ee0ef6023621cff504d758262d4e04895a5af4a2`, which this head requires

> **Known limitation of this preview.** On two-sided social rating scales the head has not been trained on (for
> example "strongly respectful … neither … strongly disrespectful"), its confidence follows how extreme the text is
> rather than how much people agree, and it is a worse guide to which answers to check than the prompted base model.
> This is the one release criterion it fails (Measuring Hate Speech's survey items). On such a task, calibrate each
> level on a few hundred of your own ratings (task calibration, guide §10.6) or fine-tune on them. See
> [Limitations](#limitations-and-biases).
>
> **Calibrate before you set a review threshold.** Zero-shot, the head is usually underconfident on a new task.
> Fitting two numbers on about 100 labelled items of your task fixes this
> ([guide §10.6](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/10-calibration.md)).

## Summary

| | |
|---|---|
| Question types | Choice (2–255 unordered options), Score (2–10 ordered levels), Noul (probability a statement is true) |
| Input | text or JSON state up to 8,192 tokens (longer states are shortened head and tail) |
| Output | a probability per option or level, the chosen option or expected level, a confidence; for Noul one probability |
| Independence | each question, and each option, is scored on its own: adding or removing questions or shuffling options does not change other answers |
| Calibration | temperatures per question type and number of options, and a Noul bias, fitted on human-labelled questions of the training mix; on a new task, calibrate on about 100 of your own labelled items |
| Size | 17.4M trained parameters in the backbone (LoRA and query-token rows) and a 2.6M-parameter scorer, on top of the frozen backbone |

## Intended use

- Routing, triage and tagging of messages, tickets and documents.
- Rubric scoring and quality judgements with a review step for uncertain items.
- Coding qualitative data, such as survey answers and open-ended responses, as a first coder whose confidence says
  which items a human coder should see.
- Screening and extraction expressed as yes/no statements (Nouls).
- Measuring a new decision task zero-shot before deciding whether to train a task-specific fine-tune.

Check accuracy and the review threshold on a labelled sample of your own task before relying on the head
([guide chapter 7](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/07-using-the-general-head.md)).

## Out-of-scope use

- Unreviewed decisions that affect people: clinical, credit, hiring, housing, legal, policing or benefits decisions.
  Such uses need your own labelled evaluation, checks of errors across the groups affected, and a person in the loop.
- Free-text generation, explanation or summarisation.
- Choice questions with more than 255 options.
- Images and audio: the head was trained and evaluated on text and JSON only.
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

## Training data

The training mix is release tier only: every source carries a licence that allows training and redistribution of the
result (guide chapter 16). 205,433 training requests (976,638 questions: 227,215 Choice, 190,571 Score, 558,852
Noul) of three kinds:

| Kind | Sources | Labels |
|---|---|---|
| Human-labelled public datasets | Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0), HelpSteer2 (CC BY 4.0), Civil Comments (CC0 1.0), GoEmotions (Apache-2.0), LEDGAR (CC BY 4.0) | the share of human raters who chose each label |
| Rewritten questions on the same states | subsets of the intent sets' options, options removed with a self-contained "other", intents asked as yes/no statements | carried over from the human labels |
| Teacher-written question sets and states | question sets in many settings (support, finance, moderation, documents, qualitative coding of free-text answers and more), with states written to be easy, borderline, ambiguous, irrelevant or open-ended, and plainly worded Scores with a neutral middle; and new question sets about real states of the public datasets | Gemma 4 26B A4B's label probabilities, averaged over option orderings and calibrated on human labels |

Each release publishes its mix manifest: every source, its licence, the row counts per split and the sha256 of the
mix. The public datasets are linked rather than re-hosted, and the teacher-written rows and their labels are published
in the companion dataset repository. Details: [guide chapter 13](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/13-how-the-general-head-was-trained.md)
and [chapter 14](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/14-datasets.md).

Splits are assigned per source item and question family before any rewriting, so no test item or its rewrites reach
training. The frozen held-out suite (below) is listed in `configs/holdout.yaml`, and its text was checked for overlap
with every trainable row.

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
| Selection | validation every 500 steps on 1,500 held-back rows; the checkpoint with the lowest calibrated validation loss is kept (this release: step 9,000 of 25,680, after about 12 hours on one RTX PRO 6000) |
| Calibration | one temperature per question type and option-count bucket (2, 3–5, 6–15, 16–63, 64–255), with buckets of fewer than 100 questions sharing their type's temperature, and a Noul temperature and bias, fitted on the human-labelled `gold_calib` rows of the mix. No global correction for new tasks: a scale fitted on one held-out task made the head overconfident on its training tasks and could not fix errors that go both ways on new ones, so users calibrate on their own task instead (guide §10.6) |

The exact settings of each release, including the number of epochs and the selected step, are recorded in its
`calibration.json`.

## Evaluation

The head is judged first on **review readiness**, whether its confidence says which answers to check, then on
calibration and accuracy, always beside the frozen Gemma 4 E4B prompted to answer the same questions (P0) and the
Gemma 4 26B A4B teacher. The release criteria are written down before any candidate is scored
([RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md)).

### Frozen held-out suite

Four public, human-labelled sets that no training, selection or calibration decision ever sees. Each candidate is
scored on it once. Scores below are on each set's test split less the 100 rows used for the per-task calibration
step; ECE is shown zero-shot and after that step. Full report: `results/suite_general_v3d_e4b_r8_seed0.md`.

| Task | What it tests | ρ(confidence, agreement) (P0) | tie AUROC (P0) | errors caught in a 20% review (P0) | ECE zero-shot / after 100 rows | primary metric | P0 primary metric |
|---|---|---|---|---|---|---|---|
| HelpSteer3 preference (Score, 6 levels) | judging chatbot responses beyond the training set | 0.063 (0.043) | 0.508 (0.518) | 21.7% (21.0%) | 0.029 / 0.029 | **1.010** (MAE) | 1.292 |
| Measuring Hate Speech (9 Scores) | calibration against real rater disagreement | **0.142 (0.201)** | **0.556 (0.578)** | **22.5% (23.8%)** | 0.092 / 0.080 | **0.620** (MAE) | 0.657 |
| Measuring Hate Speech (Noul) | the same, as a yes/no judgement | | | | 0.061 / 0.061 | 0.865 (AUROC) | 0.855 |
| CUAD, 10 held-out clause types (Noul) | unseen questions in a known domain | | | | 0.015 / 0.015 | 0.951 (AUROC) | 0.945 |
| MASSIVE, 10 held-out languages (Choice, 60 intents) | zero-shot language transfer | | | | 0.115 / 0.023 | **0.676** (accuracy) | 0.658 |

Bold: significantly different from P0 (paired 95% bootstrap). On Measuring Hate Speech's Scores, P0 is significantly
better on all three review measures; that is this preview's failed criterion.

Release criteria (pass or fail, with paired bootstrap confidence intervals against P0):

| Criterion | Result |
|---|---|
| 1. Review readiness not worse than P0 | **fails on Measuring Hate Speech's Scores**; passes on HelpSteer3 |
| 2. Calibration (top-label ECE ≤ 0.10 on every task, after the per-task step) | passes (0.015–0.080) |
| 3. Accuracy not worse than P0 | passes; better on 3 of 5 tasks |
| 4. Close to the task-specific fine-tunes on the training datasets' own test sets | passes on all 7 seed tasks |
| 5. Contract: conformance, independence and order | passes: conformance 100% on every suite item (41,642 requests); reversing options, packing another request first or adding a question moves no probability by more than 0.02 (bf16 rounding; an identical request reproduces exactly) |
| 6. Provenance | passes: release-tier mix with its manifest; Gemma 4 teacher and backbone; clean-room checks pass |
| 7. Snapshot audit | passes: allowlisted paths only, the test suite green, no secrets, private paths or private data |
| 8. A better starting point for fine-tunes (1.0; reported for 0.x) | passes: fine-tunes on 100 or 300 items of the coding study and MultiNLI, started from this head, are never significantly worse than ones started from the base model, and better on accuracy and review readiness at 100 items (coding: 91.1% against 86.6%) |

This head is released as a 0.x preview although it fails criterion 1 on one task. That decision was made after its
suite result was seen, and the criteria's change log records it
([RELEASE.md](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/RELEASE.md), 2026-10-27).

### Training datasets' own test sets

| Dataset | Metric | Task-specific fine-tune | This head |
|---|---|---|---|
| Banking77 | accuracy | 0.902 | 0.892 |
| CLINC150 | accuracy | 0.920 | 0.947 |
| HelpSteer2 | mean MAE | 0.468 | 0.478 |
| Civil Comments | mean AUROC | 0.982 | 0.978 |
| GoEmotions | accuracy / mean AUROC | 0.760 / 0.952 | 0.744 / 0.948 |
| LEDGAR | accuracy | 0.821 | 0.806 |

Every difference is inside the pre-registered margin (3 points of accuracy or AUROC, 0.05 levels of MAE), and on the
multi-rater sets (HelpSteer2, GoEmotions) the head's review readiness is not significantly different from the
fine-tune's.

### Development sets

These sets were used to choose between candidate heads, so they are reported as development results, not tests.
P0 and the teacher are calibrated the same way as the head.

| Set | What it shows | This head | P0 |
|---|---|---|---|
| Coding study: 792 free-text answers, 8 codes, 4 human coders (Alhasan, Diaz Monfort & Richardson, under review; data not distributed) | a rule-heavy coding task, zero-shot | κ with coders 0.703 (coders with each other 0.739); errors caught in a 20% review 52.3%; ρ 0.316; ECE 0.257 zero-shot | κ 0.649; errors caught 41.9%; ρ 0.385 |
| Seven public benchmarks (AG News, Banking77, DailyDialog, emotion, financial topics, SST-2, tweet topics) | zero-shot classification | more accurate than P0 on 3, level on 1, less on 3; ECE ≤ 0.09 after 100 rows (0.05–0.25 zero-shot) | ECE ≤ 0.08 after 100 rows |
| MultiNLI (5 annotators per pair; held out) | review readiness on a task outside the mix | ρ 0.344 (Choice), 0.236 (Score); errors caught 48.0% (Choice) | ρ 0.267, 0.191; errors caught 37.7% |
| Wikipedia toxicity (about 10 raters per comment; held out) | review readiness on an unseen two-sided social scale | ρ −0.126; tie AUROC 0.487; MAE 0.484 | ρ +0.059; tie AUROC 0.541; MAE 0.381 |

The last row is the limitation above, reproduced on development data. Wikipedia toxicity was added because of the
suite result, which is why it is reported here. Two later candidates trained with more human ratings (politeness;
humour and sarcasm) learned to be unsure on those tasks but did not improve on toxicity, and were not released
(paper, Section 9). With task calibration on toxicity's 2,040 calibration rows (a scale and an offset per level,
fitted on the task's own ratings), this head's ρ is +0.305 and MAE 0.227, against P0's +0.293 and 0.277 with the same
step: raters there rarely use the ends of the scale, which the head could not know zero-shot.

## Limitations and biases

- **Review readiness on unseen two-sided scales.** On social rating scales with a neutral middle that the head has
  not been trained on, its confidence tracks how extreme the text is, not how much people agree. On Measuring Hate
  Speech's survey items and on Wikipedia toxicity it is a worse guide to which answers to check than the prompted base
  model, and it chooses the neutral middle less often than people do. With a few hundred of the task's own ratings,
  task calibration (guide §10.6) or a fine-tune fixes this on development data, so use one of them for such a task.
- **Calibration on a new task.** Confidence is well calibrated on the tasks the head was trained on. On a new task it
  is usually underconfident zero-shot, and on some (the suite's 60-intent task in languages it was not trained on)
  overconfident. Calibrate on about 100 labelled items of your task before setting a review threshold (guide §10.6);
  the release criteria measure calibration after that step.
- **One pass, no reasoning.** The head reads each option once with the state. Long instructions with precedence rules
  ("code B unless C is also mentioned") are answered less well than by a model prompted to reason step by step; a
  fine-tune on labelled examples closes much of that gap.
- **Inherited judgement.** Part of the training labels come from Gemma 4 26B A4B. The head can share that model's
  blind spots and biases (its avoidance of neutral middles is one), and the human-labelled datasets carry the views of
  their raters.
- **Language.** Training is mostly English. Other languages are answered through the multilingual backbone and
  measured on the held-out suite.
- **Self-contained options.** Each option is judged without seeing the others, so an "other" option needs a
  description of what it covers ("about something other than payments, outages or pricing").
- **Text only.** This head is not trained or evaluated on images or audio. The toolkit fine-tunes on them, and the
  guide's image and audio examples (§9.9) start from this head.

## Roadmap

Each version is released only when it passes the release criteria in force for it, so dates are estimates and move
rather than the bar.

| Version | Estimated | Main change |
|---|---|---|
| 0.1 | Early November 2026 | This preview: English text and JSON on Gemma 4 E4B |
| 0.2 | Mid November 2026 | Review readiness on two-sided scales: 600 public texts rated by five people each through our open rating app (agree–disagree, sure–unsure, formal–informal, serious–joking, convincing–unconvincing), first as a check set, then as training data; the deferred training portions of two suite tasks join the mix |
| 0.3 | Late November 2026 | A general head on Gemma 4 26B A4B; human-labelled data in languages other than English |
| 0.4 | Early December 2026 | Images (initial pass): photographs, documents and charts as states, with multi-rater image labels and an image suite |
| 0.5 | Mid December 2026 | Audio (initial pass): speech and other sound as states, with human-rated audio tasks and an audio suite |
| 1.0 | By the end of December 2026 | Better than the prompted base model on review readiness across the suite, the stricter calibration gate, and a better starting point for fine-tunes at 100 items |

## Licence

The code and this head are released under Apache-2.0. The head is used with Gemma 4 E4B, released by Google DeepMind
under Apache-2.0; see its model card for its terms. Training datasets keep their own licences (above); attribution for
CC BY sources is in [guide chapter 14](https://github.com/xkiwilabs/decisive-lm/blob/main/docs/guide/14-datasets.md).

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
