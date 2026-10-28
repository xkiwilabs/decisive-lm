# 1. Introduction

DecisiveLM is an open-source "System One" decision model built on Google's Gemma 4. You give it a state (a piece of
text, a JSON object or an image) and a set of typed questions about that state. It answers every question in a single
forward pass, with a calibrated probability distribution, and the answer is always one of the options you supplied. This
chapter explains what that means, why the confidence matters as much as the answer, and what the project provides.

## 1.1 What a System One decision model is

Many practical uses of language models are not about writing text. They are decisions: which team should handle this
ticket, how severe is this complaint, does this clause cap liability, which of eight codes fits this survey answer. A
general chat model can make these decisions when prompted, but it produces free text that has to be parsed, it can
answer outside the options, it costs a full generation per question, and its stated certainty means little.

A System One model is built for the decision itself:

- **Constrained.** Every answer is a distribution over the options or levels in the request, never anything else.
- **Fast.** All questions about one state are answered together in one pass of the model, with no text generation.
  Adding a question costs little extra time.
- **Independent.** Each question is judged on its own. Adding, removing or reordering questions, or shuffling the
  options of a question, does not change any other answer. This is guaranteed by the way the input is built
  (chapter 6), not just encouraged by training.
- **Calibrated.** When the model says 0.8, it is right about 80% of the time, measured against human labels.

The name follows the "System One / System Two" distinction from psychology: quick, intuitive judgement as opposed to
slow, step-by-step reasoning. A System One model gives the quick judgement and, through its confidence, tells you when
the slow kind is needed.

## 1.2 The three question types

Every question is one of three types (chapter 4 has the full format):

| Type | Asks | You supply | You get back |
|---|---|---|---|
| **Choice** | Which one of these unordered options fits? | 2–255 named options, each with a description | the chosen option, a probability per option, a confidence |
| **Score** | Which level on an ordered scale fits? | 2–10 level descriptions, in order | a fractional score (the expected level), a probability per level, a confidence |
| **Noul** | Is this statement true? | the statement, optionally what "yes" and "no" mean | one probability that the statement is true |

"Noul" is short for a continuous boolean: a yes/no question answered with a probability instead of a hard yes or no.

These three types compose. An extraction task becomes a set of Nouls ("the invoice is overdue", "the invoice is in
euros"); a rubric becomes a set of Scores; a codebook becomes a Choice, or one Noul per code.

## 1.3 Why the confidence matters

For many decision tasks a capable model is already about as accurate as a careful person. What it usually cannot do is
say *which* of its answers to trust. If every answer comes back at 0.98, a human reviewer has to check everything or
nothing.

DecisiveLM is judged first on **review readiness**: whether its confidence tells a person which answers to check. The
yardstick is human coders. When several people label the same item, some items get unanimous labels and others split
the coders. A review-ready model is confident on the first kind and unsure on the second, and a reviewer who checks
only its least confident answers catches a large share of its errors. The measures come from a study of coding
free-text responses (Alhasan, Diaz Monfort & Richardson, *Knowing which responses to check*, under review):

- **Tracking**: the rank correlation between the model's confidence and how much the human coders agreed on each
  item, overall and within each category.
- **Tie detection**: how well low confidence picks out the items on which the coders had no majority.
- **Dynamic range**: how far the model's average confidence moves between the items people agree on and the items they
  split on. A confidence that only moves from 0.986 to 0.998 ranks items but cannot carry a review threshold.
- **Confidence-guided review**: send the least confident 20% of answers to a person; how many of the model's errors
  does that catch, and how accurate is the rest?

Accuracy and calibration are reported too (chapter 11), but a model that is accurate and unhelpfully certain is not
the goal. To make confidence follow human disagreement, every model in this project is trained on the share of raters
who chose each label, never on a majority vote.

## 1.4 What the project provides

The repository holds two deliverables built with the same code:

1. **The toolkit**: the model code, the data format and converters, a pipeline that labels data with an open teacher
   model served by vLLM, training, calibration and evaluation scripts. With it you can train a decision model on your
   own data, on your own hardware, without sending anything to a third party.
2. **The general head**: a trained adapter for Gemma 4 E4B that answers new questions in any domain without examples
   (zero-shot). It is published on Hugging Face with a model card ([docs/model-card.md](../model-card.md)) and is the
   quickest way to start.

A **head** (or adapter bundle) is the small set of trained weights that turns Gemma 4 into a decision model: low-rank
adapter (LoRA) weights for the backbone, a scorer network, three query-token embeddings, and the calibration
temperatures. The backbone itself is unchanged and loaded from Google's release.

With the toolkit you can also train **task-specific fine-tunes**: a head trained on your own labelled examples for one
task, such as a coding scheme for survey answers. A fine-tune is usually more accurate and better at review readiness
on its task than the general head, at the cost of needing a few hundred labelled questions (chapter 9).

## 1.5 Backbones

DecisiveLM runs on two Gemma 4 models. Nothing in the code is tied to one of them; each has a profile in
`configs/model.yaml` and its own heads.

| | **Gemma 4 E4B** (default) | **Gemma 4 26B A4B** |
|---|---|---|
| Size | about 4B effective parameters | 25.2B parameters, a mixture of experts with about 3.8B active per token |
| Inputs DecisiveLM uses | text, JSON, images | text, JSON, images |
| Memory for the weights | fits a single consumer or workstation GPU, or Apple silicon | about 50 GB in bf16 |
| General head | published | not published |

The E4B backbone itself also accepts audio, but DecisiveLM does not yet handle audio input; a request with audio media
is rejected. Both backbones are developed by Google DeepMind and released under Apache-2.0.

## 1.6 Who it is for

- **Researchers who code text**, such as survey answers, interview excerpts or open-ended responses, and want a
  model that matches trained coders and flags the items worth a second look.
- **Developers** who need routing, triage, moderation, tagging or rubric scoring with predictable outputs and
  latency, running locally.
- **Teams that cannot send data out**: everything runs on your own machine, from labelling to serving.
- **Teams with model-origin requirements**: the backbone and the teacher are from Google DeepMind, and every model used
  anywhere in the pipeline has its developer and origin recorded (chapter 16).

## 1.7 How this guide is organised

| Part | Chapters |
|---|---|
| Getting started | [2. Installation](02-installation.md) · [3. Quickstart](03-quickstart.md) |
| Asking questions | [4. Requests and responses](04-requests-and-responses.md) · [5. Writing good questions](05-writing-good-questions.md) |
| Understanding the model | [6. How it works](06-how-it-works.md) · [7. Using the general head](07-using-the-general-head.md) |
| Training your own | [8. Data format](08-data-format.md) · [9. Fine-tuning for your task](09-fine-tuning-for-your-task.md) · [10. Calibration](10-calibration.md) · [11. Evaluation](11-evaluation.md) · [12. Teacher labelling](12-teacher-labelling.md) |
| How the general head was made | [13. How the general head was trained](13-how-the-general-head-was-trained.md) · [14. Datasets](14-datasets.md) · [15. Release criteria](15-release-criteria.md) |
| Rules and reference | [16. Provenance, licences and responsible use](16-provenance-licences-and-responsible-use.md) · [17. Reference](17-reference.md) |

If you only want to use the general head, read chapters 2, 3, 4, 5 and 7. If you want to train a head for your own
task, continue with chapters 8 to 11.

Next: [2. Installation](02-installation.md)
