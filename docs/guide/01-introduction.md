# 1. Introduction

DecisiveLM is an open-source "System One" decision model built on Google's Gemma 4. You give it a state (a piece of
text, a JSON object, an image or an audio clip) and a set of typed questions about that state. It answers every
question in a single forward pass (one pass of the model, with no text generation), with a calibrated probability
distribution, and the answer is always one of the options you supplied. This chapter explains what that means, why
the confidence matters as much as the answer, and what the project provides.

## 1.1 What a System One decision model is

Many practical uses of language models are decisions rather than writing: which team should handle this ticket, how
severe is this complaint, does this clause cap liability, which of eight codes fits this survey answer. A general chat
model can make these decisions when prompted, but it produces free text that has to be parsed, it can answer outside
the options, it costs a full generation per question, and its stated certainty means little.

A System One model is built for the decision itself:

- **Constrained.** Every answer is a distribution over the options or levels in the request, never anything else.
- **Fast.** All questions about one state are answered together in one pass of the model, with no text generation.
  The state is read once, and each question adds only the tokens of its own text and options. A 1,000-token state
  with four five-option questions is 1,240 tokens, answered in 75 ms with E4B on one NVIDIA RTX PRO 6000 GPU
  ([results/M0.md](../../results/M0.md)).
- **Independent.** Each question is judged on its own. Adding, removing or reordering questions, or shuffling the
  options of a question, does not change any other answer. This is guaranteed by the way the input is built
  (chapter 6), not left to training. Rounding still moves answers slightly, by less than 0.0002 in fp32
  (32-bit) arithmetic and by about 0.03 on average in bf16 (16-bit), which is the default (section 11.7).
- **Calibrated.** When the model says 0.8, it is right about 80% of the time, measured against human labels. On a task
  it was not trained on, calibrate it on about 100 of your own labelled items first (section 10.6).

The name follows the "System One / System Two" distinction from psychology (Kahneman, *Thinking, Fast and Slow*,
2011): quick, intuitive judgement as opposed to slow, step-by-step reasoning. A System One model gives the quick
judgement and, through its confidence, tells you when the slow kind is needed.

## 1.2 The three question types

Every question is one of three types (chapter 4 has the full format):

| Type | Asks | You supply | You get back |
|---|---|---|---|
| **Choice** | Which one of these unordered options fits? | 2–255 named options, each with a description | the chosen option, a probability per option, a confidence |
| **Score** | Which level on an ordered scale fits? | 2–10 level descriptions, in order | a fractional score (the expected level), a probability per level, a confidence |
| **Noul** | Is this statement true? | the statement, optionally what "yes" and "no" mean | one probability that the statement is true |

"Noul" is the published API's name for this type (section 4.1), which it describes as a continuous boolean. A Noul is
a yes/no question answered with a probability instead of a hard yes or no.

These three types compose. An extraction task becomes a set of Nouls ("the invoice is overdue", "the invoice is in
euros"); a rubric becomes a set of Scores; a codebook becomes a Choice, or one Noul per code.

## 1.3 Why the confidence matters

For many decision tasks a capable model is already about as accurate as a careful person. It usually cannot say
*which* of its answers to trust. If every answer comes back at 0.98, a human reviewer has to check everything or
nothing.

DecisiveLM is judged first on **review readiness**: whether its confidence tells a person which answers to check. The
yardstick is human coders. When several people label the same item, some items get unanimous labels and others split
the coders. A review-ready model is confident on the first kind and unsure on the second. A reviewer who checks only
its least confident 20% of answers then catches more than the 20% of its errors that a confidence with no information
would catch. On a study of coding free-text responses (Alhasan, Diaz Monfort & Richardson, *Knowing which
responses to check*, under review), the least confident 20% of the general head's answers (section 1.4) held 60.6% of
its errors zero-shot, against 41.9% for the prompted base model ([model card](../model-card.md)). The measures come
from that study:

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

A **head** is the small set of trained additions that turns Gemma 4 into a decision model (about 20 million
parameters in the general head). It has four parts:

- **LoRA weights** (low-rank adaptation): small trained corrections added to the language model's layers
  (section 6.6);
- **three query-token embeddings**: one reserved token per question type, placed after each option, level or
  statement, where the model's judgement of it is read (section 6.1);
- **a scorer network**, which turns the model's internal state at each query token into a score (section 6.5);
- **calibration temperatures**, which scale those scores so that the stated probabilities match how often answers are
  right on human-labelled data (chapter 10).

A head is saved and shipped as a directory called a **bundle**. The **backbone**, the Gemma 4 model a head runs on, is
not part of the bundle. Its weights are loaded from Google's release and are never changed. Loading a head adds its
LoRA layers beside them in memory (section 3.2).

The repository holds two deliverables built with the same code:

1. **The toolkit**: the model code, the data format and converters, a pipeline that labels data with a teacher model
   served by vLLM (an open-source server for language models, chapter 12), and training, calibration and evaluation
   scripts. With it you can train a decision model on your own data, on your own hardware, without sending anything
   to a third party. The teacher is a larger open model, Gemma 4 26B A4B by default, whose answer probabilities become
   training labels for rows that have no human labels. If your data already has human labels, you need neither the
   teacher nor vLLM (chapter 9).
2. **The general heads**: two heads, one for Gemma 4 E4B and one for Gemma 4 26B A4B, trained on the same data. They
   answer new questions about text and JSON states, and about photos, without examples (zero-shot), in the kinds of
   task they were trained and tested on: routing and tagging, rubric scoring and quality judgements, the coding of
   free-text answers, and screening expressed as yes/no statements. Each is published on Hugging Face with a model
   card ([E4B](../model-card.md), [26B A4B](../model-card-26b.md)), which lists its known shortfalls. The E4B head is
   the default and the quickest way to start; where this guide says "the general head" without naming a backbone, it
   means the E4B head, and section 7.10 compares the two.

With the toolkit you can also train **task-specific fine-tunes**: a head trained on your own labelled examples for one
task, such as a coding scheme for survey answers. A fine-tune is usually more accurate and better at review readiness
on its task than the general head, but it needs a few hundred labelled questions (chapter 9).

## 1.5 Backbones

DecisiveLM runs on two Gemma 4 models. Nothing in the code is tied to one of them. Each has a profile in
`configs/model.yaml` and its own heads.

| | **Gemma 4 E4B** (default) | **Gemma 4 26B A4B** |
|---|---|---|
| Size | about 4B effective parameters | 25.2B parameters, a mixture of experts with about 3.8B active per token |
| Inputs DecisiveLM uses | text, JSON, images, audio | text, JSON, images |
| Memory for the weights | fits a single consumer or workstation GPU, or Apple silicon | about 50 GB in bf16 |
| General head | published (`decisivelm-e4b-general`) | published (`decisivelm-26b-a4b-general`), needs a GPU with about 72 GB free |

Audio needs E4B, because 26B A4B has no audio encoder. Both backbones are developed by Google DeepMind and released
under Apache-2.0.

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
