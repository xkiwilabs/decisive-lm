# 7. Using the general head

The general head turns Gemma 4 E4B into a decision model for questions it has never seen: you write the question and
its options, and it answers without examples. This chapter describes what the head is for, how to use its confidence
to decide which answers a person should check, the limits on its input, and when a fine-tune for your own task is the
better choice. Results are reported on the model card ([docs/model-card.md](../model-card.md)) and against the
release criteria in [chapter 15](15-release-criteria.md).

## 7.1 What the general head is for

The general head answers Choice, Score and Noul questions about text and JSON states in any domain, zero-shot. It is
trained on a broad mix (chapter 13):

- human-labelled public datasets covering customer intents, the quality of chatbot responses, toxicity, emotions and
  legal contract provisions;
- rewritten versions of those datasets' questions, so the same states are asked about with different option sets,
  names and Noul statements;
- question sets and states written by an open teacher model across many settings, from support tickets and finance to
  moderation, documents and the qualitative coding of short free-text answers, labelled with the teacher's calibrated
  probabilities.

It is a good fit when:

- you have a decision to make about many items and no labelled examples yet;
- the decision can be explained in the question and the option descriptions;
- you want to know which answers are worth checking, not just the answers.

It is also the starting point for evaluating a task: run it on a sample, look at where it is unsure, and decide whether
its answers are good enough, need a review step, or call for a fine-tune.

## 7.2 How to read its answers

Every answer is calibrated on human labels. Calibration fits one temperature per question type and number of options
(2, 3–5, 6–15, 16–63 and 64–255 options) and a bias for Nouls, on held-back human-labelled questions of the tasks the
head was trained on. On a task it has not seen, it is usually underconfident until you calibrate it on about 100 of
your own labelled items (chapter 10.6). After that step, in practice:

- **Choice and Score probabilities** can be read as the chance that a person would pick each option. Across many
  answers given at 0.7, about 70% match the human label.
- **Noul** values are the probability that the statement is true. A value near 0.5 means the model cannot tell, not
  that the statement is half true.
- **Score** `score` is the expected level. A score of 1.5 on a 0–3 scale with probability split between levels 1 and 2
  says the item is on the boundary; the `probabilities` show how the mass is spread.
- **Confidence** (Choice and Score) summarises how peaked the distribution is. It ranks answers from most to least
  certain; the probabilities themselves carry the calibrated meaning.

Calibration holds on average over many items of the kinds the head was calibrated on. On a new task the probabilities
are a good starting point, but check them on a labelled sample of your own (section 7.4) before relying on a threshold.

## 7.3 Reviewing the least confident answers

The intended workflow for a large batch:

1. Run the head on every item (chapter 3, `scripts/predict_bundle.py` for files).
2. Sort the answers by confidence.
3. Send the least confident share to a person, and accept the rest.

The share to review is a trade-off between effort and accuracy. Two numbers describe it:

- **errors caught**: the share of the model's errors that fall inside the reviewed set;
- **accuracy of the rest**: how accurate the answers you accept without review are.

If the model's confidence carried no information, reviewing 20% of the answers would catch 20% of its errors. A
review-ready model catches much more, because its errors concentrate among its least confident answers. Its
confidence also tends to be low exactly where people disagree, so the reviewed set holds the genuinely ambiguous items
as well as the mistakes.

For a Noul, rank by how far the probability is from a decision: `abs(noul − 0.5)`, smallest first.

## 7.4 Choosing a review threshold on your own sample

Before you rely on a review share, measure it on a few hundred items you have labelled yourself. `decisivelm.eval`
provides the measures:

```python
import numpy as np
from decisivelm.eval import review_curve

# conf: the model's confidence per item; pred and gold: option indices (gold = -1 where your labellers tied)
conf, pred, gold = np.array(conf), np.array(pred), np.array(gold)
scored = gold >= 0
fractions = [0.1, 0.2, 0.3]
caught, rest_accuracy = review_curve(conf, pred == gold, scored, fractions)
for f, c, r in zip(fractions, caught, rest_accuracy):
    print(f"review {f:.0%}: catches {c:.0%} of errors, the rest is {r:.1%} accurate")
```

`review_curve(conf, correct, scored, fractions)` sends the least confident fraction `f` of all items to review and
returns, for each `f`, the share of errors inside the reviewed set and the accuracy of the unreviewed rest. Items
without a human consensus (`scored` false) are ranked like the others but counted in neither number.

Pick the smallest share that gives the accuracy you need for the accepted answers. Then fix the threshold as a
confidence value, not a share, so it carries over to new batches: the confidence of the item at that position in your
sorted sample.

If several people labelled your sample, you can also check that the model's confidence follows their agreement:

```python
from decisivelm.eval import pair_agreement, spearman, within_category_spearman

# labels: int array [items, raters], -1 where a rater did not label the item
agreement = pair_agreement(labels)                 # share of rater pairs that agree, per item
print(spearman(conf, agreement))                   # rank correlation of confidence with agreement
print(within_category_spearman(conf, agreement, consensus_label, min_n=25))
```

A positive correlation within each category shows that the confidence finds the hard items inside a category, not
only which categories are hard. Chapter 11 describes these measures and the scripts that report them with confidence
intervals.

## 7.5 Input limits

| Limit | Value | What happens beyond it |
|---|---|---|
| State length | 8,192 tokens | the state is shortened to its first 70% and last 30% of the budget, joined by `[…]` |
| Choice options | 2–255 | the request is rejected |
| Score levels | 2–10 | the request is rejected |
| Images | 4 per request | the request is rejected |
| Audio | not supported | the request is rejected |

- **JSON states** are written compactly, with keys in the order you gave them. Put the fields that matter early if the
  state may be shortened.
- **Instructions and criteria** may be strings, objects or arrays, for example `{"what": …, "examples": […]}`; objects
  and arrays are written as compact JSON.
- **Cost** grows with the length of the state and the total length of the options. The state is read once per request
  however many questions you ask; each option adds only its own description.

## 7.6 Languages and modalities

The general head is trained mostly on English text. It answers in other languages through the multilingual Gemma 4
backbone, and the frozen held-out suite measures this on ten languages it was not trained on (chapter 15). Check its
accuracy on a sample before relying on it in another language.

The head was trained and evaluated on text and JSON states only. The toolkit accepts images and audio
([chapter 4](04-requests-and-responses.md), section 4.5), and image and audio tasks are served by a fine-tune trained
on labelled examples of your own ([chapter 9](09-fine-tuning-for-your-task.md), section 9.9). Starting that fine-tune
from the general head can help even though the head never saw media.

## 7.7 When to fine-tune instead

The general head does the most with questions whose meaning is in their wording. A fine-tune on your own labels
(chapter 9) is the better choice when:

- **the rules are long or have precedence**: a codebook that says "code as B unless the response also mentions C,
  in which case D" asks for careful reading of a long instruction, which a single pass does less well than a model that
  has seen labelled examples;
- **your labels follow local conventions**: the line between two codes is drawn by your team's practice rather than by
  the words of the codebook;
- **you need the best review readiness on one task**: a fine-tune learns where *your* coders disagree. With a few
  hundred ratings but no wish to train, task calibration ([chapter 10](10-calibration.md), section 10.6) fits how your
  raters use each option or level; on a two-sided rating scale the head has not seen, it is what turns the head's
  confidence into a review signal;
- **your states are images or audio**, or a kind of text far from the training mix;
- **you have labelled data already**: a few hundred labelled questions are enough to train and calibrate a fine-tune,
  and the general head's answers on a sample tell you how much you would gain.

## 7.8 Intended uses

- Routing, triage and tagging of messages, tickets and documents.
- Rubric scoring and quality judgements, with a review step for the uncertain items.
- Coding qualitative data (survey answers, open-ended responses, transcripts) as a first coder whose confidence says
  which items a human coder should see.
- Screening and extraction expressed as Nouls ("the document mentions a termination date").
- Measuring a task before deciding whether to fine-tune.

## 7.9 Out-of-scope uses

- **Decisions about people without your own checks.** Do not use the head, unreviewed, for clinical, credit, hiring,
  housing, legal or other decisions that affect people. If a decision affects people, evaluate the head on your own
  labelled data, check its errors across the groups concerned, and keep a person in the loop.
- **Free text.** The head does not explain, summarise or extract arbitrary spans; express what you need as Choice,
  Score and Noul questions.
- **More than 255 options.** Split the decision into stages, for example a coarse Choice and then a fine one.
- **Facts the state does not contain.** The head judges the state you give it; questions about the outside world are
  answered from what the backbone happens to know, with no guarantee.

Next: [8. Data format](08-data-format.md)
