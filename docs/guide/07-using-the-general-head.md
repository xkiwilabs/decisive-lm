# 7. Using the general head

The general head turns Gemma 4 E4B into a decision model for questions it has never seen. You write the question and its
options, and it answers without examples. There is a second general head, for Gemma 4 26B A4B, trained on the same data.
In this chapter, "the general head" means the E4B head unless a sentence says otherwise, and section 7.10 helps you
choose between them. This chapter describes what the head is for, how to use its confidence to decide which answers a
person should check, the limits on its input, and when a fine-tune for your own task is the better choice. Results are
reported on the model card ([docs/model-card.md](../model-card.md)) and against the release criteria in
[chapter 15](15-release-criteria.md). The 26B A4B head's results are on its own model card
([docs/model-card-26b.md](../model-card-26b.md)).

## 7.1 What the general head is for

The general head answers Choice, Score and Noul questions about text and JSON states in any domain, and about photos,
zero-shot. It is trained on a broad mix (chapter 13):

- human-labelled public datasets covering customer intents, the quality of chatbot responses, toxicity, emotions and
  legal contract provisions, and photos taken by blind people with a question about each;
- rewritten versions of those datasets' questions, so the same states are asked about with different option sets,
  names and Noul statements;
- question sets and states written by an open teacher model across many settings, from support tickets and finance to
  moderation, documents and the qualitative coding of short free-text answers, labelled with the teacher's calibrated
  probabilities.

It is a good fit when:

- you have a decision to make about many items and no labelled examples yet;
- the decision can be explained in the question and the option descriptions;
- you want to know which answers to check, not only the answers.

It is also the starting point for evaluating a task: run it on a sample, look at where it is unsure, and decide whether
its answers are good enough, need a review step, or call for a fine-tune.

## 7.2 How to read its answers

Every answer is calibrated on human labels. Calibration fits one temperature per question type and number of options
(2, 3–5, 6–15, 16–63 and 64–255 options) and a bias for Nouls, on held-back human-labelled questions of the tasks the
head was trained on. A temperature divides the model's raw scores before they become probabilities. Above 1 it makes
answers less confident, below 1 more confident, and it never changes which option comes first. The Noul bias shifts
how readily the model says yes ([chapter 10](10-calibration.md), section 10.1).

Calibration holds on average over many items of the kinds the head was calibrated on. On a task it has not seen, the
head is usually underconfident until you calibrate it on about 100 of your own labelled items (chapter 10, section
10.6). Check it on a labelled sample of your own (section 7.4) before relying on a threshold. After calibration on
your own items, in practice:

- **Choice and Score probabilities** can be read as the chance that a person would pick each option. Across many
  answers given at 0.7, about 70% match the human label.
- **Noul** values are the probability that the statement is true. A value near 0.5 means the model cannot tell, not
  that the statement is half true.
- **Score** `score` is the expected level. A score of 1.5 on a 0–3 scale with probability split between levels 1 and 2
  says the item is on the boundary. The `probabilities` show how the mass is spread.
- **Confidence** (Choice and Score) summarises how peaked the distribution is. It ranks answers from most to least
  certain. The probabilities themselves carry the calibrated meaning.

## 7.3 Reviewing the least confident answers

The intended workflow for a large batch:

1. Run the head on every item (chapter 3, `scripts/predict_bundle.py` for files).
2. Sort the answers by confidence.
3. Send the least confident share to a person, and accept the rest.

The share to review is a trade-off between effort and accuracy. Two numbers measure it:

- **errors caught**: the share of the model's errors that fall inside the reviewed set;
- **accuracy of the rest**: how accurate the answers you accept without review are.

If the model's confidence carried no information, reviewing 20% of the answers would catch 20% of its errors. A
review-ready model catches more, because its errors concentrate among its least confident answers. In a 20% review,
the general head caught 60.6% of its errors on the coding study and 47.2% on MultiNLI, but only 21.7–24.5% on
HelpSteer3 and Measuring Hate Speech, close to the 20% that uninformative confidence would catch
([model card](../model-card.md)). Its confidence also tends to be low exactly where people disagree, so the reviewed
set holds the ambiguous items as well as the mistakes.

For a Noul, rank by how far the probability is from 0.5, where the model cannot tell: `abs(noul − 0.5)`, smallest
first.

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
confidence value, not a share, so it carries over to new batches. Take the confidence of the most confident item
inside that share of your sorted sample, and on a new batch send every answer at or below that confidence to review.

If several people labelled your sample, you can also check that the model's confidence follows their agreement:

```python
from decisivelm.eval import pair_agreement, spearman, within_category_spearman

# labels: int array [items, raters], -1 where a rater did not label the item
# consensus_label: the raters' majority label per item; categories with fewer than min_n items are skipped
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
| Audio clips | 4 per request, E4B only (the 26B A4B head's server returns HTTP 400) | the request is rejected |

- **JSON states** are written compactly, with keys in the order you gave them. Put the fields that matter early if the
  state may be shortened.
- **Instructions and criteria** may be strings, objects or arrays, for example `{"what": …, "examples": […]}`. Objects
  and arrays are written as compact JSON.
- **Cost** grows with the length of the state and the total length of the options. The state is read once per request
  however many questions you ask. Each option adds only its own description.

## 7.6 Languages and modalities

The general head is trained mostly on English text. It answers in other languages through the multilingual Gemma 4
backbone. The frozen held-out suite ([chapter 15](15-release-criteria.md)) measures this on ten languages the head was
not trained on. The suite is four public, human-labelled datasets that no training or calibration decision ever sees,
and each candidate head is scored on it once. On 60 intents in those languages the head's accuracy was 0.672, against
0.658 for the prompted base model, and it was overconfident until calibrated on 100 items
([model card](../model-card.md)). Check its accuracy on a sample before relying on it in another language. The 26B A4B
head scored 0.680 on the same intents, against 0.731 for the prompted 26B A4B, so, unlike the E4B head, it is below its
prompted base model there ([its model card](../model-card-26b.md)).

Of images, the head was trained only on photo questions: VizWiz-VQA's photos, taken by blind people to ask a question,
with whether each photo can answer its question and which answer it supports. The E4B head has been checked zero-shot on
one other image set, CIFAR-10H (section 3.7), where it was more accurate than the prompted base model but its
probabilities needed calibrating on the task. It has not been trained or evaluated on audio, and the 26B A4B head cannot
take audio, because its backbone has no audio input. The toolkit accepts images and audio
([chapter 4](04-requests-and-responses.md), section 4.5). For an image or audio task of your own, calibrate the head on
about 100 labelled items (chapter 10, section 10.6) or fine-tune it on your labelled examples
([chapter 9](09-fine-tuning-for-your-task.md), section 9.9). Starting that fine-tune from the E4B general head helped on
two of three public examples with 300 items. Across three seeds, voice-emotion accuracy rose by 0.021–0.042 and the mean
absolute error of a living-thing rating fell on two seeds (by 0.062 and 0.019), while photo quality showed no consistent
change (chapter 9, section 9.9).

## 7.7 When to fine-tune instead

The general head does the most with questions whose meaning is in their wording. A fine-tune on your own labels
(chapter 9) is the better choice when:

- **the rules are long or have precedence**: a codebook that says "code as B unless the response also mentions C,
  in which case D" asks for careful reading of a long instruction. The general head has to apply such a rule from its
  wording alone, and does so less well than a fine-tune that has learned it from labelled examples;
- **your labels follow local conventions**: the line between two codes is drawn by your team's practice rather than by
  the words of the codebook;
- **you need the best review readiness on one task**: a fine-tune learns where *your* coders disagree;
- **your states are images or audio**, or a kind of text far from the training mix;
- **you have labelled data already**: a few hundred labelled questions are enough to train and calibrate a fine-tune,
  and the general head's accuracy and review readiness on a labelled sample (section 7.4) are the baseline the
  fine-tune has to beat.

With a few hundred ratings but no wish to train, use task calibration instead ([chapter 10](10-calibration.md),
section 10.6). It fits a scale on the question's temperature and an offset per option or level on your own labelled
items, so the probabilities follow how your raters use the scale. A two-sided rating scale runs between two opposite
poles through a neutral middle ("strongly respectful … neither … strongly disrespectful"). On such a scale the head
has not seen, its confidence follows how extreme the text is rather than how much people agree. Task calibration
turns that confidence into a review signal. On Wikipedia toxicity it moved the correlation of the head's confidence
with rater agreement from −0.010 to +0.349 ([model card](../model-card.md)).

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
- **Free text.** The head does not explain, summarise or extract arbitrary spans. Express what you need as Choice,
  Score and Noul questions.
- **More than 255 options.** Split the decision into stages, for example a coarse Choice and then a fine one.
- **Facts the state does not contain.** The head judges the state you give it. Questions about the outside world are
  answered from what the backbone happens to know, with no guarantee.

## 7.10 Choosing between the E4B and 26B A4B heads

Both heads were trained on the same mix (chapter 13) and answer the same requests, so you can run both on a labelled
sample of your task and compare their accuracy and review thresholds (sections 7.4 and 10.6) before relying on either.

- **Use the E4B head by default.** It runs on a 24 GB GPU (chapter 2), answers a 20-block request in about 0.10 s over
  HTTP, accepts audio input (though it is not trained on audio), and runs on Apple silicon.
- **Use the 26B A4B head when review readiness matters more than speed and you have a GPU with about 72 GB free.** On
  the frozen held-out suite, scored against the E4B head's own numbers (each from its own report, not paired), it is
  better on every measure the comparison shows but one: a review of its least confident 20% of hate-speech answers
  catches slightly fewer of its errors (23.4% against 24.5%). For example, on Measuring Hate Speech's Scores the
  correlation of its confidence with rater agreement is 0.177 against 0.078, and the mean absolute error is 0.571
  against 0.653 ([model card of the 26B A4B head](../model-card-26b.md)). It takes about 0.47 s for the same request and
  computes in float32.
- **The 26B A4B head has no audio input.** The server refuses audio with HTTP 400.
- **The 26B A4B head is compared with a stronger model.** Each head is judged against its own backbone prompted to
  answer the same questions, and the prompted 26B A4B is a much stronger model than the prompted E4B (0.731 against
  0.658 on the ten-language intents). Compared against that prompted 26B A4B, the 26B A4B head fails five checks of the
  release criteria, so it is a preview ([chapter 15](15-release-criteria.md), section 15.3). Its model card lists them:
  lower accuracy on the ten-language intents (0.680 against 0.731), a review of its least confident 20% of hate-speech
  answers that catches fewer errors (23.4% against 26.7%), the lowest within-category correlations on the hate-speech
  survey and HelpSteer3, and a Brier score (a calibration measure, lower is better, section 11.2) on the hate-speech
  Scores that calibration makes worse by 0.0004. It also rates edited headlines as funnier than people do (Humicroedit,
  humour ratings of edited news headlines, chapter 14).
- **The 26B A4B head was trained once** (seed 0). Two E4B runs that differ only in their seed differ by up to 0.17 in
  the correlation of confidence with rater agreement on the development sets, so small differences between the heads are
  not reliable.

Next: [8. Data format](08-data-format.md)
