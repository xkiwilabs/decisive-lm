# 11. Evaluation

A decision model is judged here on three things, in this order:

1. **Review readiness:** whether its confidence tells a person which answers to check.
2. **Calibration:** whether its stated confidence can be trusted.
3. **Accuracy.**

Every number is measured on human-labelled data that played no part in training or calibration. Every model is
compared with the same backbone prompted directly (P0), on the same items, with paired confidence intervals. This
chapter defines each metric, the scripts that compute them, and how to read a report. The review-readiness definitions
follow Alhasan, Diaz Monfort & Richardson, *Knowing which responses to check* (under review).

## 11.1 Ground rules

- **Human labels only.** Reported results use the `gold_test` split (or a held-out dataset's test rows), scored against
  human targets. Teacher-labelled `val` rows are used for model selection only.
- **Never choose on the test set.** Stopping rules, checkpoints, temperatures and prompts are chosen on `val` and
  `gold_calib`. A number from `gold_test` is reported, not optimised.
- **Always against a baseline.** P0, the frozen backbone prompted per question ([chapter 9](09-fine-tuning-for-your-task.md)),
  is the number to beat. A model **beats** a baseline when the primary metric improves and the paired 95% confidence
  interval of the difference excludes zero.
- **Hard and soft labels.** Accuracy and calibration use **hard** labels: the raters' unique plurality, or for a Noul a
  yes-share of at least 0.5. Items where raters tie have no hard label and are left out of these metrics. Soft metrics
  and review readiness use the raters' **shares**.

## 11.2 Metrics per question type

`scripts/eval_dataset.py` computes these for every question of any dataset (`decisivelm/eval.py` holds the shared
functions):

**Choice**

| Metric | Meaning |
|---|---|
| accuracy | Top option matches the raters' unique plurality. |
| macro-F1 | F1 per option, averaged with every option weighted equally, so rare options count. |
| soft NLL | Cross-entropy against the raters' shares, divided by `log n` (uniform = 1). |
| ECE | Top-label expected calibration error, 15 bins ([chapter 10](10-calibration.md)). |
| Brier | Squared distance to the one-hot label, summed over options. |
| conf AUROC | How well the top probability separates right answers from wrong ones (0.5 = no better than chance). |
| `other` accuracy | Accuracy on items whose answer is a catch-all `other` option, reported separately because such an option cannot see its siblings. |

**Score**

| Metric | Meaning |
|---|---|
| MAE | Mean absolute difference between the predicted score (`Σ level × p`) and the raters' mean level. |
| exact | Most likely level matches the raters' unique plurality. |
| EMD | Earth mover's distance: the summed gap between the predicted and human cumulative distributions over levels. It penalises mass placed far from the human levels more than mass placed nearby. |
| soft NLL | As for Choice. |

**Noul**

| Metric | Meaning |
|---|---|
| AUROC | How well `p(yes)` separates items humans call yes from items they call no. |
| acc@.5 | Answer at the 0.5 threshold matches the human side. |
| Brier, ECE | Calibration of `p(yes)`; ECE uses `max(p, 1 − p)` as the confidence. |
| soft BCE | Binary cross-entropy against the yes-share. |
| ρ(p, share) | Spearman correlation between `p(yes)` and the human yes-share: does the model's probability rise with the share of people saying yes? |

**Primary metric per type:** Choice accuracy, Score MAE (lower is better) and Noul AUROC. Gates and "beats" are
decided on these. For a dataset with several questions of a type, the primary metric is averaged over them.

## 11.3 Review readiness

A model is **review-ready** when sending its least confident answers to a person catches its mistakes, and when its
confidence follows how much people themselves agree. These metrics need items rated by several people: at least
three per item for the gates.

**Per-item human agreement** is the share of rater pairs that gave the same label. With four raters it takes the
values 0, .17, .33, .5 and 1; with three raters, 0, ⅓ and 1. It is computed from the stored shares and `n_raters`.

| Metric | Question it answers | Good |
|---|---|---|
| **ρ(top p, agreement)** | Spearman correlation between the model's top probability and per-item agreement. Is the model surer where people agree? | Positive, higher |
| **Within-category ρ** | The same correlation inside each consensus category (each label the raters converged on) with at least 25 items. Only this shows item-level tracking: an overall ρ can arise just from some categories being harder than others. | Positive in every category |
| **Tie AUROC** | How well low confidence finds the items with no unique human majority (ties). | Above 0.5, higher |
| **Dynamic range** | Highest minus lowest mean top probability across agreement levels. A signal running from .986 to .998 ranks items but cannot carry a review threshold. | At least 0.2 |
| **Confidence-guided review** | Send the least confident fraction *f* of all items to a person. Report the share of the model's errors inside the reviewed set ("errors caught") and the accuracy of the unreviewed rest. Tied items are ranked like any other but counted in neither figure. | Many errors caught, high rest accuracy |
| **Ties in review** | The share of human ties that fall inside the reviewed fraction. | High |

Two confidence **signals** are reported:

- **top p:** the probability of the chosen option.
- **1 − H(p)/log n:** read from the spread of the whole distribution (`decisivelm.eval.entropy_confidence`).

A model whose errors are spread evenly through its confidence ranking catches about *f* of its errors with a review of
fraction *f*. A review-ready model catches far more.

Where these are computed:

- `scripts/eval_dataset.py` reports ρ(top p, agreement), tie AUROC, errors caught at 20% and the rest accuracy at 20%
  for every Choice or Score question where most items have three or more raters.
- `scripts/compare_coding.py` reports the full set for a coding task: both signals, within-category ρ, dynamic range,
  review at 10% and 20%, ties in review, and the paired differences between methods.
- `scripts/score_suite.py` applies them as release criteria ([chapter 15](15-release-criteria.md)).

## 11.4 Confidence intervals and paired comparisons

Every interval is a 95% **item-level bootstrap**. Test items are resampled with replacement (1,000 times by default in
`eval_dataset.py`, 2,000 in `compare_coding.py`), the metric is recomputed on each resample, and the 2.5th and 97.5th
percentiles are reported.

Comparisons are **paired**. Every method is scored on the same resamples, and the interval of the difference comes from
the per-resample differences. Two models whose own intervals overlap can still differ reliably when they are right and
wrong on the same items, and pairing detects that. `eval_dataset.py` scores every method on the rows that all of them
answered, and says so when that is fewer than all rows.

## 11.5 Running an evaluation

```bash
uv run python scripts/eval_dataset.py --data data/processed/my_task.jsonl \
    --predictions "P3=data/predictions/P3_my_task_e4b_r8_seed0.jsonl" \
                  "P0=data/predictions/P0_my_task_e4b.jsonl" \
    --reference P0 [--split gold_test] [--n-boot 1000] [--seed 0] --out results/my_task_gold_test.md
```

- `--predictions` takes `NAME=path` pairs. The files are prediction rows as written by the training scripts,
  `run_p0.py` and `predict_bundle.py`.
- `--reference` is the method every Δ is measured against.

The report has one table per question:

```
## team (choice, 3 options)

| Method | soft NLL | accuracy | macro-F1 | ECE | Brier | conf AUROC | errors in review 20% | rest acc 20% | ρ(top p, agreement) | tie AUROC | accuracy 95% CI | Δaccuracy vs ref | 95% CI |
```

Read it in this order:

1. **Coverage.** The header says how many rows every method answered. A method that skipped items is not comparable.
2. **Review readiness**, where shown. Are ρ and tie AUROC at least P0's, and does a 20% review catch a larger share of
   errors than P0's own review?
3. **Calibration.** ECE well under 0.1 is usable for thresholds. Compare with P0's.
4. **Primary metric.** The Δ against the reference and its interval. An interval that includes zero means no
   detectable difference on this test set, not equality.

Things to keep in mind:

- A few hundred items give wide intervals. An ECE difference of 0.02 is rarely significant at that size.
- On items where raters tie, no answer is wrong; those items are excluded from accuracy, not counted against the
  model.
- A dataset that has been used to choose between models is a **development set**. Its numbers are reported as such,
  and claims about new tasks rest on held-out data ([chapter 15](15-release-criteria.md)).

## 11.6 Comparing with human coders

Agreement with a consensus label is not the whole story for a coding task: the useful comparison is with the coders
themselves. `scripts/compare_coding.py` reports:

- **Cohen's κ** against each coder separately, averaged, with a bootstrap interval. κ is agreement corrected for
  chance agreement.
- **The human ceiling:** the mean pairwise κ between the coders. A model near the ceiling agrees with a coder about as
  well as coders agree with each other.
- **Accuracy against the consensus**, and recall per consensus category.
- **Soft NLL, ECE, Brier and confidence AUROC** for probabilistic methods.
- **Review readiness** (§11.3), and the paired differences from `--reference`.

It reads the coders' own labels from a CSV with one column per coder (`--codes`), and the models' predictions as
`NAME=path` pairs. The script is written for a table laid out like the coding study's: the coder columns are listed
in `HUMANS`, and each row's id is built from its participant and trial columns. Adapt both to your table.

```bash
uv run python scripts/compare_coding.py --data data/processed/my_coding_task.jsonl --codes coders.csv \
    --predictions P3=data/predictions/P3_my_coding_task_e4b_r8_seed0.jsonl P0=data/predictions/P0_my_coding_task_e4b.jsonl \
    --reference P3 [--review 0.1 0.2] [--n-boot 2000] --out results/my_coding_task.md
```

## 11.7 Conformance, independence and order

These check that the model behaves as the request contract says ([chapter 4](04-requests-and-responses.md)), whatever
its accuracy.

**Conformance.** Every response must meet all of these:

- `probabilities` keys exactly match the request's options or levels, and the values sum to 1;
- `score` lies in `[0, n − 1]`;
- a Noul answer is in `[0, 1]` and has no `confidence`;
- `choice` is the most probable option.

Responses are validated as they are built, and the test suite checks conformance on a tiny random Gemma 4
configuration (`uv run pytest`).

**Independence and order.** Each option is scored in its own attention branch, so these must leave an answer
unchanged:

- adding or removing other questions;
- shuffling a Choice question's options;
- renaming a question id;
- packing another request alongside.

In exact arithmetic the change is zero. In practice the tests allow 1e-5 in fp32. In bf16 the allowance is set from
the measured rounding noise. The test suite checks this on the tiny configuration. On a trained bundle and the real
backbone, measure what a user would see:

```bash
uv run python scripts/check_invariance.py --bundle bundles/P3_my_task_e4b_r8_seed0 --data data/processed/my_task.jsonl \
    [--split gold_test] [--n 200] [--dtype float32] --out results/my_task_invariance.md
```

For a sample of rows it compares every answer with the same answer under these variants, reported per question type
(a Noul's answer is yes or no):

- `repeat`: the identical pass again, a determinism check;
- `reversed_options` (Choice only, since Score levels are ordered);
- `after_another_request`: another request packed first;
- `extra_question`: one more question added.

| Column | Meaning |
|---|---|
| max \|Δp\| / mean / p99 | Largest, mean and 99th-percentile change of any option's probability |
| answer flips | Items whose top option changed |
| top-p rank ρ | Rank correlation of the top probability before and after: is the order of confidence stable? |
| review-set churn (20%) | Share of the 20% least confident items that would change |

On the dense E4B backbone these are small. On the 26B A4B mixture-of-experts backbone, bf16 rounding can flip a token's
choice of experts when the packing changes. `--dtype float32` shows how much of the movement is rounding, and fp32
inference removes most of it.

`scripts/check_bundle.py` complements this. It confirms that a saved bundle, reloaded through `DecisiveLM.system_one`,
reproduces the predictions written at training time for every question, and that every response conforms
([chapter 9](09-fine-tuning-for-your-task.md)).

Next: [12. Teacher labelling](12-teacher-labelling.md)
