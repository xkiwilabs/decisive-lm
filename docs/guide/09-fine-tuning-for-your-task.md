# 9. Fine-tuning for your task

A **task fine-tune** is a small adapter trained on your own labelled decisions, so that one forward pass answers your
questions the way your human raters would and says how sure it is. This chapter covers the whole path: preparing
labels, measuring the baseline to beat, training, calibrating, evaluating and putting the result to use. A fine-tune
starts either from base Gemma 4 or from the general head (§9.6), and needs one GPU.

As in [chapter 1](01-introduction.md), a **head** is the whole trained bundle you load, and the general head is one.
Where this chapter says "the general head" without naming a backbone, it means the E4B head. The **scorer** is the small
network inside a head that turns one of the backbone's hidden states into one score ([chapter 6](06-how-it-works.md),
section 6.5).

## 9.1 The path at a glance

| Step | What you do | Tool |
|---|---|---|
| 1 | Convert your labelled data into canonical records, with rater shares | a converter ([chapter 8](08-data-format.md)) |
| 2 | Check the data | `scripts/inspect_data.py` |
| 3 | Measure the baseline: base Gemma prompted directly (P0) | `scripts/run_p0.py` |
| 4 | Optional: a quick probe that trains the scorer only (P1) | `scripts/cache_features.py`, `scripts/train_p1.py` |
| 5 | Train the fine-tune (P3): a LoRA adapter, the query tokens and the scorer (§9.5), from base or from the general head | `scripts/train_p3.py` |
| 6 | Evaluate against P0 and against your raters | `scripts/eval_dataset.py`, `scripts/compare_coding.py` |
| 7 | Check the bundle, then use it | `scripts/check_bundle.py`, `scripts/check_invariance.py`, `attach_bundle` |

There is no P2 step, because P2, in the design's numbering, trains the general head
([chapter 13](13-how-the-general-head-was-trained.md)).

Calibration is part of training. The training scripts fit temperatures on `gold_calib`, the calibration split of §9.2,
and write calibrated predictions. A temperature is one number per question type and size that scales how confident
the answers are ([chapter 10](10-calibration.md)).

## 9.2 Preparing labelled data

**Ask the questions you will ask in production.** Write each question once, in the request format of
[chapter 4](04-requests-and-responses.md). The same `instructions` and option descriptions go into every row, and the
fine-tune learns to judge exactly those words. [Chapter 5](05-writing-good-questions.md) explains how to write options
that can be judged one at a time, including self-contained `other` options.

**Use several raters where you can, and train on their shares.**

- If three people label an item and two choose `billing` and one `technical`, the target is
  `{"billing": 0.667, "technical": 0.333, "sales": 0.0}` with `n_raters: 3`.
- Never reduce this to the majority vote. The shares teach the model to be unsure where people are unsure,
  and that is the property the fine-tune is judged on: whether its confidence tells you which answers to check.
- With one rater per item the targets are one-hot. The model can still learn to be accurate and calibrated, but its
  confidence cannot be checked against human agreement.
- Record `n_raters` on every human target. The review-readiness metrics in [chapter 11](11-evaluation.md) use items
  with three or more raters.

**Split by the unit that could leak.** Assign splits by a hash of the item, or of the document, conversation or
participant the item comes from, so near-identical items never sit on both sides of a split (§8.4). A workable plan:

| Split | Share of labelled items | Purpose |
|---|---|---|
| `train` | 60–70% | Training |
| `val` | 10% | Early stopping: deciding when training stops and which snapshot to keep (§9.5) |
| `gold_calib` | ~10–15% (≥ 100 questions per question type) | Calibration |
| `gold_test` | ~15–20% | All reported results |

The more items there are, the better the fine-tune. As a minimum:

- 20 labelled questions to start training;
- at least 100 labelled questions per question type in `gold_calib` for a dependable temperature;
- enough in `gold_test` for a confidence interval narrow enough to act on. A few hundred items gives intervals of a few
  points on accuracy.

A study of coding free-text responses is a typical task. In it, several people assign each participant's answer to one
category of a written codebook (Alhasan, Diaz Monfort & Richardson, *Knowing which responses to check*, under review).
The codebook becomes one Choice question, each category an option with its codebook definition, and each response a
row whose target is the share of coders who chose each category.

Check the converted file before training:

```bash
uv run python scripts/inspect_data.py data/processed/my_task.jsonl
```

## 9.3 The baseline: P0

P0 is the frozen backbone prompted directly, with no training. It is the number every fine-tune must beat.

- The model is asked the question with lettered options, and its reply is started for it with "Answer:", so the next
  token it writes is an option's letter.
- Its label probabilities are read directly.
- Every rotation of the options is averaged (or 8 seeded orderings above 8 options) to cancel position bias, a
  preference for an option because of where it sits in the list.
- Questions with more than 20 options are numbered and read digit by digit ([chapter 12](12-teacher-labelling.md)).
- One temperature is fitted on `gold_calib`.

```bash
uv run python scripts/run_p0.py --data data/processed/my_task.jsonl --question team [--backbone e4b]
```

- `--question` names the Choice question to score (one per run).
- `--splits` defaults to `gold_calib gold_test`.
- `--chunk` sets how many rows are written per resumable chunk.

Raw distributions go to `data/predictions/P0_<data>_<backbone>_raw.jsonl`. The job resumes where it stopped. When
every item is done, calibrated predictions are written to `data/predictions/P0_<data>_<backbone>.jsonl`.

`run_p0.py` scores one Choice question on text rows. For Score and Noul questions, and for rows with images or audio,
run P0 with `scripts/label_teacher.py` instead. It is the teacher labeller of [chapter 12](12-teacher-labelling.md),
here run with the base model on this machine's GPU (section 9.9):

```bash
uv run python scripts/label_teacher.py --teacher gemma4_e4b --local --data data/processed/my_task.jsonl
```

## 9.4 Optional: a scorer-only probe (P1)

P1 trains only the scorer, on hidden states cached from the frozen backbone (the backbone's internal vectors at each
query token, which the scripts call features). It runs in minutes once the features are cached, and it shows how much
of the task the base model already represents. It is a diagnostic. A P1 scorer can also warm-start the P3 scorer (§9.5).

```bash
uv run python scripts/cache_features.py --data data/processed/my_task.jsonl --out data/features/my_task_e4b.pt
uv run python scripts/train_p1.py --features data/features/my_task_e4b.pt --data data/processed/my_task.jsonl \
    --standardize --seed 0
```

- `cache_features.py` runs the backbone once per row and saves progress every `--save-every` rows, so it resumes
  after an interruption.
- `train_p1.py` trains on `train` with the same per-type losses as adapter training ([chapter 6](06-how-it-works.md),
  section 6.7), stops early on `val` (`--patience`, by default 15 epochs, that is, 15 passes over `train`) and fits
  calibration on `gold_calib`. It writes predictions and a bundle named `P1_<features>[_z]_seed<seed>`.
- `--standardize` z-scores each feature dimension before the scorer. It subtracts the dimension's mean and divides by
  its standard deviation, both measured on training states. Without it, a few very large dimensions of Gemma 4's states
  dominate the rest (section 6.5). It is required on 26B A4B and helps on E4B.

## 9.5 Training the fine-tune (P3)

P3 trains three things together ([chapter 6](06-how-it-works.md), section 6.6):

- a LoRA (low-rank adaptation) adapter on the language model's decoder projections, a small trainable correction to
  the frozen weights that puts the judgement "how well does this option fit this state" into the hidden state the
  scorer reads;
- the three query-token embedding rows, a trainable vector for each of the tokens that mark where an answer is read;
- the scorer.

How much LoRA adds over P1's frozen backbone depends on the task (section 6.6). P3 stops early on `val`, fits
calibration on `gold_calib` and writes the bundle and calibrated predictions.

```bash
uv run python scripts/train_p3.py --data data/processed/my_task.jsonl --standardize-rows 2000 --seed 0
```

This command is a complete first run from the base model. The other flags change the training schedule, fit training
to the memory you have (below), start from another bundle, or select the larger backbone.

| Flag | Default | Meaning |
|---|---|---|
| `--data` | | Canonical records with `train`, `val`, `gold_calib` and `gold_test` rows. |
| `--backbone` | `e4b` | `e4b` or `26b-a4b` (profiles in `configs/model.yaml`). |
| `--rank`, `--alpha`, `--dropout` | 8, 16, 0.05 | LoRA rank, scaling and dropout. |
| `--epochs` | 2 | Passes over `train`. Training usually stops earlier. |
| `--rows-per-step` | 16 | Requests per optimiser step. |
| `--max-tokens` | 8192 | Packed tokens per micro-batch (memory; see below). |
| `--lr-lora`, `--lr-head` | 2e-4, 1e-3 | Learning rates for the LoRA weights and for the scorer plus query rows. |
| `--warmup-frac` | 0.03 | Linear warm-up share of the steps, then cosine decay. |
| `--eval-every`, `--patience` | 40, 3 | Validate every N steps; stop after N validations without improvement. |
| `--limit-val` | all | Early-stop on a seeded random sample of this many `val` rows. |
| `--select` | `loss` | The validation criterion that picks the kept snapshot (below). |
| `--predict-best` | none | Also write a bundle and predictions from each listed criterion's best snapshot, named `<name>_by<criterion>`. |
| `--standardize-rows` | 0 | Z-score the scorer's input (§9.4) using the base model's states on this many `train` rows. At 0 the input is not z-scored. The commands in this chapter use 2000 when training from base. |
| `--init-head` | | A bundle (e.g. a P1 run) whose scorer warm-starts this one. |
| `--init-bundle` | | A trained bundle (e.g. the general head) whose LoRA, query rows and scorer this run trains further (§9.6). |
| `--n-train` | all | Train on a seeded random sample of this many `train` rows (sampled with `--seed`). |
| `--no-checkpointing` | off | Turn off gradient checkpointing: faster, but needs much more memory. |
| `--dtype` | `bfloat16` | `float32` runs every matmul in fp32 (recommended for 26B A4B). |
| `--fp32-router` | off | MoE routers only in fp32 (26B A4B). |
| `--image-budget`, `--media-root` | | Soft tokens per image, and the directory relative media paths resolve against. |
| `--adapter` | `user` | Which holdouts to enforce. `general` also restricts training rows to release-tier data. |
| `--limit-train` | | A timing run on this many rows: no calibration or predictions. |
| `--name`, `--seed` | | Bundle name (default `P3_<data>_<backbone>_r<rank>_seed<seed>`) and random seed. |

**Choosing the kept snapshot.** Every validation computes three criteria, and `--select` picks one:

- `loss`: the training loss on `val`.
- `cal_loss`: the same loss after one temperature per question type is fitted on `val`. A snapshot that is only over-
  or under-confident is not penalised, because calibration corrects that afterwards.
- `acc`: whether the top answer matches the raters' unique plurality (Choice and Score) or the share's side of 0.5
  (Noul).

`loss` is the default. `cal_loss` suits tasks with much rater disagreement, where the raw loss rewards hedging.
`--predict-best loss acc` writes the alternatives as well, so the rules can be compared on `gold_test` afterwards.
Choose the rule before you look at `gold_test`, and report the one you chose.

**Warm-starting from P1.** `--init-head bundles/P1_...` starts the scorer from a trained P1 scorer and brings that
scorer's standardisation, so it cannot be combined with `--standardize-rows`. Set `--lr-head` near `1e-5` when
warm-starting, because an optimiser's first steps move every weight by about the learning rate, and at `1e-3` that
knocks a converged scorer off its optimum.

**Memory.** `--max-tokens` bounds the packed tokens per micro-batch. Several short requests are packed together until
the budget is reached, and a request over budget runs alone.

| Hardware | Suggested settings |
|---|---|
| A GPU with 24 GB or more, E4B | the defaults, `--max-tokens 8192` with checkpointing. A 300-item fine-tune peaked at 21.2 GiB with them, and at 16.7 GiB with `--max-tokens 2048`, which was also faster (0.72 s a row against 1.14) |
| A GPU shared with other work | `--max-tokens 2048` |
| Apple silicon with large unified memory, E4B | `--no-checkpointing --max-tokens 2048`: one request per micro-batch. Time per row grows with its length: about 0.14 s for short messages, about 2 s for the coding study's longer rows. |
| Apple silicon, 26B A4B | not measured for fine-tuning. Answering with the 26B A4B head fits in 96 GB or more of unified memory (58.6 GiB at the peak, section 2.1) |

On an NVIDIA GPU, `train_p3.py` and `predict_bundle.py` print PyTorch's peak memory at the end of a run. The GPU needs
about 0.7 GiB more for the CUDA context.

Packing several requests into a long micro-batch saves little, because the attention mask is dense and the full square
of the sequence is computed. Turning off checkpointing on a large micro-batch multiplies memory several times over. An
8,192-token E4B micro-batch without checkpointing needed more than 180 GB on Apple silicon.

**26B A4B.** The larger backbone needs about 50 GB for its weights. Train it with
`--backbone 26b-a4b --standardize-rows 2000 --dtype float32`. The default compute type is bf16 (16-bit floating point),
and `--dtype float32` runs every matrix multiplication in fp32 (32-bit) instead. The fp32 compute keeps expert routing
(which 8 of the 128 experts each token goes through, chapter 2) stable when requests are packed differently. The weights
stay in bf16, the checkpoint's own precision, so converting them to fp32 is exact. Caching 26B A4B features took 0.51 s
a row in fp32 against 0.16 in bf16, about three times as long (`results/invariance.md`). A 26B A4B bundle cannot take
audio.

**What P3 writes.**

- `bundles/<name>/`:
  - `adapter.pt`: the LoRA weights and query-token rows;
  - `scorer.pt`: the scorer, with its standardisation;
  - `calibration.json`: the temperatures, the Noul bias, the backbone profile and revision, the LoRA settings, the
    render version, the training data file, the selection rule and step, and the validation history.
- `data/predictions/<name>.jsonl`: calibrated predictions for every question of `gold_calib` and `gold_test`, in the
  format the evaluation scripts read.

Before training, `train_p3.py` checks the `train` rows with `check_mix`. It stops with an error if a row belongs to a
dataset held out for the chosen adapter kind in `configs/holdout.yaml` (§8.6) or, with `--adapter general`, is not
release-tier data. Rows of your own data pass unless they come from one of those datasets.

## 9.6 Starting from the general head

With few labels (the comparison below used 100 and 300 items), start from the general head instead of the base model.
The fine-tune then begins with a model that already knows how to answer typed questions and how to be unsure, and
your labels only have to teach it your task.

```bash
uv run python scripts/train_p3.py --data data/processed/my_task.jsonl \
    --init-bundle bundles/decisivelm-e4b-general --lr-lora 5e-5 --lr-head 1e-5 --seed 0
```

- `--init-bundle` loads the bundle's LoRA weights, query-token rows and scorer, standardisation included, and trains
  them further. The LoRA rank, scaling and target modules come from the bundle, so `--rank` and `--alpha` are ignored.
  The bundle must match `--backbone`.
- It cannot be combined with `--init-head` or `--standardize-rows`, because the bundle already brings a scorer and its
  standardisation.
- The bundle's own validation score, before any step, counts as a candidate snapshot. A fine-tune that never improves on
  the general head keeps the general head's weights.
- Calibration is refitted on your `gold_calib` as usual. The general head's temperatures
  ([chapter 10](10-calibration.md)) are not carried over.

**Lower learning rates.** Use about `5e-5` for the LoRA weights and `1e-5` for the scorer, not the defaults of `2e-4`
and `1e-3`. As with a P1 warm start (§9.5), the defaults knock the general head's trained weights off their optimum
before your labels can guide them.

**From the 26B A4B head.** Use `--backbone 26b-a4b --dtype float32 --init-bundle bundles/decisivelm-26b-a4b-general`
with the same learning rates. The fine-tune computes in float32 (section 9.5) and needs a GPU with about 72 GiB free for
the head itself. Memory for a fine-tune has not been measured (training the 26B A4B head on its full mix peaked at 66.0
GiB with 4,096-token micro-batches). The comparison below was run with the E4B head only. Whether the 26B A4B head is a
better starting point than the base model (release criterion 8) has not yet been measured. It is scheduled for 7 to 8
November 2026, and until then use the E4B head or the base model if you need a result you can rely on.

**How much it helps (E4B head).** We compared fine-tunes from the general head with fine-tunes from the base model
(release criterion 8, [chapter 15](15-release-criteria.md)). Each was trained on 100 and on 300 labelled items of two
tasks outside the general head's training mix. Each start (an arm) was trained three times with different random seeds.
Both arms used the same items and the same settings (`--epochs 10 --eval-every 5 --patience 4 --limit-val 200 --select
loss`). The base arm used its default learning rates and `--standardize-rows` on its training items, and the general arm
used the rates above. Each fine-tune was calibrated on its own task's `gold_calib` and scored on `gold_test`. Values are
means over the three seeds. Δ is the general start minus the base start, with a 95% interval from item-level bootstrap
resamples ([chapter 11](11-evaluation.md), section 11.4).

The measures are defined in chapter 11 (sections 11.2 and 11.3):

- **ρ of confidence with rater agreement**: the Spearman correlation between the model's top probability and the
  share of rater pairs that agree on the item. Above 0, the model is surer where people agree, and higher is better.
- **Tie AUROC**: how well low confidence picks out the items with no unique human majority. 0.5 is chance and 1 is
  perfect.
- **Errors caught by reviewing the least confident 20%**: the share of the model's errors that fall among the 20% of
  items it is least sure of. Confidence unrelated to errors catches about 0.20.
- **ECE** (expected calibration error): the average gap between the model's top probability and how often it is
  right. 0 is perfect, and well under 0.1 is usable for a review threshold.
- **Soft NLL** (negative log-likelihood): the mean of −log p over the raters' answers, divided by `log n`, where n is
  the number of options, so that a uniform guess scores 1. It is lowest when the probabilities match the raters' shares.

MultiNLI (whether the second sentence follows from the first; 3 options, five annotators per pair, 4,823 test pairs):

| Measure | 100 items: base | 100: general | Δ (95% CI) | 300 items: base | 300: general | Δ (95% CI) |
|---|---|---|---|---|---|---|
| Accuracy against the raters' plurality | 0.843 | 0.861 | +0.018 [+0.011, +0.025] | 0.854 | 0.873 | +0.019 [+0.012, +0.026] |
| ρ of confidence with rater agreement | 0.351 | 0.399 | +0.048 [+0.032, +0.062] | 0.377 | 0.430 | +0.052 [+0.038, +0.065] |
| Tie AUROC | 0.736 | 0.754 | +0.018 [−0.009, +0.046] | 0.754 | 0.781 | +0.027 [+0.000, +0.053] |
| Errors caught by reviewing the least confident 20% | 0.513 | 0.546 | +0.033 [+0.008, +0.058] | 0.541 | 0.574 | +0.033 [+0.005, +0.060] |
| ECE (lower is better) | 0.070 | 0.061 | −0.009 [−0.016, −0.003] | 0.064 | 0.059 | −0.005 [−0.011, +0.001] |
| Soft NLL (lower is better) | 0.525 | 0.488 | −0.037 [−0.042, −0.031] | 0.502 | 0.473 | −0.030 [−0.035, −0.024] |

A private coding study (8 codebook categories, four coders, 792 test responses; its data is not distributed):

| Measure | 100 items: base | 100: general | Δ (95% CI) | 300 items: base | 300: general | Δ (95% CI) |
|---|---|---|---|---|---|---|
| Accuracy against the coders' plurality | 0.866 | 0.919 | +0.052 [+0.036, +0.068] | 0.919 | 0.932 | +0.012 [+0.001, +0.024] |
| ρ of confidence with coder agreement | 0.473 | 0.577 | +0.104 [+0.076, +0.133] | 0.575 | 0.622 | +0.047 [+0.017, +0.076] |
| Tie AUROC | 0.744 | 0.750 | +0.006 [−0.017, +0.028] | 0.806 | 0.808 | +0.002 [−0.028, +0.038] |
| Errors caught by reviewing the least confident 20% | 0.666 | 0.750 | +0.084 [+0.012, +0.167] | 0.670 | 0.815 | +0.145 [+0.053, +0.225] |
| ECE (lower is better) | 0.072 | 0.060 | −0.012 [−0.027, +0.002] | 0.064 | 0.051 | −0.012 [−0.022, −0.001] |
| Soft NLL (lower is better) | 0.329 | 0.269 | −0.060 [−0.072, −0.048] | 0.277 | 0.243 | −0.033 [−0.044, −0.024] |

On both tasks and at both sizes, starting from the general head gave a fine-tune whose confidence followed human
agreement more closely and whose review caught more of its errors, and it was not significantly worse on any measure.
Accuracy was higher on both tasks at both sizes, by 1.2 to 5.2 points. On the coding study the gains in accuracy and in
ρ are largest with 100 items, and 100 items from the general head came within 1.6 points of a fine-tune from base on all
2,592 training items (0.935). On MultiNLI the gains are about the same at both sizes. Tie AUROC was slightly higher from
the general head on MultiNLI at 300 items, and did not differ clearly otherwise. The intervals reflect the sampling of
test items, not the spread between seeds. The per-seed values are in `results/c8_general_v5c_e4b_r8_seed0.md`, written
by `scripts/score_c8.py`.

## 9.7 Evaluating

Compare the fine-tune with P0 on `gold_test`, using the same items and the same bootstrap resamples:

```bash
uv run python scripts/eval_dataset.py --data data/processed/my_task.jsonl \
    --predictions "P3=data/predictions/P3_my_task_e4b_r8_seed0.jsonl" \
                  "P0=data/predictions/P0_my_task_e4b.jsonl" \
    --reference P0 --out results/my_task_gold_test.md
```

`results/` is tracked by git. For private data, write the report to a folder git ignores instead, for example
`--out data/my_task_gold_test.md`.

For each question this reports accuracy, calibration and, where most items have three or more raters, review
readiness. It also gives the primary metric with a 95% confidence interval and the paired difference from the
reference. [Chapter 11](11-evaluation.md) explains every column. Read it in this order:

1. **Conformance and coverage.** Every method answered every item.
2. **Review readiness.** Does the fine-tune's confidence track human agreement (ρ, tie AUROC) at least as well as
   P0's, and does reviewing its least confident 20% catch more of its errors?
3. **Calibration.** Is the ECE well under 0.1, and lower than P0's?
4. **Accuracy.** Is the primary metric's paired difference from P0 above zero, with an interval that excludes zero?

**Comparing with individual human coders.** `scripts/compare_coding.py` produces the full report for a multi-coder
coding task:

- Cohen's κ (agreement corrected for chance) against each coder, beside the human ceiling (the mean pairwise κ
  between coders);
- accuracy against the coders' consensus;
- soft NLL, ECE and Brier (section 11.2);
- review readiness for two confidence signals, a confidence-guided review at 10% and 20%, and recall per category.

It reads each coder's own labels from a CSV with one column per coder. The script is written for the coding study's
table: the coder column names are fixed in its `HUMANS` list, row ids are built as `lrc-<SubjectID>-<TrialNumber>`, and
`--question` defaults to `intent`. To use it on your own coders, change the `HUMANS` list to your coder columns, change
the two places that build row ids from `SubjectID` and `TrialNumber` (one for `--codes`, one for `--baselines`) so the
ids match your rows' `id`s, and pass `--question` with your question id ([chapter 11](11-evaluation.md)). If you have
rater shares but not each coder's labels, `eval_dataset.py` above reports everything except the per-coder κ.

## 9.8 Checking and using the bundle

Two checks before you rely on a bundle:

```bash
uv run python scripts/check_bundle.py --bundle bundles/P3_my_task_e4b_r8_seed0 --data data/processed/my_task.jsonl --n 20
uv run python scripts/check_invariance.py --bundle bundles/P3_my_task_e4b_r8_seed0 --data data/processed/my_task.jsonl \
    --n 200 --out results/my_task_invariance.md
```

- `check_bundle.py` reloads the bundle through the inference path (`DecisiveLM.load`, then `system_one`) and compares
  every answer on the first `--n` test rows with the predictions written at training time. It passes when every
  probability is within `--tolerance` of the written one (default 0.10 for a bf16 bundle, 0.001 for one trained in
  float32) and every answer that changed was a near tie in the written prediction, with its top two options within twice
  the tolerance. A larger difference means the bundle is wired wrongly. It also checks every response's conformance
  (chapter 11), exits with an error if one fails, and prints a latency figure.
- `check_invariance.py` measures, per question type, how far answers move when only the packing changes: options
  reversed (Choice), another request packed first, one more question added, or the identical pass repeated. It reports
  the largest probability change, answer flips, and churn in the least-confident 20%. Section 11.7 gives the values
  measured on E4B in bf16 and in fp32 for comparison ([chapter 11](11-evaluation.md)).

To use the bundle, load it with the backbone it was trained on:

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/P3_my_task_e4b_r8_seed0")
response = dlm.system_one({
    "state": "I was charged twice for my subscription this month.",
    "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                           "criteria": {"billing": "Payment, refund or subscription issues",
                                        "technical": "Bugs, outages or integration problems",
                                        "sales": "Pricing, plans or new-account questions"}}},
})
print(response["answers"]["team"])
```

- Ask the questions with the same wording you trained on. The fine-tune has learned those words, and other wording
  gets weaker answers.
- A fine-tune answers any question it is given, including options and wording it never trained on. Only a question of a
  type it never trained on, such as a Noul asked of a fine-tune trained on Choice questions, gets a warning in
  `dlm.warnings`. Its calibration was fitted only on the questions in your `gold_calib`, so treat answers to other
  questions as uncalibrated. Ask the general head those questions instead, and calibrate it on about 100 labelled items
  of each before you set a review threshold (section 10.6).
- A bundle belongs to one backbone. `calibration.json` records its model id and revision. `DecisiveLM.load` loads that
  backbone, and `attach_bundle` and the check scripts refuse a different one.
- To score a whole file with a bundle, use `scripts/predict_bundle.py` ([section 3.8](03-quickstart.md)).

## 9.9 Images and audio

A task whose states are photos or recordings is fine-tuned like a text task. Each row lists its media
([chapter 8](08-data-format.md)), with paths relative to the repository root:

```json
{"state": "A photo.", "media": [{"type": "image", "path": "data/raw/my_photos/0001.jpg"}], "questions": {...}}
```

The steps are the same as for text:

```bash
# P0: the base model prompted with each row's image or clip, on this machine's GPU
uv run python scripts/label_teacher.py --teacher gemma4_e4b --local --data data/processed/my_task.jsonl
# the fine-tune: from the general head (shown), or from the base model with --standardize-rows N in place of
# --init-bundle and the default learning rates (section 9.6)
uv run python scripts/train_p3.py --data data/processed/my_task.jsonl --name my_task_e4b_seed0 --seed 0 \
    --backbone e4b --adapter user --rows-per-step 16 --limit-val 200 --select loss --max-tokens 4096 --patience 4 \
    --epochs 3 --eval-every 50 --init-bundle bundles/decisivelm-e4b-general --lr-lora 5e-5 --lr-head 1e-5
uv run python scripts/eval_dataset.py --data data/processed/my_task.jsonl --reference P0 \
    --predictions P0=data/predictions/teacher_my_task__gemma4_e4b.jsonl P3=data/predictions/my_task_e4b_seed0.jsonl
```

- Audio needs the `audio` extra (`uv sync --extra audio`) and the E4B backbone. 26B A4B has no audio encoder.
- Images are encoded at 280 soft tokens by default and audio at one token per 40 ms (up to 30 s), so a row costs more
  than a short text. Use `--max-tokens 4096`, as in the command above (the default is 8192).
- For judgements about fine visual detail, such as photo quality, give images more tokens with `--image-budget 560` or
  `--image-budget 1120` (Gemma 4 also accepts 70 and 140). The bundle records it, and `predict_bundle.py` uses the
  same. Training takes longer: on photo quality, about 40 minutes at 1,120 against 8 at 280.
- P0's prompt places the image or clip where the state starts, as the fine-tune's input does.

Three public examples show what to expect. Each has many raters per item, so the targets carry real disagreement. "The
general head" in these examples is the released E4B head (General v5c). None has been run from the 26B A4B head. None of
the three tasks is in General v5c's mix, whose only image set is VizWiz-VQA and which has no audio. On the test split,
seed 0 (the 300-item runs were repeated on two more seeds, each with its own 300 items):

| Example | Raters | P0 | 300 items, from base | 300 items, from the general head | All items, from base | All items, from the general head |
|---|---|---|---|---|---|---|
| Photo quality (KonIQ-10k), MAE / ρ | ~107 per photo | 0.389 / 0.12 | 0.228 / 0.09 | 0.209 / 0.10 | **0.145** / **0.36** | 0.159 / 0.32 |
| Voice emotion (CREMA-D), accuracy / ρ | ~10 per clip | 0.551 / 0.01 | 0.577 / 0.11 | 0.605 / 0.13 | **0.755** / **0.51** | 0.734 / 0.48 |
| Is it alive? (THINGSplus), MAE / ρ | ~37 per object | 0.365 / 0.48 | 0.239 / 0.50 | 0.177 / 0.64 | 0.186 / 0.61 | **0.162** / **0.67** |

MAE is the mean absolute error, the mean distance in levels between the predicted score and the raters' mean level
(lower is better). ρ is the correlation of the top probability with per-item rater agreement (chapter 11). What they
show:

- **Prompted, the base model barely does two of these tasks.** On photo quality it beats always answering the most
  common level, but its confidence hardly follows the raters' agreement. On voice emotion it answers "neutral" for 91%
  of clips, with confidence unrelated to the listeners' agreement. It does better on the living-thing rating, where its
  ρ (0.48) is close to the 300-item fine-tune from base (0.50), though its MAE is higher (0.365 against 0.239).
- **With enough items, a fine-tune learns them**, and its confidence follows where people disagree. With all the
  items, the starting point matters less. The fine-tune from the base model is best on voice emotion and photo quality,
  and the one from the general head on the living-thing task.
- **With 300 items, it depends on the task.** From the base model, voice emotion is not learned on any of the three
  seeds (it scores the same as always answering "neutral"), and photo quality gets more accurate but its confidence does
  not yet follow agreement. Starting from the general head helps voice emotion on every seed (accuracy +0.021 to
  +0.042). It helps the living-thing task on two of the three seeds (MAE 0.062 and 0.019 lower, and level on the third),
  where the judgement is about the named object rather than the pixels. With 300 objects it does as well as a fine-tune
  from the base model on all 1,128. It gives no consistent gain on photo quality (MAE 0.018 lower, 0.008 higher and
  level).
- **More image detail helps photo quality.** At 1,120 image tokens instead of 280, MAE falls on every seed: by 0.033,
  0.014 and 0.017 from the base model, and by 0.012, 0.021 and 0.015 from the general head.
- **The earlier head gave the same results.** The same runs from General v3d, an earlier head trained on text only,
  differed from these by no more than the spread between seeds.

To run an example yourself (`koniq10k`, `crema_d` or `things_living`):

```bash
uv run python scripts/fetch_dataset.py things_living
uv run python scripts/convert_dataset.py things_living
```

Results: `results/media_<example>_gold_test_v5c.md`; the three seeds and the image-detail runs:
`results/media_seeds_gold_test_300_v5c.md`; the comparison with v3d's runs: `results/media_v5c.md`. The datasets and
their licences are in
[chapter 14](14-datasets.md) (section 14.3).

Next: [10. Calibration](10-calibration.md)
