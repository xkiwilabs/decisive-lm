# 9. Fine-tuning for your task

A **task fine-tune** is a small adapter trained on your own labelled decisions, so that one forward pass answers your
questions the way your human raters would and says how sure it is. This chapter covers the whole path: preparing
labels, measuring the baseline to beat, training, calibrating, evaluating and putting the result to use. A fine-tune
starts either from base Gemma 4 or from the general head (§9.6), and needs one GPU.

## 9.1 The path at a glance

| Step | What you do | Tool |
|---|---|---|
| 1 | Convert your labelled data into canonical records, with rater shares | a converter ([chapter 8](08-data-format.md)) |
| 2 | Check the data | `scripts/inspect_data.py` |
| 3 | Measure the baseline: base Gemma prompted directly (P0) | `scripts/run_p0.py` |
| 4 | Optional: a quick probe that trains the head only (P1) | `scripts/cache_features.py`, `scripts/train_p1.py` |
| 5 | Train the fine-tune (P3): a LoRA adapter, the query tokens and the head, from base or from the general head | `scripts/train_p3.py` |
| 6 | Evaluate against P0 and against your raters | `scripts/eval_dataset.py`, `scripts/compare_coding.py` |
| 7 | Check the bundle, then use it | `scripts/check_bundle.py`, `scripts/check_invariance.py`, `attach_bundle` |

Calibration is part of training: the training scripts fit temperatures on `gold_calib` and write calibrated
predictions ([chapter 10](10-calibration.md)).

## 9.2 Preparing labelled data

**Ask the questions you will ask in production.** Write each question once, in the request format of
[chapter 4](04-requests-and-responses.md). The same `instructions` and option descriptions go into every row, and the
fine-tune learns to judge exactly those words. [Chapter 5](05-writing-good-questions.md) explains how to write options
that can be judged one at a time, including self-contained `other` options.

**Use several raters where you can, and train on their shares.**

- If three people label an item and two choose `billing` and one `technical`, the target is
  `{"billing": 0.667, "technical": 0.333, "sales": 0.0}` with `n_raters: 3`.
- Never reduce this to the majority vote. The shares are what teach the model to be unsure where people are unsure,
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
| `val` | 10% | Early stopping |
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

- The model is asked the question with lettered options and the answer prefilled, so the next token is the label.
- Its label probabilities are read directly.
- Every rotation of the options is averaged (or 8 seeded orderings above 8 options) to cancel position bias.
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

For any question type, and for rows with images or audio, run P0 with the labeller on this machine's GPU instead
(section 9.9):

```bash
uv run python scripts/label_teacher.py --teacher gemma4_e4b --local --data data/processed/my_task.jsonl
```

## 9.4 Optional: a head-only probe (P1)

P1 trains only the small scorer head on hidden states cached from the frozen backbone. It runs in minutes once the
features are cached, and it shows how much of the task the base model already represents. It is a diagnostic; a
P1 head can also warm-start P3 (§9.5).

```bash
uv run python scripts/cache_features.py --data data/processed/my_task.jsonl --out data/features/my_task_e4b.pt
uv run python scripts/train_p1.py --features data/features/my_task_e4b.pt --data data/processed/my_task.jsonl \
    --standardize --seed 0
```

- `cache_features.py` runs the backbone once per row and saves progress every `--save-every` rows, so it resumes
  after an interruption.
- `train_p1.py` trains on `train` with the same per-type losses as adapter training, stops early on `val` (`--patience`,
  default 15 epochs) and fits calibration on `gold_calib`. It writes predictions and a bundle named
  `P1_<features>[_z]_seed<seed>`.
- `--standardize` z-scores each feature dimension before the head. It is required on 26B A4B and helps on E4B.

## 9.5 Training the fine-tune (P3)

P3 trains three things together:

- a LoRA adapter on the language model's decoder projections;
- the three query-token embedding rows;
- the scorer head.

It stops early on `val`, fits calibration on `gold_calib` and writes the bundle and calibrated predictions.

```bash
uv run python scripts/train_p3.py --data data/processed/my_task.jsonl --standardize-rows 2000 --seed 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | | Canonical records with `train`, `val`, `gold_calib` and `gold_test` rows. |
| `--backbone` | `e4b` | `e4b` or `26b-a4b` (profiles in `configs/model.yaml`). |
| `--rank`, `--alpha`, `--dropout` | 8, 16, 0.05 | LoRA rank, scaling and dropout. |
| `--epochs` | 2 | Passes over `train`. Training usually stops earlier. |
| `--rows-per-step` | 16 | Requests per optimiser step. |
| `--max-tokens` | 8192 | Packed tokens per micro-batch (memory; see below). |
| `--lr-lora`, `--lr-head` | 2e-4, 1e-3 | Learning rates for the LoRA weights and for the head plus query rows. |
| `--warmup-frac` | 0.03 | Linear warm-up share of the steps, then cosine decay. |
| `--eval-every`, `--patience` | 40, 3 | Validate every N steps; stop after N validations without improvement. |
| `--limit-val` | all | Early-stop on a seeded random sample of this many `val` rows. |
| `--select` | `loss` | The validation criterion that picks the kept snapshot (below). |
| `--predict-best` | none | Also write a bundle and predictions from each listed criterion's best snapshot, named `<name>_by<criterion>`. |
| `--standardize-rows` | 0 | Z-score the head's input using the base model's states on this many `train` rows. |
| `--init-head` | | A bundle (e.g. a P1 run) whose head warm-starts this one. |
| `--init-bundle` | | A trained bundle (e.g. the general head) whose LoRA, query rows and head this run trains further (§9.6). |
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

**Warm-starting from P1.** `--init-head bundles/P1_...` starts the head from a trained P1 head and brings that head's
standardisation, so it cannot be combined with `--standardize-rows`. Set `--lr-head` near `1e-5` when warm-starting:
an optimiser's first steps move every weight by about the learning rate, and at `1e-3` that knocks a converged head
off its optimum.

**Memory.** `--max-tokens` bounds the packed tokens per micro-batch. Several short requests are packed together until
the budget is reached, and a request over budget runs alone.

| Hardware | Suggested settings |
|---|---|
| A 96 GB GPU, E4B | `--max-tokens 4096` with checkpointing (the default) |
| A GPU shared with other work | `--max-tokens 2048` |
| Apple silicon with large unified memory, E4B | `--no-checkpointing --max-tokens 2048`: one request per micro-batch, about 2 s per row |

Packing several requests into a long micro-batch saves little, because the attention mask is dense and the full
square of the sequence is computed. Turning off checkpointing on a large micro-batch multiplies memory several times
over.

**26B A4B.** The larger backbone needs about 50 GB for its weights. Train it with `--backbone 26b-a4b
--standardize-rows 2000 --dtype float32`. The fp32 compute keeps expert routing stable when requests are packed
differently; the weights stay in bf16, so this costs no extra precision loss. A 26B A4B bundle cannot take audio.

**What P3 writes.**

- `bundles/<name>/`:
  - `adapter.pt`: the LoRA weights and query-token rows;
  - `scorer.pt`: the head, with its standardisation;
  - `calibration.json`: the temperatures, the Noul bias, the backbone profile and revision, the LoRA settings, the
    render version, the training data file, the selection rule and step, and the validation history.
- `data/predictions/<name>.jsonl`: calibrated predictions for every question of `gold_calib` and `gold_test`, in the
  format the evaluation scripts read.

Before training, `train_p3.py` checks the `train` rows with `check_mix`: no held-out rows for the chosen adapter kind,
and for `--adapter general` release tier only.

## 9.6 Starting from the general head

With few labels, start from the general head instead of the base model. The fine-tune then begins with a model that
already knows how to answer typed questions and how to be unsure, and your labels only have to teach it your task.

```bash
uv run python scripts/train_p3.py --data data/processed/my_task.jsonl \
    --init-bundle bundles/<general head> --lr-lora 5e-5 --lr-head 1e-5 --seed 0
```

- `--init-bundle` loads the bundle's LoRA weights, query-token rows and head, standardisation included, and trains them
  further. The LoRA rank, scaling and target modules come from the bundle, so `--rank` and `--alpha` are ignored. The
  bundle must match `--backbone`.
- It cannot be combined with `--init-head` or `--standardize-rows`: the bundle already brings a head and its
  standardisation.
- The bundle's own validation score, before any step, counts as a candidate snapshot. A fine-tune that never improves on
  the general head keeps the general head's weights.
- Calibration is refitted on your `gold_calib` as usual. The general head's temperatures
  ([chapter 10](10-calibration.md)) are not carried over.

**Lower learning rates.** Use about `5e-5` for the LoRA weights and `1e-5` for the head, not the defaults of `2e-4` and
`1e-3`. An optimiser's first steps move every weight by about the learning rate. At the defaults, that knocks a
converged head off its optimum before your labels can guide it.

**How much it helps.** We compared fine-tunes from the general head with fine-tunes from the base model (release
criterion 8, [chapter 15](15-release-criteria.md)). Each was trained on 100 and on 300 labelled items of two tasks
outside the general head's training mix, with three seeds per arm. Both arms used the same items and the same
settings (`--epochs 10 --eval-every 5 --patience 4 --limit-val 200 --select loss`). The base arm used its default
learning rates and `--standardize-rows` on its training items; the general arm used the rates above. Each fine-tune was
calibrated on its own task's `gold_calib` and scored on `gold_test`. Values are means over the three seeds. Δ is the
general start minus the base start, with a 95% interval from item-level bootstrap resamples.

MultiNLI (does the second sentence follow from the first?; 3 options, five annotators per pair, 4,823 test pairs):

| Measure | 100 items: base | 100: general | Δ (95% CI) | 300 items: base | 300: general | Δ (95% CI) |
|---|---|---|---|---|---|---|
| Accuracy against the raters' plurality | 0.843 | 0.862 | +0.019 [+0.011, +0.026] | 0.854 | 0.872 | +0.019 [+0.012, +0.025] |
| ρ of confidence with rater agreement | 0.351 | 0.402 | +0.050 [+0.035, +0.065] | 0.377 | 0.423 | +0.046 [+0.032, +0.060] |
| Tie AUROC | 0.736 | 0.757 | +0.022 [−0.003, +0.049] | 0.754 | 0.779 | +0.025 [−0.002, +0.054] |
| Errors caught by reviewing the least confident 20% | 0.513 | 0.560 | +0.047 [+0.021, +0.073] | 0.541 | 0.576 | +0.035 [+0.009, +0.061] |
| ECE (lower is better) | 0.070 | 0.057 | −0.013 [−0.020, −0.006] | 0.064 | 0.055 | −0.009 [−0.016, −0.003] |
| Soft NLL (lower is better) | 0.525 | 0.487 | −0.038 [−0.044, −0.033] | 0.502 | 0.473 | −0.029 [−0.035, −0.024] |

A private coding study (8 codebook categories, four coders, 792 test responses; its data is not distributed):

| Measure | 100 items: base | 100: general | Δ (95% CI) | 300 items: base | 300: general | Δ (95% CI) |
|---|---|---|---|---|---|---|
| Accuracy against the coders' plurality | 0.866 | 0.922 | +0.056 [+0.041, +0.070] | 0.919 | 0.931 | +0.012 [+0.001, +0.023] |
| ρ of confidence with coder agreement | 0.473 | 0.548 | +0.075 [+0.048, +0.102] | 0.575 | 0.605 | +0.030 [+0.002, +0.058] |

On both tasks and at both sizes, starting from the general head gave a more accurate fine-tune whose confidence
followed human agreement more closely. It was not worse on any measure. The gain is largest with 100 items. On the coding
study, 100 items from the general head came within 1.3 points of a fine-tune from base on all 2,592 training items
(0.935). Tie AUROC did not differ clearly on either task. The intervals reflect the sampling of test items, not
the spread between seeds; the per-seed values are in the criterion-8 report (`scripts/score_c8.py`).

## 9.7 Evaluating

Compare the fine-tune with P0 on `gold_test`, using the same items and the same bootstrap resamples:

```bash
uv run python scripts/eval_dataset.py --data data/processed/my_task.jsonl \
    --predictions "P3=data/predictions/P3_my_task_e4b_r8_seed0.jsonl" \
                  "P0=data/predictions/P0_my_task_e4b.jsonl" \
    --reference P0 --out results/my_task_gold_test.md
```

For each question this reports accuracy, calibration and, where most items have three or more raters, review
readiness. It also gives the primary metric with a 95% confidence interval and the paired difference from the
reference. [Chapter 11](11-evaluation.md) explains every column. Read it in this order:

1. **Conformance and coverage.** Every method answered every item.
2. **Review readiness.** Does the fine-tune's confidence track human agreement (ρ, tie AUROC) at least as well as
   P0's, and does reviewing its least confident 20% catch more of its errors?
3. **Calibration.** Is the ECE small, and lower than P0's?
4. **Accuracy.** Is the primary metric's paired difference from P0 above zero, with an interval that excludes zero?

**Comparing with individual human coders.** `scripts/compare_coding.py` produces the full report for a multi-coder
coding task:

- Cohen's κ against each coder, beside the human ceiling (the mean pairwise κ between coders);
- accuracy against the coders' consensus;
- soft NLL, ECE and Brier;
- review readiness for two confidence signals, a confidence-guided review at 10% and 20%, and recall per category.

It reads each coder's own labels from a CSV with one column per coder. The coder columns (`HUMANS`) and the id
mapping follow the coding study's table, so adapt them to yours ([chapter 11](11-evaluation.md)).

## 9.8 Checking and using the bundle

Two checks before you rely on a bundle:

```bash
uv run python scripts/check_bundle.py --bundle bundles/P3_my_task_e4b_r8_seed0 --data data/processed/my_task.jsonl --n 20
uv run python scripts/check_invariance.py --bundle bundles/P3_my_task_e4b_r8_seed0 --data data/processed/my_task.jsonl \
    --n 200 --out results/my_task_invariance.md
```

- `check_bundle.py` reloads the bundle through the inference path (`DecisiveLM.load`, then `system_one`) and compares
  every answer on the first `--n` test rows with the predictions written at training time. Any difference beyond bf16
  rounding means the bundle is wired wrongly. It also checks every response's conformance (chapter 11), exits with an
  error if one fails, and prints a latency figure.
- `check_invariance.py` measures, per question type, how far answers move when only the packing changes: options
  reversed (Choice), another request packed first, one more question added, or the identical pass repeated. It reports the largest probability change,
  answer flips, and churn in the least-confident 20% ([chapter 11](11-evaluation.md)).

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
- A bundle belongs to one backbone. `calibration.json` records its model id and revision; `DecisiveLM.load` loads that
  backbone, and `attach_bundle` and the check scripts refuse a different one.
- To score a whole file with a bundle, use `scripts/predict_bundle.py` ([chapter 7](07-using-the-general-head.md)).

## 9.9 Images and audio

A task whose states are photos or recordings is fine-tuned exactly like a text task. Each row lists its media
([chapter 8](08-data-format.md)), with paths relative to the repository root:

```json
{"state": "A photo.", "media": [{"type": "image", "path": "data/raw/my_photos/0001.jpg"}], "questions": {...}}
```

The steps are the same as for text:

```bash
# P0: the base model prompted with each row's image or clip, on this machine's GPU
uv run python scripts/label_teacher.py --teacher gemma4_e4b --local --data data/processed/my_task.jsonl
# the fine-tune: from the general head (shown) or from the base model (--standardize-rows N instead of --init-bundle)
uv run python scripts/train_p3.py --data data/processed/my_task.jsonl --name my_task_e4b_seed0 --seed 0 \
    --backbone e4b --adapter user --rows-per-step 16 --limit-val 200 --select loss --max-tokens 4096 --patience 4 \
    --epochs 3 --eval-every 50 --init-bundle bundles/general_v3d_e4b_r8_seed0 --lr-lora 5e-5 --lr-head 1e-5
uv run python scripts/eval_dataset.py --data data/processed/my_task.jsonl --reference P0 \
    --predictions P0=data/predictions/teacher_my_task__gemma4_e4b.jsonl P3=data/predictions/my_task_e4b_seed0.jsonl
```

- Audio needs the `audio` extra (`uv sync --extra audio`) and the E4B backbone; 26B A4B has no audio encoder.
- Images are encoded at 280 soft tokens and audio at one token per 40 ms (up to 30 s), so a row costs more than a
  short text. Keep `--max-tokens 4096`.
- P0's prompt places the image or clip where the state starts, as the fine-tune's input does.

Three public examples show what to expect. Each has many raters per item, so the targets carry real disagreement.
One seed each, on the test split:

| Example | Raters | P0 | 300 items, from base | 300 items, from the general head | All items, from base | All items, from the general head |
|---|---|---|---|---|---|---|
| Photo quality (KonIQ-10k), MAE / ρ | ~107 per photo | 0.389 / 0.12 | 0.228 / 0.09 | 0.220 / 0.05 | **0.145** / 0.36 | 0.155 / **0.37** |
| Voice emotion (CREMA-D), accuracy / ρ | ~10 per clip | 0.551 / 0.01 | 0.577 / 0.11 | 0.613 / 0.15 | **0.755** / **0.51** | 0.745 / 0.49 |
| Is it alive? (THINGSplus), MAE / ρ | ~37 per object | 0.365 / 0.48 | 0.239 / 0.50 | **0.184** / **0.64** | 0.186 / 0.61 | 0.168 / 0.67 |

ρ is the correlation of the top probability with per-item rater agreement (chapter 11). What they show:

- **Prompted, the base model barely does these tasks.** On photo quality its error is worse than always answering
  the most common level, and on voice emotion it answers "neutral" for 91% of clips, with confidence unrelated to the
  listeners' agreement.
- **With enough items, a fine-tune learns them**, and its confidence follows where people disagree.
- **With 300 items, it depends on the task.** On photo quality and voice emotion, a fine-tune from the base model
  learns little more than the most common answer. On the living-thing task, where the judgement is about the named
  object rather than the pixels, starting from the general head (trained on text only) helps clearly: with 300
  objects it matches a fine-tune from the base model on all 1,128.

To run an example yourself (`koniq10k`, `crema_d` or `things_living`):

```bash
uv run python scripts/fetch_dataset.py things_living
uv run python scripts/convert_dataset.py things_living
```

Results: `results/media_<example>_gold_test.md`. The datasets and their licences are in
[chapter 14](14-datasets.md) (section 14.3).

Next: [10. Calibration](10-calibration.md)
