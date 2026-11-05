# 12. Teacher labelling

A *teacher* is an open model, usually larger than the student, that labels rows which have no human answers, so that the
head can learn from many more examples than people could rate. DecisiveLM asks the teacher through a vLLM server's
OpenAI-compatible API. vLLM is an open-source server that runs a language model on a GPU and answers requests in the
format of OpenAI's API. You can start one on your own GPU or use an existing one, named by `base_url` in the teacher's
entry (12.1). A language model gives every token it could write next a probability, and the server returns the
logarithms of the most likely ones (*log-probabilities*). The labeller reads a probability distribution over the answer
labels from them, never a single sampled answer. This chapter covers how the teacher is prompted and how its answers
become calibrated training targets. The same method, run on the student's own backbone, is the P0 baseline every model
is compared with.

## 12.1 The teacher and its configuration

Teachers are registered in `configs/teachers.yaml`. The default teacher is Gemma 4 26B A4B (`gemma4_26b_a4b`). Gemma 4
E4B prompted directly, the backbone of an E4B student, is registered as `gemma4_e4b`. A 26B A4B student's backbone is
the default teacher itself. Each entry records what the labeller needs and what every teacher target stores:

| Key | Meaning |
|---|---|
| `base_url`, `model` | The server's OpenAI-compatible endpoint (`…/v1`) and the served model name |
| `revision` | The Hugging Face revision of the served weights, stored in every target |
| `tokenizer_backbone` | The `configs/model.yaml` profile whose tokenizer and chat template render the prompts |
| `licence`, `developer`, `base_model_origin` | Licence and origin, for the model-origin rule (16.2) |
| `modalities` | What the model accepts (`text`, `image`, `audio`) |
| `vllm_version` | The server's vLLM version, stored in every target |
| `max_logprobs` | How many top log-probabilities the server returns per token (vLLM's default is 20) |
| `logprobs_mode` | `raw_logprobs`: log-probabilities before the server applies its sampling temperature and penalties. This sampling temperature is unrelated to the calibration temperatures of 12.9 |
| `prefix_caching` | Whether the server caches shared prompt prefixes (it should) |
| `api_key_env` | The environment variable holding the server's API key, sent as a bearer token when set; `null` for a server without one |
| `schedule` | How many requests to keep open, by time of day (12.7) |

Any other open model can be added the same way. A teacher must:

- be served by vLLM with the OpenAI-compatible API;
- have a licence that permits training on its outputs;
- meet the model-origin rule.

## 12.2 How a question is asked

The labeller and the student see the same input. Both call `decisivelm/preprocess.py` on the serialised state, so a
long state is truncated identically for both (8,192 tokens by default, with 70% of that budget taken from the start of
the state and 30% from its end). Any difference in what they saw would become label noise.

Each question type has a versioned prompt template in `decisivelm/synth/prompts/`:

| Template | Used for |
|---|---|
| `choice_v1.yaml` | Choice with up to 20 options, lettered A, B, C … |
| `choice_num_v1.yaml` | Choice with more than 20 options, numbered 01, 02, … |
| `score_v1.yaml` | Score, levels lettered in the order shown |
| `noul_v1.yaml` | Noul, three phrasings of a yes/no question |

All four work the same way:

- the prompt is rendered with the teacher's own chat template;
- thinking (Gemma 4's option to write out its reasoning before it answers) is turned off;
- the assistant turn is pre-filled with `Answer:` (or `Answer: ` for numbered options).

The next token the model would write is therefore the label itself, not reasoning or formatting. The labeller asks the
server for that one token's top log-probabilities (`/v1/completions` with the prompt sent as token ids, `max_tokens: 1`,
`temperature: 0`). It then reads the probability of each allowed label.

In every template the state comes last. Everything before it (instructions, question, options) is the same for every
row that asks the same question. The labeller sends those rows together, so the server's prefix cache reuses the
shared part.

## 12.3 Choice

Options are shown with their names and descriptions, each behind a label. A model tends to favour options by position
(often the first), so the question is asked several times with the options in different orders and the results are
averaged:

- **up to 8 options**: every rotation (n orderings);
- **more than 8 options**: 8 permutations, seeded from the row's id, so a rerun asks the same orderings.

Averaging over orderings cancels the position preference while keeping any preference for an option's content.

**Lettered labels (up to 20 options).** Each option gets a capital letter. Every letter is a single token in Gemma's
tokenizer, and a letter and its space-prefixed form ("A" and " A") are merged by adding their probabilities. The
server returns the top 20 log-probabilities, so every letter can be read in one request.

**Numbered labels (more than 20 options).** With more options than the server returns log-probabilities for, the
options are numbered with zero-padded labels of equal width ("01" … "60"; "001" … "151" for 151 options). The label is
read one digit at a time:

1. One request reads the first digit. Only digits that begin a real label count, renormalised among themselves.
2. For each first digit holding at least 0.1% of the probability, one more request reads the next digit after it.
3. A prefix below 0.1% is not expanded. Its probability is spread evenly over the labels it would lead to.

Gemma's tokenizer splits numbers into single-digit tokens, and the separating space is its own token. The prompt
therefore ends in `Answer: ` and the next token is the first digit. This per-option scoring reads every option's
probability, however many options there are and however few log-probabilities the server returns. It costs a few
requests per ordering.

## 12.4 Score

A Score's levels are ordered, and shuffling them would set the teacher a different task from the one the student
learns, so they are never shuffled. Each Score is asked twice, with the levels lettered in natural order (A = the first
level) and in reversed order (A = the last). The two distributions are mapped back to levels and averaged, which
cancels a preference for the first or last letter.

## 12.5 Noul

A Noul is asked as a yes/no question in three phrasings (`noul_v1.yaml`). Where the request gives `criteria`, the
prompt includes them as "Yes means: …" and "No means: …". The answer for one phrasing is
P(Yes) / (P(Yes) + P(No)), with "Yes" and " Yes" (and "No" and " No") merged. The three phrasings are averaged.

## 12.6 Allowed-label mass

Each request records its **allowed mass**: the total probability the model put on valid labels. For numbered options
this is the probability of the first digit being one that starts a real label. A low allowed mass means the model
wanted to answer with something else (a word, an apology, a different format), so its distribution over the labels
says little.

- Orderings or phrasings with allowed mass below **0.5** are left out of the average.
- If none of them reaches 0.5, all are kept, and the row records its low mean allowed mass.
- When teacher targets are written (12.9), a question whose mean allowed mass is below 0.5 gets no target. It is
  masked in the loss, so it adds nothing to the training loss while the row's other questions still train.

Every result records the method (`logprob`), the number of orderings asked and kept, and the mean allowed mass.

## 12.7 Running a labelling job

`scripts/label_teacher.py` labels rows from canonical data files:

```bash
uv run python scripts/label_teacher.py --teacher gemma4_26b_a4b \
    --data data/processed/helpsteer2.jsonl data/processed/civil_comments.jsonl:2000 \
    --splits gold_calib gold_test
```

- `--data PATH[:N]` takes one or more files. `:N` samples N rows per split, seeded so that a rerun picks the same rows.
- `--splits` chooses which splits to label (default `gold_calib gold_test`). Use `--splits train val` for generated
  rows.
- `--chunk` sets how many rows are written at a time. `--groups-in-parallel` sets how many question groups are labelled
  at once. Rows asking the identical question form one group and share prompts.
- Raw distributions go to `data/teacher/<stem>__<teacher>.jsonl`, one line per row, appended chunk by chunk.
- Jobs are **resumable**: a restart skips rows already in the output file, so a job can run overnight or over several
  days.
- When a file is complete and has `gold_calib` rows, the teacher's temperatures are fitted there against the human
  labels, the same way the student's are (chapter 10). Calibrated predictions then go to
  `data/predictions/teacher_<stem>__<teacher>.jsonl`, ready for `scripts/eval_dataset.py`. A file with `gold_test` rows
  but no `gold_calib` rows (a test-only benchmark) gets its raw, uncalibrated distributions there instead.

Every output line records the teacher model, its revision, the vLLM version, `max_logprobs` and the SHA-256 hash of each
prompt template. A label can always be traced to the exact prompt and weights that produced it. A mixture-of-experts
teacher such as Gemma 4 26B A4B sends each token to 8 of its 128 experts, and where two experts score almost equally,
the choice can change with what else the server is batching (`results/M0.md`). Its log-probabilities for the same row
can therefore differ between runs, so results are compared as distributions, not as exact numbers.

**Sharing a server.** A labelling job can run on a server that other people use. The `schedule` block of a teacher
limits how many requests are open at once:

| Key | Meaning |
|---|---|
| `day_hours: [start, end]` | Local hours counted as "day" |
| `day_max_in_flight` | Most requests open at once during the day |
| `night_max_in_flight` | Most requests open at once outside those hours |
| `yield_when_others_waiting` | During the day, send nothing new while the server reports queued requests from others |

The limit is re-read before every request, so a job that runs past the end of the day speeds up by itself. The
server's queue is read from vLLM's `/metrics` endpoint (`vllm:num_requests_waiting`).

Recommended server settings:

- prefix caching on;
- the default raw log-probabilities mode;
- a `--max-logprobs` of at least 20, recorded in the teacher's entry.

Guided (constrained) decoding is not used, because the log-probabilities it returns may be taken before or after the
constraint depending on the vLLM version.

On a server, the labeller skips rows with images or audio and labels text rows only. To label those rows, run it
locally with `--local` (section 12.8).

## 12.8 The same method as a baseline (P0)

Prompting the student's own frozen backbone this way, with no training, is the **P0 baseline**. It is the number every
trained model must be compared with, because a user gets it from the base model with no fine-tuning.
There are three ways to run it:

- **On a vLLM server**: `scripts/label_teacher.py --teacher gemma4_e4b`, exactly as for the teacher. For a student on
  26B A4B, use `--teacher gemma4_26b_a4b`, the teacher itself.
- **Locally, on one data file**: `scripts/run_p0.py --data … [--backbone e4b]` loads the backbone with transformers
  and scores one question per row (`--question`, default `intent`). It writes raw distributions to
  `data/predictions/P0_<data>_<backbone>_raw.jsonl`. When every item is done, it fits one temperature on `gold_calib`
  and writes calibrated predictions next to them.
- **Locally, through the labeller**: `scripts/label_teacher.py --teacher gemma4_e4b --local` runs the same model on
  this machine's GPU, every question type, and also rows with images or audio, which the server path leaves out. Each
  row's media are encoded as the student encodes them and placed where the state starts
  (`decisivelm.synth.teacher.LocalTeacher`). Output goes where the server path's does.

All use the same prompts and label mechanics as the teacher (`decisivelm/synth/label.py`).

## 12.9 From teacher answers to training targets

Raw teacher distributions are not yet good targets. A prompted model's log-probabilities are usually over- or
under-confident, and the student cannot be better calibrated than the targets it learns from.
`scripts/teacher_targets.py` calibrates the teacher against human labels and writes the calibrated probabilities as
targets:

```bash
uv run python scripts/teacher_targets.py --data data/processed/civil_comments.jsonl --teacher gemma4_26b_a4b
```

This fits the teacher's temperatures on the file's own `gold_calib` rows (one per question type and option-count
bucket, chapter 10). It then writes `data/processed/<stem>_teacher.jsonl`: the `train` rows the teacher labelled, with
teacher targets in place of the human ones. Every other split is copied unchanged, so validation, calibration and test
stay human-labelled. If some train rows were not labelled, it also writes `<stem>_human_matched.jsonl`, holding the same
rows with their human targets for a like-for-like comparison.

Generated rows have no human labels of their own, so their teacher is calibrated on other datasets' human gold:

```bash
uv run python scripts/teacher_targets.py --data data/processed/synth_v1.jsonl --splits train val \
    --balanced --set score=choice \
    --calib data/processed/banking77_explode.jsonl data/processed/clinc150_explode.jsonl \
            data/processed/helpsteer2.jsonl data/processed/civil_comments.jsonl data/processed/goemotions.jsonl
```

`--set score=choice` gives Score the Choice temperature, chosen by judgement rather than fitted. Among these five
datasets the only human Score labels are HelpSteer2's subjective ratings, and their temperature (9.8) would make every
generated scale near-flat.

| Option | Effect |
|---|---|
| `--splits` | Which splits receive teacher targets (default `train`; generated rows use `train val`) |
| `--calib PATH …` | Fit the teacher's calibration on the pooled `gold_calib` rows of these datasets, each with its own teacher file |
| `--balanced` | One temperature per question type, fitted on an equal-sized seeded sample (up to 500 questions per type) from each `--calib` dataset, with the Noul bias at 0, so no single dataset's difficulty or base rate decides it |
| `--noul-bias {fit,zero}` | Without `--balanced`: fit the Noul bias or keep it at 0 (a bias fitted on one dataset mostly records its base rate) |
| `--set TYPE=T` or `TYPE=OTHER` | After fitting, override a type's temperature with a number or with another type's (`score=choice`) |
| `--blend TEACHER=W` | Mix a second teacher in: each target becomes (1 − W) × this teacher's calibrated probabilities + W × the second's, both calibrated the same way |
| `--out` | Output path (default `data/processed/<stem>_teacher.jsonl`) |

Each target records the method, orderings, allowed mass, teacher model and revision, vLLM version, prompt hash and
the temperature applied (`calibrated_T`).

Temperatures are always fitted on human-labelled rows, never on teacher-labelled ones, because fitting on the
teacher's own labels would calibrate the student to the teacher rather than to people.

## 12.10 What the teacher cannot fix

The student learns from the teacher, so it inherits the teacher's blind spots. A student whose backbone is the teacher,
such as the 26B A4B general head, can share them more directly (section 13.7). Four safeguards limit this:

- **Human gold.** The seed datasets' human labels train the head alongside the teacher's.
- **Human calibration.** Every temperature, the teacher's and the student's, is fitted on human-labelled rows.
- **Human gates.** Release gates use only human-labelled data the head never trained on (chapter 15).
- **The P0 comparison.** P0 is reported beside every result, so a trained head that is no better than prompting the
  base model is visible.

Next: [13. How the general head was trained](13-how-the-general-head-was-trained.md)
