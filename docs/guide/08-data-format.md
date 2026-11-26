# 8. Data format

Everything the toolkit trains on, calibrates on or evaluates is stored as **canonical records**: one JSON object per
line, validated against a single schema. This chapter describes that record and the rules around it:

- what a target is;
- what each split is for;
- how splits are assigned so that evaluation data never leaks into training;
- how licences, holdouts and the clean-room rule are enforced;
- how to convert your own data.

## 8.1 Rows and records

A **row** is one request: one state plus the set of questions asked about it. Dataset sizes are always counted in
rows. Each row is a `Record` (`decisivelm/data/schema.py`), and every file under `data/processed/` is a JSONL file of
records. Every writer validates against the schema, and every reader (`decisivelm.data.dataset.load_records`) validates
again.

```json
{
  "schema_version": 1,
  "id": "helpsteer2-validation-00412",
  "lineage": {"source_dataset": "helpsteer2", "source_id": "validation-00412",
              "family_id": "helpsteer2-prompt-3f9a0c1d2e4b", "parent_id": null, "multiplier": null},
  "license": "CC-BY-4.0",
  "tier": "R",
  "modality": ["text"],
  "language": "en",
  "state": {"conversation": "User: How do I ...", "response": "You can ..."},
  "media": [],
  "questions": {
    "helpfulness": {"type": "score",
                    "instructions": "How helpful is the response to the conversation's last user turn overall?",
                    "criteria": ["Not helpful at all", "Slightly helpful: misses most of what was asked",
                                 "Partially helpful: addresses the request with notable gaps or problems",
                                 "Mostly helpful: addresses the request well with minor gaps",
                                 "Fully helpful: completely addresses the request"]}
  },
  "targets": {
    "helpfulness": {"human": {"probabilities": {"0": 0.0, "1": 0.0, "2": 0.2, "3": 0.6, "4": 0.2}, "n_raters": 5}}
  },
  "preprocess": {"render_version": 1, "max_state_tokens": 8192, "truncated": false},
  "split": "gold_test"
}
```

| Field | Meaning |
|---|---|
| `schema_version` | Always `1`. |
| `id` | Unique row id. Seeded choices (option orderings, sampling) derive from it, so it must be stable. |
| `lineage` | Where the row came from: `source_dataset`, `source_id` (the item in the source), `family_id` (the leakage unit, §8.4), and for derived rows (rows made from another row, such as the variants of §8.10) `parent_id` and `multiplier`. |
| `license`, `tier` | The source's SPDX licence id (the standard short name from the SPDX License List, e.g. `CC-BY-4.0`, or `LicenseRef-<name>` for a licence not on it, such as your own data's) and its licence tier (§8.5). Both must match `configs/datasets.yaml`. |
| `modality` | Any of `text`, `image`, `audio`. |
| `language` | A language code (e.g. `en`) or a locale (e.g. `pl-PL`). |
| `state` | A string or a JSON object. JSON is serialised compactly, in insertion order. |
| `media` | Images or audio, each `{"type": "image" \| "audio"}` with exactly one of `"path"`, `"data"` or `"url"`, as in a request (section 4.5). A `url` is accepted in a record but refused when the head answers, so give a path or data. A relative `path` resolves against the repository root (section 9.9). |
| `questions` | The request's questions in the request format of [chapter 4](04-requests-and-responses.md), keyed by question id. |
| `targets` | Per question id, the labels to learn from (§8.2). A question with no target is masked out of the loss. |
| `preprocess` | The render version and state budget used, and whether the state was truncated. |
| `split` | One of `train`, `val`, `gold_calib`, `gold_test`, `holdout` (§8.3). |

The schema rejects unknown fields, so a misspelt key fails at once and is never silently ignored.

## 8.2 Targets

Each question may carry a **human** target, a **teacher** target, or both.

- **Choice and Score** targets are distributions in `probabilities`. A Choice target is keyed by option name. A Score
  target is keyed by level index as a string, `"0"` to `"n-1"`. The keys must match the question's options or levels
  exactly, the values must be non-negative, and they must sum to 1.
- **Noul** targets are one number in `noul`, between 0 and 1: the share of raters who said yes.
- A human target may record `n_raters`, the number of people who labelled the item. Review-readiness metrics need it
  ([chapter 11](11-evaluation.md)), so record it wherever you know it.
- A teacher target also records how it was produced: the method, number of orderings, allowed-label mass, teacher model
  and revision, server version, prompt hash and calibration temperature ([chapter 12](12-teacher-labelling.md)
  explains each). A question may carry a list of teacher targets when several teachers labelled it. Data with human
  labels only has no teacher targets, so these fields do not arise.

**Store the raters' label shares, never only the majority vote.** If three of five annotators chose level 3 and one each
chose levels 2 and 4, the target is `{"2": 0.2, "3": 0.6, "4": 0.2}`. The share lets a model's confidence follow human
disagreement. Majority votes tell a model that every item is certain. If each item has a single label, the target is
one-hot. For example, `{"billing": 1.0, "technical": 0.0, "sales": 0.0}` with `n_raters` left out or set to 1.

Training uses the human target where there is one, and otherwise the teacher target. Calibration and every acceptance
gate (a pass-or-fail release criterion for the general head, [chapter 15](15-release-criteria.md)) use human targets
only.

## 8.3 Splits

| Split | Used for | Never used for |
|---|---|---|
| `train` | Fitting the adapter and the scorer. | Any reported number. |
| `val` | Early stopping (ending training once results on `val` stop improving) and choosing between training snapshots. | Calibration, reported results. |
| `gold_calib` | Calibration: fitting the temperatures and the Noul bias, which rescale the model's probabilities so that its stated confidence matches how often it is right ([chapter 10](10-calibration.md)). Human labels only. | Training, reported results. |
| `gold_test` | Every reported result and acceptance gate. Human labels only. | Training, calibration, model selection. |
| `holdout` | Rows kept out of every training mix (§8.6). | Training. |

`gold_calib` and `gold_test` are split once, from human-labelled data, and are never mixed. A temperature fitted on
the test rows would make calibration look better than it is.

As a guide for your own data, send about 60–70% of labelled items to `train`, 10% to `val` and the rest to the gold
splits, with about 40% of the gold items in `gold_calib` and 60% in `gold_test`
([chapter 9](09-fine-tuning-for-your-task.md), section 9.2, gives the full plan). Calibration needs at least 100
labelled questions per question type to fit well. A public dataset's own test split becomes `gold_calib`/`gold_test`
and is never trained on.

## 8.4 Lineage: splitting before multiplying

Splits are assigned at the level of the **source item** and the **schema family** before any data is multiplied.

- A **source item** is one item in the original dataset (`lineage.source_id`).
- A **schema family** (`lineage.family_id`) is the unit that must not be split across train and test. Rows that
  share something a model could memorise share a family. That covers a base question with every rewording, renaming,
  coarsening and Noul split made from it, and rows that share a source prompt, document, conversation or speaker. A
  converter chooses the family key. HelpSteer2's responses to the same prompt share one family, for example, so the
  same prompt is never split across train and test.
- **Derived rows inherit their parent's split, licence, tier and family.** A row made from a `gold_test` row by
  rewording or subsetting its options is itself `gold_test`.

Converters assign splits deterministically from a hash of the family key, so rebuilding a dataset gives the same
splits:

```python
import hashlib

def _u(text: str) -> float:   # a uniform number in [0, 1) from a string
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16) / 16 ** 15

split = "val" if _u("mydata-split:" + family_key) < 0.10 else "train"
```

`scripts/convert_dataset.py` prints a hash of each split's ids, so you can check that a rebuild did not move rows.

## 8.5 Licence tiers

Every source has an entry in `configs/datasets.yaml` with its SPDX licence id and **tier**:

| Tier | Meaning | Allowed in |
|---|---|---|
| **R** (release) | Permissive, commercial use allowed, no share-alike (CC0, CC BY, MIT, Apache-2.0, …). | The published general head, task fine-tunes, evaluation. |
| **S** (share-alike) | CC BY-SA, ODbL, GPL. | Private fine-tunes and evaluation. |
| **X** (restricted) | Non-commercial, research-only, terms that leave it unclear whether use is allowed (a dataset with no licence and no restricting terms is R, section 14.1), third-party-copyright media, or containing proprietary-model outputs. | Task fine-tunes for your own use, private experiments and evaluation. Never in released weights. |

- The tier is derived from the licence (`decisivelm.data.dataset.tier_of`).
- An entry may place a source in a *more* restrictive tier with an `override` reason. No entry can make a source less
  restrictive.
- `load_records` refuses any row whose `license` or `tier` disagrees with the registry, and any source missing from
  it.
- A derived row takes the most restrictive tier of its inputs.

Mixes are built per tier (`decisivelm.data.mix.MIX_TIERS`):

| Mix | Tiers |
|---|---|
| `release` | R |
| `private` | R, S |
| `research` | R, S, X |

The general head is trained on a `release` mix only. A task fine-tune (`scripts/train_p3.py --adapter user`,
[chapter 9](09-fine-tuning-for-your-task.md)) is checked against the `research` mix, so it may train on rows of all
three tiers.

**Private data:** register your own sources in `configs/datasets.local.yaml` (section 8.8 shows an entry). It has the
same format, is gitignored and is read alongside the public registry. A local entry may add sources but may not
redefine a public one.

## 8.6 Holdouts

`configs/holdout.yaml` lists data that a training mix must never contain. Each entry names a `dataset`
(`lineage.source_dataset`) and the `adapters` it is barred from (`general` for the general head, `user` for a task
fine-tune, or both). It may narrow the holdout with selectors: `source_ids`, `family_ids` or `languages`. With no
selector, the whole dataset is held out.

```yaml
holdouts:
  - dataset: measuring_hate_speech
    adapters: [general, user]
    family_ids: [measuring_hate_speech-eval]
    reason: Frozen held-out suite; the other comments may train.
```

`decisivelm.data.mix.check_mix` runs before every training job (`scripts/build_mix.py`, `scripts/train_p3.py`). It
**refuses** a mix, rather than filtering it, if any row:

- matches a holdout for that adapter;
- is not in the `train` split;
- has a tier the mix does not allow.

A held-out row reaching a mix means a split or lineage error upstream, and it should be fixed there. The frozen
held-out suite used for releases is described in [chapter 15](15-release-criteria.md).

## 8.7 Clean-room inputs

If your data includes logs from another decision service, keep only the **inputs** and relabel them, by people or
with the teacher pipeline of [chapter 12](12-teacher-labelling.md). Never keep that service's answers
([chapter 16](16-provenance-licences-and-responsible-use.md) explains why). `decisivelm/data/cleanroom.py` enforces
this:

- `whitelist_request(log)` keeps only `state`, `questions` and `media` from a logged request and drops everything
  else.
- `scan_state(state)` flags a state that contains an answer-shaped object. This catches an earlier answer pasted into a
  later request. A flagged item needs a human decision before it is used.
- `find_canaries(root, canaries)` lists the files under a directory that contain any canary string. The test suite
  plants unique canary answers in a fixture log and fails if any of them appears under `data/processed/`. Run
  `uv run pytest tests/test_cleanroom.py` after converting new data.

## 8.8 Converting your own data

A converter is a function that reads raw files and returns a list of `Record`s. The built-in converters are in
`decisivelm/data/converters/`. `helpsteer2.py` (Score questions with rater shares) and `civil_comments.py` (Noul
questions with rater shares) are good models.

**1. Register the source.** Every row names its source dataset, and the tools refuse a row whose source is not
registered. A private source goes in `configs/datasets.local.yaml`, which git ignores, so it is never committed.
Create the file with an entry like this:

```yaml
version: 1
datasets:
  support_tickets:
    description: Customer-support messages, each labelled by three staff into one of three teams
    source: Our help desk, exported 2026-10-01
    license: LicenseRef-private
    tier: X
```

`LicenseRef-private` with tier X marks data you cannot publish (section 8.5). It can train your own fine-tunes and is
kept out of anything released.

**2. Write the converter.** A minimal converter for a CSV with one text column and several raters' labels, saved as
`data/converters/support_tickets.py` (the `data/` folder is ignored by git, so your converter stays private too):

```python
import collections, hashlib, pathlib
import pandas as pd

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "support_tickets"           # must have an entry in configs/datasets.local.yaml
QUESTION = {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payment, refund or subscription issues",
                         "technical": "Bugs, outages or integration problems",
                         "sales": "Pricing, plans or new-account questions"}}

def _u(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16) / 16 ** 15

def _split(key):
    u = _u("support_tickets-split:" + key)
    return "train" if u < 0.6 else "val" if u < 0.7 else "gold_calib" if u < 0.82 else "gold_test"

def convert(source_dir, tokenizer=None, max_state_tokens=MAX_STATE_TOKENS):
    df = pd.read_csv(pathlib.Path(source_dir) / "tickets.csv")   # columns: ticket_id, text, rater_1..rater_3
    out = []
    for d in df.to_dict("records"):
        labels = [d[c] for c in ("rater_1", "rater_2", "rater_3") if isinstance(d[c], str)]
        counts = collections.Counter(labels)
        target = HumanTarget(probabilities={k: counts[k] / len(labels) for k in QUESTION["criteria"]},
                             n_raters=len(labels))
        text, truncated = (truncate_state(d["text"], tokenizer, max_state_tokens) if tokenizer
                           else (d["text"], False))
        sid = str(d["ticket_id"])
        out.append(Record(
            id=f"{SOURCE}-{sid}",
            lineage=Lineage(source_dataset=SOURCE, source_id=sid, family_id=f"{SOURCE}-{sid}"),
            license="LicenseRef-private", tier="X", modality=["text"], language="en", state=text,
            questions={"team": QUESTION}, targets={"team": QuestionTargets(human=target)},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens,
                                  truncated=truncated),
            split=_split(sid)))
    return out
```

**3. Run it** from the repository root, with your CSV at `data/raw/support_tickets/tickets.csv`:

```bash
uv run python scripts/convert_dataset.py --converter data/converters/support_tickets.py:convert --name support_tickets
```

`convert_dataset.py` runs your converter the same way as the built-in ones. It passes the backbone's tokenizer
(`--backbone`, default `e4b`, which suits both backbones because they share one tokenizer), checks every row against the
registry, writes `data/processed/support_tickets.jsonl` and prints the number of rows in each split with a hash of each
split's ids. `--raw DIR` reads the raw data from a folder other than `data/raw/<name>`.

Guidelines:

- **Truncate with the shared preprocessor.** `truncate_state` keeps the first 70% and last 30% of the token budget
  around a marker. The teacher labeller uses the same function, so teacher and student (the model being trained) see
  identical text.
- **Choose the family key carefully.** If several rows share a source document, conversation or prompt, key the split
  on that, not on the row.
- **Write questions the model can answer one leaf at a time.** Every option is judged alone, without seeing the others
  ([chapter 5](05-writing-good-questions.md)).
- **Return the rows instead of writing them.** `convert_dataset.py` checks and writes them, so a converter only builds
  `Record`s.
- **Give `gold_calib` enough items.** Temperatures are fitted on `gold_calib` and need about 100 questions of each type
  (chapter 10). The shares above put 12% of items there, which reaches 100 only at about 830 items. A dataset of 200
  items gets 24. For a smaller dataset, either raise the `gold_calib` share or accept a temperature fitted on fewer
  questions, which is less certain. A share of 25% reaches 100 at 400 items. In `_split` that means, for example,
  `train` below 0.5, `val` below 0.6 and `gold_calib` below 0.85, which takes the extra from `train`. Calibration prints
  a warning when a question type has fewer than 100 questions to fit on.

The public seed datasets are fetched and converted with two commands. `fetch_dataset.py` downloads the files pinned in
`configs/datasets.yaml` into `data/raw/<name>/`, with a sha256 manifest. `convert_dataset.py` runs the registered
converter and checks every row against the registry.

```bash
uv run python scripts/fetch_dataset.py banking77
uv run python scripts/convert_dataset.py banking77
```

## 8.9 Inspecting data

```bash
uv run python scripts/inspect_data.py data/processed/support_tickets.jsonl
```

This prints rows per split, questions per type, a Choice option-count histogram by calibration bucket (the
option-count groups 2, 3–5, 6–15, 16–63 and 64–255 that get their own temperature, section 6.5), Score level
counts, the spread of Noul targets, whether each question has a human target, a teacher target or none, media counts,
and the truncation rate. Rows are loaded through the licence check, so a file that disagrees with the registry fails
here first.

## 8.10 Multiplying data: schema explosion

One labelled item can teach many question schemas (versions of a question with a different option set, or recast as
Nouls). `scripts/explode.py` derives new rows from a Choice question without a teacher. It keeps the state and the
human label and changes only the question. The general head was trained on such rows so that it answers option sets
of every size, not only a dataset's full list, and recognises when no listed option fits (chapter 13, section 13.3).

```bash
uv run python scripts/explode.py data/processed/banking77.jsonl --question intent
```

For each row it writes:

- **Subsets** (`--subsets`, default 2): the same question with fewer options. A subset keeps every option that has
  some of the target probability (target mass) and adds a random number of the other options (distractors), up to
  `--max-k` (default 60). The target is rescaled to sum to 1 over the kept options.
- **Drop-gold plus `other`:** the gold option (the labelled answer) and the two options whose descriptions share the
  most words with it (its most confusable options) are removed, and a self-contained `other` option is added that
  names what the kept options cover, so the answer becomes `other`. One-hot targets only.
- **Noul splits** (`--nouls`, default 2): "is the answer X?" Nouls, about 40% of them positive.

The output, `data/processed/<name>_explode.jsonl`, keeps every row's split, tier and family, and records
`lineage.parent_id` and `lineage.multiplier`. Gold rows produce gold-only variants for evaluation, and training mixes
take `train` rows only.

Next: [9. Fine-tuning for your task](09-fine-tuning-for-your-task.md)
