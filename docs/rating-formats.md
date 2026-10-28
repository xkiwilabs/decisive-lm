# Rating formats: hv-items/1 and hv-ratings/1

Human labels with several raters per item are what the model is trained and judged on: the share of raters choosing
each answer, and how much they agree (guide chapters [9](guide/09-fine-tuning-for-your-task.md) and
[11](guide/11-evaluation.md)). To collect them, records are exported as **items** for a rating app, and the app's
**ratings** are imported back as human targets. This page describes the two line formats exchanged with the app, as
written and read by `decisivelm/data/rating.py`, `scripts/export_for_rating.py` and `scripts/import_ratings.py`. Any
rating app that reads and writes them can be used.

Both files are JSON Lines: one JSON object per line.

## Items out: hv-items/1

```bash
uv run python scripts/export_for_rating.py --data data/processed/my_task.jsonl --split gold_test --n 300 \
    --task-set my_task_pilot --out data/rating/my_task_pilot.items.jsonl
```

One line per record:

| Field | Contents |
|---|---|
| `format` | `"hv-items/1"` |
| `item_id` | The record's id |
| `task_set` | The name given with `--task-set` |
| `source` | `{"dataset", "id", "family"}` from the record's lineage |
| `license`, `tier` | The record's licence and licence tier ([chapter 14](guide/14-datasets.md)) |
| `data_class` | `public` (downloadable from a pinned public source), `synthetic` (written by the teacher) or `private` |
| `language` | The record's language |
| `state` | The state, exactly as the model sees it |
| `questions` | `{question id: question}` in the request format of [chapter 4](guide/04-requests-and-responses.md), only those named with `--questions` if given |
| `role` | `rate`, `training` or `check` (below) |
| `content_warnings` | The kinds of content raters should be warned about (`--content-warning`) |
| `reference` | Training and check items only: `{question id: answer}` |
| `reference_source` | Training and check items only: `"human_consensus"` |
| `meta_questions` | Data-check items only (`--data-check`): the ids of the meta-questions added to `questions` |

An item never carries a target, a teacher's answer or a model's answer. A rater should judge the item without seeing
what anyone else said.

**Roles.**

- `rate`: an ordinary item. Its answers become human targets.
- `training`: shown to new raters with the reference answer, to learn the task. Its answers are never counted.
- `check`: an item with an unambiguous answer, used to check that raters understood the task. By default only records
  on which every human rater agreed are exported as check items (`--check-min-share 1.0`, the share of the most
  chosen answer on every question).

A reference is always the human consensus: the most chosen option or level, or yes when at least half the raters said
yes. It is never a teacher's or a model's answer, and a record without a human target cannot be a training or check
item.

**Data-check items.** For rows written by a teacher, a data-check project asks raters about the item as well as for
their answer. `--data-check` adds up to three yes/no meta-questions after the item's own questions:

| Id | Statement |
|---|---|
| `meta_answerable` | Each question above can be answered from the text |
| `meta_distinct` | In each question above, the options or levels are clearly different from one another (left out when every question is yes/no) |
| `meta_realistic` | The text reads like something a real person would write |

These ids are reserved: a record whose own question ids clash with them is refused.

**Newer fields.** `reference_source` and `meta_questions` were added after the first version of the format. They are
written by default; `--no-a1-fields` leaves them out for an app that predates them (and refuses `--data-check`).

**Uploading.** An item's `task_set` must be the name of the task set it is uploaded into. Uploading the same file again
changes nothing, but an item id that comes back with different content is refused, so export a new question subset
of the same rows under a new task set.

**Private data.** Records whose source is not downloadable from a pinned public source are refused unless
`--allow-private` is given. A rating app should refuse them too unless its deployment allows private data. Items with
images or audio are not supported yet.

## Ratings back: hv-ratings/1

```bash
uv run python scripts/import_ratings.py --ratings data/rating/my_task_pilot.ratings.jsonl \
    --data data/processed/my_task.jsonl --out data/processed/my_task_rated.jsonl [--min-raters 3]
```

One line per item:

```json
{"format": "hv-ratings/1", "item_id": "r0", "meta_questions": [],
 "questions": {"team": {"type": "choice", "responses": [
     {"rater": "p1", "value": "billing", "kind": "first"},
     {"rater": "p2", "value": "technical", "kind": "first"},
     {"rater": "p1", "value": "billing", "kind": "repeat"},
     {"rater": "p3", "value": null, "kind": "first"},
     {"rater": "p4", "value": "billing", "kind": "first", "over_quota": true},
     {"rater": "p5", "value": "technical", "kind": "first", "excluded": true}]}}}
```

| Field | Contents |
|---|---|
| `format` | `"hv-ratings/1"` |
| `item_id` | The item's id, which is the record's id |
| `meta_questions` | Echoed from the item, if it had any |
| `questions` | `{question id: {"type", "responses"}}`; `type` must match the record's question type |

Each response:

| Field | Meaning |
|---|---|
| `rater` | A pseudonymous rater id |
| `value` | An option name (Choice), a level index (Score), `true` or `false` (Noul), or `null` when the rater could not judge the item |
| `kind` | `first` (the default) for a rater's first answer, `repeat` when the rater saw the item again |
| `over_quota` | Optional: a first answer that arrived after the item already had its planned number of raters |
| `excluded` | Optional: the app's administrator excluded this rater's answers, with a reason |

Other fields (timings, export dates) are ignored.

**How responses become targets.** Each question's human target is the share of counted answers on each option
(Choice), level (Score) or "yes" (Noul), with the number of raters. An answer is counted when it is a rater's first
answer, has a value, and is not excluded.

- **Repeats** are never counted. They measure each rater's consistency, which the import prints per rater.
- **Over-quota answers** are counted like any other first answer: a late answer is still an independent judgement.
  The import prints how many there were.
- An answer that is not an option of the question is an error.
- A question with fewer than `--min-raters` counted answers keeps its old target. Teacher targets are kept beside the
  new human ones. Only rated records are written.

The import also prints the mean pair agreement per question: the share of rater pairs that gave the same answer.

**Meta-questions** judge the item, not its content, and never become targets. The import prints the mean yes share of
each one and how many items were flagged, an item being flagged when any meta-question has a yes share below .5.
`--meta-out PATH` writes one line per data-check item:

```json
{"item_id": "r0", "meta": {"meta_answerable": [1.0, 3], "meta_realistic": [0.333, 3]}, "flagged": true}
```

Each value is the yes share and the number of counted answers.
