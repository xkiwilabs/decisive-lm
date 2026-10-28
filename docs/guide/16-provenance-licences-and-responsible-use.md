# 16. Provenance, licences and responsible use

DecisiveLM is built to be used where provenance matters. Every model and every row of training data it uses can be
traced to a public source with a known licence, and two project rules are enforced in code, not left to good
intentions. This chapter states those rules, the licences that apply, how evaluation data is kept clean, and what the
model should and should not be used for.

## 16.1 The clean-room rule

DecisiveLM's request and response format follows the **public documentation** of TypeSafe's Jev System One API, so that
an application written for that API can switch to a local model with little code. The documented contract defines:

- the three question types;
- the field names;
- the published confidence formula for Choice.

DecisiveLM is an independent implementation and is not affiliated with TypeSafe.

The project is built clean-room:

- **No requests are ever sent to Jev or any TypeSafe service**, in any step: data generation, training, evaluation,
  benchmarking or debugging.
- **No Jev output is used anywhere.** This includes response logs from earlier use of the service.
- Only the public documentation of the API contract informs the design. The service is never probed to learn how it
  works.
- A user's own **inputs** (the states they sent and the questions they wrote) are theirs and may be reused. Any answers
  that came back with them are discarded, and the inputs are relabelled by an open teacher (chapter 12).

Comparisons with published results of other systems cite their published aggregate figures with the source. Those
figures are never reproduced by calling the service.

**How it is enforced** (`decisivelm/data/cleanroom.py`, `tests/test_cleanroom.py`):

| Mechanism | What it does |
|---|---|
| Whitelist | Importing a request log keeps only `state`, `questions` and `media` (`whitelist_request`). Every answer field is dropped. Dataset converters likewise copy only the fields they need. |
| Canary | A test fixture holds a request log with unique marker strings in its answers. The test suite fails if any of them appears anywhere under `data/processed/`. |
| Scanner | `scan_state` flags states that contain answer-shaped content (`probabilities`, `noul`, `confidence` or `legend` objects), for example an earlier answer pasted into a later request. A flagged row needs a human decision before it is used. |

## 16.2 Model-origin rule

Every model in the pipeline comes from Google's **Gemma 4** family:

| Role | Model |
|---|---|
| Backbone (the student) | Gemma 4 E4B (default) or Gemma 4 26B A4B |
| Teacher | Gemma 4 26B A4B |
| P0 baseline | The student's own backbone, prompted |

As a project policy, **no model developed by a Chinese organisation, or derived from one, is used in any role**:

- backbone;
- teacher;
- ensemble member;
- embedding model;
- speech or OCR model.

A fine-tune of such a model by another organisation still counts as derived. This keeps DecisiveLM usable by
organisations whose policies exclude such models.

Each backbone in `configs/model.yaml` and each teacher in `configs/teachers.yaml` records its developer and base-model
origin next to its licence. Adding a model means filling in those fields and checking them against the rule.

Datasets are governed by their licences (16.3), not by this rule. A public, human-rated dataset may contain text that
other models wrote. It is used as data, and its models are never run.

## 16.3 Licences

| Component | Licence |
|---|---|
| DecisiveLM code | Apache-2.0 (`LICENSE`) |
| Published adapter weights (the general head) | Apache-2.0 |
| Gemma 4 weights | Released by Google under Apache-2.0, as recorded in `configs/model.yaml`; check the model card of the revision you use |
| Teacher-written rows | Apache-2.0: Gemma 4 places no terms on its outputs |
| Public datasets | Each its own (chapter 14): CC BY 4.0, CC BY 3.0, CC0 1.0 and Apache-2.0 for training data; evaluation-only sets as listed |

**Licence tiers** decide where a row may be used (14.1):

- **R**: permissive licences. The only tier that reaches the published general head.
- **S**: share-alike licences. User and private adapters only.
- **X**: restricted or unclear licences. Private experiments and evaluation only.

A row derived from others inherits the most restrictive tier of its inputs. The mix builder refuses a release mix
with any non-R row (`decisivelm/data/mix.py`).

The release tier excludes:

- data whose labels came from proprietary models;
- sources whose terms forbid training language models;
- any output of the service whose API format DecisiveLM follows.

**Attribution.** CC BY datasets require credit. Anything you publish that was derived from them (weights, generated
rows written about their states, results) should cite the dataset authors, link the source and licence, and say what
changed (14.8).

## 16.4 Data hygiene

Results only mean something if the test data was never seen in training. The project enforces this at several points:

- **Splits before variants.** Rows are assigned to `train`, `val`, `gold_calib`, `gold_test` or `holdout` at the
  level of the source item and the schema family, before any variant is derived. Every derived row (schema explosion,
  fan-out) inherits its parent's split, so no variant of a test row can reach training.
- **Separate calibration and test sets.** `gold_calib` is used only to fit temperatures and `gold_test` only to
  report. They never mix.
- **Training on train only.** A training mix may contain only `train` rows. Public test splits are never trained on.
- **Enforced holdouts.** `configs/holdout.yaml` lists data barred from the general head, user adapters or both. The
  mix builder refuses a mix that overlaps it, and the test suite checks this.
- **Overlap filtering.** The generator drops any generated question or state that shares word n-grams with an
  evaluation or held-out row. `scripts/overlap_audit.py` audits the frozen suite against every trainable source in
  both directions.
- **Human gates.** Every release gate uses human labels only, never teacher labels (chapter 15).

## 16.5 Responsible use

**Intended uses.** DecisiveLM is for constrained judgements about text and images, where an answer is one of a fixed
set of options, a level on a scale, or a yes/no. Examples:

- routing support tickets;
- triaging moderation queues;
- coding free-text survey answers;
- tagging documents;
- checking forms for required content.

Its confidence is designed to say which answers a person should check. The intended workflow sends the least
confident answers to human review.

**Out of scope without your own data and checks.** Decisions that significantly affect people should not rest on the
general head alone, nor on any adapter, without labelled data from your own setting, an evaluation on it and a
fairness check. Examples are credit, insurance, employment and hiring, housing, healthcare, education admissions and
legal outcomes.

The general head was not trained on historical outcomes in these areas. Its generated hiring questions are limited to
factual questions about what a document says. It must not be used to infer protected attributes such as age, gender,
ethnicity or health from text, faces or voices.

**Known limits.**

- **Teacher blind spots.** The head learns partly from a teacher model, so it can share that teacher's blind spots.
- **Long, rule-heavy instructions.** A single forward pass can be weaker than a model that reasons step by step,
  especially on long codebooks with precedence rules. A task-specific fine-tune (chapter 9) closes much of that gap.
- **Isolated options.** A Choice option is judged without seeing its siblings, so an `other` option needs a
  self-contained description (chapter 5).
- **Option limit.** Choice is limited to 255 options.
- **Languages.** Languages other than English are weaker than English.
- **Audio.** Audio input is not supported by the inference path.

Check the model card for measured performance and failure modes before relying on it.

**Human review.** Calibrated confidence makes review cheaper, not unnecessary. Choose a review threshold on your own
human-labelled data (chapter 11), and keep checking a sample of confident answers too.

## 16.6 Privacy

DecisiveLM runs on your own hardware. The model, adapters and inference make no network calls once the weights are
downloaded; set `HF_HUB_OFFLINE=1` to make sure of it. Requests, states and media never leave your machine.

When labelling with a teacher, rows are sent to the vLLM server you configure, which can also be your own. Adapters
trained on private data stay wherever you keep them. Keep private datasets in tier X so they cannot reach a release
mix.

Next: [17. Reference](17-reference.md)
