# 16. Provenance, licences and responsible use

DecisiveLM is built to be used where provenance matters. Every model and every row of training data it uses can be
traced to a public source with a known licence. Two project rules are enforced in code: the clean-room rule (16.1),
by a field whitelist, a canary test and a scanner, and the model-origin rule (16.2), by a check that runs whenever a
backbone or teacher is loaded. This chapter states those rules, the licences that apply, how evaluation data is kept
clean, and what the model should and should not be used for.

## 16.1 The clean-room rule

DecisiveLM's request and response format follows the **public documentation** of TypeSafe's Jev System One API, so that
an application written for that API can switch to a local model with little code. The documented contract defines
the three question types, the field names and the published confidence formula for Choice.

DecisiveLM is an independent implementation and is not affiliated with TypeSafe.

The project is built clean-room:

- **No requests are ever sent to Jev or any TypeSafe service**, in any step: data generation, training, evaluation,
  benchmarking or debugging.
- **No Jev output is used anywhere.** This includes response logs from earlier use of the service.
- Only the public documentation of the API contract informs the design. The service is never probed to learn how it
  works.
- A user's own **inputs** (the states they sent and the questions they wrote) are theirs and may be reused. Any answers
  that came back with them are discarded, and the inputs are relabelled by an open teacher (Gemma 4 26B A4B, a
  larger model that labels rows which have no human answers, chapter 12).

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

As a project policy, **no model developed by a Chinese organisation, or derived from one, is used in any role**,
whether as backbone, teacher, ensemble member, embedding model, or speech or OCR model.

A fine-tune of such a model by another organisation still counts as derived. This keeps DecisiveLM usable by
organisations whose policies exclude such models.

Each backbone in `configs/model.yaml` and each teacher in `configs/teachers.yaml` records its developer and base-model
origin next to its licence. The code loads a backbone or teacher only if both are in `approved_origins` at the top of
`configs/model.yaml` (`decisivelm.model.check_origin`, tested in `tests/test_model_origin.py`), so a model of unchecked
origin fails at load time. Adding a model means filling in those fields and checking them against the rule; adding an
organisation to `approved_origins` means checking the organisation.

Datasets are governed by their licences (16.3), not by this rule. A public, human-rated dataset may contain text that
other models wrote. It is used as data, and its models are never run.

## 16.3 Licences

| Component | Licence |
|---|---|
| DecisiveLM code | Apache-2.0 (`LICENSE`) |
| Published adapter weights (the two general heads) | Apache-2.0 |
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
- any output of TypeSafe's Jev System One API (16.1).

**Attribution.** CC BY datasets require credit. Anything you publish that was derived from them (weights, generated
rows written about their states, results) should cite the dataset authors, link the source and licence, and say what
changed (14.8).

## 16.4 Data hygiene

Results only mean something if the test data was never seen in training. The project enforces this at several points:

- **Splits before variants.** Rows are assigned to `train`, `val`, `gold_calib`, `gold_test` or `holdout` at the level
  of the source item and the schema family (rows that share something a model could memorise, such as a base question
  and its rewordings, chapter 8.4), before any variant is derived. Every derived row inherits its parent's split,
  whether it comes from schema explosion (new questions built from a row's options, chapter 13.3) or fan-out (new
  questions the teacher writes about a seed dataset's state, chapter 13.4), so no variant of a test row can reach
  training.
- **Separate calibration and test sets.** `gold_calib` is used only to fit temperatures (the numbers that sharpen or
  soften a model's probabilities to match human labels, chapter 10) and `gold_test` only to report. They never mix,
  so no reported number comes from rows the calibration was fitted on.
- **Training on train only.** A training mix may contain only `train` rows. Public test splits are never trained on.
- **Enforced holdouts.** `configs/holdout.yaml` lists data barred from the general head, user adapters or both. The
  mix builder refuses a mix that overlaps it, and the test suite checks this.
- **Overlap filtering.** The generator drops any generated question or state that shares word n-grams (runs of
  consecutive words) with an evaluation or held-out row. `scripts/overlap_audit.py` audits the frozen suite against
  every trainable source in both directions.
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
general head alone, nor on any adapter, without labelled data from your own setting, an evaluation on it, and a
check of its errors across the groups affected. Examples are credit, insurance, employment and hiring, housing,
healthcare, education admissions and legal outcomes.

The general head was not trained on historical outcomes in these areas. Its teacher-written training data includes
hiring paperwork, but the questions on it are limited to factual yes/no statements about what a CV or cover letter says
(for example, whether it lists a qualification), and none asks whether to hire (chapter 13.4). It must not be used to
infer protected attributes such as age, gender, ethnicity or health from text, faces or voices.

**Known limits.**

- **Teacher blind spots.** The head learns partly from a teacher model, so it can share that teacher's blind spots.
- **Long, rule-heavy instructions.** A single forward pass can be weaker than a model that reasons step by step,
  especially on long codebooks with precedence rules. A task-specific fine-tune (chapter 9, section 9.6) closes much
  of that gap.
- **Isolated options.** A Choice option is judged without seeing its siblings, so an `other` option needs a
  self-contained description (chapter 5).
- **Option limit.** Choice is limited to 255 options.
- **Languages.** The training data is mostly English. On the held-out suite's 60-intent task in ten languages the
  head was not trained on, the E4B head's accuracy is 0.672 (P0 0.658; the 26B A4B head's is 0.680 against 0.731 for
  its P0), and the E4B head's calibration needs the per-task step (expected calibration error 0.094 zero-shot and
  0.018 after it, chapter 15). The same task is not scored in English, so how
  these languages compare with English is not measured.
- **Images and audio.** Of images, the general head was trained only on VizWiz-VQA's photo questions, and it has been
  checked zero-shot on one other image set (CIFAR-10H, model card). It was not trained on audio. Image and audio tasks
  need calibration or a fine-tune on your own labelled examples (chapter 10, section 10.6, and chapter 9, section 9.9),
  and audio needs the E4B backbone (the 26B A4B general head takes text and images only).

Check the model card for measured performance and failure modes before relying on it.

**Human review.** Calibrated confidence makes review cheaper, not unnecessary. Choose a review threshold on your own
human-labelled data (chapter 11), and keep checking a sample of confident answers too.

## 16.6 Privacy

DecisiveLM runs on your own hardware. The model, adapters and inference make no network calls once the weights are
downloaded. Set `HF_HUB_OFFLINE=1` to make sure of it. Requests, states and media stay on your machine when DecisiveLM
is called from Python. The HTTP server (section 4.10) receives them over the network from its callers, and with
`DECISIVELM_AUTH=litellm` it sends each caller's key, but not the request, to the LiteLLM proxy you name.

When labelling with a teacher, rows are sent to the vLLM server you configure, which can also be your own. Nothing in
DecisiveLM uploads adapters, so adapters trained on private data stay where you save them. Keep private datasets in
tier X so they cannot reach a release mix.

Next: [17. Reference](17-reference.md)
