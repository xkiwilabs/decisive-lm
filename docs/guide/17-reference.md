# 17. Reference

This chapter lists every script, configuration file, package module and test group in the repository, with one line
on what each does. Use it to find the right tool; the chapters linked from each section explain them in full. Every
script prints its full usage with `--help`, and its docstring at the top of the file gives worked examples.

## 17.1 Scripts

Run every script from the repository root as `uv run python scripts/<name>.py …`.

**Data**

| Script | What it does | Key flags |
|---|---|---|
| `fetch_dataset.py` | Downloads a registered dataset's pinned files into `data/raw/<name>/` with a SHA-256 `MANIFEST.json` (14.5) | dataset names |
| `convert_dataset.py` | Converts fetched public datasets into canonical rows in `data/processed/<name>.jsonl`, checked against the registry; leaves out suite exclusions (14.5) | dataset names, `--backbone` (tokenizer for truncation) |
| `inspect_data.py` | Summarises canonical rows: per-type counts, option counts, Score levels, Noul targets, media, truncation | data files |
| `explode.py` | Schema explosion of one Choice question: subsets, drop-gold plus `other`, Noul splits; writes `<name>_explode.jsonl` (13.3) | data file, `--question`, `--subsets`, `--max-k`, `--nouls`, `--seed` |
| `synth_generate.py` | Teacher-written question sets and states, and fan-out on real states, filtered and resumable (13.4) | `--name`, `--spec-version`, `--sets`, `--states`, `--fanout PATH:N`, `--val-share`, `--workers`, `--stop-at`, `--seed`, `--teacher`, `--build-only` |
| `overlap_audit.py` | Word 8-gram overlap between the frozen suite and every trainable source, both directions (15.2) | `--share`, `--out`, `--drop` |
| `build_mix.py` | Builds a training mix and its manifest, enforcing tiers and holdouts (13.6) | `--name`, `--adapter {general,user}`, `--tiers`, `--train PATH[:N]`, `--eval PATH[:N]`, `--zero-shot PATH[:N]`, `--cap MULTIPLIER=SHARE`, `--seed`, `--out-dir` |

**Teacher and baselines**

| Script | What it does | Key flags |
|---|---|---|
| `label_teacher.py` | Labels rows with a teacher on a vLLM server, resumably, then calibrates on `gold_calib` and writes predictions (12.7); `--local` runs the model on this machine and labels image and audio rows too (the media P0) | `--teacher`, `--data PATH[:N]`, `--splits`, `--chunk`, `--groups-in-parallel`, `--seed`, `--local`, `--media-root` |
| `teacher_targets.py` | Replaces train-row targets with calibrated teacher targets (12.9) | `--data`, `--teacher`, `--splits`, `--calib`, `--balanced`, `--noul-bias`, `--set`, `--blend TEACHER=W`, `--out` |
| `run_p0.py` | P0 baseline on a local backbone: prompt per question, read label log-probabilities, fit one temperature (12.8) | `--data`, `--backbone`, `--question`, `--splits`, `--chunk` |
| `baseline_tfidf.py` | TF-IDF plus logistic-regression baseline for every question, on soft human targets (needs `uv run --with scikit-learn`) | `--data`, `--grid` |

**Human ratings** (formats in [docs/rating-formats.md](../rating-formats.md))

| Script | What it does | Key flags |
|---|---|---|
| `export_for_rating.py` | Writes records as hv-items/1 items for a rating app: the state and questions only, never an answer | `--data`, `--split`, `--n`, `--seed`, `--questions`, `--task-set`, `--role {rate,training,check}`, `--check-min-share`, `--data-check`, `--no-a1-fields`, `--content-warning`, `--allow-private`, `--out` |
| `import_ratings.py` | Turns hv-ratings/1 ratings into human targets (shares of first answers), and reports pair agreement, rater consistency and data-check meta-questions | `--ratings`, `--data`, `--out`, `--min-raters`, `--meta-out` |

**Training and calibration**

| Script | What it does | Key flags |
|---|---|---|
| `cache_features.py` | Caches the frozen backbone's query-token states once per row, for head-only training (P1); resumable | `--data`, `--out`, `--backbone`, `--save-every`, `--dtype` |
| `train_p1.py` | Trains the scorer head only on cached features (P1), calibrates, writes a bundle | `--features`, `--data`, `--seed`, `--lr`, `--batch`, `--patience`, `--standardize`, `--name` |
| `train_p3.py` | Trains a LoRA adapter with query rows and head (a user adapter, or the general head with `--adapter general`), calibrates, writes a bundle and predictions (chapters 9 and 13) | `--data`, `--backbone`, `--adapter`, `--rank`, `--alpha`, `--epochs`, `--rows-per-step`, `--max-tokens`, `--lr-lora`, `--lr-head`, `--warmup-frac`, `--eval-every`, `--patience`, `--select`, `--predict-best`, `--limit-val`, `--standardize-rows`, `--init-head`, `--init-bundle` (fine-tune from a trained bundle such as the general head, 9.6), `--n-train`, `--no-checkpointing`, `--dtype`, `--name` |
| `recalibrate.py` | Refits a trained run's calibration from its saved predictions, without the model; with `--new-task`, scales the Choice and Score temperatures on another task's `gold_calib` and writes a new bundle with that calibration (a development tool; the released head ships without such a scale, 10.6) | `--name`, `--suffix`, `--new-task STEM`, `--replace` |
| `task_calibrate.py` | Task calibration of any method's predictions on a dataset: a scale and an offset per option or level for each question, fitted on its `gold_calib` rows; writes `<file>_task.jsonl` (10.6) | `--data`, `--predictions PATH …` |

**Inference and checks**

| Script | What it does | Key flags |
|---|---|---|
| `predict_bundle.py` | Runs a bundle on data files with its own calibration; writes `data/predictions/<bundle>__<stem>.jsonl` (7) | `--bundle`, `--data`, `--splits`, `--max-tokens` |
| `check_bundle.py` | Loads a bundle on its own backbone, runs it through `DecisiveLM.system_one`, compares every answer with the predictions training wrote, and checks conformance | `--bundle`, `--data`, `--n`, `--predictions` |
| `check_invariance.py` | How far a bundle's answers move, per question type, when only the packing changes: options reversed (Choice), another request packed first, one more question (11) | `--bundle`, `--backbone`, `--data`, `--split`, `--n`, `--dtype`, `--out` |
| `m0_real.py` | Backbone checks on real Gemma 4 weights: config facts, reserved token ids, mask equivalence, bf16 noise floor, memory and latency | `--backbone`, `--device`, `--fp32-device`, `--skip-latency` |

**Evaluation and release**

| Script | What it does | Key flags |
|---|---|---|
| `eval_dataset.py` | Per-type metrics on any dataset's `gold_test`, with bootstrap CIs and paired differences between methods (11) | `--data`, `--predictions NAME=PATH …`, `--reference`, `--split`, `--n-boot`, `--out` |
| `compare_coding.py` | Scores methods on a dataset coded by several human coders: κ against each coder, accuracy against consensus, and review readiness (11) | `--data`, `--codes`, `--predictions`, `--baselines`, `--reference`, `--question`, `--review`, `--n-boot`, `--out` |
| `score_suite.py` | Scores a release candidate against release criteria 1–4 on the frozen suite, once (15.4) | `--candidate`, `--calib-data`, `--sets`, `--n-boot`, `--seed`, `--out` |
| `score_c8.py` | Release criterion 8: fine-tunes from the general head against fine-tunes from base, per task and training size, on accuracy and review readiness (9.6) | `--general`, `--n-boot`, `--seed`, `--out` |

## 17.2 Configuration

All configuration is YAML in `configs/`. No paths are hard-coded in the code.

| File | Contents |
|---|---|
| `model.yaml` | Backbone profiles (`e4b`, `26b-a4b`): model id and pinned revision, licence and origin, hidden size, layer pattern, sliding window, modalities, image token budget, LoRA target regex; the reserved query-token ids |
| `teachers.yaml` | Teachers on vLLM servers: endpoint, model and revision, tokenizer profile, licence and origin, vLLM version, log-probability settings, request schedule (12.1) |
| `datasets.yaml` | Every data source: description, source, SPDX licence, tier, optional override reason, and pinned `fetch` entries (14) |
| `holdout.yaml` | Enforced holdouts: dataset, the adapters it is barred from (`general`, `user`), optional `source_ids`, `family_ids` or `languages` selectors, and a reason |
| `release_seeds.yaml` | Release criterion 4: each seed's data file and reference predictions, and the tolerance |
| `suite_exclusions.txt` | Ids of frozen-suite rows left out because their text overlaps trainable data |
| `local.example.yaml` | Template for the machine-local, untracked `configs/local.yaml`: `hf_cache` (where Hugging Face files are read and written) and `offline` (load only from that cache) |

## 17.3 Package modules

The `decisivelm` package (chapter 6 explains how the pieces fit together):

| Module | What it holds |
|---|---|
| `api_types` | Request and response models, validation and linting (`SystemOneRequest`, `ChoiceQuestion`, `ScoreQuestion`, `NoulQuestion`, `lint_request`) |
| `render` | Block text templates for question segments and leaves; `RENDER_VERSION` |
| `preprocess` | State truncation shared by the teacher and the model (head and tail, with a marker) |
| `media` | Images in the shared prefix: loading, encoding and placeholder tokens |
| `tree` | Request tree to packed token ids, logical positions, and the global and sliding-window attention masks |
| `model` | Backbone loading from a profile (`load_backbone`, `load_tokenizer`, `load_image_processor`); fp32 options for the MoE backbone |
| `lora` | LoRA on decoder projections and trainable query-token rows |
| `head` | The scorer head, calibration lookup (`Calibration`) and readout |
| `confidence` | Confidence statistics (`peak`, `entropy`, `margin`, `ordinal`) |
| `losses` | Training losses per question type |
| `train` | The adapter training loop: packing, micro-batches, validation criteria |
| `calibrate` | Fitting temperatures and the Noul bias, and the optional new-task scales; writing predictions |
| `infer` | `DecisiveLM` and `system_one`: one request in, one response out |
| `bundle` | `load_bundle` (also `DecisiveLM.load`): a bundle on the backbone it was trained on; `attach_bundle`: a bundle on a loaded backbone, checked against its revision |
| `invariance` | Probability-level independence and order checks |
| `eval` | Metrics shared by every report: ECE, Brier, AUROC, pair agreement, review curves |
| `data.schema` | The canonical record (chapter 8) |
| `data.dataset` | Loading rows, with the licence and tier checks against the registry |
| `data.mix` | Licence tiers for mixes and the holdout check (`check_mix`) |
| `data.cleanroom` | Whitelist, canary search and the answer-shape scanner (16.1) |
| `data.rating` | Items for a rating app and ratings back into human targets (hv-items/1, hv-ratings/1), data-check meta-questions |
| `data.converters` | One converter per source |
| `synth.label` | Label log-probability method: prompts, orderings, lettered and numbered labels, the local labeller |
| `synth.teacher` | The vLLM teacher client, request schedule, and Score and Noul labelling |
| `synth.generate` | Seeded specs and prompts for teacher-written question sets and states, and their filters |
| `synth.multipliers.schema_explosion` | Subsets, drop-gold plus `other`, Noul splits |
| `synth/prompts/` | Versioned prompt templates for labelling and generation |

## 17.4 Tests

```bash
uv run pytest                                   # the whole suite: CPU, tiny random Gemma 4 configs, seconds
uv run pytest tests/test_tree.py                # one file
uv run pytest tests/test_tree.py::test_sliding_window_uses_logical_not_physical_distance   # one test
```

The suite runs without downloaded weights. Model tests use tiny random Gemma 4 configurations (`tests/tiny_gemma.py`),
dense and mixture-of-experts, so masks, independence and conformance are checked in seconds. A `real_model` marker is
reserved for tests that need downloaded weights, and it is deselected by default. Checks on the real backbones run
through `scripts/m0_real.py`.

| Group | Files |
|---|---|
| Masks and packing | `test_tree.py`, `test_mask_equivalence.py`, `test_media.py`, `test_moe_precision.py` |
| Request contract | `test_api_types.py`, `test_render.py`, `test_preprocess.py`, `test_confidence.py`, `test_system_one.py`, `test_invariance.py` |
| Model and training | `test_lora.py`, `test_scorer.py`, `test_losses.py`, `test_train.py`, `test_bundle.py`, `test_calibrate.py` |
| Data | `test_data_registry.py` (licences, tiers, holdouts), `test_cleanroom.py` (whitelist, canary, scanner), the converter tests (including `test_multinli.py`), `test_schema_explosion.py`, `test_suite_converters.py`, `test_rating.py` (rating formats) |
| Teacher and generation | `test_label.py`, `test_generate.py` |
| Evaluation and release | `test_eval.py`, `test_eval_dataset.py`, `test_score_suite.py` |
| Local settings | `test_hub.py` (`configs/local.yaml` handling) |

## 17.5 Environment variables

| Variable | Use |
|---|---|
| `HF_HUB_CACHE` | Where Hugging Face weights and tokenizers are stored. Set it on a machine whose default cache belongs to another application, or use `hf_cache` in `configs/local.yaml` |
| `HF_HUB_OFFLINE=1` | Load only what is already downloaded; a missing pinned revision fails instead of downloading |
| `CUDA_VISIBLE_DEVICES` | Choose the GPU(s) a script uses |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces memory fragmentation in long training runs with varying sequence lengths |
| `PYTHONUNBUFFERED=1` | Prints progress lines immediately, useful when a long job's output goes to a log file |

## 17.6 Repository layout

```
decisivelm/        the package (17.3)
scripts/           command-line tools (17.1)
configs/           YAML configuration (17.2)
tests/             the test suite (17.4)
docs/              this guide, the release criteria and the model card
results/           evaluation reports written by the scripts
bundles/           trained adapter bundles (not tracked)
data/              raw, processed, teacher, mixes and predictions (not tracked)
```

Back to the [README](../../README.md)
