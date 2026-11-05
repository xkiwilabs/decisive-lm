# 17. Reference

This chapter lists the scripts, configuration files, package modules and test groups in the repository, with one line on
what each does. Use it to find the right tool. The chapters linked from each section explain them in full. Every script
prints its full usage with `--help`, and its docstring at the top of the file gives worked examples.

## 17.1 Scripts

Run every script from the repository root as `uv run python scripts/<name>.py …`. The flags listed are the ones you will
usually set, and `--help` lists all of them.

**Data**

| Script | What it does | Key flags |
|---|---|---|
| `fetch_dataset.py` | Downloads a registered dataset's pinned files into `data/raw/<name>/` with a SHA-256 `MANIFEST.json` (14.5) | dataset names, `--jobs` (files downloaded at once, default 1) |
| `convert_dataset.py` | Converts fetched public datasets into canonical rows in `data/processed/<name>.jsonl`, checked against the registry; leaves out suite exclusions (14.5). Runs your own converter the same way (8.8) | dataset names, `--converter FILE.py:FUNCTION` with `--name` (your source, registered in `configs/datasets.local.yaml`) and `--raw` (its raw-data directory), `--backbone` (tokenizer for truncation) |
| `inspect_data.py` | Summarises canonical rows: per-type counts, option counts, Score levels, Noul targets, media, truncation | data files |
| `explode.py` | Schema explosion of one Choice question: subsets, drop-gold plus `other`, Noul splits; writes `<name>_explode.jsonl` (13.3) | data file, `--question`, `--subsets`, `--max-k`, `--nouls`, `--seed` |
| `synth_generate.py` | Teacher-written question sets and states, and fan-out on real states, filtered and resumable (13.4) | `--name`, `--spec-version`, `--sets`, `--states`, `--fanout PATH:N`, `--val-share`, `--workers`, `--stop-at`, `--seed`, `--teacher`, `--build-only` |
| `overlap_audit.py` | Word 8-gram overlap between the frozen suite and every trainable source, both directions (15.2) | `--share`, `--out`, `--drop` |
| `build_mix.py` | Builds a training mix and its manifest, enforcing tiers and holdouts (13.6) | `--name`, `--adapter {general,user}`, `--tiers`, `--train PATH[:N]`, `--eval PATH[:N]`, `--zero-shot PATH[:N]`, `--cap MULTIPLIER=SHARE`, `--seed`, `--out-dir` |
| `drop_teacher_score.py` | A copy of a mix with every teacher-labelled Score question removed from its `train` and `val` rows, to test whether the teacher's Score targets cause a head's Score failure; writes `data/mixes/<name>.jsonl` and its manifest | `--mix`, `--name` |

**Teacher and baselines**

| Script | What it does | Key flags |
|---|---|---|
| `label_teacher.py` | Labels rows with a teacher on a vLLM server, resumably, then calibrates on `gold_calib` and writes predictions (12.7). With `--local` it runs the model on this machine and labels image and audio rows too (the media P0) | `--teacher`, `--data PATH[:N]`, `--splits`, `--chunk`, `--groups-in-parallel`, `--seed`, `--local`, `--media-root` |
| `teacher_targets.py` | Replaces train-row targets with calibrated teacher targets (12.9) | `--data`, `--teacher`, `--splits`, `--calib`, `--balanced`, `--noul-bias`, `--set`, `--blend TEACHER=W`, `--out` |
| `run_p0.py` | P0 baseline on a local backbone: prompt per question, read label log-probabilities, fit one temperature (12.8) | `--data`, `--backbone`, `--question`, `--splits`, `--chunk` |
| `baseline_tfidf.py` | TF-IDF plus logistic-regression baseline for every question, on soft human targets (needs `uv run --with scikit-learn`) | `--data`, `--grid` |

**Human ratings** (formats in [docs/rating-formats.md](../rating-formats.md))

| Script | What it does | Key flags |
|---|---|---|
| `export_for_rating.py` | Writes records as hv-items/1 items for a rating app: the state and questions only, never an answer | `--data`, `--split`, `--n`, `--seed`, `--questions`, `--task-set`, `--role {rate,training,check}`, `--check-min-share`, `--data-check`, `--a1-fields` / `--no-a1-fields`, `--content-warning`, `--allow-private`, `--out` |
| `import_ratings.py` | Turns hv-ratings/1 ratings into human targets (shares of first answers), and reports pair agreement, rater consistency and data-check meta-questions | `--ratings`, `--data`, `--manifest`, `--out`, `--min-raters`, `--meta-out` |
| `build_dev_v1.py` | Items for a development rating project: one question each from validation rows no head was trained on (intents, toxicity, sentiment, helpfulness, preference), split into `gold_calib` and `gold_test`, leaving out prompts already in other rating files | `--processed`, `--exclude-items`, `--seed`, `--out` |
| `build_two_sided.py` | Items for a two-sided rating project: args.me arguments and HelpSteer2 responses with 5-level Scores whose middle is neutral, several raters per item | `--args-me`, `--helpsteer2`, `--exclude-items`, `--seed`, `--out` |

**Training and calibration**

| Script | What it does | Key flags |
|---|---|---|
| `cache_features.py` | Caches the frozen backbone's query-token states once per row, for head-only training (P1); resumable | `--data`, `--out`, `--backbone`, `--save-every`, `--dtype`, `--fp32-router`, `--image-budget`, `--media-root` |
| `train_p1.py` | Trains the scorer head only on cached features (P1), calibrates, writes a bundle | `--features`, `--data`, `--seed`, `--lr`, `--batch`, `--patience`, `--standardize`, `--max-train`, `--name` |
| `train_p3.py` | Trains a LoRA adapter with query rows and head (a user adapter, or the general head with `--adapter general`), calibrates, writes a bundle and predictions (chapters 9 and 13) | `--data`, `--backbone`, `--adapter`, `--rank`, `--alpha`, `--epochs`, `--rows-per-step`, `--max-tokens`, `--lr-lora`, `--lr-head`, `--warmup-frac`, `--eval-every`, `--patience`, `--select`, `--predict-best`, `--limit-val`, `--standardize-rows`, `--init-head`, `--init-bundle` (fine-tune from a trained bundle such as the general head, 9.6), `--n-train`, `--limit-train`, `--seed`, `--dropout`, `--no-checkpointing`, `--dtype`, `--fp32-router`, `--image-budget` (image detail in soft tokens, 9.9), `--media-root`, `--name` |
| `recalibrate.py` | Refits a trained run's calibration from its saved predictions, without the model; with `--new-task`, scales the Choice and Score temperatures on another task's `gold_calib` and writes a new bundle with that calibration (a development tool, and neither released head ships with such a scale, 10.6) | `--name`, `--suffix`, `--new-task STEM`, `--replace` |
| `task_calibrate.py` | Task calibration of any method's predictions on a dataset: a scale and an offset per option or level for each question, fitted on its `gold_calib` rows; writes `<file>_task.jsonl` (10.6). With `--calib-rows N`, fits on a random N of those rows instead and writes `<file>_task_n<N>_s<seed>.jsonl` | `--data`, `--predictions PATH …`, `--calib-rows`, `--seed` |

**Inference and checks**

| Script | What it does | Key flags |
|---|---|---|
| `predict_bundle.py` | Runs a bundle on data files with its own calibration, in the compute type it was trained in (bf16 or float32), including image and audio rows; writes `data/predictions/<bundle>__<stem>.jsonl` (7) | `--bundle`, `--data`, `--splits`, `--max-tokens`, `--image-budget` (default: the bundle's), `--media-root` (where relative media paths start; default the repository) |
| `check_bundle.py` | Loads a bundle on its own backbone, runs it through `DecisiveLM.system_one`, compares every answer with the predictions training wrote, and checks conformance. Passes when every probability is within the tolerance and every changed answer was a near tie (its top two within twice the tolerance) | `--bundle`, `--data`, `--n`, `--predictions`, `--tolerance` (default 0.10 for a bf16 bundle, 0.001 for float32) |
| `check_invariance.py` | How far a bundle's answers move, per question type, when only the packing changes: options reversed (Choice), another request packed first, one more question (11) | `--bundle`, `--backbone`, `--data`, `--split`, `--n`, `--seed`, `--dtype`, `--fp32-router`, `--out` |
| `m0_real.py` | Backbone checks on real Gemma 4 weights: config facts, reserved token ids, mask equivalence, bf16 noise floor, memory and latency | `--backbone`, `--device`, `--fp32-device`, `--skip-latency` |

**Serving**

| Command | What it does | Settings |
|---|---|---|
| `uv run --extra serve python -m decisivelm.serve` | Serves one backbone and any number of same-shape bundles over HTTP: `GET /health`, `GET /v1/models`, `POST /v1/systemone` (4.10). Also built as a container from the `Dockerfile` | `DECISIVELM_*` environment variables (17.5) |

**Evaluation and release**

| Script | What it does | Key flags |
|---|---|---|
| `eval_dataset.py` | Per-type metrics on any dataset's `gold_test`, with bootstrap CIs and paired differences between methods (11) | `--data`, `--predictions NAME=PATH …`, `--reference`, `--split`, `--n-boot`, `--seed`, `--out` |
| `compare_coding.py` | Scores methods on a dataset coded by several human coders: κ against each coder, accuracy against consensus, and review readiness (11) | `--data`, `--codes`, `--predictions`, `--baselines`, `--reference`, `--question`, `--review`, `--n-boot`, `--seed`, `--out` |
| `split_half.py` | How well half of each item's raters foretell the other half's agreement, over many random splits: split-half ρ, its Spearman-Brown full-panel estimate, and half A's top share as a confidence. A comparison for ρ(top p, agreement), not a ceiling (11) | `--data`, `--split`, `--question`, `--splits`, `--seed` |
| `taskcal_report.py` | Task calibration of a bundle against P0 on rated development sets: the four rows (each with and without task calibration), the split-half figures, and fits on 100 and 300 `gold_calib` rows over 5 draws (10.6) | `--bundle`, `--data`, `--name`, `--out` |
| `media_seed_report.py` | The image and audio examples' 300-item runs over three seeds, general minus base per seed, and photo quality at more image detail (9.9) | `--general` (which head's runs: `general` = v3d, `generalv5c` = v5c), `--out` |
| `score_suite.py` | Scores a release candidate against release criteria 1–4 on the frozen suite, once (15.4) | `--candidate`, `--calib-data`, `--sets`, `--n-boot`, `--seed`, `--seeds-only`, `--out` |
| `score_c8.py` | Release criterion 8: fine-tunes from the general head against fine-tunes from base, per task and training size, on accuracy and review readiness (9.6) | `--general`, `--n-boot`, `--seed`, `--out` |

## 17.2 Configuration

All configuration is YAML in `configs/`. Machine-specific paths (the Hugging Face cache, private data) go in the
untracked `configs/local.yaml` and `configs/datasets.local.yaml`. Scripts take data and output paths as flags, with
defaults under the repository.

| File | Contents |
|---|---|
| `model.yaml` | Backbone profiles (`e4b`, `26b-a4b`): model id and pinned revision, licence and origin, hidden size, layer pattern, sliding window, modalities, image token budget, LoRA target regex; the reserved query-token ids; `approved_origins`, the developers a backbone or teacher may come from (16.2) |
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
| `media` | Images and audio in the shared prefix: loading, encoding to soft tokens, and placeholder tokens |
| `tree` | Request tree to packed token ids, logical positions, and the global and sliding-window attention masks |
| `model` | Backbone loading from a profile (`load_backbone`, `load_tokenizer`, `load_image_processor`, `load_feature_extractor` for audio, `vision_kwargs`); `check_origin`, which refuses a backbone or teacher whose origin is not approved; `hub_kwargs`, the Hugging Face cache and offline settings from `configs/local.yaml`; fp32 options for the MoE backbone |
| `lora` | LoRA on decoder projections and trainable query-token rows |
| `head` | The scorer head, calibration lookup (`Calibration`) and readout |
| `confidence` | Confidence statistics (`peak`, `entropy`, `margin`, `ordinal`) |
| `losses` | Training losses per question type |
| `train` | The adapter training loop: packing, micro-batches, validation criteria |
| `calibrate` | Fitting temperatures and the Noul bias, and the optional new-task scales; writing predictions |
| `infer` | `DecisiveLM` and `system_one`: one request in, one response out; `RequestTooLarge` when a packed request exceeds a token limit |
| `serve` | The HTTP server: bundles hot-swapped on one backbone, API-key or LiteLLM-gateway auth, a GPU memory cap, per-key usage logs (4.10) |
| `bundle` | `load_bundle` (also `DecisiveLM.load`): a bundle on the backbone it was trained on; `attach_bundle`: a bundle on a loaded backbone, checked against its revision |
| `invariance` | Probability-level independence and order checks |
| `eval` | Metrics shared by every report: ECE, Brier, AUROC, pair agreement, review curves |
| `data.schema` | The canonical record (chapter 8) |
| `data.dataset` | Loading rows, with the licence and tier checks against the registry |
| `data.mix` | Licence tiers for mixes and the holdout check (`check_mix`) |
| `data.cleanroom` | Whitelist, canary search and the answer-shape scanner (16.1) |
| `data.rating` | Items for a rating app and ratings back into human targets (hv-items/1, hv-ratings/1), data-check meta-questions |
| `data.converters` | One converter per source, registered by name in `scripts/convert_dataset.py` |
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
| Masks and packing | `test_tree.py`, `test_mask_equivalence.py`, `test_media.py`, `test_audio.py`, `test_moe_precision.py` |
| Request contract | `test_api_types.py`, `test_render.py`, `test_preprocess.py`, `test_confidence.py`, `test_system_one.py`, `test_invariance.py` |
| Model and training | `test_lora.py`, `test_scorer.py`, `test_losses.py`, `test_train.py`, `test_bundle.py`, `test_calibrate.py`, `test_model_origin.py` (every configured backbone and teacher passes the origin rule) |
| Serving | `test_serve.py` (bundle switching gives the same answers as loading each bundle alone, auth, key-check cache, refusals, settings) |
| Data | `test_data_registry.py` (licences, tiers, holdouts), `test_cleanroom.py` (whitelist, canary, scanner), `test_fetch_dataset.py`, the converter tests (mostly `test_<source>.py`, e.g. `test_multinli.py`, `test_cifar10h.py`; `test_intent_converters.py` for the intent sets, `test_vqa.py` for the photo question-answering sets), `test_convert_dataset.py` (your own converter), `test_schema_explosion.py`, `test_suite_converters.py` |
| Human ratings | `test_rating.py` (rating formats), `test_build_dev_v1.py`, `test_build_two_sided.py` |
| Teacher and generation | `test_label.py`, `test_teacher.py`, `test_generate.py` |
| Evaluation and release | `test_eval.py`, `test_eval_dataset.py`, `test_score_suite.py` |
| Local settings | `test_hub.py` (`configs/local.yaml` handling) |

## 17.5 Environment variables

| Variable | Use |
|---|---|
| `HF_HUB_CACHE` | Where Hugging Face weights and tokenizers are stored. Set it on a machine whose default cache belongs to another application, or use `hf_cache` in `configs/local.yaml` |
| `HF_HUB_OFFLINE=1` | Load only what is already downloaded. A missing pinned revision fails instead of downloading |
| `CUDA_VISIBLE_DEVICES` | Choose the GPU(s) a script uses |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces memory fragmentation in long training runs with varying sequence lengths |
| `PYTHONUNBUFFERED=1` | Prints progress lines immediately, useful when a long job's output goes to a log file |

The server (`decisivelm.serve`, 4.10) reads only environment variables:

| Variable | Use |
|---|---|
| `DECISIVELM_BUNDLES` | Required: a bundle directory, a directory of bundles, or several of either joined by commas. All must be LoRA bundles of the same shape on the same backbone revision |
| `DECISIVELM_DEFAULT` | The bundle used when a request names none (default: the first) |
| `DECISIVELM_HOST` | Address(es) to bind, joined by commas (default `127.0.0.1`) |
| `DECISIVELM_PORT` | Default 8003 |
| `DECISIVELM_DEVICE` | e.g. `cuda:0` (default: CUDA, else MPS, else CPU) |
| `DECISIVELM_MEMORY_GB` | A hard cap on the process's GPU memory (CUDA), so an oversized request fails in this server, not in a neighbour |
| `DECISIVELM_MAX_TOKENS` | The longest packed request accepted (default 12,288: a state at its 8,192-token budget plus about 4,000 tokens of questions). Longer ones get `413` |
| `DECISIVELM_AUTH` | `none` (default), `keys` or `litellm`. With `none`, the server refuses to bind anything but loopback unless `DECISIVELM_ALLOW_UNAUTHENTICATED=1` |
| `DECISIVELM_API_KEYS` | Bearer tokens accepted with `keys`, joined by commas |
| `DECISIVELM_LITELLM_URL` | With `litellm`, a token is accepted if this LiteLLM proxy accepts it. Answers are cached for 5 minutes (refusals for 10 s) |
| `DECISIVELM_ALLOW_UNAUTHENTICATED` | `1` allows a non-loopback bind without auth, e.g. inside a container on a private network |

## 17.6 Repository layout

```
decisivelm/        the package (17.3)
scripts/           command-line tools (17.1)
configs/           YAML configuration (17.2)
tests/             the test suite (17.4)
docs/              this guide, the release criteria and the model card
Dockerfile         the server image (4.10)
results/           evaluation reports written by the scripts
bundles/           trained adapter bundles (not tracked)
data/              raw, processed, teacher, mixes and predictions (not tracked)
```

Back to the [README](../../README.md)
