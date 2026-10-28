# 2. Installation

DecisiveLM is a Python package managed with `uv`. This chapter covers the hardware you need, installing the package,
getting the Gemma 4 weights, pointing the toolkit at a model cache, and checking that everything works. Using the
general head needs only the steps up to section 2.5; training adds nothing beyond the same environment.

## 2.1 Requirements

| | Minimum | Notes |
|---|---|---|
| Operating system | Linux or macOS | Linux with CUDA for training at scale; macOS on Apple silicon works for inference and small fine-tunes |
| Python | 3.13 | installed for you by `uv` |
| Package manager | [uv](https://docs.astral.sh/uv/) | pins every dependency through `uv.lock` |
| Accelerator | an NVIDIA GPU with CUDA, or Apple silicon (MPS) | the CPU works for the test suite, not for real models |
| Disk | about 15 GB for Gemma 4 E4B, about 50 GB for 26B A4B | plus your data |

Measured memory use:

| Backbone | Inference (bf16) | Training a head (bf16, LoRA) |
|---|---|---|
| Gemma 4 E4B | 15 GB (14.8 GB of weights) | about 20 GB with 4,096-token micro-batches and gradient checkpointing |
| Gemma 4 26B A4B | 49 GB (48.1 GB of weights) | 59–69 GB with 4,096-token micro-batches |

Gemma 4 26B A4B is a mixture-of-experts model. In bf16 its expert routing is sensitive to rounding, so a head on it is
best run with fp32 computation (`--dtype float32` in the training and checking scripts), which keeps the weights in
bf16 and needs about 54 GB for inference. Chapter 6 explains why.

## 2.2 Install the package

```bash
git clone https://github.com/xkiwilabs/decisive-lm.git
cd decisive-lm
uv sync
```

`uv sync` creates `.venv/` with Python 3.13, PyTorch and the pinned version of Hugging Face `transformers`. The
`transformers` version is pinned exactly in `pyproject.toml` because the tree attention masks (chapter 6) depend on
Gemma 4's internal mask handling; do not upgrade it on its own.

Run every command through `uv run`, which uses the project environment:

```bash
uv run python -c "import decisivelm; print(decisivelm.__file__)"
```

## 2.3 Get the Gemma 4 weights

DecisiveLM loads Gemma 4 from Hugging Face at an exact revision, recorded per backbone in `configs/model.yaml`:

| Profile | Model id | Used for |
|---|---|---|
| `e4b` | `google/gemma-4-E4B-it` | the default backbone and the general head |
| `26b-a4b` | `google/gemma-4-26B-A4B-it` | the larger backbone, and the default teacher (chapter 12) |

Gemma 4 is released by Google DeepMind under Apache-2.0. Read the model card on Hugging Face before use. If your
account must accept terms to download the model, accept them on the model page and log in once:

```bash
uv run hf auth login
```

The first load downloads the weights into your Hugging Face cache. To download ahead of time at the pinned revision:

```bash
uv run hf download google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2
```

The revision is part of every head. A head records the backbone id and revision it was trained on
(`calibration.json` → `backbone`). `DecisiveLM.load` loads that revision, and `attach_bundle` and the scripts refuse a
head on a different one.

## 2.4 Choose where models are cached (optional)

By default the standard Hugging Face cache is used. To keep DecisiveLM's files apart, for example on a machine where
the default cache belongs to a model server, create a machine-local settings file:

```bash
cp configs/local.example.yaml configs/local.yaml
```

`configs/local.yaml` is never committed. It has two keys, read by `decisivelm.model.hub_kwargs` for every model and
tokenizer load:

| Key | Effect |
|---|---|
| `hf_cache` | the directory DecisiveLM reads and writes Hugging Face files in |
| `offline` | `true` loads models only from `hf_cache`; a missing pinned revision then fails instead of downloading |

Dataset downloads (`scripts/fetch_dataset.py`, chapter 14) always go online. If you set `offline: true`, download the
pinned model revisions into `hf_cache` first, for example with `uv run hf download --cache-dir <hf_cache> …`.

## 2.5 Check the installation

Run the test suite:

```bash
uv run pytest
```

It runs on the CPU in under a minute, against tiny randomly initialised Gemma 4 configurations (a dense one and a
mixture-of-experts one, with a sliding window of 8 tokens so short inputs exercise it). It checks, among other things:

- that the packed tree attention gives the same hidden states as running every branch on its own (mask equivalence);
- that answers do not change when other questions are added or options are shuffled (independence and order);
- that every response conforms to the request (probabilities over exactly the given options, summing to 1);
- the data rules: licence tiers, held-out datasets, and the clean-room checks of chapter 16.

Run a single test with its path, for example:

```bash
uv run pytest tests/test_tree.py::test_sliding_window_uses_logical_not_physical_distance
```

The `real_model` marker is reserved for tests that need downloaded Gemma 4 weights; `pyproject.toml` deselects it by
default, and `uv run pytest -m real_model` selects it.

## 2.6 Check a real backbone

`scripts/m0_real.py` runs the same structural checks on the real model and prints a report:

```bash
uv run python scripts/m0_real.py --backbone e4b
uv run python scripts/m0_real.py --backbone 26b-a4b --fp32-device cpu
```

It reports:

1. the configuration facts DecisiveLM relies on: the pattern of sliding-window and global attention layers, the
   sliding window size, key-value sharing, and the attention backends available;
2. that the three reserved token ids used as query tokens are genuinely unused by the tokenizer;
3. mask equivalence in fp32, with an image, a state longer than the sliding window, and branches whose combined length
   exceeds it (on 26B A4B with expert selection held fixed, so the check tests the masks rather than routing);
4. the bf16 noise floor: how much hidden states move when the same request is packed at different offsets;
5. a first latency figure for a 1,000-token state with 20 answer blocks.

| Flag | Default | Use |
|---|---|---|
| `--backbone` | `e4b` | `e4b` or `26b-a4b` |
| `--device` | `cuda` if available, else `mps`, else `cpu` | where the bf16 sections run; on CUDA the model spreads over every visible GPU |
| `--fp32-device` | the `--device` | where the fp32 mask check runs; `cpu` for 26B A4B on a single GPU (its fp32 copy is about 104 GB) |
| `--skip-latency` | off | skip the timing section |

## 2.7 Layout of the repository

| Path | Contents |
|---|---|
| `decisivelm/` | the package: request types, rendering, tree masks, model, head, training, calibration, evaluation |
| `decisivelm/data/` | the canonical record, dataset registry and licence checks, converters, mix checks |
| `decisivelm/synth/` | teacher labelling, question and state generation, prompt templates |
| `scripts/` | command-line entry points, one per step |
| `configs/` | backbone profiles, teachers, datasets, held-out lists, release settings |
| `tests/` | the CPU test suite |
| `bundles/` | trained heads (one directory each; not committed) |
| `data/` | raw, processed and generated data, predictions (not committed) |

Next: [3. Quickstart](03-quickstart.md)
