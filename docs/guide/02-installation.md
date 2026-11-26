# 2. Installation

DecisiveLM is a Python package managed with `uv`. This chapter covers the hardware you need, installing the package,
getting the Gemma 4 weights, pointing the toolkit at a model cache, and checking that everything works. Using the
general head (either one) needs only the steps up to section 2.5. Training needs the same environment and nothing more.

## 2.1 Requirements

| | Minimum | Notes |
|---|---|---|
| Operating system | Linux or macOS | Linux with CUDA for training at scale; macOS on Apple silicon works for inference and small fine-tunes |
| Python | 3.13 | installed for you by `uv` |
| Package manager | [uv](https://docs.astral.sh/uv/) | pins every dependency through `uv.lock`. Install it as its [installation page](https://docs.astral.sh/uv/getting-started/installation/) describes, or with `brew install uv` on macOS |
| Git | any recent version | to clone the repository |
| Accelerator | an NVIDIA GPU with CUDA, or Apple silicon (MPS) | the CPU works for the test suite, not for real models |
| Disk | about 15 GB for Gemma 4 E4B, about 50 GB for 26B A4B | plus about 1 GB for the Python environment, and your data |

Measured memory use:

| Backbone | Inference | Training a head (LoRA) |
|---|---|---|
| Gemma 4 E4B (bf16) | 16.0 GiB predicting with `--max-tokens 2048` (the most tokens processed at once), 17.8 GiB with 8192, and about 18.6 GiB for one request at the server's default limit (section 4.10). The weights are 14.8 GiB of it (16.0 GB on disk) | 16.7 GiB with `--max-tokens 2048`, 18.5 GiB with 4096 and 21.2 GiB with the default 8192, with gradient checkpointing (on by default). Section 9.5 lists settings for smaller GPUs and Apple silicon |
| Gemma 4 26B A4B (bf16) | 49 GiB (48.1 GiB of weights, 51.6 GB on disk) | up to 69 GiB with `--max-tokens 4096`, on one coding fine-tune |
| 26B A4B general head, through the server (fp32 computation) | 55.8 GiB after loading, 67.0 GiB at the peak, at the server's default limit of 12,288 tokens per request | 66.0 GiB training the head with `--max-tokens 4096` |
| 26B A4B general head on Apple silicon (MPS, fp32 computation) | 56.0 GiB after loading and 58.6 GiB at the peak over 20 requests of up to 1,599 tokens, on an M5 Max with 128 GB of unified memory, so a Mac needs 96 GB or more. About 1.9 s per request once warm | not measured |

The E4B figures are the peaks `nvidia-smi` reported for each process on one RTX PRO 6000, CUDA context included, for a
300-item fine-tune and for predicting 792 rows. The 26B A4B figures are `nvidia-smi` readings of the whole card, an RTX
PRO 6000 Blackwell Max-Q (96 GB) with nothing else on it, sampled every 0.2 s while serving and every 10 s while
training. The serving peak came on the longest requests, about 12,000 tokens with 110 to 130 options. The bf16 26B A4B
row is PyTorch's own peak, which leaves out the CUDA context of about 0.5–1 GiB. A 24 GB GPU is enough for E4B, and a 16
GB GPU is not at these settings.

bf16 and fp32 are 16-bit and 32-bit floating-point number formats. Models load and run in bf16 by default, which halves
memory but rounds more coarsely. Gemma 4 26B A4B is a mixture-of-experts model, which sends each token through 8 of its
128 expert sub-networks. In bf16 the choice of experts is sensitive to rounding, so a head on 26B A4B is best run with
fp32 computation (`--dtype float32` in `train_p3.py`, `cache_features.py` and `check_invariance.py`, or
`dtype=torch.float32` in `DecisiveLM.load`). This keeps the expert weights, about 90% of the model, in bf16 and converts
them to fp32 for each multiplication, so it needs about 56 GiB for inference rather than the 104 GB of a full fp32 copy.
The 26B A4B general head does this on its own when you load it, and it needs a GPU with about 72 GiB free or a Mac with
96 GB or more of unified memory. Section 6.9 gives the details. E4B is a dense model and much less sensitive to rounding
(section 11.7).

## 2.2 Install the package

```bash
git clone https://github.com/xkiwilabs/decisive-lm.git
cd decisive-lm
uv sync
```

`uv sync` creates `.venv/` with Python 3.13, PyTorch and the pinned version of Hugging Face `transformers`. The
`transformers` version is pinned exactly in `pyproject.toml` because the tree attention masks (chapter 6) depend on
Gemma 4's internal mask handling. Do not upgrade it on its own.

Two optional extras add features the core install leaves out:

```bash
uv sync --extra audio     # audio states (section 4.5)
uv sync --extra serve     # the HTTP server (section 4.10)
```

Run every command through `uv run`, which uses the project environment:

```bash
uv run python -c "import decisivelm; print(decisivelm.__file__)"
```

## 2.3 Get the Gemma 4 weights

DecisiveLM loads Gemma 4 from Hugging Face at an exact revision, recorded per backbone in `configs/model.yaml`:

| Profile | Model id | Used for |
|---|---|---|
| `e4b` | `google/gemma-4-E4B-it` | the default backbone and the E4B general head |
| `26b-a4b` | `google/gemma-4-26B-A4B-it` | the larger backbone, the 26B A4B general head, and the default teacher (chapter 12) |

Gemma 4 is released by Google DeepMind under Apache-2.0. Read the model card on Hugging Face before use. The models
are not gated, so no Hugging Face account or login is needed to download them.

The first load downloads the weights into your Hugging Face cache. To download ahead of time at the pinned revision:

```bash
uv run hf download google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2
# for the 26B A4B head (about 50 GB):
uv run hf download google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
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
| `offline` | `true` loads models only from `hf_cache`. A missing pinned revision then fails instead of downloading |

Dataset downloads (`scripts/fetch_dataset.py`, chapter 14) always go online. If you set `offline: true`, download the
pinned model revisions into `hf_cache` first, for example with `uv run hf download --cache-dir <hf_cache> …`.

## 2.5 Check the installation

Run the test suite:

```bash
uv run pytest
```

It runs on the CPU in about a minute, against tiny randomly initialised Gemma 4 configurations (a dense one and a
mixture-of-experts one, with a sliding window of 8 tokens so short inputs exercise it). It checks, among other things:

- that each answer is computed as if its question were asked alone (mask equivalence: packing every question into one
  sequence gives the same hidden states as running each one separately);
- that answers do not change when other questions are added or options are shuffled (independence and order);
- that every response conforms to the request (probabilities over exactly the given options, summing to 1);
- the data rules: licence tiers, held-out datasets, and the clean-room checks of chapter 16.

In a fresh clone of version 0.1, a clean run ends with a line like `384 passed, 2 skipped, 36 warnings` (the counts grow
as tests are added). One skipped test is the HTTP server's, which needs the `serve` extra. The other is the canary test
of chapter 16, which looks under `data/processed/` and is skipped until that folder exists. The warnings come from tests
of degenerate inputs: numpy dividing by zero, and calibration warnings that the tests trigger on purpose. All of these
are expected. Any `failed` or `error` is not.

Run a single test with its path, for example:

```bash
uv run pytest tests/test_tree.py::test_sliding_window_uses_logical_not_physical_distance
```

The `real_model` marker is reserved for tests that need downloaded Gemma 4 weights. `pyproject.toml` deselects it by
default, and `uv run pytest -m real_model` selects it.

## 2.6 Check a real backbone

This step is optional. Using the general head needs only sections 2.1 to 2.5. `scripts/m0_real.py` runs the same
structural checks on the real model, on your hardware, and prints a report:

```bash
uv run python scripts/m0_real.py --backbone e4b
uv run python scripts/m0_real.py --backbone 26b-a4b --fp32-device cpu
```

It reports:

1. the configuration facts DecisiveLM relies on: the pattern of sliding-window and global attention layers, the sliding
   window size, key-value sharing (layers that reuse an earlier layer's attention keys and values), and the attention
   backends available;
2. that the tokenizer never produces the three reserved token ids used as query tokens;
3. mask equivalence in fp32, with an image, a state longer than the sliding window, and branches whose combined length
   exceeds it (on 26B A4B with expert selection held fixed, so the check tests the masks rather than routing);
4. the bf16 noise floor: how much hidden states move when the same request is packed at different offsets;
5. a first latency figure for a 1,000-token state with 20 answer blocks.

Two lines say whether the backbone works: under `## 2. Reserved query-token ids`, the line
`emitted by the tokenizer …: none`, and under `## 3. Mask equivalence, fp32`, a line ending in `PASS`. The other
sections are measurements. `results/M0.md` records them for both backbones, for example 75 ms for the latency check
with E4B on one NVIDIA RTX PRO 6000 GPU.

| Flag | Default | Use |
|---|---|---|
| `--backbone` | `e4b` | `e4b` or `26b-a4b` |
| `--device` | `cuda` if available, else `mps`, else `cpu` | where the bf16 sections run. On CUDA the model spreads over every visible GPU |
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
| `bundles/` | trained heads, one bundle directory each (not committed) |
| `data/` | raw, processed and generated data, predictions (not committed) |

Next: [3. Quickstart](03-quickstart.md)
