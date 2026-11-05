# 3. Quickstart

This chapter takes you from an installed toolkit to your first answers: download the general head, load it onto Gemma 4
E4B (the default, and the 26B A4B head loads the same way, section 3.1), send a request from Python, and read the
response. It ends with running a head over a whole file of requests. It assumes the installation of chapter 2.

## 3.1 Download the general head

There are two general heads: `decisivelm-e4b-general` for Gemma 4 E4B, and `decisivelm-26b-a4b-general` for Gemma 4 26B
A4B. This chapter uses the E4B head, which runs on a 24 GB GPU and is the faster of the two. Section 7.10 helps you
choose. Each head is published on Hugging Face as a bundle, a directory of three files:

| File | Contents |
|---|---|
| `adapter.pt` | the LoRA weights for the backbone's decoder layers and the three query-token embeddings |
| `scorer.pt` | the scorer network that turns a hidden state into a score, with its input standardisation |
| `calibration.json` | the calibration temperatures and Noul bias, the backbone id and revision, and the training settings |

You do not need to open these files. Section 6.11 describes each one, and chapter 10 the calibration. Download the
bundle into `bundles/`:

```bash
uv run hf download xkiwilabs/decisivelm-e4b-general --local-dir bundles/decisivelm-e4b-general
```

The repository id is the one given on the model card ([docs/model-card.md](../model-card.md)). Each release is a
tagged revision, which you can pin with `--revision`. If the download reports that the model is not found, check the id
against the model card. A head you train yourself from the base model (section 9.5) works in every example below in
place of `bundles/decisivelm-e4b-general`.

To use the 26B A4B head instead, download `xkiwilabs/decisivelm-26b-a4b-general` into
`bundles/decisivelm-26b-a4b-general` and load that directory in the examples below. It has the same three files and the
same calls, except that `load_backbone` and `load_image_processor` take `"26b-a4b"` where the examples pass `"e4b"`. It
computes in float32 (the bundle sets this), so it needs a GPU with about 72 GB free and takes about 0.47 s for a
20-block request over HTTP, against about 0.10 s for the E4B head. It accepts text and images but not audio
([model card](../model-card-26b.md)).

## 3.2 Load the model

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/decisivelm-e4b-general")
```

`DecisiveLM.load(bundle, dtype=None, device=None, media_root=None)` (the same as `decisivelm.bundle.load_bundle`)
reads the head's `calibration.json`, finds the backbone profile in `configs/model.yaml` with the model id and revision
the head was trained on, loads that backbone in the dtype the head was trained in, applies the head, and sets up the
image path. With `device=None` it uses CUDA if available, then Apple silicon (MPS), then the CPU. A head whose backbone
has no profile is refused.

To attach a head to a backbone you have already loaded, use the two steps it is made of:

```python
from decisivelm.bundle import attach_bundle
from decisivelm.model import load_backbone

backbone, tokenizer = load_backbone("e4b")          # Gemma 4 E4B at the revision in configs/model.yaml
dlm = attach_bundle("bundles/decisivelm-e4b-general", backbone, tokenizer)
```

`attach_bundle` refuses a backbone from `load_backbone` whose model id or revision differs from the one the head
records. Loading takes a few seconds once the weights are cached. `attach_bundle` adds the head's LoRA layers to the
loaded backbone in memory. Gemma 4's own weights, in memory and on disk, stay unchanged, but a loaded backbone carries
one head at a time, so load a fresh backbone for each head you attach.

## 3.3 Your first request

A request is a dictionary with a `state` and named `questions`. Each question has a `type`, `instructions` and, for
Choice and Score, `criteria`:

```python
request = {
    "state": "Our API started returning 500s 20 minutes ago and we can't process any orders.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Payment, invoice or subscription problems",
                "technical": "Bugs, outages or integration problems",
                "sales": "Pricing, plans or account questions",
            },
        },
    },
}

response = dlm.system_one(request)
answer = response["answers"]["department"]
print(answer["choice"], answer["confidence"])
print(answer["probabilities"])
```

`system_one` validates the request, builds the input, runs one forward pass and returns a dictionary. With the values
shown purely as an illustration:

```python
{
    "model": None,
    "answers": {
        "department": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.94,
            "probabilities": {"billing": 0.01, "technical": 0.96, "sales": 0.03},
        }
    },
    "usage": {"input_tokens": 118, "blocks": 3, "latency_ms": 85.0,
              "confidence_method": "peak", "adapter": "decisivelm-e4b-general"},
}
```

- `probabilities` covers exactly the options you gave and sums to 1.
- `choice` is the most probable option.
- `confidence` is a 0–1 summary of how peaked the distribution is: 1 means all the probability is on one option, 0
  means it is spread evenly (section 3.6). It is not the chance that the answer is right. That chance is the top
  probability, 0.96 here, once the head is calibrated for the task (section 3.6).
- `usage` reports the number of input tokens, the number of answer blocks (one per option here), the time taken, the
  confidence method and the head's name (its bundle directory). In Python the optional request field `model` is
  echoed back unchanged. Over HTTP it chooses the head that answers (section 4.10).

## 3.4 Several questions at once

Questions about the same state go in one request. They are answered together in one pass, and each is answered
independently of the others:

```python
request = {
    "state": "Our API started returning 500s 20 minutes ago and we can't process any orders.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Payment, invoice or subscription problems",
                "technical": "Bugs, outages or integration problems",
                "sales": "Pricing, plans or account questions",
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message reports a problem that needs action within the hour",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the writer appear?",
            "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
        },
    },
}
response = dlm.system_one(request)
```

The three answer types come back in different shapes:

```python
response["answers"]["is_urgent"]     # {"type": "noul", "noul": 0.97}
response["answers"]["frustration"]   # {"type": "score", "score": 1.12, "confidence": 0.67,
                                     #  "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil",
                                     #             "2": "Very angry, strong language"},
                                     #  "probabilities": {"0": 0.05, "1": 0.78, "2": 0.17}}
```

- **Noul**: `noul` is the probability that the statement is true. There is no separate confidence, because the
  probability itself says how sure the model is.
- **Score**: `probabilities` is over the levels, numbered from 0 in the order you listed them. `score` is the expected
  level, Σ level × probability, so it can fall between levels (1.12 here). `legend` maps each level number back to its
  description.

The question names (`department`, `is_urgent`, `frustration`) are keys for you. The model never sees them, so the
`instructions` must say everything the question means. Chapter 5 covers how to write questions that work well.

## 3.5 Warnings

Some requests are valid but likely to be answered badly. `system_one` stores warnings for the last request in
`dlm.warnings` rather than refusing it:

```python
response = dlm.system_one(request)
for w in dlm.warnings:
    print(w)
```

There are two warnings. The first is for an `other` option without a self-contained description. Each option is judged
on its own, so "Other" gives the model nothing to compare against. Write what it covers instead, for example "The
message is about something other than payments, outages or pricing" (chapter 5). The second is for a question of a type
the bundle was never trained on (chapter 9).

Invalid requests raise a `pydantic.ValidationError`: an unknown question type, fewer than 2 or more than 255 Choice
options, fewer than 2 or more than 10 Score levels, more than 4 images, or unknown fields.

## 3.6 Choosing a confidence measure

The default confidence measure, `peak`, rescales the top probability so that a uniform distribution gives 0 and a
certain one gives 1: (n · p_max − 1) / (n − 1) for n options. Others are available in `decisivelm.confidence`:

| Method | Definition | Good for |
|---|---|---|
| `peak` | (n · p_max − 1) / (n − 1), clipped to [0, 1] | the default, and the formula of the published System One API (section 4.1) |
| `entropy` | 1 − H(p) / log n | uses the whole distribution, not only the top option |
| `margin` | p_top1 − p_top2 | separating close calls between two options |
| `ordinal` | 1 − Var(level) / Var_max | Score questions: spreading over neighbouring levels costs less than over distant ones |

Switch with:

```python
dlm.confidence_method = "entropy"
```

The method in use is reported in `usage.confidence_method`.

**Which number to threshold.** To choose the answers a person should check, rank them by the top probability,
`max(probabilities.values())`. The top probability is the chance that the answer is right, calibrated on human labels
(on a new task, after the step in section 10.6), and it is the signal chapters 10 and 11 measure. For one question,
`peak` ranks answers in the same order, because it rescales the top probability by the number of options. Across
questions with different numbers of options the two rank differently, and only the top probability keeps its calibrated
meaning. For a Noul, rank by `max(noul, 1 − noul)`. Chapter 7 explains how to pick a review threshold.

## 3.7 Images

A request can carry up to four images in `media`, placed before the state text in the model's input. Of images, the
general head was trained only on photo questions (VizWiz-VQA: photos that blind people took to ask a question about
them), so on any other image task it answers zero-shot (see the end of this section). Give each image as a file path or
as base64 data:

```python
dlm = DecisiveLM.load("bundles/decisivelm-e4b-general", media_root="my_images")   # relative paths resolve here

response = dlm.system_one({
    "state": "Photo of the parcel as delivered.",
    "media": [{"type": "image", "path": "parcel_0412.jpg"}],
    "questions": {"damaged": {"type": "noul", "instructions": "The parcel shown is visibly damaged"}},
})
```

`DecisiveLM.load` sets up the backbone's image processor. After `attach_bundle`, set it yourself:
`dlm.image_processor = decisivelm.model.load_image_processor("e4b")` and `dlm.media_root`. Each image is encoded at a
fixed budget of 280 image tokens. Image URLs are refused. Download the image and pass it as a path or as `data`.

Zero-shot on one image set, CIFAR-10H (small photos of ten kinds of object, each labelled by about 51 people), the E4B
general head was more accurate than the prompted base model (0.909 against 0.875), and its confidence was closer than
the base model's to how much people agreed. Its raw probabilities on these images were too low (underconfident),
however, until it was calibrated on the task ([model card](../model-card.md#zero-shot-on-other-images), and section
7.6). The 26B A4B head has not been scored on another image set, and neither head has been evaluated on audio. For an
image task of your own, calibrate the general head on about 100 of your labelled images (section 10.6) before you rely
on its confidence, or train a fine-tune on them (chapter 9).

## 3.8 Running a head over a file

To answer many requests, put them in a JSONL file of canonical rows (chapter 8), one request per line. A row without
labels has an empty `targets`. `predict_bundle.py` answers only the rows whose `split` is listed in `--splits`, which
defaults to `gold_test` (in a labelled dataset, the rows held back for testing, section 8.3), so the row below uses
`gold_test`. A file with no rows in the listed splits gives an empty output. Shown here over several lines, the row
must be a single line in the file:

```json
{"id": "req-1",
 "lineage": {"source_dataset": "my_requests", "source_id": "1", "family_id": "my_requests-1"},
 "license": "LicenseRef-private", "tier": "X", "modality": ["text"], "language": "en",
 "state": "My card was declined twice today.",
 "questions": {"department": {"type": "choice", "instructions": "Which team should handle this message?",
   "criteria": {"billing": "Payments, refunds and invoices", "technical": "Bugs and outages",
                "sales": "Plans and pricing"}}},
 "targets": {}, "preprocess": {"render_version": 1, "max_state_tokens": 8192, "truncated": false},
 "split": "gold_test"}
```

The source dataset, here `my_requests`, must be registered: a public dataset in `configs/datasets.yaml`, and private
data in the uncommitted `configs/datasets.local.yaml` (section 8.8 shows an entry). Then run:

```bash
uv run python scripts/predict_bundle.py --bundle bundles/decisivelm-e4b-general \
    --data data/processed/my_requests.jsonl --splits gold_test
```

| Flag | Default | Meaning |
|---|---|---|
| `--bundle` | required | the head's directory |
| `--data` | required | one or more JSONL files of canonical rows |
| `--splits` | `gold_test` | answer only the rows whose `split` is one of these |
| `--max-tokens` | `4096` | tokens per packed batch (lower it if you run out of memory) |
| `--media-root` | the repository root | where relative media paths resolve |
| `--image-budget` | 280 (the backbone profile's) | image tokens per image, for a head that records no budget of its own |

The script checks the head's backbone revision, packs several requests into each forward pass, and writes
`data/predictions/<bundle name>__<data file stem>.jsonl`, one line per row:

```json
{"id": "req-1", "split": "gold_test",
 "answers": {"department": {"type": "choice", "probabilities": {"billing": 0.93, "technical": 0.05, "sales": 0.02}}},
 "probabilities": {"billing": 0.93, "technical": 0.05, "sales": 0.02}}
```

Rows with a single Choice or Score question also carry its `probabilities` at the top level. The file is in the format
the evaluation scripts read (chapter 11), not the response format. It holds probabilities only, with a Noul's under `p`
and no `choice` or `confidence`. The answer is the most probable option, and its confidence follows from the
probabilities as in section 3.6. The script runs the backbone in the compute type the head records (bf16 for the E4B
head, float32 for the 26B A4B head), and it answers rows with images or audio as well as text. To run the E4B head in
float32, load it with `DecisiveLM.load(..., dtype=torch.float32)` and call `system_one`.

## 3.9 Where to go next

- The full request and response format: [chapter 4](04-requests-and-responses.md).
- Writing questions the model answers well: [chapter 5](05-writing-good-questions.md).
- What the general head is good at, and how to use its confidence: [chapter 7](07-using-the-general-head.md).
- Training a head on your own labelled data: [chapter 9](09-fine-tuning-for-your-task.md).

Next: [4. Requests and responses](04-requests-and-responses.md)
