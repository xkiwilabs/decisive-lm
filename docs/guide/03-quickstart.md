# 3. Quickstart

This chapter takes you from an installed toolkit to your first answers: download the general head, load it onto Gemma 4
E4B, send a request from Python, and read the response. It ends with running a head over a whole file of requests.
It assumes the installation of chapter 2.

## 3.1 Download the general head

The general head is published on Hugging Face as a directory of three files:

| File | Contents |
|---|---|
| `adapter.pt` | the LoRA weights for the backbone's decoder layers and the three query-token embeddings |
| `scorer.pt` | the scorer network that turns a hidden state into a score, with its input standardisation |
| `calibration.json` | the calibration temperatures and Noul bias, the backbone id and revision, and the training settings |

Download it into `bundles/`:

```bash
uv run hf download xkiwilabs/decisivelm-e4b-general --local-dir bundles/decisivelm-e4b-general
```

The repository id is the one given on the model card ([docs/model-card.md](../model-card.md)); each release is a tagged
revision, which you can pin with `--revision`.

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
records. Loading takes a few seconds once the weights are cached. The backbone is modified in place, so load a fresh
backbone for each head you attach.

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
  means it is spread evenly (section 3.6).
- `usage` reports the number of input tokens, the number of answer blocks (one per option here), the time taken, the
  confidence method and the head's name. The optional request field `model` is echoed back unchanged.

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

- **Noul**: `noul` is the probability that the statement is true. There is no separate confidence; the probability
  itself says how sure the model is.
- **Score**: `probabilities` is over the levels, numbered from 0 in the order you listed them. `score` is the expected
  level, Σ level × probability, so it can fall between levels (1.12 here). `legend` maps each level number back to its
  description.

The question names (`department`, `is_urgent`, `frustration`) are keys for you; the model never sees them, so the
`instructions` must say everything the question means. Chapter 5 covers how to write questions that work well.

## 3.5 Warnings

Some requests are valid but likely to be answered badly. `system_one` stores warnings for the last request in
`dlm.warnings` rather than refusing it:

```python
response = dlm.system_one(request)
for w in dlm.warnings:
    print(w)
```

The main warning is for an `other` option without a self-contained description: each option is judged on its own, so
"Other" gives the model nothing to compare against. Write what it covers instead, for example "The message is about
something other than payments, outages or pricing" (chapter 5).

Invalid requests raise a `pydantic.ValidationError`: an unknown question type, fewer than 2 or more than 255 Choice
options, fewer than 2 or more than 10 Score levels, more than 4 images, or unknown fields.

## 3.6 Choosing a confidence measure

The default confidence measure, `peak`, rescales the top probability so that a uniform distribution gives 0 and a
certain one gives 1: (n · p_max − 1) / (n − 1) for n options. Others are available in `decisivelm.confidence`:

| Method | Definition | Good for |
|---|---|---|
| `peak` | (n · p_max − 1) / (n − 1), clipped to [0, 1] | the default; compatible with the published API |
| `entropy` | 1 − H(p) / log n | uses the whole distribution, not just the top option |
| `margin` | p_top1 − p_top2 | separating close calls between two options |
| `ordinal` | 1 − Var(level) / Var_max | Score questions: spreading over neighbouring levels costs less than over distant ones |

Switch with:

```python
dlm.confidence_method = "entropy"
```

The method in use is reported in `usage.confidence_method`. Chapter 7 explains how to pick a review threshold.

## 3.7 Images

A request can carry up to four images in `media`, placed before the state text in the model's input. Give each as a
file path or as base64 data:

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
fixed budget of 280 image tokens. Image URLs are refused; download the image and pass it as a path or as `data`. The general head was trained on text; for image tasks, train a fine-tune on labelled images of
your own (chapter 9).

## 3.8 Running a head over a file

To answer many text requests, put them in a JSONL file of canonical rows (one request per line, chapter 8) and run:

```bash
uv run python scripts/predict_bundle.py --bundle bundles/decisivelm-e4b-general \
    --data data/processed/my_requests.jsonl --splits gold_test
```

| Flag | Default | Meaning |
|---|---|---|
| `--bundle` | required | the head's directory |
| `--data` | required | one or more JSONL files of canonical rows |
| `--splits` | `gold_test` | which rows to answer, by their `split` field |
| `--max-tokens` | `4096` | tokens per packed batch; lower it if you run out of memory |

The script checks the head's backbone revision, packs several requests into each forward pass, and writes
`data/predictions/<bundle name>__<data file stem>.jsonl`, one line per row:

```json
{"id": "row-17", "split": "gold_test",
 "answers": {"department": {"type": "choice", "probabilities": {"billing": 0.01, "technical": 0.96, "sales": 0.03}},
             "is_urgent": {"type": "noul", "p": 0.97}}}
```

Rows with a single Choice or Score question also carry its `probabilities` at the top level. These files are what the
evaluation scripts read (chapter 11). Every row's source dataset must be listed in `configs/datasets.yaml`, or for
private data in the uncommitted `configs/datasets.local.yaml` (chapter 8).

## 3.9 Where to go next

- The full request and response format: [chapter 4](04-requests-and-responses.md).
- Writing questions the model answers well: [chapter 5](05-writing-good-questions.md).
- What the general head is good at, and how to use its confidence: [chapter 7](07-using-the-general-head.md).
- Training a head on your own labelled data: [chapter 9](09-fine-tuning-for-your-task.md).

Next: [4. Requests and responses](04-requests-and-responses.md)
