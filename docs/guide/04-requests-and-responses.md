# 4. Requests and responses

Every call to DecisiveLM takes one **request**: a state plus a set of typed questions. It returns one **response**:
a constrained, calibrated answer to every question, from a single forward pass of the model. This chapter is the
complete reference for both: every field, every limit, what is rejected, what only raises a warning, and how the
numbers in a response are computed.

The request and response are defined as pydantic classes in `decisivelm/api_types.py`. `DecisiveLM.system_one` in
`decisivelm/infer.py` accepts either a plain `dict` or a `SystemOneRequest`, and returns a plain `dict` that has been
validated against `SystemOneResponse`.

## 4.1 Compatibility

The request and response follow the publicly documented System One API of Jev, TypeSafe's decision-model service:
the same three question types, field names and published confidence formula. An application written against that API
uses the same request and response fields with DecisiveLM. Section 4.10 gives DecisiveLM's own HTTP endpoint and
authentication. DecisiveLM is an independent implementation built only from that public documentation. It shares no
code, weights or outputs with Jev. `media` is an extension of the format. Chapter
[16](16-provenance-licences-and-responsible-use.md) sets out the clean-room rule this project follows.

## 4.2 A first example

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/<bundle>")     # a general head (E4B or 26B A4B) or your own, on its backbone

response = dlm.system_one({
    "model": "decisivelm-e4b-general",
    "state": "Our API started returning 500s 20 minutes ago; we can't process orders.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payment or subscription issues",
                                    "technical": "Bugs or integration problems",
                                    "sales": "Pricing or account questions"}},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
        "frustration": {"type": "score", "instructions": "How frustrated does the customer appear?",
                        "criteria": ["Calm, just stating facts", "Frustrated but civil",
                                     "Very angry, strong language"]},
    },
})
response["answers"]["department"]["choice"]      # e.g. "technical"
```

A response to this request has the following shape. The numbers are illustrative, and section 4.9 works through the
arithmetic.

```json
{
  "model": "decisivelm-e4b-general",
  "answers": {
    "department":  {"type": "choice", "choice": "technical", "confidence": 0.94,
                    "probabilities": {"billing": 0.01, "technical": 0.96, "sales": 0.03}},
    "is_urgent":   {"type": "noul", "noul": 0.97},
    "frustration": {"type": "score", "score": 1.12, "confidence": 0.67,
                    "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil",
                               "2": "Very angry, strong language"},
                    "probabilities": {"0": 0.05, "1": 0.78, "2": 0.17}}
  },
  "usage": {"input_tokens": 412, "blocks": 7, "latency_ms": 85.0, "confidence_method": "peak",
            "adapter": "<bundle>"}
}
```

## 4.3 The request

| Field | Type | Required | Meaning |
|---|---|---|---|
| `model` | string or null | no | In Python, a label for the caller's own bookkeeping, echoed in the response and with no effect on the answers. The HTTP server (section 4.10) uses it to choose which bundle (the directory a head is saved in) answers. |
| `state` | string or JSON object | yes | What the questions are about: a message, a document, a record, a transcript. |
| `media` | list of media items | no (default `[]`) | Images and audio clips that belong to the state (section 4.5). |
| `questions` | object `{question_id: question}` | yes, at least one | The questions to answer. |

Unknown fields are rejected at every level of the request. A misspelt key such as `"critera"` fails validation and is
never silently ignored.

**The state.** A string is used as it is. A JSON object is serialised compactly (no spaces after `,` and `:`), in the
order its keys were written, with non-ASCII characters kept as they are. Keys are never sorted, so the model sees the
object exactly as you built it. A JSON array is not a valid state. Wrap it in an object (`{"items": [...]}`).

**Long states.** States longer than 8,192 tokens (`decisivelm.preprocess.MAX_STATE_TOKENS`) are not rejected. They are
cut to the first 70% and the last 30% of the budget, joined by a marker line (`[…]`). The labelling pipeline uses the
same function (`truncate_state`), so a model is always trained on states cut the same way it will see them.

**Question ids** are the keys of `questions`. They identify answers in the response and are **never shown to the
model**. Renaming `department` to `q1` does not change any answer, so the `instructions` must carry the whole
question.

**Instructions and criteria** may each be a string, a JSON object or a JSON array, for example
`{"what": "...", "examples": ["...", "..."]}`. Objects and arrays reach the model as compact JSON in insertion order.

## 4.4 The three question types

There are three question types and no others. Any other `type` is rejected.

### Choice: which one of these options?

```json
{"type": "choice",
 "instructions": "Which team should handle this?",
 "criteria": {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems"}}
```

- `criteria` is an object mapping each **option name** to its **description**. It must have **2 to 255 options**.
  Names are unique because they are object keys. In JSON text a repeated key keeps only its last value, so check
  generated requests for duplicates.
- The options are unordered. Every option is judged on its own (chapter [6](06-how-it-works.md)), and the answer for an
  option does not depend on where it appears in the object.
- An empty description is allowed when the name says everything (`"criteria": {"yes": "", "no": ""}`).

### Score: which level on an ordered scale?

```json
{"type": "score",
 "instructions": "How frustrated does the customer appear?",
 "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]}
```

- `criteria` is an array of **2 to 10 level descriptions**, from lowest to highest. The levels are numbered `0` to
  `n − 1` by their position in the array.
- A level never sees its number or its neighbours, so describe each level fully, in words that place it on the scale
  without the others (section 5.5). The order matters only in the answer, where `score` is the expected level.

### Noul: is this statement true?

A Noul (the published API's "continuous boolean") is a statement about the state. It returns the probability that
the statement is true, and no `confidence`. The model is trained on the share of raters who said yes, so for a graded
statement such as urgency the value is the chance that a rater would say yes, not the degree of urgency.

```json
{"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}
{"type": "noul", "instructions": "The customer asks for a refund",
 "criteria": "Asking about the refund policy without requesting one does not count"}
{"type": "noul", "instructions": "The reviewer recommends the product",
 "criteria": {"yes": "They say others should buy it", "no": "They advise against it or do not say"}}
```

`criteria` takes one of four forms:

| Form | Meaning |
|---|---|
| absent or `null` | The statement alone. |
| a string | Extra criteria for judging the statement. |
| `{"yes": ..., "no": ...}` | What a yes and a no each mean. Either key may be left out. |
| `{"true": ..., "false": ...}` | The same, accepted as aliases of `yes` and `no`. |

## 4.5 Media

`media` lists images and audio clips that belong to the state. They enter the model before the state text.

```json
"media": [{"type": "image", "path": "photos/receipt.png"},
          {"type": "image", "data": "<base64-encoded PNG or JPEG>"},
          {"type": "audio", "path": "calls/0412.wav"}]
```

- Each item has `type` and **exactly one** of `path`, `data` (base64) or `url`.
- At most **4 images** and **4 audio clips** per request, placed before the state in the order given.
- A relative `path` resolves against the instance's `media_root` (`DecisiveLM.load(..., media_root=...)`).
- A `url` is refused, because the model never fetches anything. Fetch the file yourself and pass it as `path` or
  `data`.
- Every image of a request is encoded at one fixed budget of soft tokens (the vectors the vision encoder places in the
  input where text tokens would otherwise go). The budget is 280 by default, and Gemma 4 offers 70, 140, 280, 560 or
  1120. A larger budget keeps more of the image's detail and adds as many tokens to the request (section 6.10). The
  same budget is used for labelling, training and inference.
- Images need an instance with a vision path: a Gemma 4 backbone plus its image processor. `DecisiveLM.load` sets it
  up. After `attach_bundle`, set `dlm.image_processor = decisivelm.model.load_image_processor("e4b")` yourself
  (`"26b-a4b"` for a 26B A4B head).
- **Audio** (Gemma 4 E4B only, because 26B A4B has no audio encoder): any format libsndfile reads (WAV, FLAC, OGG, MP3),
  mixed down to mono, resampled to 16 kHz and cut at 30 seconds. Each 40 ms becomes one soft token, so a 10-second
  clip adds about 250 tokens. It needs the `audio` extra (`uv sync --extra audio`). `DecisiveLM.load` sets up the
  feature extractor. After `attach_bundle`, set
  `dlm.feature_extractor = decisivelm.model.load_feature_extractor("e4b")` yourself.

## 4.6 What is rejected and what is only a warning

**Rejected** with a pydantic `ValidationError`, raised by `system_one` before the model runs:

- a question `type` other than `choice`, `score` or `noul`;
- a Choice with fewer than 2 or more than 255 options;
- a Score with fewer than 2 or more than 10 levels;
- a request with no questions, or with a state that is neither a string nor an object;
- an unknown field anywhere in the request;
- a media item with no source, or with more than one;
- more than 4 images or more than 4 audio clips.

**Warned about, not rejected.** `decisivelm.api_types.lint_request` checks each Choice for an `other`-style option
(`other`, `none`, `none of the above`, `n/a`, …) whose description is empty, is itself a bare `other`, or has fewer than
four words. Such an option is still scored, but it is likely to be scored badly, because it never sees the options it is
meant to be "other than". A question of a type the loaded bundle was never trained on is also answered with a warning
(chapter 9). After each call the warnings are on the instance:

```python
dlm.system_one(request)
dlm.warnings   # ["department: option 'other' has no self-contained description; say what it covers, ..."]
```

Chapter [5](05-writing-good-questions.md) shows how to write these options so that they work.

## 4.7 The response

| Field | Meaning |
|---|---|
| `model` | In Python, echoed from the request (`null` if it had none). Over HTTP, the name of the bundle that answered (section 4.10). |
| `answers` | One answer per question, under the same question id. |
| `usage.input_tokens` | Tokens in the packed sequence: the shared prefix, every question segment and every option, level or statement. |
| `usage.blocks` | Answerable units scored: one per Choice option, one per Score level, one per Noul. |
| `usage.latency_ms` | Wall-clock time of the call, in milliseconds. |
| `usage.confidence_method` | The confidence statistic used (section 4.8). |
| `usage.adapter` | The head that produced the answers, by the name of its bundle directory. |

**Choice answer.** `probabilities` maps every option name to a probability. `choice` is the option with the highest
probability (the first such option, in request order, if two are exactly equal). `confidence` is computed from
`probabilities` (section 4.8).

**Score answer.** `probabilities` maps every level number, as a string (`"0"`, `"1"`, …), to a probability. `score`
is the expected level, `Σ i · pᵢ`, a fractional number between 0 and `n − 1`. `legend` repeats each level's
description under its number, so the response can be read on its own. `confidence` is computed as for Choice.

**Noul answer.** `noul` is the probability that the statement is true, between 0 and 1. A Noul has no `confidence`
and no `probabilities`, because its one number is already the full answer.

**Conformance.** Every response satisfies these guarantees, and the test suite checks them:

- `probabilities` has exactly the request's option names or level numbers, no more and no fewer;
- each Choice or Score distribution sums to 1 (to floating-point precision);
- `score` lies in `[0, n − 1]`;
- `noul` lies in `[0, 1]`, with no `confidence`;
- `choice` is the argmax of `probabilities`.

**Calibration.** Every probability is scaled by the bundle's temperatures, one per question type and number of
options (and, for Nouls, a bias). A temperature divides the head's raw scores before they become probabilities. A
temperature above 1 spreads probability out, one below 1 sharpens it, and neither changes which answer is on top. The
temperatures are fitted on human-labelled data so that stated probabilities match how often answers are right. The
general head's were fitted on held-back questions of the tasks it was trained on, and a fine-tune's are fitted on your
`gold_calib` rows during training (chapter 9). On a new task the general head is usually underconfident until you
calibrate it on about 100 of your own labelled items (section 10.6). Chapter [10](10-calibration.md) covers fitting
and refitting.

**Independence.** Each answer depends only on the state, its own question and its own options. Adding, removing or
renaming other questions, or reordering a Choice's options, does not change it. Chapter [6](06-how-it-works.md)
explains why.

## 4.8 Confidence

`confidence` summarises how peaked a Choice or Score distribution is: 1.0 means all probability on one answer, and 0.0
means probability spread evenly. `decisivelm/confidence.py` provides four statistics, for `n` options with
probabilities `p`:

| Method | Formula | Notes |
|---|---|---|
| `peak` (default) | `clip((n · p_max − 1) / (n − 1), 0, 1)` | The published System One formula. 1 when `p_max = 1`, 0 when `p` is uniform. |
| `entropy` | `1 − H(p) / log n`, with `H(p) = −Σ pᵢ log pᵢ` | Uses the whole distribution, not only its top value. |
| `margin` | `p_top1 − p_top2` | Small when two answers are close. |
| `ordinal` | `1 − Var(level) / ((n − 1)² / 4)` | Treats the options as a scale in the order given, so use it for Scores, or for Choices whose options are listed in order. Mass on neighbouring levels costs less than mass at both ends of the scale. |

The method is set per instance, and applies to every Choice and Score answer:

```python
from decisivelm.infer import DecisiveLM
dlm.confidence_method = "entropy"          # after attach_bundle; or DecisiveLM(..., confidence_method="entropy")
```

`ordinal` treats option order as a scale, so it is only meaningful when every Choice in the request is also ordered,
or when the request has only Scores. `usage.confidence_method` records the method used.

To choose which answers a person should check, rank them by the top probability, `max(probabilities.values())`. The top
probability is the chance that the answer is right once the head is calibrated for the task, and it is the signal
chapter [11](11-evaluation.md) measures against human agreement. For one question, `peak` ranks answers in the same
order, because it rescales the top probability by the number of options. Across questions with different numbers of
options the two rank differently, and the other methods can rank differently from either. For a Noul,
`max(noul, 1 − noul)` plays the same role.

## 4.9 Worked example

The response in section 4.2, checked by hand:

- **department** (Choice, `n = 3`): `p_max = 0.96`, so `peak = (3 × 0.96 − 1) / (3 − 1) = 1.88 / 2 = 0.94`. The
  answer is `technical`, the argmax.
- **frustration** (Score, `n = 3`):
  - `score = 0 × 0.05 + 1 × 0.78 + 2 × 0.17 = 0.78 + 0.34 = 1.12`, a little above "Frustrated but civil";
  - `peak = (3 × 0.78 − 1) / 2 = 1.34 / 2 = 0.67`;
  - the other methods for comparison: `margin = 0.78 − 0.17 = 0.61`; `ordinal`: the variance of the level is
    `0.05 × 1.12² + 0.78 × 0.12² + 0.17 × 0.88² ≈ 0.206`, against a maximum of `(3 − 1)² / 4 = 1`, so
    `ordinal ≈ 0.79`.
- **is_urgent** (Noul): `noul = 0.97`, with no confidence field.
- **usage.blocks** = 3 options + 3 levels + 1 statement = **7**.

## 4.10 Calling DecisiveLM over HTTP

`decisivelm/serve.py` serves the same requests over HTTP, so other programs and other machines can call a model
without loading it themselves. One server holds one backbone and any number of bundles trained on it (the general
head and your fine-tunes). Each request picks one by name in its `model` field, or gets the default.

```bash
uv sync --extra serve
DECISIVELM_BUNDLES=bundles/my-task DECISIVELM_MEMORY_GB=20 uv run python -m decisivelm.serve   # 127.0.0.1:8003
```

| Method and path | What it does |
|---|---|
| `GET /health` | `200` once the model is loaded; needs no key, for health probes |
| `GET /v1/models` | The bundles this server holds, and the default |
| `POST /v1/systemone` | A request as in this chapter; the response as in 4.7, with `model` set to the bundle that answered |

```bash
curl -s http://127.0.0.1:8003/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "my-task", "state": "My card was declined twice today.",
       "questions": {"angry": {"type": "noul", "instructions": "The customer is angry"}}}'
```

Settings come only from environment variables (the full list is in the module's docstring):

- **What it serves**: `DECISIVELM_BUNDLES` is a bundle, a folder of bundles, or several joined by commas. Every bundle
  must be a LoRA bundle (one with an `adapter.pt`) with the same `rank`, `alpha` and `target_regex` under `lora` in its
  `calibration.json`, on the same backbone revision. The server refuses to start otherwise. `DECISIVELM_DEFAULT` names
  the default.
- **Where it listens**: `DECISIVELM_HOST` (default `127.0.0.1`, or several addresses joined by commas) and
  `DECISIVELM_PORT` (default 8003). With `DECISIVELM_AUTH=none`, the server refuses to start on any address other than
  the loopback addresses (`127.0.0.1`, `::1`, `localhost`). Override this with `DECISIVELM_ALLOW_UNAUTHENTICATED=1`
  only where nobody but trusted callers can reach that address.
- **Limits**: `DECISIVELM_MEMORY_GB` caps the process's GPU memory, so an oversized request fails in this server, not in
  another program on the same GPU. `DECISIVELM_MAX_TOKENS` (default 12,288) refuses longer requests with `413`. A state
  is cut at 8,192 tokens (4.3), so the default leaves about 4,000 tokens for the questions and their options. A request
  at the default limit peaks at about 18.6 GiB on E4B, inside a 20 GB `DECISIVELM_MEMORY_GB` cap. The 26B A4B general
  head, which computes in float32, peaks at 68.6 GB at that limit (57.2 GB after loading), so give it
  `DECISIVELM_MEMORY_GB=72`. It answers a 20-block request in about 0.47 s over HTTP, against about 0.10 s for the E4B
  head, and the server refuses audio for it with `400`.
- **Who may call it**: `DECISIVELM_AUTH` is `none` (the default, for a server only this machine can reach), `keys` (a
  `Bearer` token from `DECISIVELM_API_KEYS`), or `litellm` (any key a LiteLLM proxy at `DECISIVELM_LITELLM_URL` accepts,
  so the people who already have a key for that proxy can call this server through it).

Differences from calling `system_one` in Python:

- **Media must be base64** `data` (plain, or as a `data:image/png;base64,…` URL). A server never reads a file path
  from a request (`400`), and data that is not base64 or not a readable image or sound file is a `400` too.
- **Errors** are JSON, `{"error": {"message": ..., "type": ...}}`: `400` for an invalid request, `401` for a missing or
  wrong key, `404` for an unknown bundle, `413` for a request over the token limit, `503` when the GPU is out of memory
  for this request or the key service is down.
- **Warnings** (4.6) come back in the `X-DecisiveLM-Warnings` header, as a JSON list.
- Requests are answered **one at a time** on the GPU. Switching bundle between requests copies the new bundle's LoRA
  weights into the shared backbone, about 70 MB for an E4B bundle.

The `Dockerfile` at the repository root builds the same server as an image: mount a Hugging Face cache holding the
backbone at `/models/hub` (read offline) and your bundles at `/bundles`, set `DECISIVELM_BUNDLES=/bundles`, and give
the container one GPU. The container listens on all its interfaces, so it starts only with `DECISIVELM_AUTH` set, or
with `DECISIVELM_ALLOW_UNAUTHENTICATED=1` when its port is published to this machine alone
(`docker run -p 127.0.0.1:8003:8003 …`). The comment at the top of the `Dockerfile` has a full `docker run` line.

Next: [5. Writing good questions](05-writing-good-questions.md)
