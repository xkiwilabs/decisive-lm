# 4. Requests and responses

Every call to DecisiveLM takes one **request**: a state plus a set of typed questions. It returns one **response**:
a constrained, calibrated answer to every question, from a single forward pass of the model. This chapter is the
complete reference for both: every field, every limit, what is rejected, what only raises a warning, and how the
numbers in a response are computed.

The models live in `decisivelm/api_types.py` (pydantic). `DecisiveLM.system_one` in `decisivelm/infer.py` accepts
either a plain `dict` or a `SystemOneRequest`, and returns a plain `dict` that has been validated against
`SystemOneResponse`.

## 4.1 Compatibility

The request and response follow the publicly documented System One API of TypeSafe's Jev: the same three question
types, the same field names and the same published confidence formula. An application written against that API can
switch to DecisiveLM with little code. DecisiveLM is an independent implementation built only from that public
documentation. It shares no code, weights or outputs with Jev. `media` is an extension of the format. Chapter
[16](16-provenance-licences-and-responsible-use.md) sets out the clean-room rule this project follows.

## 4.2 A first example

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/<bundle>")     # the general head or your own, on the backbone it was trained on

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

A response to this request has the following shape. The numbers are illustrative.

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

Section 4.7 works through the arithmetic.

## 4.3 The request

| Field | Type | Required | Meaning |
|---|---|---|---|
| `model` | string or null | no | A label for the caller's own bookkeeping. It is echoed in the response and has no effect on the answers. |
| `state` | string or JSON object | yes | What the questions are about: a message, a document, a record, a transcript. |
| `media` | list of media items | no (default `[]`) | Images that belong to the state (section 4.5). |
| `questions` | object `{question_id: question}` | yes, at least one | The questions to answer. |

Unknown fields are rejected at every level of the request. A misspelt key such as `"critera"` fails validation; it is
never silently ignored.

**The state.** A string is used as it is. A JSON object is serialised compactly (no spaces after `,` and `:`), in the
order its keys were written, with non-ASCII characters kept as they are. Keys are never sorted, so the model sees the
object exactly as you built it. A JSON array is not a valid state; wrap it in an object (`{"items": [...]}`).

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
- A level never sees its number or its neighbours. The order matters only in the answer: `score` is the expected
  level.

### Noul: is this statement true?

A Noul ("continuous boolean") is a statement that is more or less true of the state. It returns a probability, and
no `confidence`.

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

`media` lists images that belong to the state. They enter the model before the state text.

```json
"media": [{"type": "image", "path": "photos/receipt.png"},
          {"type": "image", "data": "<base64-encoded PNG or JPEG>"},
          {"type": "audio", "path": "calls/0412.wav"}]
```

- Each item has `type` and **exactly one** of `path`, `data` (base64) or `url`.
- At most **4 images** and **4 audio clips** per request, placed before the state in the order given.
- A relative `path` resolves against the instance's `media_root` (`DecisiveLM.load(..., media_root=...)`).
- A `url` is refused: the model never fetches anything. Fetch the file yourself and pass it as `data`.
- Every image of a request is encoded at one fixed soft-token budget (280 by default). The same budget is used for
  labelling, training and inference.
- Images need an instance with a vision path: a Gemma 4 backbone plus its image processor. `DecisiveLM.load` sets it
  up; after `attach_bundle`, set `dlm.image_processor = decisivelm.model.load_image_processor("e4b")` yourself.
- **Audio** (Gemma 4 E4B only; 26B A4B has no audio encoder): any format libsndfile reads (WAV, FLAC, OGG, MP3),
  mixed down to mono, resampled to 16 kHz and cut at 30 seconds. Each 40 ms becomes one soft token, so a 10-second
  clip adds about 250 tokens. It needs the `audio` extra (`uv sync --extra audio`); `DecisiveLM.load` sets up the
  feature extractor, and after `attach_bundle` set `dlm.feature_extractor =
  decisivelm.model.load_feature_extractor("e4b")`.

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
(`other`, `none`, `none of the above`, `n/a`, …) whose description is empty, is itself a bare `other`, or has fewer
than four words. Such an option cannot be judged: it never sees the options it is meant to be "other than". After
each call the warnings are on the instance:

```python
dlm.system_one(request)
dlm.warnings   # ["department: option 'other' has no self-contained description; say what it covers, ..."]
```

Chapter [5](05-writing-good-questions.md) shows how to write these options so that they work.

## 4.7 The response

| Field | Meaning |
|---|---|
| `model` | Echoed from the request (`null` if it had none). |
| `answers` | One answer per question, under the same question id. |
| `usage.input_tokens` | Tokens in the packed sequence: the shared prefix, every question segment and every option, level or statement. |
| `usage.blocks` | Answerable units scored: one per Choice option, one per Score level, one per Noul. |
| `usage.latency_ms` | Wall-clock time of the call, in milliseconds. |
| `usage.confidence_method` | The confidence statistic used (section 4.8). |
| `usage.adapter` | The bundle that produced the answers (its directory name). |

**Choice answer.** `probabilities` maps every option name to a probability. `choice` is the option with the highest
probability (the first such option, in request order, if two are exactly equal). `confidence` is computed from
`probabilities` (section 4.8).

**Score answer.** `probabilities` maps every level number, as a string (`"0"`, `"1"`, …), to a probability. `score`
is the expected level, `Σ i · pᵢ`, a fractional number between 0 and `n − 1`. `legend` repeats each level's
description under its number, so the response can be read on its own. `confidence` is computed as for Choice.

**Noul answer.** `noul` is the probability that the statement is true, between 0 and 1. A Noul has no `confidence`
and no `probabilities`: its one number is already the full answer.

**Conformance.** Every response satisfies these guarantees, and the test suite checks them:

- `probabilities` has exactly the request's option names or level numbers, no more and no fewer;
- each Choice or Score distribution sums to 1 (to floating-point precision);
- `score` lies in `[0, n − 1]`;
- `noul` lies in `[0, 1]`, with no `confidence`;
- `choice` is the argmax of `probabilities`.

**Calibration.** All probabilities are calibrated: the bundle's temperatures (and, for Nouls, a bias) have been
applied, so that the stated probabilities match how often answers are right on human-labelled data. See chapter
[10](10-calibration.md).

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
| `ordinal` | `1 − Var(level) / ((n − 1)² / 4)` | For Score only. Mass on neighbouring levels costs less than mass at both ends of the scale. |

The method is set per instance, and applies to every Choice and Score answer:

```python
from decisivelm.infer import DecisiveLM
dlm.confidence_method = "entropy"          # after attach_bundle; or DecisiveLM(..., confidence_method="entropy")
```

`ordinal` treats option order as a scale, so it is only meaningful when every Choice in the request is also ordered,
or when the request has only Scores. `usage.confidence_method` records the method used.

For deciding which answers a person should check, the top probability (`max(probabilities.values())`) is also a good
signal. Chapter [11](11-evaluation.md) measures how well each signal tracks human agreement. For a Noul,
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

Next: [5. Writing good questions](05-writing-good-questions.md)
