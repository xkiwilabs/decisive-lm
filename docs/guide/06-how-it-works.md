# 6. How it works

DecisiveLM turns a Gemma 4 language model into a decision model. It never generates text. It reads the state and
every possible answer together in one forward pass, and scores each answer with a small network, the scorer. This
chapter describes the architecture:

- how a request becomes a packed token sequence;
- how attention is masked so that answers stay independent;
- what the scorer computes, and what is trained;
- what a saved adapter bundle contains.

## 6.1 Blocks and query tokens

Every answerable unit of a request is a **block**:

- a Noul is 1 block (its statement);
- a Choice with `n` options is `n` blocks, one per option;
- a Score with `n` levels is `n` blocks, one per level.

Each block ends with a **query token**: one of three reserved Gemma 4 token ids, one per question type (`choice` = 6,
`score` = 7, `noul` = 8; `configs/model.yaml`).

- These ids are unused entries in Gemma 4's vocabulary (`<unused0>` to `<unused2>`). The tokenizer never produces
  them from text, so they are inserted as ids, never as strings.
- The vocabulary is **not resized**. A resize would also have to grow the tied embedding and, on E-series models, the
  per-layer embeddings that are sized to the original vocabulary.
- The query token tells the backbone which kind of judgement it is making. Its embedding row is trained (section 6.6).

The backbone's final **hidden state** at each query token (the vector the last layer produces at that position) is
passed to the scorer, which turns it into **one number**, a logit (a raw score that can be any real number). A
question's answer comes from grouping its blocks' logits and turning them into probabilities (section 6.5).

This is a **cross-encoder**: each option is read together with the state by the full model, rather than embedded
separately and compared afterwards. It is more accurate than comparing separate embeddings, and together with the
attention masks (section 6.3) it makes answers independent by construction.

## 6.2 The packed tree

A request is laid out as a tree of token segments:

```
                         ┌─ [Q: department instructions] ─┬─ [Option: billing — …     <q_choice>]
                         │                                ├─ [Option: technical — …   <q_choice>]
                         │                                └─ [Option: sales — …       <q_choice>]
[<bos> images  State: …] ─┼─ (no segment) ──────────────────── [Statement: … urgency.  <q_noul>]
     shared prefix       │
                         └─ [Q: frustration instructions] ─┬─ [Level: Calm, …          <q_score>]
                                                           ├─ [Level: Frustrated …     <q_score>]
                                                           └─ [Level: Very angry, …    <q_score>]
```

- **Prefix**: the beginning-of-sequence token, then each image's placeholder tokens, then `State:` and the state
  text. It is computed once and shared by every question.
- **Question segment**: `Q:` and the question's instructions (`decisivelm/render.py`). A Noul has none. Its statement
  lives in its leaf.
- **Leaf**: one option (`Option: {name} — {description}`), one level (`Level: {description}`) or one statement
  (`Statement: … Yes means: … No means: …`), followed by the query token.

Question ids, option positions, level numbers and sibling options are never rendered. The text templates are versioned
(`RENDER_VERSION` in `decisivelm/render.py`), and a bundle records the version it was trained with.

**Physical layout and logical positions.** `decisivelm/tree.py` lays the tokens out one after another: the prefix,
then each question's segment followed by its leaves. Each token also gets a **logical position id** that follows the
tree rather than the layout:

- a question segment continues from the last position of the prefix;
- a leaf continues from the last position of its question segment;
- sibling leaves therefore **reuse the same positions**.

So each leaf sees positions exactly as if it were the only continuation of its question, which is how the language
model was pretrained to read text.

**Several requests in one pass.** `pack` can place several requests in one sequence for training or batched
inference. Requests never attend to each other (the masks are block-diagonal), and each request's positions restart
at 0.

## 6.3 Attention masks: two, not one

Inside the backbone, each token's hidden state is computed from the tokens it **attends to**, that is, the tokens
whose information it may read. An attention mask lists, for every token, the tokens it may attend to. In the tree, a
token may attend to:

- every token of its **ancestor segments** (a leaf sees its question segment and the prefix);
- the **earlier tokens of its own segment** (causal, as in normal language modelling).

It never sees a sibling leaf, another question or another request. This is the **global** mask (`ancestor_or_self`).

Gemma 4 has two kinds of attention layer. Most are **sliding-window** layers, in which a token sees only the previous
W positions. Every sixth layer, and the last, is a **global** layer that sees everything allowed. W is read from the
model configuration: 512 on E4B, 1024 on 26B A4B. DecisiveLM therefore builds two masks
(`decisivelm.tree.attention_masks`):

- **global** (`full_attention`): allowed by the tree;
- **sliding** (`sliding_attention`): allowed by the tree and `pos[query] − pos[key] < W`, measured in logical
  positions.

Both are passed to the backbone as a mapping keyed by layer type (`hf_masks`). A single custom mask would be applied
unchanged to every layer, which silently turns the sliding layers into full attention. The model would still run and
still return numbers, but its sliding layers would see far more context than they were trained on, and its answers
would degrade. The window is also measured in logical positions, not physical ones. A leaf placed physically after
many siblings is still logically right after its question, and must see the end of the state exactly as it would
alone.

On 26B A4B, the soft tokens of one image (the slots that carry the vision encoder's output for that image, section 6.10)
also attend to each other in both directions on sliding layers, as Gemma 4 does natively. The sliding mask reproduces
this (`bidirectional_images`).

The test suite checks the masks directly:

- `tests/test_tree.py` checks positions, ancestry, causality and the logical window;
- `tests/test_mask_equivalence.py` checks that the packed tree gives the same hidden states as running each
  root-to-leaf path on its own, and that one mask for every layer is detected as wrong.

## 6.4 Hidden states, not logits

`decisivelm.model.load_backbone` returns the `Gemma4Model` inside `Gemma4ForConditionalGeneration`. That module
merges images and per-layer embeddings and returns `last_hidden_state`, but never computes the language-model head.
DecisiveLM gathers the hidden states at the query positions only. The full vocabulary projection (about 262,000
outputs per token) is never computed, and intermediate layers' states are never kept. Only the final states at the
query positions are needed, so skipping the rest saves memory and time.

## 6.5 The scorer and the readout

The scorer (`decisivelm.head.Scorer`) maps one hidden state to one logit:

```
hidden state → per-dimension z-score → LayerNorm → Linear(hidden, 1024) → GELU → Linear(1024, 1) → logit
```

- **The z-score** subtracts a per-dimension mean and divides by a per-dimension standard deviation. Both are measured
  once on training query states (`Scorer.standardize`) and saved with the scorer. Gemma 4's final states contain a few
  dimensions with very large activations. LayerNorm rescales each vector by its overall spread, so these dimensions
  would otherwise set that scale and squash every other dimension.
- One scorer is shared by all three question types. The query token tells the backbone which type it is scoring.

**Readout** (`decisivelm.head.readout`) groups a question's logits into its answer:

| Type | Probabilities |
|---|---|
| Choice | `softmax(logits / T)` over the question's options |
| Score | `softmax(logits / T)` over its levels; `score = Σ i · pᵢ` |
| Noul | `sigmoid((logit + b_noul) / T)` |

A **softmax** turns a question's logits into probabilities that sum to 1, and the largest logit gets the largest
share. A **sigmoid** turns one logit into a probability between 0 and 1.

`T` is a **temperature**. It is fitted on human-labelled calibration data per question type and option-count bucket
(2, 3–5, 6–15, 16–63, 64–255 options), with a pooled value per type for buckets that have too little data. `b_noul` is
a bias added to every Noul's logit, which shifts how readily the model says yes. Choice and Score need no bias,
because adding the same number to all of a question's logits leaves a softmax unchanged. Temperatures change how
peaked a distribution is, never its ranking. Chapter [10](10-calibration.md) describes the fitting.

## 6.6 What is trained

The backbone's own weights are never changed. Training adds three small parts:

1. **LoRA adapters** (`decisivelm/lora.py`). LoRA (low-rank adaptation) adds a small trainable correction `B·A` to a
   frozen weight matrix, scaled by `alpha / rank`.
   - Adapters go on the language-model decoder's attention and MLP (multilayer perceptron, the feed-forward block of
     each layer) projections (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) only, chosen by a regular expression on
     module paths (`lora_target_regex` in `configs/model.yaml`).
   - The expression is anchored on the language model, so the vision and audio encoders' identically named
     projections are left alone.
   - On 26B A4B the expert layers (128 feed-forward sub-networks, of which each token uses 8) and the router (the small
     layer that picks those 8) stay frozen. LoRA adapts attention and the dense MLP that every layer runs beside the
     experts.
   - `B` starts at zero, so a new adapter reproduces the base model exactly. Adapter weights are kept in fp32 (32-bit
     floating point). The backbone runs in bf16 by default, a 16-bit format with the same range as fp32 but fewer
     digits of precision.
2. **Query-token rows** (`QueryRows`): a trainable vector per query token, added to the embedding output wherever
   that token appears. It starts at zero.
3. **The scorer** (section 6.5).

LoRA moves the judgement "how well does this option fit this state" into the hidden state at the query token. A frozen
chat model's state there was never trained to encode it. How much LoRA adds over a scorer trained on the frozen backbone
alone (P1, [chapter 9](09-fine-tuning-for-your-task.md), section 9.4) depends on the task and the backbone. On E4B it
raised Civil Comments AUROC from 0.875–0.884 to 0.954 but moved Banking77 accuracy only from 0.887–0.900 to 0.902. Each
range is three seeds of the scorer alone, against one LoRA run (seed 0), and both measures are defined in chapter 11. On
26B A4B it raised agreement with human coders on the coding study (Cohen's κ, agreement corrected for chance, section
11.6) from κ 0.699 to 0.805.

## 6.7 Losses

Training adjusts the trained parts to make a **loss** smaller. The loss is one number that measures how far the
model's probabilities are from each question's **target**, the probabilities it should give. Training targets are
**probability distributions**, not single labels (`decisivelm/losses.py`):

- **human targets** are the share of raters who chose each option or level, or, for a Noul, the share who said yes;
- **teacher targets** are a teacher model's calibrated probabilities (chapter [12](12-teacher-labelling.md));
- a question uses its human target if it has one and otherwise its teacher target, and a question with neither is
  left out of the loss.

Training on rater shares rather than a majority vote lets confidence follow human disagreement, because an item on
which raters split teaches the model to be unsure.

| Type | Loss |
|---|---|
| Choice | soft cross-entropy against the target distribution, divided by `log n` |
| Score | the same, plus 0.5 × EMD² (earth mover's distance), the mean squared difference between the predicted and target cumulative distributions |
| Noul | binary cross-entropy against the target share |

- **Dividing by `log n`** keeps a 255-option question from outweighing a Noul.
- **The EMD term** exists because levels never see each other. It is the only place the model learns that mass on a
  neighbouring level is a smaller error than mass at the far end of the scale.
- Losses are averaged per question, then per request.

## 6.8 Independence and order, by construction

Because of the masks:

- an option's logit depends only on the state, its question's instructions and its own text;
- adding or removing a question, renaming a question id, or reordering a Choice's options leaves every other logit
  unchanged.

The only numerical effect of a different packing is floating-point rounding, from different kernel shapes and summation
orders. In bf16 this is large enough to flip an answer whose top two options are close (section 11.7 gives the measured
sizes, and section 6.9 the larger effect on 26B A4B in bf16). In fp32 it is negligible.

This is tested at two levels:

- **in the test suite** (`tests/test_system_one.py`, `tests/test_invariance.py`), on a tiny random Gemma 4 in fp32:
  adding and removing questions, shuffling options, renaming question ids, and packing another request first must not
  change the answers;
- **on a trained bundle and the real backbone**, with `scripts/check_invariance.py`. It reports, for each question type
  and packing change (options reversed for Choice, another request packed first, one more question): the largest change
  in any probability, answer flips, rank stability of the top probability, and churn in the least confident 20% of
  answers. It also runs the identical pass twice, as a determinism check.

## 6.9 Backbones

| | **E4B** (default) | **26B A4B** |
|---|---|---|
| Model | `google/gemma-4-E4B-it` | `google/gemma-4-26B-A4B-it` (mixture of experts) |
| Size | about 4B effective parameters | 25.2B total, about 3.8B active per token |
| Hidden size | 2560 | 2816 |
| Sliding window W | 512 | 1024 |
| Input | text, images, audio | text, images |

- Each backbone has a profile in `configs/model.yaml`: model id, pinned revision, hidden size, layer pattern, window,
  modalities, image budget and LoRA expression. Code reads these values and never assumes them.
- A bundle is trained for one backbone and records that backbone's profile, including the revision.

**Numerics on 26B A4B.** In a mixture-of-experts model each token is routed to its top 8 of 128 experts. In bf16,
the rounding differences between two packings of the same request are enough to change some routing decisions. The
change then grows through later layers, so answers can move more than on a dense model. Loading the backbone in
float32 removes nearly all of it. On 200 coding questions, the answers that flipped when only the packing changed fell
from 38–44 to 0 (`results/invariance.md`). To load it in float32:

```python
import torch
from decisivelm.model import load_backbone
backbone, tokenizer = load_backbone("26b-a4b", dtype=torch.float32)
```

This keeps the checkpoint's bf16 weights (about 56 GiB of GPU memory once loaded, within the 72 GiB free that chapter 2
asks for) and runs every matrix multiplication in fp32 (`decisivelm.model.fp32_compute`). The released checkpoint is
itself bf16, so converting its weights to fp32 for each multiplication loses nothing, and the answers equal those of a
full fp32 copy of the model, which would need about 104 GB. Use it wherever answers must not depend on what else is in a
batch, and measure the remaining effect on your own bundle with `scripts/check_invariance.py`. E4B is a dense model and
is less sensitive to this. In bf16, repacking flipped 3–4.5% of its answers, against 19–22% on 26B A4B
(`results/M0.md`). For this reason the published 26B A4B general head is trained, checked and served in float32, and its
bundle sets that when loaded.

## 6.10 Images

Images enter the shared prefix before the state text (`decisivelm/media.py`). Each image is processed by Gemma 4's
vision encoder into a run of soft tokens:

- an image-start token;
- one soft token per pooled patch group;
- an image-end token.

The vision encoder's outputs fill the soft-token slots in order. Every image of a request uses one fixed soft-token
budget (Gemma 4 offers 70, 140, 280, 560 or 1120; 280 by default). The same budget is used for labelling, training
and inference, so a model is never asked about an image at a resolution it was not trained on.

## 6.11 The adapter bundle

An **adapter bundle** is the unit that is saved, shipped and loaded: a directory under `bundles/`.

| File | Contents |
|---|---|
| `calibration.json` | The temperatures and `b_noul`; the backbone profile (model id and revision); the LoRA settings (`rank`, `alpha`, `dropout`, `target_regex`); `render_version`; the numeric type; the standardisation and image-budget settings; the training data path, adapter kind and seed; the selection rule and the training history. |
| `scorer.pt` | The scorer, including its z-score statistics. |
| `adapter.pt` | The LoRA `A` and `B` matrices and the query-token rows. Absent from a P1 bundle, whose scorer was trained on the frozen backbone (section 6.6). |

`DecisiveLM.load(bundle)` (`decisivelm.bundle.load_bundle`) loads the backbone a bundle records and attaches the bundle.
`decisivelm.bundle.attach_bundle(bundle, backbone, tokenizer)` attaches it to a backbone already loaded. Two rules:

- **Match the revision.** A bundle is only valid on the backbone it was trained on. `load_bundle` loads that revision,
  and `attach_bundle` refuses a backbone from `decisivelm.model.load_backbone` with a different model id or revision.
- **One bundle per backbone instance.** `attach_bundle` applies the LoRA layers in place, so in your own code attach
  each bundle to its own freshly loaded backbone. The server ([chapter 4](04-requests-and-responses.md), section
  4.10) holds several bundles on one backbone by attaching the LoRA layers once and, when a request names another
  bundle, copying that bundle's LoRA weights into them.

Next: [7. Using the general head](07-using-the-general-head.md)
