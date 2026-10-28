"""M0 checks on a real Gemma 4 backbone (the M0 bring-up checks). Prints a report for results/M0.md.

    uv run python scripts/m0_real.py [--backbone e4b|26b-a4b] [--device cuda|mps|cpu] [--fp32-device cpu] [--skip-latency]

On CUDA the model is spread over every visible GPU. fp32 26B A4B is ~104 GB: with one free GPU, run the fp32 mask
check in host RAM (--fp32-device cpu) and the bf16 sections on the GPU. 26B A4B makes image tokens
bidirectional on sliding layers; tree.attention_masks builds that (bidirectional_images), and the path run alone
gets the stock vision masks through mm_token_type_ids.

1. Config facts: layer-type pattern, sliding window W, KV sharing, attention backends.
2. Reserved query-token ids: genuinely unused?
3. Mask equivalence in fp32 with one image, a prefix longer than W, and three
   branches under one question whose combined length exceeds W. On MoE backbones the
   masks are tested with expert selection forced to the packed run's (RoutingProbe);
   free-routing error is reported beside it as noise.
4. bf16 noise floor: the same request at different packing positions.
5. First latency number: 1k-token text state, 20 blocks, batch 1, bf16.
"""
import argparse, statistics, time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from transformers.models.gemma4 import image_processing_gemma4

from decisivelm.media import EncodedImage
from decisivelm.model import backbone_profile, hub_kwargs
from decisivelm.tree import Question, TreeRequest, attention_masks, hf_masks, pack

QUERY_IDS = {"choice": 6, "score": 7, "noul": 8}     # <unused0..2>
TOL = 1e-3

TEXT = (
    "The warehouse in Newcastle received {n} pallets of ceramic tiles on Tuesday, of which {m} "
    "arrived cracked. The supplier's contract allows returns within 14 days if damage is reported "
    "with photographs. Shift supervisor notes mention a forklift fault on dock {d}. "
)


def text_ids(tok, n_tokens, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_tokens:
        out += tok.encode(TEXT.format(n=rng.integers(10, 99), m=rng.integers(1, 9), d=rng.integers(1, 6)),
                          add_special_tokens=False)
    return out[:n_tokens]


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def forward(model, ids, mm, positions=None, masks=None):
    kw = dict(input_ids=torch.tensor([ids], device=model.device), **mm)
    if positions is not None:
        kw.update(position_ids=positions[None].to(model.device),
                  attention_mask={k: m.to(model.device) for k, m in masks.items()})
    return model(**kw).last_hidden_state[0]


class RoutingProbe:
    """MoE backbones: top-k expert selection is discontinuous, so fp32 summation-order noise (~1e-6) between a packed
    run and a path run alone flips near-tied experts and the difference grows to ~1e-2 by the last layers. To test
    masks and positions rather than routing, record each token's selection in the packed run, then in an alone run
    either count the tokens whose top-k set differs ("observe") or impose the packed selection on the matching tokens
    ("force"), with weights from the alone run's own router probabilities."""

    def __init__(self, layers):
        self.mode, self.packed, self.index, self.flips = "record", {}, None, {}
        for i, layer in enumerate(layers):
            if getattr(layer, "router", None) is not None:
                layer.router.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(router, inputs, out):
            probs, weights, idx = out
            if self.mode == "record":
                self.packed[i] = idx.clone()
                return None
            ref = self.packed[i][self.index.to(idx.device)]
            self.flips[i] = int((idx.sort(-1).values != ref.sort(-1).values).any(-1).sum())
            if self.mode == "observe":
                return None
            w = probs.gather(-1, ref)
            w = w / w.sum(-1, keepdim=True) * router.per_expert_scale[ref]
            return probs, w.to(weights.dtype), ref
        return hook


def path_index(req, tree):
    """For each (question, leaf), the packed physical index of every token of its root-to-leaf path run alone."""
    spans, at, out = iter(tree.leaf_spans), len(req.prefix), []
    for q in req.questions:
        segment = list(range(at, at + len(q.segment)))
        at += len(q.segment)
        for _ in q.leaves:
            s, e = next(spans)
            out.append(torch.tensor(list(range(len(req.prefix))) + segment + list(range(s, e))))
            at = e
    return out


def load(profile, dtype, device):
    kw = dict(revision=profile["revision"], dtype=dtype, attn_implementation="sdpa", **hub_kwargs())
    if device == "cuda":
        return Gemma4ForConditionalGeneration.from_pretrained(profile["model_id"], device_map="auto", **kw).eval()
    return Gemma4ForConditionalGeneration.from_pretrained(profile["model_id"], **kw).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="e4b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--fp32-device", help="device for the fp32 mask check (default: --device)")
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()

    profile = backbone_profile(a.backbone)
    print(f"# {profile['model_id']} @ {profile['revision']}, device {a.device}\n")
    processor = AutoProcessor.from_pretrained(profile["model_id"], revision=profile["revision"], **hub_kwargs())
    tok = processor.tokenizer
    full = load(profile, torch.float32, a.fp32_device or a.device)
    model = full.model                       # decoder-only path: no 262k-way lm_head (docs/guide/06 §6.4)
    tc = full.config.text_config
    W = tc.sliding_window

    print("## 1. Config")
    print(f"- layers {tc.num_hidden_layers}; types: " + ", ".join(
        f"{t} x{tc.layer_types.count(t)}" for t in sorted(set(tc.layer_types))))
    print(f"- sliding window W = {W}; KV-shared layers = {tc.num_kv_shared_layers}; "
          f"PLE dim = {tc.hidden_size_per_layer_input}; hidden = {tc.hidden_size}; "
          f"bidirectional = {tc.use_bidirectional_attention}")
    print(f"- supports flex attention: {full._supports_flex_attn}; sdpa: {full._supports_sdpa}")
    print(f"- image max_soft_tokens choices: {image_processing_gemma4._SUPPORTED_SOFT_TOKENS}; "
          f"default {processor.image_processor.max_soft_tokens}")

    print("\n## 2. Reserved query-token ids")
    emb = model.language_model.embed_tokens.weight.detach().float()
    norms = emb.norm(dim=1)
    for name, i in QUERY_IDS.items():
        print(f"- {name}: id {i} = {tok.convert_ids_to_tokens(i)!r}, row norm {norms[i]:.4f} "
              f"(vocab median {norms.median():.4f})")
    sample = tok.encode(TEXT * 50 + " <unused0> unused0 " + "".join(chr(c) for c in range(32, 1200)),
                        add_special_tokens=False)
    print(f"- emitted by the tokenizer on a 1,200-codepoint sample + literal '<unused0>': "
          f"{sorted(set(sample) & set(QUERY_IDS.values())) or 'none'}")

    print("\n## 3. Mask equivalence, fp32")
    bidir = bool(profile["bidirectional_image_attention"])
    img = Image.fromarray((np.random.default_rng(0).random((448, 448, 3)) * 255).astype(np.uint8))
    enc = processor(text=[f"Screenshot: {processor.image_token}\nTicket text:"], images=[[img]],
                    return_tensors="pt")
    mm = {k: v.to(model.device) for k, v in enc.items()
          if k in ("pixel_values", "image_position_ids")}
    prefix = enc["input_ids"][0].tolist() + text_ids(tok, int(1.4 * W), 1)     # lengths scale with W
    n_image = sum(t == processor.image_token_id for t in prefix)
    start = prefix.index(processor.image_token_id)
    image = EncodedImage(token_ids=prefix[start - 1:start + n_image + 1], pixel_values=enc["pixel_values"][0],
                         position_ids=enc["image_position_ids"][0], budget=processor.image_processor.max_soft_tokens)
    req = TreeRequest(prefix=prefix, images=[image], image_starts=[start], questions=[
        Question(segment=text_ids(tok, 40, 2),
                 leaves=[text_ids(tok, n, 10 + n) + [QUERY_IDS["choice"]]
                         for n in (int(0.37 * W), int(0.45 * W), int(0.31 * W))]),
        Question(segment=[], leaves=[text_ids(tok, 60, 3) + [QUERY_IDS["noul"]]]),
    ])
    branch_total = sum(len(l) for l in req.questions[0].leaves)
    print(f"- prefix {len(prefix)} tokens ({n_image} image soft tokens"
          f"{', bidirectional on sliding layers' if bidir else ''}) > W = {W}; "
          f"question 1 branches total {branch_total} > W")
    tree = pack([req])
    probe = RoutingProbe(model.language_model.layers) if profile.get("moe") else None
    modes = ["force", "observe"] if probe else [None]
    worst = {m: [0.0, 0.0] for m in modes}
    first_flip = None
    with torch.no_grad():
        packed = forward(model, tree.input_ids.tolist(), mm, tree.position_ids,
                         hf_masks(attention_masks(tree, W, bidir), torch.float32))
        spans, index = iter(tree.leaf_spans), iter(path_index(req, tree))
        for q in req.questions:
            for leaf in q.leaves:
                head = req.prefix + q.segment
                s, e = next(spans)
                if probe:
                    probe.index = next(index)
                for m in modes:
                    if probe:
                        probe.mode, probe.flips = m, {}
                    path = head + leaf
                    stock = {"mm_token_type_ids": torch.tensor([[int(t == processor.image_token_id) for t in path]],
                                                               device=model.device)} if bidir else {}
                    alone = forward(model, path, {**mm, **stock})
                    worst[m][0] = max(worst[m][0], rel_err(packed[s:e], alone[len(head):]))
                    worst[m][1] = max(worst[m][1], rel_err(packed[e - 1], alone[-1]))
                    if m == "observe":
                        flipped = [i for i, n in sorted(probe.flips.items()) if n]
                        if flipped and (first_flip is None or flipped[0] < first_flip):
                            first_flip = flipped[0]
    state, query = worst[modes[0]]
    verdict = "PASS" if max(state, query) < TOL else "FAIL"
    if probe:
        print(f"- masks (expert selection forced to the packed run's): worst leaf {state:.2e}, query {query:.2e} "
              f"(threshold {TOL:.0e}): {verdict}")
        print(f"- free routing (noise, not a mask test): worst leaf {worst['observe'][0]:.2e}, "
              f"query {worst['observe'][1]:.2e}, first flip layer {first_flip}")
    else:
        print(f"- worst relative error, leaf states {state:.2e}, query states {query:.2e} "
              f"(threshold {TOL:.0e}): {verdict}")
    del full, model
    if a.device == "mps":
        torch.mps.empty_cache()
    elif a.device == "cuda":
        torch.cuda.empty_cache()

    print("\n## 4. bf16 noise floor")
    full = load(profile, torch.bfloat16, a.device)
    model = full.model
    mm16 = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in mm.items()}
    text_req = TreeRequest(prefix=text_ids(tok, 300, 5), questions=[
        Question(segment=text_ids(tok, 20, 6), leaves=[text_ids(tok, 12, 20 + i) + [QUERY_IDS["choice"]]
                                                      for i in range(6)])])
    filler = TreeRequest(prefix=text_ids(tok, 257, 7), questions=[Question(segment=[], leaves=[[QUERY_IDS["noul"]]])])

    def query_states(reqs, which):
        t = pack(reqs)
        out = forward(model, t.input_ids.tolist(), {}, t.position_ids, hf_masks(attention_masks(t, W), torch.bfloat16))
        qi = t.query_index[t.request_of_question[t.leaf_question] == which]
        return out[qi.to(out.device)]

    with torch.no_grad():
        base = query_states([text_req], 0)
        repeat = query_states([text_req], 0)
        shifted = query_states([filler, text_req], 1)
        rev = TreeRequest(prefix=text_req.prefix, questions=[
            Question(segment=text_req.questions[0].segment, leaves=text_req.questions[0].leaves[::-1])])
        reordered = query_states([rev], 0).flip(0)
    print(f"- identical input repeated: rel err {rel_err(repeat, base):.2e}")
    print(f"- same request packed after another (offset 258): rel err {rel_err(shifted, base):.2e}")
    print(f"- leaves in reverse packing order: rel err {rel_err(reordered, base):.2e}")

    if not a.skip_latency:
        print("\n## 5. Latency, bf16, batch 1")
        lat_req = TreeRequest(prefix=text_ids(tok, 1000, 8), questions=[
            Question(segment=text_ids(tok, 15, 30 + j),
                     leaves=[text_ids(tok, 8, 40 + 5 * j + i) + [QUERY_IDS["choice"]] for i in range(5)])
            for j in range(4)])
        t = pack([lat_req])
        masks = hf_masks(attention_masks(t, W), torch.bfloat16)
        sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(a.device, lambda: None)
        times = []
        with torch.no_grad():
            for i in range(25):
                sync(); t0 = time.perf_counter()
                forward(model, t.input_ids.tolist(), {}, t.position_ids, masks)
                sync(); times.append(time.perf_counter() - t0)
        times = times[5:]
        print(f"- {len(t.input_ids)} packed tokens, {len(t.query_index)} blocks on {a.device}: "
              f"p50 {statistics.median(times) * 1000:.0f} ms, p95 {np.percentile(times, 95) * 1000:.0f} ms "
              f"(mask build excluded)")


if __name__ == "__main__":
    main()
