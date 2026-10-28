"""Backbone loading (docs/guide/06 §6.4, §6.9). Returns the decoder-only path: never the lm_head."""
import pathlib

import torch
import yaml
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer, Gemma4ForConditionalGeneration

from decisivelm import CONFIGS


def backbone_profile(name: str) -> dict:
    return yaml.safe_load((CONFIGS / "model.yaml").read_text())["backbones"][name]


def hub_kwargs(path: pathlib.Path = CONFIGS / "local.yaml", offline: bool = True) -> dict:
    """Extra Hugging Face download arguments from the machine-local, gitignored configs/local.yaml:
    `hf_cache` → cache_dir, `offline: true` → local_files_only (pass offline=False for dataset downloads).
    Empty when the file is absent. Set it where the default cache belongs to a serving stack."""
    cfg = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    kw = {}
    if cfg.get("hf_cache"):
        kw["cache_dir"] = str(pathlib.Path(cfg["hf_cache"]).expanduser())
    if offline and cfg.get("offline"):
        kw["local_files_only"] = True
    return kw


def fp32_routers(model) -> int:
    """Run every MoE router in fp32 (its input upcast, its parameters fp32); the experts stay in the model dtype.
    In bf16 the 128 router scores are rounded to ~3 significant digits, so the 8th and 9th expert are often tied
    or near-tied and which one wins depends on the packing (results/invariance.md). Returns the number patched."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter
    routers = [m for m in model.modules() if isinstance(m, Gemma4TextRouter)]
    for r in routers:
        r.float()
        r.register_forward_pre_hook(lambda mod, args: (args[0].float(), *args[1:]))
    return len(routers)


def fp32_compute(model) -> int:
    """Give an MoE backbone loaded in bf16 the numerics of a full fp32 load. Every parameter except the expert weights
    becomes fp32; each expert upcasts its own weights for its matmuls. The released checkpoint is bf16, so the
    upcast is exact and the result equals an fp32 load, while the experts (~90% of 26B A4B) keep bf16 memory.
    bf16 noise flips near-tied experts, and the flips compound with depth (results/invariance.md). Returns the
    number of expert modules patched."""
    import types

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts, Gemma4TextScaledWordEmbedding
    experts = [m for m in model.modules() if isinstance(m, Gemma4TextExperts)]
    for m in model.modules():
        if not isinstance(m, Gemma4TextExperts):
            m._apply(lambda t: t.float() if t.is_floating_point() else t, recurse=False)
        if isinstance(m, Gemma4TextScaledWordEmbedding):   # built in bf16 at load (53.0 for 53.07): recompute it
            m.embed_scale = torch.tensor(m.scalar_embed_scale, device=m.weight.device)
    for m in experts:
        m.forward = types.MethodType(_fp32_experts_forward, m)
    return len(experts)


def _fp32_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Gemma4TextExperts.forward in fp32, grouped by expert as transformers' grouped_mm path is (one host sync per
    layer), and summed over the top-k without atomics so a repeat reproduces exactly."""
    n, k = top_k_index.shape
    flat = top_k_index.reshape(-1)
    order = flat.argsort(stable=True)
    x = hidden_states.float()[order // k]
    out = x.new_empty(n * k, x.shape[-1])
    start = 0
    for e, count in enumerate(torch.bincount(flat, minlength=self.num_experts).tolist()):
        if count:
            gate, up = torch.nn.functional.linear(x[start:start + count], self.gate_up_proj[e].float()).chunk(2, -1)
            out[start:start + count] = torch.nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e].float())
            start += count
    out = out[order.argsort()].view(n, k, -1)
    return (out * top_k_weights[..., None].float()).sum(1).to(hidden_states.dtype)


def load_tokenizer(name: str = "e4b"):
    p = backbone_profile(name)
    return AutoTokenizer.from_pretrained(p["model_id"], revision=p["revision"], **hub_kwargs())


def load_image_processor(name: str = "e4b"):
    """The backbone's own Gemma 4 image processor (decisivelm.media encodes with it at a fixed soft-token budget)."""
    p = backbone_profile(name)
    return AutoImageProcessor.from_pretrained(p["model_id"], revision=p["revision"], **hub_kwargs())


def load_feature_extractor(name: str = "e4b"):
    """The backbone's own Gemma 4 audio feature extractor (log-mel frames at 16 kHz; decisivelm.media)."""
    p = backbone_profile(name)
    if "audio" not in p.get("modalities", []):
        raise ValueError(f"backbone {name} has no audio tower")
    return AutoFeatureExtractor.from_pretrained(p["model_id"], revision=p["revision"], **hub_kwargs())


def vision_kwargs(name: str, rows, budget: int | None = None, media_root=None) -> dict:
    """DecisiveLM's media arguments when any row (a Record or its dict) carries media, else {}: the image processor
    and budget (which defaults to the backbone profile's image_max_soft_tokens and is recorded with every cache and
    bundle) for images, the feature extractor for audio."""
    media = [m for r in rows for m in ((r.get("media") if isinstance(r, dict) else r.media) or [])]
    if not media:
        return {}
    kinds = {m["type"] if isinstance(m, dict) else m.type for m in media}
    kw = {"media_root": pathlib.Path(media_root) if media_root else None}
    if "image" in kinds:
        kw |= {"image_processor": load_image_processor(name),
               "image_budget": budget or backbone_profile(name)["image_max_soft_tokens"]}
    if "audio" in kinds:
        kw["feature_extractor"] = load_feature_extractor(name)
    return kw


def load_backbone(name: str = "e4b", dtype=torch.bfloat16, device: str | None = None, attn: str = "sdpa"):
    """(Gemma4Model, tokenizer) for a backbone profile in configs/model.yaml. The model records the profile's name, id
    and revision as `decisivelm_backbone`, which decisivelm.bundle.attach_bundle checks a bundle against."""
    p = backbone_profile(name)
    model = _load(p, dtype, device, attn).model.eval()
    model.decisivelm_backbone = {"name": name, "model_id": p["model_id"], "revision": p["revision"]}
    return model, load_tokenizer(name)


def load_causal_lm(name: str = "e4b", dtype=torch.bfloat16, device: str | None = None, attn: str = "sdpa"):
    """(full model with lm_head, tokenizer): for prompting baselines and the labeller (P0, docs/guide/12 §12.8), never for DecisiveLM."""
    return _load(backbone_profile(name), dtype, device, attn).eval(), load_tokenizer(name)


def _load(p: dict, dtype, device: str | None, attn: str):
    """On CUDA, load straight onto the GPU (26B A4B is ~52 GB in bf16; staging it in host RAM first is wasteful).
    device_map crashes the process on MPS, so elsewhere load on the host and move. An MoE backbone in float32 loads
    in bf16 and runs fp32_compute: the same numbers as a full fp32 load (104 GB for 26B A4B) from about 55 GB of weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    moe_fp32 = dtype == torch.float32 and "moe" in p
    kw = dict(revision=p["revision"], dtype=torch.bfloat16 if moe_fp32 else dtype, attn_implementation=attn,
              **hub_kwargs())
    if device.startswith("cuda"):
        model = Gemma4ForConditionalGeneration.from_pretrained(p["model_id"], device_map=device, **kw)
    else:
        model = Gemma4ForConditionalGeneration.from_pretrained(p["model_id"], **kw).to(device)
    if moe_fp32:
        fp32_compute(model)
    return model
