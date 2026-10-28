"""Tiny random Gemma 4 text configs for CI (docs/guide/11 §11.7).

Two variants mirror the two supported backbones (docs/guide/06 §6.9):
- `dense`: like E4B. Per-Layer Embeddings on, the last half of the layers
  share KV with earlier layers.
- `moe`: like 26B A4B. Mixture-of-experts blocks, K = V on global layers,
  no PLE, no KV sharing.

The sliding window is 8 so the window bites on short inputs.
"""
from functools import cache

import torch
from transformers import AutoTokenizer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

from decisivelm.model import hub_kwargs

SLIDING_WINDOW = 8
VOCAB = 512


def tiny_text_config(variant: str, vocab: int = VOCAB) -> Gemma4TextConfig:
    common = dict(
        vocab_size=vocab, vocab_size_per_layer_input=vocab,
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, global_head_dim=32,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
        sliding_window=SLIDING_WINDOW, max_position_embeddings=4096,
    )
    if variant == "dense":
        common.update(hidden_size_per_layer_input=8, num_kv_shared_layers=3)
    elif variant == "moe":
        common.update(hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                      enable_moe_block=True, num_experts=4, top_k_experts=2,
                      moe_intermediate_size=32, attention_k_eq_v=True,
                      num_global_key_value_heads=1)
    else:
        raise ValueError(variant)
    return Gemma4TextConfig(**common)


@cache
def tokenizer():
    """The real Gemma 4 tokenizer (small download), for end-to-end tests with a full-vocab tiny model."""
    return AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
                                         **hub_kwargs())


def tiny_decoder(variant: str, attn_implementation: str, seed: int = 0, vocab: int = VOCAB) -> Gemma4TextModel:
    torch.manual_seed(seed)
    model = Gemma4TextModel._from_config(tiny_text_config(variant, vocab),
                                         attn_implementation=attn_implementation,
                                         dtype=torch.float32)
    # Random init at std 0.02 makes every token look alike; widen it so that
    # attending to the wrong token visibly changes the hidden states.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.2)
    return model.eval()


TINY_IMAGE_IDS = {"boi_token_id": VOCAB - 12, "image_token_id": VOCAB - 11, "eoi_token_id": VOCAB - 10,
                  "video_token_id": VOCAB - 9, "audio_token_id": VOCAB - 8, "boa_token_id": VOCAB - 7,
                  "eoa_token_index": VOCAB - 6}


def tiny_multimodal(variant: str, attn_implementation: str = "sdpa", seed: int = 0, vocab: int = VOCAB,
                    bidirectional: bool | None = None, audio: bool = False):
    """A Gemma4Model (vision tower + text decoder, no lm_head) around the tiny text config. With the small vocab the
    image token ids are moved inside it; with the real vocab they stay Gemma 4's. `bidirectional` defaults to the
    variant's real counterpart: 26B A4B (moe) lets an image attend to itself on sliding layers, E4B (dense) doesn't.
    `audio` adds a tiny audio tower (E4B has one, 26B A4B doesn't)."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4Config, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
    torch.manual_seed(seed)
    text = tiny_text_config(variant, vocab)
    if bidirectional if bidirectional is not None else variant == "moe":
        text.use_bidirectional_attention = "vision"
    vision = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=2, head_dim=16, pooling_kernel_size=3, patch_size=16)
    ids = TINY_IMAGE_IDS if vocab == VOCAB else {}
    audio_config = Gemma4AudioConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                     output_proj_dims=text.hidden_size) if audio else None
    config = Gemma4Config(text_config=text, vision_config=vision, audio_config=audio_config, **ids)
    model = Gemma4Model._from_config(config, attn_implementation=attn_implementation, dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.2)
    return model.eval()


def random_image(seed: int, size=(40, 56)):
    import numpy as np
    from PIL import Image
    return Image.fromarray((np.random.default_rng(seed).random((size[1], size[0], 3)) * 255).astype(np.uint8))
