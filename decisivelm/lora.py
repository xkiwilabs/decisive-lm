"""LoRA on the decoder projections and trainable query-token rows (docs/guide/06 §6.6).

The base weights stay frozen and unchanged. LoRA B starts at zero and the query
rows at a zero delta, so a fresh adapter reproduces the base model exactly.
Adapter parameters are kept in fp32 and cast to the activation dtype.
"""
import math
import re

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        delta = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + (delta * self.scale).to(x.dtype)


class QueryRows(nn.Module):
    """A trainable delta added to the embedding output wherever a query-token id appears."""

    def __init__(self, embed: nn.Embedding, query_ids: list[int]):
        super().__init__()
        self.register_buffer("ids", torch.tensor(query_ids, device=embed.weight.device), persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(query_ids), embed.weight.shape[1], device=embed.weight.device))
        self._handle = embed.register_forward_hook(self._hook)

    def _hook(self, module, args, out):
        match = (args[0][..., None] == self.ids).to(self.delta.dtype)
        return out + (match @ self.delta).to(out.dtype)

    def remove(self):
        self._handle.remove()


def apply_lora(model: nn.Module, target_regex: str, rank: int, alpha: float, query_ids: list[int],
               dropout: float = 0.0) -> nn.Module:
    """Freeze the model, wrap every nn.Linear whose path matches `target_regex`, add query rows as `model.query_rows`."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and re.search(target_regex, n)]
    if not targets:
        raise ValueError(f"no module matches {target_regex!r}")
    for name in targets:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, child, LoRALinear(getattr(owner, child), rank, alpha, dropout))
    embeds = [m for n, m in model.named_modules() if re.search(r"(^|\.)embed_tokens$", n)]
    if len(embeds) != 1:
        raise ValueError(f"expected one embed_tokens module, found {len(embeds)}")
    model.query_rows = QueryRows(embeds[0], query_ids)
    return model


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """The adapter's own parameters (LoRA A/B and query rows), on CPU."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith(("lora_A", "lora_B", "query_rows.delta"))}


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]):
    params = dict(model.named_parameters())
    missing = set(lora_state(model)) - set(state)
    unknown = set(state) - set(params)
    if missing or unknown:
        raise ValueError(f"adapter state mismatch: missing {sorted(missing)[:3]}, unknown {sorted(unknown)[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t)
