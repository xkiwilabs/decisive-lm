"""Scorer head and readout (docs/guide/06 §6.5).

The scorer turns each query token's hidden state into one logit. Distributions
come from grouping the logits by question: softmax for Choice and Score,
sigmoid for Noul. Temperatures are looked up by (type, option-count bucket).
"""
import hashlib
import json
from dataclasses import dataclass, field

import torch
from torch import nn


class Scorer(nn.Module):
    def __init__(self, hidden_size: int, width: int = 1024):
        super().__init__()
        self.register_buffer("shift", torch.zeros(hidden_size))
        self.register_buffer("scale", torch.ones(hidden_size))
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU(),
                                 nn.Linear(width, 1))
        self.register_load_state_dict_pre_hook(_identity_if_unstandardized)

    @torch.no_grad()
    def standardize(self, states: torch.Tensor):
        """Z-score each dimension with statistics of training query states, before the LayerNorm. 26B A4B's final
        states have massive-activation dims (one holds 4% of the variance, 100 hold 30%; E4B 0.4% and 13%), and
        they set LayerNorm's per-state scale, squashing the rest (results/invariance.md)."""
        flat = states.reshape(-1, states.shape[-1]).float()
        self.shift.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0).clamp_min(1e-6))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net((states - self.shift) / self.scale).squeeze(-1)


def _identity_if_unstandardized(module, state_dict, prefix, *args):
    """scorer.pt files saved before standardisation existed have no shift/scale: load them as the identity."""
    for name, fill in (("shift", torch.zeros_like), ("scale", torch.ones_like)):
        state_dict.setdefault(prefix + name, fill(getattr(module, name)))


BUCKETS = [(2, 2), (3, 5), (6, 15), (16, 63), (64, 255)]


def bucket(n_options: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n_options <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    raise ValueError(f"no bucket for {n_options} options")


@dataclass
class Calibration:
    """Fitted at the calibration phase (docs/guide/10); identity until then. `tasks` holds task calibration: for a
    question fitted on its own labelled rows (decisivelm.calibrate.fit_task_calibration), keyed by `task_key`, a scale
    on its temperature and an offset per option or level: {"scale": s, "offsets": {name: b}} (Noul: {"yes": b})."""
    temperatures: dict[str, float] = field(default_factory=dict)   # "choice:3-5" -> T, "noul" -> T
    b_noul: float = 0.0                                              # "choice:*" -> the type's pooled T
    tasks: dict[str, dict] = field(default_factory=dict)

    def temperature(self, qtype: str, n: int) -> float:
        if qtype == "noul":
            return self.temperatures.get("noul", 1.0)
        return self.temperatures.get(f"{qtype}:{bucket(n)}", self.temperatures.get(f"{qtype}:*", 1.0))


def task_key(q) -> str:
    """A question's identity for task calibration: its type, instructions and options, never its id. Choice options
    are sorted, so reordering them finds the same calibration; a Score's levels keep their order."""
    d = q.model_dump(include={"type", "instructions", "criteria"}, mode="json")
    if d["type"] == "choice":
        d["criteria"] = sorted(d["criteria"].items())
    return d["type"] + ":" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def readout(qtype: str, logits: torch.Tensor, calib: Calibration, key: str | None = None,
            names: list[str] | None = None) -> torch.Tensor:
    """Probabilities over a question's blocks (Choice/Score) or P(yes) as a 1-element tensor (Noul). With the
    question's `key` and option `names` (Score: "0", "1", …; Noul: ["yes"]), its task calibration applies if it has
    one."""
    t = calib.temperature(qtype, len(logits))
    task = calib.tasks.get(key) if key is not None else None
    s, b = 1.0, torch.zeros(len(logits))
    if task is not None:
        names = names if names is not None else ["yes"] if qtype == "noul" else [str(i) for i in range(len(logits))]
        s, b = task["scale"], torch.tensor([task["offsets"].get(k, 0.0) for k in names])
    b = b.to(logits.device, logits.dtype)
    if qtype == "noul":
        return torch.sigmoid((logits + calib.b_noul) / (t * s) + b)
    return torch.softmax(logits / (t * s) + b, dim=-1)
