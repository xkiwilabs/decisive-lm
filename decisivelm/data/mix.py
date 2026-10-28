"""Training mixes: licence tiers and enforced holdouts (docs/guide/08 §8.5, §8.6).

A mix is refused, not filtered, when it overlaps a holdout: a held-out row
reaching a training mix is a bug upstream (split or lineage), not something to
paper over here.
"""
import pathlib

import yaml

from decisivelm import CONFIGS
from decisivelm.data.schema import Record

MIX_TIERS = {"release": {"R"}, "private": {"R", "S"}, "research": {"R", "S", "X"}}
SELECTORS = {"source_ids": lambda r: r.lineage.source_id,
             "family_ids": lambda r: r.lineage.family_id,
             "languages": lambda r: r.language}


def load_holdouts(path: pathlib.Path = CONFIGS / "holdout.yaml") -> list[dict]:
    return yaml.safe_load(pathlib.Path(path).read_text())["holdouts"]


def held_out(record: Record, holdout: dict) -> bool:
    if record.lineage.source_dataset != holdout["dataset"]:
        return False
    return all(get(record) in holdout[key] for key, get in SELECTORS.items() if key in holdout)


def check_mix(records: list[Record], holdouts: list[dict], adapter: str, tiers: set[str] = MIX_TIERS["research"]):
    """Raise if any row is not a `train` row, is held out for this adapter ("general" or "user"),
    or has a tier the mix doesn't allow. val and gold splits are never trained on (docs/guide/08 §8.3)."""
    active = [h for h in holdouts if adapter in h["adapters"]]
    bad = [r.id for r in records if any(held_out(r, h) for h in active)]
    if bad:
        raise ValueError(f"{len(bad)} rows overlap a {adapter} holdout (configs/holdout.yaml), e.g. {bad[:5]}")
    not_train = [r.id for r in records if r.split != "train"]
    if not_train:
        raise ValueError(f"{len(not_train)} rows are not in the train split, e.g. {not_train[:5]}")
    wrong_tier = [r.id for r in records if r.tier not in tiers]
    if wrong_tier:
        raise ValueError(f"{len(wrong_tier)} rows have a tier outside {sorted(tiers)}, e.g. {wrong_tier[:5]}")
