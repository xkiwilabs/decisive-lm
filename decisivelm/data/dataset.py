"""Dataset registry and licence checks (docs/guide/14 §14.1).

configs/datasets.yaml records each source's expected SPDX licence and tier. A
tier is derived from the licence; a source may be placed in a *more* restrictive
tier with an `override` reason, never a less restrictive one without it.
Every row read for training is checked against its source's entry.
"""
import json
import pathlib

import yaml

from decisivelm import CONFIGS
from decisivelm.data.schema import Record

RANK = {"R": 0, "S": 1, "X": 2}
RELEASE = {"CC0-1.0", "PDDL-1.0", "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "Unlicense",
           "CDLA-Permissive-1.0", "CDLA-Permissive-2.0", "ODC-By-1.0",
           "LicenseRef-OANC",   # Open ANC texts: free of restrictions on use and redistribution (no SPDX id)
           "NOASSERTION"}       # no licence and no restricting terms stated (docs/guide/14 §14.1)
SHARE_ALIKE = {"ODbL-1.0", "CDLA-Sharing-1.0"}


def tier_of(spdx: str) -> str:
    """R for permissive licences, S for share-alike, X for everything else (non-commercial, no-derivatives, unknown)."""
    if spdx in RELEASE:
        return "R"
    if spdx in SHARE_ALIKE or spdx.startswith(("CC-BY-SA-", "GPL-", "LGPL-", "AGPL-")):
        return "S"
    if spdx.startswith("CC-BY-") and spdx.split("-")[2][0].isdigit():
        return "R"
    return "X"


LOCAL_REGISTRY = CONFIGS / "datasets.local.yaml"


def load_registry(path: pathlib.Path = CONFIGS / "datasets.yaml", local: pathlib.Path | None = LOCAL_REGISTRY) -> dict:
    """The public registry, plus the machine-local, gitignored `local` one for private sources (same format), which
    may add sources but never redefine a public one."""
    registry = yaml.safe_load(pathlib.Path(path).read_text())["datasets"]
    if local is not None and pathlib.Path(local).exists():
        extra = yaml.safe_load(pathlib.Path(local).read_text()).get("datasets") or {}
        if clash := set(extra) & set(registry):
            raise ValueError(f"{local} redefines public sources {sorted(clash)}")
        registry = {**registry, **extra}
    return registry


def check_registry(registry: dict):
    for name, d in registry.items():
        expected = tier_of(d["license"])
        if RANK[d["tier"]] < RANK[expected]:
            raise ValueError(f"{name}: tier {d['tier']} is less restrictive than {d['license']} ({expected}); "
                             f"no override can lower a tier")
        if d["tier"] != expected and not d.get("override"):
            raise ValueError(f"{name}: tier {d['tier']} differs from {d['license']} ({expected}); "
                             f"give an override reason")


def check_licence(record: Record, registry: dict):
    name = record.lineage.source_dataset
    if name not in registry:
        raise ValueError(f"{record.id}: source {name!r} is not in configs/datasets.yaml")
    d = registry[name]
    if (record.license, record.tier) != (d["license"], d["tier"]):
        raise ValueError(f"{record.id}: licence {record.license}/{record.tier} does not match "
                         f"configs/datasets.yaml ({d['license']}/{d['tier']})")


def load_records(path: pathlib.Path, registry: dict | None = None) -> list[Record]:
    """Validated rows from a JSONL file, each checked against the registry."""
    registry = load_registry() if registry is None else registry
    check_registry(registry)
    rows = [Record.model_validate(json.loads(line)) for line in open(path, encoding="utf-8")]
    for r in rows:
        check_licence(r, registry)
    return rows
