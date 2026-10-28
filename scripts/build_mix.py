"""Build a training mix for one adapter, with a manifest (docs/guide/13 §13.6, docs/guide/08 §8.4).

    uv run python scripts/build_mix.py --name general_v0 --adapter general --tiers release \
        --train data/processed/banking77.jsonl:5000 data/processed/banking77_explode.jsonl:10000 ... \
        --eval data/processed/banking77.jsonl data/processed/banking77_explode.jsonl:1000 ... \
        --zero-shot data/processed/llm_response_coding.jsonl

- `--train PATH[:N]`: N train rows sampled (seeded) from PATH, all of them without N. Checked with `check_mix`: train
  split only, no holdout for this adapter, tiers allowed by `--tiers`.
- `--eval PATH[:N]`: val, gold_calib and gold_test rows of PATH (N per split when given), for early stopping,
  calibration and the report. Never trained on.
- `--cap MULTIPLIER=SHARE`: in every `--train` source, rows made by that multiplier (`lineage.multiplier`) are at
  most SHARE of the rows sampled from it; e.g. `schema_explosion:drop_gold=0.05`, since a high share of drop-gold
  rows teaches a catch-all `other` (results/general_v0.md).
- `--zero-shot PATH[:N]`: gold_test rows only, so nothing from that dataset (not even calibration) reaches the
  adapter; e.g. a dataset held out of the general adapter.

Writes data/mixes/<name>.jsonl and <name>.manifest.json (sources, row counts by split, sha256 of the file).
"""
import argparse
import collections
import hashlib
import json
import pathlib
import random

from decisivelm.data.dataset import load_records
from decisivelm.data.mix import MIX_TIERS, check_mix, load_holdouts

REPO = pathlib.Path(__file__).resolve().parent.parent


def spec(s: str) -> tuple[pathlib.Path, int | None]:
    path, _, n = s.partition(":")
    return pathlib.Path(path), int(n) if n else None


def sample(rows, n, rng):
    return rows if n is None or n >= len(rows) else [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]


def capped(rows, caps: dict[str, float], n, rng):
    """Sample n rows (all without n) with each capped multiplier's rows at most its share of the sample. Without
    caps this is `sample`, so earlier mixes rebuild identically; rows keep their order in the source."""
    if not caps:
        return sample(rows, n, rng)
    total = len(rows) if n is None else min(n, len(rows))
    out = []
    for m, share in caps.items():
        mine = [r for r in rows if r.lineage.multiplier == m]
        out += sample(mine, int(share * total), rng)
    rest = [r for r in rows if r.lineage.multiplier not in caps]
    out += sample(rest, total - len(out), rng)
    order = {id(r): i for i, r in enumerate(rows)}
    return sorted(out, key=lambda r: order[id(r)])


def cap(s: str) -> tuple[str, float]:
    m, _, share = s.rpartition("=")
    return m, float(share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", choices=["general", "user"], required=True)
    ap.add_argument("--tiers", choices=sorted(MIX_TIERS), default="release")
    ap.add_argument("--train", nargs="+", type=spec, required=True)
    ap.add_argument("--eval", nargs="*", type=spec, default=[])
    ap.add_argument("--zero-shot", nargs="*", type=spec, default=[])
    ap.add_argument("--cap", nargs="*", type=cap, default=[], help="MULTIPLIER=SHARE: at most SHARE of each train source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO / "data" / "mixes")
    a = ap.parse_args()

    cache = {}
    load = lambda p: cache.setdefault(p, load_records(p))
    out, sources = [], []
    for path, n in a.train:
        rows = capped([r for r in load(path) if r.split == "train"], dict(a.cap), n,
                      random.Random(f"{a.seed}-train-{path.name}"))
        check_mix(rows, load_holdouts(), adapter=a.adapter, tiers=MIX_TIERS[a.tiers])
        out += rows
        sources.append({"path": str(path), "role": "train", "rows": len(rows)})
    for role, splits, specs in (("eval", ("val", "gold_calib", "gold_test"), a.eval),
                                ("zero_shot", ("gold_test",), a.zero_shot)):
        for path, n in specs:
            got = {}
            for s in splits:
                rows = sample([r for r in load(path) if r.split == s], n, random.Random(f"{a.seed}-{s}-{path.name}"))
                out += rows
                got[s] = len(rows)
            sources.append({"path": str(path), "role": role, "rows": got})
    ids = collections.Counter(r.id for r in out)
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        raise SystemExit(f"{len(dup)} rows appear twice (a file listed under two roles?), e.g. {dup[:3]}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"{a.name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    manifest = {
        "name": a.name, "adapter": a.adapter, "tiers": sorted(MIX_TIERS[a.tiers]), "seed": a.seed, "caps": dict(a.cap),
        "sources": sources,
        "rows_by_split": dict(collections.Counter(r.split for r in out)),
        "rows_by_dataset_and_split": {f"{d}/{s}": n for (d, s), n in
                                      sorted(collections.Counter((r.lineage.source_dataset, r.split) for r in out).items())},
        "questions_by_type": dict(collections.Counter(q["type"] if isinstance(q, dict) else q.type
                                                      for r in out if r.split == "train" for q in r.questions.values())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (a.out_dir / f"{a.name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("rows_by_split", "questions_by_type", "sha256")}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
