"""Convert fetched public datasets (data/raw/<name>/) into canonical rows (data/processed/<name>.jsonl).

    uv run python scripts/fetch_dataset.py banking77 && uv run python scripts/convert_dataset.py banking77

Rows are checked against configs/datasets.yaml before they are written; split id hashes are printed. Frozen-suite
rows listed in configs/suite_exclusions.txt (text that overlaps trainable data) are left out.
"""
import argparse, collections, hashlib, inspect, pathlib


import functools

from decisivelm.data.converters import (civil_comments, crema_d, csc_sarcasm, cuad, goemotions, helpsteer2,
                                        helpsteer2_preference, helpsteer3,
                                        humicroedit, jev_public, koniq10k, ledgar, massive, measuring_hate_speech, multinli,
                                        politeness, sib200, things_living, wiki_toxicity)
from decisivelm.data.converters.intents import convert_banking77, convert_clinc150
from decisivelm.data.dataset import check_licence, load_registry
from decisivelm.model import load_tokenizer

REPO = pathlib.Path(__file__).resolve().parent.parent
CONVERTERS = {"banking77": convert_banking77, "clinc150": convert_clinc150, "helpsteer2": helpsteer2.convert,
              "civil_comments": civil_comments.convert, "goemotions": goemotions.convert,
              "ledgar": ledgar.convert, "multinli": multinli.convert, "sib200": sib200.convert,
              "politeness": politeness.convert, "wiki_toxicity": wiki_toxicity.convert,
              "humicroedit": humicroedit.convert, "csc_sarcasm": csc_sarcasm.convert,
              "helpsteer2_preference": helpsteer2_preference.convert,
              "crema_d": crema_d.convert, "koniq10k": koniq10k.convert,   # audio and image fine-tune examples
              "things_living": things_living.convert,
              # the frozen held-out suite (docs/RELEASE.md, SUITE)
              "helpsteer3": helpsteer3.convert, "measuring_hate_speech": measuring_hate_speech.convert,
              "cuad": cuad.convert, "massive": massive.convert,
              **{n: functools.partial(jev_public.convert_dhruv, n) for n in jev_public.DHRUV},
              **{n: functools.partial(jev_public.convert_elcronos, n) for n in jev_public.ELCRONOS}}
SUITE = {"helpsteer3", "measuring_hate_speech", "cuad", "massive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(CONVERTERS))
    ap.add_argument("--backbone", default="e4b", help="tokenizer for state truncation (docs/guide/04 §4.6)")
    a = ap.parse_args()
    registry = load_registry()
    tok = load_tokenizer(a.backbone)
    for name in a.names:
        convert = CONVERTERS[name]
        kwargs = {"tokenizer": tok} if "tokenizer" in inspect.signature(convert).parameters else {}
        rows = convert(REPO / "data" / "raw" / name, **kwargs)
        if name in SUITE:
            excluded = {line.strip() for line in open(REPO / "configs" / "suite_exclusions.txt")
                        if line.strip() and not line.startswith("#")}
            kept = [r for r in rows if r.id not in excluded]
            print(f"{name}: {len(rows) - len(kept)} rows left out (configs/suite_exclusions.txt)")
            rows = kept
        for r in rows:
            check_licence(r, registry)
        out = REPO / "data" / "processed" / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(r.model_dump_json(exclude_none=True) + "\n")
        splits = collections.Counter(r.split for r in rows)
        hashes = {s: hashlib.sha256("\n".join(sorted(r.id for r in rows if r.split == s)).encode()).hexdigest()[:16]
                  for s in splits}
        truncated = sum(r.preprocess.truncated for r in rows)
        print(f"{name}: {len(rows)} rows -> {out.relative_to(REPO)}; {dict(splits)}; {truncated} truncated; "
              f"split hashes {hashes}")


if __name__ == "__main__":
    main()
