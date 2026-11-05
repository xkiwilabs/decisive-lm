"""Convert fetched public datasets (data/raw/<name>/) into canonical rows (data/processed/<name>.jsonl).

    uv run python scripts/fetch_dataset.py banking77 && uv run python scripts/convert_dataset.py banking77

Your own converter (docs/guide/08 §8.8) runs the same way, from its file, with its source registered in
configs/datasets.local.yaml:

    uv run python scripts/convert_dataset.py --converter my_tickets.py:convert --name support_tickets

Rows are checked against configs/datasets.yaml before they are written; split id hashes are printed. Frozen-suite
rows listed in configs/suite_exclusions.txt (text that overlaps trainable data) are left out.
"""
import argparse, collections, hashlib, importlib.util, inspect, pathlib


import functools

from decisivelm.data.converters import (cifar10h, civil_comments, crema_d, csc_sarcasm, cuad, emozionalmente,
                                        goemotions, helpsteer2, helpsteer2_preference, helpsteer3, humicroedit,
                                        jev_public, koniq10k, ledgar, massive, measuring_hate_speech, multinli, odaq,
                                        politeness, sib200, sonyc_ust, textvqa, things_living, vizwiz_vqa,
                                        wiki_toxicity)
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
              "cifar10h": cifar10h.convert,
              "emozionalmente": emozionalmente.convert, "sonyc_ust": sonyc_ust.convert,   # 1.0 audio training
              "vizwiz_vqa": vizwiz_vqa.convert,   # 1.0 image training
              "odaq": odaq.convert, "textvqa": textvqa.convert,   # 1.0 frozen audio and image suites
              # the frozen held-out suite (docs/RELEASE.md, SUITE)
              "helpsteer3": helpsteer3.convert, "measuring_hate_speech": measuring_hate_speech.convert,
              "cuad": cuad.convert, "massive": massive.convert,
              **{n: functools.partial(jev_public.convert_dhruv, n) for n in jev_public.DHRUV},
              **{n: functools.partial(jev_public.convert_elcronos, n) for n in jev_public.ELCRONOS}}
SUITE = {"helpsteer3", "measuring_hate_speech", "cuad", "massive"}


def user_converter(spec: str):
    """The function FUNCTION from FILE.py, given as FILE.py:FUNCTION."""
    path, _, function = spec.rpartition(":")
    if not path or not function:
        raise SystemExit(f"--converter takes FILE.py:FUNCTION, not {spec!r}")
    module_spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    if module_spec is None or not pathlib.Path(path).is_file():
        raise SystemExit(f"--converter: no Python file {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not callable(convert := getattr(module, function, None)):
        raise SystemExit(f"--converter: {path} has no function {function}")
    return convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", metavar="name", help=f"built-in converters: {', '.join(sorted(CONVERTERS))}")
    ap.add_argument("--converter", metavar="FILE.py:FUNCTION",
                    help="your own converter: a function taking the raw-data directory and returning Records")
    ap.add_argument("--name", help="with --converter: the source name, as in configs/datasets.local.yaml; rows are "
                                   "written to data/processed/<name>.jsonl")
    ap.add_argument("--raw", type=pathlib.Path,
                    help="with --converter: the raw-data directory (default data/raw/<name>)")
    ap.add_argument("--backbone", default="e4b", help="tokenizer for state truncation (docs/guide/04 §4.6)")
    a = ap.parse_args()
    if unknown := [n for n in a.names if n not in CONVERTERS]:
        ap.error(f"no built-in converter {', '.join(unknown)}; for your own, use --converter FILE.py:FUNCTION --name")
    jobs = [(n, CONVERTERS[n], REPO / "data" / "raw" / n) for n in a.names]
    if a.converter:
        if not a.name:
            ap.error("--converter needs --name")
        jobs.append((a.name, user_converter(a.converter), a.raw or REPO / "data" / "raw" / a.name))
    if not jobs:
        ap.error("name a built-in converter or pass --converter")
    registry = load_registry()
    tok = load_tokenizer(a.backbone)
    for name, convert, raw in jobs:
        kwargs = {"tokenizer": tok} if "tokenizer" in inspect.signature(convert).parameters else {}
        rows = convert(raw, **kwargs)
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
