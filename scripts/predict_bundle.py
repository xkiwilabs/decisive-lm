"""Predict with a saved bundle on other datasets, using the bundle's own calibration (zero-shot for anything it was
not trained or calibrated on).

    uv run python scripts/predict_bundle.py --bundle bundles/general_v0_e4b_r8_seed0 \
        --data data/processed/bench_agnews.jsonl data/processed/bench_sst2.jsonl

Writes data/predictions/<bundle>__<data stem>.jsonl in the decisivelm.calibrate format, for scripts/eval_dataset.py.
Rows with images or audio get the backbone's media processors; a bundle never trained on media answers them zero-shot,
at its own image budget if it records one (else --image-budget, else the profile's).
"""
import argparse
import json
import pathlib
import random
import time

import torch

from decisivelm.bundle import attach_bundle
from decisivelm.calibrate import predictions, question_logits, write_predictions
from decisivelm.data.dataset import load_records
from decisivelm.model import backbone_profile, load_backbone, vision_kwargs
from decisivelm.train import make_items, micro_batches

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=pathlib.Path, required=True, help="the bundle directory")
    ap.add_argument("--data", type=pathlib.Path, nargs="+", required=True,
                    help="one or more data/processed/<name>.jsonl files")
    ap.add_argument("--splits", nargs="+", default=["gold_test"], help="splits to predict")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="packed tokens per forward pass; lower it if memory runs out")
    ap.add_argument("--image-budget", type=int, help="soft tokens per image (default: the bundle's, else the profile's)")
    ap.add_argument("--media-root", type=pathlib.Path, default=REPO, help="relative media paths resolve against this")
    a = ap.parse_args()

    calib = json.loads((a.bundle / "calibration.json").read_text())
    name = next(k for k, p in (("e4b", backbone_profile("e4b")), ("26b-a4b", backbone_profile("26b-a4b")))
                if p["model_id"] == calib["backbone"]["model_id"])
    if backbone_profile(name)["revision"] != calib["backbone"]["revision"]:
        raise SystemExit(f"bundle was trained on {calib['backbone']['model_id']}@{calib['backbone']['revision']}")
    dtype = calib.get("dtype", "bfloat16")   # the dtype it was trained in, as decisivelm.bundle.load_bundle loads it
    backbone, tok = load_backbone(name, dtype=getattr(torch, dtype))
    print(f"{name} in {dtype}", flush=True)
    dlm = attach_bundle(a.bundle, backbone, tok)
    for path in a.data:
        rows = [r for r in load_records(path) if r.split in a.splits]
        for k, v in vision_kwargs(name, rows, calib.get("image_budget") or a.image_budget, a.media_root).items():
            setattr(dlm, k, v)
        rec = {r.id: r for r in rows}
        t0, qs = time.perf_counter(), []
        with torch.no_grad():
            for b in micro_batches(make_items(dlm, rows), a.max_tokens, random.Random(0)):
                tree, logits = dlm.logits([it.tree for it in b])
                leaf_request = tree.request_of_question[tree.leaf_question]
                for k, it in enumerate(b):
                    qs += question_logits(rec[it.id], logits[leaf_request == k])
        out = REPO / "data" / "predictions" / f"{a.bundle.name}__{path.stem}.jsonl"
        write_predictions(out, predictions(qs, dlm.calibration))
        print(f"{path.name}: {len(rows)} rows in {time.perf_counter() - t0:.0f} s -> {out.relative_to(REPO)}", flush=True)
    if torch.cuda.is_available():   # what a card must hold: PyTorch's peak, and its cache (nvidia-smi adds the context)
        print(f"peak GPU memory: allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
