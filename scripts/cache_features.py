"""Cache query-token hidden states for P1 (docs/guide/09 §9.4): the frozen backbone runs once per row.

    uv run python scripts/cache_features.py --data data/processed/llm_response_coding.jsonl \
        --out data/features/llm_response_coding_e4b.pt

Resumable: progress is saved every --save-every rows and skipped on restart.
"""
import argparse, json, pathlib, time

import torch

from decisivelm.head import Scorer
from decisivelm.infer import DecisiveLM
from decisivelm.model import backbone_profile, fp32_routers, load_backbone, vision_kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--backbone", default="e4b")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="backbone compute dtype; float32 removes bf16 packing noise (results/invariance.md)")
    ap.add_argument("--fp32-router", action="store_true", help="MoE routers in fp32 (decisivelm.model.fp32_routers)")
    ap.add_argument("--image-budget", type=int, help="soft tokens per image (default: the backbone profile's)")
    ap.add_argument("--media-root", type=pathlib.Path, help="directory relative media paths resolve against")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    cache = torch.load(a.out, weights_only=True) if a.out.exists() else {"backbone": backbone_profile(a.backbone),
                                                                         "dtype": a.dtype, "fp32_router": a.fp32_router,
                                                                         "image_budget": None, "states": {}}
    if (cache.get("dtype", "bfloat16"), cache.get("fp32_router", False)) != (a.dtype, a.fp32_router):
        raise SystemExit(f"{a.out} was cached with other settings; use another --out")
    todo = [r for r in rows if r["id"] not in cache["states"]]
    print(f"{len(cache['states'])} cached, {len(todo)} to go", flush=True)
    backbone, tok = load_backbone(a.backbone, dtype=getattr(torch, a.dtype))
    if a.fp32_router:
        print(f"{fp32_routers(backbone)} routers in fp32", flush=True)
    vision = vision_kwargs(a.backbone, rows, a.image_budget, a.media_root)
    dlm = DecisiveLM(backbone, tok, Scorer(backbone.config.text_config.hidden_size), **vision)
    if vision.get("image_budget"):
        if cache.get("image_budget") not in (None, vision["image_budget"]):
            raise SystemExit(f"{a.out} was cached at another image budget; use another --out")
        cache["image_budget"] = vision["image_budget"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        _, q = dlm.query_states([dlm.tree_for({"state": r["state"], "media": r.get("media", []), "questions": r["questions"]})])
        cache["states"][r["id"]] = q.to("cpu", getattr(torch, a.dtype))
        if i % a.save_every == 0 or i == len(todo):
            torch.save(cache, a.out)
            print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f} s/row", flush=True)


if __name__ == "__main__":
    main()
