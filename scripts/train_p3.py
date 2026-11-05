"""P3 (docs/guide/09 §9.5): user adapter from base Gemma — LoRA + query rows + scorer on the user's human labels.

    uv run python scripts/train_p3.py --data data/processed/llm_response_coding.jsonl \
        [--init-head bundles/P1_llm_response_coding_e4b_seed0] [--epochs 2] [--limit-train 32 --limit-val 16]
    # from the general head, on a seeded sample of 100 train rows (release criterion 8):
    uv run python scripts/train_p3.py --data ... --init-bundle bundles/general_v3c_e4b_r8_seed0_nt --n-train 100 ...

Trains on `train`, early-stops on `val`, fits one temperature per (type, bucket) and the
Noul bias on gold_calib, and writes calibrated predictions for every question of gold_calib
and gold_test to data/predictions/ and the bundle to bundles/.
The training mix is checked against configs/holdout.yaml for the user adapter.

--init-bundle starts from a trained bundle instead of base: its LoRA (rank, alpha and target modules come from the
bundle), query-token rows and head, standardisation included, are trained further. Its step-0 validation counts as a
candidate, so a fine-tune that never improves on the bundle returns the bundle's own weights.
"""
import argparse, collections, json, pathlib, random, time

import torch

from decisivelm.data.dataset import load_records
from decisivelm.data.mix import MIX_TIERS, check_mix, load_holdouts
from decisivelm.calibrate import calibration_json, fit_calibration, predictions, question_logits, write_predictions
from decisivelm.head import Scorer
from decisivelm.infer import QUERY_IDS, DecisiveLM
from decisivelm.lora import apply_lora, load_lora_state
from decisivelm.model import backbone_profile, fp32_routers, load_backbone, vision_kwargs
from decisivelm.render import RENDER_VERSION
from decisivelm.train import make_items, micro_batches, train_adapter

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True,
                    help="the rows: trains on train, early-stops on val, calibrates on gold_calib")
    ap.add_argument("--backbone", default="e4b", help="e4b or 26b-a4b (configs/model.yaml)")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank")
    ap.add_argument("--alpha", type=float, default=16, help="LoRA alpha (the update is scaled by alpha / rank)")
    ap.add_argument("--dropout", type=float, default=0.05, help="dropout on the LoRA input")
    ap.add_argument("--epochs", type=int, default=2, help="passes over the train rows (early stopping can end sooner)")
    ap.add_argument("--rows-per-step", type=int, default=16,
                    help="requests whose gradients are summed per optimiser step")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="packed tokens per micro-batch (2048 on a Mac: one request at a time)")
    ap.add_argument("--lr-lora", type=float, default=2e-4,
                    help="learning rate of the LoRA weights and query-token rows")
    ap.add_argument("--lr-head", type=float, default=1e-3,
                    help="learning rate of the scorer head (keep near 1e-5 with --init-head)")
    ap.add_argument("--warmup-frac", type=float, default=0.03,
                    help="share of the steps spent warming the learning rate up")
    ap.add_argument("--no-checkpointing", action="store_true", help="faster when memory allows (e.g. 128 GB unified)")
    ap.add_argument("--eval-every", type=int, default=40, help="optimiser steps between validation checks")
    ap.add_argument("--patience", type=int, default=3, help="checks without improvement before stopping")
    ap.add_argument("--init-head", type=pathlib.Path, help="bundle whose scorer.pt warm-starts the head (e.g. P1)")
    ap.add_argument("--init-bundle", type=pathlib.Path, help="start from this bundle's LoRA and head (e.g. a general head)")
    ap.add_argument("--n-train", type=int, help="train on a seeded random sample of this many train rows (--seed)")
    ap.add_argument("--limit-train", type=int,
                    help="keep only the first N train rows (a quick test; --n-train samples instead)")
    ap.add_argument("--limit-val", type=int, help="early-stop on a seeded random sample of this many val rows")
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling, shuffling and initialisation")
    ap.add_argument("--select", choices=["loss", "cal_loss", "acc"], default="loss",
                    help="val criterion for early stopping and the kept adapter (decisivelm.train.val_metrics)")
    ap.add_argument("--predict-best", nargs="*", default=[], choices=["loss", "cal_loss", "acc"],
                    help="also write predictions and a bundle from each of these criteria's best snapshots "
                         "(named <name>_by<criterion>), to compare selection rules on gold_test")
    ap.add_argument("--name",
                    help="run name for the bundle and predictions (default P3_<data>_<backbone>_r<rank>_seed<seed>)")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="backbone compute dtype; float32 removes 26B A4B's routing noise (results/invariance.md)")
    ap.add_argument("--fp32-router", action="store_true", help="MoE routers in fp32 (decisivelm.model.fp32_routers)")
    ap.add_argument("--standardize-rows", type=int, default=0,
                    help="z-score the head's input on the base model's states for this many train rows "
                         "(Scorer.standardize; needed on 26B A4B). An --init-head bundle brings its own.")
    ap.add_argument("--image-budget", type=int, help="soft tokens per image (default: the backbone profile's)")
    ap.add_argument("--media-root", type=pathlib.Path, help="directory relative media paths resolve against")
    ap.add_argument("--adapter", choices=["user", "general"], default="user",
                    help="holdouts to enforce; general also limits train rows to release-tier data")
    a = ap.parse_args()
    if a.standardize_rows and (a.init_head or a.init_bundle):
        raise SystemExit("--init-head and --init-bundle bring their head's standardisation; drop --standardize-rows")
    if a.init_head and a.init_bundle:
        raise SystemExit("--init-bundle already brings a head; drop --init-head")
    if a.init_bundle:
        lora = json.loads((a.init_bundle / "calibration.json").read_text())["lora"]
        a.rank, a.alpha = lora["rank"], lora["alpha"]

    rows = load_records(a.data)
    split = collections.defaultdict(list)
    for r in rows:
        split[r.split].append(r)
    check_mix(split["train"], load_holdouts(), adapter=a.adapter,
              tiers=MIX_TIERS["release" if a.adapter == "general" else "research"])
    train_rows, val_rows = split["train"][:a.limit_train], split["val"]
    if a.n_train and a.n_train < len(train_rows):   # source order can be grouped by label: sample, don't slice
        train_rows = [train_rows[i] for i in sorted(random.Random(a.seed).sample(range(len(train_rows)), a.n_train))]
    if a.limit_val and a.limit_val < len(val_rows):   # source order can be grouped by label: sample, don't slice
        val_rows = [val_rows[i] for i in sorted(random.Random(a.seed).sample(range(len(val_rows)), a.limit_val))]

    profile = backbone_profile(a.backbone)
    if a.init_bundle and lora["target_regex"] != profile["lora_target_regex"]:
        raise SystemExit(f"{a.init_bundle}: LoRA targets {lora['target_regex']!r}, not the {a.backbone} profile's")
    backbone, tok = load_backbone(a.backbone, dtype=getattr(torch, a.dtype))
    if a.fp32_router:
        print(f"{fp32_routers(backbone)} routers in fp32", flush=True)
    apply_lora(backbone, profile["lora_target_regex"], rank=a.rank, alpha=a.alpha,
               query_ids=list(QUERY_IDS.values()), dropout=a.dropout)
    device = next(backbone.parameters()).device
    scorer = Scorer(profile["hidden_size"]).to(device)
    if a.init_head:
        scorer.load_state_dict(torch.load(a.init_head / "scorer.pt", weights_only=True))
    if a.init_bundle:
        load_lora_state(backbone, torch.load(a.init_bundle / "adapter.pt", weights_only=True, map_location=device))
        scorer.load_state_dict(torch.load(a.init_bundle / "scorer.pt", weights_only=True))
    vision = vision_kwargs(a.backbone, rows, a.image_budget, a.media_root)
    dlm = DecisiveLM(backbone, tok, scorer, adapter=a.name,
                     **vision)
    if a.standardize_rows:   # LoRA B starts at zero, so these are the base model's states
        sample = make_items(dlm, train_rows[:a.standardize_rows])
        with torch.no_grad():
            scorer.standardize(torch.cat([dlm.query_states([it.tree for it in b])[1]
                                          for b in micro_batches(sample, a.max_tokens, random.Random(0))]))
        print(f"standardized the head's input on {len(sample)} train rows", flush=True)
    n_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"trainable adapter parameters: {n_lora / 1e6:.1f}M; train {len(train_rows)}, val {len(val_rows)}", flush=True)

    t0 = time.perf_counter()
    result = train_adapter(dlm, make_items(dlm, train_rows), make_items(dlm, val_rows), epochs=a.epochs,
                           rows_per_step=a.rows_per_step, max_tokens=a.max_tokens, lr_lora=a.lr_lora,
                           lr_head=a.lr_head, warmup_frac=a.warmup_frac, eval_every=a.eval_every, patience=a.patience, seed=a.seed,
                           checkpointing=not a.no_checkpointing, select=a.select,
                           log=lambda s: print(s, flush=True))
    print(f"training took {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if a.limit_train:   # timing run: no calibration or predictions
        return
    name = a.name or f"P3_{a.data.stem}_{a.backbone}_r{a.rank}_seed{a.seed}"
    best = {m: {"value": b["value"], "step": b["step"]} for m, b in result["best"].items()}
    print(f"best per criterion {best}", flush=True)
    runs = [(name, result["adapter"], result["scorer"], a.select)]
    for m in a.predict_best:
        if m != a.select:
            st = result["best"][m]["state"]
            runs.append((f"{name}_by{m}", st["adapter"], st["scorer"], m))
    for run_name, adapter, scorer_state, criterion in runs:
        load_lora_state(backbone, adapter)
        scorer.load_state_dict(scorer_state)
        backbone.eval(); scorer.eval()
        # Every question of every gold row, cut out of the packed trees in leaf order (decisivelm.calibrate);
        # temperatures per (type, bucket) and the Noul bias fitted on gold_calib (docs/guide/10).
        qs = {}
        for s in ("gold_calib", "gold_test"):
            rec = {r.id: r for r in split[s]}
            qs[s] = []
            for b in micro_batches(make_items(dlm, split[s]), a.max_tokens, random.Random(0)):
                tree, logits = dlm.logits([it.tree for it in b])
                leaf_request = tree.request_of_question[tree.leaf_question]
                for k, it in enumerate(b):
                    qs[s] += question_logits(rec[it.id], logits[leaf_request == k])
        calib = fit_calibration(qs["gold_calib"])
        print(f"{run_name}: selected by val {criterion} at step {best[criterion]['step']}; "
              f"calibration {calibration_json(calib)}", flush=True)

        pred_path = REPO / "data" / "predictions" / f"{run_name}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_predictions(pred_path, predictions(qs["gold_calib"] + qs["gold_test"], calib))
        bundle = REPO / "bundles" / run_name
        bundle.mkdir(parents=True, exist_ok=True)
        torch.save(adapter, bundle / "adapter.pt")
        torch.save(scorer_state, bundle / "scorer.pt")
        (bundle / "calibration.json").write_text(json.dumps({
            **calibration_json(calib), "backbone": profile,
            "lora": {"rank": a.rank, "alpha": a.alpha, "dropout": a.dropout, "target_regex": profile["lora_target_regex"]},
            "render_version": RENDER_VERSION, "init_head": str(a.init_head) if a.init_head else None,
            "init_bundle": str(a.init_bundle) if a.init_bundle else None, "n_train": len(train_rows),
            "dtype": a.dtype, "fp32_router": a.fp32_router, "standardize_rows": a.standardize_rows,
            "data": str(a.data), "adapter_kind": a.adapter, "seed": a.seed, "image_budget": vision.get("image_budget"),
            "question_types": dict(collections.Counter(q.type for r in train_rows for q in r.questions.values())),
            "select": criterion, "best_step": best[criterion]["step"],
            "best_val": best[criterion]["value"], "best": best, "history": result["history"]}, indent=1))
        print(f"wrote {pred_path.relative_to(REPO)} and {bundle.relative_to(REPO)}/", flush=True)
    if torch.cuda.is_available():   # what a card must hold: PyTorch's peak, and its cache (nvidia-smi adds the context)
        print(f"peak GPU memory: allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
