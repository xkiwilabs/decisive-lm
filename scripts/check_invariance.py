"""How far a bundle's answers move when only the packing changes (docs/guide/11 §11.7; decisivelm/invariance.py).

    uv run python scripts/check_invariance.py --bundle bundles/P1_llm_response_coding_26b-a4b_seed0 \
        --backbone 26b-a4b --data data/processed/llm_response_coding.jsonl [--n 200] [--out results/...md]

Per question type and variant (options reversed, for Choice only; another request packed first; one more question;
`repeat` = the identical pass again, a determinism check): the largest |Δp| per item, answer flips, rank stability of
the top probability, and churn in the least-confident 20% a human would review. --dtype float32 shows how much is
bf16 rounding.
"""
import argparse, json, pathlib, random

import torch

from decisivelm.bundle import attach_bundle
from decisivelm.data.dataset import load_records
from decisivelm.invariance import VARIANTS, answer_probabilities, compare, variant_requests
from decisivelm.model import backbone_profile, fp32_routers, load_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=pathlib.Path, required=True)
    ap.add_argument("--backbone", default="e4b")
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="gold_test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], help="default: the dtype the bundle was trained in")
    ap.add_argument("--fp32-router", action="store_true", help="MoE routers in fp32 (decisivelm.model.fp32_routers)")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    calib = json.loads((a.bundle / "calibration.json").read_text())
    profile = backbone_profile(a.backbone)
    if profile["revision"] != calib["backbone"]["revision"]:
        raise SystemExit(f"bundle was trained on {calib['backbone']['model_id']}@{calib['backbone']['revision']}")
    backbone, tok = load_backbone(a.backbone, dtype=getattr(torch, a.dtype or calib.get("dtype", "bfloat16")))
    routers = fp32_routers(backbone) if a.fp32_router or calib.get("fp32_router") else 0
    dlm = attach_bundle(a.bundle, backbone, tok)

    rows = [r for r in load_records(a.data) if r.split == a.split]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    reqs = [{"state": r.state, "questions": r.questions} for r in rows]
    by_type = {}
    for i, r in enumerate(rows):
        for qid, q in r.questions.items():
            by_type.setdefault(q.type, []).append((i, qid))
    lines = [f"{a.bundle.name} on {profile['model_id']}@{profile['revision'][:8]}, "
             + ", ".join(f"{len(c)} {t}" for t, c in by_type.items()) + f" questions from {a.split} (seed {a.seed}), "
             f"dtype {next(backbone.parameters()).dtype}{f', {routers} routers in fp32' if routers else ''}. "
             "A Noul's answer is yes or no; options are reversed for Choice questions only.", "",
             "| Type | Variant | max \\|Δp\\| | mean | p99 | answer flips | top-p rank ρ | review-set churn (20%) |",
             "|---|---|---|---|---|---|---|---|"]
    for qtype, cases in by_type.items():
        base = [answer_probabilities(dlm, [reqs[i]], 0, qid) for i, qid in cases]
        results = {"repeat": compare(base, [answer_probabilities(dlm, [reqs[i]], 0, qid) for i, qid in cases])}
        for name in VARIANTS:
            if name == "reversed_options" and qtype != "choice":
                continue
            moved = []
            for i, qid in cases:
                requests, k = variant_requests(reqs[i], qid, name, reqs[(i + 1) % len(reqs)])
                moved.append(answer_probabilities(dlm, requests, k, qid))
            results[name] = compare(base, moved)
        for name, r in results.items():
            lines.append(f"| {qtype} | {name} | {r['max_dp']:.2e} | {r['mean_dp']:.2e} | {r['p99_dp']:.2e} | "
                         f"{r['flips']} / {r['n']} | {r['top_p_rho']:.4f} | {r['review_churn']:.1%} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
