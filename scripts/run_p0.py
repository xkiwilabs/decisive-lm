"""P0 (docs/guide/09 §9.3): prompt the frozen backbone per question and read label log-probs (docs/guide/12).

    uv run python scripts/run_p0.py --data data/processed/llm_response_coding.jsonl [--backbone e4b]

Resumable: raw per-item distributions go to data/predictions/P0_<data>_<backbone>_raw.jsonl.
When every item is done, one temperature is fitted on gold_calib (docs/guide/10) and calibrated
predictions are written next to it, ready for compare_coding.py.
"""
import argparse, json, math, pathlib, time

import torch

from decisivelm.data.dataset import load_records
from decisivelm.eval import fit_temperature
from decisivelm.model import backbone_profile, load_causal_lm
from decisivelm.preprocess import truncate_state
from decisivelm.synth.label import LocalLM, choice_labels, prompt_template

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True,
                    help="the rows to label (a data/processed/<name>.jsonl file)")
    ap.add_argument("--backbone", default="e4b",
                    help="the frozen model that answers: e4b or 26b-a4b (configs/model.yaml)")
    ap.add_argument("--question", default="intent", help="the id of the Choice question to answer (one per run)")
    ap.add_argument("--splits", nargs="+", default=["gold_calib", "gold_test"],
                    help="splits to answer; gold_calib fits the temperature")
    ap.add_argument("--chunk", type=int, default=48, help="rows prompted per batch; progress is saved after each")
    a = ap.parse_args()

    rows = [r for r in load_records(a.data) if r.split in a.splits]
    name = f"P0_{a.data.stem}_{a.backbone}"
    raw_path = REPO / "data" / "predictions" / f"{name}_raw.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(raw_path)} if raw_path.exists() else set()
    todo = [r for r in rows if r.id not in done]
    print(f"{len(done)} done, {len(todo)} to go")

    if todo:
        model, tok = load_causal_lm(a.backbone)
        scorer = LocalLM(model, tok).scorer(len(todo[0].questions[a.question].criteria))
        _, prompt_hash = prompt_template()
        meta = {"teacher_model": backbone_profile(a.backbone)["model_id"],
                "teacher_revision": backbone_profile(a.backbone)["revision"], "prompt_hash": prompt_hash}
        t0 = time.perf_counter()
        for start in range(0, len(todo), a.chunk):
            chunk = todo[start:start + a.chunk]
            q = chunk[0].questions[a.question]
            states = [truncate_state(r.state, tok, r.preprocess.max_state_tokens)[0] for r in chunk]
            labels = choice_labels(q, states, scorer, tok, [r.id for r in chunk])
            with open(raw_path, "a") as f:
                for r, (p, info) in zip(chunk, labels):
                    f.write(json.dumps({"id": r.id, "split": r.split, "probabilities": p, **info, **meta}) + "\n")
            n = start + len(chunk)
            print(f"{n}/{len(todo)}  {(time.perf_counter() - t0) / n:.2f} s/item", flush=True)

    raw = [json.loads(l) for l in open(raw_path)]
    names = list(rows[0].questions[a.question].criteria)
    target = {r.id: [r.targets[a.question].human.probabilities[c] for c in names] for r in rows}
    calib = [d for d in raw if d["split"] == "gold_calib"]
    logits = torch.tensor([[math.log(max(d["probabilities"][c], 1e-12)) for c in names] for d in calib])
    T = fit_temperature(logits, torch.tensor([target[d["id"]] for d in calib]))
    low = sum(d["allowed_mass"] < 0.5 for d in raw)
    print(f"T = {T:.3f}; mean allowed-label mass {sum(d['allowed_mass'] for d in raw) / len(raw):.3f}; "
          f"{low} items below 0.5")
    with open(raw_path.with_name(f"{name}.jsonl"), "w") as f:
        for d in raw:
            lp = torch.tensor([math.log(max(d["probabilities"][c], 1e-12)) for c in names]) / T
            f.write(json.dumps({"id": d["id"], "split": d["split"],
                                "probabilities": dict(zip(names, torch.softmax(lp, -1).tolist()))}) + "\n")
    print(f"wrote {raw_path.with_name(name + '.jsonl').relative_to(REPO)}")


if __name__ == "__main__":
    main()
