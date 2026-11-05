"""Run a trained bundle (P1 or P3) through DecisiveLM.system_one and compare with the predictions it wrote.

    uv run python scripts/check_bundle.py --bundle bundles/P3_my_task_e4b_r8_seed0 \
        --data data/processed/my_task.jsonl [--n 20] [--predictions data/predictions/<bundle>.jsonl]

Every question of the first --n gold_test rows that have a prediction is answered through the serving path
(decisivelm.bundle.load_bundle, on the backbone the bundle was trained on) and compared with the prediction training
wrote. The two paths share the tree and the backbone call, so any difference beyond bf16 noise means the bundle
(head, temperature, option order) is wired wrongly. Also checks each response's conformance (docs/guide/11 §11.7).

It passes (exit 0) when every probability is within --tolerance of the written one and every answer that differs was
a near tie in the written prediction (its top two options within 2 × --tolerance). bf16 moves the same request's
probabilities by up to a few hundredths (one walkthrough bundle: max |Δp| 0.078, 19 of 20 answers the same), so the
default tolerance is 0.10 for a bf16 bundle and 0.001 for one trained in float32.
"""
import argparse, json, pathlib

from decisivelm.bundle import load_bundle
from decisivelm.data.dataset import load_records

REPO = pathlib.Path(__file__).resolve().parent.parent


def conformance(ans: dict, q) -> list[str]:
    """Departures from the conformance definition (docs/guide/11 §11.7) for one answer."""
    if ans["type"] == "noul":
        return [] if 0 <= ans["noul"] <= 1 and "confidence" not in ans else ["noul outside [0, 1] or has a confidence"]
    p = ans["probabilities"]
    keys = list(q.criteria) if ans["type"] == "choice" else [str(i) for i in range(len(q.criteria))]
    bad = [] if list(p) == keys else ["probability keys differ from the request's options or levels"]
    if abs(sum(p.values()) - 1) > 1e-6:
        bad.append(f"probabilities sum to {sum(p.values())}")
    if ans["type"] == "choice" and ans["choice"] != max(p, key=p.get):
        bad.append("choice is not the argmax")
    if ans["type"] == "score" and not 0 <= ans["score"] <= len(keys) - 1:
        bad.append("score outside [0, n − 1]")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=pathlib.Path, required=True, help="the bundle directory")
    ap.add_argument("--data", type=pathlib.Path, required=True,
                    help="the rows it was trained on (its gold_test is used)")
    ap.add_argument("--n", type=int, default=20, help="gold_test rows to check")
    ap.add_argument("--predictions", type=pathlib.Path, help="default: data/predictions/<bundle name>.jsonl")
    ap.add_argument("--tolerance", type=float,
                    help="largest |Δp| that passes (default 0.10 for a bf16 bundle, 0.001 for float32)")
    a = ap.parse_args()

    dtype = json.loads((a.bundle / "calibration.json").read_text()).get("dtype", "bfloat16")
    tolerance = a.tolerance if a.tolerance is not None else (0.001 if dtype == "float32" else 0.10)
    dlm = load_bundle(a.bundle)
    path = a.predictions or REPO / "data" / "predictions" / f"{a.bundle.name}.jsonl"
    preds = {d["id"]: d for d in map(json.loads, open(path))}
    rows = [r for r in load_records(a.data) if r.split == "gold_test" and r.id in preds][:a.n]
    worst, agree, n_questions, problems, flips = 0.0, 0, 0, [], []
    for r in rows:
        resp = dlm.system_one({"model": a.bundle.name, "state": r.state, "media": r.media, "questions": r.questions})
        written = preds[r.id].get("answers") or {next(iter(r.questions)): {"probabilities": preds[r.id]["probabilities"]}}
        for qid, ans in resp["answers"].items():
            problems += [f"{r.id} {qid}: {p}" for p in conformance(ans, r.questions[qid])]
            if qid not in written:
                continue
            got, want = (({"yes": ans["noul"], "no": 1 - ans["noul"]}, {"yes": written[qid]["p"], "no": 1 - written[qid]["p"]})
                         if ans["type"] == "noul" else (ans["probabilities"], written[qid]["probabilities"]))
            worst = max(worst, max(abs(got[c] - want[c]) for c in want))
            same = max(got, key=got.get) == max(want, key=want.get)
            top = sorted(want.values(), reverse=True)
            if not same and top[0] - top[1] > 2 * tolerance:
                flips.append(f"{r.id} {qid}: answer changed although the written margin was {top[0] - top[1]:.3f}")
            agree += same
            n_questions += 1
    print(f"{len(rows)} rows, {n_questions} questions: answer agrees on {agree}; max |Δp| = {worst:.2e}; "
          f"last latency {resp['usage']['latency_ms']} ms for {resp['usage']['input_tokens']} tokens, "
          f"{resp['usage']['blocks']} blocks")
    print(f"conformance: {'100%' if not problems else f'{len(problems)} problems'}")
    for p in problems[:20]:
        print(" ", p)
    close = worst <= tolerance and not flips
    print(f"agreement with the written predictions ({dtype}, tolerance {tolerance}): "
          f"{'passes' if close else 'FAILS'}")
    for f in flips[:20]:
        print(" ", f)
    if problems or not close:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
