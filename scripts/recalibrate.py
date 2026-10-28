"""Refit a trained head's calibration (decisivelm.calibrate.fit_calibration, as it is now) without running the model.

    uv run python scripts/recalibrate.py --name general_v2_ep2_e4b_r8_seed0 [--suffix _recal]

Each prediction is a readout of the head's raw logit through the bundle's stored calibration, so the logits come back
exactly (up to a per-question constant, which softmax ignores): log p × T for Choice and Score, T·logit(p) − b for
Noul. The calibration is refitted on the same gold_calib rows of the bundle's mix that training used (human-labelled
only), and every prediction file of the run (data/predictions/<name>.jsonl and <name>__*.jsonl) is rewritten
through it as <name><suffix>….jsonl, with the new calibration in data/predictions/<name><suffix>.calibration.json.
The bundle itself is left as it was.

    uv run python scripts/recalibrate.py --name general_v3c_e4b_r8_seed0 --new-task multinli --suffix _nt

With --new-task, the bundle's own temperatures are kept and the Choice temperatures and the Score temperatures are each
scaled by one factor fitted against the human majority answer on that dataset's gold_calib rows (data/processed/<stem>.jsonl, predictions
in data/predictions/<name>__<stem>.jsonl): a task outside the mix, so the factor says how the head behaves on new tasks
(decisivelm.calibrate.fit_new_task_scale). A new bundle, bundles/<name><suffix>/, is written with that calibration
(adapter and head hard-linked), so the bundle that is shipped carries it. The same step calibrates a released head
for a task of your own on about 100 labelled items (guide §10.6).
"""
import argparse
import json
import os
import pathlib

from decisivelm.calibrate import (QuestionLogits, calibration_json, fit_calibration, invert,
                                  new_task_scales, predictions, scaled, write_predictions)
from decisivelm.data.schema import Record
from decisivelm.head import Calibration
from decisivelm.train import _target

REPO = pathlib.Path(__file__).resolve().parent.parent


def question_logits(path: pathlib.Path, calib: Calibration, records: dict[str, Record]) -> list[QuestionLogits]:
    qs = []
    for line in open(path):
        row = json.loads(line)
        for qid, ans in row["answers"].items():
            names, logits = invert(ans, calib)
            r = records.get(row["id"])
            target = _target(r.questions[qid], r.targets.get(qid)) if r is not None else None
            if target is not None and ans["type"] != "noul":   # in the prediction's option order
                order = list(r.questions[qid].criteria) if ans["type"] == "choice" else \
                    [str(i) for i in range(len(r.questions[qid].criteria))]
                target = target[[order.index(k) for k in names]]
            qs.append(QuestionLogits(row["id"], row["split"], qid, ans["type"], names, logits, target))
    return qs


def gold_calib(path: pathlib.Path) -> dict[str, Record]:
    records = {}
    for line in open(path, encoding="utf-8"):
        if '"split": "gold_calib"' in line or '"split":"gold_calib"' in line:
            r = Record.model_validate_json(line)
            if r.split == "gold_calib":
                records[r.id] = r
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--suffix", default="_recal")
    ap.add_argument("--new-task", metavar="STEM", help="fit Choice and Score scales on this unseen dataset's gold_calib")
    ap.add_argument("--replace", action="store_true",
                    help="rewrite an existing new-task bundle's calibration.json (its weights must be links to the source's)")
    a = ap.parse_args()
    meta = json.loads((REPO / "bundles" / a.name / "calibration.json").read_text())
    old = Calibration(temperatures=meta["temperatures"], b_noul=meta["b_noul"])
    main_path = REPO / "data" / "predictions" / f"{a.name}.jsonl"
    out_dir = main_path.parent
    if a.new_task:
        records = gold_calib(REPO / "data" / "processed" / f"{a.new_task}.jsonl")
        calib_qs = [q for q in question_logits(out_dir / f"{a.name}__{a.new_task}.jsonl", old, records)
                    if q.split == "gold_calib" and q.target is not None]
        s = new_task_scales(calib_qs, old)
        new = scaled(old, s)
        print(f"{a.name}: new-task scales {s} on {len(calib_qs)} {a.new_task} gold_calib questions\n  "
              f"new {calibration_json(new)}", flush=True)
        # a downloaded bundle has no predictions on its mix: then only the run's other prediction files are rewritten
        qs = question_logits(main_path, old, {}) if main_path.exists() else []
        bundle = REPO / "bundles" / f"{a.name}{a.suffix}"
        if a.replace and bundle.exists():
            for f in ("adapter.pt", "scorer.pt"):
                if not os.path.samefile(bundle / f, REPO / "bundles" / a.name / f):
                    raise SystemExit(f"{bundle / f} is not {a.name}'s {f}: not replacing")
        else:
            bundle.mkdir(exist_ok=False)
            for f in ("adapter.pt", "scorer.pt"):
                os.link(REPO / "bundles" / a.name / f, bundle / f)
        method = {"method": "new_task_scale", "scales": s, "fitted_on": f"{a.new_task} gold_calib",
                  "questions": len(calib_qs), "from_bundle": a.name, "from_temperatures": meta["temperatures"]}
        tmp = bundle / "calibration.json.tmp"   # a training run may be reading the bundle
        tmp.write_text(json.dumps({**meta, **calibration_json(new), "calibration": method}, indent=1))
        os.replace(tmp, bundle / "calibration.json")
        print(f"  wrote {bundle.relative_to(REPO)}/", flush=True)
    else:
        records = gold_calib(REPO / meta["data"])
        qs = question_logits(main_path, old, records)
        calib_qs = [q for q in qs if q.split == "gold_calib" and q.target is not None]
        new = fit_calibration(calib_qs)
        print(f"{a.name}: refitted on {len(calib_qs)} gold_calib questions\n  old {calibration_json(old)}\n  "
              f"new {calibration_json(new)}", flush=True)
    (out_dir / f"{a.name}{a.suffix}.calibration.json").write_text(json.dumps(calibration_json(new), indent=1))
    for path in [main_path] * main_path.exists() + sorted(out_dir.glob(f"{a.name}__*.jsonl")):
        rows = predictions(qs if path == main_path else question_logits(path, old, {}), new)
        out = out_dir / path.name.replace(a.name, a.name + a.suffix, 1)
        write_predictions(out, rows)
        print(f"  wrote {out.relative_to(REPO)} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
