"""Task calibration of any method's predictions on a dataset (decisivelm.calibrate.fit_task_calibration; guide 10).

    uv run python scripts/task_calibrate.py --data data/processed/wiki_toxicity.jsonl \
        --predictions data/predictions/teacher_wiki_toxicity__gemma4_e4b.jsonl \
                      data/predictions/general_v4a_e4b_r8_seed0__wiki_toxicity.jsonl

For each prediction file, the probabilities are read back as logits (log p, so the method's own calibration stays in),
a scale and offsets are fitted per question on the dataset's gold_calib rows, and every row is rewritten through them
as <file>_task.jsonl, ready for scripts/eval_dataset.py. Give every method compared the same step, P0 included.
The fitted calibration is printed and saved next to it as <file>_task.calibration.json.

--calib-rows N fits on a random N of the gold_calib rows instead (seeded by --seed), the step a user takes with a
few hundred ratings; the outputs are then <file>_task_n<N>_s<seed>.jsonl and .calibration.json.
"""
import argparse
import json
import pathlib
import random

from decisivelm.calibrate import (MIN_TASK, QuestionLogits, calibration_json, fit_task_calibration, invert,
                                  predictions, write_predictions)
from decisivelm.data.dataset import load_records
from decisivelm.head import Calibration, task_key
from decisivelm.train import _target


def question_logits(path: pathlib.Path, records: dict) -> list[QuestionLogits]:
    qs = []
    for line in open(path):
        row = json.loads(line)
        r = records.get(row["id"])
        if r is None:
            continue
        for qid, ans in row["answers"].items():
            q = r.questions[qid]
            names, logits = invert(ans, Calibration())
            target = _target(q, r.targets.get(qid))
            if target is not None and ans["type"] != "noul":   # in the prediction's option order
                order = list(q.criteria) if ans["type"] == "choice" else [str(i) for i in range(len(q.criteria))]
                target = target[[order.index(k) for k in names]]
            qs.append(QuestionLogits(row["id"], row["split"], qid, ans["type"], names, logits, target, task_key(q)))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--predictions", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--calib-rows", type=int, help="fit on a random sample of this many gold_calib rows")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    records = {r.id: r for r in load_records(a.data)}
    for path in a.predictions:
        qs = question_logits(path, records)
        fit = [q for q in qs if q.split == "gold_calib"]
        if a.calib_rows:
            ids = sorted({q.id for q in fit})
            keep = set(random.Random(a.seed).sample(ids, min(a.calib_rows, len(ids))))
            fit = [q for q in fit if q.id in keep]
        calib = Calibration(tasks=fit_task_calibration(fit, Calibration(),
                                                       min_questions=min(MIN_TASK, a.calib_rows or MIN_TASK)))
        if not calib.tasks:
            print(f"{path.name}: no question has enough gold_calib rows; nothing written", flush=True)
            continue
        out = path.with_name(path.stem + (f"_task_n{a.calib_rows}_s{a.seed}" if a.calib_rows else "_task") + ".jsonl")
        write_predictions(out, predictions(qs, calib))
        out.with_suffix(".calibration.json").write_text(json.dumps(calibration_json(calib), indent=1))
        print(f"{path.name}: {json.dumps(calib.tasks)}\n  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
