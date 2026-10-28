"""Task calibration of any method's predictions on a dataset (decisivelm.calibrate.fit_task_calibration; guide 10).

    uv run python scripts/task_calibrate.py --data data/processed/wiki_toxicity.jsonl \
        --predictions data/predictions/teacher_wiki_toxicity__gemma4_e4b.jsonl \
                      data/predictions/general_v4a_e4b_r8_seed0__wiki_toxicity.jsonl

For each prediction file, the probabilities are read back as logits (log p, so the method's own calibration stays in),
a scale and offsets are fitted per question on the dataset's gold_calib rows, and every row is rewritten through them
as <file>_task.jsonl, ready for scripts/eval_dataset.py. Give every method compared the same step, P0 included.
The fitted calibration is printed and saved next to it as <file>_task.calibration.json.
"""
import argparse
import json
import pathlib

from decisivelm.calibrate import (QuestionLogits, calibration_json, fit_task_calibration, invert, predictions,
                                  write_predictions)
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
    a = ap.parse_args()
    records = {r.id: r for r in load_records(a.data)}
    for path in a.predictions:
        qs = question_logits(path, records)
        calib = Calibration(tasks=fit_task_calibration([q for q in qs if q.split == "gold_calib"], Calibration()))
        if not calib.tasks:
            print(f"{path.name}: no question has enough gold_calib rows; nothing written", flush=True)
            continue
        out = path.with_name(path.stem + "_task.jsonl")
        write_predictions(out, predictions(qs, calib))
        out.with_suffix(".calibration.json").write_text(json.dumps(calibration_json(calib), indent=1))
        print(f"{path.name}: {json.dumps(calib.tasks)}\n  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
