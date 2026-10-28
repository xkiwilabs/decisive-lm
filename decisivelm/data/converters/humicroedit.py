"""Humicroedit (SemEval-2020 Task 7, subtask 1; Hossain et al. 2020): news headlines with one word edited to make them
funny, each graded by about 5 crowd workers from 0 (not funny) to 3 (funny). FunLines (the task's extra training
data) is graded the same way.

Human ratings with several raters per item from a kind of judgement unrelated to the suite's tasks (docs/RELEASE.md,
change log 2026-10-26): how much raters agree varies a lot from headline to headline.

- state: the original headline and the edited one
- one Score over the 4 grades
- target: the share of the headline's graders at each grade (n_raters); headlines with fewer than MIN_RATERS graders
  are dropped (some FunLines rows have one), and so are the few FunLines headlines with a grade off the scale (4)
- splits: the task's own splits, which never share an original headline: its test split is gold, 60% gold_test and
  40% gold_calib by a hash of the original headline; dev is val; train and FunLines are train
"""
import collections
import csv
import hashlib
import io
import pathlib
import re
import zipfile

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "humicroedit"
ARCHIVE = "semeval-2020-task-7-dataset.zip"
FUNNY = {"type": "score",
         "instructions": "How funny is the edited headline?",
         "criteria": ["Not funny", "Slightly funny", "Moderately funny", "Funny"]}
FILES = (("train", "train"), ("dev", "val"), ("test", "gold"), ("train_funlines", "train"))
MIN_RATERS, CALIB_SHARE = 3, 0.4
_TAG = re.compile(r"<([^<>]*)/>")


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def _state(original: str, edit: str) -> str:
    return (f"Original headline: {_TAG.sub(lambda m: m.group(1), original)}\n"
            f"Edited headline: {_TAG.sub(edit, original)}")


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    out = []
    with zipfile.ZipFile(pathlib.Path(source_dir) / ARCHIVE) as z:
        for name, split in FILES:
            with z.open(f"semeval-2020-task-7-dataset/subtask-1/{name}.csv") as f:
                rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
            for d in rows:
                grades = [int(c) for c in d["grades"]]
                if len(grades) < MIN_RATERS or max(grades) > 3:
                    continue
                counts = collections.Counter(grades)
                key = hashlib.sha256(d["original"].encode()).hexdigest()[:12]
                text = _state(d["original"], d["edit"])
                state, truncated = truncate_state(text, tokenizer, max_state_tokens) if tokenizer else (text, False)
                out.append(Record(
                    id=f"{SOURCE}-{d['id']}",
                    lineage=Lineage(source_dataset=SOURCE, source_id=d["id"], family_id=f"{SOURCE}-{key}"),
                    license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=state,
                    questions={"funny": FUNNY},
                    targets={"funny": QuestionTargets(human=HumanTarget(
                        probabilities={str(i): counts[i] / len(grades) for i in range(4)}, n_raters=len(grades)))},
                    preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens,
                                          truncated=truncated),
                    split=split if split != "gold" else "gold_calib" if _u(key) < CALIB_SHARE else "gold_test"))
    return out
