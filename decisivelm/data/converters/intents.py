"""Choice seeds: Banking77 and CLINC150 (docs/guide/14 §14.2).

The base schema: one Choice question over every intent, option names are the
dataset's label ids, descriptions are the ids written out ("card_arrival" →
"Card arrival"). Schema explosion (docs/guide/08 §8.10) rewrites these later.

Splits are a stable hash of the item: source train → train / val (10%), source
validation → val, source test → gold_calib / gold_test (40 / 60). Public test
items are never trained on (docs/guide/08 §8.3).
"""
import hashlib
import pathlib

import pandas as pd
import yaml

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

OTHER = "other"
VAL_SHARE, CALIB_SHARE = 0.10, 0.40
CLINC_DOMAINS = ("banking, credit cards, kitchen and dining, home, auto and commute, travel, utilities, work, "
                 "small talk, and questions about the assistant itself")


def item_split(source: str, source_id: str, source_split: str) -> str:
    u = int(hashlib.sha256(f"{source}:{source_id}".encode()).hexdigest()[:15], 16) / 16 ** 15
    if source_split == "train":
        return "val" if u < VAL_SHARE else "train"
    if source_split == "validation":
        return "val"
    if source_split == "test":
        return "gold_calib" if u < CALIB_SHARE else "gold_test"
    raise ValueError(source_split)


def describe(label: str) -> str:
    text = label.replace("_", " ").strip()
    return text[:1].upper() + text[1:].lower()


def _records(source, license, rows, criteria, instructions) -> list[Record]:
    question = {"type": "choice", "instructions": instructions, "criteria": criteria}
    out = []
    for source_split, idx, text, label in rows:
        source_id = f"{source_split}-{idx:05d}"
        out.append(Record(
            id=f"{source}-{source_id}",
            lineage=Lineage(source_dataset=source, source_id=source_id, family_id=f"{source}-base"),
            license=license, tier="R", modality=["text"], language="en", state=str(text),
            questions={"intent": question},
            targets={"intent": QuestionTargets(human=HumanTarget(probabilities={c: float(c == label) for c in criteria}))},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
            split=item_split(source, source_id, source_split)))
    return out


def convert_banking77(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    rows, labels = [], set()
    for split in ("train", "test"):
        df = pd.read_csv(source_dir / f"{split}.csv")
        labels |= set(df.category)
        rows += [(split, i, t, c) for i, (t, c) in enumerate(zip(df.text, df.category))]
    criteria = {c: describe(c) for c in sorted(labels, key=str.lower)}
    return _records("banking77", "CC-BY-4.0", rows, criteria,
                    "Which of these intents does this online-banking customer's message express?")


def convert_clinc150(source_dir: pathlib.Path) -> list[Record]:
    """The 'plus' configuration. Out-of-scope queries become a self-contained `other` option, listed last."""
    source_dir = pathlib.Path(source_dir)
    meta = yaml.safe_load((source_dir / "README.md").read_text().split("---")[1])
    plus = next(c for c in meta["dataset_info"] if c["config_name"] == "plus")
    names = {int(k): v for k, v in plus["features"][1]["dtype"]["class_label"]["names"].items()}
    criteria = {n: describe(n) for _, n in sorted(names.items()) if n != "oos"}
    # Self-contained: a leaf never sees its siblings (docs/guide/06 §6.8), so `other` can't say "none of the above".
    criteria[OTHER] = (f"Out of scope: the request is not one of the specific tasks a virtual assistant handles in "
                       f"{CLINC_DOMAINS}")
    name_of = {k: (OTHER if v == "oos" else v) for k, v in names.items()}
    rows = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(source_dir / f"{split}.parquet")
        rows += [(split, i, t, name_of[int(y)]) for i, (t, y) in enumerate(zip(df.text, df.intent))]
    return _records("clinc150", "CC-BY-3.0", rows, criteria,
                    "Which of these intents does this request to a virtual assistant express?")
