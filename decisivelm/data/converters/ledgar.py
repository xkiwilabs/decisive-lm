"""Long-tail legal Choice seed: LEDGAR from LexGLUE (docs/guide/14 §14.2).

One Choice over all 100 provision types; option names are LexGLUE's label names ("Governing Laws"), with no
descriptions, so the model has to read the names. Targets are one-hot (one label per provision; the labels were
derived semi-automatically from the provisions' headings by the dataset's authors).

Seeded samples keep the natural label frequencies: source train → train, source validation → val, source test →
gold_calib / gold_test (40 / 60) by a stable hash. Public test items are never trained on (docs/guide/08 §8.3).
"""
import pathlib

import numpy as np
import pandas as pd
import yaml

from decisivelm.data.converters.intents import item_split
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "ledgar"
INSTRUCTIONS = "Which type of contract provision is this clause?"


def label_names(readme: pathlib.Path) -> list[str]:
    """The ledgar config's class names from the LexGLUE card's YAML header."""
    header = yaml.safe_load(readme.read_text().split("---")[1])
    (cfg,) = [c for c in header["dataset_info"] if c["config_name"] == "ledgar"]
    (feature,) = [f for f in cfg["features"] if f["name"] == "label"]
    names = feature["dtype"]["class_label"]["names"]
    return [names[str(i)] if str(i) in names else names[i] for i in range(len(names))]


def _sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.iloc[np.sort(np.random.default_rng(seed).choice(len(df), n, replace=False))]


def convert(source_dir: pathlib.Path, n_train: int | None = 20_000, n_val: int | None = 1_000,
            n_test: int | None = 4_000, seed: int = 0, tokenizer=None,
            max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    names = label_names(source_dir / "README.md")
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {n: "" for n in names}}
    out = []
    for source_split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        df = _sample(pd.read_parquet(source_dir / f"{source_split}.parquet"), n, seed)
        for idx, d in zip(df.index, df.to_dict("records")):
            source_id = f"{source_split}-{idx:05d}"
            split = item_split(SOURCE, source_id, source_split)
            if source_split == "train":
                split = "train"     # LexGLUE has its own validation split
            state, truncated = (truncate_state(d["text"], tokenizer, max_state_tokens) if tokenizer
                                else (d["text"], False))
            gold = names[int(d["label"])]
            out.append(Record(
                id=f"{SOURCE}-{source_id}",
                lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=f"{SOURCE}-base"),
                license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=state,
                questions={"provision": question},
                targets={"provision": QuestionTargets(human=HumanTarget(probabilities={c: float(c == gold) for c in names}))},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
                split=split))
    return out
