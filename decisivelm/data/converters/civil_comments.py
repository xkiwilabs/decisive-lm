"""Noul seed with human soft labels: Civil Comments (docs/guide/14 §14.2).

One Noul per attribute; the target is the share of raters who marked it, so the
labels carry real human disagreement for calibration. Statements follow the
Jigsaw annotation definitions.

The source has ~1.8M training comments, so each portion is a uniform random
sample (fixed seed), which keeps the natural base rates: source train → train,
source validation → val, source test → gold_calib / gold_test (40 / 60) — the
held-out calibration eval portion (configs/holdout.yaml).
"""
import hashlib
import pathlib

import numpy as np
import pandas as pd

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "civil_comments"
ATTRIBUTES = {
    "toxicity": "The comment is toxic: rude, disrespectful or unreasonable, likely to make someone leave a discussion",
    "severe_toxicity": "The comment is very toxic: very hateful, aggressive or disrespectful",
    "obscene": "The comment contains obscene or vulgar language",
    "threat": "The comment contains a threat of harm against a person or group",
    "insult": "The comment insults a person or group",
    "identity_attack": "The comment attacks or demeans people because of a group identity such as race, religion, "
                       "gender or sexual orientation",
    "sexual_explicit": "The comment contains sexually explicit content",
}
CALIB_SHARE = 0.40


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df[df.text.str.strip().astype(bool)]
    if n >= len(df):
        return df
    return df.iloc[np.sort(np.random.default_rng(seed).choice(len(df), n, replace=False))]


def convert(source_dir: pathlib.Path, n_train: int = 50_000, n_val: int = 5_000, n_eval: int = 10_000,
            seed: int = 0, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    columns = ["text", *ATTRIBUTES]
    train = pd.concat([pd.read_parquet(p, columns=columns) for p in sorted(source_dir.glob("train-*.parquet"))],
                      ignore_index=True)
    portions = [("train", _sample(train, n_train, seed)),
                ("validation", _sample(pd.read_parquet(source_dir / "validation.parquet", columns=columns), n_val, seed)),
                ("test", _sample(pd.read_parquet(source_dir / "test.parquet", columns=columns), n_eval, seed))]
    questions = {a: {"type": "noul", "instructions": s} for a, s in ATTRIBUTES.items()}
    out = []
    for source_split, df in portions:
        for idx, d in zip(df.index, df.to_dict("records")):
            source_id = f"{source_split}-{idx:07d}"
            if source_split == "test":
                u = int(hashlib.sha256(f"{SOURCE}:{source_id}".encode()).hexdigest()[:15], 16) / 16 ** 15
                split = "gold_calib" if u < CALIB_SHARE else "gold_test"
            else:
                split = {"train": "train", "validation": "val"}[source_split]
            state, truncated = (truncate_state(d["text"], tokenizer, max_state_tokens) if tokenizer
                                else (d["text"], False))
            out.append(Record(
                id=f"{SOURCE}-{source_id}",
                lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=f"{SOURCE}-base"),
                license="CC0-1.0", tier="R", modality=["text"], language="en", state=state, questions=questions,
                targets={a: QuestionTargets(human=HumanTarget(noul=float(d[a]))) for a in ATTRIBUTES},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
                split=split))
    return out
