"""Frozen-suite set: Measuring Hate Speech (Kennedy et al. 2020; docs/guide/14 §14.4; docs/RELEASE.md).

- state: the comment text
- ten questions, one per survey item of the study. In the study an annotator first named the group or groups a
  comment is about and then rated it towards them; here each question says "the group or groups the comment is about".
  Nine are Scores on 0–4, ordered from least to most hateful as the data codes them; the hate speech item (no /
  unclear / yes) is a Noul whose target counts "unclear" as half a yes.
- targets: the share of the comment's raters at each level (n_raters). Only the comment id, text and the ten item
  columns are read: annotator demographics, target identities and IRT estimates never reach a row.
- eval portion (the frozen suite): 20% of the comments with ≥ 3 raters, chosen by a hash of the comment id, are
  gold_test in family `measuring_hate_speech-eval` (held out in configs/holdout.yaml). Every other comment is train
  (val 10%) in family `measuring_hate_speech-base`, for later mixes.
"""
import hashlib
import pathlib

import pandas as pd

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "measuring_hate_speech"
GROUP = "the group or groups the comment is about"
AGREE = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
SCORES = {
    "sentiment": ("What is the sentiment of the comment?",
                  ["Strongly positive", "Somewhat positive", "Neutral", "Somewhat negative", "Strongly negative"]),
    "respect": (f"Is the comment respectful towards {GROUP}?",
                ["Strongly respectful", "Respectful", "Neither respectful nor disrespectful", "Disrespectful",
                 "Strongly disrespectful"]),
    "insult": (f"How much do you agree: the comment is insulting towards {GROUP}.", AGREE),
    "humiliate": (f"How much do you agree: the comment is humiliating towards {GROUP}.", AGREE),
    "status": (f"How much do you agree: the comment states that {GROUP} are inferior to another group.", AGREE),
    "dehumanize": (f"How much do you agree: the comment dehumanizes {GROUP}, for example by comparing them to an "
                   "animal, insect, pest, disease or filth.", AGREE),
    "violence": (f"How much do you agree: the comment calls for using violence against {GROUP}.", AGREE),
    "genocide": (f"How much do you agree: the comment calls for the deliberate killing of a large number of people "
                 f"from {GROUP}.", AGREE),
    "attack_defend": (f"Is the comment attacking or defending {GROUP}?",
                      ["Strongly defending", "Defending", "Neither attacking nor defending", "Attacking",
                       "Strongly attacking"]),
}
HATE_SPEECH = ("The comment contains hate speech: bias-motivated, hostile and malicious language aimed at a person or "
               "group because of their actual or perceived innate characteristics.")
COLUMNS = ["comment_id", "text", *SCORES, "hatespeech"]
EVAL_SHARE, VAL_SHARE, MIN_EVAL_RATERS = 0.20, 0.10, 3


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def _shares(values, levels: int) -> HumanTarget:
    counts = [0] * levels
    for v in values:
        counts[int(v)] += 1
    return HumanTarget(probabilities={str(i): c / len(values) for i, c in enumerate(counts)}, n_raters=len(values))


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    df = pd.read_parquet(pathlib.Path(source_dir) / "train.parquet", columns=COLUMNS)
    questions = {k: {"type": "score", "instructions": ins, "criteria": levels} for k, (ins, levels) in SCORES.items()}
    questions["hatespeech"] = {"type": "noul", "instructions": HATE_SPEECH}
    out = []
    for cid, g in df.groupby("comment_id", sort=True):
        n = len(g)
        targets = {k: QuestionTargets(human=_shares(g[k], 5)) for k in SCORES}
        targets["hatespeech"] = QuestionTargets(human=HumanTarget(noul=float(g.hatespeech.mean() / 2), n_raters=n))
        source_id = f"{int(cid):05d}"
        if n >= MIN_EVAL_RATERS and _u("eval:" + source_id) < EVAL_SHARE:
            family, split = f"{SOURCE}-eval", "gold_test"
        else:
            family, split = f"{SOURCE}-base", "val" if _u("val:" + source_id) < VAL_SHARE else "train"
        text = str(g.text.iloc[0])
        state, truncated = truncate_state(text, tokenizer, max_state_tokens) if tokenizer else (text, False)
        out.append(Record(
            id=f"{SOURCE}-{source_id}",
            lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=family),
            license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=state, questions=questions,
            targets=targets,
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split=split))
    return out
