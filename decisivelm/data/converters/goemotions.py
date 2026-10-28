"""Multi-rater affect seed: GoEmotions (docs/guide/14 §14.2).

Each Reddit comment was rated by 3–5 raters for 27 emotions plus neutral (the per-rater `raw` release). Targets are
rater shares, never a majority vote:
- one Noul per emotion: the share of raters who marked it;
- one Choice over the paper's sentiment groups (positive, negative, ambiguous, neutral): each rater's vote is split
  equally across the groups of the emotions they marked.

Train rows ask the sentiment Choice, `neutral`, and a seeded uniform sample of `nouls` − 1 other emotions, so each
emotion is asked about at its natural rate. (General v1 asked about every marked emotion, up to half the sample: it
then saw emotions everywhere, and on daily_dialog called 1,282 of 7,740 turns `surprise`, gold 116.) Eval rows ask
all 28.

Splits by comment id: the `simplified` validation ids → gold_calib, test ids → gold_test (the held-out calibration
eval portion, configs/holdout.yaml); everything else (the simplified train ids and the comments the simplified release
dropped for lack of rater agreement) → train, with a seeded `val_share` of it → val.
"""
import hashlib
import pathlib
import random

import pandas as pd

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "goemotions"
EMOTIONS = {
    "admiration": "admiration: respect or warm approval of someone or something",
    "amusement": "amusement: finding something funny or entertaining",
    "anger": "anger: strong displeasure or hostility",
    "annoyance": "annoyance: mild anger or irritation",
    "approval": "approval: agreeing with or accepting something as good",
    "caring": "care: concern for someone's wellbeing or wish to help",
    "confusion": "confusion: not understanding something or being uncertain",
    "curiosity": "curiosity: interest in learning or knowing more",
    "desire": "desire: wanting something or wishing for it to happen",
    "disappointment": "disappointment: sadness that hopes or expectations were not met",
    "disapproval": "disapproval: an unfavourable opinion or not agreeing",
    "disgust": "disgust: revulsion or strong disapproval at something offensive",
    "embarrassment": "embarrassment: self-consciousness, shame or awkwardness",
    "excitement": "excitement: eager enthusiasm or high energy",
    "fear": "fear: being afraid, worried or anxious about a threat",
    "gratitude": "gratitude: thankfulness or appreciation",
    "grief": "grief: deep sorrow, especially over a loss or death",
    "joy": "joy: happiness, pleasure or delight",
    "love": "love: strong affection or fondness",
    "nervousness": "nervousness: apprehension or unease",
    "optimism": "optimism: hope or confidence about the future",
    "pride": "pride: satisfaction in an achievement of one's own or of someone close",
    "realization": "realisation: becoming aware of something",
    "relief": "relief: reassurance after anxiety or distress has passed",
    "remorse": "remorse: regret or guilt for something one did",
    "sadness": "sadness: unhappiness or sorrow",
    "surprise": "surprise: being astonished or caught off guard",
    "neutral": "no particular emotion: a neutral statement",
}
GROUPS = {
    "positive": ["admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love",
                 "optimism", "pride", "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief",
                 "nervousness", "remorse", "sadness"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}
GROUP_OF = {e: g for g, es in GROUPS.items() for e in es}
SENTIMENT = {"type": "choice", "instructions": "What overall emotional tone does the comment express?",
             "criteria": {
                 "positive": "Positive emotion, such as joy, gratitude, admiration, amusement, love or optimism",
                 "negative": "Negative emotion, such as anger, annoyance, sadness, disgust, fear or disappointment",
                 "ambiguous": "Emotion that is neither clearly positive nor negative, such as surprise, curiosity, "
                              "confusion or realisation",
                 "neutral": "No particular emotion: a neutral statement"}}


def noul(emotion: str) -> dict:
    return {"type": "noul", "instructions": f"The comment expresses {EMOTIONS[emotion]}"}


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def sentiment_shares(votes: pd.DataFrame) -> dict[str, float]:
    """Each rater's vote split equally over the groups of the emotions they marked; raters who marked none are
    left out (none if no rater marked anything)."""
    shares, n = dict.fromkeys(GROUPS, 0.0), 0
    for row in votes[list(EMOTIONS)].to_numpy():
        groups = {GROUP_OF[e] for e, v in zip(EMOTIONS, row) if v}
        for g in groups:
            shares[g] += 1 / len(groups)
        n += bool(groups)
    return {g: v / n for g, v in shares.items()} if n else {}


def convert(source_dir: pathlib.Path, nouls: int = 6, val_share: float = 0.05, seed: int = 0, tokenizer=None,
            max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    raw = pd.read_parquet(source_dir / "raw.parquet", columns=["id", "text", "rater_id", *EMOTIONS])
    eval_split = {i: "gold_calib" for i in pd.read_parquet(source_dir / "simplified_validation.parquet").id}
    eval_split |= {i: "gold_test" for i in pd.read_parquet(source_dir / "simplified_test.parquet").id}
    out = []
    for cid, votes in raw.groupby("id", sort=True):
        text = votes.text.iloc[0]
        if not text.strip():
            continue
        split = eval_split.get(cid) or ("val" if _u(f"val:{cid}") < val_share else "train")
        share = votes[list(EMOTIONS)].mean().to_dict()
        if split in ("gold_calib", "gold_test"):
            asked = list(EMOTIONS)
        else:
            rng = random.Random(f"{seed}:{cid}")
            asked = {"neutral", *rng.sample([e for e in EMOTIONS if e != "neutral"], nouls - 1)}
            asked = [e for e in EMOTIONS if e in asked]   # canonical order
        questions = {e: noul(e) for e in asked}
        targets = {e: QuestionTargets(human=HumanTarget(noul=float(share[e]), n_raters=len(votes))) for e in asked}
        if sent := sentiment_shares(votes):
            questions["sentiment"] = SENTIMENT
            targets["sentiment"] = QuestionTargets(human=HumanTarget(probabilities=sent, n_raters=len(votes)))
        state, truncated = truncate_state(text, tokenizer, max_state_tokens) if tokenizer else (text, False)
        out.append(Record(
            id=f"{SOURCE}-{cid}", lineage=Lineage(source_dataset=SOURCE, source_id=cid, family_id=f"{SOURCE}-base"),
            license="Apache-2.0", tier="R", modality=["text"], language="en", state=state, questions=questions,
            targets=targets,
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split=split))
    return out
