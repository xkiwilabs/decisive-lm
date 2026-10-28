"""Conversational Sarcasm Corpus (Jang & Frassinelli 2024): people's replies to short everyday scenarios about a friend
or colleague called Steve, each rated by 4 or 6 other people for how sarcastic it is, from 1 (not at all) to 6
(completely).

Human ratings with several raters per item from a kind of judgement unrelated to the suite's tasks (docs/RELEASE.md,
change log 2026-10-26).

- state: the scenario, Steve's relationship to the speaker, and the reply
- one Score over the 6 levels, in the corpus's own words
- target: the share of the reply's evaluators at each level (n_raters); the speaker's rating of their own reply is
  not used
- splits by scenario (about 68 replies share each of the 104): 15% gold_test, 10% gold_calib, 5% val, the rest train
"""
import collections
import csv
import hashlib
import pathlib

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "csc_sarcasm"
SARCASTIC = {"type": "score",
             "instructions": "How sarcastic is the reply?",
             "criteria": ["Not at all", "Mostly not", "Not so much", "Somewhat", "Mostly", "Completely"]}
RELATIONSHIP = {"best_friend": "your best friend", "colleague": "your colleague"}
SPLITS = (("gold_test", 0.15), ("gold_calib", 0.25), ("val", 0.30))


def _split(key: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:split:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return next((s for s, top in SPLITS if u < top), "train")


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    replies = collections.defaultdict(list)
    with open(pathlib.Path(source_dir) / "data_full.csv", encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f):
            replies[(d["global_context_name"], d["global_speaker"])].append(d)
    out = []
    for (context, speaker), ds in sorted(replies.items()):
        d = ds[0]
        if any((x["context_text"], x["response_text"]) != (d["context_text"], d["response_text"]) for x in ds):
            raise ValueError(f"reply {context}/{speaker} has more than one text")
        levels = [int(float(x["sarcasm_score_by_evaluator"])) - 1 for x in ds]
        counts = collections.Counter(levels)
        text = (f"Scenario: {d['context_text'].strip()}\n"
                f"Steve is {RELATIONSHIP[d['interlocutor_relationship']]}.\n"
                f"Reply to Steve: {d['response_text'].strip()}")
        state, truncated = truncate_state(text, tokenizer, max_state_tokens) if tokenizer else (text, False)
        out.append(Record(
            id=f"{SOURCE}-{context}-{speaker}",
            lineage=Lineage(source_dataset=SOURCE, source_id=f"{context}-{speaker}", family_id=f"{SOURCE}-{context}"),
            license="MIT", tier="R", modality=["text"], language="en", state=state,
            questions={"sarcastic": SARCASTIC},
            targets={"sarcastic": QuestionTargets(human=HumanTarget(
                probabilities={str(i): counts[i] / len(levels) for i in range(6)}, n_raters=len(levels)))},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split=_split(context)))
    return out
