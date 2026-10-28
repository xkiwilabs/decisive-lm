"""Score seed: NVIDIA HelpSteer2 (docs/guide/14 §14.2).

- state: JSON {"conversation", "response"}; NeMo turn markers become "User:" / "Assistant:" lines
- five Score questions (0–4). Instructions follow the dataset card's attribute definitions; the
  level descriptions are ours (the card defines the attributes, not each level)
- targets: the distribution of the individual annotators' ratings (disagreements file) where
  the item is there, else the released score (the mean of the closest three, rounded)
- splits by prompt, since several responses share one: source train → train / val (10%),
  source validation → gold_calib / gold_test (40 / 60)
"""
import ast
import hashlib
import pathlib

import pandas as pd

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "helpsteer2"
ATTRIBUTES = {
    "helpfulness": ("How helpful is the response to the conversation's last user turn overall?",
                    ["Not helpful at all", "Slightly helpful: misses most of what was asked",
                     "Partially helpful: addresses the request with notable gaps or problems",
                     "Mostly helpful: addresses the request well with minor gaps",
                     "Fully helpful: completely addresses the request"]),
    "correctness": ("How correct is the response: does it include all pertinent facts without errors?",
                    ["Completely incorrect, or misses the pertinent facts", "Mostly incorrect",
                     "Partly correct: some errors or missing facts", "Mostly correct: minor errors or omissions",
                     "Fully correct: all pertinent facts and no errors"]),
    "coherence": ("How coherent is the response: is it consistent and clearly expressed?",
                  ["Incoherent", "Mostly incoherent or confusing", "Somewhat coherent: parts are unclear or inconsistent",
                   "Mostly coherent: minor lapses in clarity", "Fully coherent and clear"]),
    "complexity": ("How much intellectual depth did writing the response require?",
                   ["Basic: anyone with basic language skills could write it", "Simple: everyday knowledge",
                    "Intermediate: some education or familiarity with the topic",
                    "Advanced: university-level knowledge of the topic", "Expert: deep domain expertise"]),
    "verbosity": ("How much detail does the response give relative to what the conversation asks for?",
                  ["Very terse", "Succinct", "Moderate", "Detailed", "Very detailed and long"]),
}
VAL_SHARE, CALIB_SHARE = 0.10, 0.40


def _u(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16) / 16 ** 15


def conversation(prompt: str) -> str:
    text = "User: " + prompt
    return text.replace("\n<extra_id_1>Assistant\n", "\nAssistant: ").replace("\n<extra_id_1>User\n", "\nUser: ")


def _split(prompt: str, source_split: str) -> str:
    u = _u("helpsteer2-split:" + prompt)
    if source_split == "train":
        return "val" if u < VAL_SHARE else "train"
    return "gold_calib" if u < CALIB_SHARE else "gold_test"


def _target(ratings) -> HumanTarget:
    counts = [0] * 5
    for x in ratings:
        counts[int(x)] += 1
    return HumanTarget(probabilities={str(i): c / len(ratings) for i, c in enumerate(counts)},
                       n_raters=len(ratings) if len(ratings) > 1 else None)


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    raters = {}
    for d in pd.read_json(source_dir / "disagreements.jsonl.gz", lines=True, dtype=False).to_dict("records"):
        raters[(d["prompt"], d["response"])] = {a: ast.literal_eval(str(d[a])) for a in ATTRIBUTES}
    questions = {a: {"type": "score", "instructions": ins, "criteria": levels} for a, (ins, levels) in ATTRIBUTES.items()}
    out = []
    for source_split in ("train", "validation"):
        df = pd.read_json(source_dir / f"{source_split}.jsonl.gz", lines=True, dtype=False)
        for i, d in enumerate(df.to_dict("records")):
            state = {"conversation": conversation(d["prompt"]), "response": d["response"]}
            truncated = False
            if tokenizer is not None:
                text, truncated = truncate_state(state["response"], tokenizer, max_state_tokens)
                state["response"] = text
            per = raters.get((d["prompt"], d["response"]))
            targets = {a: QuestionTargets(human=_target(per[a] if per and per[a] else [d[a]])) for a in ATTRIBUTES}
            source_id = f"{source_split}-{i:05d}"
            out.append(Record(
                id=f"{SOURCE}-{source_id}",
                lineage=Lineage(source_dataset=SOURCE, source_id=source_id,
                                family_id=f"{SOURCE}-prompt-{hashlib.sha256(d['prompt'].encode()).hexdigest()[:12]}"),
                license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=state,
                questions=questions, targets=targets,
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
                split=_split(d["prompt"], source_split)))
    return out
