"""First use case: free-text "what is this person's intent?" responses coded by four humans.

Source: the authors' study data (not distributed). Read by explicit column whitelist
only; nothing else in the study's files, such as experiment metadata (including each
video's true intention), may enter this repo.

- state: the participant's transcription, exactly what the human coders saw
- one Choice question built from the codebook, whose category list (and order) it defines
- human target: the share of coders choosing each category (soft label)
- splits by participant, so no participant's responses span splits
"""
import pathlib
import random
import re

import pandas as pd

from decisivelm.data.schema import Lineage, Preprocess, QuestionTargets, HumanTarget, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "llm_response_coding"
RESPONSE_COLUMNS = ["SubjectID", "TrialNumber", "Transcription"]
HUMAN_COLUMNS = ["human001", "human002", "human003", "human004"]
LICENSE = "LicenseRef-pending"              # see configs/datasets.yaml
CODES_FILE = "full_coding_matrix.csv"      # the source project's normalised human labels (summary_outputs/)
# Cumulative split boundaries by participant. 60 participants -> 36 / 6 / 7 / 11.
# gold_calib : gold_test = 40 : 60, as for every human-labelled set.
SPLIT_BOUNDS = [("train", 0.60), ("val", 0.70), ("gold_calib", 0.82), ("gold_test", 1.0)]


def codebook_question(path: pathlib.Path) -> dict:
    """One Choice question from the codebook.

    Precedence rules sit in a category's bullet as an "Important Note", but they
    decide between several categories. A leaf sees only its own option (each option
    is scored independently, docs/guide/06 §6.8), so those rules move into the question
    instructions, which every leaf sees. The closing "### Instructions" section is
    about generating a one-word answer and is dropped.
    """
    text = pathlib.Path(path).read_text(encoding="utf-8")
    preamble, rest = text.split("### Category Definitions", 1)
    body = rest.split("### Instructions", 1)[0]
    parts = re.split(r"(?m)^\*\s+\*\*(.+?)\*\*\s*$", body)
    criteria, rules = {}, []
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        block = "\n".join(line.strip() for line in parts[i + 1].strip().splitlines())
        if "**Important Note**" in block:
            block, note = block.split("**Important Note**", 1)
            rules.append("Important Note" + note.rstrip())
        criteria[name] = block.strip()
    if len(criteria) < 2:
        raise ValueError(f"{path}: found categories {list(criteria)}; a codebook needs at least two")
    instructions = preamble.strip()
    if rules:
        instructions += "\n\nRules that apply across categories:\n" + "\n".join(rules)
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def split_subjects(subjects, seed: int = 0) -> dict:
    order = sorted(set(subjects))
    random.Random(seed).shuffle(order)
    n, out, start = len(order), {}, 0
    for name, bound in SPLIT_BOUNDS:
        end = round(bound * n)
        for s in order[start:end]:
            out[s] = name
        start = end
    return out


def convert(source_dir: pathlib.Path, seed: int = 0, codebook: pathlib.Path | None = None,
            tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    """Canonical rows. With a tokenizer, states over budget are truncated here (decisivelm.preprocess) and flagged."""
    source_dir = pathlib.Path(source_dir)
    responses = pd.read_csv(source_dir / "response_data.csv", usecols=RESPONSE_COLUMNS)
    codes = pd.read_csv(source_dir / CODES_FILE, usecols=["SubjectID", "TrialNumber"] + HUMAN_COLUMNS)
    question = codebook_question(codebook or source_dir / "coder_instructions_new.md")
    categories = list(question["criteria"])
    canon = {c.lower(): c for c in categories}
    split_of = split_subjects(responses.SubjectID, seed)

    df = responses.merge(codes, on=["SubjectID", "TrialNumber"], how="left", validate="one_to_one")
    rows = []
    for r in df.itertuples(index=False):
        labels = [getattr(r, h) for h in HUMAN_COLUMNS if isinstance(getattr(r, h), str)]
        unknown = [x for x in labels if x.strip().lower() not in canon]
        if unknown:
            raise ValueError(f"{r.SubjectID}-{r.TrialNumber}: unknown labels {unknown}")
        if not labels:
            continue
        counts = {c: 0 for c in categories}
        for x in labels:
            counts[canon[x.strip().lower()]] += 1
        sid, trial = int(r.SubjectID), int(r.TrialNumber)
        state, truncated = (truncate_state(str(r.Transcription), tokenizer, max_state_tokens) if tokenizer
                            else (str(r.Transcription), False))
        rows.append(Record(
            id=f"lrc-{sid}-{trial}",
            lineage=Lineage(source_dataset=SOURCE, source_id=f"{sid}-{trial}", family_id=f"lrc-subject-{sid}"),
            license=LICENSE,
            tier="X",
            modality=["text"], language="en",
            state=state,
            questions={"intent": question},
            targets={"intent": QuestionTargets(human=HumanTarget(
                probabilities={c: n / len(labels) for c, n in counts.items()}, n_raters=len(labels)))},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split=split_of[r.SubjectID],
        ))
    return rows
