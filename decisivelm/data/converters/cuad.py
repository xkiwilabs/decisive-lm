"""Frozen-suite set: CUAD v1, the 10 held-out clause types (Hendrycks et al. 2021; docs/guide/14 §14.4; docs/RELEASE.md).

- state: one section of a contract, cut at line breaks into pieces of at most SECTION_CHARS characters (~2.5k
  tokens), since whole contracts (median ~8k tokens, longest ~80k) are far longer than anything the head trains on
- questions: one Noul per held-out clause type, "this excerpt contains a <type> clause", with CUAD's own definition
- targets: the expert labels as published (one annotation per contract, n_raters None): yes if a labelled span lies
  wholly inside the section, no if no span touches it. A span cut by the section boundary makes that clause type's
  answer unclear for the section, so the question is left out of that row.
- every row is gold_test in family `cuad-heldout-types` (held out in configs/holdout.yaml). The other 31 clause types
  and a contract-level split come with a training converter, if CUAD is ever trained on.
"""
import json
import pathlib

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "cuad"
HELD_OUT = ["Cap On Liability", "Audit Rights", "Change Of Control", "Non-Compete", "Uncapped Liability",
            "Covenant Not To Sue", "Liquidated Damages", "Non-Disparagement", "Most Favored Nation",
            "Source Code Escrow"]
SECTION_CHARS = 10_000


def key(clause: str) -> str:
    return clause.lower().replace(" ", "_").replace("-", "_")


def sections(text: str, limit: int = SECTION_CHARS) -> list[tuple[int, int]]:
    """[start, end) character ranges covering `text`, cut after a line break where one falls in the second half of
    the window, else at the limit."""
    out, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + limit // 2, end)
            end = cut + 1 if cut != -1 else end
        out.append((start, end))
        start = end
    return out


def convert(source_dir: pathlib.Path) -> list[Record]:
    data = json.load(open(pathlib.Path(source_dir) / "CUAD_v1.json", encoding="utf-8"))["data"]
    out = []
    for c, doc in enumerate(data):
        [para] = doc["paragraphs"]
        text, spans, questions = para["context"], {}, {}
        for qa in para["qas"]:
            clause = qa["id"].split("__")[-1]
            if clause in HELD_OUT:
                spans[clause] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
                questions[clause] = {"type": "noul", "instructions": (
                    f'This contract excerpt contains a "{clause}" clause, or part of one. CUAD defines the clause type '
                    f'by the question: {qa["question"].split("Details:")[-1].strip()}')}
        assert set(spans) == set(HELD_OUT), doc["title"]
        for s, (lo, hi) in enumerate(sections(text, SECTION_CHARS)):
            asked, targets = {}, {}
            for clause in HELD_OUT:
                inside = any(lo <= a and b <= hi for a, b in spans[clause])
                touches = any(a < hi and b > lo for a, b in spans[clause])
                if inside or not touches:
                    asked[key(clause)] = questions[clause]
                    targets[key(clause)] = QuestionTargets(human=HumanTarget(noul=float(inside)))
            if not asked:
                continue
            source_id = f"{c:03d}-{s:02d}"
            out.append(Record(
                id=f"{SOURCE}-{source_id}",
                lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=f"{SOURCE}-heldout-types"),
                license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=text[lo:hi], questions=asked,
                targets=targets,
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
                split="gold_test"))
    return out
