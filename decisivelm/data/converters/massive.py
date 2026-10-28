"""Frozen-suite set: MASSIVE 1.1, the 10 held-out languages (FitzGerald et al. 2022; docs/guide/14 §14.4; docs/RELEASE.md).

- state: the utterance, in its own language
- one Choice over MASSIVE's 60 intents; option names are the intent ids, descriptions the ids written out
  ("alarm_set" → "Alarm set"); the question is in English, so the test is answering an English schema about text in a
  language the head never trained on
- targets: the published intent (one annotation, n_raters None)
- only the test partition of the held-out locales (configs/holdout.yaml, the `massive` entry) is converted, all
  gold_test; `language` is the locale. MASSIVE is parallel: an utterance keeps one id and one partition in every
  locale, so the family is the id, and a later training converter must take train partitions only, in other locales.
"""
import json
import pathlib
import tarfile

from decisivelm.data.converters.intents import describe
from decisivelm.data.mix import load_holdouts
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "massive"
ARCHIVE = "amazon-massive-dataset-1.1.tar.gz"
INSTRUCTIONS = "Which of these intents does this request to a voice assistant express?"


def held_out_locales() -> list[str]:
    [entry] = [h for h in load_holdouts() if h["dataset"] == SOURCE]
    return entry["languages"]


def convert(source_dir: pathlib.Path, locales: list[str] | None = None) -> list[Record]:
    locales = locales or held_out_locales()
    with tarfile.open(pathlib.Path(source_dir) / ARCHIVE) as tar:
        rows = {loc: [json.loads(line) for line in tar.extractfile(f"1.1/data/{loc}.jsonl")] for loc in locales}
    intents = sorted({r["intent"] for rs in rows.values() for r in rs})
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {i: describe(i) for i in intents}}
    out = []
    for loc in locales:
        for r in rows[loc]:
            if r["partition"] != "test":
                continue
            source_id = f"{loc}-{r['id']}"
            out.append(Record(
                id=f"{SOURCE}-{source_id}",
                lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=f"{SOURCE}-utt-{r['id']}"),
                license="CC-BY-4.0", tier="R", modality=["text"], language=loc, state=r["utt"],
                questions={"intent": question},
                targets={"intent": QuestionTargets(human=HumanTarget(
                    probabilities={i: float(i == r["intent"]) for i in intents}))},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
                split="gold_test"))
    return out
