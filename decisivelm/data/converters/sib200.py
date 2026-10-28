"""SIB-200: 1,004 Flores-200 news sentences, each labelled with one of 7 topics, the same sentences in 205 languages.

A development check of calibration on text that is not English (docs/RELEASE.md, change log 2026-10-24): it showed
General v3c underconfident outside English, so no non-English scale is fitted. Share-alike (S tier), so no row is
ever trained on for the general adapter (configs/holdout.yaml) and every row is gold.

- state: the sentence in one language; the question is in English, as in the suite's MASSIVE task
- one Choice over the 7 topics, each described so that it stands on its own
- target: the published topic (one label per sentence, the same in every language; n_raters left unset)
- languages: all but the frozen suite's 10 held-out MASSIVE languages, so the suite still measures zero-shot language
  transfer; English is kept as the reference
- splits by sentence (index_id), so a sentence is in gold_calib (40%) or gold_test (60%) in every language
"""
import csv
import hashlib
import pathlib

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "sib200"
TOPIC = {"type": "choice",
         "instructions": "What is the main topic of this sentence?",
         "criteria": {"science/technology": "Science, research or technology",
                      "travel": "Travel, tourism or visiting places",
                      "politics": "Politics, government or public affairs",
                      "sports": "Sports, athletes or competitions",
                      "health": "Health, medicine or disease",
                      "entertainment": "Entertainment, the arts, media or celebrities",
                      "geography": "Geography, places, nature or the physical world"}}
# Flores-200 codes of the frozen suite's held-out MASSIVE languages (configs/holdout.yaml): never used here.
SUITE_LANGUAGES = {"pol_Latn", "urd_Arab", "fin_Latn", "kat_Geor", "amh_Ethi", "swh_Latn", "tam_Taml", "kor_Hang",
                   "vie_Latn", "tgl_Latn"}
CALIB_SHARE = 0.4
FILES = ("train.tsv", "dev.tsv", "test.tsv")


def _split(index_id: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:split:{index_id}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < CALIB_SHARE else "gold_test"


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    out = []
    for lang_dir in sorted(p for p in (pathlib.Path(source_dir) / "data").iterdir() if p.is_dir()):
        lang = lang_dir.name
        if lang in SUITE_LANGUAGES:
            continue
        for name in FILES:
            with open(lang_dir / name, encoding="utf-8", newline="") as f:
                for d in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if d["category"] not in TOPIC["criteria"]:
                        raise ValueError(f"{lang}/{name}: unknown topic {d['category']!r}")
                    state, truncated = d["text"], False
                    if tokenizer is not None:
                        state, truncated = truncate_state(state, tokenizer, max_state_tokens)
                    sid = f"{lang}-{d['index_id']}"
                    out.append(Record(
                        id=f"{SOURCE}-{sid}",
                        lineage=Lineage(source_dataset=SOURCE, source_id=sid, family_id=f"{SOURCE}-{d['index_id']}"),
                        license="CC-BY-SA-4.0", tier="S", modality=["text"], language=lang, state=state,
                        questions={"topic": TOPIC},
                        targets={"topic": QuestionTargets(human=HumanTarget(
                            probabilities={k: float(k == d["category"]) for k in TOPIC["criteria"]}))},
                        preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens,
                                              truncated=truncated),
                        split=_split(d["index_id"])))
    return out
