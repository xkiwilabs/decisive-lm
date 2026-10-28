"""Stanford Politeness Corpus, Stack Exchange part (Danescu-Niculescu-Mizil et al. 2013; ConvoKit release): 6,603
requests from Stack Exchange, each rated by 5 crowd workers on a 1–25 slider from very impolite to very polite.

Human-rated training data for a two-sided Score whose neutral middle is a common answer (docs/RELEASE.md, change log
2026-10-25): the teacher seldom picks such a middle, so synthetic rows cannot teach it. The Wikipedia part is left out
because the toxicity development check is Wikipedia talk comments too.

- state: the request text
- one Score over 5 levels, from very impolite to very polite; each rating is binned symmetrically about the slider's
  middle: 1–6, 7–12, 13 (neither), 14–19, 20–25
- target: the share of the 5 raters at each level (n_raters)
- splits by text (a repeated request keeps one split): 15% gold_test, 10% gold_calib, 5% val, the rest train
- tier S: the requests are Stack Exchange posts (CC BY-SA 3.0)
- the per-rater ratings ship as a pickle of numpy integers; it is read with an unpickler that admits nothing else
"""
import collections
import hashlib
import json
import pathlib
import pickle
import zipfile

import numpy as np

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "politeness"
CORPUS = "stack-exchange-politeness-corpus"
POLITENESS = {"type": "score",
              "instructions": "How polite is this request?",
              "criteria": ["Very impolite", "Somewhat impolite", "Neither polite nor impolite", "Somewhat polite",
                           "Very polite"]}
EDGES = (6, 12, 13, 19)   # upper bound of each level but the last
SPLITS = (("gold_test", 0.15), ("gold_calib", 0.25), ("val", 0.30))


def _int_scalar(dtype, data):
    if dtype.kind not in "iu":   # an object dtype would hand back whatever the pickle built
        raise pickle.UnpicklingError(f"refused a numpy {dtype} scalar")
    from numpy._core.multiarray import scalar
    return scalar(dtype, data)


class _Ratings(pickle.Unpickler):
    """Loading a downloaded pickle can run code, so only the two globals the ratings need are admitted, and the
    scalars must be integers."""
    def find_class(self, module, name):
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) in (("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")):
            return _int_scalar
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def level(rating: int) -> int:
    return sum(rating > e for e in EDGES)


def _split(key: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:split:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return next((s for s, top in SPLITS if u < top), "train")


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    with zipfile.ZipFile(pathlib.Path(source_dir) / f"{CORPUS}.zip") as z:
        utts = [json.loads(line) for line in z.read(f"{CORPUS}/utterances.jsonl").decode("utf-8").splitlines()]
        with z.open(f"{CORPUS}/Annotations-bin.p") as f:
            ratings = _Ratings(f, encoding="latin1").load()
    if len(ratings) != len(utts):
        raise ValueError(f"{len(ratings)} rating sets for {len(utts)} requests")
    out = []
    for u, r in zip(utts, ratings):
        levels = [level(int(v)) for v in r.values()]
        counts = collections.Counter(levels)
        text = u["text"].strip()
        key = hashlib.sha256(text.encode()).hexdigest()[:12]
        state, truncated = truncate_state(text, tokenizer, max_state_tokens) if tokenizer else (text, False)
        out.append(Record(
            id=f"{SOURCE}-{u['id']}",
            lineage=Lineage(source_dataset=SOURCE, source_id=str(u["id"]), family_id=f"{SOURCE}-{key}"),
            license="CC-BY-SA-3.0", tier="S", modality=["text"], language="en", state=state,
            questions={"politeness": POLITENESS},
            targets={"politeness": QuestionTargets(human=HumanTarget(
                probabilities={str(i): counts[i] / len(levels) for i in range(5)}, n_raters=len(levels)))},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split=_split(key)))
    return out
