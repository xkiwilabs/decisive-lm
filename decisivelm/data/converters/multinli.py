"""Multi-rater reasoning data: MultiNLI 1.0 without the fiction genre (docs/guide/14 §14.4). Held out of the general mix: its
gold_calib fits the general head's new-task calibration scales (docs/RELEASE.md; configs/holdout.yaml).

- state: JSON {"premise", "hypothesis"}
- one Choice: entailment, neutral or contradiction, worded after the writers' instructions ("definitely correct",
  "might be correct", "definitely incorrect"). Gold rows also ask the same judgement as a Noul ("the hypothesis is
  definitely true": the entailment share) and as a 3-level Score (shows it false / neither / shows it true), so the
  new-task calibration scale can be fitted per question type from the same annotations
- targets: the share of annotators on each label. Dev pairs have 5 (the writer plus 4 validators); train pairs only
  the writer's label (n_raters left unset, as for HelpSteer2's single ratings). Pairs without a 3-of-5 majority
  (gold label "-") are kept: they are the disagreement the review-readiness criteria are about.
- fiction is dropped (CC BY-SA 3.0 and other works mixed with the OANC texts), so every row is R tier.
- splits by premise text, since several hypotheses share one (and 98 dev-matched premises recur in train under
  other promptIDs): dev matched → gold_calib / gold_test (40 / 60); dev mismatched → train / val (10%), its 5-rater
  shares being the point of this seed; train → train / val (10%). Train and mismatched pairs whose premise is in dev
  matched are dropped. ChaosNLI's MultiNLI items come from dev matched, so none of them is ever trained on.
- ids: pairIDs repeat when a writer gave one premise two hypotheses with the same label, so a repeat gets a suffix.
"""
import collections
import hashlib
import json
import pathlib
import zipfile

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "multinli"
LABELS = ("entailment", "neutral", "contradiction")
RELATION = {"type": "choice",
            "instructions": "Given that the premise is true, what does it say about the hypothesis?",
            "criteria": {"entailment": "The hypothesis is definitely true given the premise",
                         "neutral": "The hypothesis might be true or might not: the premise does not settle it",
                         "contradiction": "The hypothesis is definitely false given the premise"}}
FOLLOWS = {"type": "noul", "instructions": "Given that the premise is true, the hypothesis is definitely true as well"}
SUPPORT = {"type": "score", "instructions": "Given that the premise is true, how strongly does it support the hypothesis?",
           "criteria": ["It shows the hypothesis is false", "It neither shows the hypothesis true nor false",
                        "It shows the hypothesis is true"]}
FILES = {"train": "multinli_1.0_train.jsonl", "dev_matched": "multinli_1.0_dev_matched.jsonl",
         "dev_mismatched": "multinli_1.0_dev_mismatched.jsonl"}
VAL_SHARE, CALIB_SHARE = 0.1, 0.4
DROPPED_GENRES = {"fiction"}


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def _premise_key(premise: str) -> str:
    return hashlib.sha256(" ".join(premise.split()).encode()).hexdigest()[:16]


def _split(premise_key: str, source_split: str) -> str:
    u = _u(f"split:{premise_key}")
    if source_split == "dev_matched":
        return "gold_calib" if u < CALIB_SHARE else "gold_test"
    return "val" if u < VAL_SHARE else "train"


def _target(labels: list[str]) -> HumanTarget:
    labels = [x for x in labels if x in LABELS]
    return HumanTarget(probabilities={k: labels.count(k) / len(labels) for k in LABELS},
                       n_raters=len(labels) if len(labels) > 1 else None)


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS,
            source_splits=tuple(FILES)) -> list[Record]:
    out, seen = [], collections.Counter()
    with zipfile.ZipFile(pathlib.Path(source_dir) / "multinli_1.0.zip") as z:
        def pairs(source_split):
            for line in z.open(f"multinli_1.0/{FILES[source_split]}"):
                d = json.loads(line)
                if d["genre"] not in DROPPED_GENRES and any(x in LABELS for x in d["annotator_labels"]):
                    yield d
        gold = {_premise_key(d["sentence1"]) for d in pairs("dev_matched")}
        for source_split in source_splits:
            for d in pairs(source_split):
                key = _premise_key(d["sentence1"])
                if source_split != "dev_matched" and key in gold:
                    continue
                pair_id = f"{source_split}-{d['pairID']}"
                seen[pair_id] += 1
                if seen[pair_id] > 1:
                    pair_id += f"-{seen[pair_id]}"
                split = _split(key, source_split)
                t = _target(d["annotator_labels"])
                questions, targets = {"relation": RELATION}, {"relation": QuestionTargets(human=t)}
                if split in ("gold_calib", "gold_test"):
                    p = t.probabilities
                    questions |= {"follows": FOLLOWS, "support": SUPPORT}
                    targets["follows"] = QuestionTargets(human=HumanTarget(noul=p["entailment"], n_raters=t.n_raters))
                    targets["support"] = QuestionTargets(human=HumanTarget(
                        probabilities={"0": p["contradiction"], "1": p["neutral"], "2": p["entailment"]},
                        n_raters=t.n_raters))
                state, truncated = {"premise": d["sentence1"], "hypothesis": d["sentence2"]}, False
                if tokenizer is not None:
                    state["premise"], truncated = truncate_state(d["sentence1"], tokenizer, max_state_tokens)
                out.append(Record(
                    id=f"{SOURCE}-{pair_id}",
                    lineage=Lineage(source_dataset=SOURCE, source_id=pair_id, family_id=f"{SOURCE}-premise-{key}"),
                    license="LicenseRef-OANC", tier="R", modality=["text"], language="en", state=state,
                    questions=questions, targets=targets,
                    preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens,
                                          truncated=truncated),
                    split=split))
    return out
