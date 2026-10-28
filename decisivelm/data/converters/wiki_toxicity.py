"""Wikipedia Talk Labels, Toxicity (Wulczyn et al. 2017): English Wikipedia talk-page comments, each rated by about 10
crowd workers from very toxic (-2) to very healthy (2).

A development check of review readiness on a social-judgement Score rated by many people (docs/RELEASE.md, change log
2026-10-25): General v3d's confidence followed rater agreement on MultiNLI's Score but not on the suite's survey
Scores, so MultiNLI does not stand in for that kind of task. Never trained on for the general adapter
(configs/holdout.yaml), so every row is gold.

- state: the comment, with the release's NEWLINE_TOKEN / TAB_TOKEN put back and its backticks back to double quotes
- one Score over the 5 levels, ordered as the data codes them (very toxic first)
- target: the share of the comment's raters at each level (n_raters)
- rows: comments of the release's own test split with ≥ 3 raters, a sample of N_ROWS by a hash of the revision id;
  60% gold_test, 40% gold_calib
"""
import collections
import csv
import hashlib
import pathlib

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "wiki_toxicity"
TOXICITY = {"type": "score",
            "instructions": "How toxic or healthy is this comment as a contribution to a discussion?",
            "criteria": ["Very toxic: very hateful, aggressive or disrespectful, very likely to make people leave the "
                         "discussion",
                         "Toxic: rude, disrespectful or unreasonable, somewhat likely to make people leave the discussion",
                         "Neither toxic nor healthy",
                         "Healthy: reasonable, civil or polite, somewhat likely to make people want to continue the "
                         "discussion",
                         "Very healthy: very polite, thoughtful or helpful, very likely to make people want to continue "
                         "the discussion"]}
N_ROWS, CALIB_SHARE, MIN_RATERS = 5000, 0.4, 3


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def _text(comment: str) -> str:
    return comment.replace("NEWLINE_TOKEN", "\n").replace("TAB_TOKEN", "\t").replace("`", '"').strip()


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    scores = collections.defaultdict(list)
    with open(source_dir / "toxicity_annotations.tsv", encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            scores[str(int(float(d["rev_id"])))].append(int(float(d["toxicity_score"])))
    comments = {}
    with open(source_dir / "toxicity_annotated_comments.tsv", encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rid = str(int(float(d["rev_id"])))
            if d["split"] == "test" and len(scores[rid]) >= MIN_RATERS and _text(d["comment"]):
                comments[rid] = d["comment"]
    out = []
    for rid in sorted(comments, key=lambda r: _u("sample:" + r))[:N_ROWS]:
        counts = collections.Counter(scores[rid])
        n = len(scores[rid])
        text = _text(comments[rid])
        state, truncated = truncate_state(text, tokenizer, max_state_tokens) if tokenizer else (text, False)
        out.append(Record(
            id=f"{SOURCE}-{rid}",
            lineage=Lineage(source_dataset=SOURCE, source_id=rid, family_id=f"{SOURCE}-{rid}"),
            license="CC0-1.0", tier="R", modality=["text"], language="en", state=state,
            questions={"toxicity": TOXICITY},
            targets={"toxicity": QuestionTargets(human=HumanTarget(
                probabilities={str(i): counts[i - 2] / n for i in range(5)}, n_raters=n))},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split="gold_calib" if _u("split:" + rid) < CALIB_SHARE else "gold_test"))
    return out
