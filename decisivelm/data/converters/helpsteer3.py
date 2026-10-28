"""Frozen-suite set: NVIDIA HelpSteer3, preference subset (Wang et al. 2025; docs/guide/14 §14.4; docs/RELEASE.md).

- state: JSON {"conversation", "response_1", "response_2"}; turns become "User:" / "Assistant:" lines. Each part is
  truncated on its own (head and tail, decisivelm.preprocess) to a share of the state budget.
- one Score question, which response is better and by how much, on the annotators' six-level scale: the individual
  ratings run −3…−1 and 1…3 with no "about the same" level (the released overall score has a 0; it is not used)
- targets: the shares of the item's individual ratings (n_raters). The card says up to five annotators rated each item
  and only the three that agree most were released, so disagreement here is, by construction, less than it was.
- `language`: the conversation's language for the multilingual domain, else "en" (the code domain's `language` field
  names the programming language)
- the whole dataset is held out (configs/holdout.yaml); the suite is a fixed random SUITE_ITEMS of its 40,476 items,
  all gold_test, so baselines and candidates are scored on the same items at a manageable cost
"""
import gzip
import hashlib
import json
import pathlib

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "helpsteer3"
SUITE_ITEMS = 6000
SCALE = [-3, -2, -1, 1, 2, 3]
LEVELS = ["Response 1 is much better than response 2", "Response 1 is better than response 2",
          "Response 1 is slightly better than response 2", "Response 2 is slightly better than response 1",
          "Response 2 is better than response 1", "Response 2 is much better than response 1"]
QUESTION = {"type": "score", "criteria": LEVELS, "instructions": (
    "Which of the two responses is more helpful as the next assistant turn of the conversation, and by how much?")}
LANGUAGES = {"chinese": "zh", "korean": "ko", "french": "fr", "spanish": "es", "japanese": "ja", "german": "de",
             "russian": "ru", "portuguese": "pt", "italian": "it", "vietnamese": "vi", "dutch": "nl", "polish": "pl",
             "indonesian": "id"}
SHARES = {"conversation": 0.4, "response_1": 0.3, "response_2": 0.3}


def conversation(turns: list[dict]) -> str:
    return "\n".join(f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}" for t in turns)


def _rank(source_id: str) -> str:
    return hashlib.sha256(f"{SOURCE}:suite:{source_id}".encode()).hexdigest()


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS,
            n_items: int = SUITE_ITEMS) -> list[Record]:
    items = []
    for source_split in ("train", "validation"):
        with gzip.open(pathlib.Path(source_dir) / f"preference_{source_split}.jsonl.gz", "rt", encoding="utf-8") as f:
            items += [(f"{source_split}-{i:05d}", json.loads(line)) for i, line in enumerate(f)]
    items = sorted(items, key=lambda x: _rank(x[0]))[:n_items]
    out = []
    for source_id, d in sorted(items):
        state = {"conversation": conversation(d["context"]), "response_1": d["response1"], "response_2": d["response2"]}
        truncated = False
        if tokenizer is not None:
            for part, share in SHARES.items():
                state[part], cut = truncate_state(state[part], tokenizer, int(max_state_tokens * share))
                truncated |= cut
        ratings = [p["score"] for p in d["individual_preference"]]
        assert set(ratings) <= set(SCALE), (source_id, ratings)
        target = HumanTarget(probabilities={str(i): ratings.count(s) / len(ratings) for i, s in enumerate(SCALE)},
                             n_raters=len(ratings))
        out.append(Record(
            id=f"{SOURCE}-{source_id}",
            lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=f"{SOURCE}-{d['domain']}"),
            license="CC-BY-4.0", tier="R", modality=["text"],
            language=LANGUAGES[d["language"]] if d["domain"] == "multilingual" else "en",
            state=state, questions={"preference": QUESTION}, targets={"preference": QuestionTargets(human=target)},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens, truncated=truncated),
            split="gold_test"))
    return out
