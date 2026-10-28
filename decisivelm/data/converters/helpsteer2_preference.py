"""Score source: NVIDIA HelpSteer2-Preference (Wang et al. 2025; docs/guide/14 §14.2), from 0.2 on.

- state: JSON {"conversation", "response_1", "response_2"}; NeMo turn markers become "User:" / "Assistant:" lines (as
  HelpSteer2). Each part is truncated on its own to a share of the state budget.
- one Score question, which response is better and by how much, on the annotators' six-level scale: the individual
  ratings run −3…−1 and 1…3 with no "about the same" level (the released overall strength has a 0; it is not used).
  The wording is ours, not the suite's HelpSteer3 wording.
- targets: the shares of all of the pair's individual ratings (`all_preferences_unprocessed`, 2–6 raters), not the
  three most similar the authors kept. Pairs with one rating are left out.
- each pair also appears with the responses swapped and the levels reversed (multiplier `swap`), so the answer cannot
  lean on which response comes first
- splits and families are HelpSteer2's, by prompt: the pairs are HelpSteer2's responses, and HelpSteer2's validation
  prompts (gold_calib, gold_test) must stay unseen wherever they are used
- the same task as the suite's HelpSteer3 (docs/RELEASE.md, suite-task rule): in a mix from 0.2, HelpSteer3 is
  reported as a seen task
"""
import hashlib
import pathlib

import pandas as pd

from decisivelm.data.converters.helpsteer2 import _split, conversation
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.render import RENDER_VERSION

SOURCE = "helpsteer2_preference"
SCALE = [-3, -2, -1, 1, 2, 3]
LEVELS = ["Response 1 is much better", "Response 1 is better", "Response 1 is slightly better",
          "Response 2 is slightly better", "Response 2 is better", "Response 2 is much better"]
QUESTION = {"type": "score", "criteria": LEVELS,
            "instructions": "Which response is the better reply to the conversation's last user turn, and by how much?"}
SHARES = {"conversation": 0.4, "response_1": 0.3, "response_2": 0.3}


def _target(ratings: list[int]) -> HumanTarget:
    assert set(ratings) <= set(SCALE), ratings
    return HumanTarget(probabilities={str(i): ratings.count(s) / len(ratings) for i, s in enumerate(SCALE)},
                       n_raters=len(ratings))


def convert(source_dir: pathlib.Path, tokenizer=None, max_state_tokens: int = MAX_STATE_TOKENS) -> list[Record]:
    df = pd.read_json(pathlib.Path(source_dir) / "preference.jsonl.gz", lines=True, dtype=False)
    out = []
    for i, d in enumerate(df.to_dict("records")):
        ratings = [int(p["strength"]) for p in d["all_preferences_unprocessed"]]
        if len(ratings) < 2:
            continue
        convo, truncated = conversation(d["prompt"]), False
        parts = {"conversation": convo, "response_1": d["response_1"], "response_2": d["response_2"]}
        if tokenizer is not None:
            for part, share in SHARES.items():
                parts[part], cut = truncate_state(parts[part], tokenizer, int(max_state_tokens * share))
                truncated |= cut
        source_id = f"{d['split']}-{i:05d}"
        family = f"helpsteer2-prompt-{hashlib.sha256(d['prompt'].encode()).hexdigest()[:12]}"
        split = _split(d["prompt"], "train" if d["split"] == "train" else "validation")
        for swap in (False, True):
            state = ({"conversation": parts["conversation"], "response_1": parts["response_2"],
                      "response_2": parts["response_1"]} if swap else parts)
            out.append(Record(
                id=f"{SOURCE}-{source_id}" + ("-swap" if swap else ""),
                lineage=Lineage(source_dataset=SOURCE, source_id=source_id, family_id=family,
                                parent_id=f"{SOURCE}-{source_id}" if swap else None,
                                multiplier="swap" if swap else None),
                license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=state,
                questions={"preference": QUESTION},
                targets={"preference": QuestionTargets(human=_target([-r for r in ratings] if swap else ratings))},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=max_state_tokens,
                                      truncated=truncated),
                split=split))
    return out
