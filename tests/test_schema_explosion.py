"""Deterministic schema explosion (docs/guide/08 §8.10): subsets, drop-gold + `other`, Noul splits.

Labels stay human: every derived row's target is computed from the parent's.
"""
import random

import pytest

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.synth.multipliers.schema_explosion import OTHER, drop_gold_add_other, noul_splits, subset

INTENTS = {f"intent_{i}": f"Intent number {i}" for i in range(20)}


def parent(gold="intent_3", split="train", probs=None):
    probs = probs or {c: float(c == gold) for c in INTENTS}
    return Record(id="p1", lineage=Lineage(source_dataset="toy", source_id="1", family_id="toy-base"),
                  license="CC-BY-4.0", tier="R", modality=["text"], language="en", state="state text",
                  questions={"intent": {"type": "choice", "instructions": "Which intent?", "criteria": INTENTS}},
                  targets={"intent": QuestionTargets(human=HumanTarget(probabilities=probs))},
                  preprocess=Preprocess(render_version=1, max_state_tokens=8192, truncated=False), split=split)


def test_subset_keeps_the_gold_option_and_k_distractors():
    child = subset(parent(), "intent", k=4, rng=random.Random(0))
    opts = child.questions["intent"].criteria
    assert "intent_3" in opts and len(opts) == 5
    assert child.targets["intent"].human.probabilities["intent_3"] == 1.0


def test_subset_keeps_every_option_with_target_mass_and_renormalises():
    soft = {c: 0.0 for c in INTENTS} | {"intent_1": 0.5, "intent_2": 0.25, "intent_9": 0.25}
    child = subset(parent(probs=soft), "intent", k=2, rng=random.Random(0))
    p = child.targets["intent"].human.probabilities
    assert {"intent_1", "intent_2", "intent_9"} <= set(p) and len(p) == 5
    assert sum(p.values()) == pytest.approx(1.0)


def test_derived_rows_inherit_split_family_tier_and_record_lineage():
    child = subset(parent(split="gold_test"), "intent", k=3, rng=random.Random(0))
    assert child.split == "gold_test" and child.lineage.family_id == "toy-base" and child.tier == "R"
    assert child.lineage.parent_id == "p1" and child.lineage.multiplier.startswith("schema_explosion:subset")
    assert child.id != "p1"


def test_drop_gold_moves_all_mass_to_a_self_contained_other():
    child = drop_gold_add_other(parent(), "intent", k=5, rng=random.Random(0))
    opts = child.questions["intent"].criteria
    assert "intent_3" not in opts and OTHER in opts and len(opts) == 6
    kept = [c for c in opts if c != OTHER]
    assert all(INTENTS[c] in opts[OTHER] for c in kept)          # names what the kept options cover
    assert child.targets["intent"].human.probabilities == {c: float(c == OTHER) for c in opts}


def test_drop_gold_also_drops_the_most_confusable_options():
    def similar(a, b):   # intent_4 is "confusable" with the gold intent_3
        return 1.0 if {a, b} == {"Intent number 3", "Intent number 4"} else 0.0
    for seed in range(10):
        child = drop_gold_add_other(parent(), "intent", k=8, rng=random.Random(seed), similarity=similar, n_confusable=1)
        assert "intent_4" not in child.questions["intent"].criteria


def test_soft_targets_are_not_dropped_into_other():
    soft = {c: 0.0 for c in INTENTS} | {"intent_1": 0.5, "intent_2": 0.5}
    assert drop_gold_add_other(parent(probs=soft), "intent", k=5, rng=random.Random(0)) is None


def test_noul_splits_carry_the_option_mass_and_control_the_positive_rate():
    rng = random.Random(0)
    children = [c for _ in range(200) for c in noul_splits(parent(), "intent", n=2, rng=rng)]
    assert all(c.questions["intent"].type == "noul" for c in children)
    positive = [c.targets["intent"].human.noul for c in children]
    assert set(positive) <= {0.0, 1.0}
    assert 0.30 <= sum(positive) / len(positive) <= 0.50
    pos = next(c for c in children if c.targets["intent"].human.noul == 1.0)
    assert "Intent number 3" in pos.questions["intent"].instructions
