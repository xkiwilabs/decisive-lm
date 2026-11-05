"""scripts/split_half.py: votes rebuilt from human targets for Score levels and named Choice options."""
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location("split_half", pathlib.Path(__file__).parent.parent / "scripts/split_half.py")
split_half = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_half)


def target(probabilities, n):
    return SimpleNamespace(human=SimpleNamespace(n_raters=n, probabilities=probabilities))


def test_votes_index_score_levels_and_choice_options():
    score = SimpleNamespace(criteria=["low", "mid", "high"])
    assert sorted(split_half.votes(target({"0": 0.5, "2": 0.5}, 4), score)) == [0, 0, 2, 2]
    choice = SimpleNamespace(criteria={"cat": "a cat", "dog": "a dog"})
    assert sorted(split_half.votes(target({"dog": 0.75, "cat": 0.25}, 4), choice)) == [0, 1, 1, 1]


def test_identical_halves_agree_perfectly():
    items = [np.array([0] * 10), np.array([0] * 5 + [1] * 5), np.array([0] * 8 + [1] * 2)] * 5
    s = split_half.split_half(items, levels=2, n_splits=20)
    assert s["items"] == 15 and -1 <= s["split_half"] <= 1 and s["spearman_brown"] <= 1
