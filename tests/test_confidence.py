"""Confidence statistics (docs/guide/04 §4.8) and its worked examples."""
import math

import pytest

from decisivelm.confidence import METHODS, entropy, margin, ordinal, peak


def test_peak_matches_spec_choice_example():
    assert peak([0.01, 0.96, 0.03]) == pytest.approx(0.94)


def test_peak_matches_spec_score_example():
    assert peak([0.05, 0.78, 0.17]) == pytest.approx(0.67)


@pytest.mark.parametrize("fn", [peak, entropy, margin, ordinal])
def test_one_hot_is_fully_confident(fn):
    assert fn([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [peak, entropy, margin])
def test_uniform_has_zero_confidence(fn):
    assert fn([0.25] * 4) == pytest.approx(0.0)


def test_entropy_two_way():
    p = [0.9, 0.1]
    h = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert entropy(p) == pytest.approx(1 - h / math.log(2))


def test_ordinal_penalises_mass_split_across_the_scale_more_than_neighbours():
    neighbours = ordinal([0.0, 0.5, 0.5, 0.0, 0.0])
    extremes = ordinal([0.5, 0.0, 0.0, 0.0, 0.5])
    assert extremes == pytest.approx(0.0)
    assert neighbours > extremes


def test_methods_registry_has_spec_names():
    assert set(METHODS) == {"peak", "entropy", "margin", "ordinal"}
