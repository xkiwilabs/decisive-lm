"""Metrics used by every report (docs/guide/11)."""
import math

import numpy as np
import pytest
import torch

from decisivelm.eval import (agreement_profile, auroc, brier, ece, entropy_confidence, fit_temperature, kappa,
                             no_majority, pair_agreement, review_curve, share_in_review, soft_nll, spearman,
                             within_category_spearman)


def test_kappa_known_value():
    assert kappa(["x", "x", "y", "y"], ["x", "y", "y", "y"]) == pytest.approx(0.5)


def test_kappa_skips_missing_labels():
    assert kappa(["x", "y", None, "x"], ["x", "y", "x", np.nan]) == pytest.approx(1.0)


def test_ece_of_an_overconfident_model():
    conf = np.full(10, 0.9)
    correct = np.array([1, 0] * 5, dtype=bool)
    assert ece(conf, correct) == pytest.approx(0.4)


def test_ece_of_a_calibrated_model_is_zero():
    conf = np.array([0.5, 0.5, 1.0, 1.0])
    assert ece(conf, np.array([1, 0, 1, 1], dtype=bool)) == pytest.approx(0.0)


def test_soft_nll_is_one_for_uniform_predictions():
    logits = torch.zeros(4, 8)
    target = torch.softmax(torch.randn(4, 8), -1)
    assert soft_nll(logits, target) == pytest.approx(1.0)


def test_fit_temperature_recovers_a_sharpening_factor():
    torch.manual_seed(0)
    true_logits = torch.randn(2000, 5)
    target = torch.distributions.Categorical(logits=true_logits).sample()
    onehot = torch.nn.functional.one_hot(target, 5).float()
    assert fit_temperature(true_logits * 2.0, onehot) == pytest.approx(2.0, rel=0.1)


def test_brier_of_a_perfect_prediction_is_zero():
    p = np.eye(3)
    assert brier(p, np.array([0, 1, 2])) == pytest.approx(0.0)
    assert brier(np.full((1, 2), 0.5), np.array([0])) == pytest.approx(0.5)


# Knowing which responses to check (docs/guide/11 §11.3): does confidence fall where the humans disagree,
# spread widely enough to threshold, and recover accuracy when the least-confident items go to a human?

def test_pair_agreement_is_the_share_of_agreeing_coder_pairs():
    labels = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 1, 2, 3],
                       [0, 0, -1, 1], [0, -1, -1, -1]])
    got = pair_agreement(labels)
    np.testing.assert_allclose(got[:6], [1, 0.5, 1 / 3, 1 / 6, 0, 1 / 3])
    assert np.isnan(got[6])


def test_no_majority_means_the_top_count_is_shared():
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 2], [0, 1, 2, 3], [0, 0, 0, 1], [0, 0, -1, 1]])
    assert no_majority(labels).tolist() == [True, False, True, False, False]


def test_spearman_uses_average_ranks():
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / math.sqrt(95))
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_auroc_counts_positive_over_negative_pairs():
    assert auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == pytest.approx(0.75)


def test_entropy_confidence_runs_from_uniform_to_one_hot():
    p = np.array([[0.25] * 4, [1.0, 0, 0, 0]])
    np.testing.assert_allclose(entropy_confidence(p), [0.0, 1.0], atol=1e-9)


def test_agreement_profile_gives_mean_confidence_per_level_and_its_range():
    conf = np.array([0.9, 0.95, 0.6, 0.5, 0.7])
    agreement = np.array([1.0, 1.0, 0.5, 0.0, 0.5])
    means, spread = agreement_profile(conf, agreement)
    assert means == pytest.approx({0.0: 0.5, 0.5: 0.65, 1.0: 0.925})
    assert spread == pytest.approx(0.425)


def test_within_category_spearman_skips_small_categories():
    rng = np.random.default_rng(0)
    agreement = rng.random(60)
    conf = agreement + 0.01 * rng.random(60)
    category = np.array([0] * 30 + [1] * 25 + [2] * 5)
    got = within_category_spearman(conf, agreement, category, min_n=25)
    assert set(got) == {0, 1} and all(v > 0.9 for v in got.values())


def test_review_curve_counts_errors_caught_and_remainder_accuracy():
    conf = np.linspace(0.1, 1.0, 10)
    correct = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1], bool)
    caught, remainder = review_curve(conf, correct, np.ones(10, bool), [0.0, 0.2])
    np.testing.assert_allclose(caught, [0, 2 / 3])
    np.testing.assert_allclose(remainder, [0.7, 7 / 8])


def test_review_curve_ranks_unscored_items_but_leaves_them_out_of_accuracy():
    conf = np.linspace(0.1, 1.0, 10)
    correct = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1], bool)
    scored = np.array([False] + [True] * 9)       # item 0 is a human tie: reviewed, never counted
    caught, remainder = review_curve(conf, correct, scored, [0.2])
    assert caught[0] == pytest.approx(1 / 2) and remainder[0] == pytest.approx(7 / 8)


def test_share_in_review_is_the_fraction_of_flagged_items_among_the_least_confident():
    conf = np.linspace(0.1, 1.0, 10)
    tied = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], bool)
    assert share_in_review(conf, tied, 0.3) == pytest.approx(2 / 3)
