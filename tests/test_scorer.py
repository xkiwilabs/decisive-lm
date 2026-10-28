"""Scorer input standardisation (docs/guide/06 §6.5): a per-dimension z-score fitted on training states, before the LayerNorm."""
import torch

from decisivelm.head import Scorer


def _states(n=64, h=32, seed=0):
    """Query states with one massive-activation dim, as on 26B A4B (results/invariance.md)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, h, generator=g) * torch.rand(h, generator=g) + torch.randn(h, generator=g)
    x[..., 0] *= 50
    return x


def test_standardize_z_scores_each_dimension_before_the_head():
    x = _states()
    scorer = Scorer(32, width=16).eval()
    scorer.standardize(x)
    flat = x.reshape(-1, 32)
    z = (x - flat.mean(0)) / flat.std(0)
    with torch.no_grad():
        torch.testing.assert_close(scorer(x), scorer.net(z).squeeze(-1))


def test_a_scorer_never_standardized_passes_states_through():
    x = _states()
    scorer = Scorer(32, width=16).eval()
    with torch.no_grad():
        torch.testing.assert_close(scorer(x), scorer.net(x).squeeze(-1))


def test_standardization_is_saved_with_the_head():
    x = _states()
    fitted = Scorer(32, width=16).eval()
    fitted.standardize(x)
    loaded = Scorer(32, width=16).eval()
    loaded.load_state_dict(fitted.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), fitted(x))


def test_bundles_saved_before_standardization_load_unchanged():
    x = _states()
    old = Scorer(32, width=16).eval()
    state = {k: v for k, v in old.state_dict().items() if k.startswith("net.")}   # a scorer.pt from before
    loaded = Scorer(32, width=16).eval()
    loaded.standardize(_states(seed=1))
    loaded.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), old(x))
