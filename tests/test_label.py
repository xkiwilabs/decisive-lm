"""Labeller method (a), direct label log-probs (docs/guide/12), which is also the P0 baseline (docs/guide/12 §12.8)."""
import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from decisivelm.api_types import SystemOneRequest
import math

from decisivelm.synth.label import (PRUNE, LocalLM, choice_labels, choice_orders, choice_prompt, digit_ids, digit_scorer,
                                    label_ids, label_probs, number_labels)
from tiny_gemma import tiny_text_config, tokenizer

DEPT = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payment issues", "technical": "Bugs", "sales": "Pricing"}}


def question(q=DEPT):
    return SystemOneRequest.model_validate({"state": "s", "questions": {"q": q}}).questions["q"]


def test_small_choice_uses_every_rotation():
    orders = choice_orders(3, item_id="x")
    assert sorted(map(tuple, orders)) == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


def test_large_choice_uses_eight_seeded_permutations():
    orders = choice_orders(12, item_id="x")
    assert len(orders) == 8 and all(sorted(o) == list(range(12)) for o in orders)
    assert orders == choice_orders(12, item_id="x") and orders != choice_orders(12, item_id="y")


def test_prompt_lists_options_in_the_given_order_and_ends_with_the_state():
    shared, item = choice_prompt(question(), "API is down", [2, 0, 1], tokenizer())
    assert "A. sales: Pricing" in shared and "B. billing: Payment issues" in shared and "C. technical: Bugs" in shared
    assert "Which team should handle this?" in shared and "API is down" not in shared
    assert item.startswith("API is down") and item.endswith("Answer:")


def test_letter_and_space_letter_are_merged():
    ids = label_ids(tokenizer(), 3)
    logp = torch.full((len(tokenizer()),), -1e9)
    logp[ids[0][0]], logp[ids[0][1]], logp[ids[1][1]] = -1.0, -1.0, -1.0   # "A", " A", " B"
    probs, allowed = label_probs(logp.log_softmax(-1), ids)
    assert probs.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert allowed == pytest.approx(1.0)


def test_rotations_cancel_position_bias():
    """A labeller that always answers "A" carries no information: averaged over rotations it is uniform."""
    always_a = lambda shared, items: [(torch.tensor([1.0, 0.0, 0.0]), 0.9)] * len(items)
    (p, info), = choice_labels(question(), ["s"], always_a, tokenizer(), item_ids=["x"])
    assert list(p.values()) == pytest.approx([1 / 3] * 3)
    assert info == {"method": "logprob", "n_orderings": 3, "n_kept": 3, "allowed_mass": pytest.approx(0.9)}


def test_off_label_orderings_are_left_out():
    """An ordering whose allowed mass is below .5 says nothing about the labels: it is dropped from the average."""
    calls = iter([(torch.tensor([1.0, 0.0, 0.0]), 0.02), (torch.tensor([0.0, 0.0, 1.0]), 0.9),
                  (torch.tensor([0.0, 1.0, 0.0]), 0.9)])
    scorer = lambda shared, items: [next(calls) for _ in items]
    (p, info), = choice_labels(question(), ["s"], scorer, tokenizer(), item_ids=["x"])
    assert info["n_kept"] == 2 and sum(p.values()) == pytest.approx(1.0)


def test_content_preference_survives_rotation():
    def likes_bugs(shared, items):
        letter = next(line[0] for line in shared.splitlines() if line[1:3] == ". " and "technical" in line)
        return [(torch.eye(3)["ABC".index(letter)], 1.0)] * len(items)
    (p, _), = choice_labels(question(), ["s"], likes_bugs, tokenizer(), item_ids=["x"])
    assert p["technical"] == pytest.approx(1.0)


def test_items_sharing_an_order_share_one_call():
    calls = []
    def count(shared, items):
        calls.append(len(items))
        return [(torch.tensor([1.0, 0.0, 0.0]), 1.0)] * len(items)
    choice_labels(question(), ["s1", "s2", "s3"], count, tokenizer(), item_ids=["a", "b", "c"])
    assert calls == [3, 3, 3]


def many(n):
    return question({"type": "choice", "instructions": "Which intent?",
                     "criteria": {f"intent_{k}": f"Intent number {k}" for k in range(n)}})


def test_more_than_twenty_options_are_numbered_to_one_width():
    assert number_labels(25)[:2] == ["01", "02"] and number_labels(25)[-1] == "25"
    assert number_labels(151)[0] == "001"
    shared, item = choice_prompt(many(25), "s", list(range(25)), tokenizer())
    assert "01. intent_0: Intent number 0" in shared and "25. intent_24" in shared and "A. intent_0" not in shared
    assert item.endswith("Answer: ")
    shared, _ = choice_prompt(many(20), "s", list(range(20)), tokenizer())
    assert "T. intent_19" in shared   # up to 20 options keep their letters


def label_model(target: dict[str, float], off_label: float = 0.0, dense: bool = False):
    """A fake next_token_logprobs that answers with labels drawn from `target` ({label: p}), one digit at a time;
    `off_label` of the first token's mass goes to a token that is not a digit."""
    ids, calls = digit_ids(tokenizer()), []
    digit = {t: str(d) for d, t in enumerate(ids)}
    def next_token_logprobs(shared, items, extra):
        calls.append(len(items))
        out = []
        for ex in extra:
            prefix = "".join(digit[t] for t in ex)
            live = {k: v for k, v in target.items() if k.startswith(prefix)}
            scale = (1 - off_label) if not prefix else 1.0
            lp = {ids[d]: math.log(scale * sum(v for k, v in live.items() if k[len(prefix)] == str(d)) /
                                   sum(live.values()) or 1e-300) for d in range(10)}
            if dense:
                t = torch.full((len(tokenizer()),), -math.inf)
                for k, v in lp.items():
                    t[k] = v
                lp = t
            out.append(lp)
        return out
    return next_token_logprobs, calls


@pytest.mark.parametrize("dense", [False, True])
def test_digit_scorer_recovers_the_label_distribution(dense):
    target = {"07": 0.6, "12": 0.3, "23": 0.1}
    model, calls = label_model(target, off_label=0.2, dense=dense)
    (p, allowed), = digit_scorer(model, tokenizer(), 25)("", ["s"])
    labels = number_labels(25)
    assert {labels[k]: v for k, v in enumerate(p.tolist()) if v > 1e-9} == pytest.approx(target)
    assert allowed == pytest.approx(0.8)
    assert calls == [1, 3]   # first digits, then one call per live prefix (0, 1, 2)


def test_digit_scorer_spreads_a_pruned_prefix_over_its_labels():
    target = {"07": 1 - PRUNE / 2, "21": PRUNE / 4, "24": PRUNE / 4}
    model, calls = label_model(target)
    (p, _), = digit_scorer(model, tokenizer(), 25)("", ["s"])
    assert calls == [1, 1]   # prefix "2" holds less than PRUNE: not expanded
    assert p[6] == pytest.approx(1 - PRUNE / 2)
    assert p[19:25].tolist() == pytest.approx([PRUNE / 2 / 6] * 6)   # "20" … "25"
    assert float(p.sum()) == pytest.approx(1.0)


def test_numbered_choice_through_rotations_keeps_a_content_preference():
    q, n = many(25), 25
    def likes_intent_7(shared, items, extra):
        label = next(line[:2] for line in shared.splitlines() if line[2:4] == ". " and "intent_7:" in line)
        return label_model({label: 1.0})[0](shared, items, extra)
    (p, info), = choice_labels(q, ["s"], digit_scorer(likes_intent_7, tokenizer(), n), tokenizer(), item_ids=["x"])
    assert p["intent_7"] == pytest.approx(1.0) and info["n_orderings"] == 8


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_cached_prefix_matches_a_full_forward(variant):
    """Prefix KV reuse (sliding window 8, KV sharing in dense) must not change the next-token distribution."""
    torch.manual_seed(0)
    model = Gemma4ForCausalLM._from_config(tiny_text_config(variant, len(tokenizer())), attn_implementation="sdpa").eval()
    lm = LocalLM(model, tokenizer())
    shared = "<bos>" + "Options: alpha beta gamma delta epsilon zeta eta theta.\nState:\n"
    items = ["the quick brown fox jumps over the lazy dog. Answer:", "short. Answer:"]
    cached = lm.next_token_logprobs(shared, items)
    sids = tokenizer().encode(shared, add_special_tokens=False)
    assert len(sids) > 8   # the prefix alone overflows the window
    for item, got in zip(items, cached):
        ids = tokenizer().encode(shared + item, add_special_tokens=False)
        assert ids[:len(sids)] == sids   # so the cached path, not the fallback, ran
        with torch.no_grad():
            full = model(input_ids=torch.tensor([ids])).logits[0, -1].log_softmax(-1)
        torch.testing.assert_close(got, full, atol=1e-5, rtol=1e-5)
