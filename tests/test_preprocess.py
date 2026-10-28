"""State truncation (docs/guide/04 §4.6): head+tail, 70% / 30% of the budget, with a marker; shared by teacher and student."""
from decisivelm.preprocess import MARKER, truncate_state
from tiny_gemma import tokenizer

LONG = " ".join(f"word{i}" for i in range(2000))


def n_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False))


def test_short_state_is_unchanged():
    assert truncate_state("A short state.", tokenizer(), 64) == ("A short state.", False)


def test_long_state_fits_the_budget_and_is_flagged():
    text, truncated = truncate_state(LONG, tokenizer(), 100)
    assert truncated
    assert n_tokens(text) <= 100
    assert MARKER in text


def test_long_state_keeps_head_and_tail():
    text, _ = truncate_state(LONG, tokenizer(), 100)
    head, tail = text.split(MARKER)
    assert LONG.startswith(head)
    assert LONG.endswith(tail)


def test_head_gets_seventy_percent_of_the_budget():
    text, _ = truncate_state(LONG, tokenizer(), 200)
    head, tail = text.split(MARKER)
    budget = 200 - n_tokens(MARKER)
    assert abs(n_tokens(head) - 0.7 * budget) <= 2
    assert abs(n_tokens(tail) - 0.3 * budget) <= 2
