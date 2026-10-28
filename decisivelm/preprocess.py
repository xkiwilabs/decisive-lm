"""Input handling shared by the teacher labeller and the model (docs/guide/04 §4.6).

Anything the teacher sees differently from the student becomes label noise, so
both call these functions on the serialised state before rendering.
"""
MAX_STATE_TOKENS = 8192
HEAD_SHARE = 0.7
MARKER = "\n[…]\n"


def truncate_state(text: str, tokenizer, max_tokens: int = MAX_STATE_TOKENS) -> tuple[str, bool]:
    """(text, truncated). Over budget: the first 70% and last 30% of the budget, joined by MARKER."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, False
    budget = max_tokens - len(tokenizer.encode(MARKER, add_special_tokens=False))
    head = round(budget * HEAD_SHARE)
    return tokenizer.decode(ids[:head]) + MARKER + tokenizer.decode(ids[len(ids) - (budget - head):]), True
