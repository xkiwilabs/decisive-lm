"""Block text templates (docs/guide/06 §6.2). Bump RENDER_VERSION whenever any output here changes.

Question ids, option indices and sibling options are never rendered into a
leaf. Query tokens are not text: the packer appends their reserved ids.
"""
import json

from decisivelm.api_types import ChoiceQuestion, Content, NoulCriteria, NoulQuestion, ScoreQuestion

RENDER_VERSION = 1


def as_text(x: Content) -> str:
    """Strings as they are; objects and arrays as compact JSON in insertion order."""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


MEDIA_SEPARATOR = "\n"   # after each image's placeholder tokens, before the next image or the state text


def prefix_text(state_text: str) -> str:
    """Text part of the shared prefix. Media placeholders go before it (docs/guide/04 §4.5)."""
    return f"State:\n{state_text}\n"


def question_segment(q: ChoiceQuestion | ScoreQuestion | NoulQuestion) -> str:
    if isinstance(q, NoulQuestion):
        return ""        # the statement lives in the leaf
    return f"Q: {as_text(q.instructions)}\n"


def _sentence(s: str) -> str:
    return s.rstrip().rstrip(".") + "."


def leaf_texts(q: ChoiceQuestion | ScoreQuestion | NoulQuestion) -> list[str]:
    if isinstance(q, ChoiceQuestion):
        # An empty description (the option's name says it all) leaves no dangling dash.
        return [f"Option: {name} — {as_text(desc)}" if as_text(desc).strip() else f"Option: {name}"
                for name, desc in q.criteria.items()]
    if isinstance(q, ScoreQuestion):
        return [f"Level: {as_text(desc)}" for desc in q.criteria]
    text = f"Statement: {_sentence(as_text(q.instructions))}"
    c = q.criteria
    if isinstance(c, str):
        text += f" Criteria: {_sentence(c)}"
    elif isinstance(c, NoulCriteria):
        if c.yes is not None:
            text += f" Yes means: {_sentence(as_text(c.yes))}"
        if c.no is not None:
            text += f" No means: {_sentence(as_text(c.no))}"
    return [text]
