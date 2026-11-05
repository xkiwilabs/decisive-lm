"""Shared by the photo question-answering converters (VizWiz-VQA, TextVQA): ten crowd answers per question become one
Choice whose targets are the shares of the answerers who gave each answer.

- answers are compared after light normalisation (lower case, outer spaces and a final full stop removed, inner spaces
  collapsed), so "Dakota." and "dakota" count as one answer; paraphrases ("dakota", "dakota digital") stay apart, and
  the head learns to split its confidence between them as the answerers did
- a fixed number of options per question, so the count of options never tells the head how much the answerers
  disagreed: two (yes, no) for yes/no questions, else four. The four are the most given answers, filled up when fewer
  with answers to other questions from the same source, which no answerer gave here (target 0)
- answerers whose answer is not among the options are left out of `n_raters`; a question is skipped when fewer than
  half of the answerers who gave an answer are left (`MIN_COVERED`)
- "unanswerable" is never an option: VizWiz asks it as a separate yes/no question
"""
import hashlib
import random
import re
from collections import Counter

OPTIONS = 4
MIN_COVERED = 0.5
NOT_ANSWERS = {"unanswerable", "unsuitable", "unsuitable image"}
INSTRUCTIONS = "What is the answer to the question about the photo?"


def normalise(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower()).rstrip(".").strip()


def _rank(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def choice(answers: list[str], pool: list[str], key: str, yes_no: bool = False) -> tuple[dict, dict, int] | None:
    """(question, probabilities, n_raters) for one question's crowd answers, or None when too few are covered.
    `pool` is the source's distinct normalised answers in a fixed order, for filling up; `key` seeds which fillers are
    drawn and the option order."""
    given = [a for a in map(normalise, answers) if a and a not in NOT_ANSWERS]
    counts = Counter(given)
    if yes_no:
        options = ["yes", "no"]
    else:
        options = [a for a, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:OPTIONS]]
        draw = random.Random(key).sample(pool, min(len(pool), 2 * OPTIONS + len(counts)))
        options += [a for a in draw if a not in counts][:OPTIONS - len(options)]
    n = sum(counts[o] for o in options)
    if len(options) < 2 or n == 0 or n < MIN_COVERED * len(given):
        return None
    options.sort(key=lambda a: _rank(f"{key}:order:{a}"))
    names = [chr(ord("a") + i) for i in range(len(options))]
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(zip(names, options))}
    return question, {k: counts[o] / n for k, o in zip(names, options)}, n
