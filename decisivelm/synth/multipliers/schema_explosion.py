"""Schema explosion, the operations that need no teacher (docs/guide/08 §8.10).

Each derived row keeps the parent's state and human label; only the schema
changes, and the target is computed from the parent's. Derived rows inherit the
parent's split, tier and schema family (docs/guide/08 §8.4: a family is a base question
plus all its renamings, coarsenings and Noul splits) and record
`lineage.parent_id` / `lineage.multiplier`.

Renaming, re-describing, coarsening and instruction rewording need Gemma
generation and are separate (M4a).
"""
import hashlib
import random
import re

from decisivelm.data.schema import HumanTarget, Lineage, QuestionTargets, Record
from decisivelm.render import as_text

OTHER = "other"
MIN_MASS = 1e-3
POSITIVE_RATE = 0.4      # keep Noul positives at 30–50%


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def word_overlap(a: str, b: str) -> float:
    """Default confusability: Jaccard overlap of description words. Swap in embeddings when chosen."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def _probs(record: Record, qid: str) -> dict[str, float]:
    t = record.targets[qid]
    src = t.human or (t.teacher[0] if isinstance(t.teacher, list) else t.teacher)
    return src.probabilities


def _child(record: Record, qid: str, op: str, key: str, question: dict, target: HumanTarget) -> Record:
    """Validated (so the target is checked against the new question), never a shallow copy of the parent."""
    suffix = hashlib.sha256(key.encode()).hexdigest()[:8]
    lin = record.lineage
    data = record.model_dump()
    data.update({
        "id": f"{record.id}~{op}-{suffix}",
        "lineage": Lineage(source_dataset=lin.source_dataset, source_id=lin.source_id, family_id=lin.family_id,
                           parent_id=record.id, multiplier=f"schema_explosion:{op}").model_dump(),
        "questions": {**data["questions"], qid: question},
        "targets": {**data["targets"], qid: QuestionTargets(human=target).model_dump()},
    })
    return Record.model_validate(data)


def subset(record: Record, qid: str, k: int, rng: random.Random) -> Record | None:
    """Every option with target mass plus k random distractors, in the original order; target renormalised."""
    q, p = record.questions[qid], _probs(record, qid)
    must = {c for c in q.criteria if p[c] >= MIN_MASS}
    others = [c for c in q.criteria if c not in must]
    keep = must | set(rng.sample(others, min(k, len(others))))
    if len(keep) < 2:
        return None
    kept = [c for c in q.criteria if c in keep]
    z = sum(p[c] for c in kept)
    question = {"type": "choice", "instructions": q.instructions, "criteria": {c: q.criteria[c] for c in kept}}
    target = HumanTarget(probabilities={c: p[c] / z for c in kept}, n_raters=_raters(record, qid))
    return _child(record, qid, "subset", "|".join(kept), question, target)


def drop_gold_add_other(record: Record, qid: str, k: int, rng: random.Random, similarity=word_overlap,
                        n_confusable: int = 2) -> Record | None:
    """Remove the gold option and its most confusable options, keep k others, and add a self-contained `other`
    naming what the kept options cover. The answer becomes `other`. Only for one-hot targets."""
    q, p = record.questions[qid], _probs(record, qid)
    gold = max(p, key=p.get)
    if p[gold] < 1 - 1e-6 or gold == OTHER:
        return None
    desc = {c: as_text(d) for c, d in q.criteria.items()}
    candidates = [c for c in q.criteria if c not in (gold, OTHER)]
    confusable = set(sorted(candidates, key=lambda c: -similarity(desc[gold], desc[c]))[:n_confusable])
    pool = [c for c in candidates if c not in confusable]
    if len(pool) < 1:
        return None
    keep = set(rng.sample(pool, min(k, len(pool))))
    kept = [c for c in q.criteria if c in keep]
    criteria = {c: q.criteria[c] for c in kept}
    criteria[OTHER] = "Something other than any of these: " + "; ".join(desc[c] for c in kept)
    question = {"type": "choice", "instructions": q.instructions, "criteria": criteria}
    target = HumanTarget(probabilities={c: float(c == OTHER) for c in criteria}, n_raters=_raters(record, qid))
    return _child(record, qid, "drop_gold", "|".join(kept), question, target)


def noul_splits(record: Record, qid: str, n: int, rng: random.Random) -> list[Record]:
    """n Nouls "is the answer X?", each positive (an option with target mass) with probability POSITIVE_RATE."""
    q, p = record.questions[qid], _probs(record, qid)
    positives = [c for c in q.criteria if p[c] >= MIN_MASS]
    negatives = [c for c in q.criteria if p[c] < MIN_MASS]
    chosen = []
    for _ in range(n):
        pool = [c for c in (positives if rng.random() < POSITIVE_RATE else negatives) if c not in chosen]
        if pool:
            chosen.append(rng.choice(pool))
    out = []
    for c in chosen:
        statement = f"Asked “{as_text(q.instructions)}”, the correct answer is “{as_text(q.criteria[c])}”."
        question = {"type": "noul", "instructions": statement}
        out.append(_child(record, qid, "noul", c, question, HumanTarget(noul=p[c], n_raters=_raters(record, qid))))
    return out


def _raters(record: Record, qid: str):
    h = record.targets[qid].human
    return h.n_raters if h else None
