"""Exchange with a human-rating app: items out, ratings back.

Formats (docs/rating-formats.md): one JSON object per line.
- hv-items/1: one item per record, the state and questions exactly as the model sees them, and nothing a rater
  should not see (no targets, no model or teacher answers).
- hv-ratings/1: one line per item, every rater's response to each question.

Ratings become human targets: the share of first responses on each option (Choice), level (Score) or "yes" (Noul),
with the number of raters. Repeat responses (a rater seeing an item again) measure consistency and are not counted.
Training and check lines (`role`) never become targets: training raters saw the reference answer, and check items
measure the raters (`check_results`). Unknown fields are ignored: the app adds fields within a version.

Data-check items (a data-check project, docs/rating-formats.md) add meta-questions about the item itself, named in the item's
`meta_questions`; their answers judge the item and never become targets. `meta_questions` and `reference_source`
are newer fields, left out with `a1_fields=False` for an app that predates them.
"""
import collections
import itertools
import random

from decisivelm.api_types import ChoiceQuestion, ScoreQuestion
from decisivelm.data.schema import HumanTarget, QuestionTargets, Record

ITEMS_FORMAT, RATINGS_FORMAT = "hv-items/1", "hv-ratings/1"
# Meta-questions for data-check items, after the item's own questions. Raters see them as ordinary Nouls.
META = {
    "meta_answerable": {"type": "noul", "instructions": "Each question above can be answered from the text"},
    "meta_distinct": {"type": "noul", "instructions": "In each question above, the options or levels are clearly "
                                                      "different from one another"},
    "meta_realistic": {"type": "noul", "instructions": "The text reads like something a real person would write"},
}


def data_class(record: Record, registry: dict) -> str:
    """public (downloadable from a pinned public source), synthetic (written by the teacher), or private."""
    name = record.lineage.source_dataset
    if name.startswith("synth"):
        return "synthetic"
    return "public" if "fetch" in registry.get(name, {}) else "private"


def to_item(record: Record, registry: dict, task_set: str, role: str = "rate", questions: list[str] | None = None,
            content_warnings: list[str] = (), data_check: bool = False, a1_fields: bool = False) -> dict:
    if record.media:
        raise ValueError(f"{record.id}: media items are not supported yet (text first)")
    if data_check and (role != "rate" or not a1_fields):
        raise ValueError("data-check items are rate items and need the app's A1 fields (meta_questions)")
    qids = questions or list(record.questions)
    if clash := set(qids) & set(META):
        raise ValueError(f"{record.id}: question ids {sorted(clash)} are reserved for meta-questions")
    item = {"format": ITEMS_FORMAT, "item_id": record.id, "task_set": task_set,
            "source": {"dataset": record.lineage.source_dataset, "id": record.lineage.source_id,
                       "family": record.lineage.family_id},
            "license": record.license, "tier": record.tier, "data_class": data_class(record, registry),
            "language": record.language, "state": record.state,
            "questions": {q: record.questions[q].model_dump(exclude_none=True) for q in qids},
            "role": role, "content_warnings": list(content_warnings)}
    if role in ("training", "check"):
        item["reference"] = {q: reference_answer(record, q) for q in qids}
        if a1_fields:
            item["reference_source"] = "human_consensus"
    if data_check:
        meta = [m for m in META if m != "meta_distinct" or any(record.questions[q].type != "noul" for q in qids)]
        item["questions"] |= {m: META[m] for m in meta}
        item["meta_questions"] = meta
    return item


def reference_answer(record: Record, qid: str):
    """The human consensus for a training or check item: the most chosen option or level, or yes/no at a share of
    .5. Only a human target can be a reference, never a teacher's or a model's."""
    t = record.targets.get(qid)
    if t is None or t.human is None:
        raise ValueError(f"{record.id} {qid}: a training item needs a human target")
    q = record.questions[qid]
    if q.type == "noul":
        return t.human.noul >= 0.5
    best = max(t.human.probabilities, key=t.human.probabilities.get)
    return int(best) if q.type == "score" else best


def top_share(record: Record, qid: str) -> float:
    """The human share of the most chosen answer (Noul: of yes or no); 0 without a human target."""
    t = record.targets.get(qid)
    if t is None or t.human is None:
        return 0.0
    return max(t.human.noul, 1 - t.human.noul) if t.human.noul is not None else max(t.human.probabilities.values())


def sample(records: list[Record], n: int | None, seed: int) -> list[Record]:
    if n is None or n >= len(records):
        return records
    return random.Random(seed).sample(records, n)


def option_keys(q) -> list[str]:
    if isinstance(q, ChoiceQuestion):
        return list(q.criteria)
    if isinstance(q, ScoreQuestion):
        return [str(i) for i in range(len(q.criteria))]
    return ["yes", "no"]


def counted(responses: list[dict]) -> list[dict]:
    """The responses that become a target: each rater's first answer, unless they could not judge the question or the
    app marks the rater excluded (an admin's decision, with a reason)."""
    return [r for r in responses
            if r.get("kind", "first") == "first" and r.get("value") is not None and not r.get("excluded")]


def human_target(q, responses: list[dict]) -> HumanTarget | None:
    first = counted(responses)
    if not first:
        return None
    keys = option_keys(q)
    values = [("yes" if r["value"] else "no") if q.type == "noul" else str(r["value"]) for r in first]
    if unknown := set(values) - set(keys):
        raise ValueError(f"answers {sorted(unknown)} are not options of the question ({keys})")
    counts = collections.Counter(values)
    if q.type == "noul":
        return HumanTarget(noul=counts["yes"] / len(values), n_raters=len(values))
    return HumanTarget(probabilities={k: counts[k] / len(values) for k in keys}, n_raters=len(values))


def apply_ratings(record: Record, rating: dict, min_raters: int) -> tuple[Record, int]:
    """The record with each rated question's human target replaced; returns (record, questions updated)."""
    if rating.get("format") != RATINGS_FORMAT:
        raise ValueError(f"{record.id}: expected {RATINGS_FORMAT}, got {rating.get('format')!r}")
    if rating.get("role", "rate") != "rate":   # training and check answers are not independent judgements to learn
        return record, 0
    targets = dict(record.targets)
    updated = 0
    meta = set(rating.get("meta_questions", []))
    for qid, r in rating["questions"].items():
        if qid in meta:   # about the item, not its content: never a target (meta_shares)
            continue
        if qid not in record.questions:
            raise ValueError(f"{record.id}: rating for unknown question {qid!r}")
        q = record.questions[qid]
        if r["type"] != q.type:
            raise ValueError(f"{record.id} {qid}: rated as {r['type']}, the record says {q.type}")
        h = human_target(q, r["responses"])
        if h is None or h.n_raters < min_raters:
            continue
        old = targets.get(qid) or QuestionTargets()
        targets[qid] = QuestionTargets(human=h, teacher=old.teacher)
        updated += 1
    return record.model_copy(update={"targets": targets}), updated


def meta_shares(rating: dict) -> dict[str, tuple[float, int]]:
    """{meta qid: (share of counted first answers that said yes, answers)} for a data-check item's ratings."""
    out = {}
    for qid in rating.get("meta_questions", []):
        first = counted(rating["questions"][qid]["responses"])
        if first:
            out[qid] = (sum(bool(r["value"]) for r in first) / len(first), len(first))
    return out


def check_results(ratings: list[dict], records: dict[str, Record]) -> dict[str, tuple[int, int]]:
    """{rater: (check answers that matched the item's reference, check answers)}, first answers only."""
    out = collections.defaultdict(lambda: [0, 0])
    for rating in ratings:
        if rating.get("role") != "check":
            continue
        record = records[rating["item_id"]]
        for qid, r in rating["questions"].items():
            ref = reference_answer(record, qid)
            for x in counted(r["responses"]):
                out[x["rater"]][0] += x["value"] == ref
                out[x["rater"]][1] += 1
    return {k: tuple(v) for k, v in out.items()}


def pair_agreement(values: list) -> float | None:
    """The share of rater pairs that gave the same answer (the manuscript's per-item agreement)."""
    if len(values) < 2:
        return None
    pairs = list(itertools.combinations(values, 2))
    return sum(a == b for a, b in pairs) / len(pairs)


def repeat_consistency(ratings: list[dict]) -> dict[str, tuple[int, int]]:
    """{rater: (repeats answered the same as their first answer, repeats)}."""
    out = collections.defaultdict(lambda: [0, 0])
    for rating in ratings:
        for r in rating["questions"].values():
            first = {x["rater"]: x["value"] for x in r["responses"] if x.get("kind", "first") == "first"}
            for x in r["responses"]:
                if x.get("kind") == "repeat" and x["rater"] in first and x.get("value") is not None:
                    out[x["rater"]][0] += x["value"] == first[x["rater"]]
                    out[x["rater"]][1] += 1
    return {k: tuple(v) for k, v in out.items()}
