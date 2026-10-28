"""Items for the rating app's dev_v1 project: a fresh development set for choosing between general heads (0.2 on),
rated by 5 people per item, on the kinds of question the general head is trained on. The old development sets have
been used to choose between mixes, and most have one label per item, so they cannot say whether the head's confidence
follows human agreement on these tasks. The ratings are the targets.

    uv run python scripts/build_dev_v1.py --out data/processed/dev_v1.jsonl
    uv run python scripts/export_for_rating.py --data data/processed/dev_v1.jsonl --split gold_test gold_calib \
        --task-set dev_v1 --out data/rating/dev_v1.items.jsonl   # warnings set at project level in the app

Each item is one question on a row of a source's val split, never trained on: rows whose text (source id) or prompt
family is in any source's train split are left out, as are the prompts already in another rating project. Short
texts only, so an item takes a rater well under a minute:

- intent (Banking77 and CLINC150, 60 each, the schema-explosion rows with 2-8 options): which intent a message expresses
- toxicity (Civil Comments): is the comment toxic (Noul)
- emotional tone (GoEmotions): positive, negative, ambiguous or neutral
- helpfulness (HelpSteer2): how helpful the response is, five levels
- preference (HelpSteer2-Preference): which of two responses is better and by how much, six levels (40: the
  short val pairs run out)

Splits by source item (by prompt for the HelpSteer2 sets): 40% gold_calib, 60% gold_test.
"""
import argparse, collections, glob, hashlib, json, pathlib, random

from decisivelm.data.dataset import load_records
from decisivelm.data.schema import Record

SOURCE = "dev_v1"
CALIB_SHARE = 0.4
P = pathlib.Path("data/processed")
# file, question id, items, predicate on (record, question) beyond the length limits
TASKS = {
    "intent_banking": ("banking77_explode.jsonl", "intent", 60, lambda r, q: q.type == "choice" and 2 <= len(q.criteria) <= 8),
    "intent_clinc": ("clinc150_explode.jsonl", "intent", 60, lambda r, q: q.type == "choice" and 2 <= len(q.criteria) <= 8),
    "toxicity": ("civil_comments.jsonl", "toxicity", 120, lambda r, q: True),
    "tone": ("goemotions.jsonl", "sentiment", 120, lambda r, q: True),
    "helpfulness": ("helpsteer2.jsonl", "helpfulness", 120, lambda r, q: True),
    "preference": ("helpsteer2_preference.jsonl", "preference", 40, lambda r, q: True),
}
TRAINED = ["banking77.jsonl", "banking77_explode.jsonl", "clinc150.jsonl", "clinc150_explode.jsonl",
           "civil_comments.jsonl", "goemotions.jsonl", "helpsteer2.jsonl", "helpsteer2_preference.jsonl"]
MAX_WORDS = {"text": 150, "conversation": 120, "response": 250, "response_1": 250, "response_2": 250}


def short(state) -> bool:
    if isinstance(state, str):
        return len(state.split()) <= MAX_WORDS["text"]
    c = state.get("conversation", "")
    return ("Assistant:" not in c and all(len(str(state[k]).split()) <= MAX_WORDS[k] for k in state if k in MAX_WORDS))


def whole_dataset_family(family: str) -> bool:
    return family.endswith("-base")


def trained(paths: list[pathlib.Path]) -> tuple[set, set]:
    ids, families = set(), set()
    for p in paths:
        for r in load_records(p):
            if r.split == "train":
                ids.add((r.lineage.source_dataset, r.lineage.source_id))
                if not whole_dataset_family(r.lineage.family_id):
                    families.add(r.lineage.family_id)
    return ids, families


def group(r: Record) -> str:   # the unit a split and an exclusion go by
    fam = r.lineage.family_id
    return f"{r.lineage.source_dataset}-{r.lineage.source_id}" if whole_dataset_family(fam) else fam


def _split(g: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{g}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < CALIB_SHARE else "gold_test"


def pick(rows: list[Record], qid: str, n: int, keep, ids: set, families: set, used: set, seed: int) -> list[Record]:
    rows = [r for r in rows if r.split == "val" and qid in r.questions]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        g = group(r)
        if ((r.lineage.source_dataset, r.lineage.source_id) in ids or r.lineage.family_id in families or g in used
                or not short(r.state) or not keep(r, r.questions[qid])):
            continue
        used.add(g)
        out.append(r.model_copy(update={"id": f"{SOURCE}-{r.id}", "questions": {qid: r.questions[qid]},
                                        "targets": {}, "split": _split(g)}))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=pathlib.Path, default=P)
    ap.add_argument("--exclude-items", default="data/rating/*.items.jsonl",
                    help="rating files whose prompts are left out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    ids, families = trained([a.processed / f for f in TRAINED])
    used = {json.loads(l)["source"]["family"] for f in glob.glob(a.exclude_items) if SOURCE not in f for l in open(f)}
    rows = []
    for task, (file, qid, n, keep) in TASKS.items():
        got = pick(load_records(a.processed / file), qid, n, keep, ids, families, used, a.seed)
        if len(got) < n:
            raise SystemExit(f"{task}: only {len(got)} of {n} rows qualify")
        rows += got
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    print(f"{len(rows):,} items -> {a.out}",
          dict(collections.Counter((r.lineage.source_dataset, r.split) for r in rows)))


if __name__ == "__main__":
    main()
