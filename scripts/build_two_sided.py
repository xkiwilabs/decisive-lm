"""Texts and two-sided Scores for the rating app's two_sided_v1 project (docs/RELEASE.md, change log 2026-10-27): human
ratings with several raters per item on 5-level scales whose middle is neutral, the kind of rating the general head
does not learn to be unsure about from other tasks. Rated first as a fresh development check for 0.2, later as
training data.

    uv run python scripts/build_two_sided.py --out data/processed/two_sided_v1.jsonl
    uv run python scripts/export_for_rating.py --data data/processed/two_sided_v1.jsonl --split gold_test gold_calib \
        --task-set two_sided_v1 --out data/rating/two_sided_v1.items.jsonl   # warning set at project level in the app

- args.me (debate portals; CC BY 4.0): N_ARGS arguments, an equal share from each portal, one per debate, 40–180
  words; asked whether the author is for or against the debate's title, how convincing the argument is, and whether
  the author is serious or joking
- HelpSteer2 (CC BY 4.0): N_CHATS single-turn conversations from its validation rows (never trained on; the pilot's
  prompts left out), one per prompt, the user's message ≤ 120 words and the response ≤ 200; asked how formal the
  user's message is, how sure the response sounds, and whether the response agrees or disagrees with the user
- no targets: the ratings are the targets. Splits by family: 60% gold_test, 40% gold_calib
- Sentiment and respect are left out: they are Measuring Hate Speech items (docs/RELEASE.md, suite-task rule)
"""
import argparse, collections, glob, hashlib, json, pathlib, random, urllib.parse, zipfile

from decisivelm.data.dataset import load_records
from decisivelm.data.schema import Lineage, Preprocess, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "two_sided_v1"
N_ARGS, N_CHATS, CALIB_SHARE = 360, 240, 0.4
PORTALS = ("www.debate.org", "debatewise.org", "idebate.org", "www.debatepedia.org")   # hosts


def _score(instructions: str, low: str, middle: str, high: str, very=("Very", "Somewhat")) -> dict:
    return {"type": "score", "instructions": instructions,
            "criteria": [f"{very[0]} {low}", f"{very[1]} {low}", middle, f"{very[1]} {high}", f"{very[0]} {high}"]}


ARG_QUESTIONS = {
    "convincing": _score("How convincing is this argument?", "unconvincing", "Neither convincing nor unconvincing",
                         "convincing"),
    "serious": {"type": "score", "instructions": "Is the author serious or joking?",
                "criteria": ["Clearly joking", "Mostly joking", "Partly serious, partly joking", "Mostly serious",
                             "Clearly serious"]},
}
CHAT_QUESTIONS = {
    "user_formal": _score("How formal is the user's message?", "informal", "Neither formal nor informal", "formal"),
    "response_sure": _score("How sure does the response sound?", "unsure", "Neither sure nor unsure", "sure"),
    "response_agrees": {"type": "score",
                        "instructions": "Does the response agree or disagree with what the user says or assumes?",
                        "criteria": ["Clearly disagrees", "Partly disagrees", "Neither agrees nor disagrees",
                                     "Partly agrees", "Clearly agrees"]},
}


def for_against(title: str) -> dict:
    return _score(f'Is the author for or against this: "{title}"?', "against", "Neither for nor against", "for",
                  very=("Strongly", "Somewhat"))


def _split(family: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{family}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < CALIB_SHARE else "gold_test"


def _record(source: str, source_id: str, family: str, state, questions: dict) -> Record:
    return Record(id=f"{SOURCE}-{source}-{source_id}",
                  lineage=Lineage(source_dataset=source, source_id=source_id, family_id=family),
                  license="CC-BY-4.0", tier="R", modality=["text"], language="en", state=state,
                  questions=questions, targets={},
                  preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS,
                                        truncated=False),
                  split=_split(family))


def portal(argument: dict) -> str | None:
    host = urllib.parse.urlparse(argument["context"]["sourceUrl"].lower()).hostname
    return host if host in PORTALS else None


def arguments(zip_path: pathlib.Path, n: int, seed: int) -> list[Record]:
    with zipfile.ZipFile(zip_path) as z, z.open("args-me-1.0-cleaned.json") as f:
        args = json.load(f)["arguments"]
    by_portal = collections.defaultdict(dict)   # one argument per debate
    rng = random.Random(seed)
    rng.shuffle(args)
    for a in args:
        text = " ".join(p["text"] for p in a["premises"]).strip()
        title = a["context"]["discussionTitle"].strip()
        if (p := portal(a)) and title and 40 <= len(text.split()) <= 180:
            by_portal[p].setdefault(a["context"]["sourceId"], (a, title, text))
    out = []
    for p in PORTALS:
        for a, title, text in list(by_portal[p].values())[:n // len(PORTALS)]:
            out.append(_record("args_me", a["id"], f"args_me-{a['context']['sourceId']}",
                               f"Debate: {title}\nArgument: {text}", {"for_against": for_against(title), **ARG_QUESTIONS}))
    return out


def chats(helpsteer2: pathlib.Path, n: int, seed: int, exclude_families: set[str]) -> list[Record]:
    seen, out = set(), []
    rows = [r for r in load_records(helpsteer2) if r.split in ("val", "gold_calib", "gold_test")]
    random.Random(seed).shuffle(rows)
    for r in rows:
        c, resp = r.state["conversation"], r.state["response"]
        fam = r.lineage.family_id
        if ("Assistant:" in c or len(c.split()) > 120 or len(resp.split()) > 200 or fam in exclude_families
                or fam in seen):
            continue
        seen.add(fam)
        out.append(_record("helpsteer2", r.lineage.source_id, fam, {"conversation": c, "response": resp},
                           CHAT_QUESTIONS))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args-me", type=pathlib.Path, default=pathlib.Path("data/raw/args_me/argsme-1.0-cleaned.zip"))
    ap.add_argument("--helpsteer2", type=pathlib.Path, default=pathlib.Path("data/processed/helpsteer2.jsonl"))
    ap.add_argument("--exclude-items", default="data/rating/helpsteer2_pilot*.items.jsonl",
                    help="rating files whose families are left out (the pilot's prompts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    exclude = {json.loads(l)["source"]["family"] for f in glob.glob(a.exclude_items) for l in open(f)}
    rows = arguments(a.args_me, N_ARGS, a.seed) + chats(a.helpsteer2, N_CHATS, a.seed, exclude)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    print(f"{len(rows):,} texts, {sum(len(r.questions) for r in rows):,} questions -> {a.out}",
          dict(collections.Counter((r.lineage.source_dataset, r.split) for r in rows)))


if __name__ == "__main__":
    main()
