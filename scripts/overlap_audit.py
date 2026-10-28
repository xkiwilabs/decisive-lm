"""Text overlap between the frozen held-out suite and everything a mix can train on (docs/RELEASE.md).

    uv run python scripts/overlap_audit.py --out results/suite_overlap.md

Word 8-grams of states, as scripts/synth_generate.py filters generated rows. In both directions, per pair of
(training file, suite set):
- suite in train: suite rows with at least `--share` of their 8-grams in that training file's train/val states;
- train in suite: training rows with at least `--share` of their 8-grams in that suite set's states (a short training
  text copied into a long suite state, such as a LEDGAR provision inside a CUAD section).
Texts shorter than 8 words have no 8-grams and are compared exactly (after lower-casing and joining words) instead.
Suite rows found in either direction are written to --drop, one id per line, for the converters to leave out.
"""
import argparse
import collections
import json
import pathlib
import re

from decisivelm.render import as_text

REPO = pathlib.Path(__file__).resolve().parent.parent
SUITE = ["helpsteer3", "measuring_hate_speech", "cuad", "massive"]
TRAINABLE = ["banking77", "banking77_explode", "clinc150", "clinc150_explode", "helpsteer2", "civil_comments",
             "civil_comments_human_matched", "goemotions", "ledgar", "synth_v1", "synth_v2", "synth_v3",
             "measuring_hate_speech", "multinli"]
N = 8
_WORD = re.compile(r"\w+")


def grams(text: str) -> tuple[set[int], str | None]:
    """(8-gram hashes, the joined words when the text is too short to have any)."""
    words = _WORD.findall(text.lower())
    if len(words) < N:
        return set(), " ".join(words) or None
    return {hash(" ".join(words[i:i + N])) for i in range(len(words) - N + 1)}, None


def rows(name: str, suite: bool):
    for line in open(REPO / "data" / "processed" / f"{name}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if (d["split"] == "gold_test") == suite:
            yield d["id"], as_text(d["state"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results" / "suite_overlap.md")
    ap.add_argument("--drop", type=pathlib.Path, default=REPO / "data" / "suite_overlap_drop.txt")
    a = ap.parse_args()
    suite = {s: [(i, *grams(t)) for i, t in rows(s, suite=True)] for s in SUITE}
    suite_index = {s: (set().union(*(g for _, g, _ in r)), {x for _, _, x in r if x}) for s, r in suite.items()}
    table, drop = [], collections.defaultdict(set)
    for name in TRAINABLE:
        train = [(i, *grams(t)) for i, t in rows(name, suite=False)]
        train_grams = set().union(*(g for _, g, _ in train))
        train_short = {x for _, _, x in train if x}
        for s, r in suite.items():
            s_grams, s_short = suite_index[s]
            def hit(g, x, index, short):   # an empty text (no words) overlaps nothing
                return (x in short) if x else bool(g) and sum(y in index for y in g) / len(g) >= a.share
            in_train = [i for i, g, x in r if hit(g, x, train_grams, train_short)]
            in_suite = [i for i, g, x in train if hit(g, x, s_grams, s_short)]
            drop[s] |= set(in_train)
            flagged = set(in_suite)
            for g, x in ((g, x) for i, g, x in train if i in flagged):   # the suite rows that hold each one
                drop[s] |= {j for j, sg, sx in r if (x == sx if x else bool(g) and len(g & sg) / len(g) >= a.share)}
            table.append((name, s, len(train), len(r), len(in_train), len(in_suite)))
        print(f"{name}: {len(train)} train/val rows checked", flush=True)
    lines = ["# Overlap between the frozen suite and trainable data", "",
             f"Word {N}-grams of states; a row overlaps when at least {a.share:.0%} of its {N}-grams occur in the "
             f"other side (texts under {N} words: exact match). Written by `scripts/overlap_audit.py`.", "",
             "| Training file | Suite set | training rows | suite rows | suite rows found in training | "
             "training rows found in suite |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {s} | {nt:,} | {ns:,} | {a_:,} | {b:,} |" for n, s, nt, ns, a_, b in table]
    lines += ["", "Suite rows to leave out: " + ", ".join(f"{s} {len(drop[s]):,}" for s in SUITE)]
    a.out.write_text("\n".join(lines) + "\n")
    a.drop.write_text("".join(f"{i}\n" for s in SUITE for i in sorted(drop[s])))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
