"""A mix with every teacher-labelled Score question removed from its train and val rows (the probe of 2026-10-31).

    uv run python scripts/drop_teacher_score.py --mix general_v5c --name general_v5c_nts

Question: does the head's Score confidence fail to follow rater disagreement (wiki_toxicity, MHS) because most of its
Score targets come from the teacher, which avoids neutral middles? Same rows otherwise: a Score question is kept only
if its target has a human part; rows left with no targeted question are dropped. gold_calib and gold_test are copied unchanged.
"""
import argparse, collections, json, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="the mix under data/mixes/ to derive from")
    ap.add_argument("--name", required=True, help="the new mix's name")
    a = ap.parse_args()
    src, out = REPO / "data/mixes" / f"{a.mix}.jsonl", REPO / "data/mixes" / f"{a.name}.jsonl"
    dropped, rows = collections.Counter(), collections.Counter()
    with open(src) as f, open(out, "w") as g:
        for line in f:
            r = json.loads(line)
            if r["split"] in ("train", "val"):
                for qid, q in list(r["questions"].items()):
                    if q["type"] == "score" and not r["targets"].get(qid, {}).get("human"):
                        del r["questions"][qid]
                        r["targets"].pop(qid, None)
                        dropped[(r["split"], r["lineage"]["source_dataset"])] += 1
                if not any(r["targets"].get(qid) for qid in r["questions"]):   # nothing left to learn from
                    rows[r["split"]] += 1
                    continue
                line = json.dumps(r, ensure_ascii=False) + "\n"
            g.write(line)
    manifest = json.loads((REPO / "data/mixes" / f"{a.mix}.manifest.json").read_text())
    manifest.update(name=a.name, derived_from=a.mix, filter="teacher-labelled Score questions removed from train/val",
                    dropped_questions={f"{s}/{d}": n for (s, d), n in sorted(dropped.items())},
                    dropped_rows=dict(rows))
    (REPO / "data/mixes" / f"{a.name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    for (s, d), n in sorted(dropped.items()):
        print(f"{s:5s} {d:20s} {n:7d} Score questions removed")
    print(f"rows left with no targeted question, dropped: {dict(rows)}\nwrote {out}")


if __name__ == "__main__":
    main()
