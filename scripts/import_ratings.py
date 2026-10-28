"""Turn the human-rating app's export (DecisiveLM-HV, format hv-ratings/1) into human targets.

    uv run python scripts/import_ratings.py --ratings data/rating/helpsteer2_pilot.ratings.jsonl \
        --data data/processed/helpsteer2.jsonl --out data/processed/helpsteer2_rated.jsonl [--min-raters 3] \
        [--manifest data/rating/helpsteer2_pilot.manifest.json] [--meta-out data/rating/synth_v2_check.meta.jsonl]

Each rated question's human target becomes the share of raters' first answers on each option (Choice), level (Score)
or "yes" (Noul), with the number of raters; teacher targets are kept. Questions with fewer than --min-raters answers
keep their old target. Only rated rows are written. Prints the mean pair agreement per question and each rater's
consistency on repeated items. Over-quota answers (a late first answer after the item had its raters) count like any
other first answer; their number is printed.

Data-check items' meta-questions (`meta_questions`) judge the item and never become targets. Their yes shares are
summarised, and --meta-out writes one line per item ({"item_id", "meta": {qid: [yes share, answers]}, "flagged"}), an
item being flagged when any meta-question has a yes share below .5.

Training and check lines (`role`) never become targets; each rater's check answers are scored against the item's
reference (its human consensus). With --manifest (hv-manifest/1), the ratings file's sha256 and line count are checked
against it and its per-rater exclusions are printed. Items raters reported a problem with (`item_flags`) and "can't
judge" answers (`unable_reason`) are counted.
"""
import argparse, collections, hashlib, json, pathlib, statistics

from decisivelm.data.dataset import load_records
from decisivelm.data.rating import (apply_ratings, check_results, counted, meta_shares, pair_agreement,
                                    repeat_consistency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=pathlib.Path, required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--meta-out", type=pathlib.Path, help="data-check items: meta-question shares per item")
    ap.add_argument("--manifest", type=pathlib.Path, help="the export's hv-manifest/1 file, checked against --ratings")
    a = ap.parse_args()

    ratings = [json.loads(line) for line in open(a.ratings)]
    if a.manifest:
        m = json.loads(a.manifest.read_text())
        if m.get("format") != "hv-manifest/1":
            raise SystemExit(f"{a.manifest}: expected hv-manifest/1, got {m.get('format')!r}")
        digest = hashlib.sha256(a.ratings.read_bytes()).hexdigest()
        if (m["ratings_sha256"], m["ratings_lines"]) != (digest, len(ratings)):
            raise SystemExit(f"{a.ratings} does not match {a.manifest}: sha256 {digest[:12]} and {len(ratings)} lines, "
                             f"manifest {m['ratings_sha256'][:12]} and {m['ratings_lines']}")
        print(f"manifest: {m['project']['slug']} / {m['task_set']['id']}, exported {m['exported_at']}, "
              f"{m['items']['complete']:,} of {m['items']['total']:,} items complete")
        for r in m.get("per_rater", []):
            if r.get("excluded"):
                print(f"  rater {r['rater']} excluded: {r.get('exclusion_reason')}")
    by_id = {r.id: r for r in load_records(a.data)}
    if missing := [x["item_id"] for x in ratings if x["item_id"] not in by_id]:
        raise SystemExit(f"{len(missing)} rated items are not in {a.data}, e.g. {missing[:3]}")
    agreement, n_raters, updated, over_quota = collections.defaultdict(list), [], 0, 0
    roles = collections.Counter(x.get("role", "rate") for x in ratings)
    unable = collections.Counter(r["unable_reason"] for x in ratings for q in x["questions"].values()
                                 for r in q["responses"] if r.get("unable_reason"))
    with open(a.out, "w") as f:
        for x in ratings:
            if x.get("role", "rate") != "rate":
                continue
            record, k = apply_ratings(by_id[x["item_id"]], x, a.min_raters)
            updated += k
            f.write(record.model_dump_json(exclude_none=True) + "\n")
            for qid, q in x["questions"].items():
                if qid in x.get("meta_questions", []):
                    continue
                first = counted(q["responses"])
                n_raters.append(len(first))
                over_quota += sum(bool(r.get("over_quota")) for r in first)
                if (p := pair_agreement([r["value"] for r in first])) is not None:
                    agreement[qid].append(p)
    print(f"{roles['rate']:,} rated items, {updated:,} question targets from >= {a.min_raters} raters "
          f"(median {statistics.median(n_raters) if n_raters else 0} raters per question) -> {a.out}")
    if roles["training"] or roles["check"]:
        print(f"  {roles['training']:,} training and {roles['check']:,} check items, not used as targets")
    if flagged_items := sum(bool(x.get("item_flags")) for x in ratings):
        print(f"  {flagged_items:,} items reported by a rater (item_flags)")
    if unable:
        print("  can't-judge answers: " + ", ".join(f"{k} {v:,}" for k, v in unable.most_common()))
    if over_quota:
        print(f"  {over_quota:,} counted answers were over quota (submitted after the item had its raters)")
    for qid, v in sorted(agreement.items()):
        print(f"  {qid}: mean pair agreement {statistics.mean(v):.3f} over {len(v):,} items")
    meta = {x["item_id"]: m for x in ratings if (m := meta_shares(x))}
    if meta:
        by_q = collections.defaultdict(list)
        for m in meta.values():
            for qid, (share, _) in m.items():
                by_q[qid].append(share)
        flagged = {i for i, m in meta.items() if any(share < 0.5 for share, _ in m.values())}
        print(f"  data check: {len(flagged):,} of {len(meta):,} items flagged (a meta-question's yes share < .5)")
        for qid, v in sorted(by_q.items()):
            print(f"    {qid}: mean yes share {statistics.mean(v):.3f} over {len(v):,} items")
        if a.meta_out:
            with open(a.meta_out, "w") as f:
                for i, m in meta.items():
                    f.write(json.dumps({"item_id": i, "meta": {q: list(v) for q, v in m.items()},
                                        "flagged": i in flagged}) + "\n")
    for rater, (same, n) in sorted(repeat_consistency(ratings).items()):
        print(f"  rater {rater}: {same}/{n} repeats answered the same")
    for rater, (right, n) in sorted(check_results(ratings, by_id).items()):
        print(f"  rater {rater}: {right}/{n} check answers matched the reference")


if __name__ == "__main__":
    main()
