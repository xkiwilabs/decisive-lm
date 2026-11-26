# Paired bootstrap, v5c seed 0 minus P0 and minus the teacher, on the coding gold_test (completed labels): errors
# caught at a 20% review, rho(top p, agreement), kappa (mean over 4 coders) and accuracy. Same data, resamples
# (2,000, seed 0) and definitions as scripts/compare_coding.py.
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from compare_coding import HUMANS, kappa_boot, ci
from decisivelm.data.dataset import load_records
from decisivelm.eval import pair_agreement, review_curve, spearman
rows = [r for r in load_records("data/processed/llm_response_coding.jsonl") if r.split == "gold_test"]
names = list(rows[0].questions["intent"].criteria); code = {c.lower(): i for i, c in enumerate(names)}
ids = [r.id for r in rows]
soft = np.array([[r.targets["intent"].human.probabilities[c] for c in names] for r in rows])
codes = pd.read_csv("data/raw/llm_response_coding/full_coding_matrix.csv", usecols=["SubjectID", "TrialNumber", *HUMANS])
codes.index = "lrc-" + codes.SubjectID.astype(str) + "-" + codes.TrialNumber.astype(str)
humans = {h: np.array([code[v.strip().lower()] if isinstance(v, str) else -1 for v in codes.loc[ids, h]]) for h in HUMANS}
P = "data/predictions/"
files = {"v5c": P + "general_v5c_e4b_r8_seed0__llm_response_coding.jsonl", "P0": P + "P0_llm_response_coding_e4b.jsonl",
         "teacher": P + "teacher_llm_response_coding__gemma4_26b_a4b.jsonl"}
probs = {}
for m, f in files.items():
    d = {x["id"]: x.get("probabilities") or x["answers"]["intent"]["probabilities"] for x in map(json.loads, open(f))
         if x["split"] == "gold_test"}
    probs[m] = np.array([[d[i][c] for c in names] for i in ids])
top = np.sort(soft, 1); untied = top[:, -1] > top[:, -2]; consensus = soft.argmax(1)
N = len(ids); rng = np.random.default_rng(0); idx = rng.integers(0, N, (2000, N)); ident = np.arange(N)[None]
agree = pair_agreement(np.stack([humans[h] for h in HUMANS], 1)); has = ~np.isnan(agree)
stat = {}
for m, p in probs.items():
    conf, hard = p.max(1), p.argmax(1); correct = hard == consensus
    kap = lambda I: np.mean([kappa_boot(hard, humans[h], I, len(names)) for h in HUMANS], axis=0)
    caught = lambda i: review_curve(conf[i], correct[i], untied[i], [0.2])[0][0]
    stat[m] = {"errors caught at 20% (points)": (caught(np.arange(N)) * 100, np.array([caught(i) for i in idx]) * 100),
               "rho(top p, agreement)": (spearman(conf[has], agree[has]), np.array([spearman(conf[i][has[i]], agree[i][has[i]]) for i in idx])),
               "kappa (mean of 4 coders)": (kap(ident)[0], kap(idx)),
               "accuracy (points)": (correct[untied].mean() * 100, np.array([correct[i][untied[i]].mean() for i in idx]) * 100)}
for other in ("P0", "teacher"):
    print(f"\nv5c seed 0 minus {other}:")
    for k in stat["v5c"]:
        a, b = stat["v5c"][k], stat[other][k]
        fmt = ".1f" if "points" in k else ".3f"
        print(f"  {k}: {a[0]:{fmt}} vs {b[0]:{fmt}}, difference {a[0] - b[0]:+{fmt}} {ci(a[1] - b[1], fmt)}")
