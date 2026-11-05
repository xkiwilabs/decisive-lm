# Δρ(top p, pair agreement), head minus P0, with paired bootstrap CIs: scripts/paper_v5c/drho.py's method (gold_test,
# items with two or more raters, 1,000 item resamples, seed 0) on every Choice and Score question of each set.
import json, sys, numpy as np
from scipy.stats import rankdata
P="data/predictions/"
def rho(a,b):
    return np.corrcoef(rankdata(a),rankdata(b))[0,1]
def run(ds, files, ref):
    rows={}
    for l in open(f"data/processed/{ds}.jsonl"):
        r=json.loads(l)
        if r["split"]=="gold_test": rows[r["id"]]=r
    preds={nm:{} for nm in files}
    for nm,f in files.items():
        for l in open(P+f.format(ds=ds)):
            x=json.loads(l)
            if x["id"] in rows: preds[nm][x["id"]]=x["answers"]
    qids=sorted({q for r in rows.values() for q,t in r["targets"].items() if t and t.get("human") and "probabilities" in t["human"]})
    for q in qids:
        ids=[i for i,r in rows.items() if r["targets"].get(q) and r["targets"][q].get("human") and r["targets"][q]["human"]["n_raters"]>=2 and all(q in preds[nm].get(i,{}) for nm in preds)]
        if not ids: continue
        h=[rows[i]["targets"][q]["human"] for i in ids]
        C=[np.rint(np.array(list(x["probabilities"].values()))*x["n_raters"]) for x in h]   # VizWiz mixes 2 and 4 options
        agree=np.array([(c*(c-1)).sum()/(x["n_raters"]*(x["n_raters"]-1)) for c,x in zip(C,h)])
        top={nm:np.array([max(preds[nm][i][q]["probabilities"].values()) for i in ids]) for nm in preds}
        rng=np.random.default_rng(0); B=[rng.integers(0,len(ids),len(ids)) for _ in range(1000)]
        for nm in files:
            if nm==ref: continue
            d=rho(top[nm],agree)-rho(top[ref],agree)
            bs=[rho(top[nm][b],agree[b])-rho(top[ref][b],agree[b]) for b in B]
            lo,hi=np.percentile(bs,[2.5,97.5])
            print(f"{ds}\t{q}\tn={len(ids)}\t{nm}\trho {rho(top[nm],agree):+.3f}\t{ref} {rho(top[ref],agree):+.3f}\tdelta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
F={"26B head":"general_v5c_26b-a4b_r8_seed0__{ds}.jsonl","E4B head":"general_v5c_e4b_r8_seed0__{ds}.jsonl","P0 26B":"teacher_{ds}__gemma4_26b_a4b.jsonl"}
for ds in sys.argv[1:] or ["wiki_toxicity","humicroedit","csc_sarcasm","politeness","multinli","vizwiz_vqa"]:
    run(ds,F,"P0 26B")
