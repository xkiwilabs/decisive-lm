import json, sys, numpy as np
from scipy.stats import rankdata
P="data/predictions/"
def rho(a,b):
    return np.corrcoef(rankdata(a),rankdata(b))[0,1]
def run(ds, heads, ref):
    rows={}
    for l in open(f"data/processed/{ds}.jsonl"):
        r=json.loads(l)
        if r["split"]=="gold_test": rows[r["id"]]=r
    preds={nm:{} for nm in heads+[ref]}
    for nm in preds:
        f=P+(f"general_{nm}_e4b_r8_seed0__{ds}.jsonl" if nm!="P0" else f"teacher_{ds}__gemma4_e4b.jsonl")
        for l in open(f):
            x=json.loads(l)
            if x["id"] in rows: preds[nm][x["id"]]=x["answers"]
    qids=sorted({q for r in rows.values() for q,t in r["targets"].items() if t and t.get("human")})
    for q in qids:
        ids=[i for i,r in rows.items() if r["targets"].get(q) and r["targets"][q].get("human") and r["targets"][q]["human"]["n_raters"]>=2 and all(q in preds[nm].get(i,{}) for nm in preds)]
        h=[rows[i]["targets"][q]["human"] for i in ids]
        T=np.array([[x["probabilities"][k] for k in sorted(x["probabilities"],key=int)] for x in h]); N=np.array([x["n_raters"] for x in h])
        C=np.rint(T*N[:,None]); agree=(C*(C-1)).sum(1)/(N*(N-1))
        top={nm:np.array([max(preds[nm][i][q]["probabilities"].values()) for i in ids]) for nm in preds}
        rng=np.random.default_rng(0); B=[rng.integers(0,len(ids),len(ids)) for _ in range(1000)]
        for nm in heads:
            d=rho(top[nm],agree)-rho(top[ref],agree)
            bs=[rho(top[nm][b],agree[b])-rho(top[ref][b],agree[b]) for b in B]
            lo,hi=np.percentile(bs,[2.5,97.5])
            print(f"{ds}\t{q}\tn={len(ids)}\t{nm}\trho {rho(top[nm],agree):+.3f}\tP0 {rho(top[ref],agree):+.3f}\tdelta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for ds,heads in [("wiki_toxicity",["v5c","v3d","v3f","v3g"]),("humicroedit",["v5c","v3d","v3g"]),("csc_sarcasm",["v5c","v3d","v3g"]),("politeness",["v5c","v3f","v3g"])]:
    run(ds,heads,"P0")
