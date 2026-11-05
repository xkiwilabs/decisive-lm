import json, numpy as np, sys
P="data/predictions/"
rows={}
for l in open("data/processed/wiki_toxicity.jsonl"):
    r=json.loads(l)
    if r["split"]=="gold_test": rows[r["id"]]=r
ids=sorted(rows)
H=np.array([[rows[i]["targets"]["toxicity"]["human"]["probabilities"][str(k)] for k in range(5)] for i in ids])
N=np.array([rows[i]["targets"]["toxicity"]["human"]["n_raters"] for i in ids])
C=np.rint(H*N[:,None])
pair=(C*(C-1)).sum(1)/(N*(N-1))
mx=H.max(1)
strict=(H[:,2]==mx)&((H==mx[:,None]).sum(1)==1)
tied=(H[:,2]==mx)
print("n",len(ids),"strict middle plurality",strict.sum(),f"{strict.mean():.3f}","middle among tied max",tied.sum(),f"{tied.mean():.3f}")
for nm,m in [("strict",strict),("tied",tied)]:
    print(nm,"pair agreement middle %.3f other %.3f"%(pair[m].mean(),pair[~m].mean()))
# other pluralities
am=H.argmax(1)
for k in range(5): print("strict plurality level",k,((H[:,k]==mx)&((H==mx[:,None]).sum(1)==1)).sum(), "pair %.3f"%pair[(H[:,k]==mx)&((H==mx[:,None]).sum(1)==1)].mean() if ((H[:,k]==mx)&((H==mx[:,None]).sum(1)==1)).sum() else "")
def load(f):
    d={}
    for l in open(P+f):
        x=json.loads(l)
        if x["id"] in rows:
            p=x["answers"]["toxicity"]["probabilities"] if "answers" in x else x["probabilities"]
            d[x["id"]]=[p[str(k)] for k in range(5)]
    return np.array([d[i] for i in ids])
files={"26B head":"general_v5c_26b-a4b_r8_seed0__wiki_toxicity.jsonl","v5c s0":"general_v5c_e4b_r8_seed0__wiki_toxicity.jsonl","v5c s1":"general_v5c_e4b_r8_seed1__wiki_toxicity.jsonl","v3d":"general_v3d_e4b_r8_seed0__wiki_toxicity.jsonl","P0":"teacher_wiki_toxicity__gemma4_e4b.jsonl","teacher":"teacher_wiki_toxicity__gemma4_26b_a4b.jsonl"}
mean_h=(H*np.arange(5)).sum(1)
toxic=mean_h<=1.0; neutral=strict
for nm,f in files.items():
    M=load(f); pick=M.argmax(1)==2; top=M.max(1)
    ext=np.abs((M*np.arange(5)).sum(1)-2)
    from scipy.stats import spearmanr
    print(f"{nm:8s} middle pick: strict {pick[strict].mean():.3f} tied {pick[tied].mean():.3f} all {pick.mean():.3f} | top p toxic(mean<=1, n={toxic.sum()}) {top[toxic].mean():.3f} neutral-plurality {top[strict].mean():.3f} | rho(top p, |E[level]-2| model) {spearmanr(top,ext)[0]:.3f} rho(top p, |human mean-2|) {spearmanr(top,np.abs(mean_h-2))[0]:.3f} rho(top p, pair) {spearmanr(top,pair)[0]:.3f}")
print("---- variants")
tox_plur=(H[:,:2].max(1)>=mx)&(H[:,0]+H[:,1]>0)&(np.maximum(H[:,0],H[:,1])==mx)
defs={"plurality toxic/very toxic (ties incl)":tox_plur,"mean<=1.5":mean_h<=1.5,"share toxic+very>=0.5":H[:,0]+H[:,1]>=0.5,"share toxic+very>0.5":H[:,0]+H[:,1]>0.5}
for dn,m in defs.items():
    out=[]
    for nm,f in files.items():
        M=load(f); top=M.max(1); out.append(f"{nm} {top[m].mean():.3f}/{top[tied].mean():.3f}")
    print(f"{dn} n={m.sum()}: "+"; ".join(out))
# whole dataset share
allrows=[json.loads(l) for l in open("data/processed/wiki_toxicity.jsonl")]
HA=np.array([[r["targets"]["toxicity"]["human"]["probabilities"][str(k)] for k in range(5)] for r in allrows])
mA=HA.max(1); print("all 5000 tied middle", (HA[:,2]==mA).mean(), "strict", ((HA[:,2]==mA)&((HA==mA[:,None]).sum(1)==1)).mean())
# pair agreement by other plurality (ties incl), excluding middle-tied
for k in [0,1,3]:
    m=(H[:,k]==mx)&~tied; print("other plurality",k,m.sum(),"%.3f"%pair[m].mean())
print("---- more")
for dn,m in {"very toxic plurality (ties incl)":H[:,0]==mx,"mean<=0.75":mean_h<=0.75,"mean<=0.5":mean_h<=0.5,"share very toxic>=0.5":H[:,0]>=0.5}.items():
    print(dn,m.sum(),"; ".join(f"{nm} {load(f).max(1)[m].mean():.3f}" for nm,f in files.items()))
print("---- extremes")
print("raters' vote share: very toxic %.4f very healthy %.4f"%(C[:,0].sum()/C.sum(),C[:,4].sum()/C.sum()), " mean of item shares %.4f %.4f"%(H[:,0].mean(),H[:,4].mean()))
for nm,f in files.items():
    M=load(f); top=M.max(1); ext=M[:,0]+M[:,4]; am=M.argmax(1); e=(am==0)|(am==4)
    print(f"{nm:8s} P(very toxic) {M[:,0].mean():.3f} P(very healthy) {M[:,4].mean():.3f} sum {ext.mean():.3f} | argmax extreme on {e.mean():.3f} of comments, top p there {top[e].mean() if e.any() else float('nan'):.3f} vs elsewhere {top[~e].mean():.3f} | rho(top p, P0+P4) {spearmanr(top,ext)[0]:+.3f}")
