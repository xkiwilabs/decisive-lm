import json, numpy as np, collections
P="data/predictions/"
sets=["bench_agnews","bench_sst2","bench_banking77","bench_emotion","bench_tweet_topic","bench_fin_topic","bench_daily_dialog","helpsteer3","measuring_hate_speech","massive","cuad"]
for ds in sets:
    rows={}
    for l in open(f"data/processed/{ds}.jsonl"):
        r=json.loads(l)
        if r["split"]=="gold_test": rows[r["id"]]=r
    acc=collections.defaultdict(list)
    for l in open(P+f"general_v5c_e4b_r8_seed0__{ds}.jsonl"):
        x=json.loads(l)
        r=rows.get(x["id"])
        if not r: continue
        for q,a in x["answers"].items():
            t=r["targets"].get(q); 
            if not t: continue
            h=t.get("human") or t.get("teacher")
            if not t.get("human"): continue
            tp=h["probabilities"] if "probabilities" in h else None
            if a["type"]=="noul":
                p=a["p"]; y=h["noul"]
                if p is None or y is None or y==0.5: continue
                acc["noul"].append((max(p,1-p), float((p>=.5)==(y>.5))))
            else:
                ks=sorted(tp); tv=np.array([tp[k] for k in ks]); pv=np.array([a["probabilities"][k] for k in ks])
                s=np.sort(tv)
                if s[-1]<=s[-2]+1e-9: continue
                acc[a["type"]].append((pv.max(), float(pv.argmax()==tv.argmax())))
    for ty,v in acc.items():
        v=np.array(v); c,ac=v[:,0].mean(),v[:,1].mean()
        # binned ECE sign: weighted mean of (conf-acc) per bin
        print(f"{ds:22s} {ty:6s} n={len(v):6d} conf {c:.3f} acc {ac:.3f} conf-acc {c-ac:+.3f} {'over' if c>ac else 'under'}")
