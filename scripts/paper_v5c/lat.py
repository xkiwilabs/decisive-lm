import sys, time, statistics, numpy as np, torch
sys.path.insert(0, "scripts")
from m0_real import text_ids
from decisivelm.bundle import load_bundle
from decisivelm.tree import TreeRequest, Question
from decisivelm.infer import QUERY_IDS
for b in sys.argv[1:]:
    dlm = load_bundle(b)
    tok = dlm.tokenizer
    req = TreeRequest(prefix=text_ids(tok, 1000, 8), questions=[
        Question(segment=text_ids(tok, 15, 30 + j), leaves=[text_ids(tok, 8, 40 + 5 * j + i) + [QUERY_IDS["choice"]] for i in range(5)])
        for j in range(4)])
    times = []
    with torch.no_grad():
        for i in range(25):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            dlm.logits([req])
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    times = times[5:]
    n = len(req.prefix) + sum(len(q.segment) + sum(map(len, q.leaves)) for q in req.questions)
    print(f"{b}: {n} tokens, 20 blocks: p50 {statistics.median(times)*1000:.0f} ms, p95 {np.percentile(times,95)*1000:.0f} ms (mask build, LoRA and scorer included)", flush=True)
    del dlm; torch.cuda.empty_cache()
