# How far answers move when only the packing changes (docs/guide/11 §11.7)

`scripts/check_invariance.py` on 100 gold_test Choice questions of the coding data (sample seed 0). Each variant changes only how the request is packed:
- **reversed_options**: the Choice options in reverse order.
- **after_another_request**: another request packed ahead of this one in the same pass.
- **extra_question**: one more question in the request.
- **repeat**: the identical pass again.

The masks make every answer independent of these changes by construction. What remains is numerical.

## E4B, P1 seed 0 bundle, Apple M5 Max (MPS), 2026-10-06

| dtype | Variant | max \|Δp\| | mean | p99 | answer flips | top-p rank ρ | review-set churn (20%) |
|---|---|---|---|---|---|---|---|
| bf16 | repeat | 0 | 0 | 0 | 0 / 100 | 1.0000 | 0% |
| bf16 | reversed_options | .181 | .029 | .163 | 3 / 100 | .9889 | 5% |
| bf16 | after_another_request | .394 | .035 | .187 | 4 / 100 | .9847 | 10% |
| bf16 | extra_question | .068 | .003 | .055 | 0 / 100 | .9996 | 0% |
| fp32 | repeat | 0 | 0 | 0 | 0 / 100 | 1.0000 | 0% |
| fp32 | reversed_options | 6.4e-5 | 8.9e-6 | 5.8e-5 | 0 / 100 | 1.0000 | 0% |
| fp32 | after_another_request | 1.2e-4 | 9.3e-6 | 7.2e-5 | 0 / 100 | 1.0000 | 0% |
| fp32 | extra_question | 2.4e-5 | 4.9e-7 | 6.3e-6 | 0 / 100 | 1.0000 | 0% |

**On a dense backbone, the packing noise is entirely bf16 rounding.**
- **bf16:** different kernel shapes and summation orders leave 3–6% relative error on hidden states over 42 layers (results/M0.md). The trained head turns that into probability shifts of a few points, up to .39. 3–4 answers in 100 flip, and 5–10% of the least-confident fifth (the items a human would review) changes.
- **fp32:** every shift is ≤ 1.2e-4, and nothing flips or churns.
- **Same packing:** an identical request reproduces exactly in either dtype.
- **Cost:** the fp32 run took 16 min on MPS, against about 4 for bf16.

**Implications:**
- **Independence and order invariance hold in fp32.** In bf16 they hold only up to these shifts.
- **The P1 heads were trained on bf16 features,** so each item carried a fixed perturbation of this size. On E4B that costs nothing measurable (next section).
- **26B A4B adds routing noise on top** (docs/guide/06 §6.9), and in bf16 it is about five times larger (below).

## E4B: P1 on fp32 features vs bf16, RTX Pro 6000, 2026-10-06

Same coding data and P1 recipe, gold_test, seeds 0–2. Features cost 0.34 s/row in fp32 against 0.16 in bf16.

| Features | κ | soft NLL | ECE | ρ (top p) | tie AUROC | rest accuracy, 20% reviewed |
|---|---|---|---|---|---|---|
| bf16 | .756–.767 | .292–.301 | .054–.066 | .532–.552 | .787–.808 | 95.3–96.2% |
| fp32 | .767–.772 | .294–.301 | .065–.072 | .541–.545 | .790–.795 | 95.5–95.7% |

Paired differences (fp32 seed 0 minus each bf16 seed) are within ±.011 for κ and ρ, ±.015 for tie AUROC, ±0.5 points for rest accuracy, and every CI spans 0. Temperatures are 1.01–1.04 in fp32 against 1.02–1.10 in bf16, and val soft NLL is .226–.228 against .227–.232. **E4B stays bf16:** the ~4% of answers that flip under repacking do not cost P1 anything measurable.

## 26B A4B, RTX Pro 6000, 2026-10-06

**P1 seed 0 bundle, bf16, 200 gold_test questions:** 38–44 answers in 200 flip (19–22%) when only the packing changes. Top-p rank ρ is .47–.70, and 45–65% of the least-confident fifth changes. The E4B P1 bundle on the same GPU flips 6–9 in 200, with 7.5–10% churn. The 26B head fits training items about as well as E4B (soft NLL .21–.31 vs .24) but gold_test far worse (.45 vs .30): it learns each item's fixed noise. With fp32 routers the same head flips 44, 47 and 44 in 200 (bf16 routers: 38, 44, 40), with 37.5–62.5% churn: no better.

**Where the noise comes from** (a one-off diagnostic script, not kept in the repo; 20 gold_test rows, each alone vs packed after another request):

| Compute | Tokens | Exact 8th = 9th tie, share of (token, layer) | Expert choice differs, share of (token, layer) | Final-state relative error, median / p90 |
|---|---|---|---|---|
| bf16 | leaf text | 11.3% | 33.6% | .154 / .367 |
| bf16 | query | 9.3% | 45.9% | .283 / .432 |
| bf16, fp32 routers | leaf text | 0% | 32.9% | .155 / .347 |
| bf16, fp32 routers | query | 0% | 44.0% | .266 / .508 |
| fp32 (`fp32_compute`) | leaf text | 0% | 0.20% | .000 / .002 |
| fp32 (`fp32_compute`) | query | 0% | 0.33% | .000 / .002 |

- **bf16 rounding of the router scores leaves ~10% exact ties.** fp32 routers remove every one, yet the expert choices still differ about as often.
- **The router's input already carries bf16 noise,** and the 8th/9th margin is tiny (median .0015 in log-probability, p10 .0002). That noise picks the winner.
- **Flips grow with depth:** 1% of leaf-text tokens in the first layers, 56% in the last.
- **The query token is the noisiest place to read:** 66–93% of its expert choices differ in the last four layers (leaf text: about 50–56%), and its final state is about twice as noisy as leaf text.
- **Packing noise also moves 26B's validation loss:** in a P3 probe, the same 16 validation rows with the same head scored .53 at `--max-tokens 4096` and .67 at 2048.

**fp32 compute removes it.** `--dtype float32` keeps the expert weights in bf16 (the checkpoint's own dtype, so the upcast is exact) and runs every matmul in fp32 (`model.fp32_compute`; TF32 off).
- **Expert choices:** they differ in 0.1–0.3% of (token, layer) pairs, against 34–46% in bf16.
- **Final states:** they differ by about .002 at p90 for both leaf text and query tokens, so the query token is no longer an outlier.
- **The margins stay as thin as in bf16** (8th/9th median .0011–.0013). The gain comes entirely from clean router inputs.
- **What flips remain** are in layers 15 and up, at ≤ 1% per layer.
- **Memory:** 53.6 GiB of weights on the GPU, 55.7 GiB at peak in this diagnostic.
- **The bf16-trained P1 seed 0 head, run on an fp32 backbone,** flips 0 of 200 answers under every packing change (bf16: 38–44). Top-p rank ρ is .998–.9996 and review churn 2.5%, so 26B in fp32 is more packing-stable than E4B in bf16. One item still moves by .24, probably sitting on one of the remaining expert flips.
- **Cost:** feature caching runs at 0.51 s/row against 0.16 in bf16 (same 4320 rows and code), about 3.2×.

**But noise was not what held 26B's P1 back.** P1 on fp32 26B features, seeds 0–2, on gold_test:
- **κ and accuracy improve a little:** κ .678–.692 against .657–.669 in bf16, and accuracy about 1 point higher.
- **Review readiness does not improve:**
  - ρ (top p) is .23–.30, against .30–.35 in bf16;
  - tie AUROC is about .61, against .60–.68;
  - dynamic range is .19–.22, against .26–.30.
- **The gap between training and held-out items stays:** train soft NLL .22–.28 against gold_test .41–.42. E4B goes from .24 to .30.

Prompting 26B (P0) reaches κ .731 and ρ .437, so the backbone knows the task. The loss is in reading a frozen last-layer query state, which fits the 36 training participants and does not carry over to new ones. The packing noise was real and is gone, so 26B evaluations can now be trusted.

**fp32 routers alone, dropped:** a P1 head on bf16 features with fp32 routers (seeds 0–2) reaches gold_test soft NLL .387–.396, against .420–.445 with bf16 routers. That is still far from E4B's .29–.30.
