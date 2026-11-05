# Image and audio fine-tunes from the released E4B head (v5c), against the same runs from v3d

The examples of guide 9.9 rerun with the arm that starts from the general head changed from v3d (`general_v3d_e4b_r8_seed0`) to v5c (`general_v5c_e4b_r8_seed0`). The owner decided this on 2026-11-04. The recipe and the training rows are the same as before (`data/queues/media_v5c.sh`): rank-8 LoRA at 5e-5 with the scorer at 1e-5, and 300 rows on seeds 0–2 or the whole training split on seed 0. KonIQ also ran at 1,120 image tokens on seeds 0–2. The base arm's runs are the earlier ones, because each seed draws the same 300 rows. v5c was trained at the default 280 image tokens, so the 280-token runs use the same image detail as the head's own training.

Every figure is `scripts/eval_dataset.py` on gold_test, with 95% CIs from 1,000 paired item-level resamples. Full tables:
- `results/media_seeds_gold_test_300_v5c.md` (`scripts/media_seed_report.py --general generalv5c`), against `results/media_seeds_gold_test_300.md` for v3d;
- `results/media_<set>_gold_test_v5c.md`: P0, P0+task, base, v3d and v5c at 300 rows and on the whole split, Δ against P0+task.

## 300 rows: general start minus base start, same seed and rows

| Set (metric) | Seed | v5c − base [95% CI] | v3d − base [95% CI] |
|---|---|---|---|
| CREMA-D, voice emotion (accuracy) | 0 | +0.027 [+0.016, +0.038] | +0.035 [+0.022, +0.049] |
| | 1 | +0.042 [+0.026, +0.057] | +0.034 [+0.019, +0.049] |
| | 2 | +0.021 [+0.007, +0.035] | +0.030 [+0.007, +0.054] |
| THINGS, is it alive? (MAE, lower is better) | 0 | −0.062 [−0.090, −0.037] | −0.055 [−0.078, −0.034] |
| | 1 | +0.001 [−0.015, +0.016] | −0.013 [−0.026, +0.000] |
| | 2 | −0.019 [−0.031, −0.008] | −0.027 [−0.040, −0.016] |
| KonIQ-10k, photo quality (MAE) | 0 | −0.018 [−0.026, −0.011] | −0.008 [−0.014, −0.001] |
| | 1 | +0.008 [+0.001, +0.015] | +0.008 [+0.002, +0.013] |
| | 2 | +0.000 [−0.005, +0.007] | +0.000 [−0.004, +0.006] |

**v5c as a starting point behaves like v3d.**
- It helps voice emotion on every seed (+2 to +4 points of accuracy).
- It helps "is it alive?" on two of three seeds.
- It gives no consistent gain on photo quality: one seed better, one worse, one level.
- No v5c-against-v3d difference is larger than the spread between seeds.

**KonIQ at 1,120 image tokens against 280, the same start and seed** (MAE):
- v5c: −0.012, −0.021 and −0.015, all significant;
- v3d: −0.025, −0.008 and −0.019.

More image detail helps from either head, as it does from the base model.

## The whole training split, seed 0, against P0+task

| Set (metric) | P0+task | base all | v3d all | v5c all |
|---|---|---|---|---|
| CREMA-D (accuracy) | 0.576 | **0.755** | 0.745 | 0.734 |
| CREMA-D ρ(top p, agreement) | 0.013 | **0.511** | 0.488 | 0.475 |
| THINGS (MAE) | 0.346 | 0.186 | 0.168 | **0.162** |
| THINGS ρ(top p, agreement) | 0.438 | 0.612 | **0.669** | 0.665 |
| KonIQ-10k (MAE) | 0.347 | **0.145** | 0.155 | 0.159 |
| KonIQ-10k ρ(top p, agreement) | 0.107 | 0.356 | **0.370** | 0.324 |

- With the whole split, all three starts are far ahead of P0+task on every set, on accuracy or MAE and on review readiness alike.
- **Starting from the general head matters less with more data.** The base start is best on voice emotion and photo quality. The general starts are best on "is it alive?".
- These are single seeds (95% CIs in the per-set files). The base and v3d whole-split runs are the earlier ones.

**For the guide and the paper:** the 9.9 examples' conclusion holds with v5c. A general start helps most with few rows, and on voice emotion most reliably.
