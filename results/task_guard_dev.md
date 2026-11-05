# The per-task step's guard on the development sets: not adopted

The check docs/RELEASE.md (change log, 2026-11-05) set before the run: the guard (`scripts/score_suite.py
--task-guard`) is adopted only if, on the development sets, for both v5c heads and both P0s, it raises no task's Brier
after the per-task step by more than 0.002 and no task's ECE after the step by more than 0.01, compared with the step
without it.

**Result: it fails.** The largest rises are ECE +0.072 and Brier +0.019 (limits 0.01 and 0.002). 21 values exceed a limit,
in 9 distinct model–task pairs (the prompted 26B appears in both reports). The worst cases are the 26B head on MultiNLI's Choice question (ECE 0.036 → 0.108) and both
prompted models on DailyDialog (P0 26B ECE 0.097 → 0.163). It also helps in places, most of all where calibration was
far off: the prompted 26B on VizWiz-VQA's Noul (ECE 0.630 → 0.140) and the prompted E4B on MultiNLI's Noul
(0.243 → 0.088). On Wikipedia toxicity, the development stand-in for the failure it was meant to fix, it made the
E4B head's Brier worse (0.670 → 0.674), not better.

**Why, as far as this shows:** 100 rows are too few to choose between three forms whose Brier differs by about 0.01.
The choice follows noise in those rows, and a wrong choice costs more than the step's small losses it was meant to
avoid. The per-task step stays as written. The remedy for users with a few hundred labelled items stays task
calibration (guide 10.6).

**How it was run** (2026-11-05, CPU): `scripts/score_suite.py --candidate <head> --calib-data
data/mixes/general_v5c.jsonl --sets multinli wiki_toxicity humicroedit csc_sarcasm vizwiz_vqa bench_agnews
bench_banking77 bench_daily_dialog bench_emotion bench_fin_topic bench_sst2 bench_tweet_topic --n-boot 100`, with and
without `--task-guard`, for `general_v5c_e4b_r8_seed0` and `general_v5c_26b-a4b_r8_seed0`. Values are point values on
the rows the step does not use. For the 26B head, P0 and the teacher are the same model, so the 26B report's P0 rows
equal the E4B report's teacher rows.

## Forms the guard kept, and every value it changed

```
general_v5c_e4b_r8_seed0 multinli choice: candidate choice scaled, score scaled, noul calibrated; P0 choice scaled, score scaled, noul raw; teacher choice scaled, score scaled, noul raw
general_v5c_e4b_r8_seed0 multinli score: candidate choice scaled, score scaled, noul calibrated; P0 choice scaled, score scaled, noul raw; teacher choice scaled, score scaled, noul raw
general_v5c_e4b_r8_seed0 multinli noul: candidate choice scaled, score scaled, noul calibrated; P0 choice scaled, score scaled, noul raw; teacher choice scaled, score scaled, noul raw
general_v5c_e4b_r8_seed0 wiki_toxicity score: candidate score raw; P0 score calibrated; teacher score scaled
general_v5c_e4b_r8_seed0 humicroedit score: candidate score scaled; P0 score scaled; teacher score calibrated
general_v5c_e4b_r8_seed0 csc_sarcasm score: candidate score scaled; P0 score scaled; teacher score scaled
general_v5c_e4b_r8_seed0 vizwiz_vqa choice: candidate choice scaled, noul raw; P0 choice calibrated, noul raw; teacher choice scaled, noul raw
general_v5c_e4b_r8_seed0 vizwiz_vqa noul: candidate choice scaled, noul raw; P0 choice calibrated, noul raw; teacher choice scaled, noul raw
general_v5c_e4b_r8_seed0 bench_agnews choice: candidate choice scaled; P0 choice scaled; teacher choice scaled
general_v5c_e4b_r8_seed0 bench_banking77 choice: candidate choice scaled; P0 choice calibrated; teacher choice scaled
general_v5c_e4b_r8_seed0 bench_daily_dialog choice: candidate choice scaled; P0 choice calibrated; teacher choice raw
general_v5c_e4b_r8_seed0 bench_emotion choice: candidate choice scaled; P0 choice scaled; teacher choice calibrated
general_v5c_e4b_r8_seed0 bench_fin_topic choice: candidate choice scaled; P0 choice scaled; teacher choice scaled
general_v5c_e4b_r8_seed0 bench_sst2 choice: candidate choice scaled; P0 choice calibrated; teacher choice calibrated
general_v5c_e4b_r8_seed0 bench_tweet_topic choice: candidate choice scaled; P0 choice calibrated; teacher choice calibrated
general_v5c_26b-a4b_r8_seed0 multinli choice: candidate choice calibrated, score calibrated, noul calibrated; P0 choice scaled, score scaled, noul raw; teacher choice scaled, score scaled, noul raw
general_v5c_26b-a4b_r8_seed0 multinli score: candidate choice calibrated, score calibrated, noul calibrated; P0 choice scaled, score scaled, noul raw; teacher choice scaled, score scaled, noul raw
general_v5c_26b-a4b_r8_seed0 multinli noul: candidate choice calibrated, score calibrated, noul calibrated; P0 choice scaled, score scaled, noul raw; teacher choice scaled, score scaled, noul raw
general_v5c_26b-a4b_r8_seed0 wiki_toxicity score: candidate score scaled; P0 score scaled; teacher score scaled
general_v5c_26b-a4b_r8_seed0 humicroedit score: candidate score scaled; P0 score calibrated; teacher score calibrated
general_v5c_26b-a4b_r8_seed0 csc_sarcasm score: candidate score raw; P0 score scaled; teacher score scaled
general_v5c_26b-a4b_r8_seed0 vizwiz_vqa choice: candidate choice scaled, noul calibrated; P0 choice scaled, noul raw; teacher choice scaled, noul raw
general_v5c_26b-a4b_r8_seed0 vizwiz_vqa noul: candidate choice scaled, noul calibrated; P0 choice scaled, noul raw; teacher choice scaled, noul raw
general_v5c_26b-a4b_r8_seed0 bench_agnews choice: candidate choice scaled; P0 choice scaled; teacher choice scaled
general_v5c_26b-a4b_r8_seed0 bench_banking77 choice: candidate choice scaled; P0 choice scaled; teacher choice scaled
general_v5c_26b-a4b_r8_seed0 bench_daily_dialog choice: candidate choice scaled; P0 choice raw; teacher choice raw
general_v5c_26b-a4b_r8_seed0 bench_emotion choice: candidate choice scaled; P0 choice calibrated; teacher choice calibrated
general_v5c_26b-a4b_r8_seed0 bench_fin_topic choice: candidate choice scaled; P0 choice scaled; teacher choice scaled
general_v5c_26b-a4b_r8_seed0 bench_sst2 choice: candidate choice scaled; P0 choice calibrated; teacher choice calibrated
general_v5c_26b-a4b_r8_seed0 bench_tweet_topic choice: candidate choice scaled; P0 choice calibrated; teacher choice calibrated
```

| Method | Task | Measure | Without the guard | With it | Δ |
|---|---|---|---|---|---|
| P0 E4B | bench_banking77 choice | Brier | 0.4130 | 0.4110 | -0.0020 |
| P0 E4B | bench_banking77 choice | ECE | 0.0650 | 0.0670 | +0.0020 |
| P0 E4B | bench_daily_dialog choice | Brier | 0.4640 | 0.4770 | +0.0130 |
| teacher 26B | bench_daily_dialog choice | Brier | 0.4860 | 0.5050 | +0.0190 |
| P0 E4B | bench_daily_dialog choice | ECE | 0.0770 | 0.1250 | +0.0480 |
| teacher 26B | bench_daily_dialog choice | ECE | 0.0970 | 0.1630 | +0.0660 |
| teacher 26B | bench_emotion choice | Brier | 0.6010 | 0.5890 | -0.0120 |
| teacher 26B | bench_emotion choice | ECE | 0.1550 | 0.1350 | -0.0200 |
| P0 E4B | bench_sst2 choice | Brier | 0.1040 | 0.1020 | -0.0020 |
| P0 E4B | bench_sst2 choice | ECE | 0.0360 | 0.0240 | -0.0120 |
| teacher 26B | bench_sst2 choice | ECE | 0.0160 | 0.0170 | +0.0010 |
| P0 E4B | bench_tweet_topic choice | Brier | 0.3230 | 0.3220 | -0.0010 |
| teacher 26B | bench_tweet_topic choice | Brier | 0.2680 | 0.2710 | +0.0030 |
| P0 E4B | bench_tweet_topic choice | ECE | 0.0430 | 0.0310 | -0.0120 |
| teacher 26B | bench_tweet_topic choice | ECE | 0.0680 | 0.0660 | -0.0020 |
| teacher 26B | humicroedit score | Brier | 0.7320 | 0.7350 | +0.0030 |
| teacher 26B | humicroedit score | ECE | 0.0540 | 0.0640 | +0.0100 |
| P0 E4B | multinli noul | Brier | 0.2220 | 0.1100 | -0.1120 |
| teacher 26B | multinli noul | Brier | 0.1320 | 0.1010 | -0.0310 |
| P0 E4B | multinli noul | ECE | 0.2430 | 0.0880 | -0.1550 |
| teacher 26B | multinli noul | ECE | 0.0580 | 0.0780 | +0.0200 |
| P0 E4B | vizwiz_vqa choice | Brier | 0.6420 | 0.6490 | +0.0070 |
| P0 E4B | vizwiz_vqa choice | ECE | 0.0380 | 0.0520 | +0.0140 |
| E4B head | vizwiz_vqa noul | Brier | 0.1030 | 0.1000 | -0.0030 |
| P0 E4B | vizwiz_vqa noul | Brier | 0.6340 | 0.6070 | -0.0270 |
| teacher 26B | vizwiz_vqa noul | Brier | 0.5940 | 0.2230 | -0.3710 |
| E4B head | vizwiz_vqa noul | ECE | 0.0850 | 0.0710 | -0.0140 |
| P0 E4B | vizwiz_vqa noul | ECE | 0.6640 | 0.6270 | -0.0370 |
| teacher 26B | vizwiz_vqa noul | ECE | 0.6300 | 0.1400 | -0.4900 |
| E4B head | wiki_toxicity score | Brier | 0.6700 | 0.6740 | +0.0040 |
| P0 E4B | wiki_toxicity score | Brier | 0.7500 | 0.7470 | -0.0030 |
| E4B head | wiki_toxicity score | ECE | 0.0680 | 0.0750 | +0.0070 |
| P0 E4B | wiki_toxicity score | ECE | 0.0900 | 0.0880 | -0.0020 |
| P0 26B | bench_daily_dialog choice | Brier | 0.4860 | 0.5050 | +0.0190 |
| P0 26B | bench_daily_dialog choice | ECE | 0.0970 | 0.1630 | +0.0660 |
| P0 26B | bench_emotion choice | Brier | 0.6010 | 0.5890 | -0.0120 |
| P0 26B | bench_emotion choice | ECE | 0.1550 | 0.1350 | -0.0200 |
| P0 26B | bench_sst2 choice | ECE | 0.0160 | 0.0170 | +0.0010 |
| P0 26B | bench_tweet_topic choice | Brier | 0.2680 | 0.2710 | +0.0030 |
| P0 26B | bench_tweet_topic choice | ECE | 0.0680 | 0.0660 | -0.0020 |
| 26B head | csc_sarcasm score | ECE | 0.0440 | 0.0270 | -0.0170 |
| P0 26B | humicroedit score | Brier | 0.7320 | 0.7350 | +0.0030 |
| P0 26B | humicroedit score | ECE | 0.0540 | 0.0640 | +0.0100 |
| 26B head | multinli choice | Brier | 0.2290 | 0.2460 | +0.0170 |
| 26B head | multinli choice | ECE | 0.0360 | 0.1080 | +0.0720 |
| P0 26B | multinli noul | Brier | 0.1320 | 0.1010 | -0.0310 |
| P0 26B | multinli noul | ECE | 0.0580 | 0.0780 | +0.0200 |
| 26B head | multinli score | Brier | 0.2370 | 0.2400 | +0.0030 |
| 26B head | multinli score | ECE | 0.0340 | 0.0550 | +0.0210 |
| P0 26B | vizwiz_vqa noul | Brier | 0.5940 | 0.2230 | -0.3710 |
| P0 26B | vizwiz_vqa noul | ECE | 0.6300 | 0.1400 | -0.4900 |

largest rise: ECE +0.0720 (limit 0.01), Brier +0.0190 (limit 0.002)
FAILS:
  P0 E4B bench_daily_dialog choice Brier 0.4640 -> 0.4770
  teacher 26B bench_daily_dialog choice Brier 0.4860 -> 0.5050
  P0 E4B bench_daily_dialog choice ECE 0.0770 -> 0.1250
  teacher 26B bench_daily_dialog choice ECE 0.0970 -> 0.1630
  teacher 26B bench_tweet_topic choice Brier 0.2680 -> 0.2710
  teacher 26B humicroedit score Brier 0.7320 -> 0.7350
  teacher 26B humicroedit score ECE 0.0540 -> 0.0640
  teacher 26B multinli noul ECE 0.0580 -> 0.0780
  P0 E4B vizwiz_vqa choice Brier 0.6420 -> 0.6490
  P0 E4B vizwiz_vqa choice ECE 0.0380 -> 0.0520
  E4B head wiki_toxicity score Brier 0.6700 -> 0.6740
  P0 26B bench_daily_dialog choice Brier 0.4860 -> 0.5050
  P0 26B bench_daily_dialog choice ECE 0.0970 -> 0.1630
  P0 26B bench_tweet_topic choice Brier 0.2680 -> 0.2710
  P0 26B humicroedit score Brier 0.7320 -> 0.7350
  P0 26B humicroedit score ECE 0.0540 -> 0.0640
  26B head multinli choice Brier 0.2290 -> 0.2460
  26B head multinli choice ECE 0.0360 -> 0.1080
  P0 26B multinli noul ECE 0.0580 -> 0.0780
  26B head multinli score Brier 0.2370 -> 0.2400
  26B head multinli score ECE 0.0340 -> 0.0550
