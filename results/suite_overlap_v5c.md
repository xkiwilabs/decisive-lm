# Overlap between the frozen suite and trainable data

**Rerun 2026-11-02 before scoring the v5c heads on the suite**, with three trainable files added: synth_v4, vizwiz_vqa and
helpsteer2_preference. It runs on the suite as converted, so the 641 rows already left out (`configs/suite_exclusions.txt`,
from `results/suite_overlap.md`) are gone. Nonzero counts in the last column with no suite row to leave out are training
texts whose 8-grams are spread over several suite rows, none holding half of them; the original audit used the same rule.
- **v5c:** no suite row overlaps anything v5c trains on. The exclusions are unchanged for its scoring.
- **One new overlap, outside v5c:** HelpSteer3 `helpsteer3-train-34503` overlaps HelpSteer2-Preference, which is deferred to
  0.2 under the suite-task rule. It was added to `configs/suite_exclusions.txt` on 2026-11-05 (the owner's decision), after both v5c heads were scored.

Word 8-grams of states; a row overlaps when at least 50% of its 8-grams occur in the other side (texts under 8 words: exact match). Written by `scripts/overlap_audit.py`.

| Training file | Suite set | training rows | suite rows | suite rows found in training | training rows found in suite |
|---|---|---|---|---|---|
| banking77 | helpsteer3 | 11,264 | 5,966 | 0 | 0 |
| banking77 | measuring_hate_speech | 11,264 | 3,384 | 0 | 0 |
| banking77 | cuad | 11,264 | 2,553 | 0 | 0 |
| banking77 | massive | 11,264 | 29,739 | 0 | 0 |
| banking77_explode | helpsteer3 | 54,510 | 5,966 | 0 | 0 |
| banking77_explode | measuring_hate_speech | 54,510 | 3,384 | 0 | 0 |
| banking77_explode | cuad | 54,510 | 2,553 | 0 | 0 |
| banking77_explode | massive | 54,510 | 29,739 | 0 | 0 |
| clinc150 | helpsteer3 | 20,505 | 5,966 | 0 | 0 |
| clinc150 | measuring_hate_speech | 20,505 | 3,384 | 0 | 0 |
| clinc150 | cuad | 20,505 | 2,553 | 0 | 0 |
| clinc150 | massive | 20,505 | 29,739 | 0 | 0 |
| clinc150_explode | helpsteer3 | 98,553 | 5,966 | 0 | 0 |
| clinc150_explode | measuring_hate_speech | 98,553 | 3,384 | 0 | 0 |
| clinc150_explode | cuad | 98,553 | 2,553 | 0 | 0 |
| clinc150_explode | massive | 98,553 | 29,739 | 0 | 0 |
| helpsteer2 | helpsteer3 | 20,704 | 5,966 | 0 | 2 |
| helpsteer2 | measuring_hate_speech | 20,704 | 3,384 | 0 | 0 |
| helpsteer2 | cuad | 20,704 | 2,553 | 0 | 0 |
| helpsteer2 | massive | 20,704 | 29,739 | 0 | 0 |
| civil_comments | helpsteer3 | 59,026 | 5,966 | 0 | 0 |
| civil_comments | measuring_hate_speech | 59,026 | 3,384 | 0 | 0 |
| civil_comments | cuad | 59,026 | 2,553 | 0 | 0 |
| civil_comments | massive | 59,026 | 29,739 | 0 | 0 |
| civil_comments_human_matched | helpsteer3 | 24,026 | 5,966 | 0 | 0 |
| civil_comments_human_matched | measuring_hate_speech | 24,026 | 3,384 | 0 | 0 |
| civil_comments_human_matched | cuad | 24,026 | 2,553 | 0 | 0 |
| civil_comments_human_matched | massive | 24,026 | 29,739 | 0 | 0 |
| goemotions | helpsteer3 | 52,584 | 5,966 | 0 | 0 |
| goemotions | measuring_hate_speech | 52,584 | 3,384 | 0 | 0 |
| goemotions | cuad | 52,584 | 2,553 | 0 | 0 |
| goemotions | massive | 52,584 | 29,739 | 0 | 0 |
| ledgar | helpsteer3 | 22,619 | 5,966 | 0 | 0 |
| ledgar | measuring_hate_speech | 22,619 | 3,384 | 0 | 0 |
| ledgar | cuad | 22,619 | 2,553 | 0 | 275 |
| ledgar | massive | 22,619 | 29,739 | 0 | 0 |
| synth_v1 | helpsteer3 | 39,563 | 5,966 | 0 | 0 |
| synth_v1 | measuring_hate_speech | 39,563 | 3,384 | 0 | 0 |
| synth_v1 | cuad | 39,563 | 2,553 | 0 | 1 |
| synth_v1 | massive | 39,563 | 29,739 | 0 | 0 |
| synth_v2 | helpsteer3 | 40,962 | 5,966 | 0 | 0 |
| synth_v2 | measuring_hate_speech | 40,962 | 3,384 | 0 | 0 |
| synth_v2 | cuad | 40,962 | 2,553 | 0 | 24 |
| synth_v2 | massive | 40,962 | 29,739 | 0 | 0 |
| synth_v3 | helpsteer3 | 103,263 | 5,966 | 0 | 0 |
| synth_v3 | measuring_hate_speech | 103,263 | 3,384 | 0 | 0 |
| synth_v3 | cuad | 103,263 | 2,553 | 0 | 0 |
| synth_v3 | massive | 103,263 | 29,739 | 0 | 0 |
| measuring_hate_speech | helpsteer3 | 36,157 | 5,966 | 0 | 0 |
| measuring_hate_speech | measuring_hate_speech | 36,157 | 3,384 | 0 | 0 |
| measuring_hate_speech | cuad | 36,157 | 2,553 | 0 | 0 |
| measuring_hate_speech | massive | 36,157 | 29,739 | 0 | 0 |
| multinli | helpsteer3 | 328,206 | 5,966 | 0 | 0 |
| multinli | measuring_hate_speech | 328,206 | 3,384 | 0 | 0 |
| multinli | cuad | 328,206 | 2,553 | 0 | 0 |
| multinli | massive | 328,206 | 29,739 | 0 | 0 |
| synth_v4 | helpsteer3 | 17,857 | 5,966 | 0 | 0 |
| synth_v4 | measuring_hate_speech | 17,857 | 3,384 | 0 | 0 |
| synth_v4 | cuad | 17,857 | 2,553 | 0 | 0 |
| synth_v4 | massive | 17,857 | 29,739 | 0 | 0 |
| vizwiz_vqa | helpsteer3 | 22,275 | 5,966 | 0 | 0 |
| vizwiz_vqa | measuring_hate_speech | 22,275 | 3,384 | 0 | 0 |
| vizwiz_vqa | cuad | 22,275 | 2,553 | 0 | 0 |
| vizwiz_vqa | massive | 22,275 | 29,739 | 0 | 0 |
| helpsteer2_preference | helpsteer3 | 17,670 | 5,966 | 1 | 1 |
| helpsteer2_preference | measuring_hate_speech | 17,670 | 3,384 | 0 | 0 |
| helpsteer2_preference | cuad | 17,670 | 2,553 | 0 | 0 |
| helpsteer2_preference | massive | 17,670 | 29,739 | 0 | 0 |

Suite rows to leave out: helpsteer3 1, measuring_hate_speech 0, cuad 0, massive 0
