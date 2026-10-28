# Overlap between the frozen suite and trainable data

Word 8-grams of states; a row overlaps when at least 50% of its 8-grams occur in the other side (texts under 8 words: exact match). Written by `scripts/overlap_audit.py`.

| Training file | Suite set | training rows | suite rows | suite rows found in training | training rows found in suite |
|---|---|---|---|---|---|
| banking77 | helpsteer3 | 11,264 | 6,000 | 0 | 0 |
| banking77 | measuring_hate_speech | 11,264 | 3,408 | 0 | 0 |
| banking77 | cuad | 11,264 | 3,135 | 0 | 0 |
| banking77 | massive | 11,264 | 29,740 | 0 | 0 |
| banking77_explode | helpsteer3 | 54,510 | 6,000 | 0 | 0 |
| banking77_explode | measuring_hate_speech | 54,510 | 3,408 | 0 | 0 |
| banking77_explode | cuad | 54,510 | 3,135 | 0 | 0 |
| banking77_explode | massive | 54,510 | 29,740 | 0 | 0 |
| clinc150 | helpsteer3 | 20,505 | 6,000 | 0 | 2 |
| clinc150 | measuring_hate_speech | 20,505 | 3,408 | 0 | 0 |
| clinc150 | cuad | 20,505 | 3,135 | 0 | 0 |
| clinc150 | massive | 20,505 | 29,740 | 0 | 0 |
| clinc150_explode | helpsteer3 | 98,553 | 6,000 | 0 | 9 |
| clinc150_explode | measuring_hate_speech | 98,553 | 3,408 | 0 | 0 |
| clinc150_explode | cuad | 98,553 | 3,135 | 0 | 0 |
| clinc150_explode | massive | 98,553 | 29,740 | 0 | 0 |
| helpsteer2 | helpsteer3 | 20,704 | 6,000 | 11 | 31 |
| helpsteer2 | measuring_hate_speech | 20,704 | 3,408 | 0 | 0 |
| helpsteer2 | cuad | 20,704 | 3,135 | 0 | 0 |
| helpsteer2 | massive | 20,704 | 29,740 | 0 | 0 |
| civil_comments | helpsteer3 | 59,026 | 6,000 | 0 | 2 |
| civil_comments | measuring_hate_speech | 59,026 | 3,408 | 0 | 0 |
| civil_comments | cuad | 59,026 | 3,135 | 0 | 0 |
| civil_comments | massive | 59,026 | 29,740 | 1 | 4 |
| civil_comments_human_matched | helpsteer3 | 24,026 | 6,000 | 0 | 0 |
| civil_comments_human_matched | measuring_hate_speech | 24,026 | 3,408 | 0 | 0 |
| civil_comments_human_matched | cuad | 24,026 | 3,135 | 0 | 0 |
| civil_comments_human_matched | massive | 24,026 | 29,740 | 1 | 3 |
| goemotions | helpsteer3 | 52,584 | 6,000 | 0 | 2 |
| goemotions | measuring_hate_speech | 52,584 | 3,408 | 1 | 2 |
| goemotions | cuad | 52,584 | 3,135 | 0 | 0 |
| goemotions | massive | 52,584 | 29,740 | 1 | 5 |
| ledgar | helpsteer3 | 22,619 | 6,000 | 0 | 25 |
| ledgar | measuring_hate_speech | 22,619 | 3,408 | 0 | 0 |
| ledgar | cuad | 22,619 | 3,135 | 3 | 1,699 |
| ledgar | massive | 22,619 | 29,740 | 0 | 0 |
| synth_v1 | helpsteer3 | 39,563 | 6,000 | 2 | 4 |
| synth_v1 | measuring_hate_speech | 39,563 | 3,408 | 0 | 0 |
| synth_v1 | cuad | 39,563 | 3,135 | 0 | 6 |
| synth_v1 | massive | 39,563 | 29,740 | 1 | 1 |
| synth_v2 | helpsteer3 | 40,962 | 6,000 | 0 | 4 |
| synth_v2 | measuring_hate_speech | 40,962 | 3,408 | 0 | 0 |
| synth_v2 | cuad | 40,962 | 3,135 | 0 | 138 |
| synth_v2 | massive | 40,962 | 29,740 | 0 | 0 |
| synth_v3 | helpsteer3 | 74,328 | 6,000 | 0 | 0 |
| synth_v3 | measuring_hate_speech | 74,328 | 3,408 | 0 | 0 |
| synth_v3 | cuad | 74,328 | 3,135 | 0 | 5 |
| synth_v3 | massive | 74,328 | 29,740 | 1 | 1 |
| measuring_hate_speech | helpsteer3 | 36,157 | 6,000 | 0 | 0 |
| measuring_hate_speech | measuring_hate_speech | 36,157 | 3,408 | 23 | 37 |
| measuring_hate_speech | cuad | 36,157 | 3,135 | 0 | 0 |
| measuring_hate_speech | massive | 36,157 | 29,740 | 0 | 0 |

Suite rows to leave out: helpsteer3 34, measuring_hate_speech 24, cuad 582, massive 1

## MultiNLI (added 2026-10-22)

Run against the suite as published, after the exclusions above: no overlap in either direction, so no new suite rows are left out.

| Training file | Suite set | training rows | suite rows | suite rows found in training | training rows found in suite |
|---|---|---|---|---|---|
| multinli | helpsteer3 | 328,206 | 5,966 | 0 | 0 |
| multinli | measuring_hate_speech | 328,206 | 3,384 | 0 | 0 |
| multinli | cuad | 328,206 | 2,553 | 0 | 0 |
| multinli | massive | 328,206 | 29,739 | 0 | 0 |
