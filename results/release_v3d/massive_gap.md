# MASSIVE: serving path vs written predictions, per question (2026-10-27)

check_bundle_massive.txt reports max |Δp| 0.216, ten times the other suite sets. Per-question distribution on all gold_test rows (serving path, one request per pass, vs predict_bundle's packed micro-batches; bf16):

```
29739 questions: mean 5.23e-03 median 2.44e-03 p99 3.10e-02 p999 5.30e-02 max 2.16e-01; >0.05: 39, >0.1: 4
Δp 0.216 massive-sw-KE-13199 sw-KE written top 0.382 (qa_maths) serving top 0.599 (qa_maths) options 60
Δp 0.156 massive-fi-FI-2373 fi-FI written top 0.604 (play_music) serving top 0.448 (play_music) options 60
Δp 0.135 massive-fi-FI-14000 fi-FI written top 0.242 (cooking_query) serving top 0.236 (general_quirky) options 60
Δp 0.109 massive-fi-FI-4692 fi-FI written top 0.276 (takeaway_query) serving top 0.282 (play_music) options 60
Δp 0.096 massive-sw-KE-8635 sw-KE written top 0.400 (cooking_query) serving top 0.383 (takeaway_order) options 60
Δp 0.081 massive-sw-KE-4043 sw-KE written top 0.569 (iot_wemo_off) serving top 0.650 (iot_wemo_off) options 60
Δp 0.081 massive-sw-KE-15556 sw-KE written top 0.540 (social_post) serving top 0.621 (social_post) options 60
Δp 0.079 massive-fi-FI-1059 fi-FI written top 0.519 (iot_cleaning) serving top 0.440 (iot_cleaning) options 60
Δp 0.078 massive-fi-FI-9284 fi-FI written top 0.295 (iot_wemo_off) serving top 0.274 (play_radio) options 60
Δp 0.070 massive-fi-FI-6682 fi-FI written top 0.664 (news_query) serving top 0.595 (news_query) options 60
```

Median 0.0024, p99 0.031; 4 of 29,739 above 0.1, all low-confidence Swahili or Finnish questions over 60 intents, where bf16 packing noise (results/invariance.md: up to 0.39 on E4B) moves near-flat distributions most. Not a wiring difference: criterion 5 passes.
