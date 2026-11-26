# Changelog

All notable changes are recorded here. Versions follow [semantic versioning](https://semver.org/). The release criteria
for each version are in [docs/RELEASE.md](docs/RELEASE.md).

## 0.1.0 (2026-11-26)

- The toolkit: packed tree attention on Gemma 4 E4B and 26B A4B with separate global and sliding-window masks, the
  Choice, Score and Noul question types, LoRA adapters with a scorer network, calibration on human labels, and the
  evaluation and release scripts (guide chapters 6–11 and 15).
- Fine-tuning from a trained bundle (`train_p3.py --init-bundle`, `--n-train`; guide 9.6).
- The new-task factor for a general head: one factor on every Choice temperature and one on every Score temperature,
  fitted on about 100 labelled items of a task (`recalibrate.py --new-task`; guide 10.6).
- Exchange formats for a human-rating app (`export_for_rating.py`, `import_ratings.py`; docs/rating-formats.md).
- Task calibration of one question on about 100 of its own labelled items: a scale and an offset per option or level
  (`task_calibrate.py`, `decisivelm.calibrate.fit_task_calibration`; guide 10.6).
- Image questions in the general heads' training mix (VizWiz-VQA), and image and audio fine-tunes (guide 9.9).
- Audio sent to a backbone without audio input (26B A4B) is refused with a message saying so, and the HTTP server
  returns status 400.
- One HelpSteer3 row of the frozen suite (`helpsteer3-train-34503`) is excluded, because its text overlaps a dataset the
  next release trains on (`configs/suite_exclusions.txt`; docs/RELEASE.md, change log 2026-11-05).
- The first two general heads, both trained on General v5c (the training mix of guide chapter 13), each released with
  its failures of the release criteria stated (docs/RELEASE.md, change log 2026-11-03 and 2026-11-05):
  - the E4B head (`decisivelm-e4b-general`, docs/model-card.md), which fails three checks against the prompted E4B;
  - the 26B A4B head (`decisivelm-26b-a4b-general`, docs/model-card-26b.md), which runs in float32 computation and
    fails five checks against the prompted 26B A4B.
