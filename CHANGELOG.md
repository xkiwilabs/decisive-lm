# Changelog

All notable changes are recorded here. Versions follow [semantic versioning](https://semver.org/); the release
criteria for each version are in [docs/RELEASE.md](docs/RELEASE.md).

## Unreleased

- The toolkit: packed tree attention on Gemma 4 E4B and 26B A4B with separate global and sliding-window masks, the
  Choice, Score and Noul question types, LoRA adapters with a scorer head, calibration on human labels, and the
  evaluation and release scripts (guide chapters 6–11 and 15).
- Fine-tuning from a trained bundle (`train_p3.py --init-bundle`, `--n-train`; guide 9.6).
- New-task calibration scales for the general head (`recalibrate.py --new-task`; guide 10.6).
- Exchange formats for a human-rating app (`export_for_rating.py`, `import_ratings.py`; docs/rating-formats.md).
- No general head has been released yet. The first release (0.1.0) will be the first candidate that passes the
  release criteria.
