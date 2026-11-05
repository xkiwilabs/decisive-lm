# Contributing

Thank you for helping. Bug reports, fixes, new converters and documentation are all welcome. Please open an issue
before a large change, so we can agree on the approach first.

## Setting up

```bash
uv sync          # Python 3.13 environment with the pinned dependencies
uv run pytest    # the test suite: CPU only, tiny random Gemma 4 configurations, no downloads
```

Tests that need real Gemma 4 weights are marked `real_model` and are skipped by default. Every pull request must keep
`uv run pytest` passing, and new code comes with tests.

## Rules the code depends on

These are easy to break silently, so the tests check them. Please don't weaken a test to make a change pass.

- **Two attention masks, not one.** Gemma 4 interleaves sliding-window and global layers. The packed tree needs a
  global mask and a sliding-window mask, and the sliding window is measured on *logical* positions, not on positions
  in the packed sequence. One mask applied to every layer gives wrong answers without any error
  ([guide chapter 6](docs/guide/06-how-it-works.md)).
- **Independent answers.** Each answer sees the state, its own question and its own option, and nothing else: not
  other options, not other questions, and not its own index. Adding or removing a question, or reordering options,
  must not change any other answer. Question ids are never shown to the model.
- **Hidden states, not logits.** The scorer reads the decoder's last hidden state at each query token. Don't compute the
  language-model output layer (`lm_head`), and don't resize the vocabulary, because the query tokens reuse reserved
  token ids.
- **One preprocessing path.** `decisivelm/preprocess.py` is shared by the teacher labeller and the model. Any
  difference between what they see becomes label noise.
- **Never hard-code a backbone.** Read sizes such as the sliding window and the hidden size from the model config or
  `configs/model.yaml`. Both E4B and 26B A4B must keep working.
- **Configuration in YAML.** No hard-coded paths.

## Data and model rules for contributions

- **Clean room.** The request format follows the public documentation of TypeSafe's Jev System One API. Never send
  requests to Jev or any TypeSafe service, and never contribute data, labels, tests or comparisons that use Jev's
  outputs, including old response logs
  ([guide chapter 16](docs/guide/16-provenance-licences-and-responsible-use.md#161-the-clean-room-rule)).
- **Model origin.** Models developed by a Chinese organisation, or derived from one, are not used in any role:
  backbone, teacher, baseline, embeddings, speech or OCR. Record the developer and origin of any model you add.
- **Licences.** Register every new dataset in `configs/datasets.yaml` with its licence, tier and pinned files. The
  tier follows from the licence: R (release) for permissive licences, S for share-alike licences and X for restricted
  or unclear ones ([guide 14.1](docs/guide/14-datasets.md#141-licence-tiers)). A source may be put in a more
  restrictive tier with a recorded reason, never a less restrictive one. A converter copies only the fields it needs.
  Only R-tier data goes into the published general heads.
- **Held-out data stays held out.** Never train on public test splits or on anything in `configs/holdout.yaml`, which
  includes the frozen evaluation suite (the four public datasets that release candidates are scored on, fixed before
  any candidate was scored, [guide chapter 15](docs/guide/15-release-criteria.md)).
- **No data in git.** Datasets, predictions and bundles stay under `data/` and `bundles/`, which are not tracked.

## Pull requests

- Keep each pull request to one change, with a short description of what it does and why.
- Update the guide when you change behaviour a user can see.
- By contributing, you agree that your contribution is licensed under the Apache License, Version 2.0.
- Please follow the [code of conduct](CODE_OF_CONDUCT.md).
