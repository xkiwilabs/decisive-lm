# 14. Datasets

Every public dataset the project uses is registered in `configs/datasets.yaml` with its source, licence, licence tier
and the exact files to download. Each file is pinned to a commit, a Hugging Face revision or a SHA-256 checksum. This
chapter lists the datasets by role and explains how to fetch and convert them. It also covers what each release
publishes and the attribution the licences require.

## 14.1 Licence tiers

Every row carries a licence and a **tier**, derived from the licence (`decisivelm/data/dataset.py`):

| Tier | Meaning | Allowed in |
|---|---|---|
| **R** (release) | Permissive, commercial use allowed, no share-alike terms (CC0, CC BY, MIT, Apache-2.0, the OANC's terms) | The published general head, user adapters, evaluation |
| **S** (share-alike) | CC BY-SA, ODbL, GPL | User and private adapters, evaluation |
| **X** (restricted) | Non-commercial, research-only, third-party media, or containing proprietary-model outputs | Private experiments and evaluation only; never in released weights |

A source may be placed in a *more* restrictive tier than its licence implies, with a recorded reason (`override`),
but never a less restrictive one. Rows derived from other rows inherit the most restrictive tier of their inputs. The
general head trains on R-tier rows only.

A dataset whose authors state no licence and no terms restricting its use is tier R, with licence `NOASSERTION`; the
model card lists it as "no licence available". Only data whose licence or terms clearly forbid the use is excluded.

Some data is never used, whatever its licence:

- **Labels from proprietary models** are kept out of the release tier: for example UltraFeedback (GPT-4 ratings), the
  Arena preference sets, WildGuardMix (GPT-4 labels) and xLAM-60k (a research-use gate).
- **The Stack Exchange data dump** is excluded entirely: its download terms forbid training language models on it.
- **Public test splits** are never trained on; they are evaluation data only.
- **People-affecting outcomes**: nothing trains on historical credit, hiring or insurance decisions, and no question
  infers a protected attribute such as age, gender, ethnicity or health from faces, voices or names (chapter 16).

## 14.2 Training seeds

Human-labelled datasets the general head trains on (their `train` splits only). Their public test splits become
`gold_calib` and `gold_test`, which are never trained on.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| Banking77 (Casanueva et al. 2020) | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) | CC BY 4.0 | R | 13,083 online-banking queries; one Choice over 77 intents, plus exploded variants |
| CLINC150 (Larson et al. 2019) | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos), "plus" | CC BY 3.0 | R | Virtual-assistant requests; one Choice over 150 intents plus a self-contained out-of-scope option, plus exploded variants |
| HelpSteer2 (Wang et al. 2024) | [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC BY 4.0 | R | Prompt–response pairs; five Scores (0–4): helpfulness, correctness, coherence, complexity, verbosity |
| Civil Comments (Borkan et al. 2019) | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | CC0 1.0 | R | Public comments; seven Nouls whose targets are the share of raters marking each toxicity attribute |
| GoEmotions (Demszky et al. 2020) | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache-2.0 | R | 58k Reddit comments rated by 3–5 raters; emotion Nouls (27 emotions plus neutral) and a four-way sentiment Choice (positive, negative, ambiguous, neutral) |
| LEDGAR (Tuggener et al. 2020; LexGLUE, Chalkidis et al. 2022) | [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue), `ledgar` | CC BY 4.0 | R | 80k contract provisions from SEC filings; one Choice over 100 provision types |

LEDGAR's licence is as distributed in LexGLUE; the provisions themselves are public SEC EDGAR filings.

**Generated rows.** Question sets and states written by Gemma 4 26B A4B (`scripts/synth_generate.py`, chapter 13) are
registered as `synth_gemma4`. They are Apache-2.0 and R tier: Gemma 4 places no terms on its outputs. Fan-out rows,
written about a real state from a seed, keep that seed's source, licence and tier.

## 14.3 Development benchmarks

Public test sets used to compare models zero-shot during development. Each uses the published test (or validation)
split, with the item sample and question wording of an independent, publicly available benchmark harness, so results
can be set beside published figures. Data always comes from the original Hugging Face datasets.

The item samples and question wording are credited to two public harnesses:
[jevbench](https://github.com/dhruvmehra/jevbench) (MIT) for Banking77, AG News and SST-2, and
[jev-vs-open-decision-models](https://github.com/elcronos/jev-vs-open-decision-models) for emotion, TweetTopic,
Twitter financial news topics and DailyDialog. Only their sampling and wording are reused, never any model outputs
they contain.

Every row is `gold_test` with a one-hot human label, and every one of these sources is held out of the general head.
Because they were used to choose between candidate heads, they are reported as development results, not as release
gates.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| Banking77 test sample | [legacy-datasets/banking77](https://huggingface.co/datasets/legacy-datasets/banking77) | CC BY 4.0 | R | 500 items; one Choice over 77 intents with the harness's intent descriptions |
| AG News (Zhang et al. 2015) | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | none stated | X | 500 items; one Choice over 4 topics |
| SST-2 (Socher et al. 2013) | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) | none stated | X | 500 validation items; one Choice over 2 sentiments |
| emotion (Saravia et al. 2018) | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | other (research and education) | X | 2,000 tweets; one Choice over 6 emotions |
| TweetTopic (Antypas et al. 2022) | [cardiffnlp/tweet_topic_single](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) | other | X | 1,693 tweets; one Choice over 6 topics |
| Twitter financial news topics | [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | MIT (tweet text under X's terms) | X | 4,117 tweets; one Choice over 20 topics |
| DailyDialog (Li et al. 2017) | [OpenRL/daily_dialog](https://huggingface.co/datasets/OpenRL/daily_dialog) | CC BY-NC-SA 4.0 | X | 7,740 utterances; one Choice over 7 emotions |

X-tier sets are used for evaluation only and are never redistributed.

### Image and audio examples

Public multi-rater sets for the fine-tune examples of [chapter 9](09-fine-tuning-for-your-task.md) (section 9.9).
None is in the general head's training mix. Each is fetched from its authors' pinned files and converted locally;
the media stay under `data/raw/` and are never redistributed.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| KonIQ-10k (Hosu et al. 2020) | [OSF mirror](https://osf.io/hcsdy/); rating counts from [subpic/koniq](https://github.com/subpic/koniq) | database: no licence stated ("freely available to the research community"); the photos are Flickr images under their own licences; rating file MIT | X | 10,073 photos at 1024×768; one Score on technical quality over 5 levels; targets from about 107 raters per photo; the authors' split |
| CREMA-D (Cao et al. 2014) | [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | ODbL 1.0 (contents under the Database Contents License) | S | 7,442 clips; one Choice over 6 emotions; targets from about 10 listeners per clip who heard it without the video; splits by actor. The actors' demographics file is never read |
| THINGSplus (Hebart et al. 2019; Stoinski, Perkuhn & Hebart 2024) | [OSF project jum2f](https://osf.io/jum2f/) | ratings CC BY 4.0; photos CC0 | R | 1,845 objects, each with one public-domain photo; one Score on agreement that it is "something that lives", 7 levels; targets from about 37 raters per object; splits by object. Only the CC0 photos are used, never the password-protected THINGS image set; the raters' age and gender are never read |

## 14.4 The frozen held-out suite

Four public, human-labelled datasets used only for release gates (chapter 15). No training, selection, tuning or
calibration decision ever sees them. Each is held out of the general head and of user adapters in
`configs/holdout.yaml`.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| HelpSteer3 preference (Wang et al. 2025) | [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | CC BY 4.0 | R | A fixed random 6,000 response pairs (general, STEM, code, multilingual); one Score with six levels: which of two responses is better, and by how much, rated by up to 3 annotators |
| Measuring Hate Speech (Kennedy et al. 2020) | [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | CC BY 4.0 | R | The evaluation portion: 20% of the comments with at least 3 raters, chosen by comment id; nine survey items as Scores (0–4) and hate speech as a Noul |
| CUAD v1 (Hendrycks et al. 2021) | [theatticusproject/cuad](https://huggingface.co/datasets/theatticusproject/cuad) | CC BY 4.0 | R | Ten held-out clause types as Nouls on contract sections of at most 10,000 characters; one expert label each |
| MASSIVE 1.1 (FitzGerald et al. 2022) | [alexa/massive](https://github.com/alexa/massive) | CC BY 4.0 | R | The test split in ten held-out languages (Polish, Urdu, Finnish, Georgian, Amharic, Swahili, Tamil, Korean, Vietnamese, Tagalog); one Choice over 60 intents |

Notes on how they are converted:

- **HelpSteer3**: individual preference ratings run from −3 to +3 with no zero, so the six levels are the annotators'
  own. Its card says only the three most-agreeing of up to five annotations were released, so its disagreement is
  understated.
- **Measuring Hate Speech**: the converter keeps the comment text and the per-annotator item ratings only. Annotator
  demographic columns are dropped. The hate-speech Noul's target counts "unclear" ratings as half. The remaining
  comments may be used for training.
- **CUAD**: a clause type is "yes" for a section when an expert-marked span lies wholly inside it, and "no" when no
  span touches it. A span that crosses a section boundary leaves that question out for that section.
- **MASSIVE**: the dataset is parallel (one utterance id across all locales), so no test-split id is trained on in
  any locale, and only the test split of the held-out locales is scored. The archive is served from S3 rather than a
  commit, so it is pinned by its SHA-256.

**No overlap with trainable data.** `scripts/overlap_audit.py` compares the suite's text with every trainable source in
both directions, using word 8-grams (exact match for texts under 8 words). Suite rows found in training data are
listed in `configs/suite_exclusions.txt` and left out by the converters. Most of them are CUAD sections that share
text with LEDGAR provisions.

**The new-task development set.** One more human-labelled dataset is held out of the general head, for a different
purpose: it shows how the head behaves on a task it has not seen, asked three ways. It was also where an earlier
candidate's new-task calibration scales were fitted, a correction the released head no longer uses (section 10.6).
It is not part of the suite and is not a release gate, but it is never trained on by the general head
(`configs/holdout.yaml`).

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| MultiNLI 1.0 (Williams et al. 2018), without the fiction genre | [cims.nyu.edu/~sbowman/multinli](https://cims.nyu.edu/~sbowman/multinli/) | Open American National Corpus terms (free use and redistribution) | R | Premise–hypothesis pairs; one Choice: entailment, neutral or contradiction. The matched development pairs, labelled by five annotators each, become `gold_calib` (40%) and `gold_test` (60%), and also ask the same judgement as a Noul and as a 3-level Score |

- The fiction genre is dropped because its texts mix CC BY-SA 3.0 and other works with the OANC texts. Every other
  row is R tier.
- Splits are assigned by premise text, because several hypotheses share one premise. Training pairs whose premise
  appears in the matched development set are dropped.
- The archive is served from a university page rather than a commit, so it is pinned by its SHA-256.
- A user adapter may train on MultiNLI's `train` rows; only the general head holds it out.

## 14.5 Fetching and converting

Download a dataset's pinned files:

```bash
uv run python scripts/fetch_dataset.py banking77 clinc150
```

Each `fetch` entry in `configs/datasets.yaml` is either a URL pinned to a commit (`url`, `file`) or a Hugging Face
dataset file at a fixed revision (`hf`, `revision`, `path`, `file`). An optional `sha256` pins the content too. A file
that does not match is removed and the fetch stops. Files go to `data/raw/<name>/`, with a `MANIFEST.json` recording
each file's source and SHA-256.

Convert them into canonical rows (chapter 8):

```bash
uv run python scripts/convert_dataset.py banking77 clinc150 [--backbone e4b]
```

This writes `data/processed/<name>.jsonl`. Each converter copies only the fields it needs. Rows are checked against
the dataset's registered licence and tier before they are written. Long states are truncated with the backbone's
tokenizer (`--backbone`), exactly as the model will see them. Splits are assigned per source item before any variant
is derived, and the split assignment's hashes are printed so that a rebuild can be checked. Suite rows listed in
`configs/suite_exclusions.txt` are left out.

The converter names are the registry names: `banking77`, `clinc150`, `helpsteer2`, `civil_comments`, `goemotions`,
`ledgar`, `multinli`, `helpsteer3`, `measuring_hate_speech`, `cuad`, `massive`, and the benchmark sets (`bench_banking77`,
`bench_agnews`, `bench_sst2`, `bench_emotion`, `bench_tweet_topic`, `bench_fin_topic`, `bench_daily_dialog`).

Summarise any converted file:

```bash
uv run python scripts/inspect_data.py data/processed/banking77.jsonl
```

This prints per-type counts, option-count and level histograms, Noul target distributions, media and truncation
rates.

## 14.6 Your own data

Your own datasets follow the same path. Register the source in `configs/datasets.yaml` with its licence and tier,
write a converter that maps your raw files to canonical rows (chapter 8), and keep your raw data under `data/raw/`.
The whole `data/` directory is excluded from version control. Data you cannot publish belongs in tier X, which keeps
it out of any released weights.

Chapter 9 walks through fine-tuning on your own data.

## 14.7 What each release publishes

Data is never stored in the code repository. Each release publishes its data on Hugging Face as the dataset repository
`decisive-lm-data`, tagged with the release version (`vX.Y`), or links to where it came from:

- **Linked, not re-hosted:** every public seed, benchmark and suite dataset. The pinned sources in
  `configs/datasets.yaml`, `scripts/fetch_dataset.py` and `scripts/convert_dataset.py` rebuild the same rows.
- **Published:**
  - the teacher-written rows (states and question sets), with the teacher's calibrated probabilities for every
    question and the calibration that produced them;
  - the release mix manifest: source row ids, sources, licences, row counts and SHA-256, from which the
    human-labelled rows are rebuilt from their sources;
  - per-item predictions of the released head, P0 and the teacher on the public benchmarks and the frozen suite.

Before publishing, each release checks and records on the dataset card that:

- every published row is R tier;
- the teacher's licence allows its outputs to be published;
- no restricted-tier or private data, holdout row, or output of a third-party commercial decision service is included.

## 14.8 Attribution

Most datasets here are CC BY, which requires credit. When you publish anything derived from them (fan-out rows, model
weights trained on them, results tables), cite the dataset authors listed above, link the source and the licence, and
say what was changed. Fan-out rows written about a seed's state keep that seed's licence and are credited to it in the
published dataset card. CLINC150 is CC BY 3.0 and Civil Comments is CC0, which asks for no attribution, though citing
it is good practice. MultiNLI's texts come from the Open American National Corpus, which places no restrictions on
use; cite Williams, Nangia and Bowman (2018) and link the corpus page.

THINGSplus ratings are CC BY 4.0: cite Hebart et al. (2019, PLoS ONE) and Stoinski, Perkuhn and Hebart (2024,
Behavior Research Methods). CREMA-D is under the ODbL: cite Cao et al. (2014, IEEE Transactions on Affective
Computing), and anything built from it as a database must be shared alike.

The development benchmarks reuse item samples and question wording from the jevbench harness, which is MIT-licensed;
its copyright and permission notice is reproduced in the repository's `NOTICE` file.

Next: [15. Release criteria](15-release-criteria.md)
