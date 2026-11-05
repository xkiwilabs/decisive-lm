# 14. Datasets

Every public dataset the project uses is registered in `configs/datasets.yaml` with its source, licence, licence tier
and the exact files to download. Each file is pinned to a commit, a Hugging Face revision or a SHA-256 checksum. This
chapter lists the datasets by role, explains how to fetch and convert them, and covers what each release publishes and
the attribution the licences require.

## 14.1 Licence tiers

Every row carries a licence and a **tier**, derived from the licence (`decisivelm/data/dataset.py`):

| Tier | Meaning | Allowed in |
|---|---|---|
| **R** (release) | Permissive, commercial use allowed, no share-alike terms (CC0, CC BY, MIT, Apache-2.0, the Open American National Corpus (OANC) terms) | The published general head, user adapters, evaluation |
| **S** (share-alike) | CC BY-SA, ODbL, GPL | User and private adapters, evaluation |
| **X** (restricted) | Non-commercial, research-only, third-party media, or containing proprietary-model outputs | Private experiments and evaluation only; never in released weights |

A source may be placed in a *more* restrictive tier than its licence implies, with a recorded reason (`override`),
but never a less restrictive one. Rows derived from other rows inherit the most restrictive tier of their inputs. The
general head trains on R-tier rows only.

A dataset whose authors state no licence and no terms restricting its use is tier R, with licence `NOASSERTION`. The
model card lists it as "no licence available". Only data whose licence or terms clearly forbid the use is excluded. Two
development benchmarks with no stated licence, AG News and SST-2, are nonetheless placed in tier X by an `override`
(14.3).

Some data is never used, whatever its licence:

- **Labels from proprietary models** are kept out of the release tier: for example UltraFeedback (GPT-4 ratings), the
  Arena preference sets, WildGuardMix (GPT-4 labels) and xLAM-60k (a research-use gate).
- **The Stack Exchange data dump** is excluded entirely, because its download terms forbid training language models
  on it.
- **Public test splits** are used for evaluation only.
- **People-affecting outcomes**: nothing trains on historical credit, hiring or insurance decisions, and no question
  infers a protected attribute such as age, gender, ethnicity or health from faces, voices or names (chapter 16).

## 14.2 Training seeds

Human-labelled datasets the general head trains on (their `train` splits only). Their public test splits become
`gold_calib` and `gold_test`, which are never trained on.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| Banking77 (Casanueva et al. 2020) | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) | CC BY 4.0 | R | 13,083 online-banking queries; one Choice over 77 intents, plus exploded variants (schema explosion, section 13.3) |
| CLINC150 (Larson et al. 2019) | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos), "plus" | CC BY 3.0 | R | Virtual-assistant requests; one Choice over 150 intents plus a self-contained out-of-scope option, plus exploded variants |
| HelpSteer2 (Wang et al. 2024) | [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC BY 4.0 | R | Prompt–response pairs; five Scores (0–4): helpfulness, correctness, coherence, complexity, verbosity |
| Civil Comments (Borkan et al. 2019) | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | CC0 1.0 | R | Public comments; seven Nouls whose targets are the share of raters marking each toxicity attribute |
| GoEmotions (Demszky et al. 2020) | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache-2.0 | R | 58k Reddit comments rated by 3–5 raters; emotion Nouls (27 emotions plus neutral) and a four-way sentiment Choice (positive, negative, ambiguous, neutral) |
| LEDGAR (Tuggener et al. 2020; LexGLUE, Chalkidis et al. 2022) | [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue), `ledgar` | CC BY 4.0 | R | 80k contract provisions from SEC filings; one Choice over 100 provision types |
| HelpSteer2-Preference (Wang et al. 2025) | [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), `preference` | CC BY 4.0 | R | Trained on from 0.2. Neither released head trained on it. Two responses to one HelpSteer2 prompt; one Score over six levels, which response is better and by how much, with targets from the shares of all the pair's individual ratings. Each pair also appears with the responses swapped. Splits follow HelpSteer2's, by prompt. It is the same task as the suite's HelpSteer3, so from 0.2 HelpSteer3 is reported as a seen task |

LEDGAR's licence is as distributed in LexGLUE. The provisions themselves are public SEC EDGAR filings.

**Generated rows.** Question sets and states written by Gemma 4 26B A4B (`scripts/synth_generate.py`, chapter 13) are
registered as `synth_gemma4`. They are Apache-2.0 and R tier, because Gemma 4 places no terms on its outputs. Fan-out
rows, written about a real state from a seed, keep that seed's source, licence and tier.

## 14.3 Development benchmarks

Public test sets used to compare models zero-shot (with no training or calibration rows from the task) during
development. Each uses the published test (or validation) split, with the item sample and question wording of an
independent, publicly available benchmark harness, so results can be set beside published figures. Data always comes
from the original Hugging Face datasets.

The item samples and question wording are credited to two public harnesses:
[jevbench](https://github.com/dhruvmehra/jevbench) (MIT) for Banking77, AG News and SST-2, and
[jev-vs-open-decision-models](https://github.com/elcronos/jev-vs-open-decision-models) for emotion, TweetTopic,
Twitter financial news topics and DailyDialog. Only their sampling and wording are reused, never any model outputs
they contain.

Every row is `gold_test` with a one-hot human label (all probability on the one published answer, with no rater
shares), so these sets show accuracy and calibration but not whether confidence follows rater disagreement. Every one of
these sources is held out of the general head (`configs/holdout.yaml`). For Banking77 that is the 500-item sample only,
because the seed's own train split still trains the head (14.2). Because they were used to choose between candidate
heads, they are reported as development results, not as release gates.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| Banking77 test sample | [legacy-datasets/banking77](https://huggingface.co/datasets/legacy-datasets/banking77) | CC BY 4.0 | R | 500 items; one Choice over 77 intents with the harness's intent descriptions |
| AG News (Zhang et al. 2015) | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | none stated; X by `override`, because the original AG corpus is offered for non-commercial research | X | 500 items; one Choice over 4 topics |
| SST-2 (Socher et al. 2013) | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) | none stated; X by `override`, kept for evaluation only | X | 500 validation items; one Choice over 2 sentiments |
| emotion (Saravia et al. 2018) | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | other (research and education) | X | 2,000 tweets; one Choice over 6 emotions |
| TweetTopic (Antypas et al. 2022) | [cardiffnlp/tweet_topic_single](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) | other | X | 1,693 tweets; one Choice over 6 topics |
| Twitter financial news topics | [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | MIT (tweet text under X's terms) | X | 4,117 tweets; one Choice over 20 topics |
| DailyDialog (Li et al. 2017) | [OpenRL/daily_dialog](https://huggingface.co/datasets/OpenRL/daily_dialog) | CC BY-NC-SA 4.0 | X | 7,740 utterances; one Choice over 7 emotions |

X-tier sets are used for evaluation only and are never redistributed.

### Image and audio examples

Public multi-rater sets for the fine-tune examples of [chapter 9](09-fine-tuning-for-your-task.md) (section 9.9). None
is in the general head's training mix. Each is fetched from its authors' pinned files and converted locally. The media
stay under `data/raw/` and are never redistributed.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| KonIQ-10k (Hosu et al. 2020) | [OSF mirror](https://osf.io/hcsdy/); rating counts from [subpic/koniq](https://github.com/subpic/koniq) | database: no licence stated ("freely available to the research community"); the photos are Flickr images under their own licences; rating file MIT | X | 10,073 photos at 1024×768; one Score on technical quality over 5 levels; targets from about 107 raters per photo; the authors' split |
| CREMA-D (Cao et al. 2014) | [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | ODbL 1.0 (contents under the Database Contents License) | S | 7,442 clips; one Choice over 6 emotions; targets from about 10 listeners per clip who heard it without the video; splits by actor. The actors' demographics file is never read |
| THINGSplus (Hebart et al. 2019; Stoinski, Perkuhn & Hebart 2024) | [OSF project jum2f](https://osf.io/jum2f/) | ratings CC BY 4.0; photos CC0 | R | 1,845 objects, each with one public-domain photo; one Score on agreement that it is "something that lives", 7 levels; targets from about 37 raters per object; splits by object. Only the CC0 photos are used, never the password-protected THINGS image set. The raters' age and gender are never read |

### Zero-shot image check

One more multi-rater image set is held out from every mix and from every fine-tune example. It checks whether the
general head's confidence follows human disagreement on a kind of image it was not trained on. Both released heads were
trained only on VizWiz-VQA's photo questions. The E4B head was checked on this set
(`results/cifar10h_v5c_zero_shot.md`), and the 26B A4B head has not been. An earlier head, trained on text only, was
checked the same way (`results/cifar10h_v3d_zero_shot.md`).

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| CIFAR-10H (Peterson et al. 2019) | label counts from [jcpeterson/cifar-10h](https://github.com/jcpeterson/cifar-10h); images: the CIFAR-10 test set (Krizhevsky 2009) from [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) | labels CC BY-NC-SA 4.0; the images state no licence | X | 10,000 photos at 32 × 32; one Choice over the 10 classes; targets from about 50 people per photo; split 40% `gold_calib` and 60% `gold_test`. The counts are read without pickle, and the images are checked to be in the counts' order |

### Image and audio for version 1.0

Version 1.0 adds more images and audio to the general head. Three multi-rater sets are its training data, and two more
are frozen as its image and audio suites, held out of every mix and every user adapter (`configs/holdout.yaml`) before
any image or audio training. Both heads of the first release already train on 10,000 of VizWiz-VQA's training photos
(the General v5c mix, [chapter 13](13-how-the-general-head-was-trained.md)), and on none of the audio. As with every
source, the media stay under `data/raw/` and are never redistributed.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| VizWiz-VQA (Gurari et al. 2018) | [vizwiz.org](https://vizwiz.org/tasks-and-datasets/vqa/) | CC BY 4.0 | R | 24,842 photos taken by blind people, each with the question they asked; a Noul, whether the photo shows enough to answer, from the share of ten answerers who gave an answer; and, for 20,818 of them, the answer as a Choice. The train annotations become `train` and `val`, the validation annotations `gold_calib` and `gold_test`. The test split has no public answers and is not used |
| SONYC-UST v2 (Cartwright et al. 2020) | [Zenodo 3966543](https://zenodo.org/records/3966543) | CC BY 4.0 | R | 18,510 ten-second recordings from New York City street sensors; eight Nouls, one per kind of sound (engine, impact machinery, other impacts, powered saw, alert signal, music, human voice, dog), from three volunteers per clip. The SONYC team's own labels and the sensor's location and recording time are never used. SONYC's own splits, with its 2019 test recordings split by sensor into `gold_calib` and `gold_test` |
| Emozionalmente (Catania et al.) | [Zenodo 6569824](https://zenodo.org/records/6569824) | CC BY 4.0 | R | 6,902 clips of Italian actors; one Choice over seven emotions, from the listeners (about five per clip) who said which emotion they heard. Splits by actor. The listeners' demographics file is never read |
| TextVQA (Singh et al. 2019), frozen suite | [textvqa.org](https://textvqa.org) | annotations CC BY 4.0; the photos are Open Images, listed as CC BY 2.0 but not warranted per photo | X | The validation questions only: 4,985 questions on 3,161 photos that need reading the text in the photo; the answer as a Choice from ten answerers; split by photo into `gold_calib` (40%) and `gold_test` (60%) |
| ODAQ (Torcoli et al. 2024), frozen suite | [Zenodo 10405774](https://zenodo.org/records/10405774) | the scores CC BY 4.0; each recording under its own licence | X | 136 conditions of 17 excerpts: an original and a processed version (a codec, a band limit or source separation, or the original again, hidden), with one Score on how good the processed version sounds, over MUSHRA's five bands, from 26 expert listeners. Only excerpts whose audio is CC BY or CC0 are used. The 13 CC BY-NC excerpts are left out. Split by excerpt |

The photo question-answering sets have free-text answers. Each question becomes a Choice with a fixed number of options:
the four answers given most often (two, yes and no, for a yes/no question). When a question has fewer than four
different answers, the remaining options are answers to other questions, chosen from those no answerer gave to this one.
A fixed count keeps the number of options from telling the head how much the answerers disagreed. Answers are compared
after light normalisation, so "Dakota." and "dakota" are one answer, but "dakota" and "dakota digital" stay two, and the
head learns to split its confidence between them as the answerers did. A question is left out when its four options
cover fewer than half of the answerers who gave an answer (`decisivelm/data/converters/vqa.py`).

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
both directions, using word 8-grams (runs of eight consecutive words; exact match for texts under 8 words). A suite row
overlaps when at least half of its 8-grams occur in a training file. A training row overlaps when at least half of its
8-grams occur in a suite set, which catches a short training text copied into a long suite state (a LEDGAR provision
inside a CUAD section, for example). Suite rows found in either direction are listed in `configs/suite_exclusions.txt`
and left out by the converters. Of the 641 rows left out, 582 are CUAD sections, most of them sharing text with LEDGAR
provisions (`results/suite_overlap.md`). The audit was run again before the two released heads were scored, with three
more trainable sources (`synth_v4`, the version 4 teacher-written rows of chapter 13, VizWiz-VQA and
HelpSteer2-Preference), and found no overlap with anything they train on (`results/suite_overlap_v5c.md`). It found one
HelpSteer3 row that overlaps HelpSteer2-Preference, which version 0.2 trains on. That row was added to the exclusions on
2026-11-05, after both 0.1 heads were scored.

**The new-task development set.** MultiNLI is also held out of the general head, for a different purpose: it shows how
the head behaves on a task it has not seen, asked three ways. It was also where an earlier candidate's new-task
calibration scales were fitted, a correction the released head no longer uses (section 10.6). It is not part of the
suite and is not a release gate, but it is never trained on by the general head (`configs/holdout.yaml`).

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| MultiNLI 1.0 (Williams et al. 2018), without the fiction genre | [cims.nyu.edu/~sbowman/multinli](https://cims.nyu.edu/~sbowman/multinli/) | Open American National Corpus terms (free use and redistribution) | R | Premise–hypothesis pairs; one Choice: entailment, neutral or contradiction. The matched development pairs, labelled by five annotators each, become `gold_calib` (40%) and `gold_test` (60%), and also ask the same judgement as a Noul and as a 3-level Score |

- The fiction genre is dropped because its texts mix CC BY-SA 3.0 and other works with the OANC texts. Every other
  row is R tier.
- Splits are assigned by premise text, because several hypotheses share one premise. Training pairs whose premise
  appears in the matched development set are dropped.
- The archive is served from a university page rather than a commit, so it is pinned by its SHA-256.
- A user adapter may train on MultiNLI's `train` rows. Only the general head holds it out.

### Development sets for rating scales and other languages

Five more human-labelled sets check candidates during development. None is part of the suite or a release gate, and
neither released head (both trained on the General v5c mix) trained on any of them.

- **Wikipedia Talk toxicity** checks review readiness on a two-sided social scale rated by many people. It was added
  after General v3d failed criterion 1 on Measuring Hate Speech's Scores (chapter 15), and it is held out of the
  general head (`configs/holdout.yaml`), so every row is gold.
- **Stanford Politeness**, **Humicroedit** and the **Conversational Sarcasm Corpus** are human ratings with several
  raters per item. Two later candidates trained on them (one on politeness, one on humour and sarcasm) and were not
  released (docs/RELEASE.md, change log 2026-10-26 and 2026-10-27). Politeness is now tier S, so the general head
  cannot train on it. From 0.2, Wikipedia toxicity, Humicroedit and the sarcasm corpus are the development checks of
  transfer to two-sided scales the head has not seen (docs/RELEASE.md, change log 2026-10-27).
- **SIB-200** checks calibration on text that is not English. It is tier S and held out of the general head, so every
  row is gold.

| Dataset | Source | Licence | Tier | Becomes |
|---|---|---|---|---|
| Wikipedia Talk Labels, Toxicity (Wulczyn et al. 2017) | [figshare 4563973](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) | CC0 1.0 | R | Comments from the release's own test split with at least 3 raters (about 10 each); one Score over 5 levels from very toxic to very healthy; 2,040 `gold_calib` and 3,060 `gold_test` rows |
| Stanford Politeness Corpus, Stack Exchange part (Danescu-Niculescu-Mizil et al. 2013) | [ConvoKit](https://convokit.cornell.edu/documentation/stack_politeness.html) | CC BY-SA 3.0 (the Stack Exchange posts) | S | 6,603 requests, 5 raters each; one Score over 5 levels from very impolite to very polite, binned from a 1–25 slider; splits by text: 15% `gold_test`, 10% `gold_calib`, 5% `val`, the rest `train` |
| Humicroedit (Hossain et al. 2020; SemEval-2020 Task 7) | [Zenodo 3969509](https://zenodo.org/records/3969509) | CC BY 4.0 | R | About 15,000 news headlines with one word edited to be funny, plus FunLines' 8,000, about 5 graders each; one Score over 4 grades from not funny to funny; the task's own splits, its test split becoming `gold_calib` (40%) and `gold_test` (60%) |
| Conversational Sarcasm Corpus (Jang & Frassinelli 2024) | [hyewon-jang-kn/conversational-sarcasm-corpus](https://github.com/hyewon-jang-kn/conversational-sarcasm-corpus) | MIT | R | 7,036 replies to everyday scenarios, each rated by 4 or 6 other people; one Score over 6 levels of sarcasm; splits by scenario: 15% `gold_test`, 10% `gold_calib`, 5% `val`, the rest `train` |
| SIB-200 (Adelani et al. 2024) | [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | CC BY-SA 4.0 | S | 1,004 news sentences, one published topic each, in the 195 languages outside the suite's ten; one Choice over 7 topics; split by sentence into `gold_calib` (40%) and `gold_test` (60%) |

## 14.5 Fetching and converting

Download a dataset's pinned files:

```bash
uv run python scripts/fetch_dataset.py banking77 clinc150
```

Each `fetch` entry in `configs/datasets.yaml` is either a URL pinned to a commit (`url`, `file`) or a Hugging Face
dataset file at a fixed revision (`hf`, `revision`, `path`, `file`). An optional `sha256` pins the content too. A file
that does not match is removed and the fetch stops. Files go to `data/raw/<name>/`, with a `MANIFEST.json` recording
each file's source and SHA-256. `--jobs N` downloads N of a dataset's files at once, which helps with a slow host such
as Zenodo.

Convert them into canonical rows (chapter 8):

```bash
uv run python scripts/convert_dataset.py banking77 clinc150 [--backbone e4b]
```

This writes `data/processed/<name>.jsonl`. Each converter copies only the fields it needs. Rows are checked against
the dataset's registered licence and tier before they are written. Long states are truncated with the backbone's
tokenizer (`--backbone`), exactly as the model will see them. Splits are assigned per source item before any variant
is derived, and the split assignment's hashes are printed so that a rebuild can be checked. Suite rows listed in
`configs/suite_exclusions.txt` are left out.

The converter names are the registry names:

- training seeds: `banking77`, `clinc150`, `helpsteer2`, `helpsteer2_preference`, `civil_comments`, `goemotions`,
  `ledgar`;
- held-out and development sets: `helpsteer3`, `measuring_hate_speech`, `cuad`, `massive`, `multinli`, `sib200`,
  `wiki_toxicity`, `politeness`, `humicroedit`, `csc_sarcasm`;
- image and audio examples: `koniq10k`, `crema_d`, `things_living`; the zero-shot image check: `cifar10h`;
- image and audio for version 1.0: `vizwiz_vqa`, `sonyc_ust`, `emozionalmente`, and the suites `textvqa`, `odaq`;
- the benchmark sets: `bench_banking77`, `bench_agnews`, `bench_sst2`, `bench_emotion`, `bench_tweet_topic`,
  `bench_fin_topic`, `bench_daily_dialog`.

Summarise any converted file:

```bash
uv run python scripts/inspect_data.py data/processed/banking77.jsonl
```

This prints per-type counts, option-count and level histograms, Noul target distributions, media and truncation
rates.

## 14.6 Your own data

Your own datasets follow the same path. Register the source in `configs/datasets.local.yaml` with its licence and
tier (section 8.8 has an example), write a converter that maps your raw files to canonical rows (chapter 8), and keep
your raw data under `data/raw/`. `configs/datasets.local.yaml` is not tracked by git, so a private source never
reaches the public registry.
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
- no restricted-tier or private data, holdout row, or output of a third-party commercial decision service (the
  clean-room rule, section 16.1) is included.

## 14.8 Attribution

Most datasets here are CC BY, which requires credit. When you publish anything derived from them (fan-out rows, model
weights trained on them, results tables), cite the dataset authors listed above, link the source and the licence, and
say what was changed. Fan-out rows written about a seed's state keep that seed's licence and are credited to it in the
published dataset card. CLINC150 is CC BY 3.0 and Civil Comments is CC0, which asks for no attribution, though citing it
is good practice. MultiNLI's texts come from the Open American National Corpus, which places no restrictions on use.
Cite Williams, Nangia and Bowman (2018) and link the corpus page.

THINGSplus ratings are CC BY 4.0: cite Hebart et al. (2019, PLoS ONE) and Stoinski, Perkuhn and Hebart (2024,
Behavior Research Methods). CREMA-D is under the ODbL: cite Cao et al. (2014, IEEE Transactions on Affective
Computing), and anything built from it as a database must be shared alike. CIFAR-10H is CC BY-NC-SA 4.0: cite
Peterson, Battleday, Griffiths and Russakovsky (2019, ICCV) and, for the images, Krizhevsky (2009).

The annotations and scores of the version 1.0 sets are CC BY 4.0, and so are the photos and recordings of VizWiz-VQA,
SONYC-UST and Emozionalmente. Cite Gurari et al. (2018, CVPR) for VizWiz-VQA, Cartwright et al. (2020, DCASE) for
SONYC-UST, Catania et al. (the Zenodo record) for Emozionalmente, Singh et al. (2019, CVPR) for TextVQA and Torcoli et
al. (2024, ICASSP) for ODAQ. TextVQA's photos come from Open Images, listed as CC BY 2.0 but not warranted per photo.
ODAQ's recordings carry their own credits, listed per file in the dataset's `_detailed_license.csv`.

The development benchmarks reuse item samples and question wording from the jevbench harness, which is MIT-licensed. Its
copyright and permission notice is reproduced in the repository's `NOTICE` file.

Next: [15. Release criteria](15-release-criteria.md)
