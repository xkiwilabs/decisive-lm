# 13. How the general head was trained

A general head is one adapter that answers unseen questions about unseen states with calibrated confidence. Release 0.1
has two, one on Gemma 4 E4B and one on Gemma 4 26B A4B, trained on the same mix with the same recipe (section 13.7 gives
the differences). In this chapter, "the general head" means the E4B head unless it says otherwise. It is trained with
the same toolkit a user would use for their own task. Only the data differs: a broad mix of human-labelled public
datasets from release-tier sources (permissive licences that allow commercial use and have no share-alike terms), schema
variants derived from them, and question sets and states written and labelled by an open teacher. This chapter describes
that recipe and gives the commands that rebuild it. Each release publishes its exact mix manifest (sources, licences,
row counts and checksums), which is the authoritative record for that release.

## 13.1 The parts

| Part | What it is | Labels |
|---|---|---|
| Human-labelled seed datasets | Six public datasets covering intents, quality ratings, toxicity, emotions and legal provisions | The raters' own label shares |
| Schema explosion | New questions built from the intent seeds by subsetting and recombining their options | Computed from the human labels |
| Teacher-written rows | Question sets and states the teacher writes across many domains | The teacher's, calibrated on human gold |
| Fan-out on real states | New question sets the teacher writes about real states from the seeds | The teacher's, calibrated on human gold |
| Human-labelled photo questions | 10,000 photos from VizWiz-VQA, each taken by a blind person to ask a question about it | The answerers' own shares |

Every training row is a `train` row of an R-tier (release-tier) source (section 14.1, chapter 16).

## 13.2 Human-labelled seed datasets

| Seed | Becomes |
|---|---|
| Banking77 | One Choice over 77 online-banking intents |
| CLINC150 | One Choice over 150 virtual-assistant intents plus a self-contained out-of-scope option (151) |
| HelpSteer2 | Five Scores (0–4): helpfulness, correctness, coherence, complexity, verbosity of a response |
| Civil Comments | Seven Nouls: toxicity, severe toxicity, obscene, threat, insult, identity attack, sexually explicit |
| GoEmotions | Nouls for emotions (neutral plus a sample of the other 27, each at its natural rate) and a four-way sentiment Choice |
| LEDGAR | One Choice over 100 contract provision types |

**Photos.** The released heads' mix, General v5c, adds 10,000 photos from VizWiz-VQA's training split (Gurari et al.
2018; CC BY 4.0). Each photo is asked whether it shows enough to answer its question (a Noul, from the share of the ten
crowd answerers who gave an answer rather than calling the photo unanswerable). It is also asked which answer is right,
as a Choice over the four answers given most often (yes and no for a yes/no question, section 14.3), where at least
three answerers gave one of them. The photos are not a seed for criterion 4 of the release criteria, which uses the six
text datasets above (chapter 15).

Where a dataset has several raters per item, the target is the **share** of raters choosing each answer, never the
majority vote. Soft targets let the head's confidence follow human disagreement. An item that splits raters 2–1
should not be learned as certain. Links, licences and sizes are in [chapter 14](14-datasets.md).

## 13.3 Schema explosion

`scripts/explode.py` turns each intent row into new questions with the same state and a target computed from the human
label (`decisivelm/synth/multipliers/schema_explosion.py`):

- **Subsets**: the gold option plus *k* random distractors (*k* from 1 to 60), the target renormalised over the kept
  options. The head learns to choose among option sets of every size, not only the dataset's full list.
- **Drop-gold plus `other`**: the gold option and its two most confusable neighbours (the options whose descriptions
  share the most words with the gold option's) are removed and a self-contained `other` option is added, which
  becomes the right answer. This teaches the head to recognise when no listed option fits. These rows are capped at 5%
  of each source in the mix (`--cap schema_explosion:drop_gold=0.05`), since a higher share teaches the head to choose
  a catch-all too readily.
- **Noul splits**: "Does this message express *intent X*?" for the gold intent or a distractor, with positives at
  about 40%.

Derived rows keep their parent's split, licence, tier and schema family, so no variant of a test row reaches training.

## 13.4 Teacher-written question sets and states

`scripts/synth_generate.py` asks the teacher (Gemma 4 26B A4B) to write new tasks. Each job starts from a seeded
**spec** (`decisivelm/synth/generate.py`), which fixes everything that should vary. The teacher only writes text to it.
A spec chooses:

- **A domain and setting.** Twenty domains with several settings each: customer support, banking and finance, content
  moderation, documents and forms, security and guardrails, retail, logistics, healthcare administration, legal,
  education, qualitative research coding, human resources, real estate, travel, software engineering, media, public
  services, energy, agriculture and food, sports and recreation. Each domain also has the kinds of state it produces
  (emails, tickets, reviews, log excerpts, contract clauses, survey answers…).
- **The questions.** Two to eight per set, about 40% Noul, 35% Choice and 25% Score (version 4 asks for more Scores:
  30% Noul, 25% Choice and 45% Score):
  - Choice sizes follow a long tail from 2 to 20 options, mostly 3–15;
  - Scores have 2–10 levels;
  - each question has a description style (terse, descriptive, codebook-like or structured with examples).
- **Option extras.** A self-contained `other` option, a catch-all written the way people write one ("None of the
  above") as the last option, or deliberately near-duplicate options.
- **State profiles.** Easy, borderline, ambiguous, irrelevant, long, or structured JSON states.

A quarter of the question sets of spec version 2, and half of version 3's (see below), come from **qualitative coding
of free-text answers**: exit surveys, interview excerpts, patient diaries, comment cards, explanations of choices in
decision studies. These sets have 2–5 questions and 5–12 codes, mostly written as codebooks with decision rules,
inclusion and exclusion criteria and examples. Their states are written the way participants actually type: terse,
off-topic, borderline, ambiguous, or (version 3 only) **split** between two codes that fit about equally well. Such
items, on which human coders disagree, teach the head to lower its confidence exactly where a reviewer's attention is
needed.

The prompts are versioned files (`decisivelm/synth/prompts/generate_v1.yaml` to `generate_v4.yaml`, selected with
`--spec-version`). They differ in the catch-all styles they ask for, the share of coding sets, and how often states are
borderline, ambiguous or split. Version 4 adds plainly worded Scores with a middle level and an "open" state profile: a
state that bears on a Score's claim without settling it. The released heads (General v5c, version 0.1) are trained on
rows from versions 1, 2 and 4. Version 3 rows are not in their mix.

Hiring questions are limited to factual Nouls about what a document says (for example, whether a CV lists a
qualification). No question asks about or infers a personal attribute, and none asks whether to hire.

**Fan-out.** The same generator also writes question sets about real states taken from the seeds' train and validation
rows (`--fanout PATH:N`): contract provisions, Reddit comments, public comments and chatbot conversations. A fan-out
row keeps its parent's split, licence and tier.

**Filters.** A generated row is kept only if it passes all of these checks:

- every question matches its spec (type, option count, non-empty descriptions, unique option names);
- no question or option refers to another question or to other options, because each option is scored alone in its
  own *leaf* (section 6.2) and never sees them. The one exception is a requested catch-all in last place;
- the question set and the state are not exact duplicates of ones already kept;
- no question shares half or more of its word 6-grams (runs of six consecutive words), and no state half or more of its
  word 8-grams, with the evaluation and held-out rows: every non-`train` row in `data/processed/`, and every row of a
  general-head holdout.

**Labels.** Generated rows are labelled by the teacher (chapter 12) and calibrated on human gold: one temperature per
question type, fitted on an equal-sized sample of the `gold_calib` rows of five seeds (Banking77 and CLINC150
exploded, HelpSteer2, Civil Comments, GoEmotions). The Noul bias is 0 (`--balanced`), because a bias fitted on one
dataset mostly records its base rate. Score takes the Choice temperature (`--set score=choice`), because the only human
Score labels among the five are HelpSteer2's subjective ratings, whose temperature (9.8) would make every generated
scale near-flat. A question whose teacher answer put less than half its probability on valid labels gets no target.

## 13.5 What is held out

`configs/holdout.yaml` lists data the general head must never train on, and `decisivelm/data/mix.py` refuses to build
a mix that contains any of it:

- **The frozen held-out suite** (chapter 15): HelpSteer3, the evaluation portion of Measuring Hate Speech, CUAD's ten
  held-out clause types, and MASSIVE's test split in ten held-out languages.
- **The coding study.** A study of coding free-text responses (Alhasan, Diaz Monfort & Richardson, *Knowing which
  responses to check*, under review), whose four human coders labelled every response. It is a zero-shot test of a
  rule-heavy codebook, so nothing from the study (not even calibration data) reaches the head. Its data is private.
- **Development sets**: MultiNLI and Wikipedia Talk toxicity, two human-labelled tasks with several raters per item
  on which candidates are compared before any suite scoring. A task used to judge transfer must stay unseen, so none
  of either is trained on. Section 14.4 lists these and the other development sets.
- **Development benchmarks**: public test splits used to compare models zero-shot (chapter 14).
- Public test splits are never trained on, and validation and gold rows of every source never enter a training mix.

The suite's text was also audited for overlap with every trainable source in both directions (suite text found in
training data, and training text found inside a suite row), using word 8-grams (section 14.4). Overlapping suite rows
are left out (`scripts/overlap_audit.py`, `configs/suite_exclusions.txt`).

## 13.6 The mix

`scripts/build_mix.py` assembles the mix and its manifest:

- `--adapter general --tiers release`: every training row must be R tier and outside every general-head holdout;
- `--train PATH[:N]`: N `train` rows sampled from each source (all without N);
- `--cap MULTIPLIER=SHARE`: limits rows made by one multiplier to a share of each source. A multiplier is an operation
  that derives new rows from existing ones, such as schema explosion's `schema_explosion:drop_gold` (section 8.10),
  and each derived row records it in `lineage.multiplier`;
- `--eval PATH[:N]`: validation, `gold_calib` and `gold_test` rows of each source, used for early stopping,
  calibration and the report, never trained on;
- `--zero-shot PATH`: test rows only, so nothing from that dataset (not even calibration data) reaches the head.

It writes `data/mixes/<name>.jsonl` and `<name>.manifest.json`, which lists the sources, row counts by split and type,
and the file's SHA-256. The manifest is published with every release.

## 13.7 Training

The head is trained with `scripts/train_p3.py --adapter general` on Gemma 4 E4B:

| Setting | Value |
|---|---|
| Backbone | Gemma 4 E4B, frozen, bf16 |
| LoRA (low-rank adaptation, section 6.6) | rank 8, α 16, dropout 0.05, on the language-model decoder's q, k, v, o, gate, up and down projections only (`lora_target_regex` in `configs/model.yaml`) |
| Head | the scorer, a two-layer network that maps the query token's hidden state to one logit (section 6.5), trained from scratch, its input standardised per dimension on 2,000 training rows (`--standardize-rows 2000`); plus the three query-token rows |
| Learning rates | head and query rows 1e-4, LoRA 3e-5, 10% warm-up |
| Batching | 16 requests per optimiser step, packed into micro-batches of at most 4,096 tokens |
| Losses | Choice and Score soft cross-entropy divided by log n, Score plus EMD², Noul binary cross-entropy (chapter 6) |
| Validation | every 500 steps on a fixed sample of 1,500 validation rows |
| Selection | the snapshot with the lowest **calibration loss** (`--select cal_loss`): the validation loss after one temperature per type is fitted, so a snapshot is not penalised for over- or under-confidence that calibration corrects anyway |
| Calibration | temperatures per type and option-count bucket and the Noul bias, fitted on the mix's human-labelled `gold_calib` rows. No further scaling for unseen tasks: on a new task, calibrate on about 100 of your own items (section 10.6) |

LoRA adds a small trainable correction `B·A` to each chosen weight matrix of the frozen backbone. The rank is the inner
size of `B·A`, α scales the correction by α / rank, and dropout zeroes a random 5% of the correction's inputs during
training.

The standardisation divides each hidden-state dimension by its spread on the base model's states. Gemma 4's final
states have a few very large dimensions that would otherwise dominate the head's input.

The result is an adapter bundle (chapter 6): the LoRA weights, scorer, query-token rows, calibration, and the
backbone's model id and revision.

**The 26B A4B head.** It uses the same mix, losses, learning rates, batching, validation sample and selection rule, with
`--backbone 26b-a4b --dtype float32` in place of the E4B settings. The weights stay in bf16 and the computation is in
float32, because in bf16 the backbone's expert routing changed 19 to 22% of answers when only the packing of a request
changed (sections 2.1 and 6.9). LoRA covers the attention and shared MLP projections but not the 128 experts, which
gives about 9.3M adapter parameters (with the three query-token rows) and a 2.9M-parameter scorer, about 12M trained in
all. The run was seed 0 only, on one NVIDIA RTX PRO 6000 Blackwell Max-Q GPU. It took 1.38 s per training row and peaked
at 67.6 GB, with 4,096-token micro-batches and gradient checkpointing. The run was set for two epochs, at most 26,930
steps. The calibration loss on the validation rows stopped improving after step 15,000, and early stopping ended the run
at step 16,500, after about 101 hours. The bundle is the snapshot from step 15,000, the one with the lowest calibration
loss. The teacher, Gemma 4 26B A4B, is the same model as this head's backbone, so the head can share the teacher's blind
spots more directly than the E4B head does. Its results are on its model card
([docs/model-card-26b.md](../model-card-26b.md)).

## 13.8 Reproducing the general head

The steps below rebuild the head from public data and an open teacher. The teacher steps need a vLLM server serving
Gemma 4 26B A4B (chapter 12). Training needs one GPU with about 20 GB of memory for E4B with LoRA, at 4,096-token
micro-batches with gradient checkpointing, which is on by default (chapter 2). The 26B A4B head needs a GPU with about
72 GB free (67.6 GB was measured). Compare your sources and row counts with
the release's manifest.

**1. Fetch and convert the public datasets** (chapter 14):

```bash
uv run python scripts/fetch_dataset.py banking77 clinc150 helpsteer2 civil_comments goemotions ledgar multinli \
    wiki_toxicity helpsteer3 measuring_hate_speech cuad massive vizwiz_vqa
uv run python scripts/convert_dataset.py banking77 clinc150 helpsteer2 civil_comments goemotions ledgar multinli \
    wiki_toxicity helpsteer3 measuring_hate_speech cuad massive vizwiz_vqa
```

**2. Explode the intent seeds:**

```bash
uv run python scripts/explode.py data/processed/banking77.jsonl --question intent
uv run python scripts/explode.py data/processed/clinc150.jsonl --question intent
```

**3. Label the seeds' calibration rows with the teacher**, for calibrating its answers on generated rows:

```bash
uv run python scripts/label_teacher.py --splits gold_calib --data \
    data/processed/banking77_explode.jsonl data/processed/clinc150_explode.jsonl data/processed/helpsteer2.jsonl \
    data/processed/civil_comments.jsonl data/processed/goemotions.jsonl
```

**4. Generate question sets and states**, once per spec version, with fan-out on real states (General v5c uses versions
1, 2 and 4):

```bash
uv run python scripts/synth_generate.py --name synth_v1 --spec-version 1 --sets N --states 6
uv run python scripts/synth_generate.py --name synth_v2 --spec-version 2 --sets N --states 6 \
    --fanout data/processed/ledgar.jsonl:N data/processed/goemotions.jsonl:N \
             data/processed/civil_comments.jsonl:N data/processed/helpsteer2.jsonl:N
uv run python scripts/synth_generate.py --name synth_v4 --spec-version 4 --seed 4 --sets 3000 --states 6
```

`--sets` is the number of question sets (each with `--states` states) and `--fanout PATH:N` the number of real states to
write questions about. Each question set and each fan-out state is one job. For the text rows that General v3d and v5c
share, `synth_v1` ran 15,000 jobs and wrote 39,563 rows, and `synth_v2` ran 12,000 jobs and wrote 40,962 rows
(`results/suite_overlap.md`). Each job's spec is seeded from `--seed` and its job number, so a rerun writes the same
specs. Generation is resumable and can stop at a given local hour (`--stop-at`) to leave a shared server free.

**5. Label the generated rows and calibrate them into targets:**

```bash
for s in synth_v1 synth_v2 synth_v4; do
  uv run python scripts/label_teacher.py --data data/processed/$s.jsonl --splits train val
  uv run python scripts/teacher_targets.py --data data/processed/$s.jsonl --splits train val --balanced \
      --set score=choice --calib data/processed/banking77_explode.jsonl data/processed/clinc150_explode.jsonl \
      data/processed/helpsteer2.jsonl data/processed/civil_comments.jsonl data/processed/goemotions.jsonl
done
```

**6. Build the release mix.** The command that built General v5c, the mix of both 0.1 heads, is below. Check its sources
and counts against the release's manifest:

```bash
P=data/processed
uv run python scripts/build_mix.py --name general_v5c --adapter general --tiers release \
  --train $P/banking77.jsonl:5000 $P/banking77_explode.jsonl:8000 $P/clinc150.jsonl:6000 \
          $P/clinc150_explode.jsonl:10000 $P/helpsteer2.jsonl:12000 $P/civil_comments.jsonl:15000 \
          $P/goemotions.jsonl $P/ledgar.jsonl:12000 $P/synth_v1_teacher.jsonl $P/synth_v2_teacher.jsonl \
          $P/synth_v4_teacher.jsonl $P/vizwiz_vqa.jsonl:10000 \
  --cap schema_explosion:drop_gold=0.05 \
  --eval $P/banking77.jsonl $P/clinc150.jsonl $P/helpsteer2.jsonl $P/civil_comments.jsonl \
         $P/banking77_explode.jsonl:600 $P/clinc150_explode.jsonl:600 $P/goemotions.jsonl:1500 $P/ledgar.jsonl:1000 \
         $P/synth_v1_teacher.jsonl:1000 $P/synth_v2_teacher.jsonl:1000 $P/synth_v4_teacher.jsonl:1000 \
         $P/vizwiz_vqa.jsonl:1000 \
  --zero-shot $P/llm_response_coding.jsonl
```

The resulting mix has 215,433 training rows and 12,522 human-labelled `gold_calib` rows (SHA-256 `b2ecb006…`). Its text
rows are those of General v3d, an earlier head trained on text only, and it adds 10,000 VizWiz-VQA photos. Of the text
training rows, 78,422 come from teacher-written question sets and 14,257 from fan-out on the seeds' states. The coding
study's rows are `gold_test` only and private. If you do not have its data, leave out `--zero-shot`.

**7. Train:**

```bash
uv run python scripts/train_p3.py --data data/mixes/general_v5c.jsonl --adapter general --backbone e4b \
    --max-tokens 4096 --lr-lora 3e-5 --warmup-frac 0.1 --lr-head 1e-4 --standardize-rows 2000 \
    --epochs 2 --eval-every 500 --limit-val 1500 --select cal_loss --name general_v5c_e4b_r8_seed0
```

For the 26B A4B head, use `--backbone 26b-a4b --dtype float32` and `--name general_v5c_26b-a4b_r8_seed0`, which writes
its bundle to `bundles/general_v5c_26b-a4b_r8_seed0/`. The E4B command writes the bundle to
`bundles/general_v5c_e4b_r8_seed0/` and calibrated predictions for the mix's `gold_calib` and `gold_test` rows to
`data/predictions/`. Steps 8 and 9 score this bundle, and the release ships it.

**8. Check it on the development sets** before any suite scoring. Run the bundle on MultiNLI's and Wikipedia Talk
toxicity's gold rows with `scripts/predict_bundle.py`, and compare its review readiness with the base model's (P0).

**9. Score it against the release criteria** (chapter 15). `scripts/score_suite.py` compares the bundle with two
baselines, P0 (`--teacher gemma4_e4b`) and the teacher (`gemma4_26b_a4b`, the labeller's default). It needs three sets
of predictions first (section 15.4):

- the bundle's on each suite set, from `scripts/predict_bundle.py`;
- each baseline's on each suite set, from `scripts/label_teacher.py`;
- each baseline's on the mix's `gold_calib` rows, on which the baselines are calibrated the way the bundle is.

For the last, label the mix file itself. `--calib-data` then finds the labels under the mix's name
(`data/teacher/general_v5c__<teacher>.jsonl`):

```bash
uv run python scripts/label_teacher.py --teacher gemma4_e4b --data data/mixes/general_v5c.jsonl --splits gold_calib
uv run python scripts/label_teacher.py --data data/mixes/general_v5c.jsonl --splits gold_calib
uv run python scripts/score_suite.py --candidate general_v5c_e4b_r8_seed0 --calib-data data/mixes/general_v5c.jsonl
```

For the 26B A4B head, pass `--candidate general_v5c_26b-a4b_r8_seed0`. `score_suite.py` reads the backbone from the
bundle, so for this head P0 is the prompted 26B A4B, the same model as the teacher, and the `gemma4_e4b` labels are not
needed. Criterion 4 also compares the bundle with each seed's specialist, an E4B fine-tune trained on that seed alone
(the 26B A4B head is compared with the same ones), whose predictions `score_suite.py` reads from the paths in
`configs/release_seeds.yaml`. Criterion 2 (calibration) is judged after the per-task step (section 10.6), which
`score_suite.py` runs itself. On each suite set it fits each method's Choice and Score scales on 100 rows chosen by a
hash of their ids, and it scores every criterion on the remaining rows.

Next: [14. Datasets](14-datasets.md)
