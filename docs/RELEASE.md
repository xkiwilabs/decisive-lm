# Release criteria for the general head

Written 2026-10-17, before any candidate has been scored against them, so the bar cannot move to fit a result. Any
later change is added to the change log at the end with its date and reason; earlier releases keep the criteria in
force when they were made.

The question each criterion asks: **is someone who uses the general head better off than if they had prompted the base
model it is built on?** Its main value is review readiness: whether its confidence says which answers a person should
check. Accuracy comes second. Metric definitions are in [guide chapter 11](guide/11-evaluation.md) and `decisivelm/eval.py`.

## What the general head is for

The general head should be **very good at every task**: better than prompting the base model everywhere, close to a
task-specific fine-tune on the tasks it was trained on, and as good as that fine-tune at saying which answers to check.
**Excellence on one task is what fine-tuning is for**: a fine-tune spends all of its capacity on one task's labels and
conventions, while the general head shares one adapter across all of them. The criteria below gate on *very good*.
*Exceptional* is a target, reported on every model card but not a gate:

- within noise of the specialist on every seed (the paired CI of the primary metric's difference reaches 0);
- on the frozen suite, accuracy not significantly below the 26B teacher, and review readiness at least as good.

## Versions

- **0.x (preview)**: the general head is *never worse* than the frozen backbone prompted (P0) on the frozen held-out
  suite, and is calibrated well enough to set a review threshold on.
- **1.0**: the general head *beats* P0 on review readiness and passes the stricter calibration gate (criteria 1
  and 2 at their 1.0 levels, below) on the frozen held-out suite.

The first public release of the repository is the first general head that passes the 0.x bar. Until then, the
repository and its user fine-tune path are developed in private.

## The frozen held-out suite

Gates are scored on a suite that no training, selection or tuning decision ever sees: no mix, no stopping rule, no
calibration fit and no prompt change is chosen by looking at it. The one fit made on suite rows is criterion 2's
per-task step: two numbers per method, fitted on 100 rows of each set that are never scored. Each release candidate is scored on it **once**, after
it is chosen; a candidate that fails is not tuned on the result and rescored.

- Contents, confirmed 2026-10-20. All are public, CC BY 4.0 and human-labelled as published (no new
  rating); tasks with ≥ 3 raters per item carry the review-readiness criteria. Converters in
  `decisivelm/data/converters/`; each set is held out in `configs/holdout.yaml`.
  - **HelpSteer3** preference, a fixed random 6,000 of its 40,476 response pairs (3 annotators per pair): which of two
    responses is better and by how much (Score, the annotators' six levels). LLM judging beyond HelpSteer2. Its card
    says only the three most-agreeing of up to five annotations were released, so its disagreement is understated.
  - **Measuring Hate Speech**, eval portion: 20% of the comments with ≥ 3 raters, by comment id; nine survey items as
    Scores (0–4) and hate speech as a Noul. Calibration against real disagreement.
  - **CUAD**, the 10 held-out clause types (Cap on Liability, Audit Rights, Change of Control, Non-Compete, Uncapped
    Liability, Covenant Not to Sue, Liquidated Damages, Non-Disparagement, Most Favored Nation, Source Code Escrow) as
    Nouls on contract sections of ≤ 10,000 characters; one expert label each. Unseen questions in a known domain.
  - **MASSIVE**, the test split of the 10 held-out languages (Polish, Urdu, Finnish, Georgian, Amharic, Swahili,
    Tamil, Korean, Vietnamese, Tagalog): one Choice over 60 intents. Zero-shot language transfer.
- **No overlap with trainable data.** `scripts/overlap_audit.py` compares the suite's text with every trainable row
  in both directions (`results/suite_overlap.md`); overlapping suite rows are left out
  (`configs/suite_exclusions.txt`: 582 CUAD sections, mostly text shared with LEDGAR, and 59 others). The generator
  filters new synthetic rows against the suite. The audit is rerun before a candidate is scored.
- The suite, its exclusions and the release criteria are fixed before the first candidate is scored.
- **Development sets are not gates.** The coding task (`llm_response_coding`) and the public benchmark sets have been
  used to choose between General v0–v3c and so no longer test a candidate fairly. They are reported on every model
  card as development results, labelled as such. General v3c is the last mix chosen with the coding task: from then
  on it is reported but not used to choose, because a single study's codebook would otherwise shape the mix. Later
  mixes are chosen on criterion 4 (the seeds' own test sets) and the public benchmark sets.

## Criteria (all must hold)

On each task of the frozen suite, comparing the candidate with P0 using paired item-level bootstrap 95% CIs (1,000
resamples):

1. **Review readiness** (tasks with ≥ 3 raters per item):
   - 0.x: Spearman ρ(top p, per-item human agreement) and tie AUROC not significantly below P0 (the CI of the
     difference does not lie wholly below 0); 1.0: both beat P0 (CI excludes 0);
   - within-category ρ > 0 in every consensus category with ≥ 25 items (0.x: or the lowest at least P0's);
   - dynamic range of top p ≥ 0.2 (0.x: or at least P0's);
   - a confidence-guided review of the 20% least confident answers catches at least as many errors as the same review
     of P0's answers (0.x: not significantly fewer; 1.0: more), and beats unaided accuracy (CI excludes 0).
2. **Calibration, after the per-task step**: 100 rows of each suite set are set aside, and each method's Choice
   temperatures and its Score temperatures are each scaled by one factor fitted on its own answers to them, the step a
   user takes on about 100 labelled items of a new task ([guide §10.6](guide/10-calibration.md)). Then top-label ECE
   (15 bins) on hard human labels ≤ 0.10 on every task (0.x); the upper bound of its bootstrap CI < 0.07 (1.0). Brier
   after the step not significantly worse than before any calibration (the CI of the difference does not lie wholly
   above 0). ECE and Brier without the step (zero-shot) are reported on every model card, not gated.
3. **Accuracy**: the primary metric per type (Choice accuracy, Score MAE, Noul AUROC) not significantly worse than P0
   on any task (0.x); better on most tasks, worse on none (1.0).
4. **Close to the specialists on the seeds**: on each seed dataset's own gold_test and question type, against the
   reference (the previous release; for the first release, the specialist trained on that seed alone), with paired
   bootstrap 95% CIs:
   - accuracy: the primary metric at most 0.03 below the reference (Choice accuracy, Noul AUROC), or at most 0.05
     levels above it (Score MAE);
   - calibration: top-label ECE ≤ 0.10, or not significantly above the reference's (the CI of the difference does not
     lie wholly above 0);
   - review readiness, on seed tasks with ≥ 3 raters per item: ρ(top p, agreement), tie AUROC and the errors caught by
     a 20% review not significantly below the reference (the CI of the difference does not lie wholly below 0).
5. **Contract**: 100% conformance on every eval item, and the independence and order tests ([guide §11.7](guide/11-evaluation.md)) pass on the released
   bundle on the real backbone, at the tolerances recorded in the bundle.
6. **Provenance**:
   - the release mix is R tier only, with its manifest (sources, licences, row counts, sha256) published;
   - the teacher and backbone pass the model-origin rule ([guide §16.2](guide/16-provenance-licences-and-responsible-use.md));
   - the clean-room canary and scanner pass;
   - no output of Jev/TypeSafe was used in any step ([guide §16.1](guide/16-provenance-licences-and-responsible-use.md)).
7. **Snapshot audit**: the exported public tree passes the release audit: allowlisted paths
   only, the test suite green, and no secrets, private paths, private datasets or restricted-tier results.
8. **A better starting point for fine-tunes** (1.0; reported for 0.x): on two development tasks outside the training
   mix, adapters fine-tuned from the general head on 100 and on 300 labelled items, three seeds each, are compared
   with adapters fine-tuned from the base model on the same items. The general head's are not significantly worse at
   either size on the primary metric or on review readiness (where it can be measured), and better at 100 items on at
   least one of them.

`scripts/score_suite.py` scores criteria 1–4 and writes `results/suite_<candidate>.md`, refusing to overwrite one, so a
candidate is scored once. How it reads the criteria:
- a **task** is one suite set's questions of one type (Measuring Hate Speech has two: its nine Scores and its Noul), so
  the suite has five tasks; a metric over several questions is the mean of the per-question values;
- **the candidate's calibration** is its bundle's: temperatures fitted on its mix's gold_calib (from General v3d on,
  with no new-task scale; change log 2026-10-25);
- **P0 and the teacher** are calibrated the way the candidate is (`decisivelm.calibrate.fit_calibration`), on the same
  gold_calib rows of the candidate's mix (a candidate scored with a new-task scale, General v3c, had it fitted for
  them too, on their own answers to the same held-out rows);
- **the per-task step** (criterion 2) is the same for every method: its Choice and Score factors are fitted like the
  new-task scale (`decisivelm.calibrate.fit_new_task_scale`), on the method's own calibrated answers to 100 rows of
  the set with the lowest hash of their ids; Noul is left as it is, as the new-task scale leaves it. Rows that share
  a source item with those 100 (a MASSIVE utterance in its other languages) are set aside too. Every criterion is
  scored on the remaining rows: calibration after the step, everything else before it (the step does not change
  which answer is chosen);
- a **consensus category** is one question's consensus level (the raters' unique plurality); hard labels are that
  level, or for a Noul a human share ≥ 0.5;
- **dynamic range** is taken over agreement levels with ≥ 25 items (items rated by hundreds of people each have a level
  of their own);
- review readiness uses the items with ≥ 3 raters, on tasks where most items have them (HelpSteer3, Measuring Hate
  Speech's Scores);
- criterion 4 is scored the same way on each seed's gold_test, the reference taking P0's place; its margins are in
  `configs/release_seeds.yaml`. `--seeds-only` scores criterion 4 alone, for choosing between development mixes.

## What every release publishes

- **Model card**: every number for the suite, the development sets and the seeds, beside P0, the 26B teacher, the
  specialist fine-tunes and, where they exist, published aggregate figures of other systems. Each criterion is marked
  pass or fail, *including where the model loses*.
- The mix manifest, the bundle's base revision, pinned library versions, and calibration temperatures.
- A tag shared by the public repository snapshot (`vX.Y`) and the Hugging Face model revision, with the private
  development commit it was exported from.

## Data

Data never goes in git, here or in the public repository. Each release publishes its data on Hugging Face as a dataset
repository (`decisive-lm-data`), tagged `vX.Y` like the model, or links to where it came from:

- **Linked, not re-hosted:** every public seed and benchmark dataset. `configs/datasets.yaml` pins each file's source
  (Hugging Face revision or commit URL), `scripts/fetch_dataset.py` downloads it with a sha256 manifest, and
  `scripts/convert_dataset.py` rebuilds our rows, so anyone can reproduce the inputs exactly.
- **Published:**
  - the teacher-written rows (states and question sets), with the teacher's calibrated probabilities for every
    question and the calibration that produced them;
  - the release mix manifest: source row ids, sources, licences, row counts and sha256 (criterion 6), from which the
    human-labelled rows are rebuilt from their sources;
  - per-item predictions of the released head, P0 and the teacher on the public benchmarks and the frozen suite.
- **Before publishing**, checked and recorded on the dataset card:
  - every published row is R tier; rows written about a seed's state (fan-out) keep that seed's licence and are
    credited (CC BY 4.0 needs attribution);
  - the teacher's licence allows its outputs to be published;
  - no restricted-tier or private data (private image sets, X-tier datasets, the owner's coding study until the owner
    releases it), no holdout row, no Jev/TypeSafe output;
  - the export checks (criterion 7) pass on the dataset card and manifest.

## Change log

- 2026-10-17: first version.
- 2026-10-17: named the 10 MASSIVE held-out languages (`configs/holdout.yaml`); the suite is still proposed.
- 2026-10-20: the frozen suite confirmed, with its exact contents and the overlap exclusions.
- 2026-10-20: added "Data": what each release publishes on Hugging Face, what it links to instead, and the checks.
- 2026-10-21: how `scripts/score_suite.py` reads criteria 1–4 (tasks, baseline calibration, consensus categories,
  dynamic range), written before any candidate was scored.
- 2026-10-22: added "What the general head is for" (very good at every task is the gate, exceptional the reported
  target). Criterion 4 rewritten: the fixed 0.01 tolerance was smaller than the sampling noise on the seeds' test sets
  (1,000–1,900 items; one standard error of accuracy is about 0.01), so it failed candidates on noise. It became a
  0.03 margin (0.05 levels for Score MAE), with calibration and review readiness held to the specialist's level, which
  is stricter where the project's claim lies. Written after General v3b and v2-ep2 had been checked on the seeds
  (GoEmotions sentiment 0.022 and 0.027 below the specialist, with lower ECE and higher tie AUROC), and before any
  candidate was scored on the suite. Calibration on the seeds is ECE ≤ 0.10 or not significantly above the
  specialist's, because the HelpSteer2 specialist's own ECE (0.105) is above 0.10. Added criterion 8 (a better starting point for fine-tunes). The coding task stops
  choosing mixes after General v3c.
- 2026-10-23: the general head's calibration gains new-task scales (one factor on the Choice temperatures and one on
  the Score temperatures, fitted against the human majority answer on MultiNLI's gold_calib, a task outside the mix,
  whose gold rows ask the same judgement as a Choice, a Noul and a 3-level Score), and the baselines get the same.
  Chosen on development sets after General v3c was trained and before any candidate was scored on the suite: v3c's
  confidence was 0.15–0.40 below its accuracy on all seven public benchmarks and the coding task, because its
  temperatures matched rater shares on its training tasks. Fitted on MultiNLI alone, the Choice scale (0.59) brought
  every benchmark's ECE to ≤ 0.09 and the coding task's from 0.35 to 0.13. The Score scale is 0.95: applying the Choice
  scale to Scores made MultiNLI's Score ECE worse (0.045 → 0.131). Noul is unchanged (ECE 0.046 there). The seeds' ECE
  stays ≤ 0.10. MultiNLI is held out of the general mix while it plays this role.
- 2026-10-24: references to the internal design spec replaced by their definitions; no criterion changed. The 1.0
  review-readiness and calibration gates were cited as the spec's; they are the 1.0 levels of criteria 1 and 2 as
  written above, which `scripts/score_suite.py` checks. Where the spec's wording and the code differed, the code was
  already what criteria 1–3 state: the review criterion also compares the errors caught in a 20% review with P0's, and
  within-category ρ, dynamic range and the Brier comparison are checked on point values, without a CI. Written after
  General v3c (`_nt`) was scored on the suite; it changes no criterion, so no result changes.
- 2026-10-24: **General v3c (`general_v3c_e4b_r8_seed0_nt`) scored on the suite: it fails 0.x**
  (`results/suite_general_v3c_e4b_r8_seed0_nt.md`). It is not released, and is not tuned on the result and rescored.
  - Passes: accuracy is never significantly below P0's and is better on 3 of 5 tasks. ECE is 0.013–0.058 on the four
    English tasks (P0: 0.021–0.252). Criteria 4 and 8 pass.
  - Fails: MASSIVE's ECE is 0.221; P0's is 0.196 and the teacher's 0.147, so every method is overconfident in the 10
    held-out languages. The English-fitted new-task Choice scale sharpened it further (Brier 0.500 before calibration,
    0.540 after). On Measuring Hate Speech's Scores, ρ(top p, agreement) and tie AUROC are significantly below P0's
    (0.092 vs 0.163; 0.537 vs 0.562), and the dynamic range is too (0.100 vs 0.138).
  - Two criteria changed, from the next candidate on, as written above:
    - The Brier comparison had no allowance for noise. It failed four tasks on differences as small as 0.0002 (HelpSteer3
      0.7822 vs 0.7820). It is now a paired CI, like the other comparisons.
    - At 0.x ("never worse than P0"), the absolute floors on within-category ρ and dynamic range could fail a candidate
      that is better than P0. On HelpSteer3 no method meets them (dynamic range: candidate 0.014, P0 0.012, teacher
      0.025). At 0.x they now also pass when the candidate is at least P0's; 1.0 keeps the floors.
  - General v3c fails 0.x under either version. The changes are written after its result was seen, so they are
    recorded with that result. The suite has now been seen once. The next candidate's changes (language calibration,
    Score review readiness) are chosen on development data only, but they are known to be motivated by this result,
    and its model card says so.
- 2026-10-24: **criterion 2 is judged after a per-task step** (fitting two numbers per method on 100 rows of each
  suite set, as above), from the next candidate on. Zero-shot calibration is still reported, not gated.
  - Why: no method we have is calibrated zero-shot on every new task. General v3c is underconfident on the seven
    public benchmarks and the coding task, and overconfident on MASSIVE; so are P0 and the teacher on MASSIVE (ECE
    0.196 and 0.147). One general correction cannot fix errors that go both ways. A user who sets a review threshold
    on a new task calibrates on a sample of their own labels first, and that is what the criterion now measures.
  - The fix first planned, a separate scale for text that is not English, was checked on development data and
    dropped. On SIB-200 (`configs/holdout.yaml`: 7 topics, the same 1,004 sentences in 195 languages, the suite's 10
    left out) General v3c is underconfident, not overconfident, outside English: accuracy 0.720, mean confidence 0.635.
    It is overconfident in 8 of 195 languages, and underconfident by 0.05–0.12 in close relatives of every suite
    language. A scale fitted there would make MASSIVE worse, so MASSIVE's overconfidence is not a language effect we
    can measure without the suite.
  - Written after General v3c's suite result was seen, and motivated by it. The per-task step is the same for every
    method, and nothing on the 100 rows is chosen by hand.
- 2026-10-25: **the candidate is scored without the new-task scale**, from General v3d on; P0 and the teacher likewise.
  - Why: the per-task step (criterion 2) now does the scale's job on each suite set, and on the tasks the head was
    trained on the scale makes it overconfident. On the seeds, General v3d with the scale fails criterion 4 on LEDGAR
    (ECE 0.108; specialist 0.032); without it every seed passes (LEDGAR 0.043). On the development sets (MultiNLI,
    HelpSteer2) every criterion gives the same verdict either way, except HelpSteer2's lowest within-category ρ, which
    passes without the scale because P0's value moves too (it is calibrated the same way).
  - The cost: without the scale the head is underconfident zero-shot on new tasks (MultiNLI Choice ECE 0.159, against
    0.013 with it). Zero-shot calibration is reported, not gated, and the model card tells users to calibrate on about
    100 items of their own task.
  - Written after General v3c's suite result and v3d's development results were seen, and before v3d was scored on
    the suite.
- 2026-10-25: **General v3d (`general_v3d_e4b_r8_seed0`, no new-task scale) scored on the suite: it fails 0.x**
  (`results/suite_general_v3d_e4b_r8_seed0.md`), on one task. It is not released, and is not tuned on the result and
  rescored. The baselines were calibrated on `general_v3c.jsonl`'s gold_calib rows, which are the same rows as
  `general_v3d.jsonl`'s (checked by id).
  - Passes: calibration after the per-task step on all five tasks (ECE 0.015–0.080; MASSIVE 0.023, from 0.221 for
    v3c), accuracy not below P0's on any task and better on 3 of 5, criterion 4 on every seed, and every review
    criterion on HelpSteer3.
  - Fails: on Measuring Hate Speech's Scores, review readiness is significantly below P0's: ρ(top p, agreement) 0.142
    vs 0.201, tie AUROC 0.556 vs 0.578, errors caught in a 20% review 0.225 vs 0.238. v3c's gaps were −0.071 and
    −0.025 on the first two; v3d's are −0.059 and −0.022.
  - Next: MultiNLI's Score did not predict this (v3d beats P0 on its review readiness), so a development set of the
    failing kind is added: Wikipedia Talk Labels, Toxicity (CC0; 5 levels from very toxic to very healthy, about 10
    raters per comment; `configs/holdout.yaml`, never trained on). It is a different dataset from Measuring Hate
    Speech, but chosen because of its result, which the next candidate's model card says.
  - On Wikipedia Toxicity's gold_test, v3d's ρ(top p, agreement) is −0.126 (P0 +0.059; teacher −0.153), and tie AUROC
    0.487 (P0 0.541). 61% of its comments have "neither toxic nor healthy" as the raters' plurality; v3d picks that
    level for 17% of them, P0 for 32%, the teacher for 9.5%. The teacher labels every synthetic row, so more synthetic
    data (General v3e, v3d plus a second synth_v4; stopped after 15 minutes of training) cannot teach that middle.
  - The next candidate, General v3f, adds human ratings of the kind that is missing: the Stanford Politeness Corpus
    (Stack Exchange; 5 raters per request on a two-sided scale; "neither polite nor impolite" is the plurality on
    21% of requests). Politeness is not the suite's task, though it is close to one of Measuring Hate Speech's ten
    items (respectful towards a group); its Wikipedia part is left out because the toxicity check is Wikipedia talk
    comments too.
- 2026-10-26: **General v3f (`general_v3f_e4b_r8_seed0`) does not fix it, and is not scored on the suite.** On
  Wikipedia Toxicity's gold_test, its ρ(top p, agreement) minus P0's is [−0.206, −0.134] (v3d [−0.222, −0.147]); it
  picks the middle level for 14% of the comments whose plurality is "neither" (v3d 17%, P0 32%). Its seeds all pass.
  The politeness raters seldom agree (two of them give the same level about 30% of the time), so the corpus carries
  little signal about when to be confident.
  - Adding human toxicity ratings from another source was considered and dropped: toxicity is already 13% of the
    mix's questions (Civil Comments), and a near-neighbour of Measuring Hate Speech's task would make a pass partly
    a matter of training for the test. The failure is general (on two-sided scales, confidence follows how extreme
    the text is rather than how much people agree), so the next candidate adds human ratings, with several raters
    per item, from several unrelated kinds of judgement, none about toxicity, hate or offence.
  - General v3g is v3d's recipe plus Humicroedit (how funny an edited news headline is, 0–3, about 5 graders per
    headline; CC BY 4.0) and the Conversational Sarcasm Corpus (how sarcastic a reply is, 1–6, 4 or 6 raters per
    reply; MIT). In both, how much raters agree varies widely from item to item. DynaSent (sentiment, CC BY 4.0) was
    left out because sentiment is one of Measuring Hate Speech's nine Scores. Politeness is moved to tier S (its
    requests are Stack Exchange posts, CC BY-SA), so it is not in v3g; it stays a development check, with Wikipedia
    Toxicity.
  - Licences (guide 14.1): a dataset whose authors state no licence and no restricting terms may now be used
    for training, listed in the model card as "no licence available"; third-party datasets are never re-hosted.
- 2026-10-27: **General v3g (`general_v3g_e4b_r8_seed0`) does not fix it either, and is not scored on the suite.** On
  the two new sets' gold_test it learned what was asked: ρ(top p, agreement) minus P0's is [+0.212, +0.371] on
  Humicroedit and [+0.138, +0.345] on the sarcasm corpus. On the held-out checks it did not carry over: Wikipedia
  Toxicity [−0.232, −0.160] (v3d [−0.222, −0.147]), Politeness [−0.082, +0.083] with worse calibration. Early stopping
  chose step 3,500, and it fails criterion 4 on Banking77, GoEmotions and LEDGAR. Three candidates (v3e, v3f, v3g)
  agree: the head learns when to be unsure on the kinds of rating it is trained on, not on a two-sided social scale
  it has not seen.
- 2026-10-27: **the bar is changed after seeing a result, and said so here.** The owner decided to release General
  v3d (`general_v3d_e4b_r8_seed0`, scored once on the suite, change log 2026-10-25) as the first 0.x preview,
  although it fails criterion 1 on one task: Measuring Hate Speech's Scores, where its confidence is a worse guide
  to rater disagreement than P0's. It passes every other 0.x criterion. Its model card and the paper state that
  failure in the summary, not only in the tables: on two-sided social rating scales the head has not been trained
  on, use P0's confidence or fine-tune on your own ratings, which the Humicroedit and sarcasm results show works.
  This exception is for v3d only; later releases are held to the criteria as written. Human ratings on two-sided
  scales from the rating app (a fresh check set, then training data) are planned for 0.2.
- 2026-10-27: **0.2 development starts; v3d is frozen as 0.1.** The two same-task sources held back until the first
  release (the suite-task rule) may now train: HelpSteer2-Preference (which of two responses is better and by how
  much, 3–5 annotators per pair, each pair also with the responses swapped) and Measuring Hate Speech's training
  portion (its eval family stays held out). From 0.2, HelpSteer3 and Measuring Hate Speech are reported as **seen
  tasks** in every suite report, and criterion 1 on them no longer measures transfer. Transfer to two-sided scales
  the head has not seen is judged on the rating app's two-sided check set (600 public texts, five raters each) once
  it is rated, with Wikipedia Toxicity, Humicroedit and the sarcasm corpus as development checks. The first
  candidate, General v4a, is v3d's recipe plus both sources, with Measuring Hate Speech capped at 6,000 comments
  (about a fifth of the Score questions) so the head is not tilted towards hate speech.
