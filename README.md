# DecisiveLM

**An open, local decision model on Gemma 4 that tells you which of its answers to check.**

> **Status:** version 0.1, a preview. The toolkit is usable, and the first two general heads, on Gemma 4 E4B and 26B
> A4B, are published with it. Both were held to the release criteria in [chapter 15](docs/guide/15-release-criteria.md),
> and each model card explains the checks its head misses against the same backbone prompted to answer directly (P0).
> The [E4B card](docs/model-card.md) lists three. The main one is that, on one task, its review readiness is worse than
> P0's. The [26B A4B card](docs/model-card-26b.md) lists five, against a much stronger P0. The main one is accuracy on
> intents in ten languages, 5 points below P0's.

DecisiveLM is a *System One* model, in psychology's sense of quick, intuitive judgement
([1.1](docs/guide/01-introduction.md#11-what-a-system-one-decision-model-is)). You give it a state — a message, a
document, a JSON record, an image or an audio clip — and a set of typed questions about it. In a single forward pass
(one run of the model over the input, with no text generated) it returns answers drawn only from the options you
supplied. Its probabilities are calibrated against human judgement, so that on a calibrated task about 80% of the
answers it gives at 0.8 match the human label. Calibrating it for a new task takes about 100 labelled items
([guide 10.6](docs/guide/10-calibration.md)). It runs on your own hardware, its weights are open, and you can fine-tune
it on your own decisions.

A **head** is the small set of trained additions that turns Gemma 4 into this model. It has four parts: LoRA weights
(low-rank adaptation, small trained corrections to the language model's layers), three query-token embeddings (one
per question type, marking the position at which each option is judged), a scorer network that turns the model's
state at that position into a score, and calibration temperatures that set how spread out the probabilities are. A
head is saved and shipped as a directory called a **bundle**. It runs on the Gemma 4 model it was trained on, its
**backbone**, whose own weights stay as Google released them. A **general head** answers new questions with no
training of your own. There are two: the default one for Gemma 4 E4B, with about 20 million trained parameters
([model card](docs/model-card.md)), and one for Gemma 4 26B A4B, with about 12 million
([its model card](docs/model-card-26b.md)). In this README, "the general head" means the E4B one.

A decision model is useful only if its confidence also says where a person should look. That confidence should be
high where human coders would agree, and low where they would disagree or where the model is likely to be wrong.
DecisiveLM is trained on the *spread* of human ratings, not their majority vote. It is judged first on this property,
*review readiness*, against human coders and against the same base model prompted to answer directly.

## What it does

```python
from decisivelm import DecisiveLM

dlm = DecisiveLM.load("bundles/decisivelm-e4b-general")      # the head and the Gemma 4 E4B backbone it was trained on

response = dlm.system_one({
    "state": "Our API started returning 500s 20 minutes ago; we can't process orders.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payment or subscription issues",
                                    "technical": "Bugs or integration problems",
                                    "sales": "Pricing or account questions"}},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
        "frustration": {"type": "score", "instructions": "How frustrated does the customer appear?",
                        "criteria": ["Calm, just stating facts", "Frustrated but civil",
                                     "Very angry, strong language"]},
    },
})
response["answers"]["department"]   # {"type": "choice", "choice": "technical", "confidence": ..., "probabilities": {...}}
```

Every question is one of three types:

| Type | Asks | Returns |
|---|---|---|
| **Choice** | Which one of these 2–255 unordered options? | the option, a probability for each option, a confidence |
| **Score** | Which level of this ordered 2–10 level scale? | an expected level, a probability for each level, a confidence |
| **Noul** | Is this statement true? | a probability that it is |

Every question is answered independently, and every option is judged on its own. The attention masks, which set what
each part of the input can see, hide every other question and the order of the options from each answer. You can
therefore ask many questions about one state at once. The state is read once per request, and each question adds only
its own tokens. For example, four Choice questions of five options each add 240 tokens to a 1,000-token state, and with
E4B the whole request takes 75 ms on one NVIDIA RTX PRO 6000 GPU ([results/M0.md](results/M0.md)). In bf16 (16-bit)
arithmetic, which is the default, rounding still moves probabilities by about 0.03 on average (measured on a coding
fine-tune) and can flip an answer whose top two options are close. In fp32 (32-bit) the changes are below 0.0002
([11.7](docs/guide/11-evaluation.md#117-conformance-independence-and-order)).

## The models

| Model | What it is | Use it when |
|---|---|---|
| **General head** (`decisivelm-e4b-general`) | A head for Gemma 4 E4B, trained on human-labelled public datasets (photos with questions among them) and on questions across many domains labelled by a teacher (a larger model whose answer probabilities become training labels, here Gemma 4 26B A4B), and calibrated on human labels. Released as a preview ([model card](docs/model-card.md)). | You want answers to new questions with no training: classification, moderation, routing, quality judgements, coding of free-text answers. |
| **26B A4B general head** (`decisivelm-26b-a4b-general`) | The same training data on Gemma 4 26B A4B, run in float32 computation. Its confidence is a better guide to which answers to check than the E4B head's, at about 0.47 s for a 20-block request over HTTP (0.10 s for the E4B head) on a GPU with about 72 GiB free. It also runs on a Mac with 96 GB or more of unified memory, at about 1.9 s per request. Released as a preview ([model card](docs/model-card-26b.md)). | You have that GPU or Mac, and review readiness matters more than speed. |
| **Your own fine-tune** | The same architecture trained on your labelled examples with the toolkit, on your own machine. | Your task has its own conventions or long, rule-heavy instructions, and you have a few hundred labelled items, ideally rated by more than one person. |
| **Backbones** | Gemma 4 **E4B** (default; small and fast) and Gemma 4 **26B A4B** (a larger mixture-of-experts model). Each head is tied to the backbone it was trained on. | E4B for most uses; 26B A4B where accuracy matters more than speed and memory. |

Each model is compared, in every report, with two baselines: the same Gemma 4 model prompted to answer directly
(*P0*), and the larger Gemma 4 26B A4B model used as the teacher. For the 26B A4B head the two are the same model.

## Results

Each release's results, against P0 and the teacher on a frozen held-out suite, are in the model cards
([E4B](docs/model-card.md), [26B A4B](docs/model-card-26b.md)). The general head's preview is never less accurate than
P0 on the suite. After a calibration step that fits two numbers on 100 labelled items of each task (guide
[section 10.6](docs/guide/10-calibration.md)), its calibration error (ECE) is 0.05 or below on every suite task. Its
main known shortfall is on two-sided social rating scales it was not trained on. Such a scale runs between two opposite
poles through a neutral middle, for example "strongly respectful … neither … strongly disrespectful". On these scales
its confidence is a worse guide to which answers to check than P0's. For such a task, use task calibration (guide 10.6),
which fits a scale and an offset for each level on 100–300 of your own ratings, or fine-tune on them. It also misses two
smaller calibration checks, which the model card describes.

The 26B A4B head is more accurate than the E4B head on every suite task's primary metric, and on the rating scales of
the hate-speech survey (Measuring Hate Speech) its confidence ranks answers by rater agreement better than its P0's, the
check the E4B head fails. Against its P0 (the 26B A4B prompted to answer directly), it is 5 points less accurate on
intents in ten languages it was not trained on, and it misses four smaller checks
([its model card](docs/model-card-26b.md)).

Of images, both heads are trained only on photo questions (VizWiz-VQA, photos taken by blind people to ask a question).
Neither is trained on audio, and 26B A4B takes no audio input. Fine-tune examples on images and audio (photo quality,
emotion in the voice, whether an object is alive) are in the guide
([9.9](docs/guide/09-fine-tuning-for-your-task.md#99-images-and-audio)), and the model card's roadmap adds more images
and audio to the general heads in version 1.0.

## Getting started

```bash
git clone https://github.com/xkiwilabs/decisive-lm && cd decisive-lm
uv sync                     # Python 3.13 environment with the pinned dependencies
uv run pytest               # the test suite runs on CPU in about a minute
```

Then follow [2. Installation](docs/guide/02-installation.md) to fetch the Gemma 4 weights and
[3. Quickstart](docs/guide/03-quickstart.md) for your first request.

## Documentation

| Chapter | What it covers |
|---|---|
| [1. Introduction](docs/guide/01-introduction.md) | What a System One model is, why calibrated confidence matters, what the project provides |
| [2. Installation](docs/guide/02-installation.md) | Requirements, setup, Gemma 4 weights, running the tests |
| [3. Quickstart](docs/guide/03-quickstart.md) | Your first request, reading the response, predicting over a file |
| [4. Requests and responses](docs/guide/04-requests-and-responses.md) | The full request and response format, limits, validation, confidence methods |
| [5. Writing good questions](docs/guide/05-writing-good-questions.md) | Choosing a question type, self-contained options, states, worked examples |
| [6. How it works](docs/guide/06-how-it-works.md) | Blocks and query tokens, packed tree attention, the scoring head, adapter bundles |
| [7. Using the general head](docs/guide/07-using-the-general-head.md) | What it is for, using its confidence for review, limits, when to fine-tune |
| [8. Data format](docs/guide/08-data-format.md) | The canonical record, splits, lineage, licence tiers, converting your own data |
| [9. Fine-tuning for your task](docs/guide/09-fine-tuning-for-your-task.md) | Training, calibrating and evaluating a head on your own labelled data |
| [10. Calibration](docs/guide/10-calibration.md) | How confidence is fitted to human labels, and how to refit it |
| [11. Evaluation](docs/guide/11-evaluation.md) | Accuracy, calibration and review-readiness metrics, invariance checks |
| [12. Teacher labelling](docs/guide/12-teacher-labelling.md) | Labelling with an open teacher on a vLLM server |
| [13. How the general head was trained](docs/guide/13-how-the-general-head-was-trained.md) | Data sources, the training mix, the recipe, and how to reproduce it |
| [14. Datasets](docs/guide/14-datasets.md) | Every public dataset used, with links, licences and roles |
| [15. Release criteria](docs/guide/15-release-criteria.md) | The frozen held-out suite and the bar each release must clear |
| [16. Provenance, licences and responsible use](docs/guide/16-provenance-licences-and-responsible-use.md) | Clean-room and model-origin rules, licences, intended and out-of-scope uses |
| [17. Reference](docs/guide/17-reference.md) | Every script, config and module |
| [Model card](docs/model-card.md) | The general head: intended use, training, evaluation, limitations |
| [26B A4B model card](docs/model-card-26b.md) | The 26B A4B general head, compared with the prompted 26B A4B and the E4B head |

Every dataset, with its licence, tier and role, is listed in [14. Datasets](docs/guide/14-datasets.md).

## How it was built

Each general head's LoRA weights sit on its backbone's language-model layers (on 26B A4B, the attention and the shared
feed-forward layers, not the 128 experts). Each option, level or statement in a request ends with one of three reserved
query tokens (token ids the tokenizer never produces from text), and a small scoring network turns the model's hidden
state there, the vector its last layer produces at that position, into a score ([6](docs/guide/06-how-it-works.md)).
Both heads were trained on public human-labelled datasets (intent detection, response quality, toxicity, emotion,
contract provisions, questions about photos), on systematic rewrites of their questions, and on question sets and states
written by Gemma 4 26B A4B across many domains and labelled by it with probabilities that were then calibrated on human
labels. Each head's temperatures, one per question type and number of options, are fitted on human-labelled data only.
Each release is scored once, on a frozen suite of held-out human-labelled tasks that no training or selection decision
ever saw ([15. Release criteria](docs/guide/15-release-criteria.md)). The training mix manifest, the teacher-labelled
data and the per-item predictions are published with each release
([13](docs/guide/13-how-the-general-head-was-trained.md), [14](docs/guide/14-datasets.md)).

## Principles

- **Local and open.** Inference and training run on your own hardware. Code and head weights are Apache-2.0.
- **Calibrated on people.** Confidence is fitted to human labels, never to a teacher model's labels alone, and is
  evaluated against human agreement.
- **Independent answers.** No answer can influence another. The attention masks guarantee this, and the tests check
  it. Rounding still moves answers slightly, by less than 0.0002 in fp32 and by about 0.03 on average in bf16.
- **Clean-room.** The request format follows a published API specification. Nothing from any other system's outputs is
  used in any step ([16](docs/guide/16-provenance-licences-and-responsible-use.md)).
- **Model origin.** Every model in every role — backbone, teacher, baseline — is Google's Gemma 4.

## Licence and citation

Code and head weights: [Apache-2.0](LICENSE). The base model, Gemma 4, is released by Google DeepMind under Apache-2.0.
Its [model card](https://huggingface.co/google/gemma-4-E4B-it) gives its terms. Datasets keep their own licences
([14. Datasets](docs/guide/14-datasets.md)), and the attributions they require are in [NOTICE](NOTICE).

If you use DecisiveLM, please cite:

```bibtex
@misc{decisivelm,
  title  = {DecisiveLM: an open, calibrated System One decision model on Gemma 4},
  author = {Richardson, Michael J.},
  year   = {2026},
  note   = {https://github.com/xkiwilabs/decisive-lm}
}
```

The review-readiness measures follow Alhasan, Diaz Monfort & Richardson, *Knowing which responses to check* (under
review). A [CITATION.cff](CITATION.cff) file gives the same entry. When the companion paper is on arXiv, cite the paper
instead.
