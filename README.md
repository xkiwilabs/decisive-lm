# DecisiveLM

**An open, local decision model on Gemma 4 that tells you which of its answers to check.**

> **Status:** the toolkit is usable now. The first general head is not released yet: it will be published only when
> it passes the release criteria written in advance in [docs/RELEASE.md](docs/RELEASE.md).

DecisiveLM is a *System One* model. You give it a state — a message, a document, a JSON record or an image — and a
set of typed questions about it. In a single forward pass it returns answers drawn only from the options you supplied,
with probabilities calibrated against human judgement. It runs on your own hardware, its weights are open, and you can
fine-tune it on your own decisions.

Accuracy is only half of what makes such a model useful. The other half is whether its confidence can be trusted to
say where a person should look: high where human coders would agree, low where they would disagree or where the
model is likely to be wrong. DecisiveLM is trained on the *spread* of human ratings, not their majority vote, and is
judged first on that *review readiness*, compared against human coders and against simply prompting the same base
model.

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

Three question types cover every decision:

| Type | Asks | Returns |
|---|---|---|
| **Choice** | Which one of these 2–255 unordered options? | the option, a probability for each option, a confidence |
| **Score** | Which level of this ordered 2–10 level scale? | an expected level, a probability for each level, a confidence |
| **Noul** | Is this statement true? | a probability that it is |

Every question is answered independently, and every option is judged on its own: adding a question, removing one, or
reordering options never changes another answer. That makes it safe to ask many questions at once, at little extra
cost.

## The models

| Model | What it is | Use it when |
|---|---|---|
| **General head** (`decisivelm-e4b-general`) | A LoRA adapter and scoring head on Gemma 4 E4B, trained on human-labelled public datasets and on teacher-labelled questions across many domains, calibrated on human labels. To be published on Hugging Face when a candidate passes the release criteria. | You want answers to new questions with no training: classification, moderation, routing, quality judgements, coding of free-text answers. |
| **Your own fine-tune** | The same architecture trained on your labelled examples with the toolkit, on your own machine. | Your task has its own conventions or long, rule-heavy instructions, and you have a few hundred labelled items, ideally rated by more than one person. |
| **Backbones** | Gemma 4 **E4B** (default; small and fast) and Gemma 4 **26B A4B** (a larger mixture-of-experts model). Each adapter is tied to the backbone it was trained on. | E4B for most uses; 26B A4B where accuracy matters more than speed and memory. |

Each model is compared, in every report, with two baselines: the same Gemma 4 model prompted to answer directly
(*P0*), and the larger Gemma 4 26B A4B model used as the teacher.

## Results

Each release's results, against P0 and the teacher on a frozen held-out suite, are in the
[model card](docs/model-card.md). The 0.1 preview is never less accurate than P0 on the suite and is well calibrated
after a calibration step on 100 items of each task. Its one known shortfall: on two-sided social rating scales it was
not trained on, its confidence is a worse guide to which answers to check than P0's, so calibrate each level or
fine-tune on a few hundred of your own ratings for such a task. Fine-tune examples on images and audio (photo quality, emotion in
the voice, whether an object is alive) are in the guide. The model card's roadmap lists what comes next, including images and audio.

## Getting started

```bash
git clone https://github.com/xkiwilabs/decisive-lm && cd decisive-lm
uv sync                     # Python 3.13 environment with the pinned dependencies
uv run pytest               # the test suite runs on CPU in under a minute
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
| [9. Fine-tuning for your task](docs/guide/09-fine-tuning-for-your-task.md) | Training, calibrating and evaluating an adapter on your own labelled data |
| [10. Calibration](docs/guide/10-calibration.md) | How confidence is fitted to human labels, and how to refit it |
| [11. Evaluation](docs/guide/11-evaluation.md) | Accuracy, calibration and review-readiness metrics, invariance checks |
| [12. Teacher labelling](docs/guide/12-teacher-labelling.md) | Labelling with an open teacher on a vLLM server |
| [13. How the general head was trained](docs/guide/13-how-the-general-head-was-trained.md) | Data sources, the training mix, the recipe, and how to reproduce it |
| [14. Datasets](docs/guide/14-datasets.md) | Every public dataset used, with links, licences and roles |
| [15. Release criteria](docs/guide/15-release-criteria.md) | The frozen held-out suite and the bar each release must clear |
| [16. Provenance, licences and responsible use](docs/guide/16-provenance-licences-and-responsible-use.md) | Clean-room and model-origin rules, licences, intended and out-of-scope uses |
| [17. Reference](docs/guide/17-reference.md) | Every script, config and module |
| [Model card](docs/model-card.md) | The general head: intended use, training, evaluation, limitations |

Every dataset, with its licence, tier and role, is listed in [14. Datasets](docs/guide/14-datasets.md).

## How it was built

The general head is a LoRA adapter on Gemma 4 E4B's language-model layers, three reserved query tokens and a small
scoring head that turns the model's hidden state at each option into a score. It was trained on public
human-labelled datasets (intent detection, response quality, toxicity, emotion, contract provisions), on systematic
rewrites of their questions, and on question sets and states written by Gemma 4 26B A4B across many domains and
labelled by it with probabilities that were then calibrated on human labels. Its temperatures are fitted on
human-labelled data only. Each release is scored once, on a frozen suite of held-out human-labelled tasks that no
training or selection decision ever saw ([15. Release criteria](docs/guide/15-release-criteria.md)). The training mix
manifest, the teacher-labelled data and the per-item predictions are published with each release
([13](docs/guide/13-how-the-general-head-was-trained.md), [14](docs/guide/14-datasets.md)).

## Principles

- **Local and open.** Inference and training run on your own hardware. Code and adapter weights are Apache-2.0.
- **Calibrated on people.** Confidence is fitted to human labels, never to a teacher model's labels alone, and is
  evaluated against human agreement.
- **Independent answers.** No answer can influence another; this is guaranteed by the attention masks and checked by
  the tests.
- **Clean-room.** The request format follows a published API specification; nothing from any other system's outputs
  is used in any step ([16](docs/guide/16-provenance-licences-and-responsible-use.md)).
- **Model origin.** Every model in every role — backbone, teacher, baseline — is Google's Gemma 4.

## Licence and citation

Code and adapter weights: [Apache-2.0](LICENSE). The base model, Gemma 4, is released by Google DeepMind under
Apache-2.0; see its [model card](https://huggingface.co/google/gemma-4-E4B-it) for its terms. Datasets keep their own
licences ([14. Datasets](docs/guide/14-datasets.md)), and the attributions they require are in [NOTICE](NOTICE).

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
review). A [CITATION.cff](CITATION.cff) file gives the same entry; when the companion paper is on arXiv, cite it
instead.
