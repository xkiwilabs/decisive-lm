"""Request and response models, following TypeSafe's public System One API shape (docs/guide/04).

`media` is our extension. Question ids are keys only; they are never shown to the model.
"""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Instructions and criteria may be a string, an object or an array (docs/guide/04 §4.3).
Content = str | dict[str, Any] | list[Any]

MAX_IMAGES = 4
MAX_AUDIO = 4   # clips per request, each cut at 30 s (decisivelm.media.MAX_AUDIO_SECONDS)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Content] = Field(min_length=2, max_length=255)


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=10)


class NoulCriteria(_Strict):
    yes: Content | None = None
    no: Content | None = None

    @model_validator(mode="before")
    @classmethod
    def _true_false_aliases(cls, data):
        # Accept {"true": ..., "false": ...} as well as {"yes": ..., "no": ...}.
        if isinstance(data, dict) and ({"true", "false"} & data.keys()):
            data = {("yes" if k == "true" else "no" if k == "false" else k): v for k, v in data.items()}
        return data


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: Content
    criteria: str | NoulCriteria | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class Media(_Strict):
    type: Literal["image", "audio"]
    path: str | None = None       # Python API only; server mode rejects paths (docs/guide/04 §4.5)
    data: str | None = None       # base64
    url: str | None = None

    @model_validator(mode="after")
    def _one_source(self):
        if sum(x is not None for x in (self.path, self.data, self.url)) != 1:
            raise ValueError("media needs exactly one of path, data or url")
        return self


class SystemOneRequest(_Strict):
    model: str | None = None
    state: str | dict[str, Any]
    media: list[Media] = []
    questions: dict[str, Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _media_limits(self):
        for m in self.media:
            if m.url is not None:
                raise ValueError(f"{m.type} urls are not fetched; pass the {m.type} as a path or base64 data")
        if sum(m.type == "image" for m in self.media) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per request")
        if sum(m.type == "audio" for m in self.media) > MAX_AUDIO:
            raise ValueError(f"at most {MAX_AUDIO} audio clips per request")
        return self


_BARE_OTHER = {"other", "others", "none", "none of the above", "else", "something else", "n/a"}


def lint_request(req: SystemOneRequest) -> list[str]:
    """Warnings that don't block a request.

    An `other` option can't see its siblings (docs/guide/06 §6.8), so a description
    like "Other" gives the model nothing to judge.
    """
    warnings = []
    for qid, q in req.questions.items():
        if q.type != "choice":
            continue
        for name, desc in q.criteria.items():
            text = desc if isinstance(desc, str) else ""
            if name.strip().lower() in _BARE_OTHER and (
                    text.strip().lower() in _BARE_OTHER | {""} or len(text.split()) < 4):
                warnings.append(f"{qid}: option '{name}' has no self-contained description; "
                                "say what it covers, e.g. 'The message is about something other than X, Y or Z'")
    return warnings


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Content]
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class Usage(BaseModel):
    input_tokens: int
    blocks: int
    latency_ms: float
    confidence_method: str
    adapter: str | None = None


class SystemOneResponse(BaseModel):
    model: str | None
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: Usage
