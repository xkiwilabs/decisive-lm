"""Canonical record (docs/guide/08 §8.1). Every writer validates against it.

A row is one request: one state plus its question set. Each question may carry
a human target, a teacher target, or both; questions without one are masked in
the loss.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from decisivelm.api_types import ChoiceQuestion, Media, Question, ScoreQuestion

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "gold_calib", "gold_test", "holdout")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lineage(_Strict):
    source_dataset: str
    source_id: str
    family_id: str                      # schema family / leakage unit; splits are assigned per family (docs/guide/08 §8.4)
    parent_id: str | None = None
    multiplier: str | None = None


class Target(_Strict):
    probabilities: dict[str, float] | None = None    # Choice / Score
    noul: float | None = None                        # Noul


class HumanTarget(Target):
    n_raters: int | None = None


class TeacherTarget(Target):
    method: str
    n_orderings: int | None = None
    n_traces: int | None = None
    allowed_mass: float | None = None
    teacher_model: str
    teacher_revision: str | None = None
    vllm_version: str | None = None
    prompt_hash: str | None = None
    calibrated_T: float | None = None


class QuestionTargets(_Strict):
    human: HumanTarget | None = None
    teacher: TeacherTarget | list[TeacherTarget] | None = None


class Preprocess(_Strict):
    render_version: int
    max_state_tokens: int
    truncated: bool


class Record(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    id: str
    lineage: Lineage
    license: str
    tier: Literal["R", "S", "X"]
    modality: list[Literal["text", "image", "audio"]]
    language: str
    state: str | dict[str, Any]
    media: list[Media] = []
    questions: dict[str, Question]
    targets: dict[str, QuestionTargets]
    preprocess: Preprocess
    split: Literal[SPLITS]

    @model_validator(mode="after")
    def _targets_match_questions(self):
        for qid, t in self.targets.items():
            if qid not in self.questions:
                raise ValueError(f"target for unknown question {qid!r}")
            q = self.questions[qid]
            teachers = t.teacher if isinstance(t.teacher, list) else [t.teacher]
            for target in [t.human, *teachers]:
                if target is not None:
                    _check_target(qid, q, target)
        return self


def _check_target(qid, q, target: Target):
    if isinstance(q, (ChoiceQuestion, ScoreQuestion)):
        keys = list(q.criteria) if isinstance(q, ChoiceQuestion) else [str(i) for i in range(len(q.criteria))]
        p = target.probabilities
        if p is None or target.noul is not None:
            raise ValueError(f"{qid}: {q.type} targets need probabilities only")
        if set(p) != set(keys):
            raise ValueError(f"{qid}: target keys {sorted(p)} do not match options {sorted(keys)}")
        if abs(sum(p.values()) - 1.0) > 1e-6 or min(p.values()) < 0:
            raise ValueError(f"{qid}: target is not a distribution")
    else:
        if target.noul is None or target.probabilities is not None or not 0.0 <= target.noul <= 1.0:
            raise ValueError(f"{qid}: noul targets need a value in [0, 1] only")
