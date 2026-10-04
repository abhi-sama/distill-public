"""Pydantic models for ``decision.yaml`` question definitions.

The question mapping deliberately mirrors the shape accepted by Laya's
``Laya._check_question``.  It has no dependency on Laya, so validation remains
fast and does not load a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class _Question(BaseModel):
    """Fields common to every Laya question definition."""

    model_config = ConfigDict(extra="allow")

    instructions: Any


class ChoiceQuestion(_Question):
    """A categorical question with a mapping or list of options."""

    type: Literal["choice"]
    criteria: dict[Any, Any] | list[Any]

    @model_validator(mode="after")
    def has_criterion(self) -> ChoiceQuestion:
        if "labels" in self.model_extra:
            raise ValueError("'labels' is only supported for noul questions")
        if not self.criteria:
            raise ValueError("a choice question needs at least one criterion")
        return self


class ScoreQuestion(_Question):
    """An ordinal question whose levels are listed from index zero."""

    type: Literal["score"]
    criteria: list[Any]

    @model_validator(mode="after")
    def has_level(self) -> ScoreQuestion:
        if "labels" in self.model_extra:
            raise ValueError("'labels' is only supported for noul questions")
        if not self.criteria:
            raise ValueError("a score question needs at least one level")
        return self


class NoulQuestion(_Question):
    """A boolean Laya question; its semantic option order is false, then true."""

    type: Literal["noul"]
    criteria: dict[Any, Any] | None = None
    labels: dict[Any, Any] | None = None

    @model_validator(mode="after")
    def follows_laya_noul_rules(self) -> NoulQuestion:
        if self.criteria is not None:
            keys = {str(key).lower() for key in self.criteria}
            if not keys <= {"true", "false"}:
                raise ValueError("a noul question's criteria may be keyed only 'true'/'false'")

        if self.labels is not None:
            if set(self.labels) != {"false", "true"}:
                raise ValueError("noul labels must map exactly 'false' and 'true'")
            false_label = self.labels["false"]
            true_label = self.labels["true"]
            if not isinstance(false_label, str) or not isinstance(true_label, str):
                raise ValueError("noul labels must be strings")
            false_label = false_label.strip()
            true_label = true_label.strip()
            if not false_label or not true_label or false_label == true_label:
                raise ValueError("noul labels must be distinct, non-empty strings")
        return self


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


class DecisionSpec(RootModel[dict[str, Question]]):
    """The flat ``question_id -> question definition`` mapping read by Laya."""


def load_decision(path: str | Path) -> DecisionSpec:
    """Load and validate a ``decision.yaml`` file without importing or loading Laya."""
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return DecisionSpec.model_validate(data)
