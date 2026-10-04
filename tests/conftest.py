"""Shared fakes for labelling tests: no CLI calls, no network, no model loading."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from distill.labelling import LabellerAnswer, option_keys
from distill.schema import load_decision
from distill.teachers import (
    DEFAULT_LABELLER_A,
    DEFAULT_LABELLER_B,
    Teacher,
    TeacherRegistry,
)

# The default labellers: two local models.
LOCAL_A, LOCAL_B = DEFAULT_LABELLER_A, DEFAULT_LABELLER_B

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "decision.yaml"


def confident(question: Mapping[str, Any], pick: int = 0, mass: float = 0.9) -> dict[str, float]:
    """A distribution putting ``mass`` on option ``pick`` and spreading the rest evenly."""
    options = option_keys(question)
    rest = (1 - mass) / (len(options) - 1)
    return {option: mass if index == pick else rest for index, option in enumerate(options)}


class FakeLabeller(Teacher):
    """Answers labeller prompts from ``overrides[state text][question]``, else ``confident``."""

    def __init__(
        self,
        provider: str,
        questions: Mapping[str, Mapping[str, Any]],
        overrides: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
        fail_when: Callable[[str], Exception | None] | None = None,
    ) -> None:
        self.provider = provider
        self.model = None
        self.questions = questions
        self.overrides = overrides or {}
        self.fail_when = fail_when
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def ask(self, prompt: str, schema: Mapping[str, Any]) -> Any:
        with self._lock:
            self.prompts.append(prompt)
        if self.fail_when is not None and (error := self.fail_when(prompt)) is not None:
            raise error
        states = [
            json.loads(match)
            for match in re.findall(r"^item_\d+: (.*)$", prompt.split("Items:", 1)[1], re.M)
        ]
        answer = {}
        for index, state in enumerate(states):
            chosen = self.overrides.get(state["text"], {})
            answer[f"item_{index}"] = {
                question_id: {
                    "rationale": f"{self.provider} on {state['text']}",
                    "probabilities": chosen.get(question_id, confident(question)),
                }
                for question_id, question in self.questions.items()
            }
        return answer


def fake_registry(*teachers: Teacher) -> TeacherRegistry:
    """A registry over ``teachers`` keyed by provider."""
    return TeacherRegistry({teacher.provider: teacher for teacher in teachers})


def answer(questions: Mapping[str, Any], provider: str, **picks: dict[str, float]):
    """A LabellerAnswer that is ``confident`` everywhere except the given questions."""
    return LabellerAnswer(
        provider,
        {qid: picks.get(qid, confident(question)) for qid, question in questions.items()},
        {qid: f"{provider} reasoning" for qid in questions},
    )


@pytest.fixture
def questions() -> dict[str, dict[str, Any]]:
    return load_decision(EXAMPLE).model_dump(mode="json")


@pytest.fixture
def rows(questions) -> list[dict[str, Any]]:
    return [
        {"state": {"text": f"ticket {index}"}, "questions": questions, "gold": {}}
        for index in range(3)
    ]
