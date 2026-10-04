"""Opt-in smoke tests for real teachers (local Ollama); never run in CI."""

from __future__ import annotations

import os

import pytest
from conftest import EXAMPLE

from distill.teachers import (
    DEFAULT_LABELLER_A,
    DEFAULT_LABELLER_B,
    OllamaTeacher,
)


@pytest.mark.skipif(
    os.environ.get("DISTILL_LIVE_OLLAMA") != "1",
    reason="set DISTILL_LIVE_OLLAMA=1 to call the local Ollama models (free, but slow)",
)
@pytest.mark.parametrize("provider", [DEFAULT_LABELLER_A, DEFAULT_LABELLER_B])
def test_default_local_labellers_answer_a_real_labeller_batch(provider):
    from distill.labelling import labeller_prompt, labeller_schema, parse_batch
    from distill.schema import load_decision

    questions = load_decision(EXAMPLE).model_dump(mode="json")
    states = [{"text": "My card was charged twice and your chat bot keeps closing on me."}]
    teacher = OllamaTeacher.registered(provider, max_attempts=2)
    raw = teacher.ask(labeller_prompt(questions, states), labeller_schema(questions, 1))
    [answer] = parse_batch(raw, questions, 1, provider)
    assert set(answer.distributions) == set(questions)
