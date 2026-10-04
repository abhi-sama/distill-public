"""A plain provider registry: which teacher may play which role.

There is no quota reading, switching or pausing. A role names the provider, and the registry
either returns it or refuses with a one-line reason. Local Ollama models may play every role.
The hosted-API providers may only answer comparisons (`distill eval --reference`) or an
exported model's optional fallback; they never produce training data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .base import Teacher

Role = Literal["writer", "labeller", "reference", "fallback"]

# Hosted providers: a comparison point or a runtime fallback, never a source of training data.
REFERENCE_ONLY: frozenset[str] = frozenset({"anthropic-api", "openai-api"})

_REFUSAL = (
    "{provider} cannot be the {role}: it is a hosted evaluation reference, and its provider's "
    "terms govern its outputs, so use a local model (for example ollama-gemma-dense) for "
    "training data."
)


_TRAINING_ROLES: frozenset[str] = frozenset({"writer", "labeller"})


class RoleRefused(ValueError):
    """A provider was asked to play a role it is not allowed to play."""


class TeacherRegistry:
    """Maps provider names to teachers and enforces which roles each may play."""

    def __init__(self, teachers: Mapping[str, Teacher]) -> None:
        self.teachers = dict(teachers)

    def get(self, provider: str, role: Role) -> Teacher:
        if provider not in self.teachers:
            raise KeyError(f"unknown teacher provider {provider!r}")
        if role in _TRAINING_ROLES and provider in REFERENCE_ONLY:
            raise RoleRefused(_REFUSAL.format(provider=provider, role=role))
        return self.teachers[provider]

    def writer(self, provider: str) -> Teacher:
        """The teacher that drafts specs and writes synthetic states."""
        return self.get(provider, "writer")

    def labeller(self, provider: str) -> Teacher:
        """A teacher that soft-labels states."""
        return self.get(provider, "labeller")

    def reference(self, provider: str) -> Teacher:
        """A teacher whose answers `distill eval` compares the model with."""
        return self.get(provider, "reference")

    def fallback(self, provider: str) -> Teacher:
        """The teacher an exported model asks when it is not confident."""
        return self.get(provider, "fallback")
