"""Fixed reconstruction rules for decomposed decision experiments.

The rules in this module are declared before an experiment is run.  They translate
independent binary probabilities into the original multi-class answer without looking
at its evaluation labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class Reconstruction:
    """One original question reconstructed from binary question probabilities."""

    question_id: str
    options: tuple[str, ...]

    def probabilities(self, answers: Mapping[str, float]) -> tuple[float, ...]:
        """Return probabilities in ``options`` order, validating the rule's inputs."""
        raise NotImplementedError


@dataclass(frozen=True)
class ModerationAction(Reconstruction):
    """Remove for any clear violation, then review ambiguity, otherwise keep."""

    violations: tuple[str, ...]
    review: str

    def probabilities(self, answers: Mapping[str, float]) -> tuple[float, ...]:
        _require(answers, (*self.violations, self.review))
        remove = 1 - prod(1 - answers[qid] for qid in self.violations)
        review = (1 - remove) * answers[self.review]
        keep = (1 - remove) * (1 - answers[self.review])
        return _ordered(self.options, {"keep": keep, "review": review, "remove": remove})


@dataclass(frozen=True)
class OrderedSurface(Reconstruction):
    """Choose the first positive surface in a published, safety-first precedence order."""

    precedence: tuple[str, ...]

    def probabilities(self, answers: Mapping[str, float]) -> tuple[float, ...]:
        _require(answers, self.precedence)
        remaining = 1.0
        result: dict[str, float] = {}
        for qid in self.precedence:
            result[_surface_name(qid)] = remaining * answers[qid]
            remaining *= 1 - answers[qid]
        result["none"] = remaining
        return _ordered(self.options, result)


@dataclass(frozen=True)
class OrderedRisk(Reconstruction):
    """Choose the highest applicable fixed risk condition."""

    high: str
    medium: str
    low: str

    def probabilities(self, answers: Mapping[str, float]) -> tuple[float, ...]:
        _require(answers, (self.high, self.medium, self.low))
        high = answers[self.high]
        medium = (1 - high) * answers[self.medium]
        low = (1 - high) * (1 - answers[self.medium]) * answers[self.low]
        none = (1 - high) * (1 - answers[self.medium]) * (1 - answers[self.low])
        return _ordered(self.options, {"0": none, "1": low, "2": medium, "3": high})


MODERATION_RULES: tuple[Reconstruction, ...] = (
    ModerationAction(
        question_id="action",
        options=("keep", "review", "remove"),
        violations=(
            "harassment_or_hate",
            "violent_threat",
            "spam_or_scam",
            "sexual_exploitation_or_self_harm",
        ),
        review="ambiguous_or_context_dependent",
    ),
)

PR_RULES: tuple[Reconstruction, ...] = (
    OrderedRisk(
        question_id="risk_level",
        options=("0", "1", "2", "3"),
        high="has_high_risk_condition",
        medium="changes_security_control_or_exposed_surface",
        low="security_adjacent_but_contained",
    ),
    OrderedSurface(
        question_id="primary_surface",
        options=(
            "none",
            "auth",
            "crypto_secrets",
            "untrusted_input",
            "network",
            "dependencies",
            "infra_permissions",
            "personal_data",
        ),
        precedence=(
            "touches_untrusted_input",
            "touches_crypto_secrets",
            "touches_auth",
            "touches_network",
            "touches_infra_permissions",
            "touches_personal_data",
            "touches_dependencies",
        ),
    ),
)


def _surface_name(question_id: str) -> str:
    if not question_id.startswith("touches_"):
        raise ValueError(f"surface question {question_id!r} must start with 'touches_'")
    return question_id.removeprefix("touches_")


def _ordered(options: tuple[str, ...], values: Mapping[str, float]) -> tuple[float, ...]:
    if set(options) != set(values):
        raise ValueError("reconstruction values do not match its original options")
    result = tuple(values[option] for option in options)
    if any(not 0 <= value <= 1 for value in result) or abs(sum(result) - 1) > 1e-9:
        raise ValueError("reconstruction did not produce a probability distribution")
    return result


def _require(answers: Mapping[str, float], question_ids: tuple[str, ...]) -> None:
    missing = set(question_ids) - answers.keys()
    if missing:
        raise ValueError(f"reconstruction is missing answers for {sorted(missing)}")
    if any(not 0 <= answers[qid] <= 1 for qid in question_ids):
        raise ValueError("binary answer probability must be within [0, 1]")
