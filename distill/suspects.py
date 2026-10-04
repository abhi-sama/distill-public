"""Optional Cleanlab hook that flags suspect labels for relabelling or human review.

Cleanlab ranks labels by how well they agree with *out-of-sample* model probabilities, so
T4 has nothing to feed it yet: the caller is T5's relabel pass, which trains fold models and
passes their out-of-fold probabilities here. Cleanlab itself is an optional extra
(``distill[cleanlab]``) and is imported only when the default scorer runs, so any other
scorer with the same signature can stand in for it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .labelling import Disagreement, argmax, option_keys
from .review_queue import ReviewQueue, item_key

# Maps integer labels (n,) and predicted probabilities (n, k) to label-quality scores in
# [0, 1], lower meaning more likely mislabelled: the signature of
# ``cleanlab.rank.get_label_quality_scores``.
LabelQualityScorer = Callable[[np.ndarray, np.ndarray], Sequence[float]]


class CleanlabUnavailable(ImportError):
    """The default scorer was used without the optional ``cleanlab`` extra installed."""


@dataclass(frozen=True)
class SuspectLabel:
    """One ``(row, question)`` label ranked as likely wrong."""

    row: int
    question: str
    label: str
    suggested: str
    quality: float


def cleanlab_scorer(labels: np.ndarray, pred_probs: np.ndarray) -> Sequence[float]:
    """Cleanlab's self-confidence label-quality score."""
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as error:
        raise CleanlabUnavailable(
            "Cleanlab is optional; install it with `uv sync --extra cleanlab` "
            "or `pip install 'distill[cleanlab]'`"
        ) from error
    return get_label_quality_scores(labels, pred_probs)


def find_suspect_labels(
    rows: Sequence[Mapping[str, Any]],
    pred_probs: Mapping[tuple[int, str], Sequence[float]],
    *,
    top_fraction: float = 0.02,
    scorer: LabelQualityScorer = cleanlab_scorer,
) -> list[SuspectLabel]:
    """Rank every labelled ``(row, question)`` and return the least trustworthy ones.

    ``rows`` are notebook ``{state, questions, gold}`` rows. ``pred_probs`` maps
    ``(row index, question id)`` to out-of-sample probabilities in the question's option
    order (see :func:`distill.labelling.option_keys`); pairs without an entry are skipped.
    The lowest-quality ``top_fraction`` of scored pairs is returned, worst first. Questions
    are scored separately, since each has its own options.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    by_question: dict[str, list[tuple[int, int, list[float]]]] = {}
    options_by_question: dict[str, list[str]] = {}
    for (row_index, question_id), probs in pred_probs.items():
        row = rows[row_index]
        if question_id not in row["gold"]:
            continue
        options = option_keys(row["questions"][question_id])
        vector = [float(value) for value in probs]
        if len(vector) != len(options):
            raise ValueError(
                f"row {row_index} {question_id!r}: expected {len(options)} probabilities, "
                f"got {len(vector)}"
            )
        label = str(row["gold"][question_id]["label"])
        by_question.setdefault(question_id, []).append((row_index, options.index(label), vector))
        options_by_question[question_id] = options

    scored: list[SuspectLabel] = []
    for question_id, entries in by_question.items():
        options = options_by_question[question_id]
        labels = np.array([label for _, label, _ in entries], dtype=int)
        matrix = np.array([vector for _, _, vector in entries], dtype=float)
        qualities = scorer(labels, matrix)
        for (row_index, label, vector), quality in zip(entries, qualities, strict=True):
            scored.append(
                SuspectLabel(
                    row=row_index,
                    question=question_id,
                    label=options[label],
                    suggested=argmax(dict(zip(options, vector, strict=True))),
                    quality=float(quality),
                )
            )
    scored.sort(key=lambda suspect: (suspect.quality, suspect.row, suspect.question))
    return scored[: math.ceil(top_fraction * len(scored))]


def send_to_review(
    queue: ReviewQueue,
    rows: Sequence[Mapping[str, Any]],
    suspects: Sequence[SuspectLabel],
    *,
    include_reviewed: bool = False,
) -> int:
    """Flag each suspect's queue item for human review; returns how many were flagged.

    Rows are matched to queue items by content. Items a human already decided are skipped
    unless ``include_reviewed`` is set.
    """
    flagged = 0
    for suspect in suspects:
        row = rows[suspect.row]
        item = queue.find(item_key(row["state"], row["questions"]))
        if item is None:
            continue
        reason = Disagreement(
            suspect.question,
            "cleanlab",
            f"label {suspect.label!r} has quality {suspect.quality:.3f}; "
            f"the model suggests {suspect.suggested!r}",
        )
        flagged += queue.flag(item.id, reason, include_reviewed=include_reviewed)
    return flagged
