from __future__ import annotations

import sys

import numpy as np
import pytest
from conftest import answer

from distill.labelling import build_gold, option_keys
from distill.review_queue import ReviewQueue
from distill.suspects import (
    CleanlabUnavailable,
    cleanlab_scorer,
    find_suspect_labels,
    send_to_review,
)


def labelled_rows(questions, labels):
    """Rows whose only gold question is ``escalate`` with the given hard labels."""
    rows = []
    for index, label in enumerate(labels):
        gold = build_gold(
            {"escalate": questions["escalate"]}, {"escalate": {label: 0.9, _other(label): 0.1}}
        )
        rows.append({"state": {"text": f"ticket {index}"}, "questions": questions, "gold": gold})
    return rows


def _other(label: str) -> str:
    return "false" if label == "true" else "true"


def self_confidence(labels: np.ndarray, pred_probs: np.ndarray) -> list[float]:
    return [float(pred_probs[row, label]) for row, label in enumerate(labels)]


def test_the_lowest_quality_labels_come_back_worst_first(questions) -> None:
    rows = labelled_rows(questions, ["true", "false", "true", "false"])
    pred_probs = {
        (0, "escalate"): [0.1, 0.9],
        (1, "escalate"): [0.2, 0.8],  # labelled false, model says true
        (2, "escalate"): [0.6, 0.4],  # labelled true, model leans false
        (3, "escalate"): [0.95, 0.05],
        (3, "urgency"): [1, 0, 0, 0],  # no gold for urgency: ignored
    }

    suspects = find_suspect_labels(rows, pred_probs, top_fraction=0.5, scorer=self_confidence)

    assert [(s.row, s.label, s.suggested) for s in suspects] == [
        (1, "false", "true"),
        (2, "true", "false"),
    ]
    assert suspects[0].quality == pytest.approx(0.2)


def test_probabilities_must_match_the_question_options(questions) -> None:
    rows = labelled_rows(questions, ["true"])

    with pytest.raises(ValueError, match="expected 2 probabilities, got 3"):
        find_suspect_labels(rows, {(0, "escalate"): [0.2, 0.3, 0.5]}, scorer=self_confidence)
    with pytest.raises(ValueError, match="top_fraction"):
        find_suspect_labels(rows, {}, top_fraction=0, scorer=self_confidence)


def test_the_real_cleanlab_scorer_flags_an_obvious_mislabel(questions) -> None:
    pytest.importorskip("cleanlab")
    labels = ["true"] * 10 + ["false"] * 10
    labels[3] = "false"  # the model is sure item 3 is true
    rows = labelled_rows(questions, labels)
    pred_probs = {
        (index, "escalate"): [0.05, 0.95] if index < 10 else [0.9, 0.1] for index in range(20)
    }

    [suspect] = find_suspect_labels(rows, pred_probs, top_fraction=0.05)

    assert (suspect.row, suspect.label, suspect.suggested) == (3, "false", "true")


def test_cleanlab_stays_optional(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cleanlab.rank", None)

    with pytest.raises(CleanlabUnavailable, match="distill\\[cleanlab\\]"):
        cleanlab_scorer(np.array([0]), np.array([[1.0, 0.0]]))


def test_suspects_are_sent_to_the_review_queue(tmp_path, questions) -> None:
    rows = labelled_rows(questions, ["true", "false"])
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows, split="train")
        for item_id in (1, 2):
            queue.record_answer(item_id, "a", answer(questions, "ollama-qwen"))
            queue.record_answer(item_id, "b", answer(questions, "ollama-gemma"))
        queue.decide(2, {qid: {option_keys(q)[0]: 1} for qid, q in questions.items()})
        suspects = find_suspect_labels(
            rows,
            {(0, "escalate"): [0.9, 0.1], (1, "escalate"): [0.1, 0.9]},
            top_fraction=1,
            scorer=self_confidence,
        )

        assert send_to_review(queue, rows, suspects) == 1
        flagged = queue.get(1)
        assert flagged.pending and flagged.flags[-1]["trigger"] == "cleanlab"
        assert "suggests 'false'" in flagged.flags[-1]["detail"]
        assert queue.get(2).human is not None
