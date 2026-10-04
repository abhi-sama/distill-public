from __future__ import annotations

import pytest
from conftest import answer

from distill.labelling import Disagreement, InvalidDistribution, option_keys
from distill.review_queue import ReviewQueue, item_key

DISAGREE = {"escalate": {"false": 0.3, "true": 0.7}}


def label(queue: ReviewQueue, questions, item_id: int, **b_picks) -> None:
    queue.record_answer(item_id, "a", answer(questions, "ollama-qwen"))
    queue.record_answer(item_id, "b", answer(questions, "ollama-gemma", **b_picks))


def human_label(questions, **overrides):
    """Every question one-hot on its first option unless overridden."""
    return {qid: overrides.get(qid, {option_keys(q)[0]: 1.0}) for qid, q in questions.items()}


def test_the_queue_survives_a_restart(tmp_path, questions, rows) -> None:
    path = tmp_path / "queue.sqlite3"
    with ReviewQueue(path) as queue:
        assert queue.add(rows, split="train") == 3
        label(queue, questions, 1)
        label(queue, questions, 2, **DISAGREE)
        queue.record_answer(3, "a", answer(questions, "ollama-qwen"))

    with ReviewQueue(path) as queue:
        first, second, third = queue.items("all")
        assert first.merged is not None and not first.pending
        assert second.pending and second.flags[0]["trigger"] == "different_answer"
        assert second.answers["b"].distributions["escalate"] == {"false": 0.3, "true": 0.7}
        assert third.answers["b"] is None and third.merged is None
        assert [item.id for item in queue.unlabelled("b")] == [3]
        queue.decide(2, human_label(questions, escalate={"true": 1.0}), note="clear escalation")

    with ReviewQueue(path) as queue:
        decided = queue.get(2)
        assert decided.human["escalate"]["label"] == "true"
        assert decided.note == "clear escalation" and decided.reviewed_at
        assert not decided.pending
        assert queue.counts()["pending"] == 0


def test_re_adding_rows_is_idempotent_and_the_gold_set_is_capped(tmp_path, questions, rows):
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows, split="train")
        assert queue.add(rows, split="train") == 0

        gold = [{"state": {"text": f"gold {n}"}, "questions": questions} for n in range(5)]
        assert queue.add(gold, split="gold", limit=2) == 2
        assert queue.add(gold, split="gold", limit=2) == 0
        assert queue.counts()["gold"] == 2
        assert queue.find(item_key(gold[0]["state"], questions)).split == "gold"


def test_a_human_decision_overwrites_the_merged_label(tmp_path, questions, rows) -> None:
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows[:1], split="train")
        label(queue, questions, 1)
        assert queue.get(1).merged["escalate"]["label"] == "false"
        [before] = queue.export_rows("train")
        assert before["gold"]["escalate"]["label"] == "false"

        item = queue.decide(
            1, human_label(questions, escalate={"false": 0.25, "true": 0.75}, urgency={"3": 1})
        )

        assert item.final_gold == item.human
        [row] = queue.export_rows("train")
        assert set(row) == {"state", "questions", "gold"}
        assert row["gold"]["escalate"] == {
            "label": "true",
            "noul": 0.75,
            "probabilities": {"false": 0.25, "true": 0.75},
        }
        assert row["gold"]["urgency"]["label"] == 3
        # The labellers' merge is kept for audit, but no longer used.
        assert queue.get(1).merged["escalate"]["label"] == "false"


def test_exports_hold_back_undecided_flagged_and_gold_items(tmp_path, questions, rows) -> None:
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows, split="train")
        queue.add([{"state": "gold item", "questions": questions}], split="gold")
        label(queue, questions, 1)
        label(queue, questions, 2, **DISAGREE)
        label(queue, questions, 4)

        assert [row["state"] for row in queue.export_rows("train")] == [{"text": "ticket 0"}]
        assert len(queue.export_rows("train", include_unreviewed=True)) == 2
        assert queue.export_rows("gold") == []

        queue.decide(4, human_label(questions))
        assert [row["state"] for row in queue.export_rows("gold")] == ["gold item"]


def test_decisions_are_validated_before_anything_is_stored(tmp_path, questions, rows):
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows[:1], split="train")
        incomplete = human_label(questions)
        del incomplete["urgency"]

        with pytest.raises(InvalidDistribution, match="urgency"):
            queue.decide(1, incomplete)
        with pytest.raises(InvalidDistribution, match="all be zero"):
            queue.decide(1, human_label(questions, escalate={"false": 0, "true": 0}))
        assert queue.get(1).human is None


def test_external_flags_route_items_to_review_and_survive_a_re_merge(tmp_path, questions, rows):
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows[:1], split="train")
        label(queue, questions, 1)
        suspect = Disagreement("escalate", "cleanlab", "quality 0.01")

        assert queue.flag(1, suspect)
        assert queue.get(1).pending
        label(queue, questions, 1)
        assert [flag["trigger"] for flag in queue.get(1).flags] == ["cleanlab"]

        queue.decide(1, human_label(questions))
        assert not queue.flag(1, suspect)
        assert queue.get(1).human is not None
        assert queue.flag(1, suspect, include_reviewed=True)
        assert queue.get(1).human is None and queue.get(1).pending


def test_next_pending_moves_forward_and_wraps(tmp_path, questions, rows) -> None:
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        queue.add(rows, split="train")
        for item_id in (1, 2, 3):
            label(queue, questions, item_id, **({} if item_id == 2 else DISAGREE))

        assert queue.next_pending(0).id == 1
        assert queue.next_pending(1).id == 3
        assert queue.next_pending(3).id == 1
        assert [item.id for item in queue.items("review")] == [1, 3]


def test_unknown_labellers_and_splits_are_rejected(tmp_path, questions, rows) -> None:
    with ReviewQueue(tmp_path / "queue.sqlite3") as queue:
        with pytest.raises(ValueError):
            queue.add(rows, split="test")
        with pytest.raises(ValueError):
            queue.unlabelled("c")
