"""The persistent human review queue behind ``distill label`` and ``distill review``.

One SQLite file holds every item to label: its state and questions, both labellers' answers,
the merged soft label, why it was flagged, and any human decision. A human decision always
overrides the merged label. Items are keyed by their content, so re-adding a corpus is
idempotent and labelling resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .labelling import (
    DISAGREEMENT_THRESHOLD,
    LABELLERS,
    Disagreement,
    LabellerAnswer,
    build_gold,
    merge_answers,
)

SPLITS = ("train", "gold", "heldout")
VIEWS = ("review", "gold", "reviewed", "all")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    split TEXT NOT NULL CHECK (split IN ('train', 'gold', 'heldout')),
    state TEXT NOT NULL,
    questions TEXT NOT NULL,
    answer_a TEXT,
    answer_b TEXT,
    merged TEXT,
    flags TEXT NOT NULL DEFAULT '[]',
    needs_review INTEGER NOT NULL DEFAULT 0,
    human TEXT,
    note TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT
);
"""

# An item is awaiting a human when it is in the gold set or was flagged, and nobody has
# decided it yet.
_PENDING = "(split = 'gold' OR needs_review = 1) AND human IS NULL"
_VIEW_WHERE = {
    "review": _PENDING,
    "gold": "split = 'gold'",
    "reviewed": "human IS NOT NULL",
    "all": "1 = 1",
}


def item_key(state: Any, questions: Mapping[str, Any]) -> str:
    """A stable content key for a ``(state, questions)`` pair, e.g. a training row."""
    canonical = json.dumps(
        {"state": state, "questions": questions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueueItem:
    """One item as stored in the queue."""

    id: int
    key: str
    split: str
    state: Any
    questions: dict[str, dict[str, Any]]
    answers: dict[str, LabellerAnswer | None]
    merged: dict[str, dict[str, Any]] | None
    flags: list[dict[str, str]]
    needs_review: bool
    human: dict[str, dict[str, Any]] | None
    note: str
    reviewed_at: str | None

    @property
    def pending(self) -> bool:
        return (self.split == "gold" or self.needs_review) and self.human is None

    @property
    def final_gold(self) -> dict[str, dict[str, Any]] | None:
        """The label to train or evaluate on: the human decision, else the merged label."""
        return self.human if self.human is not None else self.merged


class ReviewQueue:
    """Thread-safe access to one SQLite review queue file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> ReviewQueue:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def add(
        self, rows: Iterable[Mapping[str, Any]], *, split: str, limit: int | None = None
    ) -> int:
        """Add ``{state, questions, ...}`` rows; existing items are left untouched.

        With ``limit``, stop once the split holds that many items, so a gold set never grows
        past its target size when this is re-run.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS!r}")
        added = 0
        with self._lock, self._db:
            for row in rows:
                if limit is not None and self._count(split) >= limit:
                    break
                state, questions = row["state"], row["questions"]
                cursor = self._db.execute(
                    "INSERT OR IGNORE INTO items (key, split, state, questions) "
                    "VALUES (?, ?, ?, ?)",
                    (item_key(state, questions), split, _dumps(state), _dumps(questions)),
                )
                added += cursor.rowcount
        return added

    def _count(self, split: str) -> int:
        return self._db.execute("SELECT COUNT(*) FROM items WHERE split = ?", (split,)).fetchone()[
            0
        ]

    def get(self, item_id: int) -> QueueItem | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _item(row) if row else None

    def find(self, key: str) -> QueueItem | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM items WHERE key = ?", (key,)).fetchone()
        return _item(row) if row else None

    def items(self, view: str = "all") -> list[QueueItem]:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS!r}")
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM items WHERE {_VIEW_WHERE[view]} ORDER BY id"
            ).fetchall()
        return [_item(row) for row in rows]

    def unlabelled(self, labeller: str) -> list[QueueItem]:
        column = _answer_column(labeller)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM items WHERE {column} IS NULL ORDER BY id"
            ).fetchall()
        return [_item(row) for row in rows]

    def next_pending(self, after_id: int = 0) -> QueueItem | None:
        """The next item awaiting a human after ``after_id``, wrapping to the start."""
        with self._lock:
            row = self._db.execute(
                f"SELECT * FROM items WHERE {_PENDING} ORDER BY id > ? DESC, id LIMIT 1",
                (after_id,),
            ).fetchone()
        return _item(row) if row else None

    def record_answer(
        self,
        item_id: int,
        labeller: str,
        answer: LabellerAnswer,
        *,
        threshold: float = DISAGREEMENT_THRESHOLD,
    ) -> QueueItem:
        """Store one labeller's answer; once both are in, store the merge and its flags."""
        column = _answer_column(labeller)
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE items SET {column} = ? WHERE id = ?", (_dumps(answer.to_json()), item_id)
            )
            row = self._require(item_id)
            item = _item(row)
            first, second = (item.answers[each.name] for each in LABELLERS)
            if first is not None and second is not None:
                merged, flags = merge_answers(item.questions, first, second, threshold=threshold)
                # Keep flags from other sources (such as Cleanlab); replace the labellers' own.
                kept = [flag for flag in item.flags if flag["trigger"] not in _MERGE_TRIGGERS]
                all_flags = kept + [flag.to_json() for flag in flags]
                self._db.execute(
                    "UPDATE items SET merged = ?, flags = ?, needs_review = ? WHERE id = ?",
                    (_dumps(merged), _dumps(all_flags), int(bool(all_flags)), item_id),
                )
                row = self._require(item_id)
        return _item(row)

    def flag(self, item_id: int, flag: Disagreement, *, include_reviewed: bool = False) -> bool:
        """Send an item to review for an external reason, such as a Cleanlab suspect.

        Items a human already decided are left alone unless ``include_reviewed`` is set, in
        which case the earlier decision is cleared so the item is reviewed again.
        """
        with self._lock, self._db:
            item = _item(self._require(item_id))
            if item.human is not None and not include_reviewed:
                return False
            flags = item.flags + [flag.to_json()]
            self._db.execute(
                "UPDATE items SET flags = ?, needs_review = 1, human = NULL, reviewed_at = NULL "
                "WHERE id = ?",
                (_dumps(flags), item_id),
            )
        return True

    def decide(
        self,
        item_id: int,
        distributions: Mapping[str, Mapping[str, Any]],
        *,
        note: str = "",
    ) -> QueueItem:
        """Record the human's soft label for every question; it overrides the merged label."""
        with self._lock, self._db:
            item = _item(self._require(item_id))
            gold = build_gold(item.questions, distributions)
            self._db.execute(
                "UPDATE items SET human = ?, note = ?, reviewed_at = ? WHERE id = ?",
                (_dumps(gold), note, datetime.now(UTC).isoformat(timespec="seconds"), item_id),
            )
            row = self._require(item_id)
        return _item(row)

    def counts(self) -> dict[str, int]:
        with self._lock:
            row = self._db.execute(
                f"""SELECT
                    COUNT(*) AS total,
                    SUM(split = 'train') AS train,
                    SUM(split = 'gold') AS gold,
                    SUM(split = 'heldout') AS heldout,
                    SUM(answer_a IS NOT NULL AND answer_b IS NOT NULL) AS labelled,
                    SUM(needs_review = 1) AS flagged,
                    SUM({_PENDING}) AS pending,
                    SUM(split = 'gold' AND human IS NOT NULL) AS gold_done,
                    SUM(human IS NOT NULL) AS reviewed
                FROM items"""
            ).fetchone()
        return {name: int(row[name] or 0) for name in row.keys()}

    def export_rows(
        self, split: str, *, include_unreviewed: bool = False, teacher_gold: bool = False
    ) -> list[dict[str, Any]]:
        """Notebook ``{state, questions, gold}`` rows ready for training or evaluation.

        Gold-set rows need a human decision, unless ``teacher_gold`` is set: then gold items
        both labellers agreed on (nothing flagged) are exported with their merged label, for
        runs with no human reviewer. Training rows use the human decision when there is one,
        else the merged label; flagged rows still awaiting a human are left out unless
        ``include_unreviewed`` is set. Held-out rows are teacher-labelled evaluation data:
        every labelled one is exported, so hard, disagreed cases stay in the measurement.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS!r}")
        rows = []
        for item in self.items("all"):
            if item.split != split or item.final_gold is None:
                continue
            if item.human is None and split != "heldout":
                if split == "gold" and not (teacher_gold and not item.needs_review):
                    continue
                if split == "train" and item.pending and not include_unreviewed:
                    continue
            rows.append({"state": item.state, "questions": item.questions, "gold": item.final_gold})
        return rows

    def _require(self, item_id: int) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(f"no queue item {item_id}")
        return row


_MERGE_TRIGGERS = {"different_answer", "probability_gap"}


def _answer_column(labeller: str) -> str:
    if labeller not in {each.name for each in LABELLERS}:
        raise ValueError(f"unknown labeller {labeller!r}")
    return f"answer_{labeller}"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _item(row: sqlite3.Row) -> QueueItem:
    answers = {
        labeller.name: (
            LabellerAnswer.from_json(json.loads(row[f"answer_{labeller.name}"]))
            if row[f"answer_{labeller.name}"] is not None
            else None
        )
        for labeller in LABELLERS
    }
    return QueueItem(
        id=row["id"],
        key=row["key"],
        split=row["split"],
        state=json.loads(row["state"]),
        questions=json.loads(row["questions"]),
        answers=answers,
        merged=_loads(row["merged"]),
        flags=json.loads(row["flags"]),
        needs_review=bool(row["needs_review"]),
        human=_loads(row["human"]),
        note=row["note"],
        reviewed_at=row["reviewed_at"],
    )
