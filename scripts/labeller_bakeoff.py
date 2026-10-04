"""T10 labeller-B bake-off: label one fixed gold slice with each local teacher and score it.

Usage (from the repo root, with Ollama serving every model named):

    GOLD_FILES=path/a/gold.jsonl,path/b/gold.jsonl \\
        python scripts/labeller_bakeoff.py docs/t10-bakeoff ollama-gemma ollama-nemotron ...

``GOLD_FILES`` lists the gold JSONL files to draw from (each a decision's teacher-agreed gold
set, rows shaped like ``distill label`` writes them). The published run used the first 50 rows
of three such files, 150 items; that slice is not in this repository, so bring your own gold
set. Every candidate gets the same batches of 10 through
the production path, ``OllamaTeacher`` with Distill's own labeller prompt, strict schema,
retries and ``parse_batch``. Per candidate it writes ``<provider>.jsonl`` (one line per batch,
with the normalised distributions) and ``usage-<provider>.jsonl`` (every attempt), then
``summary.json`` over every candidate found in the output directory. A re-run resumes: it
labels only the batches a provider has not answered yet, at most ``MAX_BATCHES`` of them when
that environment variable is set. Delete a provider's files to start it again.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distill.evaluation import (  # noqa: E402
    Prediction,
    accuracy,
    brier_score,
    expected_calibration_error,
)
from distill.labelling import (  # noqa: E402
    InvalidDistribution,
    find_disagreements,
    labeller_prompt,
    labeller_schema,
    option_keys,
    parse_batch,
)
from distill.teachers import (  # noqa: E402
    LOCAL_MODELS,
    MalformedTeacherResponse,
    OllamaTeacher,
    UsageLogger,
)
from distill.training import decode_state  # noqa: E402

PER_DECISION = 50
BATCH = 10


def slice_rows() -> list[tuple[str, dict[str, Any]]]:
    files = [Path(item) for item in os.environ.get("GOLD_FILES", "").split(",") if item]
    if not files:
        raise SystemExit("set GOLD_FILES to a comma-separated list of gold JSONL files")
    rows = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            decoded = [json.loads(line) for line in handle if line.strip()]
        rows.extend((path.parent.name or path.stem, row) for row in decoded[:PER_DECISION])
    return rows


def gold_index(question: dict[str, Any], entry: dict[str, Any]) -> int:
    label = str(entry["label"])
    return option_keys(question).index(label.lower() if question["type"] == "noul" else label)


def run(provider: str, out: Path, max_batches: int | None) -> None:
    """Label the batches this provider has not done yet, at most ``max_batches`` of them."""
    rows = slice_rows()
    results, usage = out / f"{provider}.jsonl", out / f"usage-{provider}.jsonl"
    if not results.exists():
        usage.unlink(missing_ok=True)
    done = {json.loads(line)["start"] for line in results.open()} if results.exists() else set()
    todo = [start for start in range(0, len(rows), BATCH) if start not in done]
    teacher = OllamaTeacher.registered(provider, usage_logger=UsageLogger(usage))
    with results.open("a", encoding="utf-8") as handle:
        for start in todo[:max_batches]:
            batch = rows[start : start + BATCH]
            decision = batch[0][0]
            assert all(name == decision for name, _ in batch), "batches stay within a decision"
            questions = batch[0][1]["questions"]
            states = [decode_state(row["state"]) for _, row in batch]
            record: dict[str, Any] = {"decision": decision, "start": start, "n": len(batch)}
            began = time.monotonic()
            try:
                raw = teacher.ask(
                    labeller_prompt(questions, states), labeller_schema(questions, len(batch))
                )
                answers = parse_batch(raw, questions, len(batch), provider)
            except (MalformedTeacherResponse, InvalidDistribution) as error:
                record.update(valid=False, error=str(error)[:500])
            else:
                record.update(valid=True, distributions=[a.distributions for a in answers])
            record["wall_s"] = round(time.monotonic() - began, 2)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(provider, decision, start, record["valid"], record["wall_s"], flush=True)


def score(provider: str, out: Path, rows: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    batches = sorted(
        (json.loads(line) for line in (out / f"{provider}.jsonl").open(encoding="utf-8")),
        key=lambda batch: batch["start"],
    )
    attempts = [
        json.loads(line)["outcome"] for line in (out / f"usage-{provider}.jsonl").open()
    ]
    predictions: list[Prediction] = []
    by_decision: dict[str, list[Prediction]] = {}
    by_question: dict[str, list[Prediction]] = {}
    for batch in batches:
        if not batch["valid"]:
            continue
        for offset, dists in enumerate(batch["distributions"]):
            decision, row = rows[batch["start"] + offset]
            for qid, question in row["questions"].items():
                options = option_keys(question)
                prediction = Prediction(
                    tuple(dists[qid][option] for option in options),
                    gold_index(question, row["gold"][qid]),
                )
                predictions.append(prediction)
                by_decision.setdefault(decision, []).append(prediction)
                by_question.setdefault(f"{decision}:{qid}", []).append(prediction)
    valid = [batch for batch in batches if batch["valid"]]
    entry = LOCAL_MODELS[provider]
    return {
        "provider": provider,
        "model": entry.model,
        "family": entry.family,
        "licence": entry.licence,
        "items_scored": sum(batch["n"] for batch in valid),
        "items": sum(batch["n"] for batch in batches),
        "question_decisions": len(predictions),
        "agreement_with_gold": accuracy(predictions) if predictions else None,
        "brier": brier_score(predictions) if predictions else None,
        "ece_15": expected_calibration_error(predictions) if predictions else None,
        "per_decision_agreement": {name: accuracy(p) for name, p in by_decision.items()},
        "per_question_agreement": {name: accuracy(p) for name, p in by_question.items()},
        "batches": len(batches),
        "batches_valid": len(valid),
        "attempts": len(attempts),
        "attempts_valid": attempts.count("success"),
        # JSON validity: the share of model answers that parsed and passed the strict schema.
        "json_validity_rate": attempts.count("success") / len(attempts) if attempts else None,
        "wall_s_total": round(sum(batch["wall_s"] for batch in batches), 1),
        "wall_s_per_batch_mean": round(sum(b["wall_s"] for b in batches) / len(batches), 1),
    }


def pair(a: str, b: str, out: Path, rows: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """How often labeller A and a candidate B would send an item to human review."""
    loaded = {}
    for provider in (a, b):
        for batch in map(json.loads, (out / f"{provider}.jsonl").open(encoding="utf-8")):
            if batch["valid"]:
                for offset, dists in enumerate(batch["distributions"]):
                    loaded.setdefault(batch["start"] + offset, {})[provider] = dists
    both = {index: d for index, d in loaded.items() if len(d) == 2}
    flagged = sum(
        any(
            find_disagreements(qid, d[a][qid], d[b][qid])
            for qid in rows[index][1]["questions"]
        )
        for index, d in both.items()
    )
    return {"items_both_valid": len(both), "flagged_rate": flagged / len(both) if both else None}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ["MAX_BATCHES"]) if os.environ.get("MAX_BATCHES") else None
    for provider in sys.argv[2:]:
        run(provider, out, limit)
    rows = slice_rows()
    providers = [p for p in LOCAL_MODELS if (out / f"{p}.jsonl").exists()]
    summary = {
        "slice": f"first {PER_DECISION} gold rows of each gold file in GOLD_FILES",
        "batch_size": BATCH,
        "candidates": {p: score(p, out, rows) for p in providers},
        "pairs_with_labeller_a": {
            p: pair("ollama-qwen", p, out, rows)
            for p in providers
            if p != "ollama-qwen" and "ollama-qwen" in providers
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for p, s in summary["candidates"].items():
        print(
            f"{p:16} agree {s['agreement_with_gold']:.3f}  json {s['json_validity_rate']:.3f}  "
            f"valid batches {s['batches_valid']}/{s['batches']}  wall {s['wall_s_total']} s"
        )


if __name__ == "__main__":
    main()
