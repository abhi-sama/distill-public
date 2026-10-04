"""Measure the per-state local share at tau, the unit the exported fallback snippet routes on.

`distill eval` reports tau's coverage per answer. The export's `route_with_fallback.py` keeps a
state local only when every one of its answers clears tau, so the share of whole states it
answers locally is lower. Run with the training environment's Python, from the repository
root, for decision directories whose model is still present locally:

    venv-laya/bin/python proof/ticket_share.py <decision-dir> [<decision-dir> ...]

It writes `<dir>/ticket-share.json` next to the directory's `eval-report/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from distill.evaluation import _decisions, _predict_laya, _read_rows

PROOF = Path(__file__).resolve().parent


def main(names: list[str]) -> None:
    for name in names:
        directory = PROOF / name
        report = json.loads((directory / "eval-report" / "metrics.json").read_text("utf-8"))
        tau = float(report["tau"]["threshold"])
        rows = _read_rows(directory / "heldout-labelled.jsonl")
        predictions = _predict_laya(directory / "model", rows, "mps", 16)["predictions"]
        states: dict[int, list] = {}
        for decision, prediction in zip(_decisions(rows), predictions, strict=True):
            states.setdefault(decision["row_index"], []).append(prediction)
        local = [
            answers for answers in states.values() if min(p.confidence for p in answers) >= tau
        ]
        result = {
            "tau": tau,
            "states": len(states),
            "local_states": len(local),
            "local_state_share": len(local) / len(states),
            "local_states_all_correct": sum(all(p.correct for p in answers) for answers in local),
            # Should match eval's tau coverage; MPS kernels are not bit-deterministic.
            "answers_at_tau": sum(p.confidence >= tau for p in predictions),
            "answers": len(predictions),
            "local_answers": sum(len(answers) for answers in local),
            "local_answers_correct": sum(p.correct for answers in local for p in answers),
        }
        (directory / "ticket-share.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
        print(name, json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
