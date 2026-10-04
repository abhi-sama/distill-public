"""Package a trained Laya decision model with its measured evaluation evidence.

This module deliberately only copies already-produced artifacts.  It never loads a
model, asks a teacher, or recalculates an evaluation metric, so export remains fast
and offline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from .training import checkpoint_layout


class ExportError(RuntimeError):
    """An actionable error from the export workflow."""


_PLACEHOLDERS = {"", "-", "n/a", "na", "none", "null", "tbd", "unknown", "<value>"}


def run_export(
    *,
    model: Path,
    evaluation: Path,
    output: Path,
    base_checkpoint: str | None = None,
    teachers: tuple[str, ...] = (),
    fallback: str | None = None,
) -> dict[str, Any]:
    """Copy a finished checkpoint and render a report from the training and eval artifacts.

    ``base_checkpoint`` and ``teachers`` are optional because the checkpoint may not
    have recorded them.  Their absence is stated explicitly in the report rather than
    being guessed from the current environment.
    """
    _check_checkpoint(model)
    training = _read_json(model / "metrics.json", "training metrics")
    report = _read_json(_metrics_path(evaluation), "evaluation metrics")
    tau = _tau(report)
    if output.exists():
        raise ExportError(f"export output already exists: {output}; choose a new --output path")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.writing-", dir=output.parent))
    try:
        _copy_checkpoint(model, staging)
        provenance = _provenance(training, base_checkpoint, teachers)
        export_data = {"tau": tau, "provenance": provenance}
        (staging / "export.json").write_text(
            json.dumps(export_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "REPORT.md").write_text(render_report(report, provenance), encoding="utf-8")
        (staging / "route_with_fallback.py").write_text(_routing_snippet(fallback), encoding="utf-8")
        (staging / "sitecustomize.py").write_text(_serve_patch(), encoding="utf-8")
        launcher = staging / "serve-local.sh"
        launcher.write_text(_serve_launcher(), encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (staging / "serve.env").write_text(_serve_environment(), encoding="utf-8")
        (staging / "SERVE.md").write_text(_serve_readme(), encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return export_data


def render_report(evaluation: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    """Render measured T6 values without placeholder values or presentation rounding."""
    tau = _tau(evaluation)
    local = _mapping(evaluation, "local")
    heldout = _mapping(local, "heldout")
    latency = _mapping(local, "latency")
    local_rows = [
        ("Accuracy", _number(heldout, "accuracy")),
        ("ECE (15 bins)", _number(heldout, "ece_15")),
        ("Brier", _number(heldout, "brier")),
        ("Tau", _number(tau, "threshold")),
        ("Local coverage", _number(tau, "coverage")),
        ("Latency (mean ms/decision)", _number(latency, "mean_ms_per_decision")),
    ]
    lines = [
        "# Distill model card",
        "",
        "## Measured local model (held-out)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        *[f"| {name} | {value} |" for name, value in local_rows],
        "",
        "## Baselines from evaluation",
        "",
        _baseline_table("Base Laya", evaluation.get("base_laya")),
        "",
        _reference_table(evaluation.get("reference")),
        "",
        "## Readiness gate",
        "",
        *_gate_lines(evaluation.get("pass_bar")),
        "",
        "## Provenance",
        "",
        f"- Base checkpoint: {provenance['base_checkpoint']}",
        f"- Labelled rows: {provenance['labelled_rows']}",
        f"- Labelling teachers: {provenance['teachers']}",
        f"- Training recipe: {provenance['training_recipe']}",
        f"- Export date: {provenance['date']}",
        "",
        "The routing snippet uses `answer_confidence`, the calibrated confidence measured by distill eval.",
        "",
    ]
    rendered = "\n".join(lines)
    if (
        re.search(r"(?:^|[^a-z0-9])(tbd|nan)(?:$|[^a-z0-9])", rendered.casefold())
        or "<value>" in rendered
    ):
        raise ExportError("REPORT.md refuses placeholder metric values")
    return rendered


def _gate_lines(gate: Any) -> list[str]:
    """State the readiness gate verdict as measured, including every failure it listed."""
    if not isinstance(gate, Mapping) or not isinstance(gate.get("ready"), bool):
        return ["Not recorded in the supplied evaluation."]
    if gate["ready"]:
        if gate.get("reference_measured"):
            return ["READY: ECE <= 0.08 and gold accuracy within 5 points of the reference."]
        return [
            "READY on ECE <= 0.08. No reference measured: gold accuracy was not compared with "
            "any reference."
        ]
    return ["NOT READY:", *[f"- {failure}" for failure in gate.get("failures", [])]]


def _check_checkpoint(model: Path) -> None:
    if not model.is_dir():
        raise ExportError(f"model checkpoint not found: {model}")
    missing = checkpoint_layout(model) - {path.name for path in model.iterdir()}
    if missing:
        raise ExportError(f"model is not a Laya checkpoint; missing {sorted(missing)}")
    if not (model / "metrics.json").is_file():
        raise ExportError(f"model checkpoint has no T5 metrics.json: {model}")


def _metrics_path(evaluation: Path) -> Path:
    return evaluation / "metrics.json" if evaluation.is_dir() else evaluation


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExportError(f"{description} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ExportError(f"{description} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExportError(f"{description} must be a JSON object: {path}")
    return value


def _copy_checkpoint(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        if child.name == "metrics.json":
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ExportError(f"evaluation output is missing {key}")
    return nested


def _number(value: Mapping[str, Any], key: str) -> str:
    number = value.get(key)
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ExportError(f"REPORT.md refuses placeholder or missing metric {key}")
    return str(number)


def _tau(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    tau = evaluation.get("tau")
    if not isinstance(tau, Mapping):
        raise ExportError(
            "evaluation selected no tau; cannot export a fallback route without a measured cutoff"
        )
    _number(tau, "threshold")
    _number(tau, "coverage")
    return tau


def _reference_table(reference: Any) -> str:
    if not isinstance(reference, Mapping):
        return "### Reference\n\nNo reference measured."
    return _baseline_table(
        f"Reference ({reference.get('provider')}, {reference.get('model')})", reference
    )


def _baseline_table(name: str, baseline: Any) -> str:
    if not isinstance(baseline, Mapping):
        return f"### {name}\n\nNot measured in evaluation output."
    heldout = baseline.get("heldout")
    latency = baseline.get("latency")
    if not isinstance(heldout, Mapping) or not isinstance(latency, Mapping):
        return f"### {name}\n\nNot measured in evaluation output."
    values = (
        ("Accuracy", heldout.get("accuracy")),
        ("ECE (15 bins)", heldout.get("ece_15")),
        ("Brier", heldout.get("brier")),
        ("Latency (mean ms/decision)", latency.get("mean_ms_per_decision")),
        ("Cost per 1k decisions (USD)", baseline.get("cost_per_1000_decisions_usd")),
    )
    lines = [f"### {name}", "", "| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {label} | {_display_metric(value)} |" for label, value in values)
    return "\n".join(lines)


def _display_metric(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Not measured in evaluation output."
    return str(value)


def _provenance(
    training: Mapping[str, Any], base_checkpoint: str | None, teachers: tuple[str, ...]
) -> dict[str, str | int]:
    data = _mapping(training, "data")
    rows = data.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise ExportError("training metrics is missing the labelled row count")
    options = _mapping(training, "options")
    recipe = ", ".join(f"{key}={value}" for key, value in sorted(options.items()))
    return {
        "base_checkpoint": _provenance_value(base_checkpoint),
        "labelled_rows": rows,
        "teachers": ", ".join(teachers)
        if teachers
        else "Not recorded in supplied training artifacts.",
        "training_recipe": recipe,
        "date": date.today().isoformat(),
    }


def _provenance_value(value: str | None) -> str:
    if value is None or value.strip().casefold() in _PLACEHOLDERS:
        return "Not recorded in supplied training artifacts."
    return value.strip()


def _routing_snippet(fallback: str | None = None) -> str:
    if fallback is None:
        return '''"""Route one decision with the exported local model. No fallback is configured.

When the local model is not confident, the local answer is still returned, flagged
low-confidence, so the caller can send it to a person or another system.
"""

import json
from pathlib import Path

import laya

MODEL_DIR = Path(__file__).resolve().parent
EXPORT = json.loads((MODEL_DIR / "export.json").read_text(encoding="utf-8"))
TAU = float(EXPORT["tau"]["threshold"])
local_model = laya.load(str(MODEL_DIR))


def decide(state, questions):
    local = local_model.predict(state, questions)
    confident = all(answer["answer_confidence"] >= TAU for answer in local["answers"].values())
    return {"source": "local", "low_confidence": not confident, "result": local}
'''
    return '''"""Route one decision: the exported local model first, then a fallback teacher."""

import json
from pathlib import Path

import laya

from distill.synthesis import make_registry

MODEL_DIR = Path(__file__).resolve().parent
EXPORT = json.loads((MODEL_DIR / "export.json").read_text(encoding="utf-8"))
TAU = float(EXPORT["tau"]["threshold"])
local_model = laya.load(str(MODEL_DIR))
teacher = make_registry().fallback(FALLBACK_PROVIDER)


def teacher_schema(questions):
    properties = {}
    for question_id, question in questions.items():
        if question["type"] == "choice":
            criteria = question["criteria"]
            properties[question_id] = {"type": "string", "enum": list(criteria)}
        elif question["type"] == "score":
            properties[question_id] = {
                "type": "integer",
                "minimum": 0,
                "maximum": len(question["criteria"]) - 1,
            }
        else:
            properties[question_id] = {"type": "boolean"}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def decide(state, questions):
    local = local_model.predict(state, questions)
    if all(answer["answer_confidence"] >= TAU for answer in local["answers"].values()):
        return {"source": "local", "result": local}
    prompt = "Classify this decision.\\nQuestions: " + json.dumps(questions) + "\\nState: " + json.dumps(state)
    return {"source": teacher.provider, "answers": teacher.ask(prompt, teacher_schema(questions))}
'''.replace("FALLBACK_PROVIDER", repr(fallback))


def _serve_patch() -> str:
    return '''"""Configure laya-serve to load this exported local checkpoint."""

import os
from pathlib import Path

import laya.serve
from laya import Router


def build_router():
    model_dir = Path(os.environ["DISTILL_EXPORT_MODEL_DIR"])
    router = Router(
        models={"english": str(model_dir)},
        device=os.environ.get("LAYA_DEVICE") or None,
    )
    if os.environ.get("LAYA_PRELOAD", "1").strip().lower() in {"1", "true", "yes", "on"}:
        router.preload(["english"])
    return router


laya.serve.build_router = build_router
'''


def _serve_launcher() -> str:
    return """#!/bin/sh
set -eu
MODEL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DISTILL_EXPORT_MODEL_DIR="$MODEL_DIR"
export PYTHONPATH="$MODEL_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec laya-serve "$@"
"""


def _serve_environment() -> str:
    return """# Source this file before ./serve-local.sh. Pick a free local port if 8001 is occupied.
LAYA_HOST=127.0.0.1
LAYA_PORT=8001
LAYA_DEVICE=cpu
LAYA_PRELOAD=1
LAYA_MODELS=english
"""


def _serve_readme() -> str:
    return """# Serve this export

Install Laya with its serve extra, then source the checked-in configuration and run the
launcher. The launcher invokes Laya's own `laya-serve` command while its local
`sitecustomize.py` maps the `english` server slot to this checkpoint.

```sh
set -a; . ./serve.env; set +a
./serve-local.sh
```

The default is loopback on port 8001 so it does not take over Laya's conventional port 8000.
"""
