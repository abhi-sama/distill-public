"""Fast offline checks for ``distill export``."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from distill.exporting import ExportError, render_report, run_export


def _checkpoint(path):
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder" / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("tiny", encoding="utf-8")
    (path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "data": {"rows": 12},
                "options": {"epochs": 4, "device": "cpu", "seed": 7},
            }
        ),
        encoding="utf-8",
    )


def _evaluation(path, tau=0.81):
    path.write_text(
        json.dumps(
            {
                "local": {
                    "heldout": {"accuracy": 0.98, "ece_15": 0.02, "brier": 0.04},
                    "latency": {"mean_ms_per_decision": 12.3456},
                },
                "base_laya": {
                    "heldout": {"accuracy": 0.7, "ece_15": 0.2, "brier": 0.3},
                    "latency": {"mean_ms_per_decision": 9.8},
                    "cost_per_1000_decisions_usd": 0.0,
                },
                "reference": None,
                "tau": {"threshold": tau, "coverage": 0.75, "accuracy": 0.99, "n": 9},
            }
        ),
        encoding="utf-8",
    )


def test_export_copies_laya_layout_and_uses_tau_from_evaluation(tmp_path):
    model = tmp_path / "trained"
    _checkpoint(model)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation, tau=0.87654)
    output = tmp_path / "export"

    result = run_export(
        model=model,
        evaluation=evaluation,
        output=output,
        base_checkpoint="local/base",
        teachers=("ollama-qwen", "ollama-gemma"),
    )

    assert result["tau"]["threshold"] == 0.87654
    assert {"model.safetensors", "encoder", "tokenizer", "rl_agent_config.json"} <= {
        item.name for item in output.iterdir()
    }
    assert json.loads((output / "export.json").read_text())["tau"]["threshold"] == 0.87654
    snippet = (output / "route_with_fallback.py").read_text()
    assert 'EXPORT["tau"]["threshold"]' in snippet
    assert "0.87654" not in snippet
    assert "laya-serve" in (output / "SERVE.md").read_text()
    assert "Router" in (output / "sitecustomize.py").read_text()
    report = (output / "REPORT.md").read_text()
    assert "0.98" in report and "12.3456" in report
    assert "ollama-qwen, ollama-gemma" in report
    assert "No reference measured." in report


def test_report_refuses_placeholder_or_missing_required_metrics(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation)
    report = json.loads(evaluation.read_text())
    report["local"]["heldout"]["accuracy"] = "TBD"

    with pytest.raises(ExportError, match="placeholder"):
        render_report(
            report,
            {
                "base_checkpoint": "base",
                "labelled_rows": 2,
                "teachers": "x",
                "training_recipe": "x",
                "date": "2026-09-24",
            },
        )


def test_export_rejects_evaluation_without_a_measured_tau(tmp_path):
    model = tmp_path / "trained"
    _checkpoint(model)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation)
    report = json.loads(evaluation.read_text())
    report["tau"] = None
    evaluation.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ExportError, match="selected no tau"):
        run_export(model=model, evaluation=evaluation, output=tmp_path / "export")


PROVENANCE_KEYS = ("base_checkpoint", "labelled_rows", "teachers", "training_recipe", "date")


def test_report_states_a_failed_readiness_gate(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation)
    data = json.loads(evaluation.read_text())
    data["pass_bar"] = {"ready": False, "failures": ["gold accuracy trails the reference by 0.0436"]}

    report = render_report(data, {key: "x" for key in PROVENANCE_KEYS})

    assert "NOT READY:\n- gold accuracy trails the reference by 0.0436" in report
    assert "Not recorded" in render_report(
        {**data, "pass_bar": None}, dict.fromkeys(PROVENANCE_KEYS, "x")
    )


def test_generated_python_files_pass_the_repository_lint(tmp_path):
    beside_python = Path(sys.executable).with_name("ruff")
    ruff = str(beside_python) if beside_python.exists() else shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff is not installed")
    model = tmp_path / "trained"
    _checkpoint(model)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation)
    output = tmp_path / "export"
    run_export(model=model, evaluation=evaluation, output=output)

    files = [str(output / "route_with_fallback.py"), str(output / "sitecustomize.py")]
    result = subprocess.run(
        [ruff, "check", "--config", str(Path(__file__).parents[1] / "pyproject.toml"), *files],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout


def test_export_fallback_defaults_to_none_and_flags_low_confidence(tmp_path):
    model = tmp_path / "trained"
    _checkpoint(model)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation)

    run_export(model=model, evaluation=evaluation, output=tmp_path / "export")

    snippet = (tmp_path / "export" / "route_with_fallback.py").read_text()
    compile(snippet, "route_with_fallback.py", "exec")
    assert '"low_confidence": not confident' in snippet
    assert "make_registry" not in snippet and "teacher" not in snippet


def test_export_fallback_is_configurable(tmp_path):
    model = tmp_path / "trained"
    _checkpoint(model)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation)

    run_export(
        model=model, evaluation=evaluation, output=tmp_path / "export", fallback="ollama-qwen"
    )

    snippet = (tmp_path / "export" / "route_with_fallback.py").read_text()
    compile(snippet, "route_with_fallback.py", "exec")
    assert "make_registry().fallback('ollama-qwen')" in snippet
    assert "claude" not in snippet and "codex" not in snippet
