"""Fast offline tests for the fixed evaluation metrics and the readiness gate."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from distill.evaluation import (
    REFERENCE_BATCH_ROWS,
    Prediction,
    _local_model_path,
    _predict_reference,
    _predict_teacher_rows,
    _write_report,
    accuracy,
    brier_score,
    expected_calibration_error,
    metric_summary,
    pass_bar,
    per_question,
    select_tau,
    selective_accuracy_curve,
)


def predictions():
    return [
        Prediction((0.90, 0.10), 0),  # correct, confidence .9
        Prediction((0.80, 0.20), 0),  # correct, confidence .8
        Prediction((0.70, 0.30), 1),  # incorrect, confidence .7
        Prediction((0.60, 0.40), 1),  # incorrect, confidence .6
    ]


def test_ece_and_brier_have_known_multiclass_answers():
    items = predictions()
    # Every item occupies its own 15-bin bucket: mean |confidence - correctness|.
    assert expected_calibration_error(items) == pytest.approx((0.1 + 0.2 + 0.7 + 0.6) / 4)
    assert brier_score(items) == pytest.approx((0.02 + 0.08 + 0.98 + 0.72) / 4)


def test_deliberately_miscalibrated_predictions_have_high_ece():
    items = [Prediction((0.99, 0.01), 1) for _ in range(10)]
    assert expected_calibration_error(items) == pytest.approx(0.99)


def test_selective_curve_and_tau_choose_maximum_coverage_at_target():
    curve = selective_accuracy_curve(predictions())
    assert curve[0] == {"threshold": 0.0, "coverage": 1.0, "n": 4, "accuracy": 0.5}
    assert select_tau(predictions(), 1.0) == {
        "threshold": 0.8,
        "coverage": 0.5,
        "n": 2,
        "accuracy": 1.0,
    }


def test_pass_bar_compares_with_a_reference_when_one_answered():
    assert pass_bar(local_ece=0.08, gold_accuracy=0.90, reference_gold_accuracy=0.95)["ready"]
    failed = pass_bar(local_ece=0.10, gold_accuracy=0.89, reference_gold_accuracy=0.95)
    assert not failed["ready"] and failed["reference_measured"]
    assert failed["ece_excess"] == pytest.approx(0.02)
    assert failed["gold_accuracy_excess"] == pytest.approx(0.01)
    assert "trails the reference" in failed["failures"][1]


def test_pass_bar_without_a_reference_reports_no_reference_measured_instead_of_failing():
    gate = pass_bar(local_ece=0.01, gold_accuracy=0.99)
    assert gate["ready"] and not gate["failures"]
    assert not gate["reference_measured"] and gate["gold_accuracy_excess"] is None
    assert any("no reference measured" in note for note in gate["notes"])
    # The ECE bound and a missing gold set still fail with no reference.
    assert not pass_bar(local_ece=0.20, gold_accuracy=0.99)["ready"]
    assert not pass_bar(local_ece=0.01, gold_accuracy=None)["ready"]


def test_base_resolution_uses_only_the_local_hub_cache(monkeypatch):
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "/tmp/cached-laya"

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    errors = ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = RuntimeError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)

    assert _local_model_path("convaiinnovations/laya") == Path("/tmp/cached-laya")
    assert calls == [
        (
            "convaiinnovations/laya",
            {
                "allow_patterns": [
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                ],
                "local_files_only": True,
            },
        )
    ]


def test_teacher_baseline_batches_states_and_scores_every_decision():
    questions = {
        "escalate": {"type": "noul", "instructions": "Escalate?"},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    }
    rows = [
        {
            "state": {"text": f"ticket {n}"},
            "questions": questions,
            "gold": {"escalate": {"label": "true"}, "urgency": {"label": n % 2}},
        }
        for n in range(12)
    ]

    class Teacher:
        def __init__(self):
            self.schemas = []

        def ask(self, prompt, schema):
            self.schemas.append(schema)
            return {name: {"escalate": "true", "urgency": "0"} for name in schema["required"]}

    teacher = Teacher()
    output, calls = _predict_teacher_rows(teacher, rows)

    assert calls == 2 and len(teacher.schemas[0]["required"]) == REFERENCE_BATCH_ROWS
    assert len(output) == 24
    assert accuracy([item["prediction"] for item in output]) == 18 / 24


def test_reference_report_names_its_provider_and_prices_only_local_models_at_zero(monkeypatch):
    from distill.teachers import Teacher, TeacherRegistry

    class Answerer(Teacher):
        def __init__(self, provider):
            self.provider, self.model = provider, "m"

        def ask(self, prompt, schema):
            return {name: {"escalate": "true"} for name in schema["required"]}

    registry = TeacherRegistry(
        {"ollama-qwen": Answerer("ollama-qwen"), "openai-api": Answerer("openai-api")}
    )
    monkeypatch.setattr("distill.synthesis.make_registry", lambda *a, **k: registry)
    questions = {"escalate": {"type": "noul", "instructions": "Escalate?"}}
    rows = [
        {"state": {"text": "t"}, "questions": questions, "gold": {"escalate": {"label": "true"}}}
    ]

    local = _predict_reference("ollama-qwen", None, rows, rows)
    hosted = _predict_reference("openai-api", None, rows, rows)

    assert local["provider"] == "ollama-qwen" and local["gold"]["accuracy"] == 1.0
    assert local["cost_per_1000_decisions_usd"] == 0.0
    assert hosted["provider"] == "openai-api" and hosted["cost_per_1000_decisions_usd"] is None


def _rows(labels):
    questions = {"escalate": {"type": "noul", "instructions": "Escalate?"}}
    return [
        {"state": f"ticket {n}", "questions": questions, "gold": {"escalate": {"label": label}}}
        for n, label in enumerate(labels)
    ]


def test_per_question_scores_each_question_id():
    rows = _rows(["true", "false"])
    result = per_question([Prediction((0.2, 0.8), 1), Prediction((0.3, 0.7), 0)], rows)

    assert result == {"escalate": {"n": 2, "accuracy": 0.5, "ece_15": pytest.approx(0.45)}}


def test_report_renders_latency_and_the_per_question_table(tmp_path):
    rows = _rows(["true"])
    summary = metric_summary([Prediction((0.2, 0.8), 1)])
    questions = per_question([Prediction((0.2, 0.8), 1)], rows)
    latency = {
        "decisions": 1,
        "total_seconds": 0.01,
        "mean_ms_per_decision": 10.0,
        "single_state_ms_median": 12.0,
        "load_seconds": 2.0,
    }
    model = {
        "heldout": summary,
        "gold": summary,
        "latency": latency,
        "per_question": {"heldout": questions, "gold": questions},
    }
    report = {
        "target_accuracy": 0.97,
        "local": model,
        "base_laya": model,
        "reference": None,
        "adversarial_slices": {},
        "tau": None,
        "pass_bar": pass_bar(local_ece=0.0, gold_accuracy=1.0, reference_gold_accuracy=None),
    }

    _write_report(tmp_path, report)

    text = (tmp_path / "REPORT.md").read_text()
    assert "12.0 ms median for one state" in text
    assert "| escalate | 1 | 100.00% | 100.00% | n/a |" in text
