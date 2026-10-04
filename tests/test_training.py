"""Fast training-unit tests: no torch, Laya, model, or network required."""

import json
import math

import pytest

from distill.training import (
    TrainingError,
    checkpoint_layout,
    cosine_learning_rate,
    decode_state,
    fit_temperature,
    optimizer_updates,
    split_rows,
    write_checkpoint,
)


def test_split_rows_is_deterministic_exact_and_never_leaks_rows():
    rows = [{"id": f"row-{index}"} for index in range(23)]
    train, calibration, test = split_rows(rows, seed=7)

    assert (len(train), len(calibration), len(test)) == (19, 2, 2)
    assert {row["id"] for row in train + calibration + test} == {row["id"] for row in rows}
    assert split_rows(rows, seed=7) == (train, calibration, test)
    assert not ({row["id"] for row in train} & {row["id"] for row in calibration})


def test_split_rows_requires_enough_data_for_all_three_slices():
    with pytest.raises(TrainingError, match="at least 10"):
        split_rows([{}] * 9)


def test_cosine_schedule_counts_partial_accumulation_and_reaches_minimum():
    assert optimizer_updates(item_count=17, micro_batch=8, grad_accum=4, epochs=4) == 4
    assert cosine_learning_rate(1e-4, 0, 4) == pytest.approx(1e-4)
    assert cosine_learning_rate(1e-4, 4, 4) == pytest.approx(1e-6)
    assert cosine_learning_rate(1e-4, 2, 4) == pytest.approx(5.05e-5)


def test_temperature_fit_recovers_soft_label_temperature_without_torch():
    samples = []
    true_temperature = 1.7
    for index in range(20):
        logits = [math.sin(index + offset) * 3 for offset in range(3 + index % 2)]
        maximum = max(value / true_temperature for value in logits)
        weights = [math.exp(value / true_temperature - maximum) for value in logits]
        samples.append((logits, [weight / sum(weights) for weight in weights]))

    assert fit_temperature(samples) == pytest.approx(true_temperature, rel=0.01)
    assert fit_temperature(samples[:9]) == 1.0


class _FakeConfig:
    def save_pretrained(self, path):
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")


class _FakeEncoder:
    config = _FakeConfig()


class _FakeModel:
    encoder = _FakeEncoder()

    def state_dict(self):
        return {"encoder.weight": "tiny"}


class _FakeTokenizer:
    def save_pretrained(self, path):
        path.mkdir()
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_checkpoint_layout_is_complete_and_is_published_only_after_writing(tmp_path):
    output = tmp_path / "model"

    def save_weights(state_dict, target):
        assert state_dict == {"encoder.weight": "tiny"}
        with open(target, "w", encoding="utf-8") as file:
            file.write("weights")

    write_checkpoint(
        output,
        _FakeModel(),
        _FakeTokenizer(),
        {"encoder": "fake", "temperature_by_options": {"choice:2": 2.0}},
        [1.1, 1.2, 1.3],
        {"items": 8},
        save_weights=save_weights,
    )

    assert checkpoint_layout(output) <= {path.name for path in output.iterdir()}
    config = json.loads((output / "rl_agent_config.json").read_text())
    assert config["fine_tuned"] is True
    assert config["temperature"] == [1.1, 1.2, 1.3]
    assert "temperature_by_options" not in config


def test_checkpoint_writer_does_not_leave_a_final_directory_when_write_fails(tmp_path):
    output = tmp_path / "model"

    def fail_weights(_state_dict, _target):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_checkpoint(
            output,
            _FakeModel(),
            _FakeTokenizer(),
            {},
            [1.0, 1.0, 1.0],
            {},
            save_weights=fail_weights,
        )
    assert not output.exists()


def test_a_state_is_decoded_only_when_it_is_json_text():
    assert decode_state('{"post": "hi"}') == {"post": "hi"}
    assert decode_state({"post": "hi"}) == {"post": "hi"}
    # distill synth writes plain-text states as they are; they must survive unchanged.
    assert decode_state("A thread about bikes.") == "A thread about bikes."
