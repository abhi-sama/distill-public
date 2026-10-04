"""The Ollama backend and its routing, against a fake transport: no model is ever called."""

from __future__ import annotations

import io
import json
import threading
from urllib.error import HTTPError, URLError

import pytest
from conftest import LOCAL_A, LOCAL_B, FakeLabeller, fake_registry

from distill.cli import TeacherProvider
from distill.labelling import label_pending
from distill.review_queue import ReviewQueue
from distill.synthesis import make_registry
from distill.teachers import (
    DEFAULT_LABELLER_A,
    DEFAULT_LABELLER_B,
    LOCAL_MODELS,
    MalformedTeacherResponse,
    OllamaTeacher,
    ProviderUnavailable,
    UsageLogger,
)
from distill.teachers.local import LOCAL_REQUEST_TIMEOUT_SECONDS, strip_wrappers

SCHEMA = {
    "type": "object",
    "properties": {
        "item_0": {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "object",
                    "properties": {
                        "false": {"type": "number", "minimum": 0, "maximum": 1},
                        "true": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["false", "true"],
                    "additionalProperties": False,
                }
            },
            "required": ["probabilities"],
            "additionalProperties": False,
        }
    },
    "required": ["item_0"],
    "additionalProperties": False,
}
ANSWER = {"item_0": {"probabilities": {"false": 0.8, "true": 0.2}}}


class FakeOllama:
    """Plays back ``/api/chat`` replies (a dict, or an exception to raise) and keeps requests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, json.loads(request.data), timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())


def chat(content, **extra):
    return {
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "prompt_eval_count": 812,
        "eval_count": 64,
        **extra,
    }


def teacher(opener, **kwargs):
    return OllamaTeacher(
        "qwen3.5:35b-mlx",
        provider="ollama-qwen",
        base_url="http://127.0.0.1:11434",
        opener=opener,
        sleep=lambda _: None,
        **kwargs,
    )


def test_structured_output_request_carries_the_schema_and_parses_probabilities(tmp_path):
    ollama = FakeOllama(chat(json.dumps(ANSWER)))
    log = UsageLogger(tmp_path / "usage.jsonl")

    answer = teacher(ollama, usage_logger=log, options={"temperature": 0.7}).ask("label", SCHEMA)

    assert answer == ANSWER
    request, body, timeout = ollama.requests[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    # Ollama's structured output, plus the schema in the prompt for engines that ignore it.
    assert body["format"] == SCHEMA
    assert json.dumps(SCHEMA) in body["messages"][0]["content"]
    assert body["stream"] is False and body["think"] is False
    assert body["options"] == {"num_ctx": 32768, "temperature": 0.7}
    assert timeout == LOCAL_REQUEST_TIMEOUT_SECONDS
    [record] = [json.loads(line) for line in (tmp_path / "usage.jsonl").open()]
    assert record["provider"] == "ollama-qwen" and record["model"] == "qwen3.5:35b-mlx"
    assert (record["outcome"], record["input_tokens"], record["output_tokens"]) == (
        "success",
        812,
        64,
    )


def test_local_timeout_is_retried_within_the_bounded_attempt_policy():
    ollama = FakeOllama(TimeoutError(), chat(json.dumps(ANSWER)))

    assert teacher(ollama).ask("label", SCHEMA) == ANSWER
    assert len(ollama.requests) == 2


@pytest.mark.parametrize(
    "content",
    [
        "```json\n" + json.dumps(ANSWER) + "\n```",
        json.dumps(ANSWER) + "<|eot|>",
        "\n" + json.dumps(ANSWER, indent=2) + "\n<|eot|>\n",
    ],
)
def test_fences_and_leaked_end_markers_are_stripped(content):
    assert teacher(FakeOllama(chat(content))).ask("label", SCHEMA) == ANSWER


def test_strip_wrappers_leaves_other_text_for_validation_to_reject():
    assert strip_wrappers("# Heading\n* not json") == "# Heading\n* not json"
    assert strip_wrappers('{"a": "<|eot|> inside"}') == '{"a": "<|eot|> inside"}'


def test_off_schema_answer_is_retried_then_accepted(tmp_path):
    # The MLX engine ignores `format`; one run wrapped its answer in a schema-shaped envelope.
    envelope = {"type": "object", "properties": ANSWER}
    ollama = FakeOllama(chat(json.dumps(envelope)), chat(json.dumps(ANSWER)))
    log = UsageLogger(tmp_path / "usage.jsonl")

    assert teacher(ollama, usage_logger=log).ask("label", SCHEMA) == ANSWER
    outcomes = [json.loads(line)["outcome"] for line in (tmp_path / "usage.jsonl").open()]
    assert outcomes == ["malformed", "success"]


def test_malformed_response_fails_loudly_with_the_raw_text():
    markdown = "# Labels\n\n* item_0: probably false"
    ollama = FakeOllama(*[chat(markdown)] * 3)

    with pytest.raises(MalformedTeacherResponse) as error:
        teacher(ollama).ask("label", SCHEMA)

    assert error.value.provider == "ollama-qwen"
    assert error.value.raw_outputs == [markdown] * 3
    assert len(ollama.requests) == 3


def test_a_probability_outside_zero_to_one_is_rejected():
    bad = {"item_0": {"probabilities": {"false": 1.4, "true": -0.4}}}
    with pytest.raises(MalformedTeacherResponse, match="maximum"):
        teacher(FakeOllama(*[chat(json.dumps(bad))] * 3)).ask("label", SCHEMA)


def test_a_truncated_answer_is_malformed_and_says_why():
    cut = chat(json.dumps(ANSWER)[:20], done_reason="length")
    with pytest.raises(MalformedTeacherResponse, match="truncated at num_ctx=32768"):
        teacher(FakeOllama(cut, cut, cut)).ask("label", SCHEMA)


def test_an_unreachable_ollama_is_not_retried():
    ollama = FakeOllama(URLError("connection refused"))
    with pytest.raises(ProviderUnavailable, match="not reachable at http://127.0.0.1:11434"):
        teacher(ollama).ask("label", SCHEMA)
    assert len(ollama.requests) == 1


def test_a_model_that_is_not_pulled_names_the_pull_command():
    missing = HTTPError(
        "http://127.0.0.1:11434/api/chat", 404, "Not Found", {}, io.BytesIO(b'{"error":"x"}')
    )
    with pytest.raises(ProviderUnavailable, match="ollama pull qwen3.5:35b-mlx"):
        teacher(FakeOllama(missing)).ask("label", SCHEMA)


def test_the_ollama_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("DISTILL_OLLAMA_URL", "http://127.0.0.1:11500/")
    ollama = FakeOllama(chat(json.dumps(ANSWER)))
    OllamaTeacher("m", opener=ollama).ask("label", SCHEMA)
    assert ollama.requests[0][0].full_url == "http://127.0.0.1:11500/api/chat"


def test_make_registry_registers_the_default_local_pair_and_every_cli_choice(tmp_path):
    registry = make_registry(tmp_path / "usage.jsonl")

    for provider in (DEFAULT_LABELLER_A, DEFAULT_LABELLER_B):
        local = registry.teachers[provider]
        assert isinstance(local, OllamaTeacher)
        assert local.model == LOCAL_MODELS[provider].model
    # Every registered teacher can be named on the command line, and nothing more.
    assert {choice.value for choice in TeacherProvider} == set(registry.teachers)
    assert {"anthropic-api", "openai-api"} <= set(registry.teachers)
    assert not any(name.endswith("-cli") for name in registry.teachers)


def test_make_registry_can_override_a_models_id(tmp_path):
    registry = make_registry(tmp_path / "usage.jsonl", models={LOCAL_A: "other:1b"})

    assert registry.teachers[LOCAL_A].model == "other:1b"
    assert registry.teachers[LOCAL_B].model == LOCAL_MODELS[LOCAL_B].model


def test_dense_synthesis_candidates_have_auditable_local_tags():
    assert {
        provider: LOCAL_MODELS[provider].model
        for provider in ("ollama-gemma-dense", "ollama-glimmer-dense", "ollama-qwen-dense")
    } == {
        "ollama-gemma-dense": "gemma4:31b-nvfp4",
        "ollama-glimmer-dense": "muse-glimmer:30b-mlx",
        "ollama-qwen-dense": "qwen3.6:27b-mlx",
    }


class HostedLabeller(FakeLabeller):
    """A fake labeller on a local model host that records how many requests overlap."""

    def __init__(self, provider, questions, host, tracker):
        super().__init__(provider, questions)
        self.local_host = host
        self.tracker = tracker

    def ask(self, prompt, schema):
        with self.tracker["lock"]:
            self.tracker["active"] += 1
            self.tracker["peak"] = max(self.tracker["peak"], self.tracker["active"])
        self.tracker["both_started"].wait(timeout=0.05)
        try:
            return super().ask(prompt, schema)
        finally:
            with self.tracker["lock"]:
                self.tracker["active"] -= 1
                if self.tracker["peak"] == 2:
                    self.tracker["both_started"].set()


@pytest.mark.parametrize(("host_b", "peak"), [("http://127.0.0.1:11434", 1), (None, 2)])
def test_labellers_on_one_local_host_take_turns(tmp_path, questions, rows, host_b, peak):
    # Two models on one Ollama would evict each other every batch; other pairs run together.
    tracker = {"lock": threading.Lock(), "active": 0, "peak": 0, "both_started": threading.Event()}
    a = HostedLabeller(LOCAL_A, questions, "http://127.0.0.1:11434", tracker)
    b = HostedLabeller(LOCAL_B, questions, host_b, tracker)
    with ReviewQueue(tmp_path / "queue.db") as queue:
        queue.add(rows, split="train")

        report = label_pending(queue, fake_registry(a, b), batch_size=1)

    assert report.labelled == {"a": 3, "b": 3}
    assert tracker["peak"] == peak
