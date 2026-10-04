from __future__ import annotations

import json

import pytest

from distill.teachers import (
    DEFAULT_LABELLER_A,
    DEFAULT_LABELLER_B,
    DEFAULT_WRITER,
    LOCAL_MODELS,
    REFERENCE_ONLY,
    AnthropicApiTeacher,
    MalformedTeacherResponse,
    OpenAIApiTeacher,
    ProviderUnavailable,
    RawResponse,
    RoleRefused,
    Teacher,
    TeacherRegistry,
    UsageLogger,
    validate_json_schema,
)
from distill.teachers.base import RetryingTeacher

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "integer", "minimum": 0}},
    "required": ["label", "score"],
    "additionalProperties": False,
}


def test_labeller_defaults_are_two_local_families():
    # Local defaults: labels carry no provider terms, and the two families stay independent.
    assert DEFAULT_LABELLER_A in LOCAL_MODELS and DEFAULT_LABELLER_B in LOCAL_MODELS
    assert DEFAULT_LABELLER_A != DEFAULT_LABELLER_B
    a, b = LOCAL_MODELS[DEFAULT_LABELLER_A], LOCAL_MODELS[DEFAULT_LABELLER_B]
    assert a.family.split()[0] != b.family.split()[0]
    assert {a.licence, b.licence} == {"Apache-2.0"}
    assert DEFAULT_WRITER in LOCAL_MODELS


class SequencedTeacher(RetryingTeacher):
    def __init__(self, responses, **kwargs):
        self.responses = iter(responses)
        super().__init__(provider="fake", model="fake-v1", **kwargs)

    def _invoke_raw(self, prompt, schema):
        return RawResponse(next(self.responses))


class FakeTeacher(Teacher):
    def __init__(self, provider):
        self.provider = provider
        self.model = "fake"
        self.prompts = []

    def ask(self, prompt, schema):
        self.prompts.append(prompt)
        return {"label": self.provider, "score": 1}


def test_retries_invalid_json_then_validates_and_audits(tmp_path):
    log = UsageLogger(tmp_path / "usage.jsonl")
    teacher = SequencedTeacher(
        ["not json", '{"label":"ok","score":1}'], usage_logger=log, sleep=lambda _: None
    )

    assert teacher.ask("classify", SCHEMA) == {"label": "ok", "score": 1}
    entries = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [entry["outcome"] for entry in entries] == ["malformed", "success"]
    assert all(entry["provider"] == "fake" for entry in entries)


def test_malformed_output_fails_loudly_with_raw_text():
    teacher = SequencedTeacher(["{}", "[]"], max_attempts=2, sleep=lambda _: None)

    with pytest.raises(MalformedTeacherResponse) as raised:
        teacher.ask("classify", SCHEMA)

    assert raised.value.raw_outputs == ["{}", "[]"]
    assert "Raw text: '[]'" in str(raised.value)


def test_schema_validation_rejects_missing_and_extra_properties():
    with pytest.raises(ValueError, match="missing required"):
        validate_json_schema({"label": "x"}, SCHEMA)
    with pytest.raises(ValueError, match="unexpected"):
        validate_json_schema({"label": "x", "score": 0, "extra": True}, SCHEMA)


def test_api_teachers_are_inert_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for teacher in (AnthropicApiTeacher(), OpenAIApiTeacher()):
        with pytest.raises(ProviderUnavailable, match="inert"):
            teacher.ask("classify", SCHEMA)


def test_api_teachers_use_configured_keys_only_with_a_fake_transport(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    requests = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def anthropic_opener(request, timeout):
        requests.append(request)
        return Response(
            {
                "content": [{"type": "text", "text": '{"label":"a","score":1}'}],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        )

    def openai_opener(request, timeout):
        requests.append(request)
        return Response(
            {
                "output_text": '{"label":"b","score":2}',
                "usage": {"input_tokens": 4, "output_tokens": 5},
            }
        )

    assert AnthropicApiTeacher(opener=anthropic_opener).ask("classify", SCHEMA)["label"] == "a"
    assert OpenAIApiTeacher(opener=openai_opener).ask("classify", SCHEMA)["label"] == "b"
    assert requests[0].get_header("X-api-key") == "test-anthropic-key"
    assert requests[1].get_header("Authorization") == "Bearer test-openai-key"


def _registry():
    teachers = {name: FakeTeacher(name) for name in (*LOCAL_MODELS, *REFERENCE_ONLY)}
    return TeacherRegistry(teachers), teachers


def test_registry_returns_the_named_teacher_for_every_role_when_local():
    registry, teachers = _registry()
    for role in (registry.writer, registry.labeller, registry.reference, registry.fallback):
        assert role("ollama-gemma-dense") is teachers["ollama-gemma-dense"]
    with pytest.raises(KeyError, match="unknown teacher provider"):
        registry.writer("not-a-provider")


def test_registry_has_no_cli_or_quota_providers():
    registry, _ = _registry()
    assert not {"claude-cli", "codex-cli"} & set(registry.teachers)
    assert REFERENCE_ONLY == {"anthropic-api", "openai-api"}


def test_hosted_api_providers_are_refused_for_training_data_with_a_one_line_reason():
    registry, teachers = _registry()
    for provider in REFERENCE_ONLY:
        for role in (registry.writer, registry.labeller):
            with pytest.raises(RoleRefused) as raised:
                role(provider)
            assert "\n" not in str(raised.value)
            assert provider in str(raised.value)
        # They stay available as an evaluation reference and as an export fallback.
        assert registry.reference(provider) is teachers[provider]
        assert registry.fallback(provider) is teachers[provider]
