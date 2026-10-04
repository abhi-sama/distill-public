"""Common structured-output and audit behavior for teacher providers."""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class SchemaValidationError(ValueError):
    """A response did not conform to the requested JSON schema."""


class MalformedTeacherResponse(RuntimeError):
    """A provider repeatedly returned text that could not be used safely."""

    def __init__(self, provider: str, raw_outputs: list[str], reason: str) -> None:
        self.provider = provider
        self.raw_outputs = raw_outputs
        self.raw_text = raw_outputs[-1] if raw_outputs else ""
        super().__init__(
            f"{provider} returned malformed structured output after "
            f"{len(raw_outputs)} attempt(s): {reason}. Raw text: {self.raw_text!r}"
        )


class RetryableTeacherError(RuntimeError):
    """A transient provider failure for which the bounded retry policy applies."""


@dataclass(frozen=True)
class RawResponse:
    """The textual response and optional provider usage metadata."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class UsageLogger:
    """Append-only JSONL audit log. It deliberately never records prompts or answers."""

    _lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(
        self,
        *,
        provider: str,
        model: str | None,
        wall_time_seconds: float,
        outcome: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        **details: Any,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_time_seconds": round(wall_time_seconds, 6),
            "outcome": outcome,
        }
        record.update(details)
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def validate_json_schema(value: JsonValue, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the portable JSON-Schema subset used by teacher responses.

    The CLI backends receive the full schema themselves; this independent check is
    intentionally small and dependency-free so malformed output is never trusted.
    """

    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, item) for item in expected_types):
            raise SchemaValidationError(
                f"{path}: expected {expected!r}, got {type(value).__name__}"
            )

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: value is not one of the allowed enum values")

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise SchemaValidationError(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise SchemaValidationError(f"{path}: schema properties must be an object")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise SchemaValidationError(f"{path}: unexpected properties {sorted(extras)!r}")
        for key, child_schema in properties.items():
            if key in value:
                _require_schema(child_schema, f"{path}.properties[{key!r}]")
                validate_json_schema(value[key], child_schema, f"{path}.{key}")

    if isinstance(value, list) and "items" in schema:
        child_schema = schema["items"]
        _require_schema(child_schema, f"{path}.items")
        for index, item in enumerate(value):
            validate_json_schema(item, child_schema, f"{path}[{index}]")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: value is above maximum")


def _require_schema(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{path}: nested schema must be an object")


def _matches_type(value: JsonValue, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


class Teacher(ABC):
    """A source of schema-constrained JSON answers."""

    provider: str
    model: str | None
    # The local model server this teacher runs on, if any; labellers that share one take turns.
    local_host: str | None = None

    @abstractmethod
    def ask(self, prompt: str, schema: Mapping[str, Any]) -> JsonValue:
        """Return one answer that has passed local schema validation."""


class RetryingTeacher(Teacher):
    """Shared retry, parsing, validation, and audit logic for real providers."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        usage_logger: UsageLogger | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.provider = provider
        self.model = model
        self.usage_logger = usage_logger
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._sleep = sleep

    def ask(self, prompt: str, schema: Mapping[str, Any]) -> JsonValue:
        raw_outputs: list[str] = []
        last_reason = "no response"
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            response: RawResponse | None = None
            try:
                response = self._invoke_raw(prompt, schema)
                raw_outputs.append(response.text)
                answer = self._parse_response(response.text)
                validate_json_schema(answer, schema)
            except (json.JSONDecodeError, SchemaValidationError, ValueError) as error:
                last_reason = str(error)
                self._log(response, started, "malformed", attempt=attempt, reason=last_reason)
                if attempt < self.max_attempts:
                    self._sleep(self.retry_delay_seconds * (2 ** (attempt - 1)))
                    continue
                raise MalformedTeacherResponse(self.provider, raw_outputs, last_reason) from error
            except RetryableTeacherError:
                self._log(response, started, "error", attempt=attempt)
                if attempt < self.max_attempts:
                    self._sleep(self.retry_delay_seconds * (2 ** (attempt - 1)))
                    continue
                raise
            except Exception:
                self._log(response, started, "error", attempt=attempt)
                raise
            self._log(response, started, "success", attempt=attempt)
            return answer
        raise AssertionError("unreachable")

    def _log(
        self, response: RawResponse | None, started: float, outcome: str, **details: Any
    ) -> None:
        if self.usage_logger is not None:
            self.usage_logger.write(
                provider=self.provider,
                model=self.model,
                input_tokens=response.input_tokens if response else None,
                output_tokens=response.output_tokens if response else None,
                wall_time_seconds=time.monotonic() - started,
                outcome=outcome,
                **details,
            )

    @abstractmethod
    def _invoke_raw(self, prompt: str, schema: Mapping[str, Any]) -> RawResponse:
        """Make a single provider request and return the candidate JSON text."""

    def _parse_response(self, text: str) -> JsonValue:
        return json.loads(text)
