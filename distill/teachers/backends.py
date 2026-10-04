"""Optional hosted-API references. Both are inert until their API key is set.

They exist only so ``distill eval --reference`` can compare a model with a hosted answer. The
registry refuses them for ``init``, ``synth`` and ``label``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .base import RawResponse, RetryingTeacher, UsageLogger


class ProviderUnavailable(RuntimeError):
    """An optional provider was selected without its already-configured credential."""


class _OptionalApiTeacher(RetryingTeacher):
    env_var: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._api_key = os.environ.get(self.env_var)
        super().__init__(*args, **kwargs)

    def _require_key(self) -> str:
        if not self._api_key:
            raise ProviderUnavailable(
                f"{self.provider} is inert because {self.env_var} is not set; no request was made"
            )
        return self._api_key


class AnthropicApiTeacher(_OptionalApiTeacher):
    """Anthropic's API, strictly inert until ``ANTHROPIC_API_KEY`` already exists."""

    env_var = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        *,
        usage_logger: UsageLogger | None = None,
        opener: Callable[..., Any] = urlopen,
        **retry_options: Any,
    ) -> None:
        self._opener = opener
        super().__init__(
            provider="anthropic-api", model=model, usage_logger=usage_logger, **retry_options
        )

    def _invoke_raw(self, prompt: str, schema: Mapping[str, Any]) -> RawResponse:
        key = self._require_key()
        request = Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": _schema_prompt(prompt, schema)}],
                }
            ).encode(),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        payload = _read_json(self._opener, request)
        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        usage = payload.get("usage", {})
        return RawResponse(text, usage.get("input_tokens"), usage.get("output_tokens"))


class OpenAIApiTeacher(_OptionalApiTeacher):
    """OpenAI's Responses API, inert until ``OPENAI_API_KEY`` already exists."""

    env_var = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "gpt-5",
        *,
        usage_logger: UsageLogger | None = None,
        opener: Callable[..., Any] = urlopen,
        **retry_options: Any,
    ) -> None:
        self._opener = opener
        super().__init__(
            provider="openai-api", model=model, usage_logger=usage_logger, **retry_options
        )

    def _invoke_raw(self, prompt: str, schema: Mapping[str, Any]) -> RawResponse:
        key = self._require_key()
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": self.model,
                    "input": prompt,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "teacher_answer",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                }
            ).encode(),
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            method="POST",
        )
        payload = _read_json(self._opener, request)
        text = payload.get("output_text") or _openai_output_text(payload)
        usage = payload.get("usage", {})
        return RawResponse(text, usage.get("input_tokens"), usage.get("output_tokens"))


def _schema_prompt(prompt: str, schema: Mapping[str, Any]) -> str:
    return f"{prompt}\n\nReturn only JSON conforming to this schema:\n{json.dumps(schema)}"


def _read_json(opener: Callable[..., Any], request: Request) -> dict[str, Any]:
    try:
        with opener(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"teacher API request failed ({error.code}): {detail}") from error


def _openai_output_text(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)
