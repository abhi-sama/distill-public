"""Local teachers served by Ollama: open-weight models, no hosted provider and no quota.

Nothing leaves the Mac. Each request carries the JSON schema twice: as Ollama's ``format``
(grammar-constrained decoding on the GGUF/llama.cpp engine) and in the prompt, because the MLX
engine ignores ``format`` and only the prompt keeps its answers on schema. Either way the
answer then goes through :class:`RetryingTeacher`'s own schema validation and bounded retries,
so a model that drifts off schema fails loudly instead of being trusted.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .backends import ProviderUnavailable, _schema_prompt
from .base import JsonValue, RawResponse, RetryableTeacherError, RetryingTeacher, UsageLogger

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# A request that has made no socket progress for 15 minutes is retried by RetryingTeacher.
# This caps a silently wedged local generation while still allowing the observed 12–14 minute
# dense-model chunks to finish.
LOCAL_REQUEST_TIMEOUT_SECONDS = 900


class LocalRequestTimeout(RetryableTeacherError):
    """Ollama did not complete a local request within its bounded timeout."""


@dataclass(frozen=True)
class LocalModel:
    """One registered local teacher: its Ollama tag, family and weights licence."""

    provider: str
    model: str
    family: str
    licence: str
    # Sampling options that override the tag's Modelfile defaults.
    options: Mapping[str, Any] = field(default_factory=dict)


LOCAL_MODELS: dict[str, LocalModel] = {
    model.provider: model
    for model in (
        # Qwen's recommended non-thinking settings. The tag's own presence_penalty of 1.5
        # would penalise JSON's repeated keys.
        LocalModel(
            "ollama-qwen",
            "qwen3.5:35b-mlx",
            "Qwen3.5-35B-A3B",
            "Apache-2.0",
            {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0},
        ),
        LocalModel("ollama-gemma", "gemma4:26b", "Gemma 4 26B-A4B", "Apache-2.0"),
        # Dense synthesis candidates for the support-escalation breadth gate. They are kept
        # separate from the default labellers so a proof run cannot alter that recipe.
        LocalModel(
            "ollama-gemma-dense", "gemma4:31b-nvfp4", "Gemma 4 31B", "Apache-2.0"
        ),
        LocalModel(
            "ollama-glimmer-dense", "muse-glimmer:30b-mlx", "Muse Glimmer 30B", "Apache-2.0"
        ),
        LocalModel("ollama-qwen-dense", "qwen3.6:27b-mlx", "Qwen3.6 27B", "Apache-2.0"),
        LocalModel(
            "ollama-nemotron",
            "nemotron-3.5-lightning:30b",
            "Nemotron 3.5 Lightning 30B-A3B",
            "NVIDIA Open Model License (pulled Ollama artifact); model card: OpenMDW-1.1",
        ),
        LocalModel("ollama-glimmer", "muse-glimmer:30b", "Meta Muse Glimmer 30B", "Apache-2.0"),
    )
}

# Chat-template end markers some tags leak after the JSON (muse-glimmer ends with <|eot|>).
_TRAILING_MARKER = re.compile(r"(?:<\|[A-Za-z0-9_]{1,32}\|>|<end_of_turn>|</s>)\s*$")
_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.S)


class OllamaTeacher(RetryingTeacher):
    """A model served by a local Ollama over ``/api/chat``, with structured output."""

    def __init__(
        self,
        model: str,
        *,
        provider: str = "ollama",
        base_url: str | None = None,
        options: Mapping[str, Any] | None = None,
        num_ctx: int = 32768,
        timeout_seconds: float = LOCAL_REQUEST_TIMEOUT_SECONDS,
        keep_alive: str = "15m",
        usage_logger: UsageLogger | None = None,
        opener: Callable[..., Any] = urlopen,
        **retry_options: Any,
    ) -> None:
        super().__init__(
            provider=provider, model=model, usage_logger=usage_logger, **retry_options
        )
        base_url = base_url or os.environ.get("DISTILL_OLLAMA_URL") or DEFAULT_OLLAMA_URL
        self.base_url = base_url.rstrip("/")
        # One Ollama keeps one of these models in memory at a time on a 64 GB Mac, so
        # labellers sharing a host run one after the other rather than evicting each other.
        self.local_host = self.base_url
        self.options = {"num_ctx": num_ctx, **(options or {})}
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self._opener = opener

    @classmethod
    def registered(cls, provider: str, *, model: str | None = None, **kwargs: Any) -> OllamaTeacher:
        entry = LOCAL_MODELS[provider]
        return cls(model or entry.model, provider=provider, options=entry.options, **kwargs)

    def _invoke_raw(self, prompt: str, schema: Mapping[str, Any]) -> RawResponse:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": _schema_prompt(prompt, schema)}],
            "format": schema,
            "stream": False,
            # Thinking agreed on 1 more of 150 decisions at 2.9x the time, so it stays off.
            "think": False,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except TimeoutError as error:
            raise LocalRequestTimeout(
                f"{self.provider}: local request timed out after {self.timeout_seconds}s"
            ) from error
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 404:
                raise ProviderUnavailable(
                    f"{self.provider}: Ollama has no model {self.model!r}; run "
                    f"`ollama pull {self.model}` ({detail.strip()})"
                ) from error
            raise RuntimeError(f"Ollama request failed ({error.code}): {detail}") from error
        except URLError as error:
            raise ProviderUnavailable(
                f"{self.provider}: Ollama is not reachable at {self.base_url} "
                f"({error.reason}); start the Ollama app or `ollama serve`"
            ) from error
        if not isinstance(payload, Mapping) or payload.get("error"):
            raise RuntimeError(f"Ollama returned an error: {payload!r}"[:500])
        message = payload.get("message")
        text = message.get("content") if isinstance(message, Mapping) else None
        if payload.get("done_reason") == "length":
            # A cut-off answer is never valid JSON; say why in the malformed-response record.
            text = f"{text or ''}\n[truncated at num_ctx={self.options['num_ctx']}]"
        return RawResponse(
            text if isinstance(text, str) else "",
            payload.get("prompt_eval_count"),
            payload.get("eval_count"),
        )

    def _parse_response(self, text: str) -> JsonValue:
        return json.loads(strip_wrappers(text))


def strip_wrappers(text: str) -> str:
    """Remove a Markdown fence and trailing chat-template markers around a JSON answer."""
    text = text.strip()
    while (stripped := _TRAILING_MARKER.sub("", text).rstrip()) != text:
        text = stripped
    if match := _FENCE.match(text):
        text = match.group(1).strip()
    return text
