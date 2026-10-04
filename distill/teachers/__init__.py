"""Structured-output teacher backends and the provider registry."""

from .backends import (
    AnthropicApiTeacher,
    OpenAIApiTeacher,
    ProviderUnavailable,
)
from .base import (
    MalformedTeacherResponse,
    RawResponse,
    Teacher,
    UsageLogger,
    validate_json_schema,
)
from .local import LOCAL_MODELS, LocalModel, OllamaTeacher
from .registry import REFERENCE_ONLY, RoleRefused, TeacherRegistry

# Local models are the default for every role. The dual labellers span two independent
# open-weight families, both run through Ollama, so labels carry no hosted provider's terms
# (see docs/T10-LABELLER-BAKEOFF.md). The writer is the dense local model that wrote the
# open-data run's states.
DEFAULT_LABELLER_A = "ollama-qwen"
DEFAULT_LABELLER_B = "ollama-gemma"
DEFAULT_WRITER = "ollama-gemma-dense"

__all__ = [
    "AnthropicApiTeacher",
    "DEFAULT_LABELLER_A",
    "DEFAULT_LABELLER_B",
    "DEFAULT_WRITER",
    "LOCAL_MODELS",
    "LocalModel",
    "MalformedTeacherResponse",
    "OllamaTeacher",
    "OpenAIApiTeacher",
    "ProviderUnavailable",
    "REFERENCE_ONLY",
    "RawResponse",
    "RoleRefused",
    "Teacher",
    "TeacherRegistry",
    "UsageLogger",
    "validate_json_schema",
]
