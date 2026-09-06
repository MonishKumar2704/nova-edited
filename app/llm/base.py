"""LLM provider interface. All concrete providers implement this ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ToolCall:
    """One requested tool invocation, normalized across providers.

    `id` is provider-supplied when available (used to correlate a later
    tool result back to this call) and is synthesized otherwise - callers
    should not assume any particular format.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_call(self) -> dict[str, Any] | None:
        """Back-compat accessor for the (common) single-tool-call case.

        Phase 9's planner is expected to use `tool_calls` directly since a
        model can legally request more than one call; this property just
        keeps any Phase-0-era caller that read `.tool_call` working.
        """
        if not self.tool_calls:
            return None
        first = self.tool_calls[0]
        return {"id": first.id, "name": first.name, "arguments": first.arguments}


@dataclass
class ProviderHealth:
    available: bool
    provider_name: str
    detail: str = ""


class LLMProvider(ABC):
    """Provider-independent interface for the rest of Nova to depend on."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float = 20.0,
    ) -> LLMResponse:
        """Generate a response, optionally with structured tool-calling.

        `tools`, when given, is a list of tool specs shaped like
        `Tool.describe()` (name/description/input_schema) - see
        `app.tools.base.Tool`. `response_schema`, when given, is a
        JSON-schema dict the provider should constrain its *text* output
        to (structured output); it is ignored by providers that don't
        support the request/`tools` combination.

        Implementations must raise `app.core.errors.LLMError` (or a more
        specific NovaError subclass such as TimeoutErrorNova,
        NetworkError, RateLimitError) on failure - never a bare/uncaught
        exception, and never execute anything the model returns without
        validation (master spec section 16). Only transient failures
        (timeouts, network errors, rate limits) should be retried
        internally; validation/auth/config failures must fail fast.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Cheaply report whether this provider is configured and reachable."""
        raise NotImplementedError
