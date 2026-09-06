"""
Null provider: used when LLM_PROVIDER is unset/"none".

The application must keep working without an LLM configured (deterministic
routing, health endpoints, etc). This provider makes that explicit instead
of the app crashing or silently pretending to be intelligent.
"""

from __future__ import annotations

from app.core.errors import LLMError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ProviderHealth


class NullProvider(LLMProvider):
    name = "none"

    def generate(self, messages: list[LLMMessage], *, tools=None, timeout: float = 20.0) -> LLMResponse:
        raise LLMError(
            "No LLM provider is configured. Set LLM_PROVIDER=gemini or LLM_PROVIDER=ollama "
            "and the corresponding credentials in your environment."
        )

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(available=False, provider_name=self.name, detail="No LLM provider configured.")
