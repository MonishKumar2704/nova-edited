"""
LLM provider factory.

The rest of Nova (agent orchestrator, services) calls `get_llm_provider()`
and depends only on the `LLMProvider` interface - never on a concrete
vendor class. Switching providers is a configuration change
(`LLM_PROVIDER=gemini|ollama`), not a code change.

Task 19 removed the dual-provider `FallbackLLMProvider` wrapper that used
to sit here (primary provider + a second provider tried on transient
failure). That was resilience for running two different LLM backends at
once, which is more than this mini project's single stated AI backend
(free local Ollama, per the roadmap's core objective) needs. Per-call
retry (`app.llm.retry.retry_call`, used inside each concrete provider) and
the orchestrator's plain-keyword-router fallback for "no LLM
configured/reachable at all" (`app.agent.planner_orchestrator`) are
unrelated, still-necessary layers and were not touched.
"""

from __future__ import annotations

from app.core.config import Config
from app.llm.base import LLMProvider
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.null_provider import NullProvider
from app.llm.providers.ollama_provider import OllamaProvider


def build_llm_provider(config: Config) -> LLMProvider:
    """Build the configured LLM provider (`LLM_PROVIDER=gemini|ollama|none`)."""
    name = config.llm_provider.lower()
    if name == "gemini":
        return GeminiProvider(api_key=config.gemini_api_key, model=config.gemini_model, max_retries=config.llm_max_retries)
    if name == "ollama":
        return OllamaProvider(base_url=config.ollama_base_url, model=config.ollama_model, max_retries=config.llm_max_retries)
    # "none", "", or anything unrecognized: fail safe to NullProvider rather
    # than crashing the whole application - surfaced via health_check().
    return NullProvider()
