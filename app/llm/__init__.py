"""
LLM provider abstraction (see master spec sections 12-15).

Nova must not be tightly coupled to any single LLM vendor. This package
defines a provider-independent interface (`base.LLMProvider`) plus a
factory (`factory.build_llm_provider`) that selects and wires up a real
implementation based on configuration (`LLM_PROVIDER=gemini|ollama|none`).

Phase 8 (FREE LLM ENGINE) implemented the concrete providers:
  * `providers.gemini_provider.GeminiProvider` - Google's free-tier Gemini
    API over REST.
  * `providers.ollama_provider.OllamaProvider` - a local/self-hosted
    Ollama server, so Nova's LLM features work with zero API key and zero
    cost.
  * `providers.null_provider.NullProvider` - explicit "no LLM configured"
    provider; the rest of the app (health checks, non-AI routes) keeps
    working without one.

(Task 19 removed a fourth provider, `FallbackLLMProvider`, which chained
two of the above together with runtime failover - more multi-provider
resilience than this mini project's single-backend AI requirement needs.
Per-call retry inside each provider, and the orchestrator's no-LLM
keyword-router fallback, are unrelated and still in place.)

The agent orchestrator (Phase 9) will depend only on `LLMProvider` -
never on a concrete vendor class - so switching providers stays a
configuration change, not a code change.
"""
