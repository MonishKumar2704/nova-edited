"""
Ollama provider (Phase 8: FREE LLM ENGINE - local model fallback).

Talks to a local (or self-hosted) Ollama server's native `/api/chat`
endpoint over REST via `requests`. This is the "always free, works
offline" side of master spec section 15 - no API key, no network egress
required beyond localhost, no vendor account at all.

Supports:
  * plain text generation
  * structured output (`response_schema` -> Ollama's `format` field,
    which accepts a JSON Schema directly as of Ollama 0.5+)
  * tool/function calling (`tools` -> OpenAI-style `tools` array, which
    is what Ollama's `/api/chat` expects)
  * timeouts + bounded retries on transient failures only
  * normalized errors (`app.core.errors`) - most commonly `NetworkError`
    when no Ollama server is running locally, which is an expected,
    non-fatal condition (see `factory.py` fallback wiring)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.core.errors import LLMError, NetworkError, RateLimitError, TimeoutErrorNova
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ProviderHealth, ToolCall
from app.llm.retry import retry_call

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, *, max_retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries

    # -- LLMProvider interface -------------------------------------------------

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float = 20.0,
    ) -> LLMResponse:
        if not messages:
            raise LLMError("Ollama generate() requires at least one message.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
                for tool in tools
            ]
        if response_schema is not None:
            payload["format"] = response_schema

        url = f"{self.base_url}/api/chat"

        def _call() -> LLMResponse:
            response = self._post(url, payload, timeout=timeout)
            return self._parse_response(response)

        return retry_call(_call, max_attempts=self.max_retries)

    def health_check(self) -> ProviderHealth:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
        except requests.exceptions.RequestException as exc:
            return ProviderHealth(
                available=False,
                provider_name=self.name,
                detail=f"Ollama server unreachable at {self.base_url}: {exc}",
            )

        if resp.status_code != 200:
            return ProviderHealth(
                available=False, provider_name=self.name, detail=f"Ollama server returned status {resp.status_code}."
            )

        try:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
        except ValueError:
            models = []

        model_matches = any(m == self.model or m.startswith(f"{self.model}:") for m in models)
        if models and not model_matches:
            return ProviderHealth(
                available=False,
                provider_name=self.name,
                detail=f"Ollama is running but model '{self.model}' is not pulled locally (run `ollama pull {self.model}`).",
            )
        return ProviderHealth(available=True, provider_name=self.name, detail=f"Ollama reachable, model '{self.model}' ready.")

    # -- transport -------------------------------------------------

    def _post(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise TimeoutErrorNova(f"Ollama request timed out after {timeout}s.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise NetworkError(f"Could not reach Ollama server at {self.base_url}: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise NetworkError(f"Ollama request failed: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("Ollama server is overloaded (429).")
        if resp.status_code >= 500:
            raise NetworkError(f"Ollama server error (status {resp.status_code}).")
        if resp.status_code == 404:
            detail = _extract_error_message(resp)
            raise LLMError(f"Ollama model '{self.model}' not found: {detail}")
        if resp.status_code != 200:
            raise LLMError(f"Ollama returned unexpected status {resp.status_code}: {_extract_error_message(resp)}")

        try:
            return resp.json()
        except ValueError as exc:
            raise LLMError("Ollama returned a non-JSON response.") from exc

    # -- response parsing -------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        if raw.get("error"):
            raise LLMError(f"Ollama error: {raw['error']}")

        message = raw.get("message") or {}
        text = message.get("content", "") or ""

        tool_calls: list[ToolCall] = []
        for i, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(call.get("id", f"ollama-call-{i}")),
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or {},
                )
            )

        finish_reason = "tool_calls" if tool_calls else ("stop" if raw.get("done", True) else "incomplete")

        return LLMResponse(text=text, tool_calls=tool_calls, finish_reason=finish_reason, raw=raw)


def _extract_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return str(body.get("error", resp.text[:200]))
    except ValueError:
        return resp.text[:200]
