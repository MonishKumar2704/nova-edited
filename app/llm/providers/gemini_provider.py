"""
Gemini provider (Phase 8: FREE LLM ENGINE).

Talks to the official Gemini API (`generativelanguage.googleapis.com`)
directly over REST via `requests` - no vendor SDK dependency, so the only
thing that changes if Google reshapes their Python SDK is this one file.
Google's free tier (an API key from https://aistudio.google.com, no
billing account required) is sufficient; nothing here requires a paid
plan (master spec section 15).

Supports:
  * plain text generation
  * structured output (`response_schema` -> `responseMimeType:
    application/json` + `responseSchema`)
  * tool/function calling (`tools` -> `functionDeclarations`)
  * timeouts + bounded retries on transient failures only
    (`app.llm.retry.retry_call`, reused rather than reinvented - master
    spec section 12)
  * normalized errors (`app.core.errors`) - callers never see a raw
    `requests` exception or an unclassified failure
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.core.errors import LLMError, NetworkError, RateLimitError, TimeoutErrorNova
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ProviderHealth, ToolCall
from app.llm.retry import retry_call

logger = logging.getLogger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, *, max_retries: int = 3) -> None:
        self.api_key = api_key
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
        if not self.api_key:
            raise LLMError("Gemini provider is not configured: GEMINI_API_KEY is not set.")

        payload = self._build_payload(messages, tools=tools, response_schema=response_schema)
        url = f"{_API_BASE}/models/{self.model}:generateContent"

        def _call() -> LLMResponse:
            response = self._post(url, payload, timeout=timeout)
            return self._parse_response(response)

        return retry_call(_call, max_attempts=self.max_retries)

    def health_check(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(available=False, provider_name=self.name, detail="GEMINI_API_KEY not set.")
        try:
            resp = requests.get(
                f"{_API_BASE}/models/{self.model}",
                params={"key": self.api_key},
                timeout=5.0,
            )
        except requests.exceptions.RequestException as exc:
            return ProviderHealth(available=False, provider_name=self.name, detail=f"Unreachable: {exc}")

        if resp.status_code == 200:
            return ProviderHealth(available=True, provider_name=self.name, detail=f"Model '{self.model}' reachable.")
        if resp.status_code in (401, 403):
            return ProviderHealth(available=False, provider_name=self.name, detail="Invalid or unauthorized API key.")
        if resp.status_code == 404:
            return ProviderHealth(
                available=False, provider_name=self.name, detail=f"Model '{self.model}' not found."
            )
        return ProviderHealth(
            available=False, provider_name=self.name, detail=f"Unexpected status {resp.status_code}."
        )

    # -- request building -------------------------------------------------

    def _build_payload(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None,
        response_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        if not contents:
            raise LLMError("Gemini generate() requires at least one non-system message.")

        payload: dict[str, Any] = {"contents": contents}

        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": _sanitize_schema(tool.get("input_schema") or {"type": "object"}),
                        }
                        for tool in tools
                    ]
                }
            ]

        generation_config: dict[str, Any] = {}
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _sanitize_schema(response_schema)
        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    # -- transport -------------------------------------------------

    def _post(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        try:
            resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise TimeoutErrorNova(f"Gemini request timed out after {timeout}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise NetworkError(f"Could not reach Gemini API: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("Gemini API rate limit exceeded.")
        if resp.status_code >= 500:
            raise NetworkError(f"Gemini API server error (status {resp.status_code}).")
        if resp.status_code in (401, 403):
            raise LLMError("Gemini API rejected the request: invalid or unauthorized API key.")
        if resp.status_code == 400:
            detail = _extract_error_message(resp)
            raise LLMError(f"Gemini API rejected the request: {detail}")
        if resp.status_code != 200:
            raise LLMError(f"Gemini API returned unexpected status {resp.status_code}: {_extract_error_message(resp)}")

        try:
            return resp.json()
        except ValueError as exc:
            raise LLMError("Gemini API returned a non-JSON response.") from exc

    # -- response parsing -------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        candidates = raw.get("candidates") or []
        if not candidates:
            block_reason = (raw.get("promptFeedback") or {}).get("blockReason")
            if block_reason:
                raise LLMError(f"Gemini blocked the prompt (reason: {block_reason}).")
            raise LLMError("Gemini API returned no candidates.")

        candidate = candidates[0]
        finish_reason = str(candidate.get("finishReason", "stop")).lower()
        parts = (candidate.get("content") or {}).get("parts") or []

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for i, part in enumerate(parts):
            if "text" in part:
                text_chunks.append(part["text"])
            elif "functionCall" in part:
                call = part["functionCall"]
                tool_calls.append(
                    ToolCall(id=f"gemini-call-{i}", name=call.get("name", ""), arguments=call.get("args") or {})
                )

        return LLMResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=raw,
        )


def _extract_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return str(((body or {}).get("error") or {}).get("message", resp.text[:200]))
    except ValueError:
        return resp.text[:200]


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop JSON-Schema keywords Gemini's `Schema` type doesn't accept.

    Our tool `input_schema`/`response_schema` dicts are plain JSON Schema
    (see `app.tools.base.Tool`); Gemini's function/response schema format
    is a strict subset. Rather than fail on unsupported keywords, drop
    them - this keeps every tool's `input_schema` usable as-is instead of
    requiring a Gemini-specific copy of each schema.
    """
    if not isinstance(schema, dict):
        return schema
    _UNSUPPORTED = {"$schema", "additionalProperties", "title", "default", "examples"}
    cleaned = {k: v for k, v in schema.items() if k not in _UNSUPPORTED}
    if "properties" in cleaned and isinstance(cleaned["properties"], dict):
        cleaned["properties"] = {k: _sanitize_schema(v) for k, v in cleaned["properties"].items()}
    if "items" in cleaned and isinstance(cleaned["items"], dict):
        cleaned["items"] = _sanitize_schema(cleaned["items"])
    return cleaned
