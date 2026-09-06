"""
Phase 8 sanity tests: real provider request/response shape + error
normalization. Not exhaustive (full LLM test suite is Phase 17 scope) -
just enough to catch an obviously broken payload/parse/error mapping.
"""

from __future__ import annotations

import requests

from app.core.errors import LLMError, NetworkError, RateLimitError, TimeoutErrorNova
from app.llm.base import LLMMessage
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.ollama_provider import OllamaProvider


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text or str(json_body)

    def json(self):
        return self._json_body


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------


def test_gemini_generate_without_api_key_raises_llm_error():
    provider = GeminiProvider(api_key="", model="gemini-1.5-flash")
    try:
        provider.generate([LLMMessage(role="user", content="hi")])
        assert False, "expected LLMError"
    except LLMError:
        pass


def test_gemini_generate_parses_text_response(monkeypatch):
    provider = GeminiProvider(api_key="k", model="gemini-1.5-flash")

    def fake_post(url, params=None, json=None, timeout=None):
        assert "generateContent" in url
        assert json["contents"][0]["parts"][0]["text"] == "hello"
        return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hi there"}]}, "finishReason": "STOP"}]})

    monkeypatch.setattr(requests, "post", fake_post)
    result = provider.generate([LLMMessage(role="user", content="hello")])
    assert result.text == "hi there"
    assert result.finish_reason == "stop"
    assert result.tool_calls == []


def test_gemini_generate_parses_tool_call(monkeypatch):
    provider = GeminiProvider(api_key="k", model="gemini-1.5-flash")

    def fake_post(url, params=None, json=None, timeout=None):
        return _FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {"parts": [{"functionCall": {"name": "youtube.search", "args": {"query": "cats"}}}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)
    result = provider.generate(
        [LLMMessage(role="user", content="find cat videos")],
        tools=[{"name": "youtube.search", "description": "search", "input_schema": {"type": "object"}}],
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "youtube.search"
    assert result.tool_calls[0].arguments == {"query": "cats"}
    assert result.tool_call == {"id": "gemini-call-0", "name": "youtube.search", "arguments": {"query": "cats"}}


def test_gemini_generate_maps_429_to_rate_limit_error(monkeypatch):
    provider = GeminiProvider(api_key="k", model="gemini-1.5-flash", max_retries=1)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(429, {"error": {"message": "quota"}}))
    try:
        provider.generate([LLMMessage(role="user", content="hi")])
        assert False, "expected RateLimitError"
    except RateLimitError:
        pass


def test_gemini_generate_maps_timeout(monkeypatch):
    provider = GeminiProvider(api_key="k", model="gemini-1.5-flash", max_retries=1)

    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr(requests, "post", raise_timeout)
    try:
        provider.generate([LLMMessage(role="user", content="hi")])
        assert False, "expected TimeoutErrorNova"
    except TimeoutErrorNova:
        pass


def test_gemini_health_check_no_key():
    health = GeminiProvider(api_key="", model="x").health_check()
    assert health.available is False


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


def test_ollama_generate_parses_text_response(monkeypatch):
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/api/chat")
        assert json["messages"][0]["content"] == "hello"
        return _FakeResponse(200, {"message": {"role": "assistant", "content": "hi there"}, "done": True})

    monkeypatch.setattr(requests, "post", fake_post)
    result = provider.generate([LLMMessage(role="user", content="hello")])
    assert result.text == "hi there"
    assert result.finish_reason == "stop"


def test_ollama_generate_parses_tool_calls(monkeypatch):
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(
            200,
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "gmail.search", "arguments": {"q": "invoice"}}}],
                },
                "done": True,
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)
    result = provider.generate(
        [LLMMessage(role="user", content="find invoice emails")],
        tools=[{"name": "gmail.search", "description": "search gmail", "input_schema": {"type": "object"}}],
    )
    assert result.tool_calls[0].name == "gmail.search"
    assert result.finish_reason == "tool_calls"


def test_ollama_generate_connection_error_maps_to_network_error(monkeypatch):
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3", max_retries=1)

    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(requests, "post", raise_conn_error)
    try:
        provider.generate([LLMMessage(role="user", content="hi")])
        assert False, "expected NetworkError"
    except NetworkError:
        pass


def test_ollama_health_check_unreachable(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(requests, "get", raise_conn_error)
    health = OllamaProvider(base_url="http://localhost:11434", model="llama3").health_check()
    assert health.available is False
