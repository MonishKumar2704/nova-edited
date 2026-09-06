import json
from unittest.mock import MagicMock

import pytest

from app.core.errors import LLMError, ValidationError
from app.llm.base import LLMResponse
from app.tools.ai.rewrite import RewriteEmailTool


def _provider_returning(text: str) -> MagicMock:
    provider = MagicMock()
    provider.generate.return_value = LLMResponse(text=text)
    return provider


def test_rewrite_requires_text():
    tool = RewriteEmailTool(llm_provider=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"style": "professional"})


def test_rewrite_defaults_style_when_omitted():
    provider = _provider_returning(json.dumps({"text": "Rewritten."}))
    tool = RewriteEmailTool(llm_provider=provider)

    result = tool.execute({"text": "hey can u send that thing"})

    assert result.success is True
    assert result.data == {"text": "Rewritten.", "style": "improve"}


def test_rewrite_passes_requested_style_through():
    provider = _provider_returning(json.dumps({"text": "Dear team, ..."}))
    tool = RewriteEmailTool(llm_provider=provider)

    result = tool.execute({"text": "hey can u send that thing", "style": "more professional"})

    assert result.data["style"] == "more professional"
    sent_messages = provider.generate.call_args.args[0]
    assert any("more professional" in m.content for m in sent_messages)
    assert any("hey can u send that thing" in m.content for m in sent_messages)


def test_rewrite_extracts_json_wrapped_in_prose():
    text = 'Here is the rewrite:\n```json\n{"text": "Shorter version."}\n```'
    provider = _provider_returning(text)
    tool = RewriteEmailTool(llm_provider=provider)

    result = tool.execute({"text": "a long rambling email", "style": "shorter"})

    assert result.data["text"] == "Shorter version."


def test_rewrite_raises_llm_error_on_unparseable_response():
    provider = _provider_returning("I can't do that.")
    tool = RewriteEmailTool(llm_provider=provider)

    with pytest.raises(LLMError):
        tool.execute({"text": "some email", "style": "polite"})


def test_rewrite_raises_llm_error_on_empty_text_field():
    provider = _provider_returning(json.dumps({"text": ""}))
    tool = RewriteEmailTool(llm_provider=provider)

    with pytest.raises(LLMError):
        tool.execute({"text": "some email", "style": "friendly"})
