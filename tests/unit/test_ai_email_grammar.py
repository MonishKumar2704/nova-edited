import json
from unittest.mock import MagicMock

import pytest

from app.core.errors import LLMError, ValidationError
from app.llm.base import LLMResponse
from app.tools.ai.grammar import CorrectEmailGrammarTool


def _provider_returning(text: str) -> MagicMock:
    provider = MagicMock()
    provider.generate.return_value = LLMResponse(text=text)
    return provider


def test_grammar_correct_requires_text():
    tool = CorrectEmailGrammarTool(llm_provider=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_grammar_correct_returns_corrected_text():
    provider = _provider_returning(json.dumps({"text": "Hey, can you send that thing?"}))
    tool = CorrectEmailGrammarTool(llm_provider=provider)

    result = tool.execute({"text": "hey can you send that thing"})

    assert result.success is True
    assert result.data == {"text": "Hey, can you send that thing?"}
    sent_messages = provider.generate.call_args.args[0]
    assert any("hey can you send that thing" in m.content for m in sent_messages)


def test_grammar_correct_extracts_json_wrapped_in_prose():
    text = 'Here is the correction:\n```json\n{"text": "Fixed sentence."}\n```'
    provider = _provider_returning(text)
    tool = CorrectEmailGrammarTool(llm_provider=provider)

    result = tool.execute({"text": "a sentence with a mistake"})

    assert result.data["text"] == "Fixed sentence."


def test_grammar_correct_raises_llm_error_on_unparseable_response():
    provider = _provider_returning("I can't do that.")
    tool = CorrectEmailGrammarTool(llm_provider=provider)

    with pytest.raises(LLMError):
        tool.execute({"text": "some email"})


def test_grammar_correct_raises_llm_error_on_empty_text_field():
    provider = _provider_returning(json.dumps({"text": ""}))
    tool = CorrectEmailGrammarTool(llm_provider=provider)

    with pytest.raises(LLMError):
        tool.execute({"text": "some email"})
