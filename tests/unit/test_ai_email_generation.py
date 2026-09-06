import json
from unittest.mock import MagicMock

import pytest

from app.core.errors import LLMError, ValidationError
from app.llm.base import LLMResponse
from app.tools.ai.email_generation import GenerateEmailDraftTool, GenerateEmailTool


def _provider_returning(text: str) -> MagicMock:
    provider = MagicMock()
    provider.generate.return_value = LLMResponse(text=text)
    return provider


def test_generate_email_requires_instruction():
    tool = GenerateEmailTool(llm_provider=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_generate_email_parses_clean_json():
    provider = _provider_returning(json.dumps({"subject": "Extension request", "body": "Dear Professor, ..."}))
    tool = GenerateEmailTool(llm_provider=provider)

    result = tool.execute({"instruction": "ask my professor for an extension"})

    assert result.success is True
    assert result.data == {"subject": "Extension request", "body": "Dear Professor, ..."}
    provider.generate.assert_called_once()


def test_generate_email_extracts_json_wrapped_in_prose():
    """A local model can ignore the "ONLY JSON" instruction and wrap the
    object in commentary or a code fence - the tool should still recover
    the email rather than failing outright."""
    text = 'Sure! Here you go:\n```json\n{"subject": "Hi", "body": "Hello there."}\n```'
    provider = _provider_returning(text)
    tool = GenerateEmailTool(llm_provider=provider)

    result = tool.execute({"instruction": "say hi"})

    assert result.data == {"subject": "Hi", "body": "Hello there."}


def test_generate_email_raises_llm_error_on_unparseable_response():
    provider = _provider_returning("I cannot help with that.")
    tool = GenerateEmailTool(llm_provider=provider)

    with pytest.raises(LLMError):
        tool.execute({"instruction": "ask my professor for an extension"})


def test_generate_email_raises_llm_error_on_missing_fields():
    provider = _provider_returning(json.dumps({"subject": "", "body": ""}))
    tool = GenerateEmailTool(llm_provider=provider)

    with pytest.raises(LLMError):
        tool.execute({"instruction": "ask my professor for an extension"})


# --- GenerateEmailDraftTool (Task 54: AI generation -> Gmail draft) ---


def _draft_summary_dict():
    return {"draft_id": "d1", "message": {"subject": "Extension request", "to": "prof@example.com"}}


def test_generate_email_draft_requires_instruction():
    tool = GenerateEmailDraftTool(llm_provider=MagicMock(), gmail_client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"to": ["prof@example.com"], "access_token": "tok"})


def test_generate_email_draft_requires_valid_recipients():
    tool = GenerateEmailDraftTool(llm_provider=MagicMock(), gmail_client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"instruction": "ask for an extension", "to": ["not-an-email"], "access_token": "tok"})


def test_generate_email_draft_requires_access_token():
    tool = GenerateEmailDraftTool(llm_provider=MagicMock(), gmail_client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"instruction": "ask for an extension", "to": ["prof@example.com"]})


def test_generate_email_draft_generates_then_saves_a_draft():
    provider = _provider_returning(json.dumps({"subject": "Extension request", "body": "Dear Professor, ..."}))
    gmail_client = MagicMock()
    draft = MagicMock()
    draft.to_dict.return_value = _draft_summary_dict()
    gmail_client.create_draft.return_value = draft

    tool = GenerateEmailDraftTool(llm_provider=provider, gmail_client=gmail_client)
    result = tool.execute(
        {"instruction": "ask my professor for an extension", "to": ["prof@example.com"], "access_token": "tok"}
    )

    assert result.success is True
    assert result.data["draft"] == _draft_summary_dict()
    assert result.data["subject"] == "Extension request"
    assert result.data["body"] == "Dear Professor, ..."
    gmail_client.create_draft.assert_called_once_with(
        access_token="tok",
        to=["prof@example.com"],
        subject="Extension request",
        body_text="Dear Professor, ...",
        cc=None,
        bcc=None,
    )


def test_generate_email_draft_never_calls_send():
    """Task 59: AI-generated emails must never be sent automatically -
    only `create_draft` should ever be invoked by this tool."""
    provider = _provider_returning(json.dumps({"subject": "Hi", "body": "Hello."}))
    gmail_client = MagicMock()
    draft = MagicMock()
    draft.to_dict.return_value = _draft_summary_dict()
    gmail_client.create_draft.return_value = draft

    tool = GenerateEmailDraftTool(llm_provider=provider, gmail_client=gmail_client)
    tool.execute({"instruction": "say hi", "to": ["a@example.com"], "access_token": "tok"})

    gmail_client.send_message.assert_not_called()
    assert not hasattr(gmail_client, "send_draft") or not gmail_client.send_draft.called
