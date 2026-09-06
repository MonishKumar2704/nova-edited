import json
from unittest.mock import MagicMock

import pytest

from app.core.errors import LLMError, ValidationError
from app.llm.base import LLMResponse
from app.tools.ai.reply import SuggestReplyTool


def _provider_returning(text: str) -> MagicMock:
    provider = MagicMock()
    provider.generate.return_value = LLMResponse(text=text)
    return provider


def _original_message(subject="Project update", from_="boss@example.com", thread_id="t1", rfc_id="<abc@mail>"):
    original = MagicMock()
    original.from_ = from_
    original.subject = subject
    original.body_text = "Can you send the report by Friday?"
    original.thread_id = thread_id
    original.rfc_message_id = rfc_id
    return original


def _draft_summary_dict():
    return {"draft_id": "d1", "message": {"subject": "Re: Project update", "to": "boss@example.com"}}


def test_suggest_reply_requires_message_id():
    tool = SuggestReplyTool(llm_provider=MagicMock(), gmail_client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tok"})


def test_suggest_reply_requires_access_token():
    tool = SuggestReplyTool(llm_provider=MagicMock(), gmail_client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1"})


def test_suggest_reply_generates_and_saves_draft_in_same_thread():
    provider = _provider_returning(json.dumps({"body": "Sure, I'll have it to you by Friday."}))
    gmail_client = MagicMock()
    gmail_client.get_message.return_value = _original_message()
    draft = MagicMock()
    draft.to_dict.return_value = _draft_summary_dict()
    gmail_client.create_draft.return_value = draft

    tool = SuggestReplyTool(llm_provider=provider, gmail_client=gmail_client)
    result = tool.execute({"message_id": "m1", "access_token": "tok"})

    assert result.success is True
    assert result.data["draft"] == _draft_summary_dict()
    assert result.data["subject"] == "Re: Project update"
    assert result.data["body"] == "Sure, I'll have it to you by Friday."
    gmail_client.get_message.assert_called_once_with(access_token="tok", message_id="m1", format="full")
    gmail_client.create_draft.assert_called_once_with(
        access_token="tok",
        to=["boss@example.com"],
        subject="Re: Project update",
        body_text="Sure, I'll have it to you by Friday.",
        thread_id="t1",
        in_reply_to="<abc@mail>",
        references="<abc@mail>",
    )


def test_suggest_reply_does_not_double_prefix_re_subject():
    provider = _provider_returning(json.dumps({"body": "Reply text."}))
    gmail_client = MagicMock()
    gmail_client.get_message.return_value = _original_message(subject="Re: Project update")
    draft = MagicMock()
    draft.to_dict.return_value = _draft_summary_dict()
    gmail_client.create_draft.return_value = draft

    tool = SuggestReplyTool(llm_provider=provider, gmail_client=gmail_client)
    result = tool.execute({"message_id": "m1", "access_token": "tok"})

    assert result.data["subject"] == "Re: Project update"


def test_suggest_reply_passes_instruction_through_to_the_model():
    provider = _provider_returning(json.dumps({"body": "I can't make it, how about next week?"}))
    gmail_client = MagicMock()
    gmail_client.get_message.return_value = _original_message()
    draft = MagicMock()
    draft.to_dict.return_value = _draft_summary_dict()
    gmail_client.create_draft.return_value = draft

    tool = SuggestReplyTool(llm_provider=provider, gmail_client=gmail_client)
    tool.execute({"message_id": "m1", "instruction": "decline and suggest next week", "access_token": "tok"})

    sent_messages = provider.generate.call_args.args[0]
    assert any("decline and suggest next week" in m.content for m in sent_messages)
    assert any("Can you send the report by Friday?" in m.content for m in sent_messages)


def test_suggest_reply_raises_validation_error_when_sender_unknown():
    gmail_client = MagicMock()
    gmail_client.get_message.return_value = _original_message(from_="")

    tool = SuggestReplyTool(llm_provider=MagicMock(), gmail_client=gmail_client)
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "access_token": "tok"})


def test_suggest_reply_raises_llm_error_on_unparseable_response():
    provider = _provider_returning("I can't do that.")
    gmail_client = MagicMock()
    gmail_client.get_message.return_value = _original_message()

    tool = SuggestReplyTool(llm_provider=provider, gmail_client=gmail_client)
    with pytest.raises(LLMError):
        tool.execute({"message_id": "m1", "access_token": "tok"})


def test_suggest_reply_never_calls_send():
    """Task 59: AI-generated replies must never be sent automatically -
    only `create_draft` should ever be invoked by this tool."""
    provider = _provider_returning(json.dumps({"body": "Reply text."}))
    gmail_client = MagicMock()
    gmail_client.get_message.return_value = _original_message()
    draft = MagicMock()
    draft.to_dict.return_value = _draft_summary_dict()
    gmail_client.create_draft.return_value = draft

    tool = SuggestReplyTool(llm_provider=provider, gmail_client=gmail_client)
    tool.execute({"message_id": "m1", "access_token": "tok"})

    gmail_client.send_message.assert_not_called()
