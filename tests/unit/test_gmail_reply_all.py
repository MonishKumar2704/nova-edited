"""Tests for Gmail *reply-all* - `gmail.reply_all` (Task 33: verify Gmail
reply-all).

`gmail.reply` (Task 32) and `gmail.forward` (Task 34) share this same
module (`app.tools.gmail.conversations`) but are separate roadmap items;
this file's scope is exactly `ReplyAllTool`, focusing on the piece that's
genuinely different from a plain reply: building the `cc` list from the
original message's `To` header plus caller-supplied `extra_cc`.
"""

from unittest.mock import MagicMock

import pytest

from app.core.errors import ValidationError
from app.tools.gmail.conversations import ReplyAllTool


def _make_original(**overrides):
    original = MagicMock()
    original.from_ = overrides.get("from_", "John Doe <john@example.com>")
    original.subject = overrides.get("subject", "Meeting tomorrow")
    original.thread_id = overrides.get("thread_id", "thread123")
    original.rfc_message_id = overrides.get("rfc_message_id", "<abc@mail.gmail.com>")
    original.to = overrides.get("to", "a@example.com, b@example.com")
    return original


def test_reply_all_requires_message_id_and_body_text():
    tool = ReplyAllTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "", "body_text": "hi", "access_token": "tkn"})
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "body_text": "   ", "access_token": "tkn"})


def test_reply_all_requires_access_token():
    tool = ReplyAllTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "body_text": "hi"})


def test_reply_all_raises_if_original_has_no_sender():
    client = MagicMock()
    client.get_message.return_value = _make_original(from_="")
    tool = ReplyAllTool(client=client)

    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})


def test_reply_all_sends_to_sender_and_ccs_original_recipients():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    sent = MagicMock()
    sent.to_dict.return_value = {"id": "m2"}
    client.send_message.return_value = sent
    tool = ReplyAllTool(client=client)

    result = tool.execute({"message_id": "m1", "body_text": "hi all", "access_token": "tkn"})

    assert result.success is True
    kwargs = client.send_message.call_args.kwargs
    assert kwargs["to"] == ["John Doe <john@example.com>"]
    assert kwargs["cc"] == ["a@example.com", "b@example.com"]
    assert kwargs["subject"] == "Re: Meeting tomorrow"
    assert kwargs["thread_id"] == "thread123"
    assert kwargs["in_reply_to"] == "<abc@mail.gmail.com>"
    assert kwargs["references"] == "<abc@mail.gmail.com>"


def test_reply_all_appends_extra_cc_to_original_recipients():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyAllTool(client=client)

    tool.execute(
        {"message_id": "m1", "body_text": "hi", "access_token": "tkn", "extra_cc": "c@example.com, d@example.com"}
    )

    assert client.send_message.call_args.kwargs["cc"] == [
        "a@example.com",
        "b@example.com",
        "c@example.com",
        "d@example.com",
    ]


def test_reply_all_cc_is_none_when_original_has_no_recipients_and_no_extra_cc():
    client = MagicMock()
    client.get_message.return_value = _make_original(to="")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyAllTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["cc"] is None


def test_reply_all_strips_whitespace_and_drops_empty_entries():
    client = MagicMock()
    client.get_message.return_value = _make_original(to="  a@example.com ,  ,b@example.com")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyAllTool(client=client)

    tool.execute(
        {"message_id": "m1", "body_text": "hi", "access_token": "tkn", "extra_cc": " , c@example.com,"}
    )

    assert client.send_message.call_args.kwargs["cc"] == ["a@example.com", "b@example.com", "c@example.com"]


def test_reply_all_does_not_double_prefix_subject_already_starting_with_re():
    client = MagicMock()
    client.get_message.return_value = _make_original(subject="RE: Meeting tomorrow")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyAllTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["subject"] == "RE: Meeting tomorrow"


def test_reply_all_requires_confirmation():
    # Sends a real email to multiple people - matches gmail.reply.
    assert ReplyAllTool.requires_confirmation is True
