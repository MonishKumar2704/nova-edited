"""Tests for Gmail *reply* - `gmail.reply` (Task 32: verify Gmail reply).

`gmail.reply_all` and `gmail.forward` are separate roadmap items (Tasks 33
and 34) and share this same module (`app.tools.gmail.conversations`); this
file's scope is exactly `ReplyTool` plus the `GmailApiClient.send_message`
RFC 2822 threading headers it relies on (`In-Reply-To`/`References`).
"""

import base64
import email
from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.conversations import ReplyTool


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


def _make_original(**overrides):
    original = MagicMock()
    original.from_ = overrides.get("from_", "John Doe <john@example.com>")
    original.subject = overrides.get("subject", "Meeting tomorrow")
    original.thread_id = overrides.get("thread_id", "thread123")
    original.rfc_message_id = overrides.get("rfc_message_id", "<abc@mail.gmail.com>")
    return original


# -- ReplyTool ----------------------------------------------------------------


def test_reply_requires_message_id_and_body_text():
    tool = ReplyTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "", "body_text": "hi", "access_token": "tkn"})
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "body_text": "   ", "access_token": "tkn"})


def test_reply_requires_access_token():
    tool = ReplyTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "body_text": "hi"})


def test_reply_fetches_original_message_in_full_format():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "Sounds good!", "access_token": "tkn"})

    assert client.get_message.call_args.kwargs == {"access_token": "tkn", "message_id": "m1", "format": "full"}


def test_reply_sends_to_original_sender_with_thread_and_references():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    sent = MagicMock()
    sent.to_dict.return_value = {"id": "m2", "labelIds": ["SENT"]}
    client.send_message.return_value = sent
    tool = ReplyTool(client=client)

    result = tool.execute({"message_id": "m1", "body_text": "Sounds good!", "access_token": "tkn"})

    assert result.success is True
    assert result.data["message"] == {"id": "m2", "labelIds": ["SENT"]}
    assert client.send_message.call_args.kwargs == {
        "access_token": "tkn",
        "to": ["John Doe <john@example.com>"],
        "subject": "Re: Meeting tomorrow",
        "body_text": "Sounds good!",
        "thread_id": "thread123",
        "in_reply_to": "<abc@mail.gmail.com>",
        "references": "<abc@mail.gmail.com>",
    }


def test_reply_raises_if_original_has_no_sender():
    client = MagicMock()
    client.get_message.return_value = _make_original(from_="")
    tool = ReplyTool(client=client)

    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})


def test_reply_does_not_double_prefix_subject_already_starting_with_re():
    client = MagicMock()
    client.get_message.return_value = _make_original(subject="Re: Meeting tomorrow")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["subject"] == "Re: Meeting tomorrow"


def test_reply_prefixes_bare_subject_with_re():
    client = MagicMock()
    client.get_message.return_value = _make_original(subject="Meeting tomorrow")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["subject"] == "Re: Meeting tomorrow"


def test_reply_handles_empty_original_subject_without_crashing():
    client = MagicMock()
    client.get_message.return_value = _make_original(subject="")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["subject"] == "Re:"


def test_reply_omits_threading_headers_when_original_has_no_rfc_message_id():
    client = MagicMock()
    client.get_message.return_value = _make_original(rfc_message_id="")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ReplyTool(client=client)

    tool.execute({"message_id": "m1", "body_text": "hi", "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["in_reply_to"] is None
    assert client.send_message.call_args.kwargs["references"] is None


def test_reply_requires_confirmation():
    # Sends a real email - matches gmail.send / gmail.draft.send.
    assert ReplyTool.requires_confirmation is True


# -- End-to-end: send_message actually encodes threading headers ------------


def test_send_message_raw_mime_includes_in_reply_to_and_references():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(
            json_body={"id": "m2", "threadId": "thread123", "labelIds": ["SENT"]}
        )
        client.send_message(
            access_token="tkn",
            to=["John Doe <john@example.com>"],
            subject="Re: Meeting tomorrow",
            body_text="Sounds good!",
            thread_id="thread123",
            in_reply_to="<abc@mail.gmail.com>",
            references="<abc@mail.gmail.com>",
        )

    body = mock_req.call_args.kwargs["json"]
    assert body["threadId"] == "thread123"
    raw = body["raw"]
    padded = raw + "=" * (-len(raw) % 4)
    msg = email.message_from_bytes(base64.urlsafe_b64decode(padded))
    assert msg["In-Reply-To"] == "<abc@mail.gmail.com>"
    assert msg["References"] == "<abc@mail.gmail.com>"
    assert msg["To"] == "John Doe <john@example.com>"
    assert msg["Subject"] == "Re: Meeting tomorrow"
