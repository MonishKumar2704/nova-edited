"""Tests for Gmail *forward* - `gmail.forward` (Task 34: verify Gmail
forward).

`gmail.reply` (Task 32) and `gmail.reply_all` (Task 33) share this same
module (`app.tools.gmail.conversations`) but are separate roadmap items.
Forward is meaningfully different from both: it goes to new recipients
(not the original sender/participants) and deliberately starts a new
conversation (no `thread_id`/`In-Reply-To`/`References`), quoting the
original message in the body instead of relying on Gmail thread linking.
"""

from unittest.mock import MagicMock

import pytest

from app.core.errors import ValidationError
from app.tools.gmail.conversations import ForwardTool


def _make_original(**overrides):
    original = MagicMock()
    original.from_ = overrides.get("from_", "John Doe <john@example.com>")
    original.subject = overrides.get("subject", "Meeting tomorrow")
    original.date = overrides.get("date", "Mon, 1 Sep 2026 10:00:00 -0700")
    original.body_text = overrides.get("body_text", "See you then!")
    return original


def test_forward_requires_message_id_and_to():
    tool = ForwardTool(client=MagicMock())
    for bad_payload in [
        {"message_id": "", "to": ["x@example.com"], "access_token": "tkn"},
        {"message_id": "m1", "to": None, "access_token": "tkn"},
        {"message_id": "m1", "to": [], "access_token": "tkn"},
        {"message_id": "m1", "to": ["not-an-email"], "access_token": "tkn"},
        {"message_id": "m1", "to": "x@example.com", "access_token": "tkn"},  # bare string, not a list
    ]:
        with pytest.raises(ValidationError):
            tool.execute(bad_payload)


def test_forward_requires_access_token():
    tool = ForwardTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "to": ["x@example.com"]})


def test_forward_fetches_original_in_full_format():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ForwardTool(client=client)

    tool.execute({"message_id": "m1", "to": ["x@example.com"], "access_token": "tkn"})

    assert client.get_message.call_args.kwargs == {"access_token": "tkn", "message_id": "m1", "format": "full"}


def test_forward_sends_to_new_recipients_without_threading_headers():
    # Forwarding starts a new conversation with new people - it should
    # NOT carry the original's thread_id/In-Reply-To/References, unlike
    # reply/reply-all.
    client = MagicMock()
    client.get_message.return_value = _make_original()
    sent = MagicMock()
    sent.to_dict.return_value = {"id": "m2"}
    client.send_message.return_value = sent
    tool = ForwardTool(client=client)

    result = tool.execute(
        {"message_id": "m1", "to": ["x@example.com"], "body_text": "FYI see below", "access_token": "tkn"}
    )

    assert result.success is True
    kwargs = client.send_message.call_args.kwargs
    assert set(kwargs.keys()) == {"access_token", "to", "subject", "body_text"}
    assert kwargs["to"] == ["x@example.com"]


def test_forward_includes_intro_and_quoted_original_when_body_text_given():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ForwardTool(client=client)

    tool.execute(
        {"message_id": "m1", "to": ["x@example.com"], "body_text": "FYI see below", "access_token": "tkn"}
    )

    body = client.send_message.call_args.kwargs["body_text"]
    assert body.startswith("FYI see below\n\n---------- Forwarded/original message ----------")
    assert "From: John Doe <john@example.com>" in body
    assert "Date: Mon, 1 Sep 2026 10:00:00 -0700" in body
    assert "Subject: Meeting tomorrow" in body
    assert body.endswith("See you then!")


def test_forward_quotes_original_without_leading_blank_lines_when_no_intro():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ForwardTool(client=client)

    tool.execute({"message_id": "m1", "to": ["x@example.com"], "access_token": "tkn"})

    body = client.send_message.call_args.kwargs["body_text"]
    assert body.startswith("---------- Forwarded/original message ----------")
    assert not body.startswith("\n")


def test_forward_treats_whitespace_only_body_text_as_no_intro():
    client = MagicMock()
    client.get_message.return_value = _make_original()
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ForwardTool(client=client)

    tool.execute({"message_id": "m1", "to": ["x@example.com"], "body_text": "   ", "access_token": "tkn"})

    body = client.send_message.call_args.kwargs["body_text"]
    assert body.startswith("---------- Forwarded/original message ----------")


def test_forward_does_not_double_prefix_subject_already_starting_with_fwd():
    client = MagicMock()
    client.get_message.return_value = _make_original(subject="Fwd: Meeting tomorrow")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ForwardTool(client=client)

    tool.execute({"message_id": "m1", "to": ["x@example.com"], "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["subject"] == "Fwd: Meeting tomorrow"


def test_forward_handles_empty_original_subject_without_crashing():
    client = MagicMock()
    client.get_message.return_value = _make_original(subject="")
    client.send_message.return_value = MagicMock(to_dict=lambda: {})
    tool = ForwardTool(client=client)

    tool.execute({"message_id": "m1", "to": ["x@example.com"], "access_token": "tkn"})

    assert client.send_message.call_args.kwargs["subject"] == "Fwd:"


def test_forward_requires_confirmation():
    # Sends a real email to new recipients - matches reply/reply-all.
    assert ForwardTool.requires_confirmation is True
