"""Tests for Gmail *reading* - `gmail.get_message` / `gmail.get_thread`
(Task 28: verify Gmail reading: message, thread, sender, subject, body).

Scoped to reading a single message/thread by ID and the header/body
parsing that backs it (`GmailApiClient._message_from_item` /
`_walk_parts`). Listing/search got its own coverage in Task 26/27;
labels, drafts, actions, conversations, and attachments get their own
coverage in later Phase 3/4 verification tasks.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import AuthenticationError, NotFoundError, ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.get_message import GetMessageTool
from app.tools.gmail.threads import GetThreadTool


def _header(name: str, value: str) -> dict:
    return {"name": name, "value": value}


def _message_item(
    message_id="m1",
    thread_id="t1",
    *,
    headers=None,
    label_ids=("INBOX",),
    body_b64=None,
    parts=None,
    snippet="a snippet",
) -> dict:
    if headers is None:
        headers = [
            _header("Subject", "Hello"),
            _header("From", "a@example.com"),
            _header("To", "me@example.com"),
            _header("Date", "Mon, 1 Jan 2024 00:00:00 +0000"),
            _header("Message-ID", "<rfc-id-1@example.com>"),
        ]
    payload = {"mimeType": "text/plain", "headers": headers}
    if parts is not None:
        payload["mimeType"] = "multipart/mixed"
        payload["parts"] = parts
    elif body_b64 is not None:
        payload["body"] = {"data": body_b64}
    return {
        "id": message_id,
        "threadId": thread_id,
        "labelIds": list(label_ids),
        "snippet": snippet,
        "payload": payload,
    }


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


# -- GmailApiClient.get_message: header parsing --------------------------


def test_get_message_parses_standard_headers_and_body():
    client = GmailApiClient()
    item = _message_item(body_b64="aGVsbG8gd29ybGQ=")  # "hello world"
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=item)
        detail = client.get_message(access_token="tkn", message_id="m1")

    assert detail.message_id == "m1"
    assert detail.thread_id == "t1"
    assert detail.subject == "Hello"
    assert detail.from_ == "a@example.com"
    assert detail.to == "me@example.com"
    assert detail.date == "Mon, 1 Jan 2024 00:00:00 +0000"
    assert detail.rfc_message_id == "<rfc-id-1@example.com>"
    assert detail.body_text == "hello world"
    assert detail.is_unread is False


def test_get_message_header_lookup_is_case_insensitive():
    # RFC 5322 field names are case-insensitive - a sender that emits
    # lower-case (or shouty) header names must still be readable.
    client = GmailApiClient()
    lower = _message_item(
        headers=[
            _header("subject", "lower-case subject"),
            _header("from", "b@example.com"),
            _header("to", "me@example.com"),
            _header("date", "Tue, 2 Jan 2024 00:00:00 +0000"),
            _header("message-id", "<rfc-id-2@example.com>"),
        ],
        body_b64="aGk=",
    )
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=lower)
        detail = client.get_message(access_token="tkn", message_id="m1")

    assert detail.subject == "lower-case subject"
    assert detail.from_ == "b@example.com"
    assert detail.rfc_message_id == "<rfc-id-2@example.com>"


def test_get_message_missing_headers_default_to_empty_string():
    client = GmailApiClient()
    item = _message_item(headers=[], body_b64="aGk=")
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=item)
        detail = client.get_message(access_token="tkn", message_id="m1")

    assert detail.subject == ""
    assert detail.from_ == ""
    assert detail.rfc_message_id == ""


def test_get_message_is_unread_reflects_unread_label():
    client = GmailApiClient()
    item = _message_item(label_ids=("INBOX", "UNREAD"), body_b64="aGk=")
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=item)
        detail = client.get_message(access_token="tkn", message_id="m1")

    assert detail.is_unread is True


# -- GmailApiClient.get_message: nested MIME body/attachment walking -----


def test_get_message_extracts_text_and_html_from_nested_multipart():
    # multipart/mixed > multipart/alternative > {text/plain, text/html}
    parts = [
        {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "aGVsbG8="}},  # "hello"
                {"mimeType": "text/html", "body": {"data": "PGI+aGVsbG88L2I+"}},  # "<b>hello</b>"
            ],
        }
    ]
    client = GmailApiClient()
    item = _message_item(parts=parts)
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=item)
        detail = client.get_message(access_token="tkn", message_id="m1")

    assert detail.body_text == "hello"
    assert detail.body_html == "<b>hello</b>"


def test_get_message_collects_attachment_metadata_without_downloading_data():
    parts = [
        {"mimeType": "text/plain", "body": {"data": "aGk="}},
        {
            "mimeType": "application/pdf",
            "filename": "invoice.pdf",
            "body": {"attachmentId": "att1", "size": 1234},
        },
    ]
    client = GmailApiClient()
    item = _message_item(parts=parts)
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=item)
        detail = client.get_message(access_token="tkn", message_id="m1")

    assert detail.has_attachments is True
    assert len(detail.attachments) == 1
    att = detail.attachments[0]
    assert att.attachment_id == "att1"
    assert att.filename == "invoice.pdf"
    assert att.size == 1234


def test_get_message_requires_message_id_via_tool():
    tool = GetMessageTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "message_id": "  "})


def test_get_message_tool_requires_access_token():
    tool = GetMessageTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1"})


def test_get_message_tool_serializes_via_to_dict():
    message = MagicMock()
    message.to_dict.return_value = {"message_id": "m1", "subject": "Hi"}
    client = MagicMock()
    client.get_message.return_value = message
    tool = GetMessageTool(client=client)

    result = tool.execute({"access_token": "tkn", "message_id": "m1"})

    assert result.success is True
    assert result.data["message"] == {"message_id": "m1", "subject": "Hi"}
    assert client.get_message.call_args.kwargs["format"] == "full"


def test_get_message_404_raises_not_found():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(status_code=404)
        with pytest.raises(NotFoundError):
            client.get_message(access_token="tkn", message_id="missing")


# -- GmailApiClient.get_thread --------------------------------------------


def test_get_thread_returns_all_messages_in_order():
    client = GmailApiClient()
    thread_body = {
        "id": "t1",
        "messages": [
            _message_item("m1", "t1", headers=[_header("Subject", "First")], body_b64="MQ=="),
            _message_item("m2", "t1", headers=[_header("Subject", "Second")], body_b64="Mg=="),
        ],
    }
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body=thread_body)
        thread = client.get_thread(access_token="tkn", thread_id="t1")

    assert thread.thread_id == "t1"
    assert [m.subject for m in thread.messages] == ["First", "Second"]
    assert [m.body_text for m in thread.messages] == ["1", "2"]


def test_get_thread_tool_requires_thread_id():
    tool = GetThreadTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "thread_id": ""})


def test_get_thread_tool_serializes_via_to_dict():
    thread = MagicMock()
    thread.to_dict.return_value = {"thread_id": "t1", "messages": []}
    client = MagicMock()
    client.get_thread.return_value = thread
    tool = GetThreadTool(client=client)

    result = tool.execute({"access_token": "tkn", "thread_id": "t1"})

    assert result.success is True
    assert result.data["thread"] == {"thread_id": "t1", "messages": []}
