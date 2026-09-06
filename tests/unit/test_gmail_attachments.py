"""Tests for Gmail *attachments* - `gmail.get_attachment`
(Task 29: verify Gmail attachment handling).

Attachment *metadata* (filename, mime type, size, attachment_id) is
surfaced on every message via `gmail.get_message` and is already covered
by `tests/unit/test_gmail_reading.py`
(`test_get_message_collects_attachment_metadata_without_downloading_data`).
This file covers the one Phase-7 addition: downloading the actual bytes
for a given `attachment_id` (`GmailApiClient.get_attachment` /
`GetAttachmentTool`).
"""

import base64
from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import AuthenticationError, NotFoundError, ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.attachments import GetAttachmentTool


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


# -- GmailApiClient.get_attachment ----------------------------------------


def test_get_attachment_requires_access_token():
    client = GmailApiClient()
    with pytest.raises(AuthenticationError):
        client.get_attachment(access_token="", message_id="m1", attachment_id="a1")


def test_get_attachment_reencodes_base64url_to_standard_base64():
    # Exercise every byte value so both base64url-specific characters
    # ('-' and '_') actually appear in the fixture, not just typical text.
    raw = bytes(range(256))
    urlsafe = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    assert "-" in urlsafe or "_" in urlsafe  # sanity: fixture exercises the tricky chars

    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            json_body={"attachmentId": "att1", "size": len(raw), "data": urlsafe}
        )
        attachment = client.get_attachment(access_token="tkn", message_id="m1", attachment_id="att1")

    assert attachment.attachment_id == "att1"
    assert attachment.size == len(raw)
    # Standard base64 (no '-'/'_') that decodes back to the exact original bytes.
    assert "-" not in attachment.data_base64
    assert "_" not in attachment.data_base64
    assert base64.b64decode(attachment.data_base64) == raw


def test_get_attachment_handles_missing_data_without_crashing():
    # A zero-byte (or otherwise data-less) attachment response.
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"attachmentId": "att2", "size": 0})
        attachment = client.get_attachment(access_token="tkn", message_id="m1", attachment_id="att2")

    assert attachment.data_base64 == ""
    assert attachment.size == 0


def test_get_attachment_falls_back_to_requested_id_if_response_omits_it():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"data": "aGk=", "size": 2})
        attachment = client.get_attachment(access_token="tkn", message_id="m1", attachment_id="requested-id")

    assert attachment.attachment_id == "requested-id"


def test_get_attachment_404_raises_not_found():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(status_code=404)
        with pytest.raises(NotFoundError):
            client.get_attachment(access_token="tkn", message_id="m1", attachment_id="missing")


def test_get_attachment_401_raises_authentication_error():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(status_code=401)
        with pytest.raises(AuthenticationError):
            client.get_attachment(access_token="expired", message_id="m1", attachment_id="a1")


# -- GetAttachmentTool -----------------------------------------------------


def test_attachment_tool_requires_message_id_and_attachment_id():
    tool = GetAttachmentTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "message_id": "", "attachment_id": "a1"})
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "message_id": "m1", "attachment_id": "  "})


def test_attachment_tool_requires_access_token():
    tool = GetAttachmentTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "attachment_id": "a1"})


def test_attachment_tool_serializes_via_to_dict():
    attachment = MagicMock()
    attachment.to_dict.return_value = {"attachment_id": "a1", "size": 10, "data_base64": "AAAA"}
    client = MagicMock()
    client.get_attachment.return_value = attachment
    tool = GetAttachmentTool(client=client)

    result = tool.execute({"access_token": "tkn", "message_id": "m1", "attachment_id": "a1"})

    assert result.success is True
    assert result.data["attachment"] == {"attachment_id": "a1", "size": 10, "data_base64": "AAAA"}
    assert client.get_attachment.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "attachment_id": "a1",
    }
