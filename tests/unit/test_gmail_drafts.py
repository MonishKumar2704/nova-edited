"""Tests for Gmail *drafts* - create/update/delete (Task 31: verify Gmail
draft operations).

`gmail.draft.list` and `gmail.draft.send` are separate roadmap items
(covered incidentally elsewhere / by Phase 6 inventory); this file's scope
is exactly what Task 31 asks for: create, update, and delete a draft -
`CreateDraftTool` / `UpdateDraftTool` / `DeleteDraftTool` and their
`GmailApiClient` counterparts (`create_draft`, `update_draft`,
`delete_draft`), including the shared `_build_raw_mime` RFC 2822 encoding
those first two rely on.
"""

import base64
import email
from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import NotFoundError, ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.compose import CreateDraftTool, DeleteDraftTool, UpdateDraftTool


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


def _decode_raw_mime(raw: str) -> email.message.Message:
    padded = raw + "=" * (-len(raw) % 4)
    return email.message_from_bytes(base64.urlsafe_b64decode(padded))


# -- GmailApiClient.create_draft --------------------------------------------


def test_create_draft_posts_to_drafts_endpoint_with_correct_raw_mime():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(
            json_body={"id": "d1", "message": {"id": "m1", "labelIds": ["DRAFT"]}}
        )
        draft = client.create_draft(
            access_token="tkn",
            to=["a@example.com"],
            subject="Hi there",
            body_text="Hello, this is the body.",
            cc=["c@example.com"],
        )

    method, url = mock_req.call_args[0]
    assert method == "post"
    assert url.endswith("/drafts")

    body = mock_req.call_args.kwargs["json"]
    assert list(body.keys()) == ["message"]
    msg = _decode_raw_mime(body["message"]["raw"])
    assert msg["To"] == "a@example.com"
    assert msg["Cc"] == "c@example.com"
    assert msg["Subject"] == "Hi there"

    assert draft.draft_id == "d1"
    assert draft.to_dict()["draft_id"] == "d1"


def test_create_draft_includes_thread_id_when_given():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(json_body={"id": "d1", "message": {"id": "m1"}})
        client.create_draft(
            access_token="tkn", to=["a@example.com"], subject="s", body_text="b", thread_id="t1"
        )

    assert mock_req.call_args.kwargs["json"]["message"]["threadId"] == "t1"


def test_create_draft_omits_thread_id_when_not_given():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(json_body={"id": "d1", "message": {"id": "m1"}})
        client.create_draft(access_token="tkn", to=["a@example.com"], subject="s", body_text="b")

    assert "threadId" not in mock_req.call_args.kwargs["json"]["message"]


def test_create_draft_falls_back_to_requested_draft_id_shape_when_id_missing():
    # If Gmail's response ever omits the draft id, DraftSummary should not
    # crash - it should just carry an empty draft_id rather than raising.
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(json_body={"message": {"id": "m1"}})
        draft = client.create_draft(access_token="tkn", to=["a@example.com"], subject="s", body_text="b")

    assert draft.draft_id == ""


# -- GmailApiClient.update_draft ---------------------------------------------


def test_update_draft_puts_to_correct_draft_id_endpoint():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(
            json_body={"id": "d1", "message": {"id": "m1", "labelIds": ["DRAFT"]}}
        )
        draft = client.update_draft(
            access_token="tkn", draft_id="d1", to=["a@example.com"], subject="Updated", body_text="New body"
        )

    method, url = mock_req.call_args[0]
    assert method == "put"
    assert url.endswith("/drafts/d1")
    msg = _decode_raw_mime(mock_req.call_args.kwargs["json"]["message"]["raw"])
    assert msg["Subject"] == "Updated"
    assert draft.draft_id == "d1"


def test_update_draft_404_raises_not_found():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(status_code=404)
        with pytest.raises(NotFoundError):
            client.update_draft(access_token="tkn", draft_id="missing", to=["a@example.com"], subject="s", body_text="b")


# -- GmailApiClient.delete_draft ----------------------------------------------


def test_delete_draft_calls_delete_on_correct_endpoint():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        resp = _mock_response(status_code=204)
        resp.content = b""
        mock_req.return_value = resp
        client.delete_draft(access_token="tkn", draft_id="d1")

    method, url = mock_req.call_args[0]
    assert method == "delete"
    assert url.endswith("/drafts/d1")


def test_delete_draft_404_raises_not_found():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(status_code=404)
        with pytest.raises(NotFoundError):
            client.delete_draft(access_token="tkn", draft_id="missing")


# -- CreateDraftTool -----------------------------------------------------------


def test_create_draft_tool_rejects_missing_or_malformed_recipients():
    tool = CreateDraftTool(client=MagicMock())
    for bad_to in (None, [], ["not-an-email"], "a@example.com"):
        with pytest.raises(ValidationError):
            tool.execute({"to": bad_to, "subject": "s", "body_text": "b", "access_token": "tkn"})


def test_create_draft_tool_requires_access_token():
    tool = CreateDraftTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"to": ["a@example.com"], "subject": "s", "body_text": "b"})


def test_create_draft_tool_calls_client_and_serializes_result():
    client = MagicMock()
    draft = MagicMock()
    draft.to_dict.return_value = {"draft_id": "d1", "message": {"id": "m1"}}
    client.create_draft.return_value = draft
    tool = CreateDraftTool(client=client)

    result = tool.execute(
        {"to": ["a@example.com"], "subject": "Hi", "body_text": "Body", "access_token": "tkn"}
    )

    assert result.success is True
    assert result.data["draft"] == {"draft_id": "d1", "message": {"id": "m1"}}
    assert client.create_draft.call_args.kwargs == {
        "access_token": "tkn",
        "to": ["a@example.com"],
        "subject": "Hi",
        "body_text": "Body",
        "cc": None,
        "bcc": None,
    }


def test_create_draft_tool_does_not_require_confirmation():
    # Drafts stay local to the account until explicitly sent, so creating
    # one is not a sensitive action (unlike gmail.send / gmail.draft.send).
    assert CreateDraftTool.requires_confirmation is False


# -- UpdateDraftTool ------------------------------------------------------------


def test_update_draft_tool_requires_draft_id():
    tool = UpdateDraftTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"draft_id": "", "to": ["a@example.com"], "subject": "s", "body_text": "b", "access_token": "tkn"})
    with pytest.raises(ValidationError):
        tool.execute({"draft_id": "  ", "to": ["a@example.com"], "subject": "s", "body_text": "b", "access_token": "tkn"})


def test_update_draft_tool_calls_client_with_draft_id():
    client = MagicMock()
    draft = MagicMock()
    draft.to_dict.return_value = {"draft_id": "d1", "message": {"id": "m1"}}
    client.update_draft.return_value = draft
    tool = UpdateDraftTool(client=client)

    result = tool.execute(
        {"draft_id": "d1", "to": ["a@example.com"], "subject": "Updated", "body_text": "New body", "access_token": "tkn"}
    )

    assert result.success is True
    assert client.update_draft.call_args.kwargs["draft_id"] == "d1"


def test_update_draft_tool_does_not_require_confirmation():
    assert UpdateDraftTool.requires_confirmation is False


# -- DeleteDraftTool ------------------------------------------------------------


def test_delete_draft_tool_requires_draft_id():
    tool = DeleteDraftTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"draft_id": "", "access_token": "tkn"})


def test_delete_draft_tool_requires_access_token():
    tool = DeleteDraftTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"draft_id": "d1"})


def test_delete_draft_tool_calls_client_and_returns_deleted_flag():
    client = MagicMock()
    tool = DeleteDraftTool(client=client)

    result = tool.execute({"draft_id": "d1", "access_token": "tkn"})

    assert result.success is True
    assert result.data == {"deleted": True, "draft_id": "d1"}
    client.delete_draft.assert_called_once_with(access_token="tkn", draft_id="d1")


def test_delete_draft_tool_requires_confirmation():
    # Permanent, irreversible - unlike create/update which stay local and
    # editable, this is destructive and matches archive/trash's precedent.
    assert DeleteDraftTool.requires_confirmation is True
