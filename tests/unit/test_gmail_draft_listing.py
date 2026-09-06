"""Tests for Gmail *draft listing* - `gmail.draft.list` (Task 10: verify
Gmail drafts).

`test_gmail_drafts.py` explicitly scopes itself to create/update/delete and
calls out `gmail.draft.list` and `gmail.draft.send` as "separate roadmap
items" - `gmail.draft.send` picked up its own coverage in
`test_gmail_send.py`, but `GmailApiClient.list_drafts`/`get_draft` and
`ListDraftsTool` were never actually covered anywhere. This file closes
that gap, mirroring the existing `list_messages` test pattern in
`test_gmail_api_client.py`.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import AuthenticationError, NotFoundError, ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.compose import ListDraftsTool


def _draft_message_item(message_id="m1", thread_id="t1", *, subject="Hello") -> dict:
    return {
        "id": message_id,
        "threadId": thread_id,
        "labelIds": ["DRAFT"],
        "snippet": "a snippet",
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "To", "value": "a@example.com"},
            ]
        },
    }


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


# -- GmailApiClient.get_draft ------------------------------------------------


def test_get_draft_requests_full_format_and_wraps_message():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            json_body={"id": "d1", "message": _draft_message_item()}
        )
        draft = client.get_draft(access_token="tkn", draft_id="d1")

    assert mock_get.call_args.args[0].endswith("/drafts/d1")
    assert mock_get.call_args.kwargs["params"]["format"] == "full"
    assert draft.draft_id == "d1"
    assert draft.message.message_id == "m1"
    assert draft.message.subject == "Hello"


def test_get_draft_falls_back_to_requested_id_when_response_omits_it():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"message": _draft_message_item()})
        draft = client.get_draft(access_token="tkn", draft_id="d1")

    assert draft.draft_id == "d1"


def test_get_draft_404_raises_not_found():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(status_code=404)
        with pytest.raises(NotFoundError):
            client.get_draft(access_token="tkn", draft_id="missing")


# -- GmailApiClient.list_drafts -----------------------------------------------


def test_list_drafts_requires_access_token():
    client = GmailApiClient()
    with pytest.raises(AuthenticationError):
        client.list_drafts(access_token="")


def test_list_drafts_resolves_each_ref_and_returns_next_page_token():
    client = GmailApiClient()
    list_body = {"drafts": [{"id": "d1"}, {"id": "d2"}], "nextPageToken": "tok2"}
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json_body=list_body),
            _mock_response(json_body={"id": "d1", "message": _draft_message_item("m1", "t1", subject="First")}),
            _mock_response(json_body={"id": "d2", "message": _draft_message_item("m2", "t2", subject="Second")}),
        ]
        drafts, next_token = client.list_drafts(access_token="tkn", max_results=25)

    assert next_token == "tok2"
    assert [d.draft_id for d in drafts] == ["d1", "d2"]
    assert [d.message.subject for d in drafts] == ["First", "Second"]

    list_call_params = mock_get.call_args_list[0].kwargs["params"]
    assert list_call_params["maxResults"] == 25


def test_list_drafts_max_results_clamped_between_1_and_50():
    client = GmailApiClient()

    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"drafts": []})
        client.list_drafts(access_token="tkn", max_results=999)
    assert mock_get.call_args.kwargs["params"]["maxResults"] == 50

    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"drafts": []})
        client.list_drafts(access_token="tkn", max_results=0)
    assert mock_get.call_args.kwargs["params"]["maxResults"] == 1


def test_list_drafts_passes_page_token_through():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"drafts": []})
        client.list_drafts(access_token="tkn", page_token="tok1")

    assert mock_get.call_args.kwargs["params"]["pageToken"] == "tok1"


def test_list_drafts_skips_refs_with_no_id():
    client = GmailApiClient()
    list_body = {"drafts": [{"id": "d1"}, {}]}
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json_body=list_body),
            _mock_response(json_body={"id": "d1", "message": _draft_message_item()}),
        ]
        drafts, _ = client.list_drafts(access_token="tkn")

    assert [d.draft_id for d in drafts] == ["d1"]
    assert mock_get.call_count == 2


def test_list_drafts_empty_page_returns_empty_list_and_no_next_token():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"drafts": []})
        drafts, next_token = client.list_drafts(access_token="tkn")

    assert drafts == []
    assert next_token is None


# -- ListDraftsTool (gmail.draft.list) ----------------------------------------


def test_list_drafts_tool_requires_access_token():
    tool = ListDraftsTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_list_drafts_tool_defaults_max_results_to_25():
    client = MagicMock()
    client.list_drafts.return_value = ([], None)
    tool = ListDraftsTool(client=client)

    tool.execute({"access_token": "tkn"})

    assert client.list_drafts.call_args.kwargs["max_results"] == 25


def test_list_drafts_tool_passes_page_token_through():
    client = MagicMock()
    client.list_drafts.return_value = ([], None)
    tool = ListDraftsTool(client=client)

    tool.execute({"access_token": "tkn", "page_token": "tok1"})

    assert client.list_drafts.call_args.kwargs["page_token"] == "tok1"


def test_list_drafts_tool_serializes_drafts_and_next_page_token():
    client = MagicMock()
    draft = MagicMock()
    draft.to_dict.return_value = {"draft_id": "d1", "message": {"id": "m1"}}
    client.list_drafts.return_value = ([draft], "tok2")
    tool = ListDraftsTool(client=client)

    result = tool.execute({"access_token": "tkn"})

    assert result.success is True
    assert result.data["drafts"] == [{"draft_id": "d1", "message": {"id": "m1"}}]
    assert result.data["next_page_token"] == "tok2"


def test_list_drafts_tool_does_not_require_confirmation():
    # Listing is read-only, like every other Phase 6/7 read tool.
    assert ListDraftsTool.requires_confirmation is False
