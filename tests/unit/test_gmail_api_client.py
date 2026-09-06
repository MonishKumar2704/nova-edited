"""Tests for `GmailApiClient.list_messages` (Task 26: verify Gmail listing).

Scoped to listing/search only, mirroring `test_youtube_api_client.py`'s
pattern for the YouTube client. Other `GmailApiClient` methods (get_message
detail parsing, threads, labels, drafts, actions, conversations,
attachments) are exercised by their own verification tasks later in
Phase 3/4 of the roadmap.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import AuthenticationError, RateLimitError
from app.integrations.gmail_api import GmailApiClient

LIST_BODY = {
    "messages": [{"id": "m1", "threadId": "t1"}, {"id": "m2", "threadId": "t2"}],
    "nextPageToken": "tok2",
}


def _message_item(message_id: str, thread_id: str, *, unread: bool) -> dict:
    label_ids = ["INBOX", "UNREAD"] if unread else ["INBOX"]
    return {
        "id": message_id,
        "threadId": thread_id,
        "labelIds": label_ids,
        "snippet": "snippet text",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Hello"},
                {"name": "From", "value": "a@example.com"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"},
            ]
        },
    }


def _mock_response(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = text
    return resp


def test_list_messages_requires_access_token():
    client = GmailApiClient()
    with pytest.raises(AuthenticationError):
        client.list_messages(access_token="", query=None, label_ids=None)


def test_list_messages_resolves_ids_and_returns_next_page_token():
    client = GmailApiClient()
    m1 = _message_item("m1", "t1", unread=True)
    m2 = _message_item("m2", "t2", unread=False)
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json_body=LIST_BODY),
            _mock_response(json_body=m1),
            _mock_response(json_body=m2),
        ]
        summaries, next_token = client.list_messages(
            access_token="tkn", query=None, label_ids=["INBOX"], max_results=25
        )

    assert next_token == "tok2"
    assert [s.message_id for s in summaries] == ["m1", "m2"]
    assert summaries[0].is_unread is True
    assert summaries[1].is_unread is False

    list_call_params = mock_get.call_args_list[0].kwargs["params"]
    assert list_call_params["labelIds"] == ["INBOX"]
    assert list_call_params["maxResults"] == 25

    detail_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert detail_call_params["format"] == "metadata"


def test_list_messages_max_results_clamped_between_1_and_50():
    client = GmailApiClient()

    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"messages": []})
        client.list_messages(access_token="tkn", query=None, label_ids=None, max_results=999)
    assert mock_get.call_args.kwargs["params"]["maxResults"] == 50

    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"messages": []})
        client.list_messages(access_token="tkn", query=None, label_ids=None, max_results=0)
    assert mock_get.call_args.kwargs["params"]["maxResults"] == 1


def test_list_messages_with_query_and_no_label_filter_searches_whole_mailbox():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"messages": []})
        client.list_messages(access_token="tkn", query="is:unread", label_ids=None, max_results=10)

    params = mock_get.call_args.kwargs["params"]
    assert params["q"] == "is:unread"
    assert "labelIds" not in params


def test_list_messages_empty_page_returns_empty_list_and_no_next_token():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={"messages": []})
        summaries, next_token = client.list_messages(access_token="tkn", query=None, label_ids=["INBOX"])

    assert summaries == []
    assert next_token is None


def test_list_messages_401_raises_authentication_error():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(status_code=401)
        with pytest.raises(AuthenticationError):
            client.list_messages(access_token="expired", query=None, label_ids=["INBOX"])


def test_list_messages_403_quota_raises_rate_limit_error():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            status_code=403, json_body={"error": {"message": "Quota exceeded for quota metric"}}
        )
        with pytest.raises(RateLimitError):
            client.list_messages(access_token="tkn", query=None, label_ids=["INBOX"])


def test_list_messages_403_non_quota_raises_authentication_error():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            status_code=403, json_body={"error": {"message": "Insufficient Permission"}}
        )
        with pytest.raises(AuthenticationError):
            client.list_messages(access_token="tkn", query=None, label_ids=["INBOX"])
