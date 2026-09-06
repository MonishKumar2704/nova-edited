"""Tests for Gmail *labels* - `gmail.list_labels`, `gmail.add_label`,
`gmail.remove_label` (Task 30: verify Gmail label operations).

Labels cover two things in this codebase:
- Listing all labels in the account (`GmailApiClient.list_labels` /
  `ListLabelsTool`, Phase 6).
- Adding/removing a label on one message (`AddLabelTool` /
  `RemoveLabelTool`, Phase 7), which - like mark read/unread, archive,
  and star/unstar - is just a thin wrapper over
  `GmailApiClient.modify_message` (`messages.modify`).

`modify_message`'s HTTP mechanics (401/404/network handling) are shared
with mark/archive/star, already covered by `tests/unit/test_gmail_tools.py`
and `tests/unit/test_gmail_api_client.py`; this file focuses on the
label-specific pieces: `list_labels` parsing, and the add/remove label
delta each tool sends.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import AuthenticationError, NotFoundError, ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.actions import AddLabelTool, RemoveLabelTool
from app.tools.gmail.labels import ListLabelsTool


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


# -- GmailApiClient.list_labels --------------------------------------------


def test_list_labels_parses_system_and_user_labels():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            json_body={
                "labels": [
                    {"id": "INBOX", "name": "INBOX", "type": "system", "messagesTotal": 10, "messagesUnread": 2},
                    {"id": "Label_1", "name": "Work", "type": "user"},
                ]
            }
        )
        labels = client.list_labels(access_token="tkn")

    assert [label.to_dict() for label in labels] == [
        {"label_id": "INBOX", "name": "INBOX", "type": "system", "messages_total": 10, "messages_unread": 2},
        {"label_id": "Label_1", "name": "Work", "type": "user", "messages_total": None, "messages_unread": None},
    ]


def test_list_labels_handles_missing_labels_key_without_crashing():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(json_body={})
        labels = client.list_labels(access_token="tkn")

    assert labels == []


def test_list_labels_requires_access_token():
    client = GmailApiClient()
    with pytest.raises(AuthenticationError):
        client.list_labels(access_token="")


def test_list_labels_401_raises_authentication_error():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.get") as mock_get:
        mock_get.return_value = _mock_response(status_code=401)
        with pytest.raises(AuthenticationError):
            client.list_labels(access_token="expired")


# -- ListLabelsTool ----------------------------------------------------------


def test_list_labels_tool_requires_access_token():
    tool = ListLabelsTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_list_labels_tool_serializes_each_label_via_to_dict():
    label = MagicMock()
    label.to_dict.return_value = {"label_id": "INBOX", "name": "INBOX", "type": "system"}
    client = MagicMock()
    client.list_labels.return_value = [label]
    tool = ListLabelsTool(client=client)

    result = tool.execute({"access_token": "tkn"})

    assert result.success is True
    assert result.data["labels"] == [{"label_id": "INBOX", "name": "INBOX", "type": "system"}]
    assert client.list_labels.call_args.kwargs == {"access_token": "tkn"}


# -- GmailApiClient.modify_message (label delta specifics) ------------------


def test_modify_message_sends_add_label_ids():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(json_body={"id": "m1", "labelIds": ["INBOX", "Label_1"]})
        client.modify_message(access_token="tkn", message_id="m1", add_label_ids=["Label_1"])

    assert mock_req.call_args.kwargs["json"] == {"addLabelIds": ["Label_1"]}


def test_modify_message_sends_remove_label_ids():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(json_body={"id": "m1", "labelIds": ["INBOX"]})
        client.modify_message(access_token="tkn", message_id="m1", remove_label_ids=["Label_1"])

    assert mock_req.call_args.kwargs["json"] == {"removeLabelIds": ["Label_1"]}


def test_modify_message_404_raises_not_found():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_req:
        mock_req.return_value = _mock_response(status_code=404)
        with pytest.raises(NotFoundError):
            client.modify_message(access_token="tkn", message_id="missing", add_label_ids=["Label_1"])


# -- AddLabelTool / RemoveLabelTool ------------------------------------------


def test_add_label_tool_requires_message_id_and_label_id():
    tool = AddLabelTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "message_id": "", "label_id": "Label_1"})
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "message_id": "m1", "label_id": "  "})


def test_add_label_tool_requires_access_token():
    tool = AddLabelTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1", "label_id": "Label_1"})


def test_add_label_tool_calls_modify_message_with_add_only():
    client = MagicMock()
    message = MagicMock()
    message.to_dict.return_value = {"id": "m1", "label_ids": ["INBOX", "Label_1"]}
    client.modify_message.return_value = message
    tool = AddLabelTool(client=client)

    result = tool.execute({"message_id": "m1", "label_id": "Label_1", "access_token": "tkn"})

    assert result.success is True
    assert result.data["message"] == {"id": "m1", "label_ids": ["INBOX", "Label_1"]}
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "add_label_ids": ["Label_1"],
    }


def test_remove_label_tool_requires_message_id_and_label_id():
    tool = RemoveLabelTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "message_id": "m1", "label_id": ""})


def test_remove_label_tool_calls_modify_message_with_remove_only():
    client = MagicMock()
    message = MagicMock()
    message.to_dict.return_value = {"id": "m1", "label_ids": ["INBOX"]}
    client.modify_message.return_value = message
    tool = RemoveLabelTool(client=client)

    result = tool.execute({"message_id": "m1", "label_id": "Label_1", "access_token": "tkn"})

    assert result.success is True
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "remove_label_ids": ["Label_1"],
    }


def test_add_and_remove_label_tools_do_not_require_confirmation():
    # Unlike archive/trash, applying or removing a label is low-stakes and
    # easily reversible, matching mark read/unread and star/unstar.
    assert AddLabelTool.requires_confirmation is False
    assert RemoveLabelTool.requires_confirmation is False
