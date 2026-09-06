"""Tests for Gmail *organization* actions - mark read/unread, star/unstar,
archive, trash/untrash (Task 35: verify Gmail organization).

These are all thin wrappers over `GmailApiClient.modify_message` (a fixed
label add/remove delta) except trash/untrash, which have their own
dedicated Gmail API endpoints (`messages.trash`/`messages.untrash`).
`modify_message`'s generic HTTP mechanics (401/403/404/network handling)
are already covered end to end by `test_gmail_labels.py`'s
`AddLabelTool`/`RemoveLabelTool` tests and `test_gmail_api_client.py`'s
`list_messages` tests; this file focuses on:

- `GmailApiClient.modify_message`/`trash_message`/`untrash_message`
  actually building the right request for each of the seven actions.
- Each `Tool` subclass in `app.tools.gmail.actions` sending the correct,
  fixed label delta (or trash/untrash call) and nothing else.
- Validation (missing `message_id`/`access_token`) matching every other
  Gmail tool.
- `requires_confirmation` matching the module's own documented policy:
  mark read/unread and star/unstar are low-stakes and instantly
  reversible (`False`); archive and trash are not (`True`); untrash - a
  deliberate undo of a confirmed trash - is `False`.
- The `gmail_message` dynamic-UI card actually offering both directions
  of every toggle (read <-> unread, star <-> unstar) so a user can act
  on a card without also needing the `mark_unread`/`unstar` tool to be
  reachable only via the AI agent.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.services.dynamic_ui import GMAIL_MESSAGE_ACTIONS, build_gmail_message_card
from app.tools.gmail.actions import (
    ArchiveMessageTool,
    MarkReadTool,
    MarkUnreadTool,
    StarMessageTool,
    TrashMessageTool,
    UnstarMessageTool,
    UntrashMessageTool,
)


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = ""
    resp.content = b"{}"
    return resp


_MESSAGE_ITEM = {"id": "m1", "threadId": "t1", "labelIds": ["INBOX"], "snippet": "hi", "payload": {"headers": []}}


# -- GmailApiClient.modify_message / trash_message / untrash_message -------


def test_modify_message_sends_add_and_remove_label_deltas():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=_MESSAGE_ITEM)
        client.modify_message(access_token="tkn", message_id="m1", add_label_ids=["STARRED"], remove_label_ids=["UNREAD"])

    assert mock_request.call_args.kwargs["json"] == {"addLabelIds": ["STARRED"], "removeLabelIds": ["UNREAD"]}
    assert "/messages/m1/modify" in mock_request.call_args.args[1]


def test_modify_message_omits_empty_deltas():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=_MESSAGE_ITEM)
        client.modify_message(access_token="tkn", message_id="m1", add_label_ids=None, remove_label_ids=None)

    assert mock_request.call_args.kwargs["json"] == {}


def test_trash_message_posts_to_trash_endpoint_with_no_body():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=_MESSAGE_ITEM)
        client.trash_message(access_token="tkn", message_id="m1")

    assert "/messages/m1/trash" in mock_request.call_args.args[1]
    assert mock_request.call_args.kwargs["json"] is None


def test_untrash_message_posts_to_untrash_endpoint_with_no_body():
    client = GmailApiClient()
    with patch("app.integrations.gmail_api.requests.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=_MESSAGE_ITEM)
        client.untrash_message(access_token="tkn", message_id="m1")

    assert "/messages/m1/untrash" in mock_request.call_args.args[1]
    assert mock_request.call_args.kwargs["json"] is None


# -- Tool-level label deltas -------------------------------------------------


def test_mark_read_removes_unread_label():
    client = MagicMock()
    client.modify_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    MarkReadTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "add_label_ids": None,
        "remove_label_ids": ["UNREAD"],
    }


def test_mark_unread_adds_unread_label():
    client = MagicMock()
    client.modify_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    MarkUnreadTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "add_label_ids": ["UNREAD"],
        "remove_label_ids": None,
    }


def test_archive_removes_inbox_label():
    client = MagicMock()
    client.modify_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    ArchiveMessageTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "add_label_ids": None,
        "remove_label_ids": ["INBOX"],
    }


def test_star_adds_starred_label():
    client = MagicMock()
    client.modify_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    StarMessageTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "add_label_ids": ["STARRED"],
        "remove_label_ids": None,
    }


def test_unstar_removes_starred_label():
    client = MagicMock()
    client.modify_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    UnstarMessageTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    assert client.modify_message.call_args.kwargs == {
        "access_token": "tkn",
        "message_id": "m1",
        "add_label_ids": None,
        "remove_label_ids": ["STARRED"],
    }


def test_trash_tool_calls_client_trash_message():
    client = MagicMock()
    client.trash_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    result = TrashMessageTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    client.trash_message.assert_called_once_with(access_token="tkn", message_id="m1")
    assert result.success is True
    assert result.data == {"message": {"id": "m1"}}


def test_untrash_tool_calls_client_untrash_message():
    client = MagicMock()
    client.untrash_message.return_value = MagicMock(to_dict=lambda: {"id": "m1"})
    result = UntrashMessageTool(client=client).execute({"message_id": "m1", "access_token": "tkn"})
    client.untrash_message.assert_called_once_with(access_token="tkn", message_id="m1")
    assert result.success is True


# -- Validation (shared across all seven tools) ------------------------------


@pytest.mark.parametrize(
    "tool_cls",
    [MarkReadTool, MarkUnreadTool, ArchiveMessageTool, StarMessageTool, UnstarMessageTool, TrashMessageTool, UntrashMessageTool],
)
def test_each_tool_requires_message_id(tool_cls):
    tool = tool_cls(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "", "access_token": "tkn"})
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn"})


@pytest.mark.parametrize(
    "tool_cls",
    [MarkReadTool, MarkUnreadTool, ArchiveMessageTool, StarMessageTool, UnstarMessageTool, TrashMessageTool, UntrashMessageTool],
)
def test_each_tool_requires_access_token(tool_cls):
    tool = tool_cls(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"message_id": "m1"})


# -- requires_confirmation policy --------------------------------------------


def test_confirmation_policy_matches_documented_reversibility():
    # Low-stakes, instantly reversible -> no confirmation.
    assert MarkReadTool.requires_confirmation is False
    assert MarkUnreadTool.requires_confirmation is False
    assert StarMessageTool.requires_confirmation is False
    assert UnstarMessageTool.requires_confirmation is False
    assert UntrashMessageTool.requires_confirmation is False
    # Touch real account state (message leaves the inbox / moves to
    # Trash) -> require confirmation before running.
    assert ArchiveMessageTool.requires_confirmation is True
    assert TrashMessageTool.requires_confirmation is True


# -- Dynamic-UI card: both directions of every toggle must be offered -------


def test_gmail_message_actions_include_both_directions_of_read_toggle():
    action_ids = {a["id"] for a in GMAIL_MESSAGE_ACTIONS}
    assert {"mark_read", "mark_unread"} <= action_ids


def test_gmail_message_actions_include_both_directions_of_star_toggle():
    action_ids = {a["id"] for a in GMAIL_MESSAGE_ACTIONS}
    assert {"star", "unstar"} <= action_ids


def test_gmail_message_actions_confirmation_flags():
    by_id = {a["id"]: a["requires_confirmation"] for a in GMAIL_MESSAGE_ACTIONS}
    assert by_id["mark_read"] is False
    assert by_id["mark_unread"] is False
    assert by_id["star"] is False
    assert by_id["unstar"] is False
    assert by_id["archive"] is True
    assert by_id["trash"] is True


def test_build_gmail_message_card_exposes_organization_actions():
    card = build_gmail_message_card({"message_id": "m1", "subject": "Hi", "is_unread": True})
    action_ids = {a["id"] for a in card["actions"]}
    assert {"mark_read", "mark_unread", "star", "unstar", "archive", "trash"} <= action_ids
