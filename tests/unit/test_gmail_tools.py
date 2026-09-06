"""Tests for `gmail.list_messages` / `gmail.search` tools (Task 26: verify
Gmail listing). Other Gmail tools (get_message, threads, labels, actions,
compose, conversations, attachments) get their own coverage in later
Phase 3/4 verification tasks.
"""

from unittest.mock import MagicMock

import pytest

from app.core.errors import ValidationError
from app.tools.gmail.list_messages import ListMessagesTool, SearchMessagesTool


def test_list_messages_tool_requires_access_token():
    tool = ListMessagesTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_list_messages_tool_defaults_to_inbox_when_no_query_or_labels():
    client = MagicMock()
    client.list_messages.return_value = ([], None)
    tool = ListMessagesTool(client=client)

    tool.execute({"access_token": "tkn"})

    assert client.list_messages.call_args.kwargs["label_ids"] == ["INBOX"]


def test_list_messages_tool_with_query_and_no_labels_searches_whole_mailbox():
    client = MagicMock()
    client.list_messages.return_value = ([], None)
    tool = ListMessagesTool(client=client)

    tool.execute({"access_token": "tkn", "query": "is:unread"})

    assert client.list_messages.call_args.kwargs["label_ids"] is None


def test_list_messages_tool_respects_explicit_label_ids_even_with_query():
    client = MagicMock()
    client.list_messages.return_value = ([], None)
    tool = ListMessagesTool(client=client)

    tool.execute({"access_token": "tkn", "query": "is:unread", "label_ids": ["STARRED"]})

    assert client.list_messages.call_args.kwargs["label_ids"] == ["STARRED"]


def test_list_messages_tool_serializes_messages_and_passes_through_page_token():
    message = MagicMock()
    message.to_dict.return_value = {"message_id": "m1"}
    client = MagicMock()
    client.list_messages.return_value = ([message], "next-tok")
    tool = ListMessagesTool(client=client)

    result = tool.execute({"access_token": "tkn"})

    assert result.success is True
    assert result.data["messages"] == [{"message_id": "m1"}]
    assert result.data["next_page_token"] == "next-tok"


def test_search_tool_requires_access_token():
    tool = SearchMessagesTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"query": "invoice"})


def test_search_tool_rejects_blank_query():
    tool = SearchMessagesTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tkn", "query": "   "})


def test_search_tool_trims_query_and_never_filters_by_label():
    client = MagicMock()
    client.list_messages.return_value = ([], None)
    tool = SearchMessagesTool(client=client)

    tool.execute({"access_token": "tkn", "query": "  from:boss@example.com  "})

    kwargs = client.list_messages.call_args.kwargs
    assert kwargs["query"] == "from:boss@example.com"
    assert kwargs["label_ids"] is None
