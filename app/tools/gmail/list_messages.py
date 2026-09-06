"""`gmail.list_messages` / `gmail.search` tools (master spec Phase 6: inbox, pagination, search foundation).

Both tools call the same underlying `GmailApiClient.list_messages` (Gmail's
`q` search-syntax parameter covers plain listing *and* search - there is no
separate "search" endpoint). They are kept as two `Tool`s rather than one
because they have different defaults and different callers: `list_messages`
defaults to the inbox and is what a "show me my inbox" command reaches for,
while `search` requires an explicit query and is what a natural-language
Gmail search (Phase 11) or an LLM tool-call reaches for.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult


class ListMessagesTool(Tool):
    name = "gmail.list_messages"
    description = (
        "List messages in the connected Gmail account, most recent first. Defaults to the inbox; "
        "optionally filter by label(s) or a Gmail search query."
    )
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "query": {"type": "string", "required": False, "description": "Gmail search syntax, e.g. 'is:unread from:x'."},
        "label_ids": {"type": "array", "required": False, "default": ["INBOX"]},
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
    }
    output_schema = {"messages": {"type": "array", "items": "MessageSummary"}, "next_page_token": {"type": "string", "nullable": True}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for gmail.list_messages.")

        label_ids = arguments.get("label_ids")
        if label_ids is None:
            # Default to the inbox only when no query is given either -
            # a bare `query` with no label filter should search the whole
            # mailbox, matching what a person typing that query into
            # Gmail's own search box would expect.
            label_ids = None if arguments.get("query") else ["INBOX"]

        messages, next_page_token = self._client.list_messages(
            access_token=access_token,
            query=arguments.get("query"),
            label_ids=label_ids,
            max_results=int(arguments.get("max_results", 25)),
            page_token=arguments.get("page_token"),
        )
        return ToolResult(
            success=True,
            data={"messages": [m.to_dict() for m in messages], "next_page_token": next_page_token},
        )


class SearchMessagesTool(Tool):
    name = "gmail.search"
    description = (
        "Search the connected Gmail account using Gmail's search syntax "
        "(e.g. 'is:unread from:college.edu', 'subject:invoice after:2026/01/01')."
    )
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "query": {"type": "string", "required": True},
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
    }
    output_schema = {"messages": {"type": "array", "items": "MessageSummary"}, "next_page_token": {"type": "string", "nullable": True}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for gmail.search.")

        query = (arguments.get("query") or "").strip()
        if not query:
            raise ValidationError("`query` is required for gmail.search.")

        messages, next_page_token = self._client.list_messages(
            access_token=access_token,
            query=query,
            label_ids=None,
            max_results=int(arguments.get("max_results", 25)),
            page_token=arguments.get("page_token"),
        )
        return ToolResult(
            success=True,
            data={"messages": [m.to_dict() for m in messages], "next_page_token": next_page_token},
        )
