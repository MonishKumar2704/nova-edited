"""`gmail.list_threads` / `gmail.get_thread` tools (master spec Phase 6: threads)."""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult


class ListThreadsTool(Tool):
    name = "gmail.list_threads"
    description = "List email threads (conversations) in the connected Gmail account, optionally filtered by label or search query."
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "query": {"type": "string", "required": False},
        "label_ids": {"type": "array", "required": False, "default": ["INBOX"]},
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
    }
    output_schema = {"threads": {"type": "array", "items": "ThreadSummary"}, "next_page_token": {"type": "string", "nullable": True}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for gmail.list_threads.")

        label_ids = arguments.get("label_ids")
        if label_ids is None:
            label_ids = None if arguments.get("query") else ["INBOX"]

        threads, next_page_token = self._client.list_threads(
            access_token=access_token,
            query=arguments.get("query"),
            label_ids=label_ids,
            max_results=int(arguments.get("max_results", 25)),
            page_token=arguments.get("page_token"),
        )
        return ToolResult(
            success=True,
            data={"threads": [t.to_dict() for t in threads], "next_page_token": next_page_token},
        )


class GetThreadTool(Tool):
    name = "gmail.get_thread"
    description = "Fetch a full email thread (all messages in a conversation) by thread ID."
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "thread_id": {"type": "string", "required": True},
    }
    output_schema = {"thread": {"type": "object", "items": "ThreadDetail"}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for gmail.get_thread.")

        thread_id = (arguments.get("thread_id") or "").strip()
        if not thread_id:
            raise ValidationError("`thread_id` is required for gmail.get_thread.")

        thread = self._client.get_thread(access_token=access_token, thread_id=thread_id)
        return ToolResult(success=True, data={"thread": thread.to_dict()})
