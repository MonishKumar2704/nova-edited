"""`gmail.get_message` tool: fetch one message's full metadata + body (master spec Phase 6)."""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult


class GetMessageTool(Tool):
    name = "gmail.get_message"
    description = "Fetch one Gmail message by ID: headers, snippet, body text/HTML, and attachment metadata."
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "message_id": {"type": "string", "required": True},
        "format": {
            "type": "string",
            "required": False,
            "default": "full",
            "enum": ["full", "metadata", "minimal"],
        },
    }
    output_schema = {"message": {"type": "object", "items": "MessageDetail"}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for gmail.get_message.")

        message_id = (arguments.get("message_id") or "").strip()
        if not message_id:
            raise ValidationError("`message_id` is required for gmail.get_message.")

        message = self._client.get_message(
            access_token=access_token,
            message_id=message_id,
            format=arguments.get("format", "full"),
        )
        return ToolResult(success=True, data={"message": message.to_dict()})
