"""`gmail.list_labels` tool (master spec Phase 6: labels)."""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult


class ListLabelsTool(Tool):
    name = "gmail.list_labels"
    description = "List all labels (system, e.g. INBOX/UNREAD, and user-created) in the connected Gmail account."
    input_schema = {"access_token": {"type": "string", "required": True}}
    output_schema = {"labels": {"type": "array", "items": "LabelSummary"}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for gmail.list_labels.")

        labels = self._client.list_labels(access_token=access_token)
        return ToolResult(success=True, data={"labels": [label.to_dict() for label in labels]})
