"""`gmail.get_attachment` tool (master spec Phase 7: attachment download).

Attachment *metadata* (filename, mime type, size, attachment_id) is already
surfaced on every message via `gmail.get_message` (Phase 6, `_walk_parts`).
This tool is the one Phase-7 addition: fetching the actual bytes for a given
`attachment_id`, base64-encoded for the caller (e.g. so the frontend can
build a download link or the LLM layer can pass it to a future "attach to
draft" flow).
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult


class GetAttachmentTool(Tool):
    name = "gmail.get_attachment"
    description = "Download one attachment's data (base64-encoded) from a Gmail message, given its attachment_id."
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "attachment_id": {"type": "string", "required": True},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"attachment": {"type": "object", "items": "AttachmentData"}}
    permissions = ["https://www.googleapis.com/auth/gmail.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = (arguments.get("message_id") or "").strip()
        attachment_id = (arguments.get("attachment_id") or "").strip()
        if not message_id or not attachment_id:
            raise ValidationError("`message_id` and `attachment_id` are required for gmail.get_attachment.")
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("This operation requires a connected Google account (`access_token`).")

        attachment = self._client.get_attachment(access_token=access_token, message_id=message_id, attachment_id=attachment_id)
        return ToolResult(success=True, data={"attachment": attachment.to_dict()})
