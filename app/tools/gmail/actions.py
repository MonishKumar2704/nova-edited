"""
`gmail.*` message-action tools (master spec Phase 7: mark read/unread,
archive, trash/untrash, star/unstar, label/unlabel).

Gmail implements every one of these as a label add/remove via
`messages.modify` (see `GmailApiClient.modify_message`) except trash/untrash,
which have their own dedicated endpoints. Each tool here just supplies the
right label delta / endpoint - all validation and HTTP concerns live in
`app.integrations.gmail_api`.

All of these mutate the user's real inbox, so (matching the YouTube
playlist-mutation precedent in `app.tools.youtube.playlists`)
`requires_confirmation = True` except the low-stakes, easily-reversible
read/unread toggle.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult

_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


def _require(arguments: dict[str, Any], key: str, tool_name: str) -> str:
    value = (arguments.get(key) or "").strip() if isinstance(arguments.get(key), str) else arguments.get(key)
    if not value:
        raise ValidationError(f"`{key}` is required for {tool_name}.")
    return value


def _require_access_token(arguments: dict[str, Any]) -> str:
    access_token = arguments.get("access_token")
    if not access_token:
        raise ValidationError("This operation requires a connected Google account (`access_token`).")
    return access_token


class _ModifyLabelsTool(Tool):
    """Base for the fixed label-delta actions (mark read/unread, archive, star, unstar)."""

    _add_labels: tuple[str, ...] = ()
    _remove_labels: tuple[str, ...] = ()

    input_schema = {"message_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"message": {"type": "object"}}
    permissions = [_MODIFY_SCOPE]

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = _require(arguments, "message_id", self.name)
        access_token = _require_access_token(arguments)
        message = self._client.modify_message(
            access_token=access_token,
            message_id=message_id,
            add_label_ids=list(self._add_labels) or None,
            remove_label_ids=list(self._remove_labels) or None,
        )
        return ToolResult(success=True, data={"message": message.to_dict()})


class MarkReadTool(_ModifyLabelsTool):
    name = "gmail.mark_read"
    description = "Mark a Gmail message as read (removes the UNREAD label)."
    _remove_labels = ("UNREAD",)
    requires_confirmation = False


class MarkUnreadTool(_ModifyLabelsTool):
    name = "gmail.mark_unread"
    description = "Mark a Gmail message as unread (adds the UNREAD label)."
    _add_labels = ("UNREAD",)
    requires_confirmation = False


class ArchiveMessageTool(_ModifyLabelsTool):
    name = "gmail.archive"
    description = "Archive a Gmail message (removes it from the inbox by removing the INBOX label)."
    _remove_labels = ("INBOX",)
    requires_confirmation = True


class StarMessageTool(_ModifyLabelsTool):
    name = "gmail.star"
    description = "Star a Gmail message."
    _add_labels = ("STARRED",)
    requires_confirmation = False


class UnstarMessageTool(_ModifyLabelsTool):
    name = "gmail.unstar"
    description = "Remove the star from a Gmail message."
    _remove_labels = ("STARRED",)
    requires_confirmation = False


class TrashMessageTool(Tool):
    name = "gmail.trash"
    description = "Move a Gmail message to Trash (recoverable via gmail.untrash until Gmail permanently deletes it)."
    input_schema = {"message_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"message": {"type": "object"}}
    permissions = [_MODIFY_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = _require(arguments, "message_id", self.name)
        access_token = _require_access_token(arguments)
        message = self._client.trash_message(access_token=access_token, message_id=message_id)
        return ToolResult(success=True, data={"message": message.to_dict()})


class UntrashMessageTool(Tool):
    name = "gmail.untrash"
    description = "Restore a Gmail message out of Trash."
    input_schema = {"message_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"message": {"type": "object"}}
    permissions = [_MODIFY_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = _require(arguments, "message_id", self.name)
        access_token = _require_access_token(arguments)
        message = self._client.untrash_message(access_token=access_token, message_id=message_id)
        return ToolResult(success=True, data={"message": message.to_dict()})


class AddLabelTool(Tool):
    name = "gmail.add_label"
    description = "Apply a label (system or user) to a Gmail message, e.g. INBOX, IMPORTANT, or a custom label ID."
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "label_id": {"type": "string", "required": True},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"message": {"type": "object"}}
    permissions = [_MODIFY_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = _require(arguments, "message_id", self.name)
        label_id = _require(arguments, "label_id", self.name)
        access_token = _require_access_token(arguments)
        message = self._client.modify_message(access_token=access_token, message_id=message_id, add_label_ids=[label_id])
        return ToolResult(success=True, data={"message": message.to_dict()})


class RemoveLabelTool(Tool):
    name = "gmail.remove_label"
    description = "Remove a label from a Gmail message."
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "label_id": {"type": "string", "required": True},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"message": {"type": "object"}}
    permissions = [_MODIFY_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = _require(arguments, "message_id", self.name)
        label_id = _require(arguments, "label_id", self.name)
        access_token = _require_access_token(arguments)
        message = self._client.modify_message(access_token=access_token, message_id=message_id, remove_label_ids=[label_id])
        return ToolResult(success=True, data={"message": message.to_dict()})
