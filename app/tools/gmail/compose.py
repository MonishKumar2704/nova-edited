"""
`gmail.draft.*` / `gmail.send` composition tools (master spec Phase 7).

Every tool that puts a message on the wire - `gmail.send`, `gmail.draft.send`
- is `requires_confirmation = True` (master spec section 39/section 10: AI
email generation must go through Preview -> Edit -> Draft -> Confirmation ->
Send, never auto-send; see Phase 10). Draft create/update/delete are local
to the account (nothing leaves it), so they're not.

Recipients are validated as non-empty and lightly shape-checked (must
contain "@") here; RFC-5322-grade address validation is left to Gmail's own
API, which will reject malformed addresses with a classified `ValidationError`
(see `GmailApiClient._write`).
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.tools.base import Tool, ToolResult

_COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"


def _require_access_token(arguments: dict[str, Any]) -> str:
    access_token = arguments.get("access_token")
    if not access_token:
        raise ValidationError("This operation requires a connected Google account (`access_token`).")
    return access_token


def _validate_recipients(arguments: dict[str, Any], tool_name: str) -> tuple[list[str], list[str] | None, list[str] | None]:
    to = arguments.get("to")
    if not to or not isinstance(to, list) or not all(isinstance(a, str) and "@" in a for a in to):
        raise ValidationError(f"`to` must be a non-empty list of email addresses for {tool_name}.")
    cc = arguments.get("cc")
    bcc = arguments.get("bcc")
    if cc is not None and (not isinstance(cc, list) or not all(isinstance(a, str) for a in cc)):
        raise ValidationError(f"`cc` must be a list of email addresses for {tool_name}.")
    if bcc is not None and (not isinstance(bcc, list) or not all(isinstance(a, str) for a in bcc)):
        raise ValidationError(f"`bcc` must be a list of email addresses for {tool_name}.")
    return to, cc, bcc


_COMPOSE_INPUT_SCHEMA = {
    "to": {"type": "array", "items": "string", "required": True},
    "subject": {"type": "string", "required": True},
    "body_text": {"type": "string", "required": True},
    "cc": {"type": "array", "items": "string", "required": False},
    "bcc": {"type": "array", "items": "string", "required": False},
    "access_token": {"type": "string", "required": True},
}


class CreateDraftTool(Tool):
    name = "gmail.draft.create"
    description = "Create a new Gmail draft (to/subject/body, optional cc/bcc). Nothing is sent."
    input_schema = _COMPOSE_INPUT_SCHEMA
    output_schema = {"draft": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        to, cc, bcc = _validate_recipients(arguments, self.name)
        subject = arguments.get("subject", "")
        body_text = arguments.get("body_text", "")
        access_token = _require_access_token(arguments)
        draft = self._client.create_draft(
            access_token=access_token, to=to, subject=subject, body_text=body_text, cc=cc, bcc=bcc
        )
        return ToolResult(success=True, data={"draft": draft.to_dict()})


class UpdateDraftTool(Tool):
    name = "gmail.draft.update"
    description = "Replace the content of an existing Gmail draft (used by AI rewrite/shorten/expand/tone transformations, Phase 10)."
    input_schema = {"draft_id": {"type": "string", "required": True}, **_COMPOSE_INPUT_SCHEMA}
    output_schema = {"draft": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        draft_id = (arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise ValidationError("`draft_id` is required for gmail.draft.update.")
        to, cc, bcc = _validate_recipients(arguments, self.name)
        subject = arguments.get("subject", "")
        body_text = arguments.get("body_text", "")
        access_token = _require_access_token(arguments)
        draft = self._client.update_draft(
            access_token=access_token, draft_id=draft_id, to=to, subject=subject, body_text=body_text, cc=cc, bcc=bcc
        )
        return ToolResult(success=True, data={"draft": draft.to_dict()})


class DeleteDraftTool(Tool):
    name = "gmail.draft.delete"
    description = "Permanently delete a Gmail draft without sending it."
    input_schema = {"draft_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"deleted": {"type": "boolean"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        draft_id = (arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise ValidationError("`draft_id` is required for gmail.draft.delete.")
        access_token = _require_access_token(arguments)
        self._client.delete_draft(access_token=access_token, draft_id=draft_id)
        return ToolResult(success=True, data={"deleted": True, "draft_id": draft_id})


class ListDraftsTool(Tool):
    name = "gmail.draft.list"
    description = "List the connected account's saved Gmail drafts."
    input_schema = {
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"drafts": {"type": "array", "items": "DraftSummary"}, "next_page_token": {"type": "string", "nullable": True}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = _require_access_token(arguments)
        drafts, next_page_token = self._client.list_drafts(
            access_token=access_token,
            max_results=int(arguments.get("max_results", 25)),
            page_token=arguments.get("page_token"),
        )
        return ToolResult(success=True, data={"drafts": [d.to_dict() for d in drafts], "next_page_token": next_page_token})


class SendDraftTool(Tool):
    name = "gmail.draft.send"
    description = "Send an existing Gmail draft as-is. This delivers a real email - requires confirmation."
    input_schema = {"draft_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"message": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        draft_id = (arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise ValidationError("`draft_id` is required for gmail.draft.send.")
        access_token = _require_access_token(arguments)
        message = self._client.send_draft(access_token=access_token, draft_id=draft_id)
        return ToolResult(success=True, data={"message": message.to_dict()})


class SendMessageTool(Tool):
    name = "gmail.send"
    description = "Compose and immediately send a new email (to/subject/body, optional cc/bcc). Delivers a real email - requires confirmation."
    input_schema = _COMPOSE_INPUT_SCHEMA
    output_schema = {"message": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        to, cc, bcc = _validate_recipients(arguments, self.name)
        subject = arguments.get("subject", "")
        body_text = arguments.get("body_text", "")
        access_token = _require_access_token(arguments)
        message = self._client.send_message(
            access_token=access_token, to=to, subject=subject, body_text=body_text, cc=cc, bcc=bcc
        )
        return ToolResult(success=True, data={"message": message.to_dict()})
