"""
`gmail.reply` / `gmail.reply_all` / `gmail.forward` (master spec Phase 7:
conversations).

Each of these fetches the original message first (for its `thread_id`,
`rfc_message_id`, subject, and participants) and then sends through
`GmailApiClient.send_message` with the right `threadId`/`In-Reply-To`/
`References` so Gmail (and any other RFC-5322-aware client) threads the
reply/forward correctly instead of starting a new conversation.

`body_text` is the caller-supplied reply/forward content (in Phase 10 this
is AI-generated and shown to the user for edit/confirmation before this
tool ever runs) - these tools never invent message content themselves.
All three send real email, so `requires_confirmation = True`.
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


def _reply_subject(original_subject: str) -> str:
    stripped = original_subject.strip()
    return stripped if stripped.lower().startswith("re:") else f"Re: {stripped}" if stripped else "Re:"


def _forward_subject(original_subject: str) -> str:
    stripped = original_subject.strip()
    return stripped if stripped.lower().startswith("fwd:") else f"Fwd: {stripped}" if stripped else "Fwd:"


def _quote_original(original) -> str:
    return f"\n\n---------- Forwarded/original message ----------\nFrom: {original.from_}\nDate: {original.date}\nSubject: {original.subject}\n\n{original.body_text}"


class ReplyTool(Tool):
    name = "gmail.reply"
    description = "Reply to the sender of a Gmail message. Sends a real email - requires confirmation."
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "body_text": {"type": "string", "required": True},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"message": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = (arguments.get("message_id") or "").strip()
        body_text = arguments.get("body_text") or ""
        if not message_id or not body_text.strip():
            raise ValidationError("`message_id` and `body_text` are required for gmail.reply.")
        access_token = _require_access_token(arguments)

        original = self._client.get_message(access_token=access_token, message_id=message_id, format="full")
        if not original.from_:
            raise ValidationError("Could not determine the sender of the original message to reply to.")

        sent = self._client.send_message(
            access_token=access_token,
            to=[original.from_],
            subject=_reply_subject(original.subject),
            body_text=body_text,
            thread_id=original.thread_id,
            in_reply_to=original.rfc_message_id or None,
            references=original.rfc_message_id or None,
        )
        return ToolResult(success=True, data={"message": sent.to_dict()})


class ReplyAllTool(Tool):
    name = "gmail.reply_all"
    description = "Reply to the sender and all other recipients of a Gmail message. Sends a real email - requires confirmation."
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "body_text": {"type": "string", "required": True},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"message": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = (arguments.get("message_id") or "").strip()
        body_text = arguments.get("body_text") or ""
        if not message_id or not body_text.strip():
            raise ValidationError("`message_id` and `body_text` are required for gmail.reply_all.")
        access_token = _require_access_token(arguments)

        original = self._client.get_message(access_token=access_token, message_id=message_id, format="full")
        if not original.from_:
            raise ValidationError("Could not determine the sender of the original message to reply to.")

        # Best-effort "all": original sender as `to`, everyone else who was
        # on the original To/Cc as `cc`. Without the connected account's own
        # address available here, this cannot exclude "myself" from that
        # list - callers that need strict RFC reply-all semantics should
        # filter `cc` client-side before this runs.
        cc_candidates = [addr.strip() for addr in f"{original.to},{arguments.get('extra_cc', '')}".split(",") if addr.strip()]

        sent = self._client.send_message(
            access_token=access_token,
            to=[original.from_],
            cc=cc_candidates or None,
            subject=_reply_subject(original.subject),
            body_text=body_text,
            thread_id=original.thread_id,
            in_reply_to=original.rfc_message_id or None,
            references=original.rfc_message_id or None,
        )
        return ToolResult(success=True, data={"message": sent.to_dict()})


class ForwardTool(Tool):
    name = "gmail.forward"
    description = "Forward a Gmail message to new recipients, with an optional intro note. Sends a real email - requires confirmation."
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "to": {"type": "array", "items": "string", "required": True},
        "body_text": {"type": "string", "required": False, "default": ""},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"message": {"type": "object"}}
    permissions = [_COMPOSE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: GmailApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = (arguments.get("message_id") or "").strip()
        to = arguments.get("to")
        if not message_id or not to or not isinstance(to, list) or not all(isinstance(a, str) and "@" in a for a in to):
            raise ValidationError("`message_id` and a non-empty `to` list of email addresses are required for gmail.forward.")
        access_token = _require_access_token(arguments)

        original = self._client.get_message(access_token=access_token, message_id=message_id, format="full")
        intro = (arguments.get("body_text") or "").strip()
        full_body = f"{intro}{_quote_original(original)}" if intro else _quote_original(original).lstrip()

        sent = self._client.send_message(
            access_token=access_token,
            to=to,
            subject=_forward_subject(original.subject),
            body_text=full_body,
        )
        return ToolResult(success=True, data={"message": sent.to_dict()})
