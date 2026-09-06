"""
`ai.email.suggest_reply` tool (Phase 5: OLLAMA EMAIL AI, Task 57).

Implements the pipeline:

    Existing Gmail email -> Ollama -> Suggested reply -> Gmail draft

Fetches the original message (for its sender, subject, body, and
threading headers - same lookup `gmail.reply` does), asks the configured
`LLMProvider` (free local Ollama by default) to write a reply body, and
saves that reply as a Gmail draft in the same thread via
`GmailApiClient.create_draft` - reusing the exact `to`/subject/threading
logic `gmail.reply` uses to send, but drafting instead of sending.

Like `ai.email.draft` (Task 54), creating a draft never sends anything,
so this tool itself does not require confirmation - actually sending the
suggested reply still goes through the existing `gmail.draft.send` /
`gmail.send` confirmation gate (Task 59: never auto-send an AI-generated
email).
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.gmail_api import GmailApiClient
from app.llm.base import LLMMessage, LLMProvider
from app.tools.ai._parsing import parse_json_fields
from app.tools.base import Tool, ToolResult

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"body": {"type": "string"}},
    "required": ["body"],
}

_SYSTEM_PROMPT = (
    "You write reply emails on the user's behalf. You will be given the ORIGINAL EMAIL "
    "(sender, subject, and body) and, optionally, an INSTRUCTION describing what the "
    "reply should say. Write a clear, appropriately-toned reply body (default to polite "
    "and professional unless asked otherwise). If no instruction is given, write a "
    "reasonable, brief acknowledgement/response to the original email's content. Do not "
    "invent facts, commitments, or details that are not implied by the original email or "
    "the instruction. Respond with ONLY a JSON object with exactly one key: \"body\", "
    "containing the reply text. Do not include a subject line or quoted original message "
    "- just the reply body. No other text, explanation, or markdown."
)


_DEFAULT_SUBJECT = "Re:"


def _reply_subject(original_subject: str) -> str:
    """Mirror `gmail.reply`'s subject convention (\"Re: ...\", not doubled up)."""
    stripped = original_subject.strip()
    return stripped if stripped.lower().startswith("re:") else f"Re: {stripped}" if stripped else _DEFAULT_SUBJECT


class SuggestReplyTool(Tool):
    name = "ai.email.suggest_reply"
    description = (
        "Read an existing Gmail message and generate a suggested reply using the local "
        "AI model (optionally guided by an instruction, e.g. 'say I can't make it and "
        "suggest next week'), then save that reply as a Gmail draft in the same thread. "
        "Does not send anything."
    )
    input_schema = {
        "message_id": {"type": "string", "required": True},
        "instruction": {
            "type": "string",
            "required": False,
            "description": "Optional guidance for what the reply should say, e.g. 'accept the invitation'.",
        },
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"draft": {"type": "object"}, "subject": {"type": "string"}, "body": {"type": "string"}}
    permissions = ["https://www.googleapis.com/auth/gmail.compose"]
    requires_confirmation = False

    def __init__(self, *, llm_provider: LLMProvider, gmail_client: GmailApiClient) -> None:
        self._llm = llm_provider
        self._gmail = gmail_client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message_id = (arguments.get("message_id") or "").strip()
        if not message_id:
            raise ValidationError("`message_id` is required for ai.email.suggest_reply.")
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("This operation requires a connected Google account (`access_token`).")

        original = self._gmail.get_message(access_token=access_token, message_id=message_id, format="full")
        if not original.from_:
            raise ValidationError("Could not determine the sender of the original message to reply to.")

        instruction = (arguments.get("instruction") or "").strip()
        user_content = (
            f"ORIGINAL EMAIL\nFrom: {original.from_}\nSubject: {original.subject}\n\n{original.body_text}"
        )
        if instruction:
            user_content += f"\n\nINSTRUCTION: {instruction}"

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]
        response = self._llm.generate(messages, response_schema=_RESPONSE_SCHEMA, timeout=30.0)
        body = _parse_body(response.text)

        subject = _reply_subject(original.subject)
        draft = self._gmail.create_draft(
            access_token=access_token,
            to=[original.from_],
            subject=subject,
            body_text=body,
            thread_id=original.thread_id,
            in_reply_to=original.rfc_message_id or None,
            references=original.rfc_message_id or None,
        )
        return ToolResult(success=True, data={"draft": draft.to_dict(), "subject": subject, "body": body})


def _parse_body(text: str) -> str:
    """Parse the model's response into the reply body text. See `app.tools.ai._parsing`."""
    return parse_json_fields(text, ["body"], context="reply")["body"]
