"""
`ai.email.generate` tool (Phase 5: OLLAMA EMAIL AI).

Turns a natural-language instruction (e.g. "write an email to my
professor asking for an extension") into a subject + body pair using the
configured `LLMProvider` (free local Ollama by default - see
`app.llm.factory`).

This tool only *generates* text - it never creates a Gmail draft or
sends anything. Wiring the generated subject/body into an actual
`gmail.draft.create` call (still requiring explicit user confirmation
before any send) is a separate, later step.
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
    "properties": {
        "subject": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["subject", "body"],
}

_SYSTEM_PROMPT = (
    "You write emails on the user's behalf. Given the user's instruction, write a "
    "clear, appropriately-toned email (default to polite and professional unless "
    "asked otherwise). Respond with ONLY a JSON object with exactly two keys: "
    '"subject" and "body". Do not include any other text, explanation, or markdown.'
)


class GenerateEmailTool(Tool):
    name = "ai.email.generate"
    description = (
        "Generate an email subject and body from a natural-language instruction "
        "(e.g. 'write an email to my professor asking for an extension'), using the "
        "local AI model. Does not create a draft or send anything - the result is "
        "just text for the user to review."
    )
    input_schema = {"instruction": {"type": "string", "required": True}}
    output_schema = {"subject": {"type": "string"}, "body": {"type": "string"}}
    permissions: list[str] = []
    requires_confirmation = False

    def __init__(self, *, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        instruction = (arguments.get("instruction") or "").strip()
        if not instruction:
            raise ValidationError("`instruction` is required for ai.email.generate.")

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=instruction),
        ]
        response = self._llm.generate(messages, response_schema=_RESPONSE_SCHEMA, timeout=30.0)

        subject, body = _parse_email(response.text)
        return ToolResult(success=True, data={"subject": subject, "body": body})


class GenerateEmailDraftTool(Tool):
    """`ai.email.draft`: the full Task 54 pipeline - user request -> Ollama ->
    generated subject/body -> Gmail draft. Reuses `GenerateEmailTool`'s own
    generation logic so there is exactly one place that talks to the LLM
    for email text, then saves the result as a draft via the same
    `GmailApiClient.create_draft` the manual `gmail.draft.create` tool
    uses. Never sends - creating a draft has no confirmation requirement
    (nothing leaves the account), same as `gmail.draft.create`.
    """

    name = "ai.email.draft"
    description = (
        "Generate an email from a natural-language instruction (e.g. 'write an email to "
        "john@example.com asking for an extension') using the local AI model, and save it "
        "as a Gmail draft. Does not send anything."
    )
    input_schema = {
        "instruction": {"type": "string", "required": True},
        "to": {"type": "array", "items": "string", "required": True},
        "cc": {"type": "array", "items": "string", "required": False},
        "bcc": {"type": "array", "items": "string", "required": False},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"draft": {"type": "object"}, "subject": {"type": "string"}, "body": {"type": "string"}}
    permissions = ["https://www.googleapis.com/auth/gmail.compose"]
    requires_confirmation = False

    def __init__(self, *, llm_provider: LLMProvider, gmail_client: GmailApiClient) -> None:
        self._llm = llm_provider
        self._gmail = gmail_client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        instruction = (arguments.get("instruction") or "").strip()
        if not instruction:
            raise ValidationError("`instruction` is required for ai.email.draft.")

        to = arguments.get("to")
        if not to or not isinstance(to, list) or not all(isinstance(a, str) and "@" in a for a in to):
            raise ValidationError("`to` must be a non-empty list of email addresses for ai.email.draft.")

        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("This operation requires a connected Google account (`access_token`).")

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=instruction),
        ]
        response = self._llm.generate(messages, response_schema=_RESPONSE_SCHEMA, timeout=30.0)
        subject, body = _parse_email(response.text)

        draft = self._gmail.create_draft(
            access_token=access_token,
            to=to,
            subject=subject,
            body_text=body,
            cc=arguments.get("cc"),
            bcc=arguments.get("bcc"),
        )
        return ToolResult(success=True, data={"draft": draft.to_dict(), "subject": subject, "body": body})


def _parse_email(text: str) -> tuple[str, str]:
    """Parse the model's response into (subject, body). See `app.tools.ai._parsing`."""
    fields = parse_json_fields(text, ["subject", "body"], context="email")
    return fields["subject"], fields["body"]
