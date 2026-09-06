"""
`ai.email.rewrite` tool (Phase 5: OLLAMA EMAIL AI, Task 55).

Takes existing email text supplied by the user and rewrites it per a
requested style - "make this more professional", "shorten it", "make it
friendlier/more polite", or a general "improve this" - using the same
`LLMProvider` (free local Ollama by default) as `ai.email.generate`.

Purely a text transformation: this tool never sends, drafts, or reads
anything from Gmail - it only takes text in and returns text out, so it
works even on email text the user typed themselves rather than an actual
Gmail message.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.llm.base import LLMMessage, LLMProvider
from app.tools.ai._parsing import parse_json_fields
from app.tools.base import Tool, ToolResult

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

_SYSTEM_PROMPT = (
    "You rewrite emails. You will be given the ORIGINAL EMAIL and a requested STYLE "
    "(e.g. more professional, shorter, more polite, more friendly, or a general "
    "improvement). Rewrite the email to fit that style while preserving the sender's "
    "original meaning and intent - do not add facts, requests, or claims that were not "
    "in the original. Respond with ONLY a JSON object with exactly one key: \"text\", "
    "containing the rewritten email. No other text, explanation, or markdown."
)

_DEFAULT_STYLE = "improve"


class RewriteEmailTool(Tool):
    name = "ai.email.rewrite"
    description = (
        "Rewrite email text the user provides, per a requested style - e.g. 'rewrite', "
        "'improve', 'more professional', 'shorter', 'more polite', or 'more friendly'. "
        "Takes plain text in, returns plain text out - does not send, draft, or touch Gmail."
    )
    input_schema = {
        "text": {"type": "string", "required": True},
        "style": {
            "type": "string",
            "required": False,
            "description": f"e.g. 'professional', 'shorter', 'polite', 'friendly'. Defaults to '{_DEFAULT_STYLE}'.",
        },
    }
    output_schema = {"text": {"type": "string"}}
    permissions: list[str] = []
    requires_confirmation = False

    def __init__(self, *, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        text = (arguments.get("text") or "").strip()
        if not text:
            raise ValidationError("`text` is required for ai.email.rewrite.")
        style = (arguments.get("style") or _DEFAULT_STYLE).strip() or _DEFAULT_STYLE

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"STYLE: {style}\n\nORIGINAL EMAIL:\n{text}"),
        ]
        response = self._llm.generate(messages, response_schema=_RESPONSE_SCHEMA, timeout=30.0)

        rewritten = _parse_text(response.text)
        return ToolResult(success=True, data={"text": rewritten, "style": style})


def _parse_text(text: str) -> str:
    """Parse the model's response into the rewritten email text. See `app.tools.ai._parsing`."""
    return parse_json_fields(text, ["text"], context="rewritten text")["text"]
