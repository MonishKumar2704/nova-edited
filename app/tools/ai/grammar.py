"""
`ai.email.grammar_correct` tool (Phase 5: OLLAMA EMAIL AI, Task 56).

Takes existing email text supplied by the user and returns a
grammar-corrected version - fixing spelling, punctuation, and grammar
mistakes only - using the same `LLMProvider` (free local Ollama by
default) as `ai.email.generate` / `ai.email.rewrite`.

Purely a text transformation, same shape as `ai.email.rewrite`: this
tool never sends, drafts, or reads anything from Gmail - it only takes
text in and returns text out. Unlike `ai.email.rewrite` it must NOT
change tone, style, or length - only fix mistakes - so it gets its own
tool and system prompt rather than being folded into `rewrite` as a
"style".
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
    "You correct grammar, spelling, and punctuation in emails. You will be given an "
    "EMAIL. Fix only grammar, spelling, and punctuation mistakes - do not change the "
    "tone, wording, structure, length, or meaning beyond what is required to correct "
    "an actual mistake, and do not add facts, requests, or claims that were not in the "
    "original. If the email already has no mistakes, return it unchanged. Respond with "
    'ONLY a JSON object with exactly one key: "text", containing the corrected email. '
    "No other text, explanation, or markdown."
)


class CorrectEmailGrammarTool(Tool):
    name = "ai.email.grammar_correct"
    description = (
        "Correct grammar, spelling, and punctuation mistakes in email text the user "
        "provides. Does not change tone or style (use ai.email.rewrite for that) and "
        "does not send, draft, or touch Gmail - takes plain text in, returns plain "
        "text out."
    )
    input_schema = {"text": {"type": "string", "required": True}}
    output_schema = {"text": {"type": "string"}}
    permissions: list[str] = []
    requires_confirmation = False

    def __init__(self, *, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        text = (arguments.get("text") or "").strip()
        if not text:
            raise ValidationError("`text` is required for ai.email.grammar_correct.")

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"EMAIL:\n{text}"),
        ]
        response = self._llm.generate(messages, response_schema=_RESPONSE_SCHEMA, timeout=30.0)

        corrected = _parse_text(response.text)
        return ToolResult(success=True, data={"text": corrected})


def _parse_text(text: str) -> str:
    """Parse the model's response into the corrected email text. See `app.tools.ai._parsing`."""
    return parse_json_fields(text, ["text"], context="corrected text")["text"]
