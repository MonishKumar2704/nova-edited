"""
Shared response parsing for the `ai.email.*` tools (Phase 5: OLLAMA EMAIL AI).

Every tool in this package asks the LLM for a small JSON object (one or
two string fields) and needs the exact same tolerant parsing: `response_schema`
makes clean JSON the common case, but a local model can still wrap its
answer in prose or a code fence, so this falls back to extracting the
first `{...}` block before giving up. This one function replaces four
copies of that logic that used to live separately in `email_generation.py`,
`rewrite.py`, `grammar.py`, and `reply.py` (Task 64: simplify backend
structure - duplicate helpers).
"""

from __future__ import annotations

import json

from app.core.errors import LLMError


def parse_json_fields(text: str, fields: list[str], *, context: str) -> dict[str, str]:
    """Parse `text` as a JSON object and return `fields` as trimmed strings.

    `context` is a short noun phrase (e.g. "email", "corrected text",
    "reply") used to build a plain-English error message if parsing or
    validation fails. Raises `LLMError` if no JSON object can be found,
    the JSON is not an object, or any requested field is missing/empty.
    """
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMError(f"The AI did not return a usable {context} (no JSON found in its response).")
        try:
            data = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise LLMError(f"The AI's response could not be parsed as a {context}.") from exc

    if not isinstance(data, dict):
        raise LLMError("The AI's response was not a JSON object.")

    result: dict[str, str] = {}
    for field in fields:
        value = str(data.get(field) or "").strip()
        if not value:
            raise LLMError(f"The AI's response was missing a usable '{field}' for the {context}.")
        result[field] = value
    return result
