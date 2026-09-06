"""
AI tools package (Phase 5: OLLAMA EMAIL AI).

Mirrors `app.tools.gmail`/`app.tools.youtube`: one `register_ai_tools()`
entry point the app factory calls with its already-built `LLMProvider`,
keeping the orchestrator dependent only on the tool registry (master
spec section 7) rather than importing concrete AI tools directly.
"""

from __future__ import annotations

from app.integrations.gmail_api import GmailApiClient
from app.llm.base import LLMProvider
from app.tools.ai.email_generation import GenerateEmailDraftTool, GenerateEmailTool
from app.tools.ai.grammar import CorrectEmailGrammarTool
from app.tools.ai.reply import SuggestReplyTool
from app.tools.ai.rewrite import RewriteEmailTool
from app.tools.registry import ToolRegistry


def register_ai_tools(registry: ToolRegistry, *, llm_provider: LLMProvider, gmail_client: GmailApiClient) -> None:
    registry.register(GenerateEmailTool(llm_provider=llm_provider))
    registry.register(GenerateEmailDraftTool(llm_provider=llm_provider, gmail_client=gmail_client))
    registry.register(RewriteEmailTool(llm_provider=llm_provider))
    registry.register(CorrectEmailGrammarTool(llm_provider=llm_provider))
    registry.register(SuggestReplyTool(llm_provider=llm_provider, gmail_client=gmail_client))
