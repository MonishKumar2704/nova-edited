"""
Agent application service.

This is the one place both the legacy `/agent` endpoint and the new
`/api/v1/agent/command` endpoint call into, so command-handling logic
lives in exactly one place instead of being duplicated across routes.
"""

from __future__ import annotations

from app.agent.context import AgentRequestContext
from app.agent.orchestrator import AgentActionResult, AgentOrchestrator
from app.core.errors import ValidationError


class AgentService:
    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self._orchestrator = orchestrator

    def handle_command(
        self, raw_command: str | None, *, context: AgentRequestContext | None = None
    ) -> AgentActionResult:
        # A confirm/cancel reply to a pending action (Phase 9) may
        # legitimately carry no new command text - the command was
        # already given (and stashed) on the turn that asked for
        # confirmation. Any other call still requires real command text.
        answering_confirmation = context is not None and context.confirm is not None
        if not answering_confirmation and (not raw_command or not raw_command.strip()):
            raise ValidationError("`command` must be a non-empty string.")
        return self._orchestrator.handle(raw_command or "", context=context)
