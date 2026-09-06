from __future__ import annotations

from pydantic import BaseModel, Field


class AgentCommandRequest(BaseModel):
    # Empty/whitespace allowed at the schema level when `confirm` is set
    # (Phase 9: confirming/cancelling a pending action needs no new
    # command text) - `AgentService.handle_command` enforces non-empty
    # command text for every other call.
    command: str = Field("", max_length=2000)
    # None: a normal new command. True: accept `AgentState.pending_confirmation`.
    # False: decline it.
    confirm: bool | None = None
