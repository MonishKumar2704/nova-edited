"""
Per-request context passed into `AgentOrchestrator.handle()` (Phase 9).

Kept separate from `AgentActionResult`/`AgentState` because it carries
*request*-scoped data (who is asking, do they have a connected Google
account, are they answering a pending confirmation) rather than
conversation-scoped memory. The API layer (`app.api.v1.agent`, the
legacy `/agent` route) builds one of these per call; orchestrators never
touch Flask's `request`/`session` directly, keeping them framework-
agnostic and unit-testable without a request context.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentRequestContext:
    #: opaque per-browser-session identifier (see `app.auth.session`),
    #: used to key `AgentState`/pending confirmations.
    session_id: str
    #: valid Google OAuth access token for this session, if one is
    #: connected - injected into tool arguments that declare an
    #: `access_token` field so the planner never has to ask the LLM for it.
    access_token: str | None = None
    #: None: this is a normal new command. True: the user is accepting
    #: `AgentState.pending_confirmation`. False: the user is declining it.
    confirm: bool | None = None
