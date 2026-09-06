"""
Per-session agent state (Phase 9: REAL AGENT ORCHESTRATOR, master spec
section 9).

Nova has no user-account system (see `app.auth.session`), so "the
conversation" is scoped to the same anonymous, cookie-backed session ID
already used for the Google OAuth connection. `AgentState` holds exactly
the structured fields the master spec calls for:

    current_youtube_video
    current_search_results
    current_playlist
    current_email
    current_thread
    current_draft
    pending_confirmation

plus a short, bounded conversation history so the planner
(`app.agent.planner_orchestrator`) can resolve references like "play the
third result", "reply to that", or "make it shorter" against what
actually happened earlier in the conversation instead of guessing.

`AgentStateStore` is the same shape as
`app.auth.token_store.InMemoryTokenStore` (in-memory, thread-safe, keyed
by session ID) for the same documented reason: no database exists yet
(master spec section 51), and state lost on process restart / not shared
across workers is an acceptable, explicit limitation at this stage.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

# How many conversation turns (one turn = one user message + one final
# assistant reply) are kept for planner context. Bounded on purpose - an
# unbounded history would grow every LLM prompt (and therefore cost/
# latency) without limit over a long-lived session.
MAX_HISTORY_TURNS = 6


@dataclass
class PendingConfirmation:
    """A tool call the planner selected but held for explicit user consent.

    Set only when the selected tool has `Tool.requires_confirmation =
    True` (master spec section 9/39: "Do not guess missing critical
    information" extends to "do not perform sensitive actions without
    asking"). `access_token` is deliberately NOT stored here - a fresh
    one is pulled from the confirming request's own context, so a stale/
    revoked token captured at planning time can't be replayed later.
    """

    tool_name: str
    arguments: dict[str, Any]
    description: str
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """Structured conversation memory for one session (master spec section 9)."""

    current_youtube_video: dict[str, Any] | None = None
    current_search_results: list[dict[str, Any]] = field(default_factory=list)
    current_playlist: dict[str, Any] | None = None
    current_email: dict[str, Any] | None = None
    current_thread: dict[str, Any] | None = None
    current_draft: dict[str, Any] | None = None
    pending_confirmation: PendingConfirmation | None = None
    # [{"role": "user"|"assistant", "content": str}, ...], oldest first,
    # trimmed to MAX_HISTORY_TURNS*2 entries (one user + one assistant
    # message per turn).
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    # Guards mutation of this specific session's state against concurrent
    # requests (e.g. a double-submit) - a single global lock across all
    # sessions would serialize unrelated users' agent calls, which the
    # concurrency architecture (Phase 1/16) explicitly avoids.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})
        max_entries = MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_entries:
            self.conversation_history = self.conversation_history[-max_entries:]

    def summarize_for_prompt(self) -> str:
        """Compact, LLM-readable snapshot of "what we were just doing".

        Deliberately terse (ids/titles/subjects only, never full email
        bodies or long descriptions) to keep the prompt small - the LLM
        can always call a `*.get_*` tool for full detail if it needs it.
        """
        lines: list[str] = []

        if self.current_youtube_video:
            v = self.current_youtube_video
            lines.append(f"- current_youtube_video: id={v.get('video_id')!r} title={v.get('title')!r}")

        if self.current_search_results:
            preview = ", ".join(
                f"[{i + 1}] {r.get('title')!r} (video_id={r.get('video_id')!r})"
                for i, r in enumerate(self.current_search_results[:10])
            )
            lines.append(f"- current_search_results: {preview}")

        if self.current_playlist:
            p = self.current_playlist
            lines.append(f"- current_playlist: id={p.get('playlist_id')!r} title={p.get('title')!r}")

        if self.current_email:
            e = self.current_email
            lines.append(
                f"- current_email: id={e.get('message_id')!r} subject={e.get('subject')!r} from={e.get('from')!r}"
            )

        if self.current_thread:
            t = self.current_thread
            lines.append(f"- current_thread: id={t.get('thread_id')!r}")

        if self.current_draft:
            d = self.current_draft
            msg = d.get("message") or {}
            lines.append(f"- current_draft: id={d.get('draft_id')!r} subject={msg.get('subject')!r}")

        if self.pending_confirmation:
            lines.append(f"- pending_confirmation: {self.pending_confirmation.description!r} (awaiting user reply)")

        if not lines:
            return "(no prior context in this session yet)"
        return "\n".join(lines)


class AgentStateStore:
    """Thread-safe, in-memory, per-session `AgentState` store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, AgentState] = {}

    def get_or_create(self, session_id: str) -> AgentState:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = AgentState()
                self._states[session_id] = state
            return state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)
