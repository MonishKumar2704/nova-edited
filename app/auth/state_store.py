"""
Short-lived CSRF `state` storage for the OAuth authorization-code flow
(master spec section 17 / section 15 SECURITY - CSRF).

Google's OAuth flow round-trips an opaque `state` value through the
user's browser. We generate it, remember which session it belongs to, and
require an exact match (single use, time-limited) on callback. This
prevents an attacker from tricking a victim into completing an OAuth flow
that gets bound to the attacker's session (a login-CSRF style attack).
"""

from __future__ import annotations

import threading
import time

_DEFAULT_TTL_SECONDS = 600  # 10 minutes - generous enough for a user to complete consent


class OAuthStateStore:
    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._states: dict[str, tuple[str, float]] = {}  # state -> (session_id, expires_at)

    def issue(self, state: str, session_id: str) -> None:
        with self._lock:
            self._sweep_locked()
            self._states[state] = (session_id, time.time() + self._ttl)

    def consume(self, state: str) -> str | None:
        """Validate and single-use consume a state token. Returns the bound session_id, or None if invalid/expired/already used."""
        with self._lock:
            entry = self._states.pop(state, None)
        if entry is None:
            return None
        session_id, expires_at = entry
        if time.time() >= expires_at:
            return None
        return session_id

    def _sweep_locked(self) -> None:
        now = time.time()
        expired = [s for s, (_, exp) in self._states.items() if now >= exp]
        for s in expired:
            self._states.pop(s, None)
