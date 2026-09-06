"""
Anonymous session identity for the Google connection (master spec section
17 - "OAuth tokens must be handled securely").

Nova has no user-account system yet (no login, no DB - see master spec
section 51). So "the user" is, for now, "whoever holds this browser's
session cookie". We use Flask's built-in signed session cookie (HMAC'd
with `SECRET_KEY`, httponly, never readable/forgeable by JS or by an
attacker who doesn't know `SECRET_KEY`) to store nothing but an opaque
random session ID. The actual Google tokens live server-side in the
`InMemoryTokenStore`, keyed by that ID - they never go anywhere near the
client.

When a real user-account system exists (out of scope until it's actually
needed), swap `get_or_create_session_id()`'s cookie-backed ID for the
authenticated user's ID - every call site here already treats the ID as an
opaque string, so nothing downstream needs to change.
"""

from __future__ import annotations

import secrets

from flask import session

_SESSION_KEY = "nova_session_id"


def get_or_create_session_id() -> str:
    sid = session.get(_SESSION_KEY)
    if not sid:
        sid = secrets.token_urlsafe(32)
        session[_SESSION_KEY] = sid
        session.permanent = True
    return sid


def get_session_id_if_present() -> str | None:
    return session.get(_SESSION_KEY)
