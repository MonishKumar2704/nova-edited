"""Shared plumbing for `app/api/v1/gmail.py` and `app/api/v1/youtube.py`.

Both route modules need to (a) reach the shared `ToolRegistry` off the
Flask app config, and (b) resolve an OAuth access token for the current
session before calling a tool. That logic used to be copy-pasted
byte-for-byte between the two files (`_registry()` identical in both;
`_required_access_token()` differing only in whether a missing token has
an API-key fallback). Task 10 (section 2.1) flagged this as safe,
genuine duplication; this module is the consolidation, done in Task 13.

No behavior changes: every call site keeps the exact same inputs,
outputs, and error text it had before.
"""

from __future__ import annotations

from flask import current_app

from app.auth.session import get_session_id_if_present
from app.core.errors import AuthenticationError
from app.tools.registry import ToolRegistry


def registry() -> ToolRegistry:
    """The shared tool registry, wired onto the app in the app factory."""
    return current_app.config["NOVA_TOOL_REGISTRY"]


def resolve_access_token() -> str | None:
    """Best-effort OAuth access token for the current session, or None.

    This is the shared core behind both call patterns used by the route
    modules:
    - YouTube's `_optional_access_token()` returns this directly (no
      session, or an invalid one, just means "no account connected" -
      several YouTube reads fall back to the public API key instead).
    - Both `require_access_token()` below and Gmail's stricter
      requirement build on this same lookup; they differ only in what
      happens when it comes back empty.
    """
    session_id = get_session_id_if_present()
    if not session_id:
        return None
    google_auth_service = current_app.config["NOVA_GOOGLE_AUTH_SERVICE"]
    try:
        return google_auth_service.get_valid_access_token(session_id=session_id)
    except AuthenticationError:
        return None


def require_access_token(message: str) -> str:
    """Access token for the current session, or a clear `AuthenticationError`.

    `message` is supplied by the caller so each route module keeps its
    own existing error text (Gmail's mentions Gmail access explicitly;
    YouTube's is a generic "connect an account" message) - consolidating
    the *lookup* logic here does not mean the two callers have to say
    the same thing when it fails.
    """
    token = resolve_access_token()
    if not token:
        raise AuthenticationError(message)
    return token
