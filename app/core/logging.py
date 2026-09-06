"""
Structured logging foundation.

Design goals (see master spec section 47 - OBSERVABILITY):
  * every log line can be tied back to a request via a correlation/request ID
  * never log secrets (passwords, OAuth tokens, API keys, full email bodies)
  * output is easy to read locally and easy to parse in production
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from flask import g, has_request_context

_REDACT_KEYS = {"password", "token", "access_token", "refresh_token", "api_key", "secret"}


class RequestIdFilter(logging.Filter):
    """Injects the current request's correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(g, "request_id", "-") if has_request_context() else "-"
        return True


def redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of `data` with sensitive keys masked.

    Use this before logging any dict that may contain user- or
    integration-supplied fields (headers, request bodies, tokens, etc).
    """
    safe = {}
    for key, value in data.items():
        if key.lower() in _REDACT_KEYS:
            safe[key] = "***REDACTED***"
        else:
            safe[key] = value
    return safe


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once, idempotently, for the whole app."""
    root = logging.getLogger()
    if getattr(root, "_nova_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | req=%(request_id)s | %(name)s | %(message)s",
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    root._nova_configured = True  # type: ignore[attr-defined]
