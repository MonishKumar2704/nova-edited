"""
Request context foundation.

Assigns a correlation ID to every incoming request so that a single
request can be traced end-to-end through the logs: request -> intent ->
tool selected -> API call -> result -> response (see master spec
section 47).
"""

from __future__ import annotations

import uuid

from flask import Flask, g, request


def register_request_context(app: Flask) -> None:
    @app.before_request
    def _assign_request_id():
        incoming = request.headers.get("X-Request-ID")
        g.request_id = incoming or str(uuid.uuid4())

    @app.after_request
    def _echo_request_id(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        return response
