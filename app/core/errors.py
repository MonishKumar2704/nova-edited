"""
Error handling foundation (see master spec section 11).

Every external / agent / tool operation should raise one of the
`NovaError` subclasses below rather than letting a bare exception
propagate or swallowing it silently. The Flask error handler registered
in `register_error_handlers` converts these into consistent, classified
JSON responses and logs the technical detail server-side without leaking
secrets to the client.
"""

from __future__ import annotations

import logging
import uuid

from flask import Flask, g, jsonify

logger = logging.getLogger(__name__)


class NovaError(Exception):
    """Base class for all classified, expected Nova errors."""

    code = "API_ERROR"
    http_status = 500

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(NovaError):
    code = "VALIDATION_ERROR"
    http_status = 400


class AuthenticationError(NovaError):
    code = "AUTHENTICATION_ERROR"
    http_status = 401


class AuthorizationError(NovaError):
    code = "AUTHORIZATION_ERROR"
    http_status = 403


class RateLimitError(NovaError):
    code = "RATE_LIMIT_ERROR"
    http_status = 429


class NetworkError(NovaError):
    code = "NETWORK_ERROR"
    http_status = 502


class TimeoutErrorNova(NovaError):
    code = "TIMEOUT_ERROR"
    http_status = 504


class ApiError(NovaError):
    code = "API_ERROR"
    http_status = 502


class LLMError(NovaError):
    code = "LLM_ERROR"
    http_status = 503


class ToolError(NovaError):
    code = "TOOL_ERROR"
    http_status = 500


class UserConfirmationRequiredError(NovaError):
    code = "USER_CONFIRMATION_REQUIRED"
    http_status = 409


class NotFoundError(NovaError):
    code = "NOT_FOUND"
    http_status = 404


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(NovaError)
    def handle_nova_error(err: NovaError):
        request_id = getattr(g, "request_id", str(uuid.uuid4()))
        logger.warning("Handled %s: %s", err.code, err.message, extra={"details": err.details})
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": err.code,
                        "message": err.message,
                        "request_id": request_id,
                    },
                }
            ),
            err.http_status,
        )

    @app.errorhandler(404)
    def handle_not_found(_err):
        request_id = getattr(g, "request_id", str(uuid.uuid4()))
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "The requested resource was not found.",
                        "request_id": request_id,
                    },
                }
            ),
            404,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception):
        request_id = getattr(g, "request_id", str(uuid.uuid4()))
        # Log full technical detail server-side; never leak internals to the client.
        logger.exception("Unhandled exception (request_id=%s)", request_id)
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "API_ERROR",
                        "message": "An unexpected error occurred.",
                        "request_id": request_id,
                    },
                }
            ),
            500,
        )
