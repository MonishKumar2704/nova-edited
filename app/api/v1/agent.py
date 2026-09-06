from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, request
from pydantic import ValidationError as PydanticValidationError

from app.agent.context import AgentRequestContext
from app.agent.orchestrator import AgentActionResult
from app.auth.session import get_or_create_session_id
from app.core.errors import AuthenticationError, ValidationError
from app.schemas.agent import AgentCommandRequest

agent_bp = Blueprint("agent", __name__)


def serialize_agent_result(result: AgentActionResult) -> dict:
    return {
        "success": result.success,
        "message": result.message,
        "url": result.url,
        "action_type": result.action_type,
        "data": result.data,
        "actions": result.actions,
    }


def _parse_command(payload: dict) -> AgentCommandRequest:
    try:
        return AgentCommandRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid request body.", details={"errors": exc.errors()}) from exc


def build_agent_context(confirm: bool | None) -> AgentRequestContext:
    """Resolve the per-request `AgentRequestContext` (Phase 9).

    Mirrors `app.api.v1.youtube._optional_access_token()`: a session
    without a connected Google account can still use the agent for
    anything that doesn't need one (public YouTube search, etc) - tools
    that do need `access_token` raise a classified error the planner
    surfaces back to the user instead of crashing.
    """
    session_id = get_or_create_session_id()
    access_token = None
    google_auth_service = current_app.config.get("NOVA_GOOGLE_AUTH_SERVICE")
    if google_auth_service is not None:
        try:
            access_token = google_auth_service.get_valid_access_token(session_id=session_id)
        except AuthenticationError:
            access_token = None
    return AgentRequestContext(session_id=session_id, access_token=access_token, confirm=confirm)


@agent_bp.route("/command", methods=["POST"])
def run_command():
    payload = request.get_json(silent=True) or {}
    req = _parse_command(payload)
    context = build_agent_context(req.confirm)

    agent_service = current_app.config["NOVA_AGENT_SERVICE"]
    result = agent_service.handle_command(req.command, context=context)

    return jsonify({**serialize_agent_result(result), "request_id": getattr(g, "request_id", None)})
