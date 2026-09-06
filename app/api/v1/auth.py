"""
`/api/v1/auth/google/...` - Google OAuth connect / callback / status /
disconnect (master spec section 17 and section 45 API DESIGN).

Route handlers stay thin: session-cookie plumbing + request parsing +
serialization only. All actual OAuth logic lives in
`app.services.google_auth_service.GoogleAuthService`.
"""

from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, redirect, request

from app.auth.session import get_or_create_session_id
from app.services.google_auth_service import GoogleAuthService

auth_bp = Blueprint("auth", __name__)


def _service() -> GoogleAuthService:
    return current_app.config["NOVA_GOOGLE_AUTH_SERVICE"]


@auth_bp.route("/google/connect", methods=["GET"])
def google_connect():
    session_id = get_or_create_session_id()
    url = _service().build_connect_url(session_id=session_id)

    # Browsers hitting this link directly expect a redirect straight into
    # Google's consent screen; API/JS clients that want the raw URL first
    # (e.g. to open it in a popup) can pass `?mode=json`.
    if request.args.get("mode") == "json":
        return jsonify({"success": True, "authorization_url": url, "request_id": getattr(g, "request_id", None)})
    return redirect(url)


@auth_bp.route("/google/callback", methods=["GET"])
def google_callback():
    session_id = get_or_create_session_id()
    status = _service().handle_callback(
        session_id=session_id,
        code=request.args.get("code"),
        state=request.args.get("state"),
        error=request.args.get("error"),
    )
    body = status.to_dict()
    body["success"] = True
    body["request_id"] = getattr(g, "request_id", None)
    # A production frontend would typically redirect to a "connected!" UI
    # page here; returning JSON keeps this phase's callback introspectable
    # (curl-able) without requiring a specific frontend route to exist yet.
    return jsonify(body)


@auth_bp.route("/google/status", methods=["GET"])
def google_status():
    session_id = get_or_create_session_id()
    status = _service().get_status(session_id=session_id)
    body = status.to_dict()
    body["success"] = True
    body["request_id"] = getattr(g, "request_id", None)
    return jsonify(body)


@auth_bp.route("/google/disconnect", methods=["POST"])
def google_disconnect():
    session_id = get_or_create_session_id()
    _service().disconnect(session_id=session_id)
    return jsonify({"success": True, "connected": False, "request_id": getattr(g, "request_id", None)})
