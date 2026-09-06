"""
`/api/v1/auth/google/...` - Google OAuth connect / callback / status /
disconnect (master spec section 17 and section 45 API DESIGN).

Route handlers stay thin: session-cookie plumbing + request parsing +
serialization only. All actual OAuth logic lives in
`app.services.google_auth_service.GoogleAuthService`.
"""

from __future__ import annotations

from urllib.parse import quote

from flask import Blueprint, current_app, g, jsonify, redirect, request

from app.auth.session import get_or_create_session_id
from app.core.errors import NovaError
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
    """Google redirects the user's browser here directly (this *is* the
    registered `GOOGLE_REDIRECT_URI`) - there is no frontend JS involved
    in this hop, so this route itself is what has to land the user back
    on the NOVA UI instead of a bare JSON page.

    Default behavior (a real browser landing here after consenting/
    denying) redirects to `/` with a `google=connected` or
    `google=error&reason=...` query flag that `index.html` reads once on
    load to show a toast and refresh the connection-status widget - never
    to mail.google.com, and never via `window.open`/a new tab, just an
    ordinary same-tab server redirect closing the OAuth round trip.

    `?mode=json` (same opt-in already used by `/google/connect`) preserves
    the original introspectable JSON response for tests/tooling.
    """
    session_id = get_or_create_session_id()
    want_json = request.args.get("mode") == "json"

    try:
        status = _service().handle_callback(
            session_id=session_id,
            code=request.args.get("code"),
            state=request.args.get("state"),
            error=request.args.get("error"),
        )
    except NovaError as exc:
        if want_json:
            raise
        return redirect(f"/?google=error&reason={quote(exc.message)}")

    if want_json:
        body = status.to_dict()
        body["success"] = True
        body["request_id"] = getattr(g, "request_id", None)
        return jsonify(body)
    return redirect("/?google=connected")


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
