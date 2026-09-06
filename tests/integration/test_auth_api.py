from __future__ import annotations

import time
from unittest.mock import Mock

from app.auth.state_store import OAuthStateStore
from app.auth.token_store import InMemoryTokenStore
from app.integrations.google_oauth import ExchangedToken, GoogleIdentity
from app.services.google_auth_service import GoogleAuthService


def test_status_disconnected_by_default(client):
    resp = client.get("/api/v1/auth/google/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["connected"] is False


def test_connect_returns_validation_error_when_not_configured(client):
    # Default test app has no GOOGLE_CLIENT_ID/SECRET set (see .env.example / Config defaults).
    resp = client.get("/api/v1/auth/google/connect?mode=json")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_disconnect_is_a_no_op_when_never_connected(client):
    resp = client.post("/api/v1/auth/google/disconnect")
    assert resp.status_code == 200
    assert resp.get_json()["connected"] is False


def _install_configured_service(app) -> Mock:
    """Swap in a GoogleAuthService whose oauth_client is mocked but reports as configured."""
    oauth_client = Mock()
    oauth_client.is_configured.return_value = True
    oauth_client.build_authorization_url.return_value = "https://accounts.google.com/o/oauth2/v2/auth?mock=1"
    service = GoogleAuthService(
        oauth_client=oauth_client,
        token_store=InMemoryTokenStore(encryption_key=None),
        state_store=OAuthStateStore(),
        default_scopes=["openid", "email"],
    )
    app.config["NOVA_GOOGLE_AUTH_SERVICE"] = service
    return oauth_client


def test_connect_redirects_to_google_when_configured(app, client):
    _install_configured_service(app)
    resp = client.get("/api/v1/auth/google/connect")
    assert resp.status_code == 302
    assert resp.headers["Location"].startswith("https://accounts.google.com/")


def test_connect_json_mode_returns_url_when_configured(app, client):
    _install_configured_service(app)
    resp = client.get("/api/v1/auth/google/connect?mode=json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["authorization_url"].startswith("https://accounts.google.com/")


def test_full_connect_callback_status_disconnect_flow(app, client):
    oauth_client = _install_configured_service(app)

    connect_resp = client.get("/api/v1/auth/google/connect?mode=json")
    auth_url = connect_resp.get_json()["authorization_url"]
    assert auth_url  # sanity

    # Pull the state the service actually issued.
    _, kwargs = oauth_client.build_authorization_url.call_args
    state = kwargs["state"]

    oauth_client.exchange_code.return_value = ExchangedToken(
        access_token="at", refresh_token="rt", expires_at=time.time() + 3600, scopes=("openid", "email")
    )
    oauth_client.get_identity.return_value = GoogleIdentity(sub="sub-1", email="user@example.com")

    callback_resp = client.get(f"/api/v1/auth/google/callback?code=abc&state={state}")
    assert callback_resp.status_code == 200
    callback_body = callback_resp.get_json()
    assert callback_body["connected"] is True
    assert callback_body["google_email"] == "user@example.com"

    status_resp = client.get("/api/v1/auth/google/status")
    assert status_resp.get_json()["connected"] is True

    disconnect_resp = client.post("/api/v1/auth/google/disconnect")
    assert disconnect_resp.get_json()["connected"] is False

    status_after = client.get("/api/v1/auth/google/status")
    assert status_after.get_json()["connected"] is False


def test_callback_with_bad_state_returns_authentication_error(app, client):
    _install_configured_service(app)
    # Establish a session cookie first so the client has one to compare against.
    client.get("/api/v1/auth/google/status")
    resp = client.get("/api/v1/auth/google/callback?code=abc&state=not-a-real-state")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_callback_with_provider_error_returns_authentication_error(app, client):
    _install_configured_service(app)
    resp = client.get("/api/v1/auth/google/callback?error=access_denied")
    assert resp.status_code == 401


def test_health_endpoint_reports_google_oauth_configuration(client):
    resp = client.get("/api/v1/health")
    body = resp.get_json()
    assert "google_oauth" in body
    assert body["google_oauth"]["configured"] is False
