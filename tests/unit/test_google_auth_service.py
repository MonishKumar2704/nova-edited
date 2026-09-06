from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from app.auth.state_store import OAuthStateStore
from app.auth.token_store import InMemoryTokenStore
from app.core.errors import AuthenticationError, ValidationError
from app.integrations.google_oauth import ExchangedToken, GoogleIdentity
from app.services.google_auth_service import GoogleAuthService


def make_service(*, configured: bool = True) -> tuple[GoogleAuthService, Mock]:
    oauth_client = Mock()
    oauth_client.is_configured.return_value = configured
    oauth_client.build_authorization_url.return_value = "https://accounts.google.com/o/oauth2/v2/auth?mock=1"
    service = GoogleAuthService(
        oauth_client=oauth_client,
        token_store=InMemoryTokenStore(encryption_key=None),
        state_store=OAuthStateStore(),
        default_scopes=["openid", "email"],
    )
    return service, oauth_client


def test_build_connect_url_raises_when_not_configured():
    service, _ = make_service(configured=False)
    with pytest.raises(ValidationError):
        service.build_connect_url(session_id="s1")


def test_build_connect_url_returns_google_url_when_configured():
    service, oauth_client = make_service(configured=True)
    url = service.build_connect_url(session_id="s1")
    assert url == "https://accounts.google.com/o/oauth2/v2/auth?mock=1"
    oauth_client.build_authorization_url.assert_called_once()


def test_handle_callback_rejects_provider_error():
    service, _ = make_service()
    with pytest.raises(AuthenticationError):
        service.handle_callback(session_id="s1", code=None, state=None, error="access_denied")


def test_handle_callback_rejects_missing_code_or_state():
    service, _ = make_service()
    with pytest.raises(ValidationError):
        service.handle_callback(session_id="s1", code=None, state="st", error=None)


def test_handle_callback_rejects_unknown_state():
    service, _ = make_service()
    with pytest.raises(AuthenticationError):
        service.handle_callback(session_id="s1", code="c", state="never-issued", error=None)


def test_handle_callback_rejects_state_bound_to_different_session():
    service, _ = make_service()
    url = service.build_connect_url(session_id="session-A")
    # extract the state the service just issued via the mocked call args
    _, kwargs = service._oauth_client.build_authorization_url.call_args
    state = kwargs["state"]
    with pytest.raises(AuthenticationError):
        service.handle_callback(session_id="session-B", code="c", state=state, error=None)
    del url  # unused, just documents intent


def test_handle_callback_success_stores_token_and_returns_status():
    service, oauth_client = make_service()
    service.build_connect_url(session_id="session-A")
    _, kwargs = oauth_client.build_authorization_url.call_args
    state = kwargs["state"]

    oauth_client.exchange_code.return_value = ExchangedToken(
        access_token="at", refresh_token="rt", expires_at=time.time() + 3600, scopes=("openid", "email")
    )
    oauth_client.get_identity.return_value = GoogleIdentity(sub="sub-1", email="user@example.com")

    status = service.handle_callback(session_id="session-A", code="the-code", state=state, error=None)

    assert status.connected is True
    assert status.google_email == "user@example.com"
    assert service.get_status(session_id="session-A").connected is True


def test_get_status_disconnected_by_default():
    service, _ = make_service()
    assert service.get_status(session_id="never-connected").connected is False


def test_disconnect_removes_token_even_if_revoke_fails():
    service, oauth_client = make_service()
    service.build_connect_url(session_id="session-A")
    _, kwargs = oauth_client.build_authorization_url.call_args
    state = kwargs["state"]
    oauth_client.exchange_code.return_value = ExchangedToken(
        access_token="at", refresh_token="rt", expires_at=time.time() + 3600, scopes=("openid",)
    )
    oauth_client.get_identity.return_value = GoogleIdentity(sub="sub-1", email="user@example.com")
    service.handle_callback(session_id="session-A", code="c", state=state, error=None)

    oauth_client.revoke.side_effect = RuntimeError("network down")
    service.disconnect(session_id="session-A")

    assert service.get_status(session_id="session-A").connected is False


def test_get_valid_access_token_raises_when_not_connected():
    service, _ = make_service()
    with pytest.raises(AuthenticationError):
        service.get_valid_access_token(session_id="never-connected")


def test_get_valid_access_token_returns_cached_token_when_fresh():
    service, oauth_client = make_service()
    service.build_connect_url(session_id="session-A")
    _, kwargs = oauth_client.build_authorization_url.call_args
    state = kwargs["state"]
    oauth_client.exchange_code.return_value = ExchangedToken(
        access_token="at", refresh_token="rt", expires_at=time.time() + 3600, scopes=("openid",)
    )
    oauth_client.get_identity.return_value = GoogleIdentity(sub="sub-1", email="e@x.com")
    service.handle_callback(session_id="session-A", code="c", state=state, error=None)

    token = service.get_valid_access_token(session_id="session-A")
    assert token == "at"
    oauth_client.refresh_access_token.assert_not_called()


def test_get_valid_access_token_refreshes_when_expired():
    service, oauth_client = make_service()
    service.build_connect_url(session_id="session-A")
    _, kwargs = oauth_client.build_authorization_url.call_args
    state = kwargs["state"]
    oauth_client.exchange_code.return_value = ExchangedToken(
        access_token="at-old", refresh_token="rt", expires_at=time.time() - 10, scopes=("openid",)
    )
    oauth_client.get_identity.return_value = GoogleIdentity(sub="sub-1", email="e@x.com")
    service.handle_callback(session_id="session-A", code="c", state=state, error=None)

    oauth_client.refresh_access_token.return_value = ExchangedToken(
        access_token="at-new", refresh_token="rt", expires_at=time.time() + 3600, scopes=("openid",)
    )
    token = service.get_valid_access_token(session_id="session-A")
    assert token == "at-new"
    oauth_client.refresh_access_token.assert_called_once_with(refresh_token="rt")


def test_get_valid_access_token_raises_when_expired_without_refresh_token():
    service, oauth_client = make_service()
    service.build_connect_url(session_id="session-A")
    _, kwargs = oauth_client.build_authorization_url.call_args
    state = kwargs["state"]
    oauth_client.exchange_code.return_value = ExchangedToken(
        access_token="at", refresh_token=None, expires_at=time.time() - 10, scopes=("openid",)
    )
    oauth_client.get_identity.return_value = GoogleIdentity(sub="sub-1", email="e@x.com")
    service.handle_callback(session_id="session-A", code="c", state=state, error=None)

    with pytest.raises(AuthenticationError):
        service.get_valid_access_token(session_id="session-A")
    assert service.get_status(session_id="session-A").connected is False
