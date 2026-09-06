from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from app.core.errors import AuthenticationError, NetworkError, TimeoutErrorNova
from app.integrations.google_oauth import GoogleOAuthClient


def make_client() -> GoogleOAuthClient:
    return GoogleOAuthClient(
        client_id="client-123",
        client_secret="secret-456",
        redirect_uri="http://localhost:8000/api/v1/auth/google/callback",
    )


def test_is_configured_true_when_all_fields_set():
    assert make_client().is_configured() is True


def test_is_configured_false_when_missing_fields():
    assert GoogleOAuthClient(client_id="", client_secret="s", redirect_uri="r").is_configured() is False


def test_build_authorization_url_contains_required_params():
    url = make_client().build_authorization_url(scopes=["openid", "email"], state="abc123")
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=client-123" in url
    assert "state=abc123" in url
    assert "access_type=offline" in url
    assert "scope=openid%20email" in url


def _mock_response(*, status_code=200, json_body=None):
    resp = Mock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    return resp


@patch("app.integrations.google_oauth.requests.post")
def test_exchange_code_success(mock_post):
    mock_post.return_value = _mock_response(
        json_body={"access_token": "at", "refresh_token": "rt", "expires_in": 3600, "scope": "openid email"}
    )
    result = make_client().exchange_code(code="the-code")
    assert result.access_token == "at"
    assert result.refresh_token == "rt"
    assert result.scopes == ("openid", "email")
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["code"] == "the-code"
    assert kwargs["data"]["grant_type"] == "authorization_code"


@patch("app.integrations.google_oauth.requests.post")
def test_exchange_code_bad_request_raises_authentication_error(mock_post):
    mock_post.return_value = _mock_response(status_code=400, json_body={"error": "invalid_grant"})
    with pytest.raises(AuthenticationError):
        make_client().exchange_code(code="bad-code")


@patch("app.integrations.google_oauth.requests.post")
def test_exchange_code_timeout_raises_timeout_error(mock_post):
    mock_post.side_effect = requests.Timeout()
    with pytest.raises(TimeoutErrorNova):
        make_client().exchange_code(code="c")


@patch("app.integrations.google_oauth.requests.post")
def test_exchange_code_network_failure_raises_network_error(mock_post):
    mock_post.side_effect = requests.ConnectionError()
    with pytest.raises(NetworkError):
        make_client().exchange_code(code="c")


@patch("app.integrations.google_oauth.requests.post")
def test_refresh_access_token_keeps_original_refresh_token_if_not_reissued(mock_post):
    mock_post.return_value = _mock_response(json_body={"access_token": "new-at", "expires_in": 3600, "scope": "openid"})
    result = make_client().refresh_access_token(refresh_token="original-rt")
    assert result.access_token == "new-at"
    assert result.refresh_token == "original-rt"


@patch("app.integrations.google_oauth.requests.get")
def test_get_identity_success(mock_get):
    mock_get.return_value = _mock_response(json_body={"sub": "12345", "email": "user@example.com"})
    identity = make_client().get_identity(access_token="at")
    assert identity.sub == "12345"
    assert identity.email == "user@example.com"


@patch("app.integrations.google_oauth.requests.get")
def test_get_identity_unauthorized_raises_authentication_error(mock_get):
    mock_get.return_value = _mock_response(status_code=401)
    with pytest.raises(AuthenticationError):
        make_client().get_identity(access_token="bad-token")
