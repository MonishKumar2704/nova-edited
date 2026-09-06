"""
Google OAuth 2.0 integration client (master spec section 17).

Deliberately implemented against Google's plain OAuth/OpenID HTTP
endpoints with `requests`, rather than pulling in `google-auth-oauthlib` /
`google-api-python-client`. Phase 2 only needs the standard authorization-
code flow (auth URL, code exchange, refresh, revoke, identity lookup) -
official REST endpoints, no scraping, no unofficial APIs (master spec
section 59). This keeps the dependency footprint small; Phase 3/6 can add
`google-api-python-client` when it's actually needed for YouTube/Gmail
calls without touching this module.

This module has ZERO Flask/session/storage knowledge - it only knows how
to talk to Google. Orchestration (state/CSRF handling, session mapping,
persistence) lives in `app.services.google_auth_service`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from app.core.errors import AuthenticationError, NetworkError, TimeoutErrorNova

logger = logging.getLogger(__name__)

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"

_REQUEST_TIMEOUT_SECONDS = 10

# RFC 6749 section 5.2 / Google's documented token-endpoint error codes,
# mapped to an actionable (still secret-free) hint for whoever is
# debugging a broken OAuth connection.
_OAUTH_ERROR_HINTS: dict[str, str] = {
    "invalid_client": (
        "Google rejected the OAuth client credentials - GOOGLE_CLIENT_ID/"
        "GOOGLE_CLIENT_SECRET do not match a valid Google Cloud OAuth client "
        "(wrong value, wrong client type, or the client was deleted/disabled)."
    ),
    "invalid_grant": (
        "Google rejected the authorization code or refresh token (already used, "
        "expired, or issued for a different redirect_uri/client)."
    ),
    "redirect_uri_mismatch": (
        "The redirect_uri Nova sent does not exactly match an authorized "
        "redirect URI configured on the Google Cloud OAuth client."
    ),
    "invalid_request": (
        "The OAuth request was missing a required parameter or was otherwise "
        "malformed."
    ),
    "unauthorized_client": (
        "This OAuth client is not authorized to use the requested grant type "
        "(check the OAuth client type/configuration in Google Cloud)."
    ),
    "access_denied": "The user (or Google's policy) denied the OAuth consent request.",
    "unsupported_grant_type": "Nova sent a grant_type Google's token endpoint does not support.",
    "invalid_scope": "One or more requested scopes are invalid or not enabled for this OAuth client.",
}


@dataclass(frozen=True)
class ExchangedToken:
    access_token: str
    refresh_token: str | None
    expires_at: float
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str
    email: str | None


class GoogleOAuthClient:
    """Thin, official-endpoints-only wrapper around Google's OAuth flow."""

    def __init__(self, *, client_id: str, client_secret: str, redirect_uri: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    def is_configured(self) -> bool:
        return bool(self._client_id and self._client_secret and self._redirect_uri)

    def build_authorization_url(self, *, scopes: list[str], state: str) -> str:
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
            # Ensures a refresh_token is returned even for a previously-
            # authorized user (Google only returns it on first consent
            # otherwise).
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
        }
        query = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in params.items())
        return f"{AUTHORIZATION_ENDPOINT}?{query}"

    def exchange_code(self, *, code: str) -> ExchangedToken:
        response = self._post(
            TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "redirect_uri": self._redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        return self._to_exchanged_token(response)

    def refresh_access_token(self, *, refresh_token: str) -> ExchangedToken:
        response = self._post(
            TOKEN_ENDPOINT,
            data={
                "refresh_token": refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "refresh_token",
            },
        )
        # Google typically does not re-issue a refresh_token on refresh;
        # the caller keeps the original one.
        exchanged = self._to_exchanged_token(response)
        return ExchangedToken(
            access_token=exchanged.access_token,
            refresh_token=exchanged.refresh_token or refresh_token,
            expires_at=exchanged.expires_at,
            scopes=exchanged.scopes,
        )

    def revoke(self, *, token: str) -> None:
        try:
            requests.post(
                REVOKE_ENDPOINT,
                params={"token": token},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out revoking the Google token.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach Google to revoke the token.") from exc
        # Revocation is best-effort from the caller's point of view: even
        # if this fails, the local token record is still deleted by the
        # service layer, so the app-side connection is always severed.

    def get_identity(self, *, access_token: str) -> GoogleIdentity:
        try:
            resp = requests.get(
                USERINFO_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out fetching Google account identity.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach Google to fetch account identity.") from exc

        if resp.status_code == 401:
            raise AuthenticationError("Google rejected the access token while fetching identity.")
        if not resp.ok:
            raise NetworkError(f"Google userinfo endpoint returned HTTP {resp.status_code}.")

        body = resp.json()
        return GoogleIdentity(sub=body.get("sub", ""), email=body.get("email"))

    # --- internal ---

    def _post(self, url: str, *, data: dict) -> dict:
        try:
            resp = requests.post(url, data=data, timeout=_REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out talking to Google's OAuth endpoint.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach Google's OAuth endpoint.") from exc

        if resp.status_code in (400, 401):
            # Google's token-endpoint error body (RFC 6749 section 5.2) carries
            # the actual reason - `error` is one of a small fixed vocabulary
            # (invalid_client, invalid_grant, redirect_uri_mismatch,
            # invalid_request, unauthorized_client, ...) and `error_description`
            # is Google's human-readable elaboration. Neither ever contains a
            # secret (client_secret/tokens are never echoed back by Google),
            # so both are safe to log/report - unlike the client_id/secret we
            # sent, which we never include here.
            error_body: dict = {}
            try:
                error_body = resp.json()
            except ValueError:
                pass
            google_error = error_body.get("error") or "unknown_error"
            google_description = error_body.get("error_description") or ""
            google_uri = error_body.get("error_uri") or ""

            logger.warning(
                "Google OAuth endpoint returned HTTP %s: error=%s description=%s error_uri=%s",
                resp.status_code,
                google_error,
                google_description,
                google_uri,
            )

            # Map the well-known error codes to a message that's actually
            # useful for fixing configuration, without ever including
            # request parameters (client_id, code, redirect_uri) that could
            # leak details we don't need to leak client-side.
            hint = _OAUTH_ERROR_HINTS.get(google_error, "Google rejected the OAuth request.")
            raise AuthenticationError(
                f"{hint} (google_error={google_error})" if google_error != "unknown_error" else hint,
                details={
                    "google_error": google_error,
                    "google_error_description": google_description,
                    "google_error_uri": google_uri,
                    "http_status": resp.status_code,
                },
            )
        if not resp.ok:
            raise NetworkError(f"Google's OAuth endpoint returned HTTP {resp.status_code}.")

        return resp.json()

    @staticmethod
    def _to_exchanged_token(body: dict) -> ExchangedToken:
        access_token = body.get("access_token")
        if not access_token:
            raise AuthenticationError("Google's OAuth response did not include an access token.")
        expires_in = float(body.get("expires_in", 3600))
        scope_str = body.get("scope", "")
        return ExchangedToken(
            access_token=access_token,
            refresh_token=body.get("refresh_token"),
            expires_at=time.time() + expires_in,
            scopes=tuple(scope_str.split()) if scope_str else (),
        )
