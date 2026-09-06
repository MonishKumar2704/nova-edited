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

import time
from dataclasses import dataclass

import requests

from app.core.errors import AuthenticationError, NetworkError, TimeoutErrorNova

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"

_REQUEST_TIMEOUT_SECONDS = 10


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

        if resp.status_code == 400:
            raise AuthenticationError("Google rejected the OAuth request (invalid/expired code or token).")
        if resp.status_code == 401:
            raise AuthenticationError("Google rejected the OAuth client credentials.")
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
