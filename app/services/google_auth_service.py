"""
Google auth application service (master spec section 17).

Orchestrates the OAuth connect/callback/status/disconnect workflow. Route
handlers in `app/api/v1/auth.py` stay thin (validation + serialization
only, per master spec section 5) and delegate everything here. This is
also the one place that knows how to keep an access token fresh
(`get_valid_access_token`), which Phase 3/6 YouTube and Gmail tools will
call before making an API request.
"""

from __future__ import annotations

import logging
import secrets

from app.auth.models import ConnectionStatus, GoogleTokenRecord
from app.auth.state_store import OAuthStateStore
from app.auth.token_store import InMemoryTokenStore
from app.core.errors import AuthenticationError, ValidationError
from app.integrations.google_oauth import GoogleOAuthClient

logger = logging.getLogger(__name__)


class GoogleAuthService:
    def __init__(
        self,
        *,
        oauth_client: GoogleOAuthClient,
        token_store: InMemoryTokenStore,
        state_store: OAuthStateStore,
        default_scopes: list[str],
    ) -> None:
        self._oauth_client = oauth_client
        self._token_store = token_store
        self._state_store = state_store
        self._default_scopes = default_scopes

    def is_configured(self) -> bool:
        return self._oauth_client.is_configured()

    def build_connect_url(self, *, session_id: str, extra_scopes: list[str] | None = None) -> str:
        if not self.is_configured():
            raise ValidationError(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET "
                "and GOOGLE_REDIRECT_URI (see .env.example)."
            )
        scopes = list(dict.fromkeys([*self._default_scopes, *(extra_scopes or [])]))  # de-dupe, keep order
        state = secrets.token_urlsafe(32)
        self._state_store.issue(state, session_id)
        return self._oauth_client.build_authorization_url(scopes=scopes, state=state)

    def handle_callback(self, *, session_id: str, code: str | None, state: str | None, error: str | None) -> ConnectionStatus:
        if error:
            raise AuthenticationError(f"Google OAuth consent was not granted: {error}")
        if not code or not state:
            raise ValidationError("Missing `code` or `state` in the OAuth callback.")

        bound_session_id = self._state_store.consume(state)
        if bound_session_id is None or bound_session_id != session_id:
            raise AuthenticationError("Invalid or expired OAuth state (possible CSRF or replay).")

        exchanged = self._oauth_client.exchange_code(code=code)
        identity = self._oauth_client.get_identity(access_token=exchanged.access_token)

        token = GoogleTokenRecord(
            access_token=exchanged.access_token,
            refresh_token=exchanged.refresh_token,
            scopes=exchanged.scopes or tuple(self._default_scopes),
            expires_at=exchanged.expires_at,
            google_email=identity.email,
            google_sub=identity.sub or None,
        )
        self._token_store.save(session_id, token)
        logger.info("Google account connected (sub=%s)", token.google_sub or "unknown")
        return ConnectionStatus.from_token(token)

    def get_status(self, *, session_id: str) -> ConnectionStatus:
        token = self._token_store.get(session_id)
        if token is None:
            return ConnectionStatus.disconnected()
        return ConnectionStatus.from_token(token)

    def disconnect(self, *, session_id: str) -> None:
        token = self._token_store.get(session_id)
        if token is not None:
            # Best-effort remote revocation; local record is removed regardless.
            try:
                self._oauth_client.revoke(token=token.refresh_token or token.access_token)
            except Exception:  # noqa: BLE001 - revocation is best-effort, never blocks local disconnect
                logger.warning("Google token revocation call failed; disconnecting locally anyway.", exc_info=True)
        self._token_store.delete(session_id)

    def get_valid_access_token(self, *, session_id: str) -> str:
        """Return a non-expired access token, refreshing it first if needed.

        This is the method Phase 3 (YouTube) and Phase 6 (Gmail) tool
        implementations should call rather than reading a stored token
        directly, so token refresh logic lives in exactly one place.
        """
        token = self._token_store.get(session_id)
        if token is None:
            raise AuthenticationError("Google account is not connected.")

        if not token.is_expired():
            return token.access_token

        if not token.refresh_token:
            self._token_store.delete(session_id)
            raise AuthenticationError("Google access token expired and no refresh token is available; reconnect required.")

        exchanged = self._oauth_client.refresh_access_token(refresh_token=token.refresh_token)
        refreshed = token.with_refreshed_access_token(access_token=exchanged.access_token, expires_at=exchanged.expires_at)
        self._token_store.save(session_id, refreshed)
        return refreshed.access_token
