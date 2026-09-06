"""Domain models for the Google OAuth connection (framework-independent)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoogleTokenRecord:
    """A single user's stored Google OAuth token.

    `access_token` and `refresh_token` are the only genuinely sensitive
    fields here. `InMemoryTokenStore` (see `token_store.py`) is
    responsible for encrypting this record at rest. Never log
    this object or serialize it directly to a client response - use
    `ConnectionStatus` (below) for anything user/API-facing.
    """

    access_token: str
    refresh_token: str | None
    scopes: tuple[str, ...]
    expires_at: float  # unix timestamp
    google_email: str | None = None
    google_sub: str | None = None  # stable Google account ID ("sub" claim)
    connected_at: float = field(default_factory=time.time)

    def is_expired(self, *, skew_seconds: float = 60.0) -> bool:
        """True if the access token is expired (or about to be, within `skew_seconds`)."""
        return time.time() >= (self.expires_at - skew_seconds)

    def with_refreshed_access_token(self, *, access_token: str, expires_at: float) -> "GoogleTokenRecord":
        """Return a copy with a new access token (refresh tokens are usually not reissued)."""
        return GoogleTokenRecord(
            access_token=access_token,
            refresh_token=self.refresh_token,
            scopes=self.scopes,
            expires_at=expires_at,
            google_email=self.google_email,
            google_sub=self.google_sub,
            connected_at=self.connected_at,
        )


@dataclass(frozen=True)
class ConnectionStatus:
    """Safe-to-serialize view of a user's Google connection state.

    Deliberately excludes tokens entirely - this is what `/api/v1/auth/google/status`
    and other API responses should be built from.
    """

    connected: bool
    google_email: str | None = None
    scopes: tuple[str, ...] = ()
    connected_at: float | None = None
    expires_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "google_email": self.google_email,
            "scopes": list(self.scopes),
            "connected_at": self.connected_at,
            "expires_at": self.expires_at,
        }

    @staticmethod
    def disconnected() -> "ConnectionStatus":
        return ConnectionStatus(connected=False)

    @staticmethod
    def from_token(token: GoogleTokenRecord) -> "ConnectionStatus":
        return ConnectionStatus(
            connected=True,
            google_email=token.google_email,
            scopes=token.scopes,
            connected_at=token.connected_at,
            expires_at=token.expires_at,
        )
