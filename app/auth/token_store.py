"""
Token storage (master spec section 17 - "OAuth tokens must be handled
securely" - and section 51 - "introduce persistence only when required").

`InMemoryTokenStore` is the only token store Nova has, and the only one
this mini project needs: it keeps tokens out of any database (none exists
yet), but it does encrypt records at rest in the process using a Fernet
key derived from `TOKEN_ENCRYPTION_KEY`, so a memory dump/log accident
doesn't trivially leak a live Google refresh token.

Known, documented limitation (tracked, not an oversight): being in-memory,
tokens are lost on process restart and are not shared across multiple
worker processes. That's acceptable for a single-process deployment
(today's target - see docs/deployment.md). If Nova ever needs a different
backend, add it then - a single-implementation interface kept "for later"
isn't worth the indirection today (nothing here constructs any store other
than this one).

Task 17 security audit fix: this used to serialize `GoogleTokenRecord`
with `pickle.dumps`/`pickle.loads`. Even though the encrypted blob is
process-local and self-produced (never attacker-controlled input), a
`pickle.loads` call is a well-known unsafe-deserialization pattern any
security review will flag on sight, and `GoogleTokenRecord` is nothing
but primitives (`str`/`float`/`tuple[str, ...]`/`None`) - there is no
reason to use a format capable of reconstructing arbitrary Python objects
when plain JSON round-trips this exact shape just as well.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from dataclasses import asdict

from app.auth.models import GoogleTokenRecord

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a valid 32-byte urlsafe-base64 Fernet key from an arbitrary secret.

    Lets operators set `TOKEN_ENCRYPTION_KEY` to any sufficiently random
    string rather than requiring them to pre-generate a Fernet key.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class InMemoryTokenStore:
    """Thread-safe, encrypted-at-rest, single-process token store.

    Encryption is best-effort defense in depth for this process's memory/
    any accidental serialization - it is NOT a substitute for transport
    security (HTTPS) or for keeping `TOKEN_ENCRYPTION_KEY` itself secret.
    """

    def __init__(self, encryption_key: str | None) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, bytes] = {}
        self._fernet = None
        if encryption_key:
            try:
                from cryptography.fernet import Fernet

                self._fernet = Fernet(_derive_fernet_key(encryption_key))
            except ImportError:  # pragma: no cover - cryptography is a declared dependency
                logger.warning(
                    "TOKEN_ENCRYPTION_KEY is set but the `cryptography` package is not "
                    "installed; storing tokens without at-rest encryption."
                )
        else:
            logger.warning(
                "TOKEN_ENCRYPTION_KEY is not set; Google OAuth tokens will be held "
                "in-memory unencrypted. Set TOKEN_ENCRYPTION_KEY in production."
            )

    def _encode(self, token: GoogleTokenRecord) -> bytes:
        raw = json.dumps(asdict(token)).encode("utf-8")
        return self._fernet.encrypt(raw) if self._fernet else raw

    def _decode(self, blob: bytes) -> GoogleTokenRecord:
        raw = self._fernet.decrypt(blob) if self._fernet else blob
        data = json.loads(raw.decode("utf-8"))
        data["scopes"] = tuple(data.get("scopes") or ())
        return GoogleTokenRecord(**data)

    def get(self, session_id: str) -> GoogleTokenRecord | None:
        with self._lock:
            blob = self._records.get(session_id)
        return self._decode(blob) if blob is not None else None

    def save(self, session_id: str, token: GoogleTokenRecord) -> None:
        blob = self._encode(token)
        with self._lock:
            self._records[session_id] = blob

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._records.pop(session_id, None)

