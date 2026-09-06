"""
Auth layer (Phase 2 - Google OAuth foundation; master spec section 17).

Responsibilities:
  * `models.py`   - plain data structures for a stored OAuth token and a
                     connection-status view of it (no framework deps).
  * `token_store.py` - `InMemoryTokenStore`, where tokens are actually
                     held (master spec section 51: introduce persistence
                     only when required - no database exists yet, so this
                     is the one store Nova has).
  * `session.py`  - maps an anonymous browser session (a random ID in a
                     signed, httponly cookie) to a token-store key. No
                     Google token ever touches the client; only an opaque
                     session ID does.

Nothing outside this package should import a concrete token-store backend
directly - go through `app.services.google_auth_service.GoogleAuthService`.
"""
