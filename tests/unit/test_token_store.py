from __future__ import annotations

import time

from app.auth.models import GoogleTokenRecord
from app.auth.state_store import OAuthStateStore
from app.auth.token_store import InMemoryTokenStore


def make_token(**overrides) -> GoogleTokenRecord:
    base = dict(
        access_token="at",
        refresh_token="rt",
        scopes=("openid", "email"),
        expires_at=time.time() + 3600,
        google_email="user@example.com",
        google_sub="sub-1",
    )
    base.update(overrides)
    return GoogleTokenRecord(**base)


def test_token_is_expired_true_past_expiry():
    token = make_token(expires_at=time.time() - 10)
    assert token.is_expired() is True


def test_token_is_expired_false_when_fresh():
    token = make_token(expires_at=time.time() + 3600)
    assert token.is_expired() is False


def test_with_refreshed_access_token_preserves_refresh_token():
    token = make_token()
    refreshed = token.with_refreshed_access_token(access_token="new-at", expires_at=time.time() + 100)
    assert refreshed.access_token == "new-at"
    assert refreshed.refresh_token == token.refresh_token
    assert refreshed.google_sub == token.google_sub


def test_in_memory_store_roundtrip_without_encryption_key():
    store = InMemoryTokenStore(encryption_key=None)
    token = make_token()
    store.save("session-1", token)
    loaded = store.get("session-1")
    assert loaded is not None
    assert loaded.access_token == "at"


def test_in_memory_store_roundtrip_with_encryption_key():
    store = InMemoryTokenStore(encryption_key="a-very-secret-value")
    token = make_token()
    store.save("session-1", token)
    loaded = store.get("session-1")
    assert loaded.access_token == "at"
    # the raw stored blob must not contain the plaintext access token
    raw_blob = store._records["session-1"]  # noqa: SLF001 - white-box test of at-rest encryption
    assert b"at" != raw_blob
    assert b"access_token" not in raw_blob


def test_in_memory_store_get_missing_returns_none():
    store = InMemoryTokenStore(encryption_key=None)
    assert store.get("nope") is None


def test_in_memory_store_delete_is_idempotent():
    store = InMemoryTokenStore(encryption_key=None)
    store.save("s1", make_token())
    store.delete("s1")
    store.delete("s1")  # should not raise
    assert store.get("s1") is None


def test_state_store_issue_and_consume():
    store = OAuthStateStore()
    store.issue("state-1", "session-1")
    assert store.consume("state-1") == "session-1"


def test_state_store_consume_is_single_use():
    store = OAuthStateStore()
    store.issue("state-1", "session-1")
    store.consume("state-1")
    assert store.consume("state-1") is None


def test_state_store_consume_unknown_state_returns_none():
    store = OAuthStateStore()
    assert store.consume("never-issued") is None


def test_state_store_expired_state_returns_none():
    store = OAuthStateStore(ttl_seconds=-1)  # already expired at issue time
    store.issue("state-1", "session-1")
    assert store.consume("state-1") is None
