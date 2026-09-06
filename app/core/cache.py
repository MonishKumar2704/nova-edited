"""
Small in-memory TTL cache (master spec section 50 - caching search results).

Deliberately not a dependency on any external cache service (Redis, etc) -
Nova has no persistence infrastructure yet (section 51: don't add one
"merely because production applications usually have one"). This is
process-local, bounded by TTL + an optional max-entry eviction, and
thread-safe. Swap for a shared cache (e.g. Redis-backed) later without
changing any call site, since callers only see `get`/`set`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    def __init__(self, *, ttl_seconds: float, max_entries: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._store: dict[K, tuple[float, V]] = {}

    def get(self, key: K) -> V | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if len(self._store) >= self._max_entries and key not in self._store:
                self._evict_oldest_locked()
            self._store[key] = (time.time() + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _evict_oldest_locked(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][0])
        self._store.pop(oldest_key, None)


def make_cache_key(*parts: Any) -> str:
    """Stable, order-preserving cache key from arbitrary hashable-ish parts."""
    return "|".join(str(p) for p in parts)
