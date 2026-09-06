"""
Retry-with-backoff infrastructure (master spec section 49 - RETRIES).

Only transient, classified errors are retried (network/timeout/rate-limit).
Validation, auth, and other "this will never succeed by retrying" errors
are never retried. Destructive operations must opt in explicitly by
passing `retryable=()`/omitting retry entirely at the call site - this
module never assumes an operation is safe to retry blindly.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Iterable, TypeVar

from app.core.errors import NetworkError, NovaError, RateLimitError, TimeoutErrorNova

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Error types that are safe to retry by default: transient infrastructure
#: failures, not application/business errors.
DEFAULT_RETRYABLE_ERRORS: tuple[type[NovaError], ...] = (NetworkError, TimeoutErrorNova, RateLimitError)


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable_errors: Iterable[type[NovaError]] = DEFAULT_RETRYABLE_ERRORS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()`, retrying on `retryable_errors` with exponential backoff + jitter.

    Raises the last exception once `max_attempts` is exhausted. Any
    exception not in `retryable_errors` propagates immediately without
    being retried.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    retryable = tuple(retryable_errors)
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except retryable as exc:
            if attempt >= max_attempts:
                logger.warning("Retryable operation failed after %d attempt(s): %s", attempt, exc)
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random())  # full jitter around the backoff window
            logger.info(
                "Retryable error on attempt %d/%d (%s); retrying in %.2fs",
                attempt,
                max_attempts,
                exc.__class__.__name__,
                delay,
            )
            sleep(delay)
