"""
Configuration foundation.

All configuration is sourced from environment variables (optionally loaded
from a local `.env` file via python-dotenv during development). Nothing in
this module hard-codes secrets.

Only variables actually used by the current phase (Phase 0: architecture
foundation) are read here. Later phases (Google OAuth, Gmail, YouTube,
LLM providers) will extend this object without requiring a rewrite of the
call sites, since the rest of the app depends on the `Config` object, not
on `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a dev convenience only
    pass


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # --- General ---
    env: str = field(default_factory=lambda: os.environ.get("FLASK_ENV", "production"))
    debug: bool = field(default_factory=lambda: _get_bool("DEBUG", False))
    secret_key: str = field(default_factory=lambda: os.environ.get("SECRET_KEY", "dev-secret-change-me"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    # --- LLM provider abstraction (Phase 0 foundation; wired up in Phase 8) ---
    # Primary provider. "none" keeps the app fully functional without any
    # LLM configured (master spec section 15 - FREE-FIRST REQUIREMENT: a
    # paid API must never be required for Nova to run).
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "none"))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
    # Retry budget for LLM calls (used by `app.llm.retry.retry_call` inside
    # each concrete provider). Per-request timeout itself is
    # `agent_llm_timeout_seconds` below, passed explicitly by the caller
    # (`PlannerOrchestrator`) into each `LLMProvider.generate()` call.
    llm_max_retries: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "3")))

    # --- Agent orchestrator (Phase 9; master spec section 9) ---
    # Bounded plan/act loop budget: how many rounds of "ask the LLM what
    # to do next" the planner will run for a single user command before
    # giving up. Bounded on purpose (same "no uncontrolled loops" principle
    # as task/concurrency infra) - a runaway tool-call loop should fail
    # closed with a clear message, not spin indefinitely burning LLM calls.
    agent_max_tool_iterations: int = field(
        default_factory=lambda: int(os.environ.get("AGENT_MAX_TOOL_ITERATIONS", "5"))
    )
    # Per-command timeout passed to each LLMProvider.generate() call.
    agent_llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("AGENT_LLM_TIMEOUT_SECONDS", "20"))
    )

    # --- Google OAuth (Phase 2 foundation; master spec section 17) ---
    google_client_id: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLIENT_SECRET", ""))
    google_redirect_uri: str = field(
        default_factory=lambda: os.environ.get(
            "GOOGLE_REDIRECT_URI", "http://localhost:8000/api/v1/auth/google/callback"
        )
    )
    # Space/comma-separated list of scopes to request. Kept minimal on
    # purpose (master spec section 17: "do not request all permissions
    # unnecessarily") - just enough to connect an account, confirm its
    # identity, and read/manage the account's own YouTube data. The
    # non-readonly `youtube` scope replaced `youtube.readonly` in Phase 5:
    # playlist/video management (create, update, delete, rate) needs
    # write access, and YouTube does not offer a narrower "read + write
    # only playlists" scope to request instead. Phase 6 added
    # `gmail.readonly` (list/get messages, threads, labels - see
    # app.tools.gmail). Phase 7 adds `gmail.modify` (mark read/unread,
    # archive, trash, star, labels - all done via the same `messages.modify`
    # endpoint, see `GmailApiClient.modify_message`) and `gmail.compose`
    # (drafts + send: `gmail.compose` alone is sufficient for
    # creating/updating/sending drafts and sending messages directly, and
    # is narrower than the full `gmail.send`/`mail.google.com` scopes).
    google_scopes: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s
            for s in os.environ.get(
                "GOOGLE_SCOPES",
                "openid https://www.googleapis.com/auth/userinfo.email "
                "https://www.googleapis.com/auth/youtube "
                "https://www.googleapis.com/auth/gmail.readonly "
                "https://www.googleapis.com/auth/gmail.modify "
                "https://www.googleapis.com/auth/gmail.compose",
            )
            .replace(",", " ")
            .split()
            if s
        )
    )
    # Key used to encrypt stored OAuth tokens at rest (see app/auth/token_store.py).
    # Any random string works; it is stretched into a Fernet key internally.
    token_encryption_key: str = field(default_factory=lambda: os.environ.get("TOKEN_ENCRYPTION_KEY", ""))

    # --- YouTube Data API v3 (Phase 3; master spec sections 18-19) ---
    # Public-data API key from Google Cloud Console (APIs & Services ->
    # Credentials -> API key, with the "YouTube Data API v3" enabled).
    # Free tier; no OAuth/user connection required for search/get_video.
    # Leave blank and youtube.search/get_video still work for any session
    # that has connected a Google account (falls back to that user's
    # OAuth token instead of a key); youtube.list_channels always
    # requires a connected account regardless of this setting.
    youtube_api_key: str = field(default_factory=lambda: os.environ.get("YOUTUBE_API_KEY", ""))
    # How long search/get_video results are cached in-process before a
    # repeat query hits the API again (master spec section 50).
    youtube_search_cache_ttl_seconds: float = field(
        default_factory=lambda: float(os.environ.get("YOUTUBE_SEARCH_CACHE_TTL_SECONDS", "300"))
    )

    def is_production(self) -> bool:
        return self.env.lower() == "production"


_config: Config | None = None


def get_config() -> Config:
    """Return a process-wide singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
