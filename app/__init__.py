"""
Nova AI application factory.

Phase 0 goals implemented here:
  * clean app factory (no module-level global Flask app)
  * configuration, logging, and error-handling foundations wired in
  * request-ID middleware for observability
  * API versioning foundation (`/api/v1/...`)
  * health endpoint
  * legacy `/` and `/agent` routes preserved for backward compatibility
    with the existing Nova UI (`app/templates/index.html`), delegating
    to the same `AgentService` the new versioned endpoint uses
"""

from __future__ import annotations

from flask import Flask, g, jsonify, render_template, request
from pydantic import ValidationError as PydanticValidationError

from app.agent.orchestrator import LegacyRuleBasedOrchestrator
from app.agent.planner_orchestrator import PlannerOrchestrator
from app.agent.state import AgentStateStore
from app.api.v1 import register_v1_blueprints
from app.api.v1.agent import build_agent_context, serialize_agent_result
from app.auth.state_store import OAuthStateStore
from app.auth.token_store import InMemoryTokenStore
from app.core.cache import TTLCache
from app.core.config import get_config
from app.core.errors import ValidationError, register_error_handlers
from app.core.logging import configure_logging
from app.core.request_context import register_request_context
from app.integrations.gmail_api import GmailApiClient
from app.integrations.google_oauth import GoogleOAuthClient
from app.integrations.youtube_api import YouTubeApiClient
from app.llm.factory import build_llm_provider
from app.schemas.agent import AgentCommandRequest
from app.services.agent_service import AgentService
from app.services.google_auth_service import GoogleAuthService
from app.tools.ai import register_ai_tools
from app.tools.gmail import register_gmail_tools
from app.tools.registry import ToolRegistry
from app.tools.youtube import register_youtube_tools


def create_app() -> Flask:
    config = get_config()
    configure_logging(config.log_level)

    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key
    app.config["NOVA_CONFIG"] = config

    # --- Session cookie hardening (master spec section 15/17) ---
    # Only an opaque session ID ever lives in this cookie (see
    # app/auth/session.py) - never a Google token. httponly blocks JS
    # access; samesite=Lax mitigates CSRF on top of the explicit OAuth
    # `state` check; secure is enabled outside local dev.
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = config.is_production()

    # --- Foundations ---
    register_request_context(app)
    register_error_handlers(app)

    # --- LLM provider (abstraction only in Phase 0; see app/llm) ---
    llm_provider = build_llm_provider(config)
    app.config["NOVA_LLM_PROVIDER"] = llm_provider

    # --- Tool registry ---
    # A fresh instance per app (not a module-level singleton), matching
    # the task manager below - each test's `create_app()` gets its own
    # registry instead of hitting "already registered" on the second call.
    tool_registry = ToolRegistry()
    app.config["NOVA_TOOL_REGISTRY"] = tool_registry

    # --- YouTube Data API v3 (Phase 3) ---
    youtube_client = YouTubeApiClient(api_key=config.youtube_api_key)
    youtube_search_cache = TTLCache(ttl_seconds=config.youtube_search_cache_ttl_seconds)
    register_youtube_tools(tool_registry, client=youtube_client, search_cache=youtube_search_cache)

    # --- Gmail API v1 (Phase 6: read-only foundation) ---
    # No API-key/search-cache knobs here (unlike YouTube): every Gmail
    # endpoint is account-specific, and inbox contents change too often
    # for a TTL cache to be worth the staleness risk at this stage.
    gmail_client = GmailApiClient()
    register_gmail_tools(tool_registry, client=gmail_client)

    # --- AI email generation (Phase 5: OLLAMA EMAIL AI) ---
    # Reuses the same `llm_provider` the agent planner uses for command
    # routing (free local Ollama by default, see app.llm.factory) - one
    # configured AI backend, not a second provider wired up separately.
    register_ai_tools(tool_registry, llm_provider=llm_provider, gmail_client=gmail_client)

    # --- Google OAuth (Phase 2: connect/callback/status/disconnect foundation) ---
    # Created before the agent service below because `PlannerOrchestrator`
    # itself never talks to Google directly (only tools do), but the
    # legacy `/agent` route's context-building needs it, same as the
    # versioned `/api/v1/agent/command` route.
    google_oauth_client = GoogleOAuthClient(
        client_id=config.google_client_id,
        client_secret=config.google_client_secret,
        redirect_uri=config.google_redirect_uri,
    )
    google_auth_service = GoogleAuthService(
        oauth_client=google_oauth_client,
        token_store=InMemoryTokenStore(encryption_key=config.token_encryption_key),
        state_store=OAuthStateStore(),
        default_scopes=list(config.google_scopes),
    )
    app.config["NOVA_GOOGLE_AUTH_SERVICE"] = google_auth_service

    # --- Agent orchestrator (Phase 9: real LLM-driven planner) ---
    # `PlannerOrchestrator` is the primary orchestrator: it interprets
    # intent, discovers/selects tools from `tool_registry` via the
    # configured `llm_provider`, plans (bounded) multi-step tool use, and
    # gates sensitive tools behind explicit confirmation. It wraps
    # `LegacyRuleBasedOrchestrator` as its own fallback for sessions where
    # no LLM provider is configured/reachable, so Nova stays usable for
    # basic commands without one (master spec section 15: free-first).
    agent_state_store = AgentStateStore()
    app.config["NOVA_AGENT_STATE_STORE"] = agent_state_store
    legacy_orchestrator = LegacyRuleBasedOrchestrator(tool_registry=tool_registry)
    planner_orchestrator = PlannerOrchestrator(
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        state_store=agent_state_store,
        legacy_fallback=legacy_orchestrator,
        max_tool_iterations=config.agent_max_tool_iterations,
        llm_timeout=config.agent_llm_timeout_seconds,
    )
    agent_service = AgentService(orchestrator=planner_orchestrator)
    app.config["NOVA_AGENT_SERVICE"] = agent_service

    # --- Versioned API ---
    register_v1_blueprints(app)

    # --- Legacy routes (backward compatible with existing frontend) ---
    _register_legacy_routes(app, agent_service)

    return app


def _register_legacy_routes(app: Flask, agent_service: AgentService) -> None:
    @app.route("/", methods=["GET"])
    def home():
        return render_template("index.html")

    @app.route("/agent", methods=["POST"])
    def legacy_agent_router():
        payload = request.get_json(silent=True) or {}
        raw_command = payload.get("command") or payload.get("text_command")
        confirm = payload.get("confirm")
        if confirm is not None and not isinstance(confirm, bool):
            confirm = None

        try:
            req = AgentCommandRequest(command=raw_command or "", confirm=confirm)
        except PydanticValidationError as exc:
            raise ValidationError("`command` or `text_command` is required.") from exc

        # Same context resolution and response shape as the versioned
        # `/api/v1/agent/command` route (see `app.api.v1.agent`) - the
        # legacy route only adds the `text_command` alias and the
        # `command`-or-`text_command` validation message above it.
        context = build_agent_context(req.confirm)
        result = agent_service.handle_command(req.command, context=context)
        return jsonify({**serialize_agent_result(result), "request_id": getattr(g, "request_id", None)})
