from __future__ import annotations

from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("", methods=["GET"])
def health():
    config = current_app.config["NOVA_CONFIG"]
    llm_provider = current_app.config["NOVA_LLM_PROVIDER"]
    llm_health = llm_provider.health_check()
    google_auth_service = current_app.config["NOVA_GOOGLE_AUTH_SERVICE"]
    tool_registry = current_app.config["NOVA_TOOL_REGISTRY"]

    return jsonify(
        {
            "success": True,
            "status": "ok",
            "env": config.env,
            "llm": {
                "provider": llm_health.provider_name,
                "available": llm_health.available,
                "detail": llm_health.detail,
            },
            "google_oauth": {
                "configured": google_auth_service.is_configured(),
            },
            "youtube": {
                "api_key_configured": bool(config.youtube_api_key),
                "tools_registered": [t.name for t in tool_registry.list_tools() if t.name.startswith("youtube.")],
            },
        }
    )
