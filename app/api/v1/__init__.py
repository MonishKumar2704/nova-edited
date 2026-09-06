"""API v1: /api/v1/health, /api/v1/agent, /api/v1/auth, /api/v1/youtube, /api/v1/gmail."""

from __future__ import annotations

from flask import Flask

from app.api.v1.agent import agent_bp
from app.api.v1.auth import auth_bp
from app.api.v1.gmail import gmail_bp
from app.api.v1.health import health_bp
from app.api.v1.youtube import youtube_bp


def register_v1_blueprints(app: Flask) -> None:
    app.register_blueprint(health_bp, url_prefix="/api/v1/health")
    app.register_blueprint(agent_bp, url_prefix="/api/v1/agent")
    app.register_blueprint(auth_bp, url_prefix="/api/v1/auth")
    app.register_blueprint(youtube_bp, url_prefix="/api/v1/youtube")
    app.register_blueprint(gmail_bp, url_prefix="/api/v1/gmail")
