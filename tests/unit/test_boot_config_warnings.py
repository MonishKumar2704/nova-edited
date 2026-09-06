from __future__ import annotations

from app import warn_on_insecure_boot_config
from app.core.config import Config


def _config(**overrides) -> Config:
    base = dict(
        env="development",
        secret_key="a-real-secret",
        google_redirect_uri="https://nova.example.com/api/v1/auth/google/callback",
    )
    base.update(overrides)
    return Config(**base)


def test_no_warnings_for_healthy_production_config():
    config = _config(env="production")
    assert warn_on_insecure_boot_config(config) == []


def test_warns_on_default_secret_key():
    config = _config(secret_key="dev-secret-change-me")
    warnings = warn_on_insecure_boot_config(config)
    assert any("SECRET_KEY" in w for w in warnings)


def test_warns_on_localhost_redirect_uri_in_production():
    config = _config(env="production", google_redirect_uri="http://localhost:8000/api/v1/auth/google/callback")
    warnings = warn_on_insecure_boot_config(config)
    assert any("GOOGLE_REDIRECT_URI" in w for w in warnings)


def test_no_localhost_warning_outside_production():
    # The localhost default is expected/fine for local dev - only a
    # production deployment pointing at localhost is a misconfiguration.
    config = _config(env="development", google_redirect_uri="http://localhost:8000/api/v1/auth/google/callback")
    assert warn_on_insecure_boot_config(config) == []


def test_no_localhost_warning_when_redirect_uri_properly_set_in_production():
    config = _config(env="production", google_redirect_uri="https://nova.example.com/api/v1/auth/google/callback")
    assert warn_on_insecure_boot_config(config) == []
