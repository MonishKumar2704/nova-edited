import pytest

from app.core.config import Config
from app.core.errors import LLMError
from app.llm.factory import build_llm_provider
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.null_provider import NullProvider
from app.llm.providers.ollama_provider import OllamaProvider


def make_config(**overrides) -> Config:
    base = dict(llm_provider="none", gemini_api_key="", gemini_model="x", ollama_base_url="x", ollama_model="x")
    base.update(overrides)
    return Config(**base)


def test_factory_defaults_to_null_provider():
    provider = build_llm_provider(make_config(llm_provider="none"))
    assert isinstance(provider, NullProvider)
    health = provider.health_check()
    assert health.available is False


def test_factory_unknown_provider_falls_back_to_null():
    provider = build_llm_provider(make_config(llm_provider="totally-unknown"))
    assert isinstance(provider, NullProvider)


def test_factory_selects_gemini():
    provider = build_llm_provider(make_config(llm_provider="gemini", gemini_api_key="k"))
    assert isinstance(provider, GeminiProvider)


def test_factory_selects_ollama():
    provider = build_llm_provider(make_config(llm_provider="ollama"))
    assert isinstance(provider, OllamaProvider)


def test_null_provider_generate_raises_llm_error():
    provider = NullProvider()
    with pytest.raises(LLMError):
        provider.generate(messages=[])
