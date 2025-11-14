import pytest
from pydantic import ValidationError

from llm_gateway.config import Settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before each test."""
    from llm_gateway import config

    config._get_settings.cache_clear()
    yield
    config._get_settings.cache_clear()


def test_settings_requires_braintrust_api_key(monkeypatch):
    """Test that missing BRAINTRUST_API_KEY raises validation error."""
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_AUTH_TOKEN", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # Don't load from .env file


def test_settings_requires_gateway_auth_token(monkeypatch):
    """Test that missing GATEWAY_AUTH_TOKEN raises validation error."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")
    monkeypatch.delenv("GATEWAY_AUTH_TOKEN", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # Don't load from .env file


def test_settings_loads_successfully_with_required_vars(monkeypatch):
    """Test that settings load with required env vars."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test-key")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "token1,token2")

    settings = Settings()

    assert settings.braintrust_api_key == "bt-test-key"
    assert settings.gateway_auth_tokens == ["token1", "token2"]


def test_settings_parses_comma_separated_tokens(monkeypatch):
    """Test that comma-separated tokens are parsed into list."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "token1,token2,token3")

    settings = Settings()

    assert len(settings.gateway_auth_tokens) == 3
    assert "token1" in settings.gateway_auth_tokens
    assert "token2" in settings.gateway_auth_tokens
    assert "token3" in settings.gateway_auth_tokens


def test_settings_loads_optional_provider_keys(monkeypatch):
    """Test that optional provider API keys are loaded."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    settings = Settings()

    assert settings.openai_api_key == "sk-openai-test"
    assert settings.anthropic_api_key == "sk-ant-test"
