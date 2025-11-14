import pytest
from fastapi import HTTPException
from llm_gateway.auth import verify_bearer_token
from llm_gateway.config import Settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before each test."""
    from llm_gateway import config

    config._get_settings.cache_clear()
    yield
    config._get_settings.cache_clear()


def test_verify_bearer_token_accepts_valid_token(monkeypatch):
    """Test that valid bearer token is accepted."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "valid-token")

    # Should not raise exception
    verify_bearer_token("Bearer valid-token")


def test_verify_bearer_token_accepts_any_valid_token_from_list(monkeypatch):
    """Test that any token from comma-separated list is accepted."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "token1,token2,token3")

    verify_bearer_token("Bearer token1")
    verify_bearer_token("Bearer token2")
    verify_bearer_token("Bearer token3")


def test_verify_bearer_token_rejects_invalid_token(monkeypatch):
    """Test that invalid bearer token raises 401."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "valid-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_bearer_token("Bearer invalid-token")

    assert exc_info.value.status_code == 401
    assert "Invalid token" in exc_info.value.detail


def test_verify_bearer_token_rejects_missing_bearer_prefix(monkeypatch):
    """Test that token without 'Bearer ' prefix raises 401."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "valid-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_bearer_token("valid-token")

    assert exc_info.value.status_code == 401
    assert "Invalid authorization header format" in exc_info.value.detail


def test_verify_bearer_token_rejects_none(monkeypatch):
    """Test that None authorization raises 401."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "valid-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_bearer_token(None)

    assert exc_info.value.status_code == 401
    assert "Missing authorization header" in exc_info.value.detail


def test_verify_bearer_token_rejects_empty_string(monkeypatch):
    """Test that empty authorization string raises 401."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "valid-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_bearer_token("")

    assert exc_info.value.status_code == 401
