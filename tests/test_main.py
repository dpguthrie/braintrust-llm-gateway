import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-bt-key")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


@pytest.fixture
def client(test_env):
    """Create test client."""
    # Import after env vars are set
    from llm_gateway.main import app

    return TestClient(app)


def test_health_endpoint_returns_200(client):
    """Test that health endpoint returns 200."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_completions_requires_auth(client):
    """Test that chat completions endpoint requires authentication."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 401


def test_chat_completions_rejects_invalid_token(client):
    """Test that invalid token is rejected."""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer invalid-token"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 401


def test_chat_completions_validates_request_body(client):
    """Test that request body is validated."""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-token"},
        json={"model": "gpt-4o-mini"},  # Missing messages
    )

    assert response.status_code == 422  # Validation error
