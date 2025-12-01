"""
Integration tests for the LLM Gateway.

These tests verify the full request flow but mock OpenAI/Anthropic responses.
For real API testing, set provider API keys and use pytest markers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before each test."""
    from llm_gateway import config

    config._get_settings.cache_clear()
    yield
    config._get_settings.cache_clear()


@pytest.fixture
def test_env(monkeypatch):
    """Set up test environment."""
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-bt-key")
    monkeypatch.setenv("GATEWAY_AUTH_TOKEN", "integration-test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


@pytest.fixture
def client(test_env):
    """Create test client with mocked Braintrust components."""
    # Mock the Braintrust init_logger and wrap_openai
    with (
        patch("llm_gateway.main.braintrust.init_logger") as mock_init_logger,
        patch("llm_gateway.main.braintrust.wrap_openai") as mock_wrap,
    ):
        # Create a mock logger
        mock_logger = MagicMock()
        mock_logger.flush = MagicMock()
        mock_logger.log = MagicMock()
        mock_init_logger.return_value = mock_logger

        # Create a mock client
        mock_client = MagicMock()
        mock_wrap.return_value = mock_client

        from llm_gateway.main import app

        yield TestClient(app)


def test_full_request_flow_non_streaming(client):
    """Test complete request flow for non-streaming."""
    import llm_gateway.main

    # Create a more complete mock response
    mock_message = MagicMock()
    mock_message.content = "Hello! How can I help?"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    # Mock the gateway_logger and openai_client
    llm_gateway.main.gateway_logger = MagicMock()
    llm_gateway.main.gateway_logger.log = MagicMock()

    llm_gateway.main.openai_client = MagicMock()
    llm_gateway.main.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer integration-test-token"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    if response.status_code != 200:
        print(f"Error response: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o-mini"
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"
    assert data["usage"]["total_tokens"] == 18
    # No braintrust_span expected since no x-braintrust-parent header was provided
    assert "braintrust_span" not in data


def test_request_with_optional_parameters(client):
    """Test that optional parameters are passed through."""
    import llm_gateway.main

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"choices": [{"message": {"content": "Response"}}]}

    # Mock the openai_client
    mock_create = AsyncMock(return_value=mock_response)
    llm_gateway.main.openai_client = MagicMock()
    llm_gateway.main.openai_client.chat.completions.create = mock_create

    client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer integration-test-token"},
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        },
    )

    # Verify OpenAI client was called with parameters
    assert mock_create.called
    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["top_p"] == 0.9


def test_error_handling(client):
    """Test that errors from OpenAI client are handled."""
    import llm_gateway.main

    # Mock the openai_client to raise an error
    llm_gateway.main.openai_client = MagicMock()
    llm_gateway.main.openai_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer integration-test-token"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 500
    assert "API Error" in response.json()["detail"]
