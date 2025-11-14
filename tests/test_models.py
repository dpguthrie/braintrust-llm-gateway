import pytest
from pydantic import ValidationError

from llm_gateway.models import ChatCompletionRequest, ChatMessage


def test_chat_message_requires_role_and_content():
    """Test that ChatMessage requires role and content."""
    with pytest.raises(ValidationError):
        ChatMessage()


def test_chat_message_validates_successfully():
    """Test that valid ChatMessage is created."""
    message = ChatMessage(role="user", content="Hello")

    assert message.role == "user"
    assert message.content == "Hello"


def test_chat_completion_request_requires_model_and_messages():
    """Test that ChatCompletionRequest requires model and messages."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest()


def test_chat_completion_request_validates_successfully():
    """Test that valid ChatCompletionRequest is created."""
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert request.model == "gpt-4o-mini"
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"


def test_chat_completion_request_supports_optional_params():
    """Test that optional parameters are supported."""
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    assert request.temperature == 0.7
    assert request.max_tokens == 100
    assert request.stream is True


def test_chat_completion_request_converts_to_dict():
    """Test that request can be converted to dict for LiteLLM."""
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
    )

    data = request.model_dump(exclude_none=True)

    assert data["model"] == "gpt-4"
    assert len(data["messages"]) == 1
    assert data["temperature"] == 0.5
    assert "max_tokens" not in data  # None values excluded
