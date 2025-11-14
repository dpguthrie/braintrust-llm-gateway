from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str | None = Field(None, description="Content of the message")
    name: str | None = Field(None, description="Optional name of the sender")
    tool_calls: list[dict[str, Any]] | None = Field(None, description="Tool calls made by assistant")
    tool_call_id: str | None = Field(None, description="ID of tool call this message responds to")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields for provider-specific params

    model: str = Field(..., description="Model to use (e.g., gpt-4, claude-3-5-sonnet-20241022)")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")

    # Optional parameters
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens to generate")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stream: bool = Field(False, description="Whether to stream the response")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    n: int | None = Field(None, gt=0, description="Number of completions to generate")
    user: str | None = Field(None, description="User identifier")

    # Tool/function calling
    tools: list[dict[str, Any]] | None = Field(None, description="Available tools")
    tool_choice: str | dict[str, Any] | None = Field(None, description="Tool choice strategy")
