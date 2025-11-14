from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required settings
    braintrust_api_key: str = Field(..., description="Braintrust API key for tracing")
    gateway_auth_token: str = Field(..., description="Comma-separated valid bearer tokens")

    # Optional provider API keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")

    # Gateway settings
    project_name: str = Field(default="llm-gateway", description="Braintrust project name")

    @property
    def gateway_auth_tokens(self) -> list[str]:
        """Parse comma-separated tokens into list."""
        return [token.strip() for token in self.gateway_auth_token.split(",") if token.strip()]


@lru_cache
def _get_settings() -> Settings:
    """Get cached settings instance. Lazily initialized."""
    return Settings()


def __getattr__(name: str):
    """Lazy initialization of module-level settings."""
    if name == "settings":
        return _get_settings()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
