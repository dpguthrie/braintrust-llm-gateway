"""Modal deployment configuration for LLM Gateway."""

import modal

# Create Modal app
app = modal.App("llm-gateway")

# Define container image with dependencies from pyproject.toml
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync()  # Uses pyproject.toml and uv.lock for dependencies
    .add_local_python_source("llm_gateway", "src/llm_gateway")  # Add source code
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_dotenv(__file__),  # Load secrets from .env file
    ],
)
@modal.asgi_app()
def fastapi_app():
    """Deploy FastAPI application."""
    from llm_gateway.main import app

    return app
