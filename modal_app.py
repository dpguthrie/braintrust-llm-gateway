"""Modal deployment configuration for LLM Gateway."""

import modal

# Create Modal app
app = modal.App("llm-gateway")

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "litellm>=1.0.0",
        "braintrust>=0.0.1",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-multipart>=0.0.6",
    )
    .copy_local_dir("src/llm_gateway", "/root/llm_gateway")
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("llm-gateway-secrets"),
    ],
)
@modal.asgi_app()
def fastapi_app():
    """Deploy FastAPI application."""
    from llm_gateway.main import app

    return app
