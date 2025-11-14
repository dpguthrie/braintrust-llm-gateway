import os
from contextlib import asynccontextmanager

import braintrust
import litellm
from braintrust.wrappers.litellm import patch_litellm
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llm_gateway.auth import verify_bearer_token
from llm_gateway.config import settings
from llm_gateway.models import ChatCompletionRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize Braintrust tracing."""
    # Set Braintrust API key
    os.environ["BRAINTRUST_API_KEY"] = settings.braintrust_api_key

    # Initialize Braintrust logger
    logger = braintrust.init_logger(project=settings.project_name)

    # Patch LiteLLM for automatic tracing
    patch_litellm()

    yield

    # Cleanup
    logger.flush()


app = FastAPI(
    title="LLM Gateway",
    description="Demo LLM gateway with automatic Braintrust tracing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: None = Depends(verify_bearer_token),
):
    """
    OpenAI-compatible chat completions endpoint.

    Automatically traced by Braintrust via patch_litellm().
    """
    try:
        # Convert request to dict for LiteLLM
        params = request.model_dump(exclude_none=True)

        # Call LiteLLM (automatically traced by Braintrust)
        if request.stream:
            # Streaming response
            response = await litellm.acompletion(**params)

            async def generate():
                async for chunk in response:
                    # Format as SSE
                    yield f"data: {chunk.model_dump_json()}\n\n"  # type: ignore[attr-defined]
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response = await litellm.acompletion(**params)
            return response.model_dump()  # type: ignore[attr-defined]

    except Exception as e:
        # Pass through LiteLLM errors
        raise HTTPException(status_code=500, detail=str(e))
