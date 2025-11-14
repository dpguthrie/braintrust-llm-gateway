import os
from contextlib import asynccontextmanager
from typing import Any

import braintrust
from anthropic import AsyncAnthropic
from braintrust import start_span
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from llm_gateway.auth import verify_bearer_token
from llm_gateway.config import settings
from llm_gateway.models import ChatCompletionRequest


# Global wrapped clients (initialized at startup)
# Note: These are Braintrust-wrapped clients, not the original client types
openai_client: Any = None
anthropic_client: Any = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: initialize Braintrust and wrapped LLM clients."""
    global openai_client, anthropic_client

    # Set Braintrust API key
    os.environ["BRAINTRUST_API_KEY"] = settings.braintrust_api_key

    # Initialize Braintrust logger for the gateway project
    logger = braintrust.init_logger(project=settings.project_name)

    # Initialize wrapped LLM clients for automatic tracing
    if settings.openai_api_key:
        openai_client = braintrust.wrap_openai(
            AsyncOpenAI(api_key=settings.openai_api_key)
        )

    if settings.anthropic_api_key:
        anthropic_client = braintrust.wrap_anthropic(
            AsyncAnthropic(api_key=settings.anthropic_api_key)
        )

    yield

    # Cleanup
    logger.flush()


app = FastAPI(
    title="LLM Gateway",
    description="Demo LLM gateway with distributed Braintrust tracing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def _route_model(model: str) -> str:
    """Determine provider from model name."""
    if model.startswith(("gpt-", "o1-", "text-")):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    else:
        # Default to OpenAI for unknown models
        return "openai"


@app.post("/v1/chat/completions")
async def chat_completions(
    req: Request,
    request: ChatCompletionRequest,
    _: None = Depends(verify_bearer_token),
    x_braintrust_parent: str | None = Header(None, alias="x-braintrust-parent"),
):
    """
    OpenAI-compatible chat completions endpoint with distributed tracing.

    Supports parent span propagation via x-braintrust-parent header.
    Creates child spans linked to client's trace hierarchy.
    """
    # Determine provider
    provider = _route_model(request.model)

    # Create gateway span with optional parent linkage
    with start_span(
        name="llm-gateway-request",
        parent=x_braintrust_parent,  # Links to client's trace
        span_attributes={
            "type": "gateway",
            "provider": provider,
            "model": request.model,
            "stream": request.stream,
            "client_ip": req.client.host if req.client else "unknown",
        },
    ) as gateway_span:
        try:
            if provider == "openai":
                if not openai_client:
                    raise HTTPException(
                        status_code=503,
                        detail="OpenAI client not configured. Set OPENAI_API_KEY.",
                    )

                # Convert request to OpenAI params
                params = request.model_dump(exclude_none=True)

                if request.stream:
                    # Streaming response
                    response = await openai_client.chat.completions.create(**params)

                    async def generate():
                        async for chunk in response:
                            # Format as SSE
                            yield f"data: {chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        generate(),
                        media_type="text/event-stream",
                    )
                else:
                    # Non-streaming response
                    response = await openai_client.chat.completions.create(**params)

                    # Export span context for client
                    span_context = gateway_span.export()

                    return {
                        **response.model_dump(),
                        "braintrust_span": span_context,
                    }

            elif provider == "anthropic":
                if not anthropic_client:
                    raise HTTPException(
                        status_code=503,
                        detail="Anthropic client not configured. Set ANTHROPIC_API_KEY.",
                    )

                # Convert OpenAI format to Anthropic format
                anthropic_params: dict[str, Any] = {
                    "model": request.model,
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    "max_tokens": request.max_tokens or 4096,
                }

                if request.temperature is not None:
                    anthropic_params["temperature"] = request.temperature
                if request.top_p is not None:
                    anthropic_params["top_p"] = request.top_p
                if request.stream:
                    anthropic_params["stream"] = True

                if request.stream:
                    # Streaming response
                    async with anthropic_client.messages.stream(
                        **anthropic_params
                    ) as stream:

                        async def generate():
                            async for chunk in stream.text_stream:
                                # Convert Anthropic format to OpenAI SSE format
                                sse_chunk = (
                                    f'{{"object": "chat.completion.chunk", '
                                    f'"choices": [{{"delta": {{"content": "{chunk}"}}, "index": 0}}]}}'
                                )
                                yield f"data: {sse_chunk}\n\n"
                            yield "data: [DONE]\n\n"

                        return StreamingResponse(
                            generate(),
                            media_type="text/event-stream",
                        )
                else:
                    # Non-streaming response
                    response = await anthropic_client.messages.create(**anthropic_params)

                    # Convert Anthropic response to OpenAI format
                    openai_format = {
                        "id": response.id,
                        "object": "chat.completion",
                        "created": int(response.model_dump().get("created_at", 0)),
                        "model": response.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.content[0].text
                                    if response.content
                                    else "",
                                },
                                "finish_reason": response.stop_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": response.usage.input_tokens,
                            "completion_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.input_tokens
                            + response.usage.output_tokens,
                        },
                    }

                    # Export span context for client
                    span_context = gateway_span.export()

                    return {
                        **openai_format,
                        "braintrust_span": span_context,
                    }

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported provider: {provider}",
                )

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log error and raise
            gateway_span.log(error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
