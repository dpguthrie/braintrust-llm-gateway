import os
from contextlib import asynccontextmanager, nullcontext
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

# Global components (initialized at startup)
# gateway_logger: For platform analytics (logs to gateway project)
# wrapped clients: Auto-log LLM calls when within active spans
gateway_logger: Any = None
openai_client: Any = None
anthropic_client: Any = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: initialize Braintrust and wrapped LLM clients."""
    global gateway_logger, openai_client, anthropic_client

    # Set Braintrust API key
    os.environ["BRAINTRUST_API_KEY"] = settings.braintrust_api_key

    # Initialize gateway logger for platform analytics
    gateway_logger = braintrust.init_logger(project=settings.project_name)

    # Initialize wrapped LLM clients for automatic tracing
    if settings.openai_api_key:
        openai_client = braintrust.wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))

    if settings.anthropic_api_key:
        anthropic_client = braintrust.wrap_anthropic(AsyncAnthropic(api_key=settings.anthropic_api_key))

    yield

    # Cleanup
    gateway_logger.flush()


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


def _log_to_gateway(
    request: ChatCompletionRequest,
    provider: str,
    client_ip: str,
    x_braintrust_parent: str | None,
    output: str | None = None,
) -> None:
    """Log request to gateway for platform analytics."""
    gateway_logger.log(
        input=[msg.model_dump() for msg in request.messages],
        output=output,
        metadata={
            "model": request.model,
            "provider": provider,
            "client_ip": client_ip,
            "has_client_trace": x_braintrust_parent is not None,
            "stream": request.stream,
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    req: Request,
    request: ChatCompletionRequest,
    _: None = Depends(verify_bearer_token),
    x_braintrust_parent: str | None = Header(None, alias="x-braintrust-parent"),
):
    """
    OpenAI-compatible chat completions endpoint with dual logging:
    1. Conditional client distributed tracing (when x-braintrust-parent provided)
    2. Always log to gateway for platform analytics
    """
    # Determine provider
    provider = _route_model(request.model)

    # Conditionally create client span for distributed tracing
    # Uses nullcontext() when no parent to avoid code duplication
    span_ctx = (
        start_span(name="llm-gateway-request", parent=x_braintrust_parent) if x_braintrust_parent else nullcontext()
    )

    with span_ctx as client_span:
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

                    # Log to gateway (input only for streaming)
                    _log_to_gateway(
                        request=request,
                        provider=provider,
                        client_ip=req.client.host if req.client else "unknown",
                        x_braintrust_parent=x_braintrust_parent,
                    )

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

                    # Log to gateway for platform analytics
                    _log_to_gateway(
                        request=request,
                        provider=provider,
                        client_ip=req.client.host if req.client else "unknown",
                        x_braintrust_parent=x_braintrust_parent,
                        output=response.choices[0].message.content if response.choices else None,
                    )

                    # Build response with optional span context
                    result = response.model_dump()
                    if client_span:
                        result["braintrust_span"] = client_span.export()

                    return result

            elif provider == "anthropic":
                if not anthropic_client:
                    raise HTTPException(
                        status_code=503,
                        detail="Anthropic client not configured. Set ANTHROPIC_API_KEY.",
                    )

                # Convert OpenAI format to Anthropic format
                anthropic_params: dict[str, Any] = {
                    "model": request.model,
                    "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
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
                    async with anthropic_client.messages.stream(**anthropic_params) as stream:
                        # Log to gateway (input only for streaming)
                        _log_to_gateway(
                            request=request,
                            provider=provider,
                            client_ip=req.client.host if req.client else "unknown",
                            x_braintrust_parent=x_braintrust_parent,
                        )

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
                                    "content": response.content[0].text if response.content else "",
                                },
                                "finish_reason": response.stop_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": response.usage.input_tokens,
                            "completion_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                        },
                    }

                    # Log to gateway for platform analytics
                    _log_to_gateway(
                        request=request,
                        provider=provider,
                        client_ip=req.client.host if req.client else "unknown",
                        x_braintrust_parent=x_braintrust_parent,
                        output=response.content[0].text if response.content else None,
                    )

                    # Build response with optional span context
                    result = openai_format
                    if client_span:
                        result["braintrust_span"] = client_span.export()

                    return result

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported provider: {provider}",
                )

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log error if client span exists
            if client_span:
                client_span.log(error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
