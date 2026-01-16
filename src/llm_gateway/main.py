import json
import os
import time
from contextlib import asynccontextmanager, nullcontext
from json import JSONDecodeError
from typing import Any
from urllib.parse import unquote

import braintrust
from anthropic import AsyncAnthropic
from braintrust import start_span
from braintrust.oai import wrap_openai
from braintrust.wrappers.anthropic import wrap_anthropic
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


def _extract_bt_parent_from_baggage(baggage: str | None) -> str | None:
    if not baggage:
        return None

    # W3C Baggage format: comma-separated list of key=value pairs
    # Example: baggage: braintrust.parent=project_name:my-project,foo=bar
    for item in baggage.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            continue

        k, v = item.split("=", 1)
        if k.strip() != "braintrust.parent":
            continue

        # Baggage values may include properties (e.g. "k=v;prop=..."); we only care about the value.
        value = unquote(v.strip()).split(";", 1)[0].strip()
        return value or None

    return None


def _parse_span_metadata(x_bt_span_metadata: str | None) -> dict[str, Any] | None:
    if not x_bt_span_metadata:
        return None
    try:
        parsed = json.loads(x_bt_span_metadata)
    except JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid x-bt-span-metadata JSON: {e}"
        ) from e

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400, detail="x-bt-span-metadata must be a JSON object"
        )

    return parsed


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
        openai_client = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))

    if settings.anthropic_api_key:
        anthropic_client = wrap_anthropic(
            AsyncAnthropic(api_key=settings.anthropic_api_key)
        )

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
    bt_parent: str | None,
    span_metadata: dict[str, Any] | None,
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
            "has_client_trace": bt_parent is not None,
            "stream": request.stream,
            "span_metadata": span_metadata,
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    req: Request,
    request: ChatCompletionRequest,
    _: None = Depends(verify_bearer_token),
    x_bt_parent: str | None = Header(None, alias="x-bt-parent"),
    x_bt_span_metadata: str | None = Header(None, alias="x-bt-span-metadata"),
    baggage: str | None = Header(None, alias="baggage"),
):
    """
    OpenAI-compatible chat completions endpoint with dual logging:
    1. Conditional client distributed tracing (when x-bt-parent provided, or W3C baggage has braintrust.parent)
    2. Always log to gateway for platform analytics
    """
    # Determine provider
    provider = _route_model(request.model)

    span_metadata = _parse_span_metadata(x_bt_span_metadata)

    bt_parent = x_bt_parent
    if bt_parent is None:
        # Optional fallback: if a client propagates `braintrust.parent` via the standard `baggage`
        # header, we can use it as the parent without requiring a custom gateway header.
        bt_parent = _extract_bt_parent_from_baggage(baggage)

    # Conditionally create client span for distributed tracing
    # Uses nullcontext() when no parent to avoid code duplication
    span_ctx = (
        start_span(
            name="llm-gateway-request",
            parent=bt_parent,
            metadata=span_metadata,
        )
        if bt_parent
        else nullcontext()
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
                    start_ts = time.time()
                    first_token_ts: float | None = None
                    output_chunks: list[str] = []
                    response = await openai_client.chat.completions.create(**params)

                    # Log to gateway (input only for streaming)
                    _log_to_gateway(
                        request=request,
                        provider=provider,
                        client_ip=req.client.host if req.client else "unknown",
                        bt_parent=bt_parent,
                        span_metadata=span_metadata,
                    )

                    async def generate():
                        nonlocal first_token_ts
                        async for chunk in response:
                            try:
                                content = (
                                    chunk.choices[0].delta.content
                                    if chunk.choices
                                    and chunk.choices[0].delta is not None
                                    else None
                                )
                            except Exception:
                                content = None

                            if content:
                                if first_token_ts is None:
                                    first_token_ts = time.time()
                                output_chunks.append(content)

                            # Format as SSE
                            yield f"data: {chunk.model_dump_json()}\n\n"

                        end_ts = time.time()
                        full_output = "".join(output_chunks) if output_chunks else None
                        metrics: dict[str, float] = {"start": start_ts, "end": end_ts}
                        if first_token_ts is not None:
                            metrics["time_to_first_token"] = first_token_ts - start_ts

                        if client_span:
                            client_span.log(output=full_output, metrics=metrics)

                        _log_to_gateway(
                            request=request,
                            provider=provider,
                            client_ip=req.client.host if req.client else "unknown",
                            bt_parent=bt_parent,
                            span_metadata=span_metadata,
                            output=full_output,
                        )

                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        generate(),
                        media_type="text/event-stream",
                    )
                else:
                    # Non-streaming response
                    start_ts = time.time()
                    response = await openai_client.chat.completions.create(**params)
                    end_ts = time.time()

                    # Log to gateway for platform analytics
                    _log_to_gateway(
                        request=request,
                        provider=provider,
                        client_ip=req.client.host if req.client else "unknown",
                        bt_parent=bt_parent,
                        span_metadata=span_metadata,
                        output=response.choices[0].message.content
                        if response.choices
                        else None,
                    )

                    # Build response with optional span context
                    result = response.model_dump()
                    if client_span:
                        client_span.log(metrics={"start": start_ts, "end": end_ts})
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
                    start_ts = time.time()
                    first_token_ts: float | None = None
                    output_chunks: list[str] = []
                    async with anthropic_client.messages.stream(
                        **anthropic_params
                    ) as stream:
                        # Log to gateway (input only for streaming)
                        _log_to_gateway(
                            request=request,
                            provider=provider,
                            client_ip=req.client.host if req.client else "unknown",
                            bt_parent=bt_parent,
                            span_metadata=span_metadata,
                        )

                        async def generate():
                            nonlocal first_token_ts
                            async for chunk in stream.text_stream:
                                if chunk:
                                    if first_token_ts is None:
                                        first_token_ts = time.time()
                                    output_chunks.append(chunk)
                                # Convert Anthropic format to OpenAI SSE format
                                sse_chunk = {
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {"delta": {"content": chunk}, "index": 0}
                                    ],
                                }
                                yield f"data: {json.dumps(sse_chunk)}\n\n"

                            end_ts = time.time()
                            full_output = (
                                "".join(output_chunks) if output_chunks else None
                            )
                            metrics: dict[str, float] = {
                                "start": start_ts,
                                "end": end_ts,
                            }
                            if first_token_ts is not None:
                                metrics["time_to_first_token"] = (
                                    first_token_ts - start_ts
                                )

                            if client_span:
                                client_span.log(output=full_output, metrics=metrics)

                            _log_to_gateway(
                                request=request,
                                provider=provider,
                                client_ip=req.client.host if req.client else "unknown",
                                bt_parent=bt_parent,
                                span_metadata=span_metadata,
                                output=full_output,
                            )
                            yield "data: [DONE]\n\n"

                        return StreamingResponse(
                            generate(),
                            media_type="text/event-stream",
                        )
                else:
                    # Non-streaming response
                    start_ts = time.time()
                    response = await anthropic_client.messages.create(
                        **anthropic_params
                    )
                    end_ts = time.time()

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

                    # Log to gateway for platform analytics
                    _log_to_gateway(
                        request=request,
                        provider=provider,
                        client_ip=req.client.host if req.client else "unknown",
                        bt_parent=bt_parent,
                        span_metadata=span_metadata,
                        output=response.content[0].text if response.content else None,
                    )

                    # Build response with optional span context
                    result = openai_format
                    if client_span:
                        client_span.log(metrics={"start": start_ts, "end": end_ts})
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
