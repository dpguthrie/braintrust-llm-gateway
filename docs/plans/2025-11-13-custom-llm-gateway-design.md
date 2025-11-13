# Custom LLM Gateway with Braintrust Tracing - Design Document

**Date:** 2025-11-13
**Status:** Approved
**Purpose:** Demo-quality LLM gateway with automatic Braintrust tracing instrumentation

## Overview

A custom LLM gateway built in Python that automatically instruments all LLM requests with Braintrust tracing. Clients using this gateway get observability out-of-the-box without any code changes. The gateway is designed to be demoable to clients showing how to implement similar functionality in their own infrastructure.

## Goals

- **Automatic Tracing**: Every LLM request automatically logged to Braintrust
- **Multi-Provider Support**: OpenAI, Anthropic, and other major LLM providers
- **OpenAI Compatibility**: Standard OpenAI SDK format for easy client integration
- **Simple Authentication**: Bearer token validation
- **Demo Quality**: Clean, readable code prioritizing simplicity over production hardening
- **Easy Deployment**: Deploy to Modal with minimal configuration

## Non-Goals

- Production-grade rate limiting or quota management
- Database-backed key management
- Advanced authentication (OAuth, JWT validation, etc.)
- Custom retry logic (rely on LiteLLM defaults)
- Multi-tenancy or user management

## Architecture

### High-Level Design

```
Client (any OpenAI-compatible SDK)
    |
    | HTTP POST /v1/chat/completions
    | Authorization: Bearer <token>
    |
    v
FastAPI Gateway (Modal)
    |
    ├─> Auth Middleware (validates token)
    |
    ├─> Braintrust Auto-Tracing (via patch_litellm)
    |
    └─> LiteLLM SDK
            |
            ├─> OpenAI API
            ├─> Anthropic API
            └─> Other Providers
```

### Key Architectural Decision: Braintrust's LiteLLM Integration

Braintrust provides `braintrust.patch_litellm()` which automatically wraps all LiteLLM calls with tracing. This eliminates the need for custom tracing middleware - all instrumentation is automatic.

**Tracing Flow:**
1. App startup: `braintrust.init_logger(project="llm-gateway")` + `braintrust.patch_litellm()`
2. Each request: LiteLLM calls automatically create Braintrust spans
3. Automatic capture of: inputs, outputs, tokens, latency, time-to-first-token, model metadata
4. No manual span management required

### Technology Stack

- **Web Framework**: FastAPI (async support, OpenAPI docs)
- **LLM Routing**: LiteLLM (unified interface to 100+ providers)
- **Observability**: Braintrust Python SDK
- **Deployment**: Modal (serverless, easy deployment)
- **Validation**: Pydantic v2

## Components

### 1. Main Application (`main.py`)

**Responsibilities:**
- FastAPI app initialization
- Braintrust logger setup and LiteLLM patching
- Modal deployment configuration
- Single endpoint: `POST /v1/chat/completions`

**Key Code Pattern:**
```python
import braintrust
import litellm

# Initialize Braintrust tracing
logger = braintrust.init_logger(project="llm-gateway")
braintrust.patch_litellm()

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    response = await litellm.acompletion(**request.model_dump())
    return response
```

### 2. Authentication Middleware (`auth.py`)

**Responsibilities:**
- Bearer token extraction and validation
- FastAPI dependency for route protection

**Implementation:**
- Extract `Authorization: Bearer <token>` header
- Validate against comma-separated `GATEWAY_AUTH_TOKEN` env var
- Return 401 for invalid/missing tokens

### 3. Configuration (`config.py`)

**Responsibilities:**
- Load and validate environment variables
- Expose configuration to app

**Environment Variables:**
- `BRAINTRUST_API_KEY` - Braintrust API key (required)
- `OPENAI_API_KEY` - OpenAI API key (optional)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)
- `GATEWAY_AUTH_TOKEN` - Comma-separated list of valid bearer tokens (required)
- Additional provider keys as needed

**Implementation:** Pydantic Settings for validation and type safety

### 4. Request/Response Models (`models.py`)

**Responsibilities:**
- Define OpenAI-compatible request/response schemas
- Validate incoming requests

**Key Models:**
- `ChatCompletionRequest` - messages, model, temperature, stream, etc.
- Pass-through to LiteLLM (no custom response models needed)

### 5. Modal Deployment (`modal_app.py` or integrated in `main.py`)

**Responsibilities:**
- Modal function and image configuration
- Secret management
- ASGI app exposure

**Key Features:**
- Modal Secrets for API keys
- Container image with Python dependencies
- Public web endpoint or custom domain
- Auto-scaling

## Data Flow

### Non-Streaming Request

1. Client sends POST to `/v1/chat/completions` with bearer token
2. Auth middleware validates token
3. Pydantic validates request body
4. `litellm.acompletion()` called (automatically traced by Braintrust)
5. LiteLLM routes to appropriate provider
6. Response returned to client
7. Braintrust span automatically logged with full trace data

### Streaming Request

1. Same as non-streaming through step 4
2. `litellm.acompletion(stream=True)` returns async generator
3. SSE stream sent to client
4. Braintrust automatically captures:
   - Time to first token
   - Full streamed response (accumulated)
   - Token counts
5. Stream completes, span closed automatically

## Error Handling

### Authentication Errors
- **Missing header**: 401 `{"error": "Missing authorization header"}`
- **Invalid token**: 401 `{"error": "Invalid token"}`
- **Malformed header**: 401 `{"error": "Malformed authorization header"}`

### Provider Errors
- **Missing API keys**: Fail fast at startup with clear error message
- **Invalid model**: Pass through LiteLLM error (400/404)
- **Rate limits**: Pass through provider 429 error
- **Provider outages**: Pass through with appropriate error codes

### Request Validation
- **Invalid JSON**: 422 with Pydantic validation details
- **Missing fields**: 422 with field-level errors
- **Invalid parameters**: LiteLLM validation errors

### Streaming Errors
- **Connection drops**: LiteLLM handles cleanup
- **Mid-stream errors**: Error event in SSE stream

## Observability

### Automatic Braintrust Tracing

Every request creates a Braintrust span with:
- **Input**: Full message array
- **Output**: Completion response
- **Metrics**:
  - `tokens` - Total tokens used
  - `prompt_tokens` - Input tokens
  - `completion_tokens` - Output tokens
  - `time_to_first_token` - Latency to first token (streaming)
- **Metadata**:
  - `model` - Model name
  - `provider` - Provider (openai, anthropic, etc.)
  - Request parameters (temperature, top_p, etc.)

### Error Tracing

Failed requests still create spans with:
- Error flag set
- Error message captured
- Partial response data if available

### Optional Python Logging

Standard Python logging for debugging:
- Request received
- Auth success/failure
- Provider calls
- Response sent

## Deployment (Modal)

### Configuration

**Secrets:**
- Create Modal secret with all API keys
- Reference in `@app.function()` decorator

**Image:**
- Python 3.11+
- Dependencies: fastapi, litellm, braintrust, pydantic, pydantic-settings

**Function Setup:**
```python
@app.function(
    image=modal.Image.debian_slim()
        .pip_install("fastapi", "litellm", "braintrust", "pydantic", "pydantic-settings"),
    secrets=[modal.Secret.from_name("llm-gateway-keys")]
)
@modal.asgi_app()
def fastapi_app():
    return app
```

### Deployment Steps

1. Install Modal CLI: `pip install modal`
2. Authenticate: `modal setup`
3. Create secrets in Modal dashboard
4. Deploy: `modal deploy modal_app.py`
5. Get public URL from deployment output

### Scaling

- Auto-scales based on request volume
- Cold start: ~2-3 seconds
- Warm instances handle requests immediately

## Security Considerations

### Demo-Quality Security

- **Bearer token auth**: Simple but sufficient for demos
- **HTTPS**: Provided by Modal automatically
- **API keys in secrets**: Never in code
- **No key rotation**: Acceptable for demo

### What's Missing (Intentionally)

- Rate limiting per key
- Request logging to persistent storage
- Advanced authentication
- DDoS protection
- Input sanitization (rely on LiteLLM)

## Testing Strategy

### Local Development

1. Set environment variables
2. Run with `uvicorn main:app --reload`
3. Test with curl or Postman
4. Verify traces in Braintrust dashboard

### Integration Testing

- Test OpenAI-compatible clients (Python SDK, Node SDK)
- Verify streaming responses
- Test error conditions
- Confirm Braintrust traces are created

### Deployment Testing

- Deploy to Modal
- Test from external clients
- Verify cold start behavior
- Check Braintrust logging in production

## Dependencies

```
fastapi>=0.104.0
litellm>=1.0.0
braintrust>=0.0.1
pydantic>=2.0.0
pydantic-settings>=2.0.0
modal>=0.57.0  # For deployment
uvicorn>=0.24.0  # For local development
```

## Future Enhancements (Out of Scope)

- Database-backed API key management
- Per-key rate limiting and quotas
- Cost tracking dashboard
- Request caching
- Model fallback/retry logic
- Custom prompt templates
- Request/response transformation
- Webhook notifications

## Success Metrics

This is a demo, so success is measured by:
- **Clarity**: Code is readable and educational
- **Completeness**: Demonstrates key concepts (auth, tracing, routing)
- **Functionality**: Actually works for demo scenarios
- **Simplicity**: Easy to understand and replicate
- **Observability**: Braintrust traces provide valuable insights

## Conclusion

This design prioritizes simplicity and educational value over production robustness. By leveraging Braintrust's built-in LiteLLM integration, we eliminate custom tracing code and keep the implementation clean. The result is a working gateway that effectively demonstrates how to add observability to LLM applications.
