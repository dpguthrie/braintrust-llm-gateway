# LLM Gateway with Braintrust Tracing

Demo-quality LLM gateway that automatically instruments all requests with distributed Braintrust tracing.

## Features

- **Distributed Tracing**: Parent span propagation for complete trace hierarchy
- **Automatic Instrumentation**: Every LLM request automatically logged to Braintrust
- **Multi-Provider**: Support for OpenAI and Anthropic (easily extensible)
- **OpenAI Compatible**: Works with any OpenAI SDK client
- **Simple Auth**: Bearer token authentication
- **Streaming**: Full support for streaming responses
- **Modal Deployment**: Easy serverless deployment with uv

## Architecture

```
[Client App] --HTTP--> [LLM Gateway] --API--> [LLM Provider]
     |                      |                      |
 [BT Span]          [BT Child Span]         [Response]
```

The gateway creates child spans that link to client application spans, maintaining complete trace hierarchy from application → gateway → LLM provider.

## Local Development

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for package management
- Braintrust account and API key
- API keys for LLM providers (OpenAI, Anthropic, etc.)

### Setup

1. Clone and install dependencies:

```bash
git clone <repo-url>
cd custom-llm-gateway
uv pip install -e ".[dev]"
```

2. Create a `.env` file (this repo does not include a `.env.example`):

```bash
cat > .env <<'EOF'
BRAINTRUST_API_KEY=...
GATEWAY_AUTH_TOKEN=your-token-1,your-token-2
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
EOF
```

Required environment variables:
- `BRAINTRUST_API_KEY`: Your Braintrust API key
- `GATEWAY_AUTH_TOKEN`: Comma-separated bearer tokens for authentication
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `ANTHROPIC_API_KEY`: Anthropic API key (optional)

3. Run the server:

```bash
uvicorn src.llm_gateway.main:app --reload
```

The server will start at `http://localhost:8000`.

### Testing

Run tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=src/llm_gateway --cov-report=html
```

Format code:

```bash
ruff format .
```

Lint code:

```bash
ruff check .
```

## Usage

### Making Requests

The gateway exposes an OpenAI-compatible API at `/v1/chat/completions`.

**Using curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Using OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-token-here",  # Your gateway token
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

**Streaming:**

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Supported Models

The gateway automatically routes requests based on model name:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`, `o1-*`, etc.
- **Anthropic**: `claude-3-5-sonnet-*`, `claude-3-opus-*`, `claude-3-haiku-*`, etc.

### Distributed tracing

The gateway supports parent span propagation for distributed tracing. This allows client applications to link their spans with gateway spans, creating a complete trace hierarchy.

**Without parent span (basic usage):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-token-here",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
# Gateway creates independent span in Braintrust
```

**With parent span (distributed tracing via `x-bt-parent`):**

```python
import braintrust
from openai import OpenAI

# Initialize your application's tracing
experiment = braintrust.init(project="my-app")

with experiment.traced(name="my-app-task") as parent_span:
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="your-gateway-token",
    )

    # Pass parent span context via header (Braintrust standard)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        extra_headers={
            "x-bt-parent": parent_span.export(),
            # Optional JSON object (must be an object, not a string/array):
            "x-bt-span-metadata": '{"request_id":"req_123","customer":"acme"}',
        }
    )

# Creates trace hierarchy: my-app-task → gateway-request → openai-completion
```

The response includes a `braintrust_span` field with the gateway span context, allowing you to continue the trace chain if needed.

**With W3C baggage (recommended for Java/OTEL ecosystems):**

If your client and gateway are OpenTelemetry-instrumented, you can propagate the Braintrust parent via W3C baggage:

- Set `baggage: braintrust.parent=project_name:my-project` (or `experiment_id:...`)
- Also propagate `traceparent` for full distributed tracing across services

The Python gateway will read `braintrust.parent` from the `baggage` header and create a child Braintrust span accordingly.

**Example (curl)**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -H "baggage: braintrust.parent=project_name:my-project" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

If you also propagate `traceparent`, you’ll get full distributed tracing across services (client ↔ gateway) in any OTEL backend; Braintrust will still use `braintrust.parent` for routing spans to the right project/experiment.

**Trace Hierarchy Example:**

```
Root Span (Client: "process-document")
└── Child Span (Gateway: "llm-gateway-request")
    └── Grandchild Span (Provider: "openai.chat.completions.create")
        ├── input: {...}
        ├── output: {...}
        ├── metrics: {tokens: 150}
        └── latency: 1.2s
```

### Viewing Traces

All requests are automatically logged to Braintrust:

1. Go to [braintrust.dev](https://braintrust.dev)
2. Navigate to your project (default: "llm-gateway")
3. View traces with full request/response data, token counts, and latency
4. If using distributed tracing, see complete hierarchy from client → gateway → provider

## Modal Deployment

### Prerequisites

- Modal account ([modal.com](https://modal.com))
- Modal CLI: `pip install modal`
- `uv` package manager installed

### Setup

1. Authenticate with Modal:

```bash
modal setup
```

2. Create a `.env` file with your API keys:

```bash
cat > .env <<'EOF'
BRAINTRUST_API_KEY=...
GATEWAY_AUTH_TOKEN=your-token-1,your-token-2
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
EOF
```

Required variables:
```
BRAINTRUST_API_KEY=your-braintrust-key
GATEWAY_AUTH_TOKEN=your-token-1,your-token-2
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

3. Ensure you have a `uv.lock` file:

```bash
uv sync
```

4. Deploy:

```bash
modal deploy modal_app.py
```

The deployment uses:
- `uv_sync()` to install dependencies from `pyproject.toml` and `uv.lock`
- `Secret.from_dotenv()` to load environment variables from your `.env` file

5. Get your deployment URL from the output:

```
✓ Created web function fastapi_app => https://your-app-id--llm-gateway-fastapi-app.modal.run
```

### Using Deployed Gateway

Update your client to use the Modal URL:

```python
client = OpenAI(
    base_url="https://your-app-id--llm-gateway-fastapi-app.modal.run/v1",
    api_key="your-gateway-token",
)
```

## Project Structure

```
.
├── src/llm_gateway/
│   ├── __init__.py
│   ├── config.py          # Configuration with Pydantic
│   ├── auth.py            # Bearer token authentication
│   ├── models.py          # Request/response models
│   └── main.py            # FastAPI app + Braintrust tracing
├── tests/
│   ├── test_config.py
│   ├── test_auth.py
│   ├── test_models.py
│   └── test_main.py
├── docs/
│   └── plans/             # Design and implementation docs
├── modal_app.py           # Modal deployment config
├── pyproject.toml         # Dependencies and config
├── ruff.toml              # Linting/formatting config
└── README.md
```

## How It Works

### Distributed Tracing Architecture

The gateway uses Braintrust's provider-specific wrappers for automatic instrumentation with distributed tracing support:

1. **On startup**:
   - Initializes Braintrust logger for the gateway project
   - Wraps OpenAI and Anthropic clients with `braintrust.wrap_openai()` and `braintrust.wrap_anthropic()`

2. **On each request**:
   - Extracts optional `x-bt-parent` header (parent span from client), or W3C baggage `braintrust.parent`
   - Creates gateway span with `start_span()`, linking to parent if provided
   - Routes to appropriate provider based on model name
   - Wrapped client automatically creates child span under gateway span
   - Returns response with `braintrust_span` field for continued tracing

3. **Trace hierarchy created**:
   ```
   Client Span (optional)
   └── Gateway Span (llm-gateway-request)
       └── Provider Span (openai.chat.completions.create)
   ```

4. **Automatic capture**: inputs, outputs, tokens, latency, time-to-first-token, model metadata

The gateway acts as a transparent proxy while maintaining complete observability through distributed tracing.

### Authentication

Simple bearer token authentication:

1. Client sends `Authorization: Bearer <token>` header
2. Gateway validates token against `GATEWAY_AUTH_TOKEN` list
3. Invalid/missing tokens return 401

For production, consider:
- JWT tokens with expiration
- Per-key rate limiting
- Database-backed key management

## Demo Use Cases

This gateway is designed for demos showing:

1. **Distributed tracing in LLM apps**: Show complete trace hierarchy from client app through gateway to provider
2. **How to add observability to LLM apps**: Show Braintrust traces with automatic instrumentation
3. **Multi-provider routing**: Automatically route to OpenAI or Anthropic based on model name
4. **OpenAI SDK compatibility**: Drop-in replacement for OpenAI API
5. **Serverless deployment**: Deploy to Modal in minutes with uv

## Limitations (By Design)

This is demo-quality code, intentionally simple:

- ❌ No rate limiting per key
- ❌ No persistent key storage
- ❌ No advanced auth (OAuth, JWT validation)
- ❌ No caching
- ❌ No custom retry logic
- ❌ No cost tracking dashboard

For production use, add these features as needed.

## Troubleshooting

**Import errors:**
```bash
# Make sure you installed in editable mode
uv pip install -e ".[dev]"
```

**Braintrust not logging:**
- Check `BRAINTRUST_API_KEY` is set correctly
- Verify project name matches your Braintrust account
- Check Braintrust dashboard for API key permissions

**LiteLLM errors:**
- Verify provider API keys are set (OPENAI_API_KEY, etc.)
- Check model name is correct for the provider
- See [LiteLLM docs](https://docs.litellm.ai/docs/) for provider-specific setup

**Modal deployment fails:**
- Ensure secret `llm-gateway-secrets` exists with all required keys
- Check Modal logs: `modal logs fastapi_app`
- Verify you're authenticated: `modal setup`

## Contributing

This is a demo project. Feel free to fork and extend for your needs!

## License

MIT
