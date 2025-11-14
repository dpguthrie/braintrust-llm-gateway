# LLM Gateway with Braintrust Tracing

Demo-quality LLM gateway that automatically instruments all requests with Braintrust tracing.

## Features

- **Automatic Tracing**: Every LLM request automatically logged to Braintrust
- **Multi-Provider**: Support for OpenAI, Anthropic, and 100+ models via LiteLLM
- **OpenAI Compatible**: Works with any OpenAI SDK client
- **Simple Auth**: Bearer token authentication
- **Streaming**: Full support for streaming responses
- **Modal Deployment**: Easy serverless deployment

## Architecture

```
Client → FastAPI Gateway → LiteLLM → OpenAI/Anthropic/etc.
                ↓
         Braintrust Tracing
```

## Local Development

### Prerequisites

- Python 3.11+
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

2. Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys
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

Any model supported by LiteLLM works. Examples:

- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`
- See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for full list

### Viewing Traces

All requests are automatically logged to Braintrust:

1. Go to [braintrust.dev](https://braintrust.dev)
2. Navigate to your project (default: "llm-gateway")
3. View traces with full request/response data, token counts, and latency

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
cp .env.example .env
# Edit .env with your actual keys
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

### Automatic Tracing

The gateway uses Braintrust's LiteLLM integration for zero-config tracing:

1. On startup: `braintrust.patch_litellm()` wraps all LiteLLM calls
2. Each request: LiteLLM automatically creates Braintrust spans
3. Captures: inputs, outputs, tokens, latency, time-to-first-token, model metadata
4. No manual span management needed!

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

1. **How to add observability to LLM apps**: Show Braintrust traces in real-time
2. **Multi-provider routing**: Switch models by changing request parameter
3. **OpenAI SDK compatibility**: Drop-in replacement for OpenAI API
4. **Serverless deployment**: Deploy to Modal in minutes

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
