# venice-py

<div align="center">

<a href="https://venice-docs.sbang.dev/"><img src="https://raw.githubusercontent.com/sethbang/venice-py/main/website/static/img/venice-py-banner.png" alt="venice-py — unofficial, community-maintained Python SDK for Venice.ai" width="720"></a>

[![PyPI version](https://img.shields.io/pypi/v/venice-py.svg)](https://pypi.org/project/venice-py/)
[![Python Versions](https://img.shields.io/pypi/pyversions/venice-py.svg)](https://pypi.org/project/venice-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-125da3.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/sethbang/venice-py/actions/workflows/ci-validation.yaml/badge.svg)](https://github.com/sethbang/venice-py/actions/workflows/ci-validation.yaml)
[![Coverage Status](https://img.shields.io/codecov/c/github/sethbang/venice-py.svg)](https://codecov.io/gh/sethbang/venice-py)
[![Security Scan](https://github.com/sethbang/venice-py/actions/workflows/security-scan.yml/badge.svg)](https://github.com/sethbang/venice-py/actions/workflows/security-scan.yml)
[![Docs](https://img.shields.io/badge/docs-venice--docs.sbang.dev-3c8fdd)](https://venice-docs.sbang.dev/)

**Production-ready Python SDK for Venice.ai with enterprise-grade rate limiting, intelligent scheduling, and comprehensive error handling**

[Documentation](https://venice-docs.sbang.dev/) | [Examples](https://github.com/sethbang/venice-py/tree/main/examples/) | [Changelog](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md) | [API Reference](https://venice-docs.sbang.dev/docs/api-reference/)

</div>

---

> **This is an unofficial, community-maintained SDK for Venice.ai.**
> Not affiliated with or endorsed by Venice AI. For official resources visit [Venice.ai](https://venice.ai/).

> **Installed as `venice-py`.** This package was published as `venice-ai` through v2.1.0.
> **Your code does not change** — the import package is still `venice_ai` and the environment
> variable is still `VENICE_API_KEY`. Only the name you install moved:
>
> ```bash
> pip install venice-py     # was: pip install venice-ai
> ```
>
> `venice-ai` stays on PyPI and nothing is yanked, so existing pins keep resolving. Its final
> release, 2.1.1, is a metadata-only bridge that pulls in `venice-py`. The bridge declares **no
> extras**, so if you install any — `[cli]`, `[x402]`, `[redis]`, `[all]` — switch the name or
> pip will warn and quietly skip them.

> **v2.x** — fully breaking rewrite over v1.3.x with enterprise-grade features. Review the [release notes](https://github.com/sethbang/venice-py/releases), the [CHANGELOG](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md), and the [Migration Guide](https://venice-docs.sbang.dev/docs/guides/migration/) before upgrading.

---

## Quick Start

> **Requires Python 3.13+.** On earlier versions pip reports that no matching distribution
> is available.

```bash
pip install 'venice-py'
export VENICE_API_KEY="your-api-key-here"
```

```python
import asyncio
from venice_ai import VeniceClient, UserMessage

async def main():
    async with VeniceClient() as client:  # reads VENICE_API_KEY from env
        model = await client.models.resolve_chat()
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Hello!")],
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

Migrating from v1.x? See the [Migration Guide](https://venice-docs.sbang.dev/docs/guides/migration/).

## Command Line Interface

The command is `venice-py`. It is not Venice's official CLI — that one is
[`veniceai-cli`](https://github.com/veniceai/venice-cli), a separate Node program invoked
as `venice`. Both can be installed at once. (This command was `venice` in v2.0.x; see the
[CHANGELOG](CHANGELOG.md) if you are upgrading.)

```bash
pip install 'venice-py[cli]'

venice-py chat start                  # Interactive chat
venice-py image generate "..."        # Image generation
venice-py image multi-edit -p "..."   # Multi-image edit
venice-py models                      # Browse models
venice-py characters reviews <slug>   # Character reviews
venice-py account keys rate-limits    # Per-model RPM/TPM limits
venice-py configure                   # Setup wizard
```

Features: streaming chat with 6 animation modes, image generation with 11+ parameters, model discovery, rich terminal UI. See the [CLI Reference](https://venice-docs.sbang.dev/docs/guides/cli/) for full CLI documentation.

---

## Build with Claude Code

Four [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills) ship with the SDK. Install them into the current project with `venice-py skills install` (or `venice-py skills install --global` for `~/.claude/skills/`); list them with `venice-py skills list`. They auto-load when their trigger contexts match — "Venice chat", "generate an image with Venice", "Venice rate limits", "Venice x402" — and steer Claude toward idiomatic v2 code (dynamic model resolution, `async with stream:`, `run_with_tools`, `client.gather(max_concurrency=N)`, `top_up_with`, etc.) instead of OpenAI-style or v1 patterns.

```bash
venice-py skills install            # → ./.claude/skills/
venice-py skills install --global   # → ~/.claude/skills/
venice-py skills list               # show bundled skills + install state
```

Catalog: `venice-py` (chat / streaming / tools / structured output), `venice-py-multimodal` (image / audio / video / music), `venice-py-production` (retries / rate limits / cost tracking / observability), `venice-py-x402` (wallet auth / SIWE / on-chain top-up). See [`tools/skills/README.md`](https://github.com/sethbang/venice-py/blob/main/tools/skills/README.md) for the full catalog and validation tooling.

---

## Core Features

### Chat Completions

```python
from venice_ai import UserMessage, SystemMessage

async with VeniceClient() as client:
    model = await client.models.resolve_chat()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Explain quantum computing simply"),
        ],
        temperature=0.7,
        max_completion_tokens=500,
    )
    print(response.choices[0].message.content)
```

[-> `examples/chat/simple_chat.py`](https://github.com/sethbang/venice-py/blob/main/examples/chat/simple_chat.py)

### Streaming

```python
model = await client.models.resolve_chat()
async with await client.chat.completions.stream(
    model=model,
    messages=[UserMessage(content="Tell me a story")],
    max_completion_tokens=200,
) as stream:
    async for text in stream.text_deltas():
        print(text, end="", flush=True)
```

For the assembled response: `response = await stream.collect()`.

[-> `examples/chat/streaming_chat.py`](https://github.com/sethbang/venice-py/blob/main/examples/chat/streaming_chat.py)

### Synchronous Usage

```python
from venice_ai import SyncVeniceClient, UserMessage

with SyncVeniceClient() as client:
    model = client.models.resolve_chat()
    response = client.chat.completions.create(
        model=model,
        messages=[UserMessage(content="Hello!")],
    )
    print(response.choices[0].message.content)
```

Streams returned by `SyncVeniceClient` iterate synchronously (`for chunk in stream:`).

### Function Calling

```python
from venice_ai import tool_from_function

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    ...

model = await client.models.resolve_chat(require_function_calling=True)
response = await client.chat.completions.create(
    model=model,
    messages=[UserMessage(content="What's the weather in NYC?")],
    tools=[tool_from_function(get_weather)],
    tool_choice="auto",
)
```

`tool_from_model(MyPydanticModel)` is also available for richer schemas.

[-> `examples/chat/tool_calling.py`](https://github.com/sethbang/venice-py/blob/main/examples/chat/tool_calling.py)

### Image Generation

```python
image_model = await client.models.resolve_image()
response = await client.image.create(
    model=image_model,
    prompt="A serene mountain landscape at sunset",
    width=512, height=512,
    enable_web_search=True,  # optional; supported models pull in recent web context
)
response.save("generated.png")          # single image
# response.save_all("output_dir")        # all images
```

Pass-through fields on `image.multi_edit()` now include `model=...`, which the
SDK forwards to the API as `modelId` (previously dropped silently).

[-> `examples/image/text_to_image.py`](https://github.com/sethbang/venice-py/blob/main/examples/image/text_to_image.py) | [-> `examples/image/web_search.py`](https://github.com/sethbang/venice-py/blob/main/examples/image/web_search.py)

### Text-to-Speech

```python
from venice_ai.types.enums import Voice, ResponseFormat

tts_model = await client.models.resolve_tts()
response = await client.audio.create_speech(
    model=tts_model,
    input="Hello! Welcome to Venice AI.",
    voice=Voice.AF_ALLOY,
    response_format=ResponseFormat.MP3,
)
response.save("speech.mp3")
```

[-> `examples/audio/text_to_speech.py`](https://github.com/sethbang/venice-py/blob/main/examples/audio/text_to_speech.py)

### Embeddings

```python
embedding_model = await client.models.resolve_embedding()
response = await client.embeddings.create(
    model=embedding_model,
    input=["Text 1", "Text 2", "Text 3"],
)
embedding = response.data[0].embedding
```

[-> `examples/embeddings/basic_embeddings.py`](https://github.com/sethbang/venice-py/blob/main/examples/embeddings/basic_embeddings.py)

### Video Generation

```python
video_model = await client.models.resolve_video()
job = await client.video.run(
    model=video_model,
    prompt="A drone shot of the Venice canals at sunrise",
    duration_seconds=5,
    aspect_ratio="16:9",
    resolution="1080p",
)
async with job:
    status = await job.wait()
    await job.download("canals.mp4", status)
```

Advanced body fields on `submit()` (all optional):
`upscale_factor`, `end_image_url`, `audio_url`, `video_url`,
`reference_image_urls` (up to 9), `elements` (up to 4 Kling-O3 structured
characters), `scene_image_urls` (up to 4). For the dedicated
`topaz-video-upscale` model, pass `video_url` + `upscale_factor` (1/2/4)
instead of `resolution`. `quote()` accepts only the pricing-relevant
subset (`model`, `duration_seconds`, `aspect_ratio`, `resolution`, `upscale_factor`,
`audio`, `video_url`) per the API spec — prompt text and reference images
don't affect price.

[-> `examples/video/text_to_video.py`](https://github.com/sethbang/venice-py/blob/main/examples/video/text_to_video.py) | [-> `examples/video/advanced_fields.py`](https://github.com/sethbang/venice-py/blob/main/examples/video/advanced_fields.py) | [-> `examples/video/upscale.py`](https://github.com/sethbang/venice-py/blob/main/examples/video/upscale.py)

### Model Selection

```python
chat_model = await client.models.resolve_chat(
    preferred_models=["llama-3.3-70b", "qwen-2.5-72b"],
    require_function_calling=True,
)
image_model = await client.models.resolve_image()
embedding_model = await client.models.resolve_embedding()

# Capability filters via the unified entry point:
vision_model = await client.models.resolve(type="chat", require_vision=True)
```

---

## Venice-Specific Features

### Character Personalities

```python
from venice_ai import VeniceParameters

model = await client.models.resolve_chat()
response = await client.chat.completions.create(
    model=model,
    messages=[UserMessage(content="What is wisdom?")],
    venice_parameters=VeniceParameters(character_slug="socrates"),
)
```

### Web Search

```python
model = await client.models.resolve_chat()
response = await client.chat.completions.create(
    model=model,
    messages=[UserMessage(content="Latest AI news?")],
    venice_parameters=VeniceParameters(enable_web_search="on", enable_web_citations=True),
)
```

### Web Scrape, Search & Text Parsing (Augment)

```python
# Scrape a URL and get markdown back
page = await client.augment.scrape(url="https://example.com")
print(page.content)

# Structured web search (Brave default; Google also supported)
hits = await client.augment.search(query="latest AI news", limit=5)
for r in hits.results:
    print(r.title, r.url)

# Parse a document (PDF / DOCX / XLSX / TXT, ≤ 25 MB)
parsed = await client.augment.parse_text(file="report.pdf")
print(parsed.text, parsed.tokens)
```

[-> `examples/augment/scrape.py`](https://github.com/sethbang/venice-py/blob/main/examples/augment/scrape.py) | [-> `examples/augment/search.py`](https://github.com/sethbang/venice-py/blob/main/examples/augment/search.py) | [-> `examples/augment/text_parser.py`](https://github.com/sethbang/venice-py/blob/main/examples/augment/text_parser.py)

### x402 Wallet Billing (optional)

The x402 billing endpoints use Ethereum wallet auth (SIWE / EIP-4361 on
Base) instead of Bearer tokens. Install the optional extra to pick up
`eth-account` + `siwe`:

```bash
pip install 'venice-py[x402]'
```

```python
from venice_ai.auth.x402 import X402Auth

auth = X402Auth(private_key=os.environ["X402_WALLET_PRIVATE_KEY"])

async with VeniceClient() as client:
    balance = await client.x402.balance(auth=auth)
    print(f"${balance.data.balanceUsd} on {auth.wallet_address}")

    txns = await client.x402.transactions(auth=auth)
    for t in txns.data.transactions[:5]:
        print(t.createdAt, t.type, t.amount)

    # Empty POST discovers x402 payment requirements; the response surfaces
    # as a 402 APIError whose body carries the accept spec.
    await client.x402.top_up()  # or top_up(payment_header=<signed b64>)
```

Prefer one-call top-ups? `client.x402.top_up_with(auth=auth, amount_usdc=5.0)` runs the
full EVM probe → sign → submit flow for you. To settle from a **Solana** wallet instead,
install `venice-py[x402-solana]` and use `SolanaX402Auth` with `top_up_with_solana`:

```python
from venice_ai.auth.x402_solana import SolanaX402Auth

auth = SolanaX402Auth(private_key=os.environ["X402_SOLANA_SECRET"])  # base58 secret
async with VeniceClient() as client:
    await client.x402.top_up_with_solana(auth=auth, amount_usdc=5.0)
```

[-> `examples/x402/balance.py`](https://github.com/sethbang/venice-py/blob/main/examples/x402/balance.py) | [-> `examples/x402/transactions.py`](https://github.com/sethbang/venice-py/blob/main/examples/x402/transactions.py) | [-> `examples/x402/top_up.py`](https://github.com/sethbang/venice-py/blob/main/examples/x402/top_up.py)

### Confidential Compute (TEE / E2EE, optional)

Venice's `e2ee-*` models run in a Trusted Execution Environment with
client-side end-to-end encryption: attest the enclave, then encrypt each
message under a key only the enclave can derive (secp256k1 ECDH -> HKDF-SHA256
-> AES-256-GCM). Install the optional extra (`cryptography`):

```bash
pip install 'venice-py[e2ee]'
```

```python
# One-shot: just turn on E2EE for an e2ee-* model. The SDK attests the enclave,
# opens a session, and encrypts/decrypts transparently.
model = await client.models.resolve_chat()  # pick an e2ee-* model
response = await client.chat.completions.create(
    model=model,
    messages=[UserMessage(content="Confidential question")],
    e2ee=True,  # equivalent: venice_parameters=VeniceParameters(enable_e2ee=True)
)

# Or drive the lifecycle yourself for low-level control:
attestation = await client.tee.get_attestation(model=model)  # fail-closed verify
with await client.tee.open_session(model=model) as session:
    headers = session.request_headers()
    blob = session.encrypt_message("Hello, confidential world.")
    # ... POST the encrypted content with `headers`; decrypt streamed deltas via
    #     session.decrypt_chunk(delta_hex)
```

#### Full client-side TDX verification (`[e2ee-verify]`)

The default path is **baseline**: it trusts Venice's server-side `verified`
claim and does not independently verify the Intel TDX quote. For threat models
that include a malicious Venice operator, install the `[e2ee-verify]` extra and
pass a `DcapTdxVerifier`, which verifies the raw quote's ECDSA signature and PCK
certificate chain to a pinned Intel SGX Root CA, the TCB status, the non-debug
flag, the REPORTDATA key binding, the RTMR event-log replay, and the dstack
compose-hash — all offline:

```bash
pip install 'venice-py[e2ee-verify]'   # dcap-qvl (+ cryptography)
```

```python
from venice_ai.tee import DcapTdxVerifier, TeeOptions

model = await client.models.resolve_chat()  # pick an e2ee-* model

# Fetch Intel-signed collateral once (the only network touch); verify() is offline.
verifier = await DcapTdxVerifier.with_fetched_collateral(
    probe_quote=(await client.tee.get_attestation(model=model)).intel_quote,
)

# Run the full verifier as part of attestation / session open:
session = await client.tee.open_session(model=model, verifier=verifier)

# Or engage it through chat E2EE:
response = await client.chat.completions.create(
    model=model,
    messages=[UserMessage(content="Confidential question")],
    e2ee=TeeOptions(verifier=verifier),
)
```

> **What it proves (Tier B).** By default `DcapTdxVerifier` proves the model
> runs on a *genuine, non-debug Intel TDX enclave* running a *self-consistent
> dstack workload*. It does **not** by itself prove this is the legitimate
> Venice workload — there are no published reference measurements today. Supply
> `expected_measurements` / `expected_compose_hash` from an independent source to
> pin workload identity (Tier A). TCB status is fail-closed reject-by-default
> (`tcb_policy="advisory"` to accept hardening-needed statuses with advisories).
> NVIDIA GPU attestation is not yet shipped.

### Reasoning Controls & Cost Tracking

```python
# Reasoning effort tier — top-level parameter on chat.completions.create().
# Seven tiers (per-model support): none / minimal / low / medium / high / xhigh / max.
response = await client.chat.completions.create(
    model=await client.models.resolve_chat(require_reasoning=True),
    messages=[UserMessage(content="Prove √2 is irrational.")],
    reasoning_effort="max",
)

# Nested form with summary verbosity. Top-level reasoning_effort takes
# precedence over reasoning.effort when both are set.
from venice_ai import ReasoningConfig
response = await client.chat.completions.create(
    model=reasoning_model,
    messages=[UserMessage(content="Explain quantum entanglement.")],
    reasoning=ReasoningConfig(effort="high", summary="concise"),
)

# Show/hide raw thinking blocks via venice_parameters
venice_params = VeniceParameters(strip_thinking_response=False, disable_thinking=False)

# Cost tracking
from venice_ai import calculate_completion_cost
cost = calculate_completion_cost(response, model_pricing=None)
print(f"Cost: ${cost['usd']:.4f}")
```

---

## Configuration

### Installation Options

Quote the specifier — `[...]` is a glob in zsh.

```bash
pip install 'venice-py'                # Core
pip install 'venice-py[cli]'           # CLI tools
pip install 'venice-py[redis]'         # Redis backend
pip install 'venice-py[enterprise]'    # Enterprise (redis + prometheus + otel)
pip install 'venice-py[adaptive]'      # Adaptive rate limiting
pip install 'venice-py[x402]'          # x402 wallet auth (eth-account + siwe)
pip install 'venice-py[x402-solana]'   # x402 Solana USDC top-up (solders)
pip install 'venice-py[e2ee]'          # TEE client-side E2EE (cryptography)
pip install 'venice-py[e2ee-verify]'   # Full client-side TDX quote verification (dcap-qvl)
pip install 'venice-py[all]'           # Everything
```

### Client Setup

```python
# Minimal (reads VENICE_API_KEY from environment)
async with VeniceClient() as client: ...

# Explicit
async with VeniceClient(api_key="your-key") as client: ...

# Factory with full configuration
from venice_ai import VeniceClientFactory, VeniceAIConfig
from venice_ai.core.config import BackendConfig, BackendType, HttpClientConfig, SchedulerConfig, SchedulerMode

config = VeniceAIConfig(
    backend=BackendConfig(backend_type=BackendType.MEMORY),
    http_client=HttpClientConfig(timeout=60.0, max_connections=50),
    scheduler=SchedulerConfig(mode=SchedulerMode.BASIC)
)
client = VeniceClientFactory.create_client(config=config, api_key=os.getenv("VENICE_API_KEY"))
```

### Environment Variables

Configure with `VENICE_` prefix (double underscores for nesting):

```bash
export VENICE_SCHEDULER__MODE=intelligent
export VENICE_BACKEND__REDIS__REDIS_URL=redis://localhost:6379
export VENICE_HTTP_CLIENT__TIMEOUT=60.0
```

> **Note:** Env var auto-loading requires `pydantic-settings`: `pip install 'venice-py[enterprise]'`

---

## Advanced Features

Rate limiting, distributed state, monitoring, observability, and performance tuning are covered in **[Advanced Features](https://venice-docs.sbang.dev/docs/guides/advanced/)**.

---

## API Resources

| Resource | Purpose | Key Methods | Example |
|----------|---------|-------------|---------|
| `chat.completions` | Chat & text generation | `create()` | [simple_chat.py](https://github.com/sethbang/venice-py/blob/main/examples/chat/simple_chat.py) |
| `responses` | Stateless multi-modal generation (Alpha) | `create()` | — |
| `image` | Image generation | `create()`, `background_remove()` | [text_to_image.py](https://github.com/sethbang/venice-py/blob/main/examples/image/text_to_image.py) |
| `video` | Async video generation | `run()` → `VideoJob`, low-level `submit()` / `quote()` / `retrieve()` / `cancel()` | [text_to_video.py](https://github.com/sethbang/venice-py/blob/main/examples/video/text_to_video.py) |
| `audio` | TTS / ASR | `create_speech()`, `transcribe()` | [text_to_speech.py](https://github.com/sethbang/venice-py/blob/main/examples/audio/text_to_speech.py) |
| `music` | Async music generation | `run()` → `MusicJob`, low-level `submit()` / `quote()` / `retrieve()` / `cancel()` | [music_generation.py](https://github.com/sethbang/venice-py/blob/main/examples/music/music_generation.py) |
| `embeddings` | Text embeddings | `create()` | [basic_embeddings.py](https://github.com/sethbang/venice-py/blob/main/examples/embeddings/basic_embeddings.py) |
| `models` | Model discovery | `list()`, `get()` | [list_models.py](https://github.com/sethbang/venice-py/blob/main/examples/models/list_models.py) |
| `billing` | Usage analytics | `get_balance()`, `get_usage_history()`, `get_usage_analytics()` | [usage_analytics.py](https://github.com/sethbang/venice-py/blob/main/examples/billing/usage_analytics.py) |
| `api_keys` | Key management | `list()`, `get_rate_limits()` | [key_management.py](https://github.com/sethbang/venice-py/blob/main/examples/api_keys/key_management.py) |
| `characters` | Character discovery & reviews | `list()`, `get()`, `reviews()` | [character_details.py](https://github.com/sethbang/venice-py/blob/main/examples/characters/character_details.py) |
| `augment` | Web scrape / search / text-parser | `scrape()`, `search()`, `parse_text()` | [scrape.py](https://github.com/sethbang/venice-py/blob/main/examples/augment/scrape.py) |
| `x402` | Wallet-billing (SIWE auth; `[x402]` extra) | `balance()`, `transactions()`, `top_up()` | [balance.py](https://github.com/sethbang/venice-py/blob/main/examples/x402/balance.py) |
| `crypto` | Multi-chain JSON-RPC proxy | `networks()`, `rpc()`, `batch_rpc()` | [networks_and_rpc.py](https://github.com/sethbang/venice-py/blob/main/examples/crypto/networks_and_rpc.py) |
| `tee` | Confidential-compute attestation & E2EE session (`[e2ee]` extra) | `get_attestation()`, `open_session()` | — |

## Type Safety

The SDK uses Pydantic v2 models throughout:

```python
from venice_ai.types.api import UserMessage, SystemMessage, AssistantMessage, ToolMessage
from venice_ai.types.api.requests import VeniceParameters, StreamOptions
from venice_ai.types.api.requests.common import Tool, ToolFunction
from venice_ai.types import JSONSchemaFormat
from venice_ai.types.chat import ChatCompletionChunk
from venice_ai.types.audio import Voice, ResponseFormat
```

## Testing

```python
from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode

async with create_test_venice_client(api_key="test-key", scheduler_mode=SchedulerMode.BASIC) as client:
    response = await client.chat.completions.create(...)
```

```bash
make test           # All tests (parallel)
make test-unit      # Unit tests only
make test-e2e       # E2E tests (requires API key)
make test-verbose   # With coverage
```

## Best Practices

1. **Always use context managers** for proper cleanup
2. **Handle errors** with specific exception types (`RateLimitError`, `AuthenticationError`, etc.)
3. **Monitor rate limits** via `response.response_rate_limits`
4. **Use environment variables** for API keys (never hardcode)
5. **Use Redis backend** for production multi-instance deployments
6. **Use streaming** for long responses to reduce time to first token

---

## Requirements

- **Python 3.13+**
- **Core deps:** aiohttp (>=3.13.4,<3.15), pydantic (^2.13.4)
- **Platform:** Linux, macOS, Windows

See [Installation Options](#installation-options) for optional dependencies.

## Contributing

```bash
git clone https://github.com/sethbang/venice-py.git && cd venice-py
make install
make test
```

Follow PEP 8, use type hints, write tests, and submit a PR.

## Support

- [Documentation](https://venice-docs.sbang.dev/)
- [Issue Tracker](https://github.com/sethbang/venice-py/issues)
- [Examples](https://github.com/sethbang/venice-py/tree/main/examples/)
- [Changelog](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md)

## License

MIT License — see [`LICENSE`](https://github.com/sethbang/venice-py/blob/main/LICENSE).

---

<div align="center">

**[Back to Top](#venice-py)**

An unofficial community SDK for Venice.ai

</div>
