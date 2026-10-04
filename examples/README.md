# Venice AI SDK Examples

This directory contains comprehensive examples demonstrating how to use the Venice AI Python SDK effectively.

## 📁 Directory Structure

```
examples/
├── README.md                     # This file
├── basic/                        # Simple getting started examples
│   ├── quick_start.py           # Minimal setup and usage
│   ├── client_setup.py          # Ways to configure the client; reads back the API root, connection_limits and retry_options (production section needs Redis)
│   └── error_handling.py        # Live SDK exceptions, APITimeoutError, billing-aware retries (paid requests never resent once processed), fallbacks
├── chat/                        # Chat completion examples
│   ├── _helpers.py              # Shared helpers: grounding checks (an answer states the tool's signed value with its unit)
│   ├── simple_chat.py           # Basic chat completions
│   ├── streaming_chat.py        # Real-time streaming responses
│   ├── tee_e2ee.py              # TEE confidential-compute client-side E2EE chat
│   ├── tool_calling.py          # Hand-driven tool round trip, parallel calls, tool_choice, tool errors; answers checked against the tool output
│   ├── structured_output.py     # Typed JSON-schema responses via parse(); values checked in code, one re-sample on a validation failure
│   ├── multi_turn_conversation.py # Context preservation
│   ├── model_feature_suffixes.py # Model feature suffixes (model:key=val) and build_model_id()
│   ├── reasoning_and_thinking.py # Reasoning and chain-of-thought
│   ├── venice_parameters.py     # Venice-specific parameters
│   ├── vision.py                # Vision / multimodal chat completions
│   ├── file_inputs.py           # Attach documents via type:file (data: URL or public URL)
│   ├── web_scraping.py          # Web scraping with chat completions
│   ├── passthrough_fields.py    # OpenAI-compat store/text/include/metadata; prompt_cache_retention tiers, with cache reads reported per tier
│   └── agent_loop.py            # Agent loop via run_with_tools (auto tool dispatch); answer checked against the tool results
├── responses/                   # OpenAI-style Responses API (Alpha)
│   └── responses_api.py         # responses.create — typed output blocks, reasoning text, truncated (incomplete) responses
├── decisions/                   # Decision ("System One") models (Beta)
│   └── ticket_routing.py        # Typed noul/choice/score judgments, checked against a calm control ticket; low confidence escalates
├── embeddings/                  # Text embedding examples
│   ├── basic_embeddings.py      # Simple text vectorization, vector size checked against the catalog
│   ├── similarity_search.py     # Semantic similarity analysis
│   └── batch_processing.py      # Processing multiple texts
├── image/                       # Image generation examples
│   ├── _helpers.py              # Shared helpers: base images, saving, size checks, retry policy, pricing
│   ├── text_to_image.py         # Basic image generation
│   ├── image_upscaling.py       # Quality enhancement
│   ├── style_variants.py        # Different artistic styles
│   ├── batch_generation.py      # Multiple image creation
│   ├── background_removal.py    # Remove image backgrounds
│   ├── image_editing.py         # Edit and modify images (resolution + timeout)
│   ├── multi_edit.py            # Multi-layer image editing
│   ├── quality_control.py       # Native quality tiers (low/medium/high)
│   └── web_search.py            # Image generation with web search context
├── voice-to-clone.wav           # Sample recording used by audio/voice_cloning.py
├── audio/                       # Audio examples
│   ├── _helpers.py              # Shared helpers: audio container detection, format checks
│   ├── text_to_speech.py        # Basic TTS generation
│   ├── speech_to_text.py        # Speech-to-text transcription
│   ├── voice_cloning.py         # Clone a voice from a sample, then synthesize
│   ├── voice_changer.py         # Re-voice an existing recording (job family)
│   ├── voice_options.py         # Different voices and settings
│   └── long_text_streaming.py   # Stream TTS audio for long-form text
├── video/                       # Video generation examples
│   ├── _helpers.py              # Shared helpers: MP4 track parsing, clip checks (resolution at least the requested tier), progress, skip exit code
│   ├── text_to_video.py         # Generate video from text prompts
│   ├── image_to_video.py        # Animate images into video
│   ├── advanced_fields.py       # Reference images/audio, transitions, elements (R2V)
│   └── upscale.py               # Upscale a source video (model from resolve_video_upscale)
├── augment/                     # Document and web augmentation examples
│   ├── scrape.py                # Scrape a short page and a long docs page to markdown; a blocked domain raises
│   ├── search.py                # Structured web search, Brave vs Google, chat answer whose quotes are checked against the cited sources
│   └── text_parser.py           # Extract text from document files
├── models/                      # Model discovery examples
│   ├── list_models.py           # Browse available models
│   ├── model_selection.py       # Semantic model discovery
│   ├── model_lifecycle.py       # context_length, deprecation, capability metadata
│   └── compatibility.py         # Migration from other APIs
├── advanced/                    # Advanced configuration and features
│   ├── custom_configuration.py  # Advanced client setup
│   ├── redis_backend.py         # Redis-backed rate limiter wiring, verified in Redis (needs Redis)
│   ├── error_recovery.py        # Retry and recovery patterns, each error triggered locally for free
│   ├── performance_optimization.py # Sequential vs concurrent requests, streaming time to first token
│   ├── prompt_caching.py        # prompt_cache_key, cache_control and cached-token usage
│   └── reasoning_effort.py      # Controlling reasoning effort levels
├── production/                  # Production-ready examples
│   ├── api_key_management.py    # Secure key handling
│   ├── logging_monitoring.py    # Proper logging setup
│   ├── async_patterns.py        # Scalable async patterns
│   └── cost_management.py       # Cost tracking, estimate vs actual, price-based model choice, budgets
├── api_keys/                    # API key management examples
│   └── key_management.py        # List keys, rate limits, create/use/delete a temporary real key
├── billing/                     # Billing and usage examples
│   └── usage_analytics.py       # Account balance vs key headroom, usage ledger and analytics
├── characters/                  # Character API examples
│   ├── character_discovery.py   # Browse available characters (free, catalog only)
│   └── character_details.py     # Character details; prices the chat with estimate_cost() before sending
├── headers/                     # Response header examples
│   └── header_access_example.py # Accessing response metadata
├── crypto/                      # Blockchain RPC proxy + supported networks
│   └── networks_and_rpc.py      # crypto.networks / rpc / batch_rpc
├── x402/                        # x402 wallet-based micropayments
│   ├── balance.py               # Read prepaid USDC balance (SIWE auth)
│   ├── transactions.py          # Transaction history
│   ├── top_up.py                # EVM/Base top-up flow
│   └── solana_settlement.py     # Solana USDC top-up (SolanaX402Auth)
├── best_practices/              # SDK best practices and patterns
│   └── pydantic_models.py       # Type-safe model usage guide
├── music/                       # Async music generation
│   └── music_generation.py      # Cheapest generator for a clip length, job and low-level flows
└── results/                     # Output directory for generated files (gitignored)
                                 # Created automatically when examples save images/audio/video
```

## 🚀 Quick Start

If you're new to Venice AI, start with these examples:

1. **[basic/quick_start.py](basic/quick_start.py)** - Get up and running in minutes
2. **[chat/simple_chat.py](chat/simple_chat.py)** - Your first chat completion
3. **[embeddings/basic_embeddings.py](embeddings/basic_embeddings.py)** - Generate text embeddings

## 📋 Prerequisites

Before running these examples, ensure you have:

1. **Python 3.13+** installed
2. **Venice AI SDK** installed — either `pip install venice-py` (end users) or
   `poetry install` from a repo checkout (development)
3. **API Key** from [Venice AI](https://venice.ai)
4. **Environment variable** set: `export VENICE_API_KEY="your-api-key"`
5. **Optional**: some examples need an extra (`e2ee`, `x402`, `x402-solana`), a wallet key, or a
   Redis server; each script's docstring lists what it needs

## 🔧 Running Examples

From a repo checkout, run examples through Poetry so the local SDK and its
dependencies resolve:

```bash
# Basic usage
poetry run python examples/basic/quick_start.py

# Chat completions
poetry run python examples/chat/simple_chat.py

# With streaming
poetry run python examples/chat/streaming_chat.py
```

If you installed the published package into your own environment instead, drop
the `poetry run` prefix (e.g. `python examples/basic/quick_start.py`).

### Exit codes

Every example checks its own results and reports them through its exit code:

| Code | Meaning |
| --- | --- |
| `0` | Passed: the example ran and verified what it demonstrates |
| `1` | Failed: an API error, or a result that did not pass the example's checks |
| `77` | Skipped: an optional prerequisite is missing (an extra not installed, an environment variable, wallet or Redis server not set up, an endpoint the account is not entitled to, or no catalog model of the kind needed) |

A whole-example skip prints a line starting with `SKIPPED:` that says what was
missing, and exits `77`. That prefix is reserved for this case.

When only part of an example cannot run (one section needs Redis, a model kind
the catalog lacks, or a paid opt-in), that section prints a line starting with
`Section skipped:` (at the start of the line, not indented) and the reason. The
example still exits `0` if its core feature was verified, and `77` (with a
`SKIPPED:` line) if it was not. A skip never hides a failure: if any section
failed, the example exits `1` whatever else was skipped. The closing summary
claims only what ran; a skipped section is never listed as demonstrated.

A resolver that finds no model of the kind needed raises
`NoMatchingModelError`, which the examples treat as a skip. When matching
models exist but every price quote failed, the resolver raises
`ModelQuotesUnavailableError`, which is a failure (`1`).

### Cost

Live examples spend credits on the key in `VENICE_API_KEY`. They pick the
cheapest suitable model from the live catalog (`prefer="cheapest"`). That
ranking includes reasoning models, so examples that need a short, direct answer
pass `exclude_reasoning=True`. Chat requests that need nothing from Venice's
default system prompt send `include_venice_system_prompt=False`, so it is not
billed on top of their own messages. Catalog, quote and model-listing calls are
free.

### Opt-in and optional settings

| Variable | Read by | Effect |
| --- | --- | --- |
| `VENICE_API_KEY` | every live example | API key for all calls (required) |
| `VENICE_API_BASE_URL` | every live example (read by the SDK) | API root, version path included (`https://api.venice.ai/api/v1`); a bare host gets `/api/v1` appended |
| `VENICE_RUN_PAID_VIDEO=1` | `video/text_to_video.py`, `image_to_video.py`, `advanced_fields.py`, `upscale.py` | Queue and download paid video jobs; without it the video examples only quote |
| `VENICE_UPSCALE_SOURCE_URL` | `video/upscale.py` | Source clip to upscale (HTTPS URL or `data:` URI) |
| `VENICE_VIDEO_ELEMENTS_MODEL` | `video/advanced_fields.py` | Model to run the elements section on |
| `VENICE_RUN_PAID_MUSIC=1` | `music/music_generation.py` | Also generate the `force_instrumental` clip (quoted for free on every run) |
| `VENICE_BACKEND__REDIS__REDIS_URL`, `VENICE_REDIS_URL`, `REDIS_URL` | `basic/client_setup.py`, `advanced/redis_backend.py` | Redis for the shared rate-limit state. The first one set wins, in this order: `VENICE_BACKEND__REDIS__REDIS_URL` is the SDK's own setting (`VeniceAIConfig.backend.redis.redis_url`), `VENICE_REDIS_URL` is a convenience name the examples read, and `REDIS_URL` is the conventional fallback |
| `VENICE_X402_TEST_PRIVATE_KEY` or `X402_WALLET_PRIVATE_KEY` | `x402/balance.py`, `transactions.py`, `top_up.py` | EVM wallet key for SIWE auth (use a test wallet) |
| `X402_DO_TOPUP=1` | `x402/top_up.py` | Submit the real, irreversible USDC top-up on Base; without it the flow is signed but not sent |
| `VENICE_X402_SOLANA_TEST_PRIVATE_KEY` | `x402/solana_settlement.py` | Solana wallet key (use a test wallet) |
| `VENICE_X402_SOLANA_RPC_URL` | `x402/solana_settlement.py` | Solana RPC endpoint for the on-chain balance read |
| `X402_SOLANA_DO_TOPUP=1` | `x402/solana_settlement.py` | Submit the real Solana USDC top-up |

`music/music_generation.py` makes two paid generations on the cheapest model on
every run; only the `force_instrumental` generation waits for the opt-in.

### Paid video runs

Every video example only quotes by default, which is free, and then exits `77`
because nothing was generated. Set `VENICE_RUN_PAID_VIDEO=1` to queue, wait and
download; `text_to_video.py` and `image_to_video.py` then generate short clips
at the lowest resolution and duration the resolved model lists. `upscale.py` takes a custom source clip from
`VENICE_UPSCALE_SOURCE_URL`.

`advanced_fields.py` picks reference-to-video and transition models with
`resolve_cheapest_video(input_mode=...)`. Neither the catalog nor the quote says
which models accept elements, so the elements section runs only when
`VENICE_VIDEO_ELEMENTS_MODEL` names one. Provider content moderation can reject
a reference job after it is queued; the section then fails and the credits are
refunded.

## 💡 Key Features Demonstrated

- **Async/Await Patterns**: All examples use proper async programming
- **Error Handling**: Comprehensive error catching and recovery
- **Type Safety**: Full type hints and Pydantic model usage
- **Best Practices**: Production-ready patterns and configurations
- **Performance**: Optimized usage patterns for scalability

## 📚 Learn More

- [Venice AI Documentation](https://docs.venice.ai)
- [API Reference](https://docs.venice.ai/api-reference/api-spec)
- [Python SDK Guide](https://venice-docs.sbang.dev)

## 🤝 Contributing

Found an issue or want to improve an example? Please open an issue or submit a pull request!