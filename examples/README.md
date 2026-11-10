# Venice AI SDK Examples

This directory contains comprehensive examples demonstrating how to use the Venice AI Python SDK effectively.

## 📁 Directory Structure

```
examples/
├── README.md                     # This file
├── basic/                        # Simple getting started examples
│   ├── quick_start.py           # Minimal setup and usage
│   ├── sync_client.py           # Blocking calls with SyncVeniceClient: chat, a stream and a paginator, and what stays async
│   ├── offline_testing.py       # Test your own code with no network, API key or spend, using a real client against a local stand-in
│   ├── client_setup.py          # Ways to configure the client; reads back the API root, connection_limits and retry_options (production section needs Redis)
│   └── error_handling.py        # Live SDK exceptions, APITimeoutError, billing-aware retries (paid requests never resent once processed), fallbacks
├── chat/                        # Chat completion examples
│   ├── _helpers.py              # Shared helpers: grounding checks (an answer states the tool's signed value with its unit)
│   ├── simple_chat.py           # Basic chat completions: stop, n, top_p, penalties, seed, logprobs
│   ├── streaming_chat.py        # Real-time streaming: text_deltas, reasoning_summary
│   ├── tee_e2ee.py              # TEE confidential-compute client-side E2EE chat: get_signature, open_session
│   ├── tool_calling.py          # Hand-driven tool round trip, parallel calls, tool_choice, tool errors, tool_from_model, parse_as; answers checked against the tool output
│   ├── streaming_tools.py       # Stream with tools=: read assembled calls from final_response, complete the round trip; answer checked against the tool output
│   ├── video_audio_input.py     # Video by public URL and audio as base64 in chat; catalog filtered on supportsVideoInput/supportsAudioInput; billed
│   ├── structured_output.py     # Typed JSON-schema responses via parse(); values checked in code, one re-sample on a validation failure
│   ├── multi_turn_conversation.py # Context preservation
│   ├── model_feature_suffixes.py # Model feature suffixes (model:key=val) and build_model_id()
│   ├── reasoning_and_thinking.py # Reasoning and chain-of-thought
│   ├── venice_parameters.py     # Venice-specific parameters
│   ├── vision.py                # Vision / multimodal chat completions: base64 and URL input, detail
│   ├── file_inputs.py           # Attach documents via type:file (data: URL or public URL)
│   ├── web_scraping.py          # Web scraping with chat completions
│   ├── passthrough_fields.py    # OpenAI-compat store/text/include/metadata, verbosity, anon_user_id, fallbacks; prompt_cache_retention tiers, with cache reads reported per tier
│   └── agent_loop.py            # Agent loop via run_with_tools (auto tool dispatch), on_tool_error, parallel tools, Conversation.run_with_tools; answer checked against the tool results
├── responses/                   # OpenAI-style Responses API (beta)
│   └── responses_api.py         # responses.create — typed output blocks, reasoning text, truncated (incomplete) responses, text.format, instructions, streaming, function-tool round trip, web_search (billed extra)
├── decisions/                   # Decision ("System One") models (Beta)
│   └── ticket_routing.py        # Typed noul/choice/score judgments with images, structured state, NoulCriteria and probabilities, checked against a calm control ticket; low confidence escalates
├── embeddings/                  # Text embedding examples
│   ├── basic_embeddings.py      # Simple text vectorization, dimensions, encoding_format, vector size checked against the catalog
│   ├── similarity_search.py     # Semantic similarity analysis
│   └── batch_processing.py      # Processing multiple texts
├── image/                       # Image generation examples
│   ├── _helpers.py              # Shared helpers: base images, saving, size and pixel checks, catalog prices, key headroom
│   ├── text_to_image.py         # Basic image generation
│   ├── generation_options.py    # format, cfg_scale, safe_mode, enhance_prompt, style_references, transparent_background, image.submit, simple_generate; prices only unless opted in
│   ├── image_upscaling.py       # Quality enhancement; prints a quote before it spends
│   ├── style_variants.py        # Different artistic styles
│   ├── batch_generation.py      # Multiple image creation
│   ├── background_removal.py    # Remove image backgrounds
│   ├── image_editing.py         # Edit and modify images (resolution + timeout)
│   ├── multi_edit.py            # Edit from several input images and one prompt; four inputs with images= behind an opt-in
│   ├── quality_control.py       # Native quality tiers (low/medium/high)
│   └── web_search.py            # Image generation with web search context
├── audio/                       # Audio examples
│   ├── _helpers.py              # Shared helpers: audio container detection, format checks
│   ├── text_to_speech.py        # Basic TTS generation
│   ├── speech_to_text.py        # Speech-to-text transcription
│   ├── voice_cloning.py         # Clone a voice from a sample, then synthesize
│   ├── voice_changer.py         # Re-voice an existing recording (job family); quotes unless opted in
│   ├── voice_options.py         # Different voices and settings
│   └── long_text_streaming.py   # Stream TTS audio for long-form text
├── video/                       # Video generation examples
│   ├── _helpers.py              # Shared helpers: MP4 track parsing, clip checks (resolution at least the requested tier), progress, skip exit code, spending ceiling, URLs safe to print
│   ├── text_to_video.py         # Generate video from text prompts
│   ├── image_to_video.py        # Animate images into video
│   ├── advanced_fields.py       # Reference images/audio, transitions, elements (R2V), keyframes, omni and consent models when named
│   ├── upscale.py               # Upscale a source video (model from resolve_video_upscale); refuses a job quoted above a ceiling
│   └── resume_job.py            # Resume a paid job from its saved model and queue_id: retrieve, save, check, release; never submits
├── augment/                     # Document and web augmentation examples
│   ├── scrape.py                # Scrape a short page and a long docs page to markdown; a blocked domain raises
│   ├── search.py                # Structured web search, Brave vs Google, chat answer whose quotes are checked against the cited sources
│   └── text_parser.py           # Extract text from document files
├── models/                      # Model discovery examples
│   ├── list_models.py           # Browse available models
│   ├── model_selection.py       # Semantic model discovery: resolve() filters, resolve_music, DynamicModelSelector
│   ├── model_lifecycle.py       # context_length, deprecation, capability metadata
│   └── compatibility.py         # Migration from other APIs, including the Anthropic response mapping
├── advanced/                    # Advanced configuration and features
│   ├── custom_configuration.py  # Advanced client setup
│   ├── env_configuration.py     # Which ways of building a client honour VENICE_* variables; free, offline, needs pydantic-settings
│   ├── redis_backend.py         # Redis-backed rate limiter wiring, verified in Redis (needs Redis)
│   ├── error_recovery.py        # Retry and recovery patterns, 429 and 503 on a local stand-in, plus free 404 and 401 from the real API
│   ├── performance_optimization.py # Sequential vs concurrent requests, streaming time to first token
│   ├── prompt_caching.py        # prompt_cache_key, cache_control on system and message markers, ttl, cached-token usage
│   └── reasoning_effort.py      # Controlling reasoning effort levels
├── production/                  # Production-ready examples
│   ├── api_key_management.py    # Secure key handling
│   ├── logging_monitoring.py    # Proper logging setup
│   ├── async_patterns.py        # Scalable async patterns
│   └── cost_management.py       # Cost tracking, estimate vs actual, price-based model choice, budgets
├── api_keys/                    # API key management examples
│   ├── key_management.py        # List keys, rate limits, create/use/update/delete a temporary real key, list this example's leftovers (deleted only on request)
│   └── web3_key.py              # Wallet key flow: token, local signing; mint only with VENICE_RUN_WEB3_MINT=1
├── billing/                     # Billing and usage examples
│   └── usage_analytics.py       # Account balance vs key headroom, usage ledger and analytics
├── characters/                  # Character API examples
│   ├── character_discovery.py   # Browse available characters (free, catalog only)
│   └── character_details.py     # Character details; prices the chat with estimate_cost() before sending
├── headers/                     # Response header examples
│   └── header_access_example.py # Accessing response metadata
├── crypto/                      # Blockchain RPC proxy + supported networks
│   └── networks_and_rpc.py      # crypto.networks / rpc / batch_rpc, idempotent replay
├── x402/                        # x402 wallet-based micropayments
│   ├── balance.py               # Read prepaid USDC balance (SIWE auth)
│   ├── transactions.py          # Transaction history
│   ├── top_up.py                # EVM/Base top-up flow
│   ├── solana_settlement.py     # Solana USDC top-up (SolanaX402Auth); payment signed and verified locally, nothing sent by default
│   └── wallet_inference.py      # Keyless inference paid from the prepaid ledger; api_key="" trap checked locally; paid call is opt-in
├── best_practices/              # SDK best practices and patterns
│   └── pydantic_models.py       # Type-safe model usage guide
├── music/                       # Async music generation
│   └── music_generation.py      # Cheapest generator for a clip length, job and low-level flows; quotes unless opted in
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

1. **Python 3.11+** installed
2. **Venice AI SDK** installed — either `pip install venice-py` (end users) or
   `poetry install` from a repo checkout (development)
3. **API Key** from [Venice AI](https://venice.ai)
4. **Environment variable** set: `export VENICE_API_KEY="your-api-key"`
5. **Optional**: some examples need an extra (`adaptive`, `e2ee`, `e2ee-verify`, `enterprise`,
   `x402`, `x402-solana`), Pillow, a wallet key, or a Redis server; each script's docstring lists
   what it needs. Plain `poetry install` and `pip install venice-py` install no extras. Add them
   with `poetry install -E e2ee -E x402` or `pip install 'venice-py[e2ee,x402]'`, or
   `poetry install --all-extras` for everything. Pillow comes with the `cli` extra
6. **Admin key, for two examples**: `api_keys/key_management.py` and
   `billing/usage_analytics.py` need an ADMIN key in `VENICE_API_KEY`. With any other valid key
   they print `SKIPPED:` and exit `77`. `key_management.py` creates, uses and deletes a real
   INFERENCE key with a $1 lifetime cap, and makes one small billed chat. It lists keys an
   earlier killed run left behind, and deletes them only with `VENICE_SWEEP_EXAMPLE_KEYS=1`

The x402 examples (`x402/balance.py`, `transactions.py`, `top_up.py`, `solana_settlement.py`,
`wallet_inference.py`) authenticate with the wallet, so they need a wallet key and no API key.

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

Scripts that share code import `_helpers` from their own directory, so run them by
path as above. `python -P` and `PYTHONSAFEPATH=1` take that directory off the import
path, and those scripts then fail with `ModuleNotFoundError`.

### Exit codes

Every example checks its own results and reports them through its exit code:

| Code | Meaning |
| --- | --- |
| `0` | Passed: the example ran and verified what it demonstrates |
| `1` | Failed: an API error, or a result that did not pass the example's checks |
| `77` | Skipped: an optional prerequisite is missing (an extra not installed, an environment variable, wallet or Redis server not set up, an endpoint the account is not entitled to, or no catalog model of the kind needed), or a job still processing |

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

Live examples spend credits on the key in `VENICE_API_KEY`. Most pick the
cheapest suitable model from the live catalog (`prefer="cheapest"`). That
ranking includes reasoning models, so examples that need a short, direct answer
pass `exclude_reasoning=True`. Some take the catalog's default ranking instead,
which does not consider price: the basic, structured-input, reasoning and truncation
demos in `responses/responses_api.py`, `chat/structured_output.py`, the strict-schema
demo in `best_practices/pydantic_models.py`, the schema demo in
`production/cost_management.py`, `audio/voice_changer.py`, `video/upscale.py`, and the
cache demos in
`production/cost_management.py` and `advanced/prompt_caching.py`, which fail without a
cache hit, and a cached-input price does not say which models serve one. A costlier
default raises what these cost. Chat requests that need nothing from Venice's
default system prompt send `include_venice_system_prompt=False`, so it is not
billed on top of their own messages. Catalog, quote and model-listing calls are
free.

What one run costs without any opt-in, on the catalog prices listed when the
figures were written. The live catalog decides the real figure. Most scripts state
their basis in the docstring.

| Per run | Scripts |
| --- | --- |
| Free: nothing billed | `advanced/env_configuration.py`, `advanced/error_recovery.py`, `api_keys/web3_key.py` (none expected), `basic/client_setup.py`, `basic/offline_testing.py`, `billing/usage_analytics.py`, `characters/character_discovery.py`, `models/list_models.py`, `models/model_lifecycle.py`, `models/model_selection.py`, `video/resume_job.py`, `x402/balance.py`, `x402/transactions.py`, and the dry runs of `x402/top_up.py`, `x402/solana_settlement.py` and `x402/wallet_inference.py` |
| Free: quotes, then exit `77` | `audio/voice_changer.py`, `image/batch_generation.py`, `image/generation_options.py`, `image/style_variants.py`, `music/music_generation.py`, `video/advanced_fields.py`, `video/image_to_video.py`, `video/text_to_video.py`, `video/upscale.py` |
| Under $0.001 | `advanced/performance_optimization.py`, `advanced/custom_configuration.py`, `advanced/redis_backend.py`, `api_keys/key_management.py`, `basic/error_handling.py`, `basic/quick_start.py`, `basic/sync_client.py`, `best_practices/pydantic_models.py`, `chat/file_inputs.py`, `chat/multi_turn_conversation.py`, `chat/simple_chat.py`, `chat/streaming_tools.py`, `chat/tee_e2ee.py`, `crypto/networks_and_rpc.py`, `decisions/ticket_routing.py`, `embeddings/basic_embeddings.py`, `embeddings/batch_processing.py`, `embeddings/similarity_search.py`, `models/compatibility.py`, `production/api_key_management.py`, `production/async_patterns.py`, `production/logging_monitoring.py` |
| Under $0.01 | `audio/speech_to_text.py`, `audio/text_to_speech.py`, `characters/character_details.py`, `chat/agent_loop.py`, `chat/passthrough_fields.py`, `chat/reasoning_and_thinking.py`, `chat/streaming_chat.py`, `chat/structured_output.py`, `chat/tool_calling.py`, `chat/video_audio_input.py`, `chat/vision.py`, `production/cost_management.py` (about $0.003 on today's default pick) |
| $0.01 to $0.03 | `advanced/prompt_caching.py` (up to $0.02 on today's default pick; about $0.92 if the default were the costliest cache-priced model), `advanced/reasoning_effort.py` (a few cents at most), `audio/long_text_streaming.py` (about $0.03), `audio/voice_options.py` (about $0.03), `augment/scrape.py` ($0.02), `augment/text_parser.py` ($0.02), `chat/model_feature_suffixes.py` (about $0.012), `headers/header_access_example.py` (up to $0.02), `responses/responses_api.py` (about $0.015) |
| $0.03 to $0.10 | `audio/voice_cloning.py` (about $0.054), `augment/search.py` ($0.04), `chat/venice_parameters.py` (about $0.04), `chat/web_scraping.py` (about $0.05), `image/multi_edit.py` (about $0.07), `image/text_to_image.py` (about $0.07) |
| Over $0.10 | `image/background_removal.py` (about $0.16), `image/image_editing.py` ($0.15 to $0.20), `image/image_upscaling.py` (about $0.15, and it prints a quote first), `image/quality_control.py` ($0.16 to $0.35), `image/web_search.py` (about $0.10, plus about $0.01 if the pricing page's $10.00 per 1,000 searches applies to image models) |

With an opt-in:

| Script | Opt-in | Per run |
| --- | --- | --- |
| `image/batch_generation.py` | `VENICE_RUN_PAID_IMAGES=1` | about $0.16 |
| `image/style_variants.py` | `VENICE_RUN_PAID_IMAGES=1` | about $0.11 |
| `image/generation_options.py` | `VENICE_RUN_PAID_IMAGES=1` | about $0.25 to $0.40 |
| `image/multi_edit.py` | `VENICE_RUN_PAID_IMAGES=1` | about $0.02 more |
| `music/music_generation.py` | `VENICE_RUN_PAID_MUSIC=1` | about $0.15 to $0.45 |
| `audio/voice_changer.py` | `VENICE_RUN_PAID_VOICE_CHANGER=1` | about $0.35 |
| `video/upscale.py` | `VENICE_RUN_PAID_VIDEO=1` | $0.13 on the default sample, plus one enhanced upscale at its printed quote, each job refused above $0.50 unless `VENICE_UPSCALE_MAX_USD` is raised |
| `video/text_to_video.py`, `video/image_to_video.py`, `video/advanced_fields.py` | `VENICE_RUN_PAID_VIDEO=1` | the printed quote, each job refused above $0.50 unless `VENICE_VIDEO_MAX_USD` is raised. A quote leaves out reference images, audio, elements and keyframes, so `advanced_fields.py` can cost more than its quote and more than the ceiling |
| `x402/wallet_inference.py` | `X402_RUN_INFERENCE=1` | one short chat from the wallet's ledger |
| `x402/top_up.py`, `x402/solana_settlement.py` | `X402_DO_TOPUP=1`, `X402_SOLANA_DO_TOPUP=1` | a real, irreversible USDC transfer |

### Opt-in and optional settings

| Variable | Read by | Effect |
| --- | --- | --- |
| `VENICE_API_KEY` | every live example except the wallet-authenticated x402 ones and `basic/offline_testing.py` | API key for all calls (required) |
| `VENICE_API_BASE_URL` | every live example (read by the SDK) | API root, version path included (`https://api.venice.ai/api/v1`); a bare host gets `/api/v1` appended |
| `VENICE_RUN_PAID_VIDEO=1` | `video/text_to_video.py`, `image_to_video.py`, `advanced_fields.py`, `upscale.py` | Queue and download paid video jobs; without it the video examples only quote |
| `VENICE_UPSCALE_SOURCE_URL` | `video/upscale.py` | Source clip to upscale (HTTPS URL or `data:` URI) |
| `VENICE_VIDEO_ELEMENTS_MODEL` | `video/advanced_fields.py` | Model to run the elements section on |
| `VENICE_VIDEO_KEYFRAMES_MODEL`, `VENICE_VIDEO_OMNI_MODEL` | `video/advanced_fields.py` | Models to run the keyframes and omni sections on. No catalog field says which models take them, so each section is skipped when its variable is unset |
| `VENICE_VIDEO_CONSENT_MODEL`, `VENICE_VIDEO_FACE_IMAGE_URL`, `VENICE_VIDEO_CONSENT_ATTESTED` | `video/advanced_fields.py` | The consent section's model, a photo you have the right to use, and `1` once you have read the policy, to send `consents` as your own confirmation. Without the attestation the section prints the policy and stops. A photo with no detected face is accepted and billed, so it also needs `VENICE_RUN_PAID_VIDEO=1` |
| `VENICE_UPSCALE_MAX_USD` | `video/upscale.py` | Highest quote a job may have, in USD (default `0.50`). A job quoted above it is refused |
| `VENICE_VIDEO_MAX_USD` | `video/text_to_video.py`, `image_to_video.py`, `advanced_fields.py` | Highest quote a job may have, in USD (default `0.50`). A job quoted above it is skipped. Quotes leave out reference images, audio, elements and keyframes, so a job using them can cost more |
| `VENICE_SWEEP_EXAMPLE_KEYS=1` | `api_keys/key_management.py` | Delete the keys an earlier killed run left behind. Without it they are listed, with how to delete them |
| `VENICE_RUN_PAID_MUSIC=1` | `music/music_generation.py` | Make every generation; without it every model is quoted and the example exits `77` |
| `VENICE_RUN_PAID_VOICE_CHANGER=1` | `audio/voice_changer.py` | Queue a conversion, about $0.35; without it the example prints the quote and exits `77` |
| `VENICE_TTS_STYLE_MODEL` | `audio/text_to_speech.py` | Model to send the style controls (`prompt`, `temperature`, `top_p`) to, for a catalog that lists no model advertising them |
| `VENICE_RUN_WEB3_MINT=1` | `api_keys/web3_key.py` | Also post the signed token to `create_web3_key()`. With the throwaway wallet the expected answer is a 400 |
| `VENICE_HTTP_CLIENT__TIMEOUT` | `advanced/env_configuration.py` sets it itself | The script sets, unsets and restores it itself to show what it does; the value in your shell does not change the result |
| `VENICE_BACKEND__REDIS__REDIS_URL`, `VENICE_REDIS_URL`, `REDIS_URL` | `basic/client_setup.py`, `advanced/redis_backend.py`, and `VENICE_REDIS_URL` in `advanced/performance_optimization.py` | Redis for the shared rate-limit state. The first one set wins, in this order: `VENICE_BACKEND__REDIS__REDIS_URL` is the SDK's own setting (`VeniceAIConfig.backend.redis.redis_url`), `VENICE_REDIS_URL` is a convenience name the examples read, and `REDIS_URL` is the conventional fallback |
| `VENICE_RUN_PAID_IMAGES=1` | `image/batch_generation.py`, `image/style_variants.py`, `image/generation_options.py`, `image/multi_edit.py` | Make the paid generations of the first three: 16 images in 15 requests, 11, and up to 9 (7 when the catalog lists no transparent-background model); without it they resolve the model, print the plan and catalog price, and exit `77`. In `multi_edit.py` it adds the four-input section; without it that section prints its price and is skipped, and the rest still runs and spends |
| `VENICE_VOICE_SAMPLE_PATH` | `audio/voice_cloning.py`, `audio/voice_changer.py` | Your own recording to use as input (`voice_changer.py` needs PCM WAV). Clone only voices you have consent to clone. Without it, the examples synthesize a short clip with Venice TTS, which is one more billed call |
| `VENICE_X402_TEST_PRIVATE_KEY` or `X402_WALLET_PRIVATE_KEY` | `x402/balance.py`, `transactions.py`, `top_up.py`, `wallet_inference.py` | EVM wallet key for SIWE auth (use a test wallet) |
| `X402_DO_TOPUP=1` | `x402/top_up.py` | Submit the real, irreversible USDC top-up on Base; without it the flow is signed but not sent |
| `X402_RUN_INFERENCE=1` | `x402/wallet_inference.py` | Pay for one short chat from the wallet's prepaid USDC ledger; without it the sign-in envelope is signed locally, nothing is sent for inference, and the example exits `77` |
| `VENICE_VIDEO_RESUME_MODEL`, `VENICE_VIDEO_RESUME_QUEUE_ID` | `video/resume_job.py` | The job to collect and release. Spends nothing new; without a queue ID the example sends nothing and exits `77` |
| `VENICE_VIDEO_RESUME_DOWNLOAD_URL` | `video/resume_job.py` | The job's `download_url`, for models that return one. It grants access on its own, so treat it as a secret |
| `VENICE_X402_SOLANA_TEST_PRIVATE_KEY` | `x402/solana_settlement.py` | Solana wallet key (use a test wallet) |
| `VENICE_X402_SOLANA_RPC_URL` | `x402/solana_settlement.py` | Solana RPC endpoint for the on-chain balance read |
| `X402_SOLANA_DO_TOPUP=1` | `x402/solana_settlement.py` | Submit the real Solana USDC top-up |

The image examples other than `batch_generation.py`, `style_variants.py` and
`generation_options.py` spend on every run.

### Paid video runs

The generation examples only quote by default, which is free, and then exit `77`
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

- **Async/Await Patterns**: Examples use async programming, except `basic/sync_client.py`, which shows the blocking client
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
