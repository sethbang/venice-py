# Venice AI SDK Examples

This directory contains comprehensive examples demonstrating how to use the Venice AI Python SDK effectively.

## 📁 Directory Structure

```
examples/
├── README.md                     # This file
├── basic/                        # Simple getting started examples
│   ├── quick_start.py           # Minimal setup and usage
│   ├── client_setup.py          # Different ways to configure the client
│   └── error_handling.py        # Basic error handling patterns
├── chat/                        # Chat completion examples
│   ├── simple_chat.py           # Basic chat completions
│   ├── streaming_chat.py        # Real-time streaming responses
│   ├── tee_e2ee.py              # TEE confidential-compute client-side E2EE chat
│   ├── tool_calling.py          # Function calling and tools
│   ├── structured_output.py     # JSON schema responses
│   ├── multi_turn_conversation.py # Context preservation
│   ├── model_feature_suffixes.py # Model feature suffix usage
│   ├── reasoning_and_thinking.py # Reasoning and chain-of-thought
│   ├── venice_parameters.py     # Venice-specific parameters
│   ├── vision.py                # Vision / multimodal chat completions
│   ├── file_inputs.py           # Attach documents via type:file (data: URL or public URL)
│   ├── web_scraping.py          # Web scraping with chat completions
│   ├── passthrough_fields.py    # Forward unmodeled fields to the API
│   └── agent_loop.py            # Agent loop via run_with_tools (auto tool dispatch)
├── responses/                   # OpenAI-style Responses API (Alpha)
│   └── responses_api.py         # responses.create — typed output blocks
├── decisions/                   # Decision ("System One") models (Beta)
│   └── ticket_routing.py        # Typed noul/choice/score judgments + confidence gating
├── embeddings/                  # Text embedding examples
│   ├── basic_embeddings.py      # Simple text vectorization
│   ├── similarity_search.py     # Semantic similarity analysis
│   └── batch_processing.py      # Processing multiple texts
├── image/                       # Image generation examples
│   ├── _helpers.py              # Shared helpers (file I/O, display utilities) used by image examples
│   ├── text_to_image.py         # Basic image generation
│   ├── image_upscaling.py       # Quality enhancement
│   ├── style_variants.py        # Different artistic styles
│   ├── batch_generation.py      # Multiple image creation
│   ├── background_removal.py    # Remove image backgrounds
│   ├── image_editing.py         # Edit and modify images (resolution + timeout)
│   ├── multi_edit.py            # Multi-layer image editing
│   ├── quality_control.py       # Native quality tiers (low/medium/high)
│   └── web_search.py            # Image generation with web search context
├── audio/                       # Audio examples
│   ├── text_to_speech.py        # Basic TTS generation
│   ├── speech_to_text.py        # Speech-to-text transcription
│   ├── voice_cloning.py         # Clone a voice from a sample, then synthesize
│   ├── voice_changer.py         # Re-voice an existing recording (job family)
│   ├── voice_options.py         # Different voices and settings
│   └── long_text_streaming.py   # Stream TTS audio for long-form text
├── video/                       # Video generation examples
│   ├── text_to_video.py         # Generate video from text prompts
│   ├── image_to_video.py        # Animate images into video
│   ├── advanced_fields.py       # Reference images/audio, transitions, elements (R2V)
│   └── upscale.py               # Upscale video with topaz-video-upscale
├── augment/                     # Document and web augmentation examples
│   ├── scrape.py                # Web scrape a URL to text/markdown
│   ├── search.py                # Web search augmentation
│   └── text_parser.py           # Extract text from document files
├── models/                      # Model discovery examples
│   ├── list_models.py           # Browse available models
│   ├── model_selection.py       # Semantic model discovery
│   ├── model_lifecycle.py       # context_length, deprecation, capability metadata
│   └── compatibility.py         # Migration from other APIs
├── advanced/                    # Advanced configuration and features
│   ├── custom_configuration.py  # Advanced client setup
│   ├── redis_backend.py         # Redis state management
│   ├── error_recovery.py        # Comprehensive error handling
│   ├── performance_optimization.py # Best practices for performance
│   ├── prompt_caching.py        # Prompt caching for efficiency
│   └── reasoning_effort.py      # Controlling reasoning effort levels
├── production/                  # Production-ready examples
│   ├── api_key_management.py    # Secure key handling
│   ├── logging_monitoring.py    # Proper logging setup
│   ├── async_patterns.py        # Scalable async patterns
│   └── cost_management.py       # Usage tracking and billing
├── api_keys/                    # API key management examples
│   └── key_management.py        # Key operations and lifecycle
├── billing/                     # Billing and usage examples
│   └── usage_analytics.py       # Usage tracking and cost analysis
├── characters/                  # Character API examples
│   ├── character_discovery.py   # Browse available characters
│   └── character_details.py     # Character information and usage
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
│   └── music_generation.py      # Generate music from a text prompt
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

## 💡 Key Features Demonstrated

- **Async/Await Patterns**: All examples use proper async programming
- **Error Handling**: Comprehensive error catching and recovery
- **Type Safety**: Full type hints and Pydantic model usage
- **Best Practices**: Production-ready patterns and configurations
- **Performance**: Optimized usage patterns for scalability

## 📚 Learn More

- [Venice AI Documentation](https://docs.venice.ai)
- [API Reference](https://docs.venice.ai/api)
- [Python SDK Guide](https://docs.venice.ai/sdk/python)

## 🤝 Contributing

Found an issue or want to improve an example? Please open an issue or submit a pull request!