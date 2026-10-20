---
sidebar_position: 1
title: Getting Started
---

# Getting Started

> **This is an unofficial, community-maintained SDK for Venice.ai.**
> Not affiliated with or endorsed by Venice AI. For official resources visit [Venice.ai](https://venice.ai/).

The Venice AI Python SDK is a production-ready, fully-typed async client for [Venice.ai](https://venice.ai).

## Install

**Python 3.11+ is required.** On earlier versions pip reports that no matching
distribution is available.

```bash
pip install 'venice-py'
export VENICE_API_KEY="your-api-key-here"
```

## Your first chat completion

The SDK resolves models dynamically — you never hardcode a model ID.

```python
import asyncio
from venice_ai import VeniceClient, UserMessage

async def main() -> None:
    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Hello, Venice!")],
        )
        print(response.text)

asyncio.run(main())
```

Prefer a synchronous client? Use `SyncVeniceClient` with a plain `with` block.

## Claude Code skills

The SDK bundles four Claude Code skills (chat, multimodal, production, x402).
Installing them uses the `venice-py` CLI, which ships as an optional extra:

```bash
pip install 'venice-py[cli]'

venice-py skills install            # → ./.claude/skills/
venice-py skills install --global   # → ~/.claude/skills/
venice-py skills list               # show bundled skills + install state
```

Open Claude Code in the project and the skills auto-load when their trigger
contexts match (e.g. "Venice chat", "generate an image with Venice",
"Venice x402").

## Next steps

- The [Guides](/docs/guides/migration) cover migration, the CLI, advanced configuration, and rate-limiting.
- The [API Reference](/docs/api-reference) documents every resource, type, and exception.
