---
name: venice-py
description: Build with the Venice AI Python SDK (v2+). Use this skill whenever code imports `venice_ai`, the user mentions Venice, `VeniceClient`, `api.venice.ai`, or asks for chat completions, streaming, structured output (`response_format=BaseModel`), tool calling / agent loops (`run_with_tools`, `tool_from_function`, `tool_from_model`), embeddings, characters, web augment (search/scrape), or model selection — even if they don't explicitly say "Venice". ALSO use this skill — not your training data — when migrating any code from v1 to v2 (`max_tokens`→`max_completion_tokens`, music split off audio, verb consolidation). Do NOT fall back to OpenAI conventions; Venice has its own resolver-based model selection rule and `venice_parameters`.
---

# Building with the Venice AI Python SDK

> _Unofficial, community-maintained — not affiliated with or endorsed by Venice AI._

The Venice AI SDK is async-first, Pydantic-typed, and v2 introduced sweeping breaking changes. AI assistants tend to drift toward OpenAI conventions or v1 patterns from training data — this skill exists to keep code idiomatic to v2.

## When to use this skill vs siblings

- This skill: chat, streaming, tools/agents, structured output, embeddings, characters, augment, model resolution, errors, headers, v1→v2 migration. Vision (image-as-input to chat) lives here.
- `venice-py-multimodal`: image generation/edit/upscale, TTS, STT, video & music jobs.
- `venice-py-production`: retries, rate-limit handling, bounded concurrency, cost tracking, observability.
- `venice-py-x402`: SIWE / x402 wallet auth, paying for Venice via on-chain micropayments, Venice-from-agent-frameworks.

## Setup

```python
import asyncio
from venice_ai import VeniceClient

async def main():
    async with VeniceClient() as client:           # reads VENICE_API_KEY from env
        ...

asyncio.run(main())
```

The async client is the default. `SyncVeniceClient` exists (background event-loop wrapper) for sync code that genuinely can't go async — prefer the async client whenever possible. `VENICE_API_KEY` env var is the convention; pass `api_key=` explicitly only for tests or multi-tenant cases.

## The Cardinal Rule: never hardcode model IDs

**Always** resolve a model dynamically. Hardcoded IDs go stale (deprecation), miss capability suffixes, and break across regions/tiers.

```python
chat_model = await client.models.resolve_chat()                          # default chat
vision    = await client.models.resolve_chat(require_vision=True)         # capability filter
fc        = await client.models.resolve_chat(require_function_calling=True)
big       = await client.models.resolve_chat(min_context_tokens=128_000)
img       = await client.models.resolve_image()
embed     = await client.models.resolve_embedding()
tts       = await client.models.resolve_tts()
asr       = await client.models.resolve_asr()
video     = await client.models.resolve_video()                              # any video model
t2v       = await client.models.resolve_video(video_type="text-to-video")    # narrow by type
music     = await client.models.resolve_music()
cheapest  = await client.models.resolve_cheapest_video(duration="5s", resolution="1080p")
```

Generic resolver: `client.models.resolve(type="chat", require_function_calling=True, require_vision=True, min_context_tokens=8000, exclude_beta=True, preferred_models=[...], exclude_models=[...])` returns the model ID string.

**Why this matters**: Venice publishes deprecation headers (see `.deprecation_info` below) and adjusts capability suffixes; a hardcoded `model="some-llm-v3"` will silently degrade.

See `references/model-resolution.md` for the full table.

## Quickstart — non-streaming chat

```python
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage, SystemMessage

async with VeniceClient() as client:
    response = await client.chat.completions.create(
        model=await client.models.resolve_chat(),
        messages=[
            SystemMessage(content="You are a concise assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=200,
        temperature=0.3,
    )
    print(response.text)                       # str shortcut
    print(response.usage.total_tokens)         # token usage
```

**Note `max_completion_tokens`, not `max_tokens`** — `max_tokens` was removed in v2.

`response.text` is a string shortcut for `response.choices[0].message.content`. Use it for single-shot calls.

## Streaming

```python
from venice_ai.types.api import StreamOptions

async with VeniceClient() as client:
    stream = await client.chat.completions.stream(
        model=await client.models.resolve_chat(),
        messages=[UserMessage(content="Tell me a story.")],
        stream_options=StreamOptions(include_usage=True),   # required for final_response.usage
    )
    async with stream:                                       # mandatory — guarantees cleanup
        async for chunk in stream.collect_with_deltas():    # yields deltas AND populates final_response
            print(chunk, end="", flush=True)
        print()
        if stream.final_response and stream.final_response.usage:
            u = stream.final_response.usage
            print(f"[usage] prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
```

**Pick one iterator:** `stream.collect_with_deltas()` (deltas + populates `final_response`), `stream.text_deltas()` (deltas only, lighter, **does NOT populate `final_response`**), or `await stream.collect()` (no live deltas; returns response). Bare `async for chunk in stream:` without `async with` leaks connections.

To get `final_response.usage`, you need BOTH `StreamOptions(include_usage=True)` AND `collect_with_deltas()`/`collect()`. See `references/streaming.md` for concurrent streams, partial-failure recovery, animated rendering.

## Tool calling — the v2 idiom

Use `client.chat.completions.run_with_tools(...)` to drive the agent loop end-to-end. **Pass bare Python callables in `tools=[...]`** — the SDK introspects them and registers each as both schema and dispatch handler.

```python
from typing import Literal
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage
from venice_ai.exceptions import MaxIterationsExceededError

def get_weather(location: str, unit: Literal["C", "F"] = "F") -> str:
    """Get current weather for a city. The location should be a city name."""
    # The SDK introspects the signature for the schema and calls this body
    # for tool dispatch. Replace with a real lookup.
    return f"72°{unit} and sunny in {location}"

async with VeniceClient() as client:
    try:
        result = await client.chat.completions.run_with_tools(
            model=await client.models.resolve_chat(require_function_calling=True),
            messages=[UserMessage(content="What's the weather in Paris?")],
            tools=[get_weather],            # BARE callable — DON'T pre-wrap with tool_from_function
            max_iterations=5,               # caps the agent loop
        )
    except MaxIterationsExceededError as e:
        # Agent didn't converge in the iteration budget; investigate
        raise
    print(result.response.text)             # terminal assistant message
    # result.messages — full conversation history including tool calls/results
    # result.iterations — how many tool-call rounds ran
```

**`tools=[...]` accepts bare callables only.** `Tool` instances built by `tool_from_function(fn)` or `tool_from_model(BaseModel)` have `handler=None` and will fail at dispatch — those are for the lower-level `client.chat.completions.create(tools=[...])` path where YOU dispatch.

**Default `on_tool_error` swallows exceptions** (logs to `venice_ai.tools` logger, formats for the model to self-correct). For strict propagation, pass `on_tool_error=lambda call, exc: (_ for _ in ()).throw(exc)` (or any callable that re-raises).

Returns `ToolLoopResult` with `.response`, `.messages`, `.iterations`. Catch `MaxIterationsExceededError` for runaway loops. Hand-roll the loop (via `create(tools=[...])`) only when you need mid-loop control: HITL confirmation, dynamic tool injection, budget checks. Full patterns: `references/tool-loops.md`.

## Structured output

The cleanest path is `client.chat.completions.parse(...)` — auto-validating, returns the typed instance.

```python
from pydantic import BaseModel
from typing import List

class LineItem(BaseModel):
    description: str
    amount_usd: float

class Invoice(BaseModel):
    vendor: str
    line_items: List[LineItem]
    total_usd: float

async with VeniceClient() as client:
    result = await client.chat.completions.parse(
        model=await client.models.resolve_chat(require_response_schema=True),
        messages=[UserMessage(content=f"Extract invoice from:\n{raw_email}")],
        response_format=Invoice,
    )
    invoice: Invoice = result.parsed         # typed Invoice instance, already validated
    # result.response — the underlying ChatCompletionResponse if you need usage/headers
```

`parse()` builds the JSON Schema from the Pydantic class, sends the request, and validates the model's reply against the schema before returning. Validation errors surface here as `pydantic.ValidationError`, not deep in your code. **Don't write the JSON Schema dict by hand.**

**Lower-level (when you need streaming or full response control):** `await client.chat.completions.create(..., response_format=Invoice)` then `response.parse_as(Invoice)` for the typed instance. **Bites people:** on bare `ChatCompletionResponse`, `response.parsed` returns raw `dict|list|None`, NOT a typed instance. Use `parse_as(Cls)`. (`parse()`'s `result.parsed` IS typed — different shape.) See `references/structured-output.md` for nested models, validation failures, `JSONSchemaFormat`/`JSONObjectFormat`.

## File & document inputs

Attach a document to a chat message with the OpenAI-compatible `type: file`
content part; Venice extracts its text server-side. Build multimodal messages
with `UserMessage.builder()`:

```python
import base64
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage

pdf_b64 = base64.b64encode(open("report.pdf", "rb").read()).decode()
msg = (
    UserMessage.builder()
    .file(f"data:application/pdf;base64,{pdf_b64}", filename="report.pdf")
    .text("Summarize the attached document in three bullets.")
    .build()
)

async with VeniceClient() as client:
    response = await client.chat.completions.create(
        model=await client.models.resolve_chat(),
        messages=[msg],
        max_completion_tokens=300,
    )
    print(response.text)
```

`.file(file_data, filename=None)` takes a `data:` URL with base64 bytes **or** a
public URL. Supported: PDF, EPUB, DOCX, PPTX, XLSX/XLS, txt, md, csv, json, and
most source-code files. The builder chains `.text()` / `.image()` / `.audio()` /
`.video()` / `.file()` parts and `.build()` returns the `UserMessage`.

## Discovering model capabilities & lifecycle

Read capability and lifecycle metadata off the catalog instead of guessing:

```python
entry = await client.models.get(await client.models.resolve_chat())
print(entry.context_length)                  # int | None — max context window (typed top-level field)
caps = entry.model_spec.capabilities         # TextModelSpec.capabilities
if caps and caps.supportsReasoningEffort:
    print(caps.reasoningEffortOptions)       # e.g. ["none", "low", "medium", "high"]
    print(caps.defaultReasoningEffort)       # e.g. "low"
dep = entry.model_spec.deprecation           # ModelDeprecation | None
if dep:
    print(dep.replacementModelId, dep.startsAt, dep.removesAt, dep.autoRemap)
```

`context_length` is a typed top-level field on `ModelResponse` (mirrors
`model_spec.availableContextTokens`). `ModelDeprecation` carries
`replacementModelId` (where to migrate) plus `startsAt` / `removesAt` lifecycle
instants — check it before pinning a model. See `references/model-resolution.md`.

## Response metadata via `_response`

Every response model inherits from `VeniceBaseModel`, which auto-attaches the raw HTTP response on a private `_response` attribute. Access metadata via these properties:

| Property | Returns | When to read |
|---|---|---|
| `response.headers` | `dict[str, str] \| None` | raw headers (dict-like) |
| `response.response_rate_limits` | `RateLimitInfo \| None` | `x-ratelimit-*` parsed |
| `response.balance_info` | `BalanceInfo \| None` | x402 prepaid USDC balance |
| `response.deprecation_info` | `DeprecationInfo \| None` | model deprecation warnings |
| `response.pagination_info` | `PaginationInfo \| None` | `x-pagination-*` parsed |

```python
if response.response_rate_limits:
    remaining = response.response_rate_limits.remaining_requests
    if remaining is not None and remaining < 5:
        log.warning(f"Only {remaining} Venice requests remaining this window")

if response.deprecation_info and response.deprecation_info.is_deprecated:
    log.warning(f"Model is deprecated: {response.deprecation_info.warning}")
```

These properties are all optional (return `None` when the relevant headers are absent) — null-check before accessing.

See `references/headers-and-metadata.md` for the full surface.

## Errors

Venice exposes a typed exception tree rooted at `VeniceError`. Catch the specific class, not bare `Exception`.

```python
from venice_ai.exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    NotFoundError,
    ModelGoneError,
    APITimeoutError,
    APIConnectionError,
    PaymentRequiredError,
    MaxIterationsExceededError,
)

try:
    response = await client.chat.completions.create(...)
except RateLimitError as e:
    # e.retry_after_seconds is set when the server returned Retry-After
    await asyncio.sleep(e.retry_after_seconds or 1.0)
except ModelGoneError:
    # 410 — the model was retired. Check models.list() for the replacement
    # (model_spec.deprecation.replacementModelId) and migrate; distinct from
    # NotFoundError (404, never existed).
    raise
except AuthenticationError:
    # Don't retry — surface to the user
    raise
```

For retry policy, backoff strategy, and the full exception map, see `venice-py-production`.

## v1 → v2 migration cheat-sheet

| If you wrote (v1 / old habit) | Use in v2 |
|---|---|
| sync `VeniceClient()` with blocking calls | `async with VeniceClient()` + `await` (async is the default); `SyncVeniceClient` if you must stay sync |
| `from venice_ai import AsyncVeniceClient` | removed — `VeniceClient` IS the async client now |
| `max_tokens=N` | `max_completion_tokens=N` |
| `client.image.generate(...)` | `client.image.create(...)` |
| `client.audio.generate_music(...)` | `client.music.run(...)` (music is its own resource) |
| `client.video.queue(...)` | `client.video.run(...)` (high-level) or `client.video.submit(...)` (low-level) |
| `resp["data"]` (embeddings/models/billing/voices) | `resp.data` — responses are typed Pydantic now |
| `client.api_keys.retrieve()` as a dict | typed `ApiKey` — read fields by attribute |
| Python 3.11 / 3.12 | Python ≥3.13 required |

Run `scripts/lint_v1_usage.py <path>` to scan for v1 / legacy / non-idiomatic patterns in a codebase.

See `references/migration-v1-to-v2.md` for the full v2 breaking-change list with rationale.

## Pitfalls AI assistants reliably get wrong

1. **Hardcoded model IDs** — "the example used `claude-3-5-sonnet`" is not a defense. Use `resolve_*` always.
2. **`max_tokens=` instead of `max_completion_tokens=`** — `max_tokens` was removed in v2; old training data still suggests it.
3. **Using `text_deltas()` and expecting `final_response.usage`** — `text_deltas()` does NOT populate `final_response`. Use `collect_with_deltas()` if you want both live deltas AND final usage.
4. **Bare `async for chunk in stream:`** instead of `async with stream:` over the result of `client.chat.completions.stream(...)` — leaks the connection.
5. **OpenAI-style `create(stream=True)` + `async for chunk in stream`** — works because the SDK accepts it, but bypasses the v2 streaming idiom (`stream(...)` + `async with stream:` + `text_deltas`/`collect_with_deltas`) and skips the safer cleanup.
6. **OpenAI-style raw-dict messages** (`{"role": "user", "content": "..."}`) instead of typed helpers (`UserMessage(content="...")`) — works but bypasses Venice's type system and `venice_parameters`.
7. **Passing `tool_from_function(fn)` results in `tools=[...]` to `run_with_tools`** — pre-2.0 the SDK accepted the schema but stored the handler as `None`, producing an unrunnable agent that only failed mid-loop when the model invoked the tool. As of 2.0 this raises `ValueError` at the call site. Either way, the fix is the same: pass bare callables to `run_with_tools`. Use `Tool` objects only with the low-level `chat.completions.create(tools=[...])` path where the caller dispatches.
8. **Expecting the default `on_tool_error` to surface tool exceptions** — by design it logs them AND formats them for the model so the model can self-correct. Pass a custom `on_tool_error=` that re-raises if you want strict propagation.
9. **Treating `response.parsed` as the typed Pydantic instance** — it returns raw `dict | list | None`. For the typed instance use `response.parse_as(Cls)`, or use `client.chat.completions.parse(..., response_format=Cls)` whose `result.parsed` IS typed.
10. **Forgetting `StreamOptions(include_usage=True)`** then complaining that `final_response.usage` is None. (Both this AND `collect_with_deltas`/`collect` are required to populate it.)
11. **Calling `client.audio.generate_music(...)`** — not a v2 method; music is its own resource — use `client.music.run(...)`.
12. **`await` on `SyncVeniceClient` methods** — they're sync; the wrapper proxies an internal event loop.
13. **Catching bare `Exception`** instead of the typed subclass — drops the structured info (`retry_after_seconds`, payment instructions, deprecation warnings).
14. **Using `response.choices[0].message.content` everywhere** when `response.text` exists for single-choice cases.
15. **Music & video model duration values are model-specific enums** — `client.music.run(model="ace-step-15", duration_seconds=30)` is a 400 because that model only accepts `[60, 90, 120, 150, 180, 210]`. As of 2.0 the SDK pre-validates against `spec.duration_options` (or `min_duration` / `max_duration`) and raises `ValueError` before the HTTP call when the spec is reachable; if the catalog can't be fetched, the server is the backstop. Either way: read `client.models.get(model_id).model_spec` for the per-model tier list when picking durations.
16. **Tool function args using PEP 604 unions like `int | None`** — `tool_from_function` accepts both `Optional[T]` and `T | None` identically; either works for an optional tool parameter.

To catch these patterns automatically in user code, run **`venice-py lint <path>`** (built into the CLI on SDK ≥ 2.0.0). Reports findings in flake8-compatible `path:line:col: CODE message` format; supports `--code` filtering and `--strict`. See the venice-py CLI docs for the full rule-code table.

## References

- `references/model-resolution.md` — every `resolve_*` method and capability filter
- `references/streaming.md` — concurrent streams, animated rendering, partial-failure recovery
- `references/tool-loops.md` — hand-rolled loops, tool-error handling, `tool_from_model`
- `references/structured-output.md` — nested models, validation failures, manual JSON Schema
- `references/headers-and-metadata.md` — full table of `_response` properties
- `references/migration-v1-to-v2.md` — full v2 breaking-change list with rationale
- `references/responses-api.md` — alpha `client.responses` (OpenAI-compat)
- `references/characters-and-augment.md` — `client.characters`, `client.augment.search/scrape/parse_text`
- `references/response-shapes.md` — where fields actually live (`model_spec` per type, billing balance nesting, augment results, audio response, etc.)
- `references/billing.md` — `client.billing.*` (`get_balance`, `get_usage_history`, `iter_usage_history`, beta analytics)
- `references/decisions.md` — beta `client.decisions` (typed `noul` / `choice` / `score` judgments from "System One" models)

## Scripts

- `scripts/lint_v1_usage.py <path>` — AST-walks a directory and flags v1 / legacy / non-idiomatic patterns: `max_tokens=`, `client.image.generate(`, `client.audio.generate_music(`, hardcoded model strings on `model=` kwargs. Legacy: prefer `venice-py lint <path>` on SDK ≥ 2.0.0; same rules, discoverable via `venice-py --help`.

## Examples to read

Paths below are relative to the SDK repo's `examples/` directory. Available at [github.com/sethbang/venice-py/tree/main/examples](https://github.com/sethbang/venice-py/tree/main/examples) or in your local clone of the SDK.

- `basic/quick_start.py` — minimal client setup
- `chat/simple_chat.py` — non-streaming chat
- `chat/streaming_chat.py` — stream + usage + animated rendering
- `chat/tool_calling.py` — `tool_from_function` + low-level `create(tools=[...])` with manual tool-call dispatch
- `chat/agent_loop.py` — `run_with_tools` agent loop end-to-end with bare callables
- `chat/structured_output.py` — `response_format=BaseModel`
- `chat/multi_turn_conversation.py` — context preservation across turns
- `chat/reasoning_and_thinking.py` — `reasoning_effort` tiers
- `chat/vision.py` — vision input to chat
- `chat/file_inputs.py` — attach documents via `type:file` (data: URL or public URL)
- `chat/venice_parameters.py` — character_slug, web search, citations
- `models/model_selection.py` — `resolve_*` + capability filters
- `models/model_lifecycle.py` — context_length, deprecation, reasoning-effort metadata
- `embeddings/basic_embeddings.py` — embeddings + `cosine_similarity` helper
- `headers/header_access_example.py` — `_response` / headers / rate limits
- `decisions/ticket_routing.py` — typed `noul` / `choice` / `score` judgments, gated on `.confidence`
- `best_practices/pydantic_models.py` — full Pydantic pattern
