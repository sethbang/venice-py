# Model resolution — every `resolve_*` method

Sourced from `Models.resolve` in `src/venice_ai/resources/models.py`. The cardinal rule is in the main `venice-py/SKILL.md`: **never hardcode model IDs**. This page is the full surface for resolving them dynamically.

## The unified `resolve()` method

```python
model_id: str = await client.models.resolve(
    type="chat",                         # "chat" | "embedding" | "image" | "video" | "tts" | "asr" | "inpaint" | "music"
    # Chat capability filters
    require_function_calling=False,
    require_vision=False,
    require_reasoning=False,
    require_code_optimization=False,
    require_response_schema=False,
    min_context_tokens=None,             # int | None
    require_private=False,
    exclude_beta=True,
    # Video-specific filters
    video_type=None,                     # "text-to-video" | "image-to-video" | None
    require_audio=False,
    min_resolution=None,                 # e.g. "720p", "1080p"
    min_duration=None,                   # e.g. "5s", "10s"
    # General
    preferred_models=None,               # list[str] | None — priority order
    exclude_models=None,                 # list[str] | None
)
```

Returns the model ID string. Raises `ValueError` if no model matches.

## Type-specific shortcuts

Each shortcut is a thin wrapper around `resolve(type=..., ...)` that exposes only the relevant filters. **Use the shortcut** — it's clearer at the call site.

### `resolve_chat`

```python
model = await client.models.resolve_chat(
    require_function_calling=False,
    require_vision=False,
    require_reasoning=False,
    require_code_optimization=False,
    require_response_schema=False,       # for `parse()` / `response_format=BaseModel`
    min_context_tokens=None,
    require_private=False,                # privacy-first models
    preferred_models=None,
    exclude_models=None,
    exclude_beta=True,
)
```

Common usages:

| Goal | Call |
|---|---|
| Default chat | `await client.models.resolve_chat()` |
| Tool-calling agent | `await client.models.resolve_chat(require_function_calling=True)` |
| Vision input | `await client.models.resolve_chat(require_vision=True)` |
| Long-context | `await client.models.resolve_chat(min_context_tokens=128_000)` |
| Reasoning model | `await client.models.resolve_chat(require_reasoning=True)` |
| Structured output | `await client.models.resolve_chat(require_response_schema=True)` |
| Code-optimized | `await client.models.resolve_chat(require_code_optimization=True)` |
| Privacy / TEE | `await client.models.resolve_chat(require_private=True)` |
| Two filters | `await client.models.resolve_chat(require_function_calling=True, require_vision=True)` |

### `resolve_embedding`

```python
model = await client.models.resolve_embedding(
    preferred_models=None,
    exclude_models=None,
)
```

No capability filters today — the embedding catalog is small. Returns the canonical embedding model.

### `resolve_image`

```python
model = await client.models.resolve_image(
    preferred_models=None,
    exclude_models=None,
)
```

Returns the canonical text-to-image model. For specific image operations (upscale, inpaint), use `resolve_inpaint` or pass an explicit `model=` to the method.

### `resolve_video`

```python
model = await client.models.resolve_video(
    video_type=None,                     # "text-to-video" | "image-to-video" | None (any)
    require_audio=False,
    min_resolution=None,
    min_duration=None,
    preferred_models=None,
    exclude_models=None,
    exclude_beta=True,
)
```

### `resolve_video_upscale`

Distinct shortcut for the video-upscale path (different model catalog from generation). Returns the canonical upscaler.

### `resolve_tts` / `resolve_asr`

```python
tts_model = await client.models.resolve_tts(preferred_models=..., exclude_models=...)
asr_model = await client.models.resolve_asr(preferred_models=..., exclude_models=...)
```

### `resolve_inpaint` / `resolve_music`

```python
inpaint_model = await client.models.resolve_inpaint(...)
music_model   = await client.models.resolve_music(...)
```

### `resolve_decision`

```python
decision_model = await client.models.resolve_decision(...)
```

Returns a decision ("System One") model for `client.decisions.create()`. Unlike `resolve_chat`, this one does **not** filter beta models — every decision model is beta-flagged today, so excluding them would leave no candidates. See `references/decisions.md`.

## `resolve_cheapest_video` — the price-aware shortcut

Video generation is the most expensive Venice modality and prices vary widely between models. `resolve_cheapest_video` issues one `POST /video/quote` per candidate and picks the lowest:

```python
result = await client.models.resolve_cheapest_video(
    duration="5s",                       # required-ish; default "5s"
    video_type="text-to-video",          # filter
    resolution="1080p",                  # filter
    audio=None,                          # bool | None
    aspect_ratio=None,                   # str | None
    exclude_models=["expensive-model-id"],
    exclude_beta=True,
)
print(f"Cheapest: {result.model} at ${result.quote_usd}")
print(f"All quotes:")
for model_id, price in result.all_quotes.items():
    print(f"  {model_id}: ${price}")
```

Returns a `CheapestVideoResult` with `.model`, `.quote_usd`, and `.all_quotes` (a `dict[str, float]` mapping each candidate model ID to its USD price, for transparency).

**Cost note**: this method makes N quote calls (where N = number of candidates). The quote endpoint is cheap but not free. Cache the result if you call this in a loop.

## When resolution fails

`resolve_*` raises `ValueError` if no model matches the criteria. Common causes:

- Filter too narrow (e.g., `require_function_calling=True` AND `require_vision=True` AND `min_context_tokens=200_000`).
- Excluded all candidates via `exclude_models`.
- Beta-only candidates with `exclude_beta=True`.

Recover gracefully:

```python
try:
    model = await client.models.resolve_chat(require_vision=True, require_function_calling=True)
except ValueError as e:
    log.warning("strict capability filters yielded no model; falling back", error=str(e))
    model = await client.models.resolve_chat(require_function_calling=True)
```

## `preferred_models` ordering

`preferred_models=["a", "b", "c"]` says: "if model `a` matches the filters, return it; else try `b`; else `c`; else any match." Useful when you have an opinion about which model to use but want the resolver to handle availability:

```python
model = await client.models.resolve_chat(
    preferred_models=["zai-org-glm-4.7", "venice-uncensored-1-2"],
    require_function_calling=True,
)
```

If neither preferred model is available (deprecation, region restrictions, etc.), the resolver falls back to any function-calling chat model.

## Capability filter cheat sheet

| Filter | Meaning | Example use |
|---|---|---|
| `require_function_calling` | Model supports `tools=[...]` and emits structured tool calls | Anything using `run_with_tools` |
| `require_vision` | Model accepts image content blocks in messages | OCR, screenshot analysis, multimodal chat |
| `require_reasoning` | Model has explicit reasoning / thinking-block support | Complex multi-step problems |
| `require_code_optimization` | Model is tuned for code generation/explanation | Coding agents, codereview |
| `require_response_schema` | Model supports strict structured output (`response_format=BaseModel`) | `client.chat.completions.parse(...)` |
| `min_context_tokens` | Model's context window ≥ this many tokens | Long documents, multi-turn agents |
| `require_private` | Model is in Venice's privacy-first / TEE-backed tier | Compliance-sensitive workloads |
| `exclude_beta` | Skip models marked beta | Production stability |
| `require_audio` (video only) | Video model includes audio track | Cinematic outputs |
| `min_resolution` (video only) | Video model supports ≥ this resolution | High-quality video output |

## Listing models

If you need the full catalog (e.g., to build a model-picker UI):

```python
catalog = await client.models.list(type="chat")
for entry in catalog.data:
    print(f"{entry.id}: {entry.model_spec.capabilities if entry.model_spec else '(unknown)'}")
```

`client.models.list()` returns the full catalog with capability metadata. `client.models.list_traits(type="text")` returns named traits (e.g., `traits.data["fastest"]` = a model ID) — useful for "give me the fastest chat model" without manual filtering.

## Model metadata: context_length, capabilities, deprecation

Each `ModelResponse` carries lifecycle and capability metadata you can read
instead of guessing or trial-and-error:

```python
from venice_ai.types.api import TextModelSpec

entry = await client.models.get(await client.models.resolve_chat())

# context_length — typed top-level field (mirrors model_spec.availableContextTokens)
print(entry.context_length)                  # int | None (None for non-text models)

spec = entry.model_spec
if isinstance(spec, TextModelSpec) and spec.capabilities:
    caps = spec.capabilities
    if caps.supportsReasoningEffort:
        print(caps.reasoningEffortOptions)   # accepted reasoning_effort tiers, e.g. ["none","low","medium","high"]
        print(caps.defaultReasoningEffort)   # the default when a request omits one

# Deprecation lifecycle (ModelSpec.deprecation) — present when retirement is scheduled
dep = spec.deprecation
if dep:
    print(dep.replacementModelId)            # where to migrate, when one exists
    print(dep.startsAt, dep.removesAt)       # ISO 8601: warnings active / dropped from GET /models
    print(dep.autoRemap)                     # True ⇒ Venice silently re-routes the retired ID
```

| Field | Lives on | Meaning |
|---|---|---|
| `context_length` | `ModelResponse` (top-level) | max context window in tokens (`int \| None`) |
| `reasoningEffortOptions` / `defaultReasoningEffort` | `TextModelSpec.capabilities` | accepted `reasoning_effort` tiers + default |
| `qualities` / `defaultQuality` | `ImageModelSpec.constraints` | image quality tiers (see `venice-py-multimodal`) |
| `deprecation` (`replacementModelId`, `startsAt`, `removesAt`, `autoRemap`, `date`) | `ModelSpec.deprecation` | retirement lifecycle |

A retired model returns `410` → `ModelGoneError` (distinct from `NotFoundError`);
check `deprecation.replacementModelId` to migrate. See `examples/models/model_lifecycle.py`.

## Common bugs

- **Hardcoding model IDs**: `model="some-llm-v3"`. The whole point of resolvers is to survive deprecation; hardcoding defeats it.
- **Calling `resolve_chat()` synchronously**: it's an async coroutine. Always `await`.
- **Constructing `client.models.resolve_chat_default()`**: that method doesn't exist. The default-shortcut method is `resolve_chat()` (no `_default` suffix).
- **Passing OpenAI-style `model="auto"`**: there's no `"auto"` resolver string. Use a real `resolve_*` call.
- **Using `cheapest=True` / `most_intelligent=True` kwargs**: those aren't real. The capability filters listed above are the real surface.

## Related references

- `migration-v1-to-v2.md` — the unified `resolve()` API replaces the v1 `create_model_selector()` + `selector.select_*()` two-step.
- `tool-loops.md` — `require_function_calling=True` is mandatory for `run_with_tools`.
- `structured-output.md` — `require_response_schema=True` is mandatory for `parse()`.
- `venice-py-multimodal/references/video.md` — `resolve_cheapest_video` patterns.
