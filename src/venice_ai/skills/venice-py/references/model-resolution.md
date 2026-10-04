# Model resolution — every `resolve_*` method

Sourced from `Models.resolve` in `src/venice_ai/resources/models.py`. The cardinal rule is in the main `venice-py/SKILL.md`: **never hardcode model IDs**. This page is the full surface for resolving them dynamically.

## The unified `resolve()` method

```python
model_id: str = await client.models.resolve(
    type="chat",                         # "chat" | "embedding" | "image" | "video" | "tts" | "asr" | "inpaint" | "music" | "decision"
    # Chat capability filters
    require_function_calling=False,
    require_vision=False,
    require_reasoning=False,
    require_code_optimization=False,
    require_response_schema=False,
    min_context_tokens=None,             # int | None
    require_private=False,
    exclude_beta=True,
    require_web_search=False,            # chat and image
    require_reasoning_effort=None,       # chat: e.g. "none"
    require_prompt_caching=False,        # chat: lists a pricing.cache_input price
    require_multiple_images=False,       # chat: several images per request
    require_e2ee=False,                  # chat: end-to-end encrypted models only
    exclude_reasoning=False,             # chat: only models that never reason
    exclude_uncensored=False,            # chat and image: skip models flagged uncensored
    require_custom_size=False,           # image: sized by width/height (no aspectRatios)
    # Video-specific filters
    video_type=None,                     # "text-to-video" | "image-to-video" (plain) | None
    input_mode=None,                     # "image" | "reference" | "first_last_frame" | "transition" | "multi_angle"
    require_audio=False,
    min_resolution=None,                 # e.g. "720p", "1080p"
    min_duration=None,                   # e.g. "5s", "10s"
    require_duration=None,               # exact: 5 | "5" | "5s"
    require_audio_configurable=False,
    # Image / inpaint / video tiers
    require_quality=None,                # image and inpaint: e.g. "high"
    require_resolution=None,             # inpaint and video: e.g. "2K", "720p"
    # Inpaint-specific filters
    require_combine_images=False,
    require_uncensored=False,
    # Music-specific filters
    exclude_non_music=False,             # skip TTS / sound-effect / voice-changer models
    require_force_instrumental=False,
    music_duration_seconds=None,         # clip length; prefer="cheapest" quotes at it
    # General
    preferred_models=None,               # list[str] | None — priority order
    exclude_models=None,                 # list[str] | None
    prefer=None,                         # None (catalog ranking) | "cheapest"
)
```

Returns the model ID string. Raises `venice_ai.NoMatchingModelError` (a `ValueError`) if no model matches.

## `prefer="cheapest"` — rank by price

By default a resolver returns Venice's trait pick or the first catalog match, which is often an expensive model. `prefer="cheapest"` (accepted by `resolve()` and every shortcut below) returns the lowest-priced model that passes every filter instead, in strict price order:

```python
embedder = await client.models.resolve_embedding(prefer="cheapest")
agent = await client.models.resolve_chat(require_function_calling=True, prefer="cheapest")
draft_image = await client.models.resolve_image(prefer="cheapest", require_quality="low")
```

Reasoning chat models compete on price like any other, and the cheapest model is often one. A reasoning model can spend a small `max_completion_tokens` budget on thinking and return empty `content`, so when you need a direct answer, either leave reasoning models out or pick one you can switch off:

```python
direct = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
switchable = await client.models.resolve_chat(prefer="cheapest", require_reasoning_effort="none")
response = await client.chat.completions.create(
    model=switchable, messages=[UserMessage(content="...")], reasoning_effort="none"
)
```

How each type is priced (`venice_ai.models.selection.model_price`):

| Type | Price compared |
|---|---|
| chat, decision | Blended per-million-token price: `(3 × input + 1 × output) / 4` |
| embedding | Input price per million tokens |
| tts | Input price per million characters |
| asr | Price per audio second |
| image, inpaint | One request at the model's default resolution and default quality — or at `require_quality` when given |
| music | Free `POST /audio/quote` per generator at `music_duration_seconds` (see `resolve_cheapest_music`) |
| video | Free `POST /video/quote` per candidate (see `resolve_cheapest_video`) |

Rules:

- Unpriced models sort last; a missing price is never treated as $0.
- Equal prices go to the model with Venice's `default` trait, then catalog order, then model ID, so a tie lands where `prefer=None` would.
- Price order is only as good as the catalog. A model whose listed capability or price is wrong upstream can still win; check the result if the request fails.
- Beta models are skipped unless you pass `exclude_beta=False` (decision models are exempt: every one is beta).
- End-to-end encrypted chat models (`e2ee-*`) are skipped unless you pass `require_e2ee=True`. `require_private=True` alone does not re-admit them, because they need the encryption flow.
- Music implies `exclude_non_music=True`, so a per-second sound-effect model can't win, and skips models that require lyrics.
- A `preferred_models` entry that passes the filters still wins over price.

For a custom `DynamicModelSelector`, the same ranking is available as `cheapest_selector()` / `cheapest_model_strategy()` in `venice_ai.models.selection`.

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
    require_web_search=False,             # honors enable_web_search
    require_reasoning_effort=None,        # value must be in reasoningEffortOptions, e.g. "none"
    require_prompt_caching=False,         # lists a pricing.cache_input price (not a guarantee of hits)
    require_multiple_images=False,        # capabilities.supportsMultipleImages
    require_e2ee=False,                   # capabilities.supportsE2EE (needs the e2ee flow)
    exclude_reasoning=False,              # only models that never reason
    exclude_uncensored=False,             # skip models flagged uncensored
    preferred_models=None,
    exclude_models=None,
    exclude_beta=True,
    prefer=None,                          # "cheapest" to rank by price
)
```

Without `prefer`, general chat (no `require_reasoning`) prefers a model that does not reason when one passes the filters, then the `default` trait, then catalog order. A `preferred_models` entry that passes the filters wins either way.

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
| Web search | `await client.models.resolve_chat(require_web_search=True)` |
| Reasoning that can be switched off | `await client.models.resolve_chat(require_reasoning=True, require_reasoning_effort="none")` |
| Prompt caching | `await client.models.resolve_chat(require_prompt_caching=True)` |
| Several images per message | `await client.models.resolve_chat(require_multiple_images=True)` |
| Two filters | `await client.models.resolve_chat(require_function_calling=True, require_vision=True)` |
| Cheapest tool-calling model | `await client.models.resolve_chat(require_function_calling=True, prefer="cheapest")` |
| Cheapest model that answers directly | `await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)` |

### `resolve_embedding`

```python
model = await client.models.resolve_embedding(
    preferred_models=None,
    exclude_models=None,
    prefer=None,                         # "cheapest": lowest input price
)
```

No capability filters today — the embedding catalog is small. Returns the canonical embedding model, which is not the cheapest; pass `prefer="cheapest"` for that.

### `resolve_image`

```python
model = await client.models.resolve_image(
    require_web_search=False,            # spec-level supportsWebSearch (enable_web_search)
    require_quality=None,                # must be in constraints.qualities, e.g. "high"
    require_custom_size=False,           # sized by width/height (no aspectRatios constraint)
    exclude_uncensored=False,            # skip models flagged uncensored
    preferred_models=None,
    exclude_models=None,
    prefer=None,
)
```

Returns the canonical text-to-image model, or the first model that passes the filters. Models that list `aspectRatios` size by `aspect_ratio` (and `resolution`) and reject or ignore `width`/`height`; pass `require_custom_size=True` when the request sends pixel dimensions. The catalog cannot tell whether such a model returns exactly the size asked for, so check the output if it matters. For specific image operations (upscale, inpaint), use `resolve_inpaint` or pass an explicit `model=` to the method.

### `resolve_video`

```python
model = await client.models.resolve_video(
    video_type=None,                     # "text-to-video" | "image-to-video" (plain) | None (any)
    input_mode=None,                     # image-to-video input: "reference", "transition", ...
    require_audio=False,
    min_resolution=None,
    min_duration=None,
    require_duration=None,               # exact: must be in constraints.durations
    require_resolution=None,             # exact: must be in constraints.resolutions
    require_audio_configurable=False,    # accepts an explicit audio=True/False
    preferred_models=None,
    exclude_models=None,
    exclude_beta=True,
)
```

`min_duration="5s"` admits a model that only offers 10s clips. `require_duration=5` does not, so use it when the request will send `duration_seconds=5`.

Venice types plain image-to-video, reference-to-video (R2V), transition, first/last-frame and multi-angle models all `image-to-video`. `video_type="image-to-video"` returns a plain image-to-video model (one start image); pass `input_mode="reference"` (or `"transition"`, `"first_last_frame"`, `"multi_angle"`) for the others. `venice_ai.video_input_mode(model)` shows how a model is classified (by id and name, the only signal the catalog has).

### `resolve_video_upscale`

Distinct shortcut for the video-upscale path (different model catalog from generation). Returns the canonical upscaler.

### `resolve_tts` / `resolve_asr`

```python
tts_model = await client.models.resolve_tts(preferred_models=..., exclude_models=...)
asr_model = await client.models.resolve_asr(preferred_models=..., exclude_models=...)
```

The catalog has no language or timestamp flags, so these take no capability filters. Pick a TTS voice from the resolved model's `model_spec.voices`.

### `resolve_inpaint`

```python
inpaint_model = await client.models.resolve_inpaint(
    require_combine_images=False,        # constraints.combineImages, needed by multi_edit
    require_quality=None,                # must be in constraints.qualities, e.g. "low"
    require_resolution=None,             # must be in constraints.resolutions, e.g. "2K"
    require_uncensored=False,            # spec-level uncensored flag
    preferred_models=None,
    exclude_models=None,
)
```

A model without a `resolutions` constraint rejects a `resolution=` argument, so pass `require_resolution` whenever the edit will send one.

### `resolve_music`

```python
music_model = await client.models.resolve_music(
    exclude_non_music=False,             # skip TTS / sound-effect / voice-changer models typed as music
    require_force_instrumental=False,    # supports_force_instrumental
    duration_seconds=None,               # skip models that cannot make this length
    preferred_models=None,
    exclude_models=None,
    prefer=None,                         # "cheapest": quote each generator at duration_seconds
)
```

Venice types text-to-speech, sound-effect and voice-changer models as `music` too, so pass `exclude_non_music=True` when you need an actual music generator.

### `resolve_cheapest_music`

Music prices depend on the clip length, and models handle length differently: some take any length in a range, some only fixed options (ace-step makes 60-second clips at minimum), some choose the length themselves. `resolve_cheapest_music` quotes each generator (free `POST /audio/quote`) at the request that fits the length you want and returns what to submit:

```python
result = await client.models.resolve_cheapest_music(
    duration_seconds=10,                 # None = each model's shortest valid request
    require_force_instrumental=False,
    lyrics_supplied=False,               # True admits models that require lyrics_prompt
)
print(result.model, result.quote_usd, result.effective_seconds)   # effective_seconds None = model-chosen length
job = await client.music.run(model=result.model, prompt="...", **result.request_params)
```

A model that cannot make a clip that long is skipped, not quoted shorter. Equal quotes go to the clip closest to the target. `venice_ai.music_request_params(spec, target_seconds)` builds the request for a model you already chose.

### `resolve_decision`

```python
decision_model = await client.models.resolve_decision(...)
```

Returns a decision ("System One") model for `client.decisions.create()`. Unlike `resolve_chat`, this one does **not** filter beta models — every decision model is beta-flagged today, so excluding them would leave no candidates. See `references/decisions.md`.

## `resolve_cheapest_video` — the price-aware shortcut

Video generation is the most expensive Venice modality and prices vary widely between models. `resolve_cheapest_video` issues one free `POST /video/quote` per candidate and picks the lowest. Each model is quoted at its *own* cheapest valid request — shortest listed duration, lowest listed resolution, 16:9 when listed, and `audio=False` where audio is configurable — so a model without a 5-second clip is still compared:

```python
result = await client.models.resolve_cheapest_video(
    video_type="text-to-video",          # filter; None = both directions (upscalers skipped)
    duration=None,                       # pin a duration; models that don't list it are skipped
    resolution=None,                     # pin a resolution tier, same rule
    audio=False,                         # sent where configurable; True also requires audio; None = model default
    aspect_ratio=None,                   # None prefers 16:9
    require_audio_configurable=False,
    exclude_models=None,
    exclude_beta=True,
    max_concurrency=8,                   # quote requests in flight; no extra retries beyond the client's policy
)
print(f"Cheapest: {result.model} at ${result.quote_usd} with {result.request_params}")
job = await client.video.submit(model=result.model, prompt="...", **result.request_params)
```

Returns a `CheapestVideoResult` with `.model`, `.quote_usd`, `.request_params` (the quote/submit kwargs the winner was priced with — submit with them to be billed that quote), `.all_quotes` (`dict[str, float]`, every successful quote) and `.skipped` (candidate → reason it was not quoted or its quote failed).

`resolve_video(prefer="cheapest")` runs the same quoting over the `resolve_video` filters and returns just the model ID. To build the cheapest request for a model you already chose, use `venice_ai.cheapest_video_params(constraints)`. With `video_type="image-to-video"` only plain image-to-video models are quoted; pass `input_mode=` for reference, transition, first/last-frame or multi-angle models.

**Cost note**: quotes are free, but this makes one request per candidate (about 50 for text-to-video). Cache the result if you call it in a loop.

## When resolution fails

`resolve_*` raises `venice_ai.NoMatchingModelError` if no model matches the criteria. It subclasses `ValueError`, so `except ValueError` also catches it. Common causes:

- Filter too narrow (e.g., `require_function_calling=True` AND `require_vision=True` AND `min_context_tokens=200_000`).
- Excluded all candidates via `exclude_models`.
- Beta-only candidates with `exclude_beta=True`.

Recover gracefully:

```python
from venice_ai import NoMatchingModelError

try:
    model = await client.models.resolve_chat(require_vision=True, require_function_calling=True)
except NoMatchingModelError as e:
    log.warning("strict capability filters yielded no model; falling back", error=str(e))
    model = await client.models.resolve_chat(require_function_calling=True)
```

The quote-ranked resolvers (`resolve_cheapest_video`, `resolve_cheapest_music`, and `resolve(..., prefer="cheapest")` for video and music) can fail a second way. When models match but every quote call fails, they raise `venice_ai.ModelQuotesUnavailableError` instead: a rate limit, a server error or outage, a connection failure, or Venice rejecting the quote request, not an empty catalog. Its `failures` maps each model ID to the exception its quote raised, and `skipped` lists the candidates rejected before quoting. Treat `NoMatchingModelError` as "skip this feature" and `ModelQuotesUnavailableError` as a failure:

```python
from venice_ai import ModelQuotesUnavailableError, NoMatchingModelError

try:
    pick = await client.models.resolve_cheapest_music(duration_seconds=30)
except NoMatchingModelError:
    return None  # no generator can make this clip: skip
except ModelQuotesUnavailableError as e:
    raise RuntimeError(f"music quotes failed: {e.failures}") from e
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
| `require_web_search` (chat, image) | Model honors `enable_web_search` | Search-grounded chat, `image.create(enable_web_search=True)` |
| `require_reasoning_effort` (chat) | Value is listed in `reasoningEffortOptions` | `reasoning_effort="none"` to switch reasoning off |
| `require_prompt_caching` (chat) | Model lists a `pricing.cache_input` price (no guarantee of hits) | Long, repeated system prompts |
| `exclude_reasoning` (chat) | Model never reasons (`supportsReasoning` false) | Direct answers under a small token budget |
| `exclude_uncensored` (chat, image) | Skip models Venice flags `uncensored` | General-audience output |
| `require_custom_size` (image only) | Model sizes by `width`/`height` (no `aspectRatios`) | `image.create(width=768, height=512)` |
| `input_mode` (video only) | What an image-to-video model takes as input | `input_mode="reference"` for R2V |
| `require_multiple_images` (chat) | Model accepts several images in one request | Comparing screenshots |
| `require_e2ee` (chat) | Model is end-to-end encrypted (`supportsE2EE`) | `chat.completions.create(e2ee=...)` |
| `exclude_non_music` (music only) | Skip TTS, sound-effect and voice-changer models typed as music | `music.generate(...)` |
| `require_force_instrumental` (music only) | Model supports `force_instrumental` | Instrumental tracks |
| `require_audio` (video only) | Video model includes audio track | Cinematic outputs |
| `require_audio_configurable` (video only) | Video model accepts an explicit `audio=True/False` | Silent clips |
| `min_resolution` (video only) | Video model supports ≥ this resolution | High-quality video output |
| `require_resolution` (inpaint, video) | Tier is listed in `constraints.resolutions` | `edit(resolution="2K")`, `video.create(resolution="720p")` |
| `require_duration` (video only) | Exact duration is listed in `constraints.durations` | `duration_seconds=5` |
| `require_quality` (image, inpaint) | Tier is listed in `constraints.qualities` | `quality="high"` |
| `require_combine_images` (inpaint only) | Model can combine several input images | `multi_edit(...)` |
| `require_uncensored` (inpaint only) | Venice flags the model `uncensored` | `edit(safe_mode=False)` |

Quality, resolution and duration values match without regard to case. Every filter narrows the pool before `preferred_models` is consulted, so a preferred model that fails a filter is skipped.

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
- **Using `cheapest=True` / `most_intelligent=True` kwargs**: those aren't real. Price ranking is `prefer="cheapest"`; the capability filters listed above are the rest of the surface.

## Related references

- `migration-v1-to-v2.md` — the unified `resolve()` API replaces the v1 `create_model_selector()` + `selector.select_*()` two-step.
- `tool-loops.md` — `require_function_calling=True` is mandatory for `run_with_tools`.
- `structured-output.md` — `require_response_schema=True` is mandatory for `parse()`.
- `venice-py-multimodal/references/video.md` — `resolve_cheapest_video` patterns.
