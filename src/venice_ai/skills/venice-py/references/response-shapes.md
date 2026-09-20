# Response shapes — what fields actually live where

Sourced from `src/venice_ai/types/api/`. This page exists because agents repeatedly write defensive `getattr` ladders for shapes that are statically typed, OR they confidently access flat fields on responses that nest one level deeper. Either way the cost is real — read this once instead of probing `dir(response)` mid-task.

## `client.models.get(...)` — `model.model_spec` per type

`ModelResponse` carries a `type: str` and a `model_spec` whose subclass is dispatched by that type. `type` is an open string, not a `Literal`: Venice adds model types to the live catalog between SDK releases (`decision` was the first), and a closed enum failed the whole `/models` parse the first time one appeared. Narrow against `KNOWN_MODEL_TYPES` (`text`, `image`, `video`, `inpaint`, `music`, `tts`, `asr`, `embedding`, `upscale`, `decision`) and treat anything outside it as a newer type rather than an invalid one. There is **no shared base** beyond `ModelSpec`; treat the per-type fields below as canonical and use `getattr(spec, name, None)` in generic helpers.

| `model.type` | Spec class | Distinguishing fields |
|---|---|---|
| `text` | `TextModelSpec` | `availableContextTokens`, `maxCompletionTokens`, `capabilities.{supportsVision, supportsFunctionCalling, supportsResponseSchema, supportsReasoning, supportsWebSearch, ...}`, `pricing.{input, output}.usd` (`LLMModelPricing`) |
| `image` | `ImageModelSpec` | `constraints.{promptCharacterLimit, steps, ...}`, `supportsWebSearch`, `pricing.generation.usd`, `pricing.upscale.*` (`ImageModelPricing`) |
| `video` | `VideoModelSpec` | `constraints.{model_type, aspect_ratios, resolutions, durations, audio, ...}`, pricing varies by model |
| `inpaint` | `InpaintModelSpec` | `constraints.{promptCharacterLimit, image_combination, ...}` |
| `music` | `MusicModelSpec` | `voices`, `default_voice`, `duration_options` (enum) OR `min_duration` / `max_duration` (range), `default_duration`, `supported_formats`, `default_format`, `prompt_character_limit`, `lyrics_character_limit`, `min_prompt_length`, `supports_lyrics`, `lyrics_required`, `supports_lyrics_optimizer`, `supports_force_instrumental`, `supports_language_code`, `supports_speed`, `min_speed`, `max_speed`, `default_speed`, `pricing.durations[<dur>].usd` (`MusicModelPricing`) |
| `tts` | `TtsModelSpec` | `voices: list[str]`, `default_voice`, `pricing.input.usd` (`AudioModelPricing`) |
| `asr` | `AsrModelSpec` | (no type-specific fields today), `pricing.per_audio_second.usd` (`ASRModelPricing`) |
| `embedding` | `EmbeddingModelSpec` | `embeddingDimensions`, `maxInputTokens`, `supportsCustomDimensions`, `pricing.{input, output}.usd` (`LLMModelPricing`) |
| `upscale` | `UpscaleModelSpec` | image upscaler — see `UpscaleModelSpec` in `src/venice_ai/types/api/models.py` |

### camelCase / snake_case is NOT consistent within `model_spec`

This trips up generic helpers. Field names mirror the wire and the wire is not consistent across types:

- **camelCase**: `availableContextTokens` (text), `maxCompletionTokens` (text), `embeddingDimensions` (embedding), `maxInputTokens` (embedding), `supportsCustomDimensions` (embedding), `supportsWebSearch` (image), `supportsVision` / `supportsFunctionCalling` / etc. (`ModelCapabilities`), `promptCharacterLimit` (image constraints).
- **snake_case**: every field on `MusicModelSpec` (`duration_options`, `min_duration`, `default_duration`, `prompt_character_limit`, `supports_lyrics`, `supports_force_instrumental`, ...), `default_voice` on `TtsModelSpec`, `model_type` / `aspect_ratios` on `VideoModelConstraints`.

Notably `prompt_character_limit` is **snake_case on music but `promptCharacterLimit` on image** — same concept, different casing. A generic "extract a prompt cap" helper has to try both spellings or hardcode per-type. There's no sentinel field on `ModelSpec` to dispatch on; use `model.type` and the table above.

### Pricing accessor varies by capability

`model.model_spec.pricing` is a discriminated union — the path depends on the model type:

| Type | Path | Unit |
|---|---|---|
| `text`, `embedding` | `.input.usd` / `.output.usd` (`LLMModelPricing`) | USD per million tokens |
| `image` | `.generation.usd` (`ImageModelPricing`) | USD per image |
| `tts` | `.input.usd` (`AudioModelPricing`) | USD per million input characters |
| `asr` | `.per_audio_second.usd` (`ASRModelPricing`) | USD per audio second |
| `music` | `.durations[<duration>].usd` (`MusicModelPricing`) | USD per duration tier |
| `video` | varies by model (`VideoResolutionPricing` keyed under `model_sets`) | varies |

There is no `pricing.unit` discriminator — branch on `model.type` (or `isinstance(spec.pricing, LLMModelPricing)` etc.) when building a unified cost surface.

## `client.billing.get_balance()` — the `.balances.` nesting

```python
balance = await client.billing.get_balance()
# BillingBalanceResponse:
#   balance.can_consume          : bool | None
#   balance.consumption_currency : Literal["USD", "VCU", "DIEM", "BUNDLED_CREDITS"] | None
#   balance.balances             : BillingBalances | None  ← NESTED
#   balance.diem_epoch_allocation: float | None
# BillingBalances:
#   balance.balances.usd  : float | None
#   balance.balances.diem : float | None
```

**The trap:** `balance.usd` does not exist. Read `balance.balances.usd` (with a None-check on `balances`). `BillingBalanceResponse` uses `populate_by_name=True`, so the snake_case attribute names work even though the wire is `canConsume` / `consumptionCurrency` / `diemEpochAllocation` (alias-mapped).

This is a different surface from x402 prepaid balance (`client.x402.balance(...)`), which has its own shape — see `venice-py-x402/references/balance-and-topup.md`. Don't confuse the two.

## `client.augment.search(...)` — fixed shape, no defensive `getattr` needed

```python
response = await client.augment.search(query="...", limit=10)
# AugmentSearchResponse:
#   response.query   : str
#   response.results : list[AugmentSearchResult]
# AugmentSearchResult:
#   result.title   : str
#   result.url     : str
#   result.content : str    ← snippet/extract, NOT `snippet` or `description`
#   result.date    : str | None
```

The shape is statically typed as `AugmentSearchResult` in `src/venice_ai/types/api/augment.py` — there is no `.hits`, no `.data`, no per-result `.snippet` or `.description`. If you find yourself writing `getattr(r, "snippet", None) or getattr(r, "description", None)`, stop — it's `r.content`.

## `client.augment.scrape(...)`

```python
response = await client.augment.scrape(url="https://...")
# AugmentScrapeResponse: url: str, content: str (markdown), format: str (always "markdown")
```

## `client.augment.parse_text(...)`

Returns either `AugmentTextParserResponse{text: str, tokens: int}` (when `response_format="json"`) or a plain `str` (when `response_format="text"`). The SDK handles both shapes transparently.

## `client.audio.create_speech(...)` — `AudioResponse.content`

```python
response = await client.audio.create_speech(model=..., input="...", voice="af_heart")
# AudioResponse:
audio_bytes = response.content           # raw audio bytes (the documented accessor)
response.save(Path("out.mp3"))           # write to disk (sync; use asyncio.to_thread for big files)
for chunk in response.iter_bytes():
    ...
```

There is no `response.audio`, no `response.bytes()`, and no `response.read()`. The raw bytes live on `response.content`; `save()` and `iter_bytes()` are the other accessors. When `stream=True` the return type changes to `AsyncIterator[bytes]` — see `venice-py-multimodal/references/audio-tts-stt.md`.

## `client.audio.get_voices(...)` — voice catalog is per model

```python
catalog = await client.audio.get_voices(model_id=await client.models.resolve_tts())
# VoiceList:
#   catalog.data                : list[VoiceDetail]
#   catalog.region_code_filter  : str | None
# VoiceDetail:
#   voice.id, voice.gender, voice.region_code, voice.language, voice.accent
```

`get_voices()` with no `model_id` returns the union across all TTS models — voices in the union are NOT portable. `af_alloy` is a `tts-kokoro` voice; calling `audio.create_speech(model="not-kokoro", voice="af_alloy")` returns 400. Always pass `model_id=` to scope, or ask the user to pick after model selection.

## `client.models.list(type=...)` — defaults to `"all"`

```python
listing = await client.models.list()                  # union of every type (~248)
text    = await client.models.list(type="text")        # text-only
image   = await client.models.list(type="image")
```

The SDK auto-passes `type="all"` when the caller omits the kwarg. The server's own default is `text`-only — don't rely on it. Valid values are typed as a `Literal`: passing `"stt"` (should be `"asr"`) or `"embeddings"` (should be `"embedding"`) is a static type error.

## `client.characters.list(...)` — `offset` (not `page`)

```python
response = await client.characters.list(limit=20, offset=0, sort_by="highlyRated")
# CharactersListResponse:
#   response.data : list[Character]
```

Pagination uses `offset`, not `page`. The return type is the response object — iterate `response.data`. For unbounded enumeration, prefer `client.characters.iter_all(...)`.

## See also

- `venice-py/references/billing.md` — full `client.billing.*` surface
- `venice-py-multimodal/references/audio-tts-stt.md` — voice scoping, formats, streaming
- `venice-py-x402/references/balance-and-topup.md` — distinct from `client.billing.get_balance()`
- `venice-py/references/headers-and-metadata.md` — `_response`, headers, rate limits, deprecation info
