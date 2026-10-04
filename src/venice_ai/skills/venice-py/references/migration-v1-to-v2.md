# Migrating from venice-ai v1 to v2

v2.0.0 is a ground-up rewrite. There are no deprecation shims. This page maps what a **v1.3.x** user must change, in rough order of impact.

## 1. The client is async by default (biggest change)

In v1, `VeniceClient` was **synchronous** and `AsyncVeniceClient` was the async one. In v2:

- `VeniceClient` **is** the async client — its methods are coroutines, `await` them.
- `AsyncVeniceClient` is **removed** (`from venice_ai import AsyncVeniceClient` → `ImportError`).
- `SyncVeniceClient` is the new sync client (`with SyncVeniceClient() as client:`), for code that genuinely can't go async.

```
# v1 (sync)
client = VeniceClient(api_key="...")
resp = client.chat.completions.create(model=m, messages=[...])

# v2 (async)
async with VeniceClient() as client:
    resp = await client.chat.completions.create(
        model=await client.models.resolve_chat(), messages=[...]
    )
```

## 2. Required environment

| Concern | v1.3.x | v2 |
|---|---|---|
| Python | ≥3.11 | **≥3.13** |
| Install | `pip install venice-ai` | `pip install venice-py`; optional extras `[x402]`, `[redis]`, `[adaptive]`, `[e2ee]` |

The v1 line was published as `venice-ai` and has no release under this name, so there is no
wrong version to land on — on Python ≤3.12 pip reports that no matching distribution exists.

If you're stuck on Python ≤3.12, stay on the latest v1 release by pinning `venice-ai<2`.

## 3. Renamed methods

| v1.3.x | v2 |
|---|---|
| `client.image.generate(...)` | `client.image.create(...)` |

`client.image.get_available_styles()` was a redundant duplicate **removed** in v2 — use `client.image.list_styles()`, which already existed in v1.3.x (not a rename).

`image.upscale`, `audio.create_speech`, `audio.get_voices`, `embeddings.create`, and `models.list` / `list_traits` / `list_compatibility` keep their names. `image.simple_generate` also keeps its name but (like every method) is now async, and its return type was renamed `SimpleImageResponse` → `SimpleImageGenerationResponse`.

## 4. Removed parameter

`max_tokens=N` (deprecated in v1) → **`max_completion_tokens=N`**. Removed without an alias; if you grep for `max_tokens=`, every hit is a bug.

## 5. Responses are now typed Pydantic models

In v1, several endpoints returned plain `TypedDict`s (subscript access). In v2 they return Pydantic models, so `resp["data"]` → `resp.data` (**subscripting now raises `TypeError`**). Envelope field names are preserved (the models `model_spec` pricing `vcu` became `diem`). Affects `embeddings.create`, `models.list` / `list_traits` / `list_compatibility`, and `audio.get_voices`. Also:

- `audio.create_speech(...)` returns `AudioResponse` (raw bytes on `.content`, or use `.save(path)`), not raw `bytes`.
- `api_keys.retrieve()` and `delete()` return typed models (`ApiKey`, `DeleteApiKeyResponse`), not dicts — read by attribute (`api_key.description`).
- `billing.get_usage(...)` is now **`billing.get_usage_history(...)`** (the `/billing/usage` endpoint was deprecated upstream in favour of `/billing/usage-history`). It moved to cursor/keyset pagination: params renamed (`startDate`/`endDate` → `startTimestamp`/`endTimestamp`, `limit` → `pageSize`) and `page`/`sortOrder` are gone (the walk is always ascending). The response is `BillingUsageHistoryResponse` (`.data` + `.nextCursor`) instead of `.data` + `.pagination`. Walk every page with `iter_usage_history(...)`, which threads the cursor for you. With `format=BillingFormatEnum.CSV` it returns a `BillingUsageHistoryCsvPage` rather than `bytes`: the document is `.content` (`.text` decoded), the continuation token from the `x-next-cursor` header is `.nextCursor`, and `.filename` is the server's file name; `iter_usage_history_csv(...)` exports a whole window page by page.
- `chat` and `characters` responses were already typed in v1 (unaffected).
- `client.get_model_pricing(model_id)` is **removed**. Read pricing off the model entry instead: `(await client.models.get(model_id)).model_spec.pricing` (an `LLMModelPricing` for chat and embedding models; image/video/music specs carry their own pricing shapes). For a whole catalog, `CostTracker.from_client(client)` builds the `{model_id: pricing}` map in one call.

## 6. Type import paths moved

Per-resource type modules moved under `venice_ai.types.api`: `venice_ai.types.image` → `venice_ai.types.api.images` (likewise `models`, `api_keys`, `billing`, `characters`, `embeddings`). Some response classes were renamed (`ChatCompletion` → `ChatCompletionResponse`, `ImageResponse` → `ImageGenerationResponse`). If you only read responses by attribute you won't notice; update direct type imports.

## New in v2 (additive — no migration needed)

Entirely new resources: `client.video`, `client.music`, `client.crypto`, `client.augment`, `client.x402`, `client.responses`, `client.tee`. New methods on existing resources: `audio.transcribe` / `audio.create_voice`, `api_keys.update`, `chat.completions.parse` / `stream` / `run_with_tools`. Plus capability-based model resolution (`client.models.resolve_*()` — use instead of hardcoded model IDs), a `venice-py` CLI, rate limiting, `client.gather([...], max_concurrency=N)`, response `.save()` / `.save_all()`, and real TEE client-side E2EE (`enable_e2ee` / `e2ee=True`) for confidential-compute models.

The new async-job resources (`video`, `music`) use a consistent verb scheme — `submit()` (low-level), `run()` (high-level lifecycle manager via `async with`), and a cleanup verb that deletes stored output: `cancel()` on video, `release()` on music.

## Patterns the linter flags (not real v2 methods)

`venice-py lint <path>` flags calls that won't run on v2 — including method names that look plausible but **do not exist in v2** (often from OpenAI habits or stale docs). These are *not* old v1 methods; they're mistakes to avoid:

- `client.audio.generate_music(...)` → use `client.music.run(...)` (V102)
- `client.audio.generate_speech(...)` → use `client.audio.create_speech(...)` (V103)
- `client.video.queue(...)` → use `client.video.run(...)` / `submit(...)` (V104)
- `client.video.complete(...)` → use `client.video.cancel(...)` (V105)
- `client.embeddings.generate(...)` → use `client.embeddings.create(...)` (V106)

plus the genuine v1 patterns it catches: `AsyncVeniceClient` (V100), `client.image.generate(...)` (V101), and `max_tokens=`.

```bash
venice-py lint src/     # ships with SDK >= 2.0.0; discoverable via venice-py --help
```

(The skill also ships a standalone copy at `scripts/lint_v1_usage.py` you can run directly.)

## Related references

- `headers-and-metadata.md` — `_response` and per-response info properties (rate limits, deprecation, balance).
- `model-resolution.md` — full `resolve_*` table.
- `tool-loops.md` — the `run_with_tools` API.
- `structured-output.md` — `parse()` vs `create() + parse_as()`.
