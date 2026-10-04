# Migration Guide: v1 → v2

**Venice AI Python SDK v2.0.0 is a ground-up rewrite.** The client is now async-first, most
responses are typed Pydantic models, and a large set of new resources (video, music,
crypto, augment, x402, a CLI, rate limiting, and more) ships alongside the v1 surface.

This guide covers everything a **v1.3.x** user must change to move to v2.0.0, in
rough order of impact. For the complete list of additions, see the
[CHANGELOG](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md).

---

## 0. Change the package name — that is also your version floor

v2 requires **Python 3.13+**, and it is published under a different name than v1. The
v1 line (1.3.x and earlier) exists only as `venice-ai`; v2 lives on PyPI as
[`venice-py`](https://pypi.org/project/venice-py/).

That rename does the job the old `>=2` pin used to do. Under `venice-ai`, a bare install
on Python 3.12 or older resolved to the newest v1 release *silently* — no error, no
warning — so the first sign of trouble was an `ImportError` on a v2 name. `venice-py` has
no v1 to fall back to, so an interpreter that is too old fails loudly and says why:

```console
$ pip install venice-py
ERROR: Ignored the following versions that require a different python version: ...
ERROR: No matching distribution found for venice-py
```

Change the name everywhere the dependency is declared, not just in the command you type —
`requirements.txt`, `pyproject.toml`, lockfiles, Dockerfiles, CI installs:

```text
# requirements.txt
venice-py
```

```toml
# pyproject.toml
dependencies = ["venice-py"]
```

**Your imports do not change.** The distribution is `venice-py`; the module it installs is
still `venice_ai`, exactly as before:

```python
from venice_ai import VeniceClient   # unchanged
```

Staying on v1 for now is a legitimate choice — keep `venice-ai<2`, which pins you to the
old name, where the v1 line lives.

---

## 1. The client is now async by default

This is the largest change. In v1, `VeniceClient` was **synchronous** and a separate
`AsyncVeniceClient` provided the async API. In v2 those roles changed:

- **`VeniceClient` is now asynchronous.** Its methods are coroutines and must be `await`ed.
- **`AsyncVeniceClient` has been removed.** `from venice_ai import AsyncVeniceClient` raises `ImportError`.
- **`SyncVeniceClient` is the new synchronous client** for code that genuinely can't go async.

**Before (v1) — synchronous `VeniceClient`:**
```python
from venice_ai import VeniceClient

client = VeniceClient(api_key="...")
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

**After (v2) — async by default:**
```python
import asyncio
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage

async def main():
    async with VeniceClient() as client:                 # reads VENICE_API_KEY from env
        response = await client.chat.completions.create(
            model=await client.models.resolve_chat(),
            messages=[UserMessage(content="Hello")],
        )
        print(response.text)

asyncio.run(main())
```

**If you must stay synchronous**, swap in `SyncVeniceClient` — same surface, no `await`:
```python
from venice_ai import SyncVeniceClient

with SyncVeniceClient() as client:
    response = client.chat.completions.create(
        model=client.models.resolve_chat(),
        messages=[UserMessage(content="Hello")],
    )
    print(response.text)
```

**What to do:**
- If you used `AsyncVeniceClient`, rename it to `VeniceClient` (it is now the async client).
- If you used the synchronous `VeniceClient`, either move your calls into an async
  context and `await` them, or switch to `SyncVeniceClient`.

---

## 2. Python ≥3.13 required

v2.0.0 requires **Python ≥3.13**. v1.3.x supported Python 3.11–3.12; support for
both has been dropped.

**What to do:** Upgrade your runtime to 3.13 or later before upgrading the SDK.

---

## 3. `max_tokens` → `max_completion_tokens`

The `max_tokens` parameter (deprecated in v1) has been **removed**. Passing it raises
a `TypeError`.

**Before (v1):**
```python
client.chat.completions.create(model=model, messages=[...], max_tokens=512)
```

**After (v2):**
```python
await client.chat.completions.create(
    model=await client.models.resolve_chat(),
    messages=[...],
    max_completion_tokens=512,
)
```

---

## 4. `client.image.generate()` → `client.image.create()`

The primary image-generation method was renamed. There is no deprecation alias —
the old name raises `AttributeError`.

```
client.image.generate(...)   → client.image.create(...)
```

`client.image.get_available_styles()` was a redundant duplicate **removed** in v2 — use
`client.image.list_styles()`, which already existed in v1.3.x (it was not renamed).

`client.image.simple_generate(...)` keeps its name (no verb rename), but it is **not
unchanged**: like every v2 method it is now async (`await` it), and its return type was
renamed `SimpleImageResponse` → `SimpleImageGenerationResponse`. It remains a thin
OpenAI-compat shim around `POST /images/generations`. For full Venice generation
features (LoRA, CFG, multi-variant) use `client.image.create(...)`; for editing use
`client.image.edit(...)` / `client.image.multi_edit(...)`.

---

## 5. Responses are now typed Pydantic models

In v1 several endpoints returned plain `TypedDict`s (read with subscripts,
`resp["data"]`). In v2 those responses are typed Pydantic models, read by attribute
(`resp.data`); **subscripting now raises `TypeError`.** (Binary endpoints — image
`create`/`edit`/`upscale`, streaming TTS — still return `bytes`/async iterators.)

### Dict → attribute access

This affects **embeddings, models, billing, and `audio.get_voices`**:

**Before (v1):**
```python
result = client.embeddings.create(model=model, input=["hello"])
vector = result["data"][0]["embedding"]
total = result["usage"]["total_tokens"]
```

**After (v2):**
```python
result = await client.embeddings.create(
    model=await client.models.resolve_embedding(), input=["hello"]
)
vector = result.data[0].embedding
total = result.usage.total_tokens
```

The same `["key"]` → `.key` change applies to `client.models.list()`,
`client.billing.get_usage_history()`, and `client.audio.get_voices()`. Envelope field names are
preserved, though some nested schemas changed — notably the models `model_spec` pricing
(`vcu` → `diem`). (`chat` and `characters` responses were already Pydantic models in v1,
so you don't switch to attribute access there — but `characters` field names became
camelCase in v2, e.g. `created_at` → `createdAt`, so update those reads.)

### `audio.create_speech()` returns `AudioResponse`, not `bytes`

**Before (v1):**
```python
audio_bytes = client.audio.create_speech(model=tts_model, input="Hello", voice=voice)
open("out.mp3", "wb").write(audio_bytes)
```

**After (v2):** the result is an `AudioResponse` — use `.save()`, or read the raw
bytes from `.content`:
```python
audio = await client.audio.create_speech(
    model=await client.models.resolve_tts(), input="Hello", voice=voice
)
audio.save("out.mp3")                          # convenience helper
# or: open("out.mp3", "wb").write(audio.content)
```

### `api_keys` return types

`client.api_keys.retrieve()` and `client.api_keys.delete()` previously returned raw
`dict`s; they now return typed models (`ApiKey` and `DeleteApiKeyResponse`). Read
their fields by attribute:

**Before (v1):**
```python
details = client.api_keys.retrieve(api_key_id=key_id)
print(details["description"])
```

**After (v2):**
```python
api_key = await client.api_keys.retrieve(api_key_id=key_id)
print(api_key.description)
```

`create()`, `get_rate_limits()`, and the web3 helpers also return typed models in v2.

### `billing.get_usage()` → `billing.get_usage_history()`

The `/billing/usage` endpoint was deprecated upstream in favour of
`/billing/usage-history`, which uses cursor (keyset) pagination. The v1
`get_usage()` method is replaced by `get_usage_history()`: `startDate`/`endDate`
become `startTimestamp`/`endTimestamp`, `limit` becomes `pageSize`, and
`page`/`sortOrder` are gone (the walk is always ascending by timestamp).

**Before (v1):**
```python
usage = client.billing.get_usage(
    startDate="2026-04-01T00:00:00Z", endDate="2026-05-01T00:00:00Z", page=1, limit=200
)
for entry in usage.data:
    print(entry.timestamp, entry.amount)
total_pages = usage.pagination.totalPages
```

**After (v2):** the response is `BillingUsageHistoryResponse` (`.data` +
`.nextCursor`). A continuation request sends **only** the cursor. The
`iter_usage_history()` helper threads it for you:
```python
async for entry in client.billing.iter_usage_history(
    startTimestamp="2026-04-01T00:00:00Z", endTimestamp="2026-05-01T00:00:00Z"
):
    print(entry.timestamp, entry.amount)
```

`format=BillingFormatEnum.CSV` returns a `BillingUsageHistoryCsvPage`, not bare
`bytes`: the CSV document is in `.content` (`.text` decodes it), the continuation
token Venice sends in the `x-next-cursor` header is `.nextCursor`, and the
server-assigned file name is `.filename`. Replace `data = await ...` /
`data.decode()` with `page = await ...` / `page.content` or `page.text`. To export
a whole window, `iter_usage_history_csv()` yields one CSV page at a time.

### `client.get_model_pricing()` removed

Pricing is no longer a dedicated client method — it lives on the model entry that
already carries it, so there is no extra round trip:

```python
# v1
pricing = client.get_model_pricing("some-model-id")

# v2
pricing = (await client.models.get("some-model-id")).model_spec.pricing
```

`model_spec.pricing` is an `LLMModelPricing` for chat and embedding models — the
shape `calculate_completion_cost` / `calculate_embedding_cost` expect. Image, video
and music specs carry their own pricing shapes.

To price a whole catalog at once, `CostTracker.from_client(client)` builds the
`{model_id: pricing}` map in a single `models.list()` call.

---

## 6. Type import paths moved

The per-resource type modules moved under a new `venice_ai.types.api` package:

```
from venice_ai.types.image  import ...   → from venice_ai.types.api.images import ...
from venice_ai.types.models import ...   → from venice_ai.types.api.models import ...
# likewise for api_keys, billing, characters, embeddings
```

Several response classes were also renamed (e.g. `ChatCompletion` →
`ChatCompletionResponse`, `ImageResponse` → `ImageGenerationResponse`). If you only
read responses by attribute you won't notice; if you imported the types directly,
update the import.

For building chat messages, prefer the typed helpers from `venice_ai.types.api`
(`UserMessage`, `SystemMessage`, `UserMessage.builder()`). Plain
`{"role": "user", "content": "..."}` dicts are still accepted.

---

## New in v2 (additive — no migration required)

None of the following change existing v1 call signatures; they're listed so v1
users know what's now available.

### New top-level resources

- **`client.video`** — text/image-to-video generation as async jobs (`submit()` / `run()`).
- **`client.music`** — music generation as async jobs (`submit()` / `run()`).
- **`client.crypto`** — JSON-RPC proxy (`rpc()`, `batch_rpc()`) with billing/idempotency headers surfaced.
- **`client.augment`** — `search()`, `scrape()`, `parse_text()` over Venice's `/augment/*` endpoints.
- **`client.x402`** — wallet billing (`balance()`, `transactions()`, `top_up()`) via SIWE / EIP-4361 auth. See [Advanced Features § x402 Wallet Authentication](./advanced.md#x402-wallet-authentication).
- **`client.responses`** — alpha OpenAI-compatible Responses API.
- **`client.tee`** — Trusted Execution Environment attestation and confidential compute.

`audio` also gained `transcribe()` (speech-to-text) and `create_voice()`.

### Model resolution

`client.models` gained `resolve_chat()`, `resolve_image()`, `resolve_embedding()`,
`resolve_tts()`, `resolve_video()`, `resolve_music()`, and friends — always prefer
these over hardcoding a model id, which goes stale on deprecation.

### CLI, rate limiting, and observability

v2 ships a `venice-py` command-line tool (`pip install` exposes the `venice-py` entry
point), pluggable rate limiting (`SIMPLE` / `ADAPTIVE` modes; the `[adaptive]` extra),
configurable backends (in-memory by default, Redis via the `[redis]` extra), cost
tracking, and structured logging.

### New optional extras

```bash
pip install 'venice-py[redis]'        # Redis-backed rate limiting / caching
pip install 'venice-py[adaptive]'     # ADAPTIVE rate limiter
pip install 'venice-py[x402]'         # SIWE wallet auth (eth-account + siwe)
pip install 'venice-py[x402-solana]'  # Solana wallet settlement
pip install 'venice-py[e2ee]'         # TEE client-side encryption (cryptography)
```

### Client-side E2EE (`enable_e2ee` / `e2ee=True`)

New in v2: real client-side end-to-end encryption for confidential-compute models.
Setting `enable_e2ee=True` (or `e2ee=True` on `client.chat.completions.create`) makes
the SDK verify the model's attestation, encrypt each user/system message to the
attested enclave key, stream the response, and decrypt it locally.

**What to do:**

- Install the extra: `pip install 'venice-py[e2ee]'` (pulls `cryptography`).
  Baseline attestation works without it; only encryption needs it.
- Use an `e2ee-*` confidential-compute model. Discover one dynamically — do not
  hardcode a model id:

  ```python
  models = await client.models.list(type="text")
  e2ee_model = next(
      entry.id
      for entry in models.data
      if getattr(entry.model_spec.capabilities, "supportsE2EE", False)
  )

  resp = await client.chat.completions.create(
      model=e2ee_model,
      messages=[{"role": "user", "content": "Confidential prompt."}],
      e2ee=True,  # or venice_parameters={"enable_e2ee": True}
  )
  ```

- Note the constraints: tool calling, web search/scraping, and multimodal
  (image/file) content are rejected with `InvalidRequestError` under E2EE.
- **Security limitation:** attestation verification is **baseline** — it trusts
  Venice's server-side `verified` claim and the nonce / report-data binding, but
  does **not** perform full client-side Intel TDX + NVIDIA quote verification. A
  one-time `UserWarning` is emitted on engagement. Supply a `FullQuoteVerifier`
  via `e2ee=TeeOptions(verifier=...)` if your threat model requires it. See
  [CHANGELOG § TEE client-side end-to-end encryption](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md).

### New kwargs on existing methods

| Method | New kwarg(s) | Notes |
|---|---|---|
| `client.image.create` | `enable_web_search` | Optional; supported models pull recent web context. |
| `client.chat.completions.create` | `store`, `text`, `include`, `metadata`, `prompt_cache_retention` | OpenAI-compat passthroughs + Venice's cache-retention tier. |

---

## Upgrading from 2.5 to 2.6

Most of 2.6 is additive. These changes can need a code change:

- **`Music.cancel()` and `MusicJob.cancel()` are removed.** They were aliases of
  `release()`, which deletes a finished job's stored media and cancels nothing.
  Call `client.music.release(model=..., queue_id=...)` or `await job.release()`.
- **`VeniceClientFactory.create_client()` no longer takes `account_key`.** It was
  never read. Pass the key as `api_key=` or `config.api_key`.
- **`RateLimitDiscovery(account_key=...)` is removed.** It was never read either;
  drop the argument.
- **Paid requests are no longer retried by default.** Image, video, music, speech
  and voice generation can be billed before a 5xx, timeout or dropped connection
  reaches you, so the default policy resends them only when they cannot have been
  processed. Free and idempotent requests are still retried, and chat, responses
  and embeddings are retried on errors Venice does not bill. The default is now
  2 attempts (was 3). `RetryOptions` is frozen (derive variants with
  `dataclasses.replace()`) and rejects 429, which the rate limiter retries.
- **Timeouts raise `APITimeoutError` on every path**, including stalled response
  bodies and calls routed through a rate limiter. It does not subclass
  `TimeoutError`, so catch `APITimeoutError` where you caught `TimeoutError` or
  `asyncio.TimeoutError`.
- **A job block that raises keeps its output.** `MusicJob`, `VideoJob` and
  `VoiceChangerJob` release stored media only when the `async with` block exits
  cleanly. A `VideoJob` or `MusicJob` left before it finishes is not released
  either (a release then deletes nothing); call `wait()` and release it after.
- **`VideoJob.cancel()` deletes the job's `download_url`** when the model returned
  one, so the private link stops working once you have saved the video.
- **Base URLs include the version path everywhere**: `https://api.venice.ai/api/v1`
  for `VeniceClient(base_url=...)`, `VeniceAIConfig.api_base_url` and
  `VENICE_API_BASE_URL`. A bare host still works as shorthand.
- **`VeniceAIConfig.api_key` is read.** The key resolves as explicit `api_key=`,
  then `config.api_key`, then `VENICE_API_KEY`.
- **CSV usage history returns a `BillingUsageHistoryCsvPage`**, not `bytes` (see
  the billing section above).
- **Resolvers raise `NoMatchingModelError` or `ModelQuotesUnavailableError`.** Both
  subclass `ValueError`, so existing `except ValueError` handlers still work.

The [CHANGELOG](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md) lists
every change.

---

## Further Reading

- [CHANGELOG](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md) — complete list of all changes in v2.0.0
- [README](https://github.com/sethbang/venice-py/blob/main/README.md) — updated usage examples for v2
