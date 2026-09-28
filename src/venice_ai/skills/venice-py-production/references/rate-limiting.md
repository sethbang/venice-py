# Rate-limit handling

Two complementary patterns:

1. **Proactive throttling** — read `response.response_rate_limits` after each call; back off BEFORE you hit the wall.
2. **Reactive retry** — catch `RateLimitError` and honor `retry_after_seconds`.

Use both together. Proactive throttling avoids most 429s; reactive retry handles the inevitable few.

## Proactive throttling — `response.response_rate_limits`

Every successful response carries `response.response_rate_limits` (a `RateLimitInfo` instance). The fields are all optional — the server emits whatever applies to your tier:

```python
class RateLimitInfo:
    limit_requests:     int | None        # max requests per window
    remaining_requests: int | None        # how many you have left RIGHT NOW
    reset_requests:     datetime | None    # when the request window resets (datetime)
    limit_tokens:       int | None        # max tokens per window (token-budgeted tiers)
    remaining_tokens:   int | None
    reset_tokens:       float | None       # absolute Unix seconds when token limit resets
    type:               str | None         # "user", "api_key", or "global"
```

Pattern:

```python
from datetime import datetime, timezone

def _seconds_until(reset: datetime | None) -> float:
    # reset_requests is a datetime; derive seconds until the window resets.
    if reset is None:
        return 1.0
    return max((reset - datetime.now(timezone.utc)).total_seconds(), 1.0)

async def call_with_throttle(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)

    rl = response.response_rate_limits
    if rl is None:
        return response

    # Throttle if we're getting close to the wall
    if rl.remaining_requests is not None and rl.remaining_requests < 5:
        wait = _seconds_until(rl.reset_requests)
        log.info("rate_limit.throttle", remaining=rl.remaining_requests, wait=wait)
        await asyncio.sleep(min(wait, 5.0))

    # Same idea for tokens (less common). reset_tokens is absolute Unix seconds.
    if rl.remaining_tokens is not None and rl.remaining_tokens < 1000:
        log.warning("rate_limit.tokens_low", remaining=rl.remaining_tokens)
        token_wait = (
            max(rl.reset_tokens - datetime.now(timezone.utc).timestamp(), 1.0)
            if rl.reset_tokens
            else 1.0
        )
        await asyncio.sleep(min(token_wait, 5.0))

    return response
```

This is "soft" throttling — the call already succeeded; we slow down for the NEXT call.

## Reactive retry — `RateLimitError`

When you DO hit a 429, the SDK raises `RateLimitError` with `e.retry_after_seconds`:

```python
from venice_ai.exceptions import RateLimitError

try:
    response = await client.chat.completions.create(...)
except RateLimitError as e:
    wait = e.retry_after_seconds or 1.0
    await asyncio.sleep(min(wait, 30.0))
    response = await client.chat.completions.create(...)
```

`e.retry_after_seconds` is parsed from the `Retry-After` header — authoritative when present. Fall back to your own backoff if it's `None` (some Venice routes don't emit `Retry-After`).

For a complete retry helper covering rate-limit + timeout + connection errors, see `retries.md`.

## Distributed: Redis-backed rate limiter

For multi-instance deployments (multiple workers / pods talking to the same Venice account), in-memory throttling doesn't work — each worker has its own view of `remaining_requests`. The SDK coordinates limits across workers via Redis using the **adaptive** rate limiter, wired through the production config preset (requires `pip install 'venice-py[adaptive]'`):

```python
import os
from venice_ai.factory import VeniceClientFactory
from venice_ai.presets import create_production_config

# create_production_config wires BOTH halves required for Redis at runtime:
#   BackendType.REDIS  +  RateLimiterMode.ADAPTIVE (redis_url=...)
# Setting the Redis backend WITHOUT adaptive mode never contacts Redis:
# VeniceClientFactory.create_client() warns and the config validator errors.
config = create_production_config(
    redis_url="redis://...",
    max_concurrent_executions=10,
)

client = VeniceClientFactory.create_client(
    config=config,
    api_key=os.environ["VENICE_API_KEY"],
    account_id="my-fleet",   # scopes the shared Redis keys for ADAPTIVE mode
)

async with client:
    response = await client.chat.completions.create(...)
```

The adaptive limiter coordinates per-account limits across all workers via the shared Redis state, learning the server-side limit empirically and proactively pacing requests. `account_id` matters — it scopes the Redis keys so multiple accounts don't collide.

This is the right architecture for production fleets. For development / single-instance jobs, the default in-memory limiter is fine. See `examples/advanced/redis_backend.py` for an end-to-end wiring that verifies Redis is actually contacted on the wire.

## When to combine throttle + retry

Big batch jobs (1k+ calls) benefit from BOTH:

```python
async def call_with_throttle_and_retry(client, **kwargs):
    backoff = 1.0
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(**kwargs)
            # Proactive: back off NOW for the next call if we're close
            rl = response.response_rate_limits
            if rl and rl.remaining_requests is not None and rl.remaining_requests < 5:
                # reset_requests is a datetime; derive seconds-until-reset.
                wait_s = (
                    max((rl.reset_requests - datetime.now(timezone.utc)).total_seconds(), 1.0)
                    if rl.reset_requests
                    else 1.0
                )
                await asyncio.sleep(min(wait_s, 5.0))
            return response
        except RateLimitError as e:
            wait = e.retry_after_seconds or backoff
            await asyncio.sleep(min(wait, 30.0))
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError("rate-limited beyond retry budget")
```

For interactive workloads (single user request), reactive retry alone is usually enough — proactive throttling adds latency you probably don't want.

## Per-route rate limits

Different Venice routes have different limits. Image generation has lower per-second caps than chat; video jobs have a tiny in-flight cap (2-3 concurrent server-side). When mixing modalities, check per-route limits separately:

```python
chat_response  = await client.chat.completions.create(...)
image_response = await client.image.create(...)

print(f"Chat remaining:  {chat_response.response_rate_limits.remaining_requests if chat_response.response_rate_limits else 'n/a'}")
print(f"Image remaining: {image_response.response_rate_limits.remaining_requests if image_response.response_rate_limits else 'n/a'}")
```

A tight per-resource cap can mean: gather across modalities is safe, gather within one modality needs a tighter `max_concurrency`.

## Common bugs

- **Treating `e.retry_after_seconds` as guaranteed-non-None** — null-check first.
- **Ignoring `retry_after_seconds` and using your own backoff** — the server's value is more accurate than your guess.
- **Throttling on `response_rate_limits` even when it's None** — null-check the property before accessing fields.
- **Polling Venice every 100ms after a 429** without honoring `retry_after_seconds` — DoS-ing yourself.
- **In-memory throttling across multi-process workers** — each worker has its own state. Use Redis.
- **Setting `max_concurrency` higher than the per-second rate limit** — bounded concurrency limits in-flight count, not request rate. With 50 RPS limit and `max_concurrency=20`, fast calls (10ms each) can still issue 2,000 RPS in bursts.

## Related references

- `retries.md` — full retry helper combining `RateLimitError` with backoff strategies.
- `concurrency.md` — `client.gather(max_concurrency=N)` complements throttling.
- `error-taxonomy.md` — `RateLimitError` is a subclass of `APIError`.
- `venice-py/references/headers-and-metadata.md` — full `response_rate_limits` property reference.
