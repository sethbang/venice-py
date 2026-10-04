---
name: venice-py-production
description: Harden Venice AI Python SDK code for production. Use this skill whenever the user says "add retries", "I'm getting 429s from Venice", "how do I track Venice spend", "log Venice usage", "parallel calls without blowing my rate limit", "production-ready Venice client", "circuit breaker for Venice", "Venice cost tracking", "Venice budget", "Redis backend Venice", or is shipping a service that hits the Venice API under load. Covers the typed exception tree (`VeniceError`, `RateLimitError.retry_after_seconds`, `PaymentRequiredError`, `MaxIterationsExceededError`, `APITimeoutError`, `VideoGenerationError.error_code`), bounded concurrency via `client.gather(max_concurrency=N)`, header-driven cost tracking (`response.balance_info`, `response.response_rate_limits`, `response.deprecation_info`), `CostTracker` and `BudgetManager`, structured logging, prompt caching, and Redis-backed distributed rate limiting. Use this — not a generic backoff library — because Venice surfaces structured retry-after, balance, and deprecation metadata that hand-rolled retries silently drop. For the basic API surface (chat, tools, etc.), use the `venice-py` skill; for image/audio/video generation under load, also load `venice-py-multimodal`.
---

# Venice AI in production

> _Unofficial, community-maintained — not affiliated with or endorsed by Venice AI._

This skill is for "ship it" — what to add to working Venice code so it survives real load: retries, rate-limit handling, bounded concurrency, cost tracking, observability. The basic API surface lives in the `venice-py` skill; this one is the layer on top.

## The five hardening axes

1. **Retries** — catch the right exceptions, respect `retry_after_seconds`, cap backoff, never retry auth/payment failures.
2. **Rate-limit handling** — read `response.response_rate_limits`, throttle proactively when remaining is low.
3. **Bounded concurrency** — `client.gather(max_concurrency=N)` instead of unbounded `asyncio.gather`.
4. **Cost tracking** — header-driven via `response.balance_info`; aggregate with `CostTracker` / `BudgetManager`.
5. **Observability** — structured logs, request-id correlation, deprecation header logging.

## Retries

Catch by specific subclass; never bare `Exception`. The structured retry hint on `RateLimitError` is authoritative.

```python
import asyncio
from venice_ai.exceptions import (
    RateLimitError, APITimeoutError, APIConnectionError,
    AuthenticationError, PaymentRequiredError,
)

async def call_with_retries(client, **kwargs):
    backoff = 1.0
    for _ in range(5):
        try:
            return await client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            await asyncio.sleep(min(e.retry_after_seconds or backoff, 30.0))
        except (APITimeoutError, APIConnectionError):
            await asyncio.sleep(min(backoff, 30.0))
        except (AuthenticationError, PaymentRequiredError):
            raise                                # never retry — auth: fix key; 402: top up
        backoff = min(backoff * 2, 30.0)
    raise RuntimeError("exhausted retries")
```

**Never retry**: `AuthenticationError` (401), `PaymentRequiredError` (402, top up), `InvalidRequestError` (400, fix request), `MaxIterationsExceededError` (logic bug). **Always cap backoff** (≤30s) — unbounded exp is a self-DoS. Full decision tree + jitter strategies + scoped `client.with_retries(RetryOptions(...))`: `references/retries.md` and `references/error-taxonomy.md`.

## Rate-limit handling

Read `response.response_rate_limits` after each call to throttle proactively rather than reactively:

```python
from datetime import datetime, timezone

async def chat_with_throttle(client, ...):
    response = await client.chat.completions.create(...)
    rl = response.response_rate_limits
    if rl and rl.remaining_requests is not None and rl.remaining_requests < 5:
        # we're close to the wall — slow down.
        # reset_requests is a datetime; derive seconds-until-reset.
        delay = (
            max((rl.reset_requests - datetime.now(timezone.utc)).total_seconds(), 1.0)
            if rl.reset_requests
            else 1.0
        )
        await asyncio.sleep(delay)
    return response
```

For distributed services, switch to the Redis-backed rate-limiter so all instances share a coordinated quota. See `references/rate-limiting.md`.

## Bounded concurrency — `client.gather`

`asyncio.gather([...])` is unbounded — it'll fan out 100 concurrent requests if you give it 100 coroutines. **Use `client.gather` with `max_concurrency`**:

```python
results = await client.gather(
    [
        client.chat.completions.create(
            model=model,
            messages=[UserMessage(content=ticket.body)],
            max_completion_tokens=120,
        )
        for ticket in tickets
    ],
    max_concurrency=5,
    return_exceptions=True,                  # don't fail-fast on a single bad ticket
)
```

`return_exceptions=True` lets you process partial successes; check `isinstance(r, Exception)` per result.

**When NOT to use `gather`**:
- Long-running jobs (video, music) — they have their own queue; parallelism is the server's concern.
- Ordered streams (one user's multi-turn conversation) — you want sequential ordering, not concurrent.

## Cost tracking via headers

`CostTracker` + `BudgetManager` (both async — `await` every method; `CostTracker` owns the `asyncio.Lock` that serializes both) wrap per-response `balance_info` and per-call usage. Recommended pattern: auto-wire on the client so the SDK calls `track()` for you.

```python
from decimal import Decimal
from venice_ai import VeniceClient
from venice_ai.costs import CostTracker, BudgetManager

# 1. Build a tracker and attach it to a single open client (SDK ≥ 2.0).
tracker = CostTracker()                          # empty pricing map
budget = BudgetManager(tracker=tracker, daily_usd=Decimal("2.00"))

async with VeniceClient(cost_tracker=tracker) as client:
    await client.attach_cost_tracker(tracker)     # hydrates pricing from /models?type=chat
    for ticket in tickets:
        if not await budget.can_afford(Decimal("0.05")):    # check BEFORE spending
            break
        response = await client.chat.completions.create(...)  # track() called for you
    summary = await tracker.summary()                          # total_cost_usd, total_tokens, ...
```

**Why `attach_cost_tracker` rather than just `from_client`?**
`CostTracker.from_client(client)` requires an *open* client (it calls
`models.list()`), but the `cost_tracker=` constructor kwarg expects the
tracker upfront — a chicken-and-egg problem.
The 2.0 helper `client.attach_cost_tracker(tracker, *, populate_pricing=True)`
solves it: pass an empty tracker at construction time, then hydrate
pricing once the client is open. Pass `populate_pricing=False` if you've
already supplied a pricing map yourself or want to skip the round-trip.

**Field names that bite** (`venice-py lint` flags V502/V503):
- `tracker.total_cost_usd` (NOT `tracker.total`)
- `tracker.total_tokens` (NOT `tracker.calls`)
- `len(tracker.requests)` for call count
- `BudgetManager(tracker=, daily_usd=, monthly_usd=)` — at least one cap; use `can_afford(estimated_cost)` BEFORE spending. **No** `limit=` kwarg, **no** `would_exceed()` method.

Manual-track variant + rollover semantics + no-helpers `response.balance_info.usd` access — see `references/cost-tracking.md`.

## Deprecation warnings

Models get retired. Log `response.deprecation_info` so you find out from your logs, not your error budget:

```python
if response.deprecation_info and response.deprecation_info.is_deprecated:
    log.warning(
        "venice.deprecated_model",
        warning=response.deprecation_info.warning,
        sunset_date=response.deprecation_info.date,
    )
```

## Error taxonomy

| Exception | When to retry | What to do |
|---|---|---|
| `RateLimitError` (429) | Yes (with `.retry_after_seconds`) | Sleep, retry |
| `APITimeoutError` | Yes (with backoff) | Retry with cap |
| `APIConnectionError` | Yes (with backoff) | Retry with cap |
| `InternalServerError` (5xx) | Yes (limited) | Retry 1-2x |
| `ServiceUnavailableError` (503) | Yes (with backoff) | Retry with cap |
| `AuthenticationError` (401) | **No** | Surface; check API key |
| `PermissionDeniedError` (403) | **No** | Surface; check ACLs |
| `PaymentRequiredError` (402) | **No** | Top up balance; carries payment instructions |
| `InvalidRequestError` (400) | **No** | Fix the request |
| `NotFoundError` (404) | **No** | Surface |
| `ModelGoneError` (410) | **No** | Migrate to `deprecation.replacementModelId` / re-resolve |
| `UnprocessableEntityError` (422) | **No** | Fix the request body |
| `ConflictError` (409) | Maybe | Depends on resource semantics |
| `MaxIterationsExceededError` | **No** | Bug in agent loop or hostile tool |
| `VideoGenerationError` / `MusicGenerationError` | Maybe (re-submit) | Inspect `.error_code` |

See `references/error-taxonomy.md` for the canonical table.

The client already retries below this layer, and only where a resend cannot bill twice: idempotent calls on any transient failure, chat/responses/embeddings once on a 500, and paid generation (image, video, music, speech) only when the connection never opened or on a documented "model at capacity" 503. So an `APITimeoutError` from a paid call means the job may have run and been billed; check before re-submitting. `client.retry_options` shows the policy; see `references/retries.md`.

## Prompt caching

For long, mostly-static prompts (system prompt + retrieved docs + per-turn user query), Venice caches the prompt prefix automatically on supporting models, adding Claude's cache markers itself. `prompt_cache_key` raises the odds that a request reaches a server holding the prefix, `prompt_cache_retention` asks for a longer lifetime, and a `cache_control={"type": "ephemeral"}` marker is only needed for something Venice would not cache on its own (a long document in a single-turn Claude request). Read hits and writes from `response.usage.cached_tokens` / `cache_write_tokens`. `resolve_chat(require_prompt_caching=True)` only checks for a listed cache price, so measure hits on the model you pick.

See `references/prompt-caching.md`. Pattern from `examples/advanced/prompt_caching.py`.

## Observability

Wrap your client calls with structured logs that correlate by request ID:

```python
import structlog
log = structlog.get_logger("venice")

async def traced_create(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)
    log.info(
        "venice.chat",
        model=kwargs.get("model"),
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        balance_usd=response.balance_info.usd if response.balance_info else None,
        request_id=response.headers.get("x-request-id") if response.headers else None,
    )
    return response
```

Pair with `client.with_retries(RetryOptions(...))` for scoped retry overrides during specific operations.

## Pitfalls AI assistants reliably get wrong

1. **Naive `for x in items: await client.chat...`** — sequential, no concurrency. Use `client.gather(..., max_concurrency=N)`.
2. **Retrying `PaymentRequiredError`** — that 402 won't go away on its own. Top up the balance, then retry the operation.
3. **Ignoring `retry_after_seconds`** — naive `time.sleep(2)` retry storms exhaust the wall-clock budget on the server.
4. **Catching `Exception`** — drops the structured info on the typed subclass. Always specific.
5. **Unbounded `asyncio.gather([...])`** — fan-out without `max_concurrency` blows your rate limit on the first call.
6. **Logging the response without `request_id`** — when something goes wrong, you can't correlate with Venice's server logs.
7. **Retrying `MaxIterationsExceededError`** — it's a logic problem, not a transient one. Investigate the loop.
8. **Treating `gather(return_exceptions=False)` as success** — a single failure aborts the batch and discards partial results.

## References

- `references/retries.md` — full retry decision tree, jitter strategies
- `references/rate-limiting.md` — simple vs adaptive vs Redis backend
- `references/cost-tracking.md` — `CostTracker` / `BudgetManager` patterns and aggregation
- `references/observability.md` — metrics, logs, tracing hooks
- `references/concurrency.md` — `gather` patterns, ordering caveats
- `references/error-taxonomy.md` — canonical table; the `venice-py` skill links here
- `references/prompt-caching.md` — when caching pays off, how to opt in

## Examples to read

Paths below are relative to the SDK repo's `examples/` directory.

- `production/async_patterns.py` — concurrent requests, `client.gather`, cancellation
- `production/api_key_management.py` — secure key handling
- `production/logging_monitoring.py` — structured logging
- `production/cost_management.py` — `CostTracker` patterns
- `advanced/error_recovery.py` — `RetryOptions`, `client.with_retries()`
- `advanced/performance_optimization.py` — connection pooling, batching
- `advanced/redis_backend.py` — distributed rate limiting
- `advanced/prompt_caching.py` — caching hit rates
- `headers/header_access_example.py` — `_response`, `balance_info`, `response_rate_limits`
- `basic/error_handling.py` — exception hierarchy walkthrough
