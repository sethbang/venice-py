# Retry strategy for Venice API calls

The retry decision tree for the typed exceptions in `venice_ai.exceptions`. Pair with `error-taxonomy.md` for the full class table.

## The 30-second mental model

1. **Catch by class, not by status code.** The SDK's typed hierarchy already encodes "what kind of failure is this" — `RateLimitError` ≠ `APIConnectionError` even though both can be 429-ish.
2. **Read the structured retry hint first.** `RateLimitError.retry_after_seconds` is authoritative when present; use it before falling back to your computed backoff.
3. **Cap everything.** Cap individual sleeps (≤30s), cap total retries (5 typical), cap idle time before declaring permanent failure.
4. **Add jitter.** Pure exponential backoff causes thundering herds. Full jitter or decorrelated jitter both work; full jitter is simpler.
5. **Surface anything you can't fix.** `AuthenticationError`, `PaymentRequiredError`, `InvalidRequestError`, `MaxIterationsExceededError` are operator concerns. Don't loop on them.

## Reference implementation

```python
import asyncio
import logging
import random
from venice_ai.exceptions import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    ServiceUnavailableError,
    AuthenticationError,
    PermissionDeniedError,
    PaymentRequiredError,
    InvalidRequestError,
    NotFoundError,
    UnprocessableEntityError,
    MaxIterationsExceededError,
)

log = logging.getLogger("venice.retry")


async def call_with_retries(coro_factory, *, max_retries: int = 5, base_backoff: float = 1.0) -> object:
    """Wrap an async callable with the standard Venice retry policy.

    Args:
        coro_factory: a zero-arg callable that returns a fresh coroutine on each call.
                      Don't pass an awaitable directly — coroutines can't be awaited twice.
        max_retries: maximum retry attempts after the initial call.
        base_backoff: starting backoff in seconds (doubles on each retry, capped at 30s).
    """
    backoff = base_backoff
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()

        # ── Retryable ──
        except RateLimitError as e:
            wait = e.retry_after_seconds if e.retry_after_seconds is not None else backoff
            wait = min(wait, 30.0)
            wait += random.uniform(0, wait)               # full jitter
            log.warning("retry.rate_limit", attempt=attempt, wait=wait)
        except (APITimeoutError, APIConnectionError):
            wait = min(backoff, 30.0) + random.uniform(0, min(backoff, 30.0))
            log.warning("retry.transient", attempt=attempt, wait=wait)
        except ServiceUnavailableError:
            wait = min(backoff, 30.0) + random.uniform(0, min(backoff, 30.0))
            log.warning("retry.503", attempt=attempt, wait=wait)
        except InternalServerError:
            if attempt >= 2:                              # 5xx: only retry 2x
                raise
            wait = min(backoff, 30.0) + random.uniform(0, min(backoff, 30.0))
            log.warning("retry.500", attempt=attempt, wait=wait)

        # ── Terminal ──
        except (AuthenticationError, PermissionDeniedError, PaymentRequiredError,
                InvalidRequestError, NotFoundError, UnprocessableEntityError,
                MaxIterationsExceededError):
            raise

        if attempt == max_retries:
            log.error("retry.exhausted", attempt=attempt)
            raise RuntimeError(f"Exhausted {max_retries} retries")

        await asyncio.sleep(wait)
        backoff = min(backoff * 2, 30.0)

    raise RuntimeError("unreachable")
```

Usage:

```python
response = await call_with_retries(
    lambda: client.chat.completions.create(
        model=await client.models.resolve_chat(),
        messages=[UserMessage(content="...")],
    ),
    max_retries=5,
)
```

## Decision tree

```
  exception
   │
   ├── AuthenticationError / PermissionDeniedError → SURFACE (auth bug)
   ├── PaymentRequiredError → SURFACE (top up — see venice-py-x402)
   ├── InvalidRequestError / NotFoundError / UnprocessableEntityError → SURFACE (fix request)
   ├── MaxIterationsExceededError → SURFACE (logic bug in agent loop)
   │
   ├── RateLimitError → wait min(retry_after_seconds, 30s) + jitter, retry up to 5x
   ├── APITimeoutError / APIConnectionError → exp backoff + jitter, retry up to 5x
   ├── ServiceUnavailableError (503) → exp backoff + jitter, retry up to 5x
   ├── InternalServerError (5xx, generic) → exp backoff + jitter, retry up to 2x
   ├── ConflictError (409) → DEPENDS — idempotent operation: retry once; mutating: surface
   ├── VideoGenerationError / MusicGenerationError → DEPENDS — check e.error_code
   │
   └── (anything else, including bare Exception) → SURFACE — don't paper over unknown failures
```

## Jitter strategies

**Full jitter** (default in the reference impl above) — simplest, evenly distributes retries:

```python
sleep = random.uniform(0, computed_backoff)
```

**Decorrelated jitter** — better at avoiding herds in extreme contention; tracks the previous sleep:

```python
prev_sleep = max(prev_sleep, base)
sleep = min(cap, random.uniform(base, prev_sleep * 3))
prev_sleep = sleep
```

**No jitter** (pure exponential) — only acceptable for single-instance scripts where there's no thundering-herd concern.

For most production workloads, **full jitter is enough**. Don't optimize for decorrelated jitter unless benchmarks show it matters.

## Idempotency considerations

Retrying a `client.chat.completions.create(...)` call when the original may have succeeded server-side BUT the client never saw the response: harmless (you pay twice but the conversation is the same). Retrying:

- **Tool-calling** (`run_with_tools`): mostly fine — the loop is stateful but the tool functions are usually idempotent.
- **`client.video.run / submit`**: NOT idempotent — each retry submits a new, separately billed job, and a job that already started keeps running server-side (`cancel()` / the `async with` exit only release its stored media). Persist the `queue_id` and resume polling with `retrieve()` rather than re-submitting.
- **`client.x402.top_up(payment_header=...)`**: NOT idempotent in general — same payment header may be rejected on retry as already-spent. The 402 → sign → submit flow has its own state.

For non-idempotent ops, use a **request-id-based deduplication key** if the endpoint supports one, or accept that retries may double-charge and limit retry counts to 1.

## `client.with_retries(RetryOptions(...))` — scoped retries

The SDK has a built-in `with_retries` async context manager that scopes a `RetryOptions` policy to a block of code:

```python
from venice_ai.middleware.retry import RetryOptions

async with client.with_retries(RetryOptions(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
)):
    response = await client.chat.completions.create(...)
```

The policy is per-call inside the block. Useful when you want a different retry posture for one specific operation without re-configuring the client.

This is at the `aiohttp` middleware layer — orthogonal to (and stackable with) your own application-level retry wrapper above. You typically don't need both; pick one.

## When NOT to wrap with retries

- **Streaming**: a partial-streamed response that fails mid-iteration can't be resumed. Retry the whole stream from scratch outside the `async with stream:` block, not inside.
- **Long-running jobs** (`async with client.video.run(...) as job:`): the SDK already handles transient errors during `wait()` polling. Don't wrap the whole `async with` in a retry.
- **Tool calls inside `run_with_tools`**: the agent loop will already retry the LLM call on transient failures (via the SDK's middleware). Adding application-level retry around `run_with_tools` mostly multiplies retry count without changing behavior.

## Common bugs

- **Bare `time.sleep(2)` in async code** — blocks the event loop for 2 seconds. Always `await asyncio.sleep(...)`.
- **Catching `Exception` and retrying everything** — drops the structured info on terminal errors and burns retry budget on bugs that won't fix.
- **Ignoring `retry_after_seconds`** — the server told you exactly how long to wait. Listen.
- **Unbounded retry loop** — write `max_retries`, write a wall-clock cap, write a budget cap. All three. (Yes, all three.)
- **Retrying `MaxIterationsExceededError`** — the model is stuck. More iterations won't help.
- **Retrying `PaymentRequiredError`** — top up the balance, then retry the original operation.

## Related references

- `error-taxonomy.md` — full exception class table with attribute access patterns.
- `rate-limiting.md` — proactive throttling using `response.response_rate_limits` (avoid the 429 in the first place).
- `concurrency.md` — interaction between bounded concurrency and retry storms.
- `cost-tracking.md` — interaction between retries and budget gating (each retry costs).
