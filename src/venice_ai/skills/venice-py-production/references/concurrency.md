# Bounded concurrency with `client.gather`

Sourced from `VeniceClient.gather` in `src/venice_ai/_client.py`. The single-most-impactful production pattern in the SDK: replace unbounded `asyncio.gather(*coros)` with `client.gather(coros, max_concurrency=N)`.

## Why bounded > unbounded

`asyncio.gather(*coros)` fans out every coroutine immediately. With 1,000 chat completions, you get 1,000 in-flight HTTP requests, you blow your tier's rate limit on the first batch, and Venice 429s most of them. Bounded concurrency caps in-flight work, smooths request rate, and avoids retry storms.

## The signature

```python
async def gather[T](
    self,
    awaitables: Iterable[Awaitable[T]],
    *,
    max_concurrency: int = 10,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    """Bounded-concurrency variant of asyncio.gather, scoped to this client."""
```

Note: `awaitables` is a **list** (or any iterable), NOT unpacked args. `client.gather(*awaitables, ...)` is wrong — it'd interpret each coroutine as a positional kwarg.

## The basic pattern

```python
import asyncio
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage


async def summarize(client, ticket: dict) -> dict:
    response = await client.chat.completions.create(
        model=await client.models.resolve_chat(),
        messages=[UserMessage(content=f"Summarize in ≤30 words: {ticket['body']}")],
        max_completion_tokens=120,
    )
    return {"id": ticket["id"], "summary": response.text}


async def main(tickets: list[dict]) -> list[dict]:
    async with VeniceClient() as client:
        results = await client.gather(
            [summarize(client, t) for t in tickets],
            max_concurrency=5,
            return_exceptions=True,
        )
    return results
```

`max_concurrency=5` means at most 5 of `summarize(...)` are in flight at any moment. The other tickets queue waiting for a slot.

## `return_exceptions=True` — process partial successes

Default `return_exceptions=True` collects per-task failures into their result slots instead of aborting the batch; pass `return_exceptions=False` for all-or-nothing fail-fast semantics. For batch jobs where you want to process whatever succeeded:

```python
results = await client.gather(coros, max_concurrency=5, return_exceptions=True)

successes, failures = [], []
for r in results:
    if isinstance(r, Exception):
        failures.append(r)
    else:
        successes.append(r)

print(f"Succeeded: {len(successes)} / Failed: {len(failures)}")
for f in failures:
    log.warning("batch.item_failed", exc=str(f), exc_type=type(f).__name__)
```

For interactive workflows (e.g., a CLI tool waiting on a single response), pass `return_exceptions=False` so you fail fast.

## When NOT to use `client.gather`

- **Long-running jobs** (`async with await client.video.run(...)`): server-side queue parallelism is what matters; client-side parallelism on `wait()` polling adds nothing. Spawn separate jobs and let the server queue them.
- **Ordered streams**: a single user's multi-turn chat is sequential by definition. Parallelizing turns breaks ordering.
- **Tool dispatch inside `run_with_tools`**: the SDK already runs tool calls in parallel where the model emits parallel tool calls. Wrapping the whole `run_with_tools` in your own gather doesn't change that.
- **One-shot calls**: just `await` directly. Reaching for `gather` with one item is unnecessary ceremony.

## Picking `max_concurrency`

Rough rules of thumb:

| Workload | Suggested cap |
|---|---|
| Chat completions, generic tier | 5-10 |
| Chat completions, enterprise tier | 20-50 (verify with `response.response_rate_limits`) |
| Embeddings | 10-20 (cheaper per call, higher rate limit) |
| Image generation | 2-3 (each generation is heavier; sequential or low concurrency is often fine) |
| Video / music jobs | 3-5 (server-side queue depth; not client throughput) |

Watch `response.response_rate_limits.remaining_requests` after the first few calls. If it's dropping fast, lower the cap. See `rate-limiting.md` for adaptive throttling.

## Combining with retries

Each in-flight task may need retries on `RateLimitError` / `APITimeoutError`. Wrap each individual coroutine in your retry helper, then gather:

```python
async def summarize_with_retry(client, ticket):
    return await call_with_retries(lambda: summarize(client, ticket))

results = await client.gather(
    [summarize_with_retry(client, t) for t in tickets],
    max_concurrency=5,
    return_exceptions=True,
)
```

See `retries.md` for the retry helper.

**Don't put `client.gather` INSIDE the retry wrapper** — that retries the whole batch on any failure, which is rarely what you want.

## Combining with budget gating

If you have a `BudgetManager`, **gate the SUBMISSION**, not the call itself. Once a coroutine is in flight, you've committed to the cost; the gate only matters for tasks not yet started.

```python
async def maybe_summarize(client, budget, ticket):
    if not await budget.can_afford(Decimal("0.05")):
        return {"id": ticket["id"], "skipped": True}
    return await summarize(client, ticket)

results = await client.gather([maybe_summarize(client, budget, t) for t in tickets], max_concurrency=5)
```

This is approximate — `can_afford` reads the tracker mid-flight while other in-flight tasks may complete and update the total. For tighter accounting, run the batch in chunks of N and check the budget between chunks.

## Cancellation

`client.gather` propagates `asyncio.CancelledError` to all in-flight coroutines:

```python
import asyncio

task = asyncio.create_task(client.gather(coros, max_concurrency=5))
await asyncio.sleep(2.0)
task.cancel()                                  # all in-flight coroutines get CancelledError
try:
    await task
except asyncio.CancelledError:
    pass
```

The internal semaphore releases slots as cancelled coroutines exit, so cancellation is clean. (For long-running individual calls — e.g., 60-second chat completions — cancellation may not interrupt the HTTP request itself; aiohttp will close the connection on next await.)

## Common bugs

- **`client.gather(*coros, max_concurrency=N)`** — wrong; use `client.gather(coros, max_concurrency=N)`. The first form unpacks coroutines into positional args, which raises.
- **Unbounded `asyncio.gather(*coros)` for many calls** — blows rate limits.
- **`max_concurrency=1`** — defeats the purpose; just use a `for` loop.
- **`return_exceptions=False` for long batches** — first failure aborts the rest.
- **Spawning N coroutines with N HTTP clients** — re-uses connections poorly. One `VeniceClient` instance for the whole batch.
- **Treating `gather`'s wall-clock as N parallel × per-call latency** — bottleneck is the cap × per-call latency. With cap=5 and 100ms/call, 100 items take ~2 seconds, not 100ms.

## Sync clients

`SyncVeniceClient.gather` is not useful: it takes awaitables, and blocking calls never produce them. If you're sync and need parallel calls, switch to `VeniceClient` (async). Mixing sync and async in the same process via `asyncio.run` per call defeats the point.

## Related references

- `retries.md` — retry semantics per coroutine; combine with `client.gather`.
- `rate-limiting.md` — proactive throttling using `response.response_rate_limits`.
- `cost-tracking.md` — budget gating in batch jobs.
- `error-taxonomy.md` — handling per-task failures inside `return_exceptions=True`.
