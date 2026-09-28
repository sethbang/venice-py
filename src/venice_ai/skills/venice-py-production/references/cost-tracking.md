# Cost tracking with `CostTracker` and `BudgetManager`

Sourced from `src/venice_ai/costs.py`. Both are async-safe (use `asyncio.Lock` internally) — **`await` every method** that mutates or reads aggregated state.

## Quick reference

| Member | Type | Description |
|---|---|---|
| **`CostTracker`** | class | Stateful accumulator for per-request API costs. |
| `CostTracker(pricing_map=None)` | sync ctor | Empty tracker. With no pricing map, all `track()` calls record zero cost. |
| `CostTracker.from_client(client)` | async classmethod | Build a tracker with the live chat-pricing map (queries `client.models.list("chat")`). |
| `tracker.track(response, *, model=None, metadata=None)` | async | Record one response. Returns the call's USD cost. |
| `tracker.summary()` | async | Returns `CostSummary{total_requests, total_cost_usd, total_tokens, average_cost_usd, average_tokens}`. |
| `tracker.by_model()` | async | Returns `dict[model_id, Decimal_cost]`. |
| `tracker.reset()` | async | Clears all tracked state. |
| `tracker.requests` | `list[CostRecord]` | Raw list of records (read directly; not async). |
| `tracker.total_cost_usd` | `Decimal` | Accumulated USD. (NOT `tracker.total`.) |
| `tracker.total_tokens` | `int` | Accumulated token count. (NOT `tracker.calls`.) |
| `tracker.pricing_map` | `dict[str, ModelPricing]` | The pricing map; mutate to add models. |
| | | |
| **`BudgetManager`** | class | Daily / monthly USD-cap enforcement layered on a `CostTracker`. |
| `BudgetManager(*, tracker, daily_usd=None, monthly_usd=None)` | sync ctor | At least one of `daily_usd`/`monthly_usd` must be set. |
| `budget.can_afford(estimated_cost_usd)` | async | `True` if adding the estimate keeps both caps satisfied. |
| `budget.remaining()` | async | Returns `BudgetRemaining` (headroom + percentages). |
| `budget.daily_usd` / `budget.monthly_usd` | `Decimal \| None` | The configured caps. |

There is no `tracker.total` (use `total_cost_usd`), no `tracker.calls` (use `total_tokens` for tokens or `len(tracker.requests)` for call count), no `BudgetManager(limit=...)` (use `daily_usd`/`monthly_usd`), and no `would_exceed(...)` (use `can_afford(...)` BEFORE spending).

## Recommended pattern: auto-wire on the client

The `VeniceClient` constructor accepts `cost_tracker=tracker` — when set, the SDK calls `tracker.track(response)` automatically after every chat / embeddings response. You don't have to thread `track()` calls through your code.

```python
import asyncio
from decimal import Decimal
from venice_ai import VeniceClient
from venice_ai.costs import CostTracker, BudgetManager
from venice_ai.types.api import UserMessage


async def main() -> None:
    # 1. Bootstrap the tracker with live pricing (one extra API call up front).
    async with VeniceClient() as bootstrap:
        tracker = await CostTracker.from_client(bootstrap)
    budget = BudgetManager(tracker=tracker, daily_usd=Decimal("2.00"))

    # 2. Hand the tracker back into a fresh client; SDK auto-tracks.
    async with VeniceClient(cost_tracker=tracker) as client:
        for question in QUESTIONS:
            # Estimate before each call — can_afford checks the live tracker total.
            if not await budget.can_afford(Decimal("0.05")):
                print(f"Budget exhausted at ${tracker.total_cost_usd}; stopping.")
                break

            response = await client.chat.completions.create(
                model=await client.models.resolve_chat(),
                messages=[UserMessage(content=question)],
                max_completion_tokens=200,
            )
            print(response.text)
            # tracker.track(response) was already called inside the SDK — no manual call needed.

    # 3. Final report.
    summary = await tracker.summary()
    print(f"Total: ${summary.total_cost_usd} / {summary.total_tokens} tokens / "
          f"{summary.total_requests} requests / avg ${summary.average_cost_usd}/call")

    # Per-model breakdown
    for model, cost in (await tracker.by_model()).items():
        print(f"  {model}: ${cost}")
```

## Manual-track pattern

When you don't want auto-wire (e.g., the client is constructed elsewhere), call `track()` yourself:

```python
async with VeniceClient() as client:
    tracker = CostTracker()                           # sync ctor; no pricing map
    response = await client.chat.completions.create(...)
    cost = await tracker.track(response)              # async — must await
    print(f"This call cost ${cost}")
```

If you skip the pricing map, costs record as `0.00` — useful for token-only accounting where you don't need USD.

## One-shot cost calc — `calculate_completion_cost`

For a single response when you don't want a full `CostTracker`, call the lower-level helper directly:

```python
from venice_ai.costs import calculate_completion_cost
# signature: (completion: ChatCompletion, model_pricing: ModelPricing | None) -> dict[str, Decimal]
```

Two things agents reliably get wrong:

1. **The first arg is the completion object, NOT a model id string.** Pass the full `ChatCompletionResponse` you got back from `client.chat.completions.create(...)`. The function reads `completion.usage` off that object: `prompt_tokens` (split into cache reads, cache writes and the uncached remainder, each at its own rate) and `completion_tokens`.
2. **The return is a dict with a `"usd"` key**, not a `Decimal` directly. Do `costs["usd"]`, not `costs`.

```python
from decimal import Decimal
from venice_ai import VeniceClient
from venice_ai.costs import CostTracker, calculate_completion_cost
from venice_ai.types.api import UserMessage

async with VeniceClient() as client:
    # Pricing map is keyed by model id; bootstrap once and reuse.
    tracker = await CostTracker.from_client(client)
    pricing_map = tracker.pricing_map                      # dict[str, ModelPricing]

    model = await client.models.resolve_chat()
    response = await client.chat.completions.create(
        model=model,
        messages=[UserMessage(content="...")],
    )

    costs = calculate_completion_cost(response, pricing_map.get(model))
    print(f"Cost: ${costs['usd']}")                        # Decimal in dict
```

If `model_pricing` is `None` or the response has no `usage`, the helper returns `{"usd": Decimal("0.00")}` rather than raising — robust by design, but it means you can silently record zero-cost rows if you forget to populate the pricing map. Source: `calculate_completion_cost` in `src/venice_ai/costs.py`.

There's a parallel `calculate_embedding_cost(embedding_response, model_pricing)` for `client.embeddings.create` responses; same dict-with-`"usd"` shape.

## What `track()` accepts

`tracker.track(response)` works for these response types:

- `ChatCompletionResponse` — used by `client.chat.completions.create`, `parse`, `run_with_tools.response`, `stream.final_response`
- `EmbeddingsResponse` — used by `client.embeddings.create`

Anything else raises `TypeError`. If you want to track image/video/music spend, you have to do it manually via `response.balance_info` (see below) — `CostTracker` doesn't compute their costs.

## Without the helpers — header-driven balance tracking

Every response with a `_response` attribute exposes `response.balance_info`:

```python
from venice_ai import BalanceInfo

response = await client.chat.completions.create(...)
if response.balance_info:                              # may be None on free tier
    bi: BalanceInfo = response.balance_info
    print(f"Remaining USD prepaid balance: ${bi.usd}")
    print(f"Remaining diem (Venice's internal unit): {bi.diem}")
```

`balance_info.usd` is the **post-call** balance — i.e., the new remaining balance, not the cost of THIS call. Compute the cost as `previous_balance - current_balance` if you're tracking spend without the helpers.

## `BudgetManager.can_afford` semantics

```python
result = await budget.can_afford(estimated_cost_usd)
```

- Reads `await tracker.summary()` to get `total_cost_usd` so far.
- Computes `projected = total_cost_usd + estimated_cost_usd`.
- Returns `True` iff `projected <= daily_usd` (when set) AND `projected <= monthly_usd` (when set).

So you must **estimate** the next call's cost before invoking it. For chat completions, use `(await client.chat.completions.estimate_cost(model=..., messages=..., expected_completion_tokens=...)).total_cost_usd`: it adds an allowance for the system prompt Venice injects unless `venice_parameters` opts out. For most apps a flat `Decimal("0.05")` ceiling per call is good enough.

## `budget.remaining()`

```python
remaining = await budget.remaining()
# remaining.daily_remaining_usd: Decimal | None
# remaining.monthly_remaining_usd: Decimal | None
# remaining.daily_used_pct: float | None
# remaining.monthly_used_pct: float | None
```

Useful for surfacing "X% of your budget used today" in dashboards.

## Rollover

`BudgetManager` does **not** call `tracker.reset()` automatically. You manage rollover (daily / monthly) yourself, e.g., by zeroing the tracker at midnight UTC:

```python
import datetime as dt

async def daily_rollover(tracker: CostTracker) -> None:
    while True:
        now = dt.datetime.now(dt.UTC)
        next_midnight = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        await asyncio.sleep((next_midnight - now).total_seconds())
        await tracker.reset()
```

For multi-instance deployments, share the tracker via Redis (see `rate-limiting.md` for the Redis-backed primitive — the same approach extends to cost tracking).

## Common bugs

- **Calling `track()` without `await`** — silently no-ops; the coroutine is discarded and the lock is never acquired.
- **Treating `tracker.total_cost_usd` as `tracker.total`** — that attribute doesn't exist.
- **Constructing `BudgetManager(limit=Decimal("2.00"))`** — wrong kwarg. Use `daily_usd=Decimal("2.00")` or `monthly_usd=Decimal("60.00")`.
- **Calling `would_exceed(...)`** — that method doesn't exist. Use `await can_afford(estimated_cost)` BEFORE spending.
- **`CostTracker()` with no pricing_map and expecting non-zero costs** — costs record as zero unless the map is populated. Use `await CostTracker.from_client(client)` for the live map.
- **Threading raw responses to `tracker.track()` from streaming runs** — pass `stream.final_response` (which IS a `ChatCompletionResponse`), not individual chunks. `final_response` is only populated when you use `collect_with_deltas()` or `collect()` — see `venice-py/references/streaming.md`.

## Related references

- `error-taxonomy.md` — `PaymentRequiredError` (when the prepaid balance is exhausted).
- `retries.md` — interaction between cost-aware budget gating and retry loops.
- `venice-py-x402/SKILL.md` — top-up flow when the prepaid balance runs out.
- `venice-py/references/headers-and-metadata.md` — full surface of `_response`-derived properties including `balance_info`.
