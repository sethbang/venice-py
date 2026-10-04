# Response metadata via `_response`

Sourced from `VeniceBaseModel` in `src/venice_ai/core/models/base.py`. Every Venice response model auto-attaches the raw HTTP response on a private `_response` attribute and exposes typed metadata via properties on the response itself.

## The five surfaced properties

| Property | Type | Source headers | When to read |
|---|---|---|---|
| `response.headers` | `dict[str, str] \| None` | raw HTTP response headers (case-sensitive lookup) | Custom server headers; one-off correlation IDs |
| `response.response_rate_limits` | `RateLimitInfo \| None` | `x-ratelimit-*` parsed | Proactive throttling before you hit a 429 |
| `response.balance_info` | `BalanceInfo \| None` | `x-venice-balance-usd` / `-diem`: what the calling key can still spend | Cost tracking without `CostTracker` |
| `response.deprecation_info` | `DeprecationInfo \| None` | model-deprecation headers | Logging "your model is being retired" warnings |
| `response.pagination_info` | `PaginationInfo \| None` | `x-pagination-*` parsed | List-endpoint paging |

All five return `None` when the relevant headers are absent. **Always null-check before accessing nested attributes.**

## `headers`

```python
response = await client.chat.completions.create(...)
if response.headers:
    request_id = response.headers.get("x-request-id")
    server_version = response.headers.get("x-venice-version")
    log.info("venice.call", request_id=request_id, server=server_version)
```

The dict is case-sensitive on lookup but stores keys in their original case — Venice sends lowercase header names, so match the wire spelling (e.g. `.get("x-request-id")`). For correlation with Venice's server logs, `x-request-id` is the canonical header.

## `response_rate_limits` — `RateLimitInfo`

Fields (all optional; the server may emit a subset):

```python
class RateLimitInfo:
    limit_requests:     int | None        # max requests per window
    remaining_requests: int | None        # how many you have left RIGHT NOW
    reset_requests:     datetime | None    # when the request window resets (datetime)
    limit_tokens:       int | None        # max tokens per window (if token-budgeted tier)
    remaining_tokens:   int | None
    reset_tokens:       float | None       # absolute Unix seconds when token limit resets
    type:               str | None         # "user", "api_key", or "global"
```

Use to throttle proactively rather than reactively:

```python
from datetime import datetime, timezone

async def call_with_throttle(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)
    rl = response.response_rate_limits
    if rl and rl.remaining_requests is not None and rl.remaining_requests < 5:
        # We're close to the wall; back off.
        # reset_requests is a datetime; derive seconds-until-reset.
        delay = (
            max((rl.reset_requests - datetime.now(timezone.utc)).total_seconds(), 1.0)
            if rl.reset_requests
            else 1.0
        )
        await asyncio.sleep(delay)
    return response
```

For reactive handling (the request itself got 429'd), see `RateLimitError.retry_after_seconds` in `error-taxonomy.md`.

## `balance_info` — `BalanceInfo`

Fields:

```python
class BalanceInfo:
    usd:  float | None       # USD the calling key could spend before this request
    diem: float | None       # the DIEM counterpart
```

`balance_info.usd` is what the calling API key can still spend: the lesser of the account balance and what remains under the key's consumption limit, as it stood **before this request was processed**. It is not the account balance (that is `client.billing.get_balance()`), and it is not the cost of this call: a key with a $1 limit reports at most $1 however much the account holds. It is the same value `client.api_keys.get_rate_limits()` returns as `data.balances.USD`. The drop between two consecutive responses is what the earlier call cost, plus anything else spent on the same key in between:

```python
prev_balance = None
total_spent = 0.0
for question in questions:
    response = await client.chat.completions.create(...)
    if response.balance_info and response.balance_info.usd is not None:
        if prev_balance is not None:
            total_spent += prev_balance - response.balance_info.usd
        prev_balance = response.balance_info.usd
```

For most use cases, prefer `CostTracker` from `venice-py-production` — it does the bookkeeping and integrates with `BudgetManager`.

`balance_info` is `None` when the response carries neither header. Don't assume it's populated.

## `deprecation_info` — `DeprecationInfo`

Fields:

```python
class DeprecationInfo:
    warning:       str | None        # human-readable explanation
    date:          datetime | None   # parsed sunset datetime (None if absent)
    # is_deprecated -> @property: True if warning or date is set
```

Models get retired. Log `deprecation_info` to find out from your logs, not your error budget:

```python
if response.deprecation_info and response.deprecation_info.is_deprecated:
    log.warning(
        "venice.deprecated_model",
        model=response.model,
        warning=response.deprecation_info.warning,
        sunset=response.deprecation_info.date,
    )
```

Pair with `client.models.resolve_*()` so your code doesn't break when the model retires — the resolver picks a successor automatically.

## `pagination_info` — `PaginationInfo`

Parsed from the `x-pagination-*` response headers on page-based list endpoints:

```python
class PaginationInfo:
    page:        int        # current page number (1-indexed)
    limit:       int        # items per page
    total:       int        # total items across all pages
    total_pages: int        # total number of pages
```

```python
pi = response.pagination_info
if pi and pi.page < pi.total_pages:
    ...  # request the next page (page + 1)
```

> Note: `client.x402.transactions(...)` paginates differently — it is **offset-based**
> and exposes its own model on the body: `resp.data.pagination` is an
> `X402TransactionsPagination` with `limit` / `offset` / `hasMore` (not
> `pagination_info`). Loop with `offset += limit` while `resp.data.pagination.hasMore`.

## Streaming responses — same surface via `final_response`

For `client.chat.completions.stream(...)`, the metadata properties live on `stream.final_response`, populated only after `collect_with_deltas()` or `collect()` finishes (NOT after `text_deltas()`):

```python
async with stream:
    async for delta in stream.collect_with_deltas():
        ...
    if stream.final_response:
        if stream.final_response.deprecation_info:
            log.warning(...)
        if stream.final_response.balance_info:
            log.info(f"key_spendable_before_call=${stream.final_response.balance_info.usd}")
```

See `streaming.md` for why `text_deltas()` doesn't populate `final_response`.

## Why `_response` is private

The raw `aiohttp.ClientResponse` (or equivalent) is on `response._response` for advanced cases — you can read `response._response.status`, `response._response.url`, etc. **Avoid relying on this in production code** — the surface isn't a stable API. Prefer the typed properties.

## Common bugs

- **Treating a property as required**: `response.balance_info.usd` raises `AttributeError` when `balance_info` is `None`. Always null-check.
- **Assuming `balance_info.usd` is the cost of THIS call, or the account balance**: it's what the key could spend before the call. Compute deltas between responses; read the account balance from `client.billing.get_balance()`.
- **Reading `.headers["X-Request-Id"]` and getting `KeyError`**: use `.get()`, and the dict is a plain (case-sensitive) dict and Venice sends lowercased header names, so use `x-request-id`.
- **Trying to read `final_response.usage` after `text_deltas()`**: doesn't populate. See `streaming.md`.

## Related references

- `streaming.md` — `final_response` lifecycle.
- `venice-py-production/references/error-taxonomy.md` — when 429 hits, `RateLimitError.retry_after_seconds`.
- `venice-py-production/references/cost-tracking.md` — `CostTracker` reads `balance_info` automatically.
- `migration-v1-to-v2.md` — properties existed in v1 but with different attribute names; v2 unified them.
