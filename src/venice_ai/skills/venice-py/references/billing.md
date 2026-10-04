# Billing — `client.billing.*`

Sourced from `src/venice_ai/resources/billing.py` and `src/venice_ai/types/api/billing.py`. The billing surface covers prepaid USD/DIEM balances, per-call usage records, and (beta) aggregated analytics. Everything below is for traditional API-key accounts; x402 wallet billing is a separate surface — see `venice-py-x402/references/balance-and-topup.md`.

## `get_balance` — current USD/DIEM headroom

```python
balance = await client.billing.get_balance()
# BillingBalanceResponse:
#   balance.can_consume          : bool | None
#   balance.consumption_currency : Literal["USD", "VCU", "DIEM", "BUNDLED_CREDITS"] | None
#   balance.balances             : BillingBalances | None    ← NESTED, can be None
#   balance.diem_epoch_allocation: float | None
# BillingBalances:
#   balance.balances.usd  : float | None
#   balance.balances.diem : float | None
```

**The `.balances.` nesting is the trap.** `balance.usd` does not exist — agents who flatten the access path get an `AttributeError` (or worse, a silent `None` from a defensive `getattr`). The wire fields are camelCase (`canConsume`, `consumptionCurrency`, `diemEpochAllocation`), but `populate_by_name=True` lets you read snake_case attributes.

```python
balance = await client.billing.get_balance()
print(f"Can consume: {balance.can_consume}")
if balance.balances:
    print(f"USD:  {balance.balances.usd}")
    print(f"DIEM: {balance.balances.diem}")
```

`balance_info` on response objects (`response.balance_info.usd`) is a *different* value — that one is flat, sourced from response headers, and reports what the calling API key can still spend: the lesser of the account balance and what remains under the key's consumption limit, before the request was processed. `get_balance()` is the account's balance. Don't confuse them.

## `get_usage_history` — cursor-paginated per-call usage records

```python
page = await client.billing.get_usage_history(
    startTimestamp="2026-04-01T00:00:00Z",           # first page only
    endTimestamp="2026-05-01T00:00:00Z",             # first page only
    currency="USD",                                  # "USD" | "DIEM" | "BUNDLED_CREDITS"; None for all
    pageSize=1000,                                    # 10..1000, default 1000
)
# BillingUsageHistoryResponse:
#   page.data       : list[BillingUsageEntry]         ← ascending timestamp order
#   page.nextCursor : str | None                      ← None on the last page
```

The endpoint is a **keyset walk**: the first request takes the filters, and every response carries a `nextCursor`. A continuation request sends **only** the cursor — the filters travel inside it, and the server rejects filters supplied alongside a cursor (the SDK raises `ValueError` before the request if you try):

```python
if page.nextCursor:
    page = await client.billing.get_usage_history(cursor=page.nextCursor)
```

Set `format=BillingFormatEnum.CSV` to receive the page as a CSV document instead:

```python
csv_page = await client.billing.get_usage_history(format=BillingFormatEnum.CSV, startTimestamp="2026-04-01T00:00:00Z")
# BillingUsageHistoryCsvPage:
#   csv_page.content    : bytes        ← one complete CSV document, header row included
#   csv_page.text       : str          ← content decoded as UTF-8
#   csv_page.nextCursor : str | None   ← from the x-next-cursor header; None on the last page
#   csv_page.filename   : str | None   ← from Content-Disposition; ordering not guaranteed
```

To export a whole window, `iter_usage_history_csv()` walks the cursor and yields one page at a time. Every page repeats the header row, so save each page as its own file or drop the header of every page after the first. Name or sort the files by their position in the walk: nothing guarantees that the server's `filename` values sort in walk order.

```python
index = 0
async for csv_page in client.billing.iter_usage_history_csv(startTimestamp="2026-04-01T00:00:00Z"):
    index += 1
    Path(f"usage-{index:04d}.csv").write_bytes(csv_page.content)
```

Billing endpoints can be slow; the SDK gives each request -- the response and, for CSV, its body -- a 10-second deadline that surfaces as `BillingTimeoutError`. The deadline wraps each page request on its own, so a longer range means more pages rather than more entries per page. To recover, lower the page size (`pageSize` / `page_size`), which is what sets the size of a page, and use a range of at least a day where data is known to exist: the endpoint can hang on ranges shorter than about 15 minutes or on filters that match nothing. An iterator has no resume handle, so restart the walk from the first page.

For unbounded enumeration use the paginator helper, which threads the cursor for you:

```python
async for entry in client.billing.iter_usage_history(currency="USD", page_size=1000):
    print(entry.timestamp, entry.amount, entry.sku)
```

`currency="USD"` returns only entries paid in USD. Spend paid from bundled credits is in the `"BUNDLED_CREDITS"` entries, so a USD-only walk understates what the account spent in USD terms. To reconcile against analytics, walk `currency="USD"` and `currency="BUNDLED_CREDITS"` and add their debits; `currency=None` also returns DIEM and refund entries.

## `get_usage_analytics` — beta aggregate dashboard

```python
analytics = await client.billing.get_usage_analytics(lookback="30d")
# OR specify both endpoints:
analytics = await client.billing.get_usage_analytics(
    startDate="2026-04-01", endDate="2026-05-01",      # YYYY-MM-DD here, NOT ISO 8601
)
```

This wraps a **beta** endpoint — schema and behavior may change. Returns aggregates by date, model, and API key. Source: `Billing.get_usage_analytics` in `src/venice_ai/resources/billing.py`.

The USD figures (`byDate[].USD`, `byModel[].totalUsd`, `byKey[].totalUsd`) are **gross USD-denominated spend**: USD debits plus bundled-credit debits counted at their USD value, with refunds not netted out. They match the usage ledger only when you add the debits in the `"USD"` and `"BUNDLED_CREDITS"` entries together and leave refunds out.

## When to read which method

- **`get_balance`** — pre-flight check before a long batch ("do we have headroom?"). Cheap (one call). Don't poll it tightly; the value moves with every paid response.
- **`response.balance_info`** — what the calling key could spend before the request, from response headers. Free (no extra request). Capped by the key's consumption limit, so it can be far below the account balance. Use this for a running tally of the key's spend during a session.
- **`get_usage_history` / `iter_usage_history`** — historical reconciliation, per-call audit, generating invoices. Heavier — walk the cursor.
- **`get_usage_analytics`** — dashboards. Beta.

## Pitfalls

- **Reading `balance.usd`** instead of `balance.balances.usd` — the nesting is real.
- **Polling `get_balance` after every call** — read `response.balance_info` instead; it's emitted on the same request. Remember it is the key's spendable amount, not the account balance.
- **Resending filters with a cursor** on `get_usage_history` — a continuation carries the cursor alone; filters alongside it are rejected. The `iter_usage_history` helper handles this for you.
- **Confusing this with x402** — different surface, different shape. `client.x402.balance(...)` returns a `data`-envelope shape (see `venice-py-x402/references/balance-and-topup.md`).

## See also

- `venice-py/references/response-shapes.md` — quick cross-reference for all balance / usage shapes
- `venice-py-production/references/cost-tracking.md` — the `CostTracker` / `BudgetManager` layer that consumes balance info automatically
- `venice-py-x402/references/balance-and-topup.md` — wallet-funded billing surface, distinct from this
