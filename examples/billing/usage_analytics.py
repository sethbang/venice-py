#!/usr/bin/env python3
"""
Venice AI SDK - Billing and Usage Analytics
===========================================

This example demonstrates how to retrieve and analyze billing data using the Venice AI SDK:
- Checking the account balance (client.billing.get_balance) and what the
  running key can still spend (client.api_keys.get_rate_limits)
- Walking the full usage ledger for a time window (client.billing.iter_usage_history)
- Aggregated usage analytics by date/model/key (client.billing.get_usage_analytics, Beta),
  reconciled day by day against the ledger
- Analyzing costs and consumption patterns
- Understanding billing entries and inference details
- Working with cursor pagination by hand (client.billing.get_usage_history)
- Exporting a ledger window as CSV pages (client.billing.iter_usage_history_csv)

Ledger conventions worth knowing before you add anything up:
- usage-history is returned in ASCENDING timestamp order, so the newest entries
  are at the END of the walk, not the start.
- Debits (spend) are negative amounts; credits (top-ups, grants, refunds) are
  positive. Summing both together gives a net figure, not your spend.
- Entries come in three currencies. USD and BUNDLED_CREDITS are both
  USD-denominated (bundled credits are spent before USD), DIEM is separate.
  A "USD spend" figure that only sums ``currency == "USD"`` misses the
  bundled-credit share.
- Usage analytics reports gross USD-denominated spend: USD plus
  bundled-credit debits, with refunds not netted out. The ledger's debits in
  those two currencies add up to it; its credits are a separate figure.
- One inference request can produce several ledger lines (input tokens, output
  tokens, cache reads/writes). Each line repeats the same ``inferenceDetails``,
  so count requests by ``requestId`` and take token counts once per request.
"""

import asyncio
import csv
import io
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    VeniceError,
)
from venice_ai.types import BillingFormatEnum
from venice_ai.types.api.billing import (
    BillingBalanceResponse,
    BillingUsageEntry,
    BillingUsageHistoryResponse,
    UsageAnalyticsResponse,
)

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/billing/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Exit codes: 0 = every section that ran verified (the per-key spending room
# and beta analytics sections may skip), 1 = a failure, 77 = skipped because
# billing needs an ADMIN key.
EXIT_SKIPPED = 77

# The server accepts pages of 10-1000 entries; large pages keep full-window walks fast.
WALK_PAGE_SIZE = 1000

# Ledger currencies denominated in USD. Bundled credits are spent before USD,
# and usage analytics reports both together as "USD".
USD_DENOMINATED = ("USD", "BUNDLED_CREDITS")

# usage-analytics is cached server-side for this long, so a day that ended
# more recently than this before the fetch can still trail the ledger.
ANALYTICS_CACHE_TTL = timedelta(minutes=10)

# Ledger amounts carry 8 decimal places. Summing a day's lines in floating
# point drifts by far less than one unit of that (about 1e-13 for thousands of
# lines), so a daily total that differs by more than this is a real mismatch.
RECONCILE_TOLERANCE_USD = 1e-9

# Rows per CSV page (the server accepts 10-1000). Each page is its own CSV
# document, header row included.
CSV_PAGE_SIZE = 1000

# The CSV export writes one file per page into this directory, replacing the
# previous run's files. Each file name starts with the page's position in the
# walk, so the files sort in walk order whatever name the server sends.
CSV_EXPORT_DIR = "venice_usage_last_7_days_csv"


class AdminKeyRequired(Exception):
    """The running key is valid but is not an ADMIN key."""


def _needs_admin_key(error: VeniceError) -> bool:
    """True when Venice refused a billing endpoint to a valid, non-admin key.

    Venice answers an INFERENCE key with 401 "Admin API key required" (an
    unknown key gets 401 "Authentication failed").
    """
    if isinstance(error, PermissionDeniedError):
        return True
    return isinstance(error, AuthenticationError) and "admin api key required" in str(error).lower()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _window(days: int, end: datetime | None = None) -> tuple[str, str]:
    end = end or datetime.now(UTC)
    return _iso(end - timedelta(days=days)), _iso(end)


def _usd(amount: float, places: int = 4) -> str:
    """Format a signed USD amount as ``-$0.0003`` / ``+$10.0000``."""
    sign = "-" if amount < 0 else "+"
    return f"{sign}${abs(amount):,.{places}f}"


async def _walk_window(client: VeniceClient, days: int, end: datetime) -> list[BillingUsageEntry]:
    """Return every ledger entry in the ``days`` days before ``end`` (oldest first)."""
    start, stop = _window(days, end)
    return [
        entry
        async for entry in client.billing.iter_usage_history(
            page_size=WALK_PAGE_SIZE, startTimestamp=start, endTimestamp=stop
        )
    ]


def _since(entries: list[BillingUsageEntry], days: int, end: datetime) -> list[BillingUsageEntry]:
    """Filter an already-walked ledger down to the ``days`` days before ``end``.

    Walking the widest window once and filtering in memory gives every section
    the same snapshot and avoids re-reading the same ledger lines.
    """
    cutoff = end - timedelta(days=days)
    return [e for e in entries if datetime.fromisoformat(e.timestamp) >= cutoff]


def _usd_spend(entries: list[BillingUsageEntry]) -> float:
    """USD-denominated spend (USD plus bundled credits), as a positive number."""
    return -sum(e.amount for e in entries if e.currency in USD_DENOMINATED and e.amount < 0)


def _usd_credits(entries: list[BillingUsageEntry]) -> float:
    """USD-denominated credits (top-ups, grants, refunds)."""
    return sum(e.amount for e in entries if e.currency in USD_DENOMINATED and e.amount > 0)


def _spend_by_currency(entries: list[BillingUsageEntry]) -> dict[str, float]:
    """Debits per ledger currency, as positive numbers."""
    spend: dict[str, float] = defaultdict(float)
    for e in entries:
        if e.amount < 0:
            spend[e.currency] += -e.amount
    return dict(spend)


def _request_ids(entries: list[BillingUsageEntry]) -> set[str]:
    return {
        e.inferenceDetails.requestId
        for e in entries
        if e.inferenceDetails is not None and e.inferenceDetails.requestId
    }


def get_recent_usage(entries: list[BillingUsageEntry]) -> bool:
    """Break the last 30 days of the ledger down by SKU."""
    print("\n📊 Recent Usage Analysis (last 30 days)")
    print("-" * 40)

    if not entries:
        print("ℹ️ No usage data found for the last 30 days")
        return True

    print(f"📅 {entries[0].timestamp} → {entries[-1].timestamp}")
    print(f"📦 Ledger entries in window: {len(entries):,}")

    currencies = Counter(e.currency for e in entries)
    print(f"🪙 Entries by currency: {dict(currencies)}")

    spend = _usd_spend(entries)
    credits = _usd_credits(entries)
    by_currency = _spend_by_currency(entries)
    print("\n💰 USD-denominated summary (USD + bundled credits):")
    print(f"   💸 Spend (debits):    ${spend:,.4f}")
    for currency in USD_DENOMINATED:
        print(f"      of which {currency}: ${by_currency.get(currency, 0.0):,.4f}")
    print(f"   💳 Credits (top-ups, refunds): ${credits:,.4f}")
    print(f"   ⚖️ Net change:        {_usd(credits - spend)}")
    if by_currency.get("DIEM"):
        print(f"   💎 DIEM spend: {by_currency['DIEM']:,.4f} DIEM")
    print(f"   🧾 Distinct inference requests: {len(_request_ids(entries)):,}")

    sku_spend: dict[str, float] = defaultdict(float)
    sku_lines: Counter[str] = Counter()
    for e in entries:
        if e.currency in USD_DENOMINATED and e.amount < 0:
            sku_spend[e.sku] += -e.amount
            sku_lines[e.sku] += 1

    top = sorted(sku_spend.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n🏷️ Top SKUs by USD-denominated spend ({len(top)} SKUs with spend):")
    for sku, cost in top[:10]:
        share = cost / spend * 100 if spend else 0.0
        print(f"   ${cost:>10,.4f}  {share:5.1f}%  {sku_lines[sku]:>5,} lines  {sku}")
    if len(top) > 10:
        rest = sum(cost for _, cost in top[10:])
        print(f"   ${rest:>10,.4f}  … {len(top) - 10} more SKUs")

    # Ascending order: the newest entries are at the end of the walk.
    print("\n🕐 Most Recent Entries (newest first):")
    for i, entry in enumerate(reversed(entries[-5:]), 1):
        print(f"\n   {i}. {entry.timestamp}")
        print(f"      🏷️ SKU: {entry.sku}")
        print(f"      💰 Amount: {_usd(entry.amount, 6)} {entry.currency}")
        print(f"      📊 Units: {entry.units:,.6g} @ ${entry.pricePerUnitUsd:,.6g}/unit")
        if entry.notes:
            print(f"      📝 Notes: {entry.notes}")

    return True


def analyze_inference_details(entries: list[BillingUsageEntry]) -> bool:
    """Aggregate token usage per inference request over the last 7 days."""
    print("\n🔍 Inference Details Analysis (last 7 days)")
    print("-" * 40)

    # Group ledger lines by request: the token counts repeat on every line of
    # a request, while the cost is split across the lines.
    requests: dict[str, list[BillingUsageEntry]] = defaultdict(list)
    for e in entries:
        if e.inferenceDetails is not None and e.inferenceDetails.requestId:
            requests[e.inferenceDetails.requestId].append(e)

    token_requests = {
        rid: lines
        for rid, lines in requests.items()
        if lines[0].inferenceDetails is not None
        and lines[0].inferenceDetails.promptTokens is not None
    }

    if not token_requests:
        print("ℹ️ No token-metered inference found in the last 7 days")
        print("💡 Token counts are reported for LLM requests")
        return True

    lines_total = sum(len(lines) for lines in requests.values())
    print(f"🧾 {lines_total:,} ledger lines belong to {len(requests):,} distinct requests")
    print(f"🧠 {len(token_requests):,} of those requests report token counts")

    total_in = 0
    total_out = 0
    for lines in token_requests.values():
        details = lines[0].inferenceDetails
        assert details is not None
        total_in += int(details.promptTokens or 0)
        total_out += int(details.completionTokens or 0)

    n = len(token_requests)
    print("\n📊 Token Usage (counted once per request):")
    print(f"   📥 Input tokens:  {total_in:,}")
    print(f"   📤 Output tokens: {total_out:,}")
    print(f"   📊 Average input per request:  {total_in / n:,.1f}")
    print(f"   📊 Average output per request: {total_out / n:,.1f}")

    newest = sorted(token_requests.items(), key=lambda kv: kv[1][-1].timestamp, reverse=True)
    print("\n🔬 Most Recent Requests (newest first):")
    for i, (rid, lines) in enumerate(newest[:3], 1):
        details = lines[0].inferenceDetails
        assert details is not None
        tokens_in = int(details.promptTokens or 0)
        tokens_out = int(details.completionTokens or 0)
        cost = -sum(line.amount for line in lines if line.currency in USD_DENOMINATED)
        print(f"\n   {i}. {lines[-1].timestamp}  ({rid})")
        print(f"      🧾 Ledger lines: {', '.join(line.sku for line in lines)}")
        print(f"      📥 Input tokens:  {tokens_in:,}")
        print(f"      📤 Output tokens: {tokens_out:,}")
        print(f"      💰 Request cost:  ${cost:,.6f} USD")
        if tokens_in + tokens_out:
            print(f"      💱 Cost per 1M tokens: ${cost / (tokens_in + tokens_out) * 1e6:,.4f}")

    return True


async def demonstrate_pagination() -> bool:
    """Demonstrate cursor pagination by hand with get_usage_history()."""
    print("\n📄 Cursor Pagination Demonstration")
    print("-" * 40)

    start, end = _window(7)
    async with VeniceClient() as client:
        try:
            # The first request takes the filters; each response carries a
            # nextCursor token. A continuation request sends ONLY that token —
            # the filters travel inside the cursor.
            print("📖 Walking usage-history pages (last 7 days, 10 per page)...")

            page_num = 0
            total_processed = 0
            cursor: str | None = None

            while page_num < 3:
                if cursor is None:
                    usage_response = await client.billing.get_usage_history(
                        format=BillingFormatEnum.JSON,
                        startTimestamp=start,
                        endTimestamp=end,
                        pageSize=10,
                    )
                else:
                    usage_response = await client.billing.get_usage_history(cursor=cursor)
                assert isinstance(usage_response, BillingUsageHistoryResponse)

                entries = usage_response.data
                page_num += 1
                total_processed += len(entries)

                print(f"\n📄 Page {page_num}: {len(entries)} entries")
                if entries:
                    print(f"   💸 USD-denominated spend:   ${_usd_spend(entries):,.6f}")
                    print(f"   💳 USD-denominated credits: ${_usd_credits(entries):,.6f}")
                    print(f"   📅 {entries[0].timestamp} → {entries[-1].timestamp}")

                cursor = usage_response.nextCursor
                if cursor is None:
                    print(f"\n✅ Reached the end of the walk ({total_processed} entries)")
                    break

            if cursor is not None:
                print(
                    f"\n💡 Stopped after {page_num} pages ({total_processed} entries). "
                    "iter_usage_history() walks every page automatically."
                )
            return True

        except VeniceError as e:
            print(f"❌ Error demonstrating pagination: {e}")
            return False


async def export_to_csv(end: datetime, window_entries: int | None) -> bool:
    """Export the last 7 days of usage as CSV, one file per page.

    ``window_entries`` is how many ledger entries the JSON walk found in the
    same window, or ``None`` when the walk failed. The CSV walk covers the same
    window, so it must hold at least that many rows (more if entries were
    posted between the two walks).
    """
    print("\n📁 CSV Export Demonstration")
    print("-" * 40)

    start, stop = _window(7, end)
    export_dir = RESULTS_DIR / CSV_EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    for stale in export_dir.glob("*.csv"):
        stale.unlink()

    header: list[str] = []
    rows: list[list[str]] = []
    pages = 0
    print(f"📥 Exporting {start} → {stop} as CSV ({CSV_PAGE_SIZE} rows per page)...")
    async with VeniceClient() as client:
        try:
            async for page in client.billing.iter_usage_history_csv(
                startTimestamp=start,
                endTimestamp=stop,
                page_size=CSV_PAGE_SIZE,
            ):
                pages += 1
                # Path(...).name keeps a server-sent name inside export_dir.
                server_name = Path(page.filename).name if page.filename else ""
                name = f"{pages:04d}-{server_name or 'usage.csv'}"
                (export_dir / name).write_bytes(page.content)
                # Parse with the csv module: a quoted field (notes, for
                # example) may hold a comma or a line break, so counting text
                # lines would miscount rows.
                table = list(csv.reader(io.StringIO(page.text)))
                if not table:
                    print(f"❌ CSV page {pages} came back empty (no header row)")
                    return False
                if pages > 1 and table[0] != header:
                    print(f"❌ CSV page {pages} has different columns from page 1: {table[0]}")
                    return False
                header = table[0]
                page_rows = [row for row in table[1:] if row]
                if len(page_rows) > CSV_PAGE_SIZE:
                    print(
                        f"❌ Page {pages} holds {len(page_rows):,} rows, "
                        f"more than pageSize={CSV_PAGE_SIZE}"
                    )
                    return False
                rows.extend(page_rows)
        except VeniceError as e:
            print(f"❌ Error exporting CSV: {e}")
            return False

    print(f"✅ {pages} CSV page(s) written to: {export_dir}")
    print(f"📊 {len(rows):,} data rows")
    print(f"🧾 Columns: {', '.join(header)}")
    if rows:
        print(f"📅 First row: {rows[0][0]}   Last row: {rows[-1][0]}")

    ok = True
    if window_entries is not None:
        # The ledger only grows, so the export can hold more rows than the JSON
        # walk saw (entries posted in between) but never fewer.
        if len(rows) < window_entries:
            print(
                f"❌ Expected at least {window_entries:,} rows: the JSON walk found "
                f"{window_entries:,} entries in this window"
            )
            ok = False
        else:
            print(
                f"📄 The export covers the whole window ({window_entries:,} entries in the walk)."
            )

    print("\n📄 CSV Preview:")
    for row in rows[:3]:
        print(f"   {','.join(row)}")

    return ok


def cost_analysis_by_timeframe(entries: list[BillingUsageEntry], end: datetime) -> bool:
    """Compare spend over the last 24 hours, 7 days and 30 days."""
    print("\n📈 Cost Analysis by Timeframe")
    print("-" * 40)

    for period_name, days in (("Last 24 hours", 1), ("Last 7 days", 7), ("Last 30 days", 30)):
        print(f"\n📅 {period_name}:")
        window = _since(entries, days, end)
        if not window:
            print("   ℹ️ No usage data found")
            continue

        spend = _usd_spend(window)
        n_requests = len(_request_ids(window))
        print(f"   💸 USD-denominated spend: ${spend:,.4f}  ({len(window):,} ledger lines)")
        print(f"   🧾 Distinct inference requests: {n_requests:,}")
        if n_requests:
            print(f"   📊 Average spend per request: ${spend / n_requests:,.6f}")
        print(f"   📊 Daily average: ${spend / days:,.4f}")

    return True


async def show_balance() -> BillingBalanceResponse | None:
    """Display the account's balance as ``billing.get_balance()`` reports it.

    This hits the stable ``GET /billing/balance`` endpoint and reports the USD
    and DIEM balances plus the DIEM epoch allocation. Every field on
    :class:`BillingBalanceResponse` is optional, so guard before dereferencing.
    Returns the response, or ``None`` when the request failed.

    This is the ACCOUNT balance. ``client.api_keys.get_rate_limits().data.balances``
    and the ``x-venice-balance-usd`` response header (``response.balance_info``)
    report something else: what the calling API key can still spend (see
    :func:`show_key_spending_room`).
    """
    print("\n💳 Account Balance")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            balance = await client.billing.get_balance()
        except VeniceError as e:
            if _needs_admin_key(e):
                raise AdminKeyRequired(str(e)) from e
            print(f"❌ Error retrieving balance: {e}")
            return None
    assert isinstance(balance, BillingBalanceResponse)

    # can_consume tells you whether the account can currently run
    # inference (i.e. it has spendable balance / is in good standing).
    if balance.can_consume is not None:
        status = "✅ yes" if balance.can_consume else "🚫 no"
        print(f"   🟢 Can consume inference: {status}")
    if balance.consumption_currency:
        print(f"   🪙 Consumption currency: {balance.consumption_currency}")

    if balance.balances is not None:
        if balance.balances.usd is not None:
            print(f"   💵 USD balance (/billing/balance):  ${balance.balances.usd:,.4f}")
        # DIEM is a staking allowance, not a token balance: each staked DIEM
        # grants $1 of credit per epoch, reset at 00:00 UTC.
        if balance.balances.diem is not None:
            print(
                f"   💎 DIEM credit left this epoch: {balance.balances.diem:,.4f} DIEM (1 DIEM = $1)"
            )
    else:
        print("   ℹ️ No per-currency balance breakdown returned")

    if balance.diem_epoch_allocation is not None:
        print(
            f"   📐 DIEM allocated this epoch (the staked amount): "
            f"{balance.diem_epoch_allocation:,.4f} DIEM"
        )
    return balance


async def show_key_spending_room(balance: BillingBalanceResponse) -> bool | None:
    """Show what the calling key can still spend, and why it can be lower.

    ``get_rate_limits().data.balances.USD`` is the account's USD balance capped
    by what is left under this key's consumption limit for the current period.
    The two agree only when the key has no limit, or plenty left.

    Returns ``True`` when the figures were read, ``None`` when this key may not
    read them (a 403, reported as a section skip), and ``False`` on any other
    error.
    """
    print("\n🔑 What This Key Can Spend")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            limits = await client.api_keys.get_rate_limits()
            keys = await client.api_keys.list()
        except PermissionDeniedError as e:
            print(f"Section skipped: this key may not read its own limits ({e})")
            return None
        except VeniceError as e:
            print(f"❌ Error reading this key's spending room: {e}")
            return False
    print(f"   🔑 USD this key can spend (/api_keys/rate_limits): ${limits.data.balances.USD:,.4f}")
    own_suffix = os.environ.get("VENICE_API_KEY", "")[-6:]
    key = next((k for k in keys if own_suffix and k.last6Chars == own_suffix), None)
    if key is None:
        print(
            "   ℹ️ This key is not in api_keys.list() under its last 6 characters; no limit to show"
        )
        return True
    cap = key.consumptionLimits.usd if key.consumptionLimits else None
    account_usd = balance.balances.usd if balance.balances is not None else None
    if cap is None:
        print("      (this key has no USD consumption limit, so it can spend the account balance)")
    elif account_usd is not None:
        used = float(key.currentPeriodUsage.usd or 0) if key.currentPeriodUsage else 0.0
        expected = min(account_usd, cap - used)
        print(
            f"      expected min(account ${account_usd:,.4f}, {key.limitPeriod} limit "
            f"${cap:,.2f} - ${used:,.4f} used) = ${expected:,.4f}"
        )
        # Other requests on this key can be billed between the reads above.
        gap = abs(expected - limits.data.balances.USD)
        if gap > 0.05:
            print(f"      ⚠️ Off by ${gap:,.4f}; requests billed in between can explain a small gap")
    return True


def reconcile_analytics(
    analytics: UsageAnalyticsResponse,
    ledger: list[BillingUsageEntry],
    ledger_start: datetime,
    ledger_end: datetime,
    fetched_at: datetime,
) -> bool:
    """Check the analytics totals against the ledger they summarize.

    Two invariants hold for usage analytics:

    - ``byDate``, ``byModel`` and ``byKey`` are three breakdowns of the same
      spend, so their USD totals agree.
    - ``byDate[d].USD`` is the gross USD-denominated spend on UTC day ``d``:
      the ledger's USD and BUNDLED_CREDITS debits that day. Refunds and
      top-ups are separate credit lines and are not netted out.

    The daily check covers only days the ledger walk spans completely and that
    ended at least ``ANALYTICS_CACHE_TTL`` before the analytics fetch. Today's
    total, and a rolling "last 7 days" window, are not comparable to a cached
    calendar-day figure.
    """
    ok = True
    by_date = sum(d.USD for d in analytics.byDate)
    by_model = sum(m.totalUsd for m in analytics.byModel)
    by_key = sum(k.totalUsd for k in analytics.byKey)
    if analytics.byModel or analytics.byKey:
        worst = max(abs(by_date - by_model), abs(by_date - by_key))
        if worst > RECONCILE_TOLERANCE_USD:
            print(
                f"   ❌ Breakdowns disagree: byDate ${by_date:,.6f}, "
                f"byModel ${by_model:,.6f}, byKey ${by_key:,.6f}"
            )
            ok = False
        else:
            print("   ✅ byDate, byModel and byKey report the same USD total")

    ledger_daily: dict[str, float] = defaultdict(float)
    for e in ledger:
        if e.currency in USD_DENOMINATED and e.amount < 0:
            ledger_daily[e.timestamp[:10]] += -e.amount

    compared = 0
    for day in analytics.byDate:
        day_start = datetime.fromisoformat(day.date).replace(tzinfo=UTC)
        day_end = day_start + timedelta(days=1)
        if day_start < ledger_start or day_end > ledger_end:
            continue
        if day_end > fetched_at - ANALYTICS_CACHE_TTL:
            continue
        compared += 1
        expected = ledger_daily.get(day.date, 0.0)
        if abs(day.USD - expected) > RECONCILE_TOLERANCE_USD:
            print(
                f"   ❌ {day.date}: analytics ${day.USD:,.6f} but the ledger debited "
                f"${expected:,.6f} (USD + bundled credits)"
            )
            ok = False
    if compared == 0:
        print("   ℹ️ No completed day inside the ledger walk to compare yet")
    elif ok:
        print(
            f"   ✅ {compared} completed day(s) match the ledger's USD + bundled-credit "
            "debits exactly"
        )
    return ok


async def show_usage_analytics(
    ledger: list[BillingUsageEntry] | None, ledger_start: datetime, ledger_end: datetime
) -> bool | None:
    """Summarize aggregated usage via ``billing.get_usage_analytics()`` (Beta).

    This wraps the beta ``GET /billing/usage-analytics`` endpoint, which returns
    pre-aggregated breakdowns by date, model, and API key — ideal for dashboards.
    ``lookback="30d"`` asks for a relative period; lookback and explicit
    start/end dates are mutually exclusive. Results are cached server-side for
    about 10 minutes. When the ledger walk succeeded, the totals are reconciled
    against it (see :func:`reconcile_analytics`).

    If the account isn't entitled to the endpoint (or it isn't deployed), the
    section returns ``None``: a clear skip, reported as such in the summary,
    rather than failing the whole run.
    """
    print("\n📊 Usage Analytics (Beta)")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            print("📥 Fetching aggregated analytics for the last 30 days...")
            # get_usage_analytics() emits a FutureWarning on every call because the
            # endpoint is beta. This example opts in knowingly, so it silences
            # that one warning category for this call only.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                analytics = await client.billing.get_usage_analytics(lookback="30d")
        except (NotFoundError, PermissionDeniedError) as e:
            print(f"Section skipped: usage analytics is not available on this account ({e})")
            print("💡 The usage-analytics endpoint is beta and may be gated.")
            return None
        except VeniceError as e:
            print(f"❌ Error retrieving usage analytics: {e}")
            return False
    fetched_at = datetime.now(UTC)
    assert isinstance(analytics, UsageAnalyticsResponse)

    active_days = [d for d in analytics.byDate if d.USD > 0 or d.DIEM > 0]
    print(f"   🗓️ Lookback window: {analytics.lookback}")
    print(f"   📅 Days in window: {len(analytics.byDate)}, days with spend: {len(active_days)}")
    # Analytics "USD" is gross USD-denominated spend: USD plus bundled-credit
    # debits, with refunds not netted out.
    print(
        "   💸 Gross USD-denominated spend (USD + bundled credits, refunds not netted): "
        f"${sum(d.USD for d in analytics.byDate):,.4f}"
    )

    if analytics.byDate:
        print("\n   📈 Most recent daily gross spend (UTC days; today's may trail the ledger):")
        for day in analytics.byDate[-5:]:
            print(f"      {day.date}: ${day.USD:,.4f} USD-denominated / {day.DIEM:,.4f} DIEM")

    if analytics.byModel:
        print("\n   🤖 Top models by gross USD-denominated spend:")
        top_models = sorted(analytics.byModel, key=lambda m: m.totalUsd, reverse=True)
        for model in top_models[:5]:
            mtype = f" [{model.modelType}]" if model.modelType else ""
            # LLM rows report totalUnits in millions of tokens even though
            # unitType says "tokens", so label them accordingly.
            if model.unitType == "tokens":
                units = f"{model.totalUnits:,.4f}M tokens"
            else:
                units = f"{model.totalUnits:,.6g} {model.unitType}"
            print(
                f"      {model.modelName}{mtype}: ${model.totalUsd:,.4f} USD-denominated / "
                f"{model.totalDiem:,.4f} DIEM ({units})"
            )

    if analytics.byKey:
        print("\n   🔑 Top API keys by gross USD-denominated spend:")
        top_keys = sorted(analytics.byKey, key=lambda k: k.totalUsd, reverse=True)
        for key in top_keys[:5]:
            print(
                f"      {key.description}: ${key.totalUsd:,.4f} USD-denominated / "
                f"{key.totalDiem:,.4f} DIEM"
            )

    if not (analytics.byDate or analytics.byModel or analytics.byKey):
        print("   ℹ️ No usage recorded in the analytics window")

    if ledger is None:
        # The ledger walk already failed and is reported as such.
        return True
    print("\n   🔎 Reconciling analytics with the ledger walk:")
    return reconcile_analytics(analytics, ledger, ledger_start, ledger_end, fetched_at)


async def main() -> int:
    """Run all billing analytics examples.

    Returns ``0`` if every section that ran succeeded (the per-key spending
    room and beta analytics sections may skip), ``1`` on any failure, and
    ``77`` when the key is not an ADMIN key.
    """
    print("🚀 Venice AI Billing & Usage Analytics Examples")
    print("=" * 70)

    try:
        balance = await show_balance()
    except AdminKeyRequired as e:
        print(f"\nSKIPPED: billing endpoints need an ADMIN key in VENICE_API_KEY ({e})")
        return EXIT_SKIPPED
    results: list[tuple[str, bool | None]] = [("show_balance", balance is not None)]
    if balance is not None:
        results.append(("show_key_spending_room", await show_key_spending_room(balance)))

    # Walk the widest window (30 days) once, anchored at one instant, and let
    # every ledger section filter that snapshot instead of re-walking it.
    now = datetime.now(UTC)
    print("\n📥 Walking the last 30 days of the usage ledger once...")
    async with VeniceClient() as client:
        try:
            ledger = await _walk_window(client, 30, now)
        except VeniceError as e:
            print(f"❌ Error walking the usage ledger: {e}")
            print("💡 Note: Billing data access requires appropriate API permissions")
            ledger = None
    window_entries: int | None = None
    if ledger is None:
        results.append(("walk_usage_ledger", False))
    else:
        print(f"   {len(ledger):,} ledger entries")
        last_7_days = _since(ledger, 7, now)
        window_entries = len(last_7_days)
        results += [
            ("get_recent_usage", get_recent_usage(ledger)),
            ("analyze_inference_details", analyze_inference_details(last_7_days)),
            ("cost_analysis_by_timeframe", cost_analysis_by_timeframe(ledger, now)),
        ]
        # Ascending order is documented; check it rather than assume it.
        stamps = [datetime.fromisoformat(e.timestamp) for e in ledger]
        if stamps != sorted(stamps):
            print("❌ The ledger walk did not come back in ascending timestamp order")
            results.append(("ledger_order", False))

    results += [
        ("demonstrate_pagination", await demonstrate_pagination()),
        ("export_to_csv", await export_to_csv(now, window_entries)),
        (
            "show_usage_analytics",
            await show_usage_analytics(ledger, now - timedelta(days=30), now),
        ),
    ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} section(s) failed: {', '.join(failed)}")
        return 1

    passed = {name for name, ok in results if ok}
    if skipped:
        print(f"\n{len(skipped)} of {len(results)} section(s) skipped: {', '.join(skipped)}")
        print(f"✨ The other {len(passed)} billing analytics sections completed.")
    else:
        print(f"\n✨ All {len(results)} billing analytics sections completed.")
    # Name only what the sections that ran actually showed.
    concepts = [
        ("show_balance", "Account balance and DIEM epoch allowance (get_balance)"),
        ("show_key_spending_room", "What this key can spend (get_rate_limits) vs the account"),
        ("get_recent_usage", "Walking a time window once with iter_usage_history"),
        ("get_recent_usage", "Separating spend (debits) from credits, per currency"),
        ("analyze_inference_details", "Counting requests by requestId; tokens once per request"),
        ("demonstrate_pagination", "Manual cursor pagination with get_usage_history"),
        ("export_to_csv", "Exporting the ledger window as CSV pages"),
        ("show_usage_analytics", "Gross usage analytics by date/model/key (Beta) vs the ledger"),
    ]
    print("\n💡 Key concepts demonstrated:")
    for name, concept in concepts:
        if name in passed:
            print(f"   - {concept}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have billing data access.", file=sys.stderr)
        sys.exit(1)
