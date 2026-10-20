"""Account commands for Venice AI CLI — Billing and usage information."""

import asyncio
import json
from typing import Any

import click

from venice_ai.cli.utils.console import console


@click.group()
def account():
    """View account billing and usage information.

    Check your balance, review usage statistics, and monitor costs.
    """
    pass


@account.command("balance")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.pass_context
def balance(ctx, output_json):
    """Show current account balance and credits.

    Displays your remaining DIEM and USD balances.

    Examples:

      venice-py account balance

      venice-py account balance --json
    """
    asyncio.run(_balance_async(ctx, output_json))


async def _balance_async(ctx, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    if plain and not is_plain_mode():
        enable_plain_mode()

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.billing.get_balance()

    # Extract balance values — handle both BillingBalanceResponse and raw dict.
    # Live API shape: {canConsume, consumptionCurrency, balances: {diem, usd}, diemEpochAllocation}
    can_consume: bool | None = None
    consumption_currency: str | None = None
    diem_balance: float | None = None
    usd_balance: float | None = None
    diem_epoch_allocation: float | None = None

    if hasattr(response, "balances"):
        balances = response.balances
        if balances is not None:
            diem_balance = getattr(balances, "diem", None)
            usd_balance = getattr(balances, "usd", None)
        diem_epoch_allocation = getattr(response, "diem_epoch_allocation", None)
        can_consume = getattr(response, "can_consume", None)
        consumption_currency = getattr(response, "consumption_currency", None)
    elif isinstance(response, dict):
        balances = response.get("balances") or {}
        if isinstance(balances, dict):
            diem_balance = balances.get("diem")
            usd_balance = balances.get("usd")
        diem_epoch_allocation = response.get("diemEpochAllocation")
        can_consume = response.get("canConsume")
        consumption_currency = response.get("consumptionCurrency")

    if output_json:
        data: dict[str, Any] = {
            "canConsume": can_consume,
            "consumptionCurrency": consumption_currency,
            "balances": {"diem": diem_balance, "usd": usd_balance},
        }
        if diem_epoch_allocation is not None:
            data["diemEpochAllocation"] = diem_epoch_allocation
        click.echo(json.dumps(data, indent=2))
        return

    from venice_ai.cli.utils.output import OutputManager

    parts = []
    if diem_balance is not None:
        parts.append(f"DIEM Balance: {diem_balance}")
    if usd_balance is not None:
        parts.append(f"USD Balance:  ${usd_balance}")
    if diem_epoch_allocation is not None:
        parts.append(f"DIEM Epoch Allocation: {diem_epoch_allocation}")
    if consumption_currency is not None:
        parts.append(f"Consumption Currency: {consumption_currency}")
    if can_consume is not None:
        parts.append(f"Can Consume: {'yes' if can_consume else 'no'}")
    OutputManager.panel("\n".join(parts), title="Account Balance", style="green")


@account.command("usage")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.option(
    "--start-date",
    default=None,
    help="Start date for usage period (YYYY-MM-DD or ISO 8601)",
)
@click.option(
    "--end-date",
    default=None,
    help="End date for usage period (YYYY-MM-DD or ISO 8601)",
)
@click.option(
    "--currency",
    type=click.Choice(["USD", "DIEM", "BUNDLED_CREDITS", "EARNED_CREDITS"]),
    default=None,
    help="Filter usage entries by currency",
)
@click.pass_context
def usage(ctx, output_json, start_date, end_date, currency):
    """Show usage statistics for your account.

    Displays billing usage data, costs, and consumption details.
    Date inputs accept YYYY-MM-DD format (e.g. 2025-01-01) or full ISO 8601
    timestamps (e.g. 2025-01-01T00:00:00Z).

    Examples:

      venice-py account usage

      venice-py account usage --start-date 2025-01-01 --end-date 2025-02-01

      venice-py account usage --currency USD

      venice-py account usage --json

      venice-py account usage --start-date 2025-01-01 --json
    """
    asyncio.run(_usage_async(ctx, output_json, start_date, end_date, currency))


def _normalize_date(date_str):
    """Convert YYYY-MM-DD to ISO 8601 format if needed."""
    if date_str is None:
        return None
    # Already has time component
    if "T" in date_str:
        return date_str
    # Simple date format — append time
    return f"{date_str}T00:00:00Z"


async def _usage_async(ctx, output_json, start_date, end_date, currency=None):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.types.enums import BillingFormatEnum

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    # Normalize date formats
    start_date_iso = _normalize_date(start_date)
    end_date_iso = _normalize_date(end_date)

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.billing.get_usage_history(
            format=BillingFormatEnum.JSON,
            currency=currency,
            startTimestamp=start_date_iso,
            endTimestamp=end_date_iso,
        )

    # Extract entries — response is BillingUsageHistoryResponse for JSON (not bytes).
    # usage-history is a cursor-paginated walk; a non-null nextCursor means this
    # first page did not reach the end of the range.
    from venice_ai.types.api.billing import BillingUsageHistoryResponse

    if isinstance(response, BillingUsageHistoryResponse):
        entries = response.data or []
        next_cursor = response.nextCursor
    elif isinstance(response, dict):
        entries = response.get("data", [])
        next_cursor = response.get("nextCursor", None)
    else:
        entries = []
        next_cursor = None

    has_more = next_cursor is not None

    if output_json:
        # Serialize entries
        serialized = []
        for entry in entries:
            if hasattr(entry, "model_dump"):
                serialized.append(entry.model_dump())
            elif isinstance(entry, dict):
                serialized.append(entry)
            else:
                serialized.append(
                    {
                        "timestamp": getattr(entry, "timestamp", None),
                        "sku": getattr(entry, "sku", None),
                        "amount": getattr(entry, "amount", None),
                        "currency": getattr(entry, "currency", None),
                        "units": getattr(entry, "units", None),
                        "pricePerUnitUsd": getattr(entry, "pricePerUnitUsd", None),
                        "notes": getattr(entry, "notes", None),
                    }
                )

        result = {"count": len(entries), "hasMore": has_more, "data": serialized}
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if not entries:
        msg = "No usage data found."
        if start_date or end_date:
            period_parts = []
            if start_date:
                period_parts.append(f"from {start_date}")
            if end_date:
                period_parts.append(f"to {end_date}")
            msg = f"No usage data found {' '.join(period_parts)}."
        if plain:
            click.echo(msg)
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        return

    # Calculate totals
    total_usd = sum(
        getattr(e, "amount", 0) or 0 for e in entries if (getattr(e, "currency", None) == "USD")
    )
    total_diem = sum(
        abs(getattr(e, "amount", 0) or 0)
        for e in entries
        if (getattr(e, "currency", None) == "DIEM")
    )
    total_units = sum(getattr(e, "units", 0) or 0 for e in entries)

    # Group by SKU
    sku_summary: dict[str, dict[str, Any]] = {}
    for entry in entries:
        sku = getattr(entry, "sku", "unknown") or "unknown"
        amount = getattr(entry, "amount", 0) or 0
        units = getattr(entry, "units", 0) or 0
        currency = getattr(entry, "currency", "USD") or "USD"
        if sku not in sku_summary:
            sku_summary[sku] = {
                "count": 0,
                "total_cost": 0.0,
                "total_units": 0.0,
                "currency": currency,
            }
        sku_summary[sku]["count"] += 1
        sku_summary[sku]["total_cost"] += abs(amount)
        sku_summary[sku]["total_units"] += units

    if plain:
        click.echo(f"Usage Summary ({len(entries)} entries)")
        click.echo("-" * 50)
        if total_usd:
            click.echo(f"Total USD:   ${total_usd:.6f}")
        if total_diem:
            click.echo(f"Total DIEM:  {total_diem:.4f}")
        click.echo(f"Total Units: {total_units:.2f}")
        if sku_summary:
            click.echo("\nBy Product SKU:")
            click.echo(f"  {'SKU':<40} {'Requests':>8} {'Cost':>12} {'Units':>10}")
            click.echo(f"  {'-' * 40} {'-' * 8} {'-' * 12} {'-' * 10}")
            for sku, data in sorted(
                sku_summary.items(), key=lambda x: x[1]["total_cost"], reverse=True
            ):
                click.echo(
                    f"  {sku:<40} {data['count']:>8} "
                    f"${data['total_cost']:>11.6f} {data['total_units']:>10.2f}"
                )
        click.echo("\nRecent Entries (last 5):")
        click.echo(f"  {'Timestamp':<25} {'SKU':<30} {'Amount':>12} {'Currency':<8}")
        click.echo(f"  {'-' * 25} {'-' * 30} {'-' * 12} {'-' * 8}")
        for entry in entries[:5]:
            ts = getattr(entry, "timestamp", "")
            ts_str = str(ts)[:24] if ts else ""
            sku = getattr(entry, "sku", "") or ""
            amount = getattr(entry, "amount", 0) or 0
            currency = getattr(entry, "currency", "") or ""
            click.echo(f"  {ts_str:<25} {sku:<30} {amount:>12.6f} {currency:<8}")
    else:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # Summary panel
        summary = Text()
        summary.append("Entries: ", style="bold")
        summary.append(f"{len(entries)}\n")
        if total_usd:
            summary.append("Total USD:   ", style="bold")
            summary.append(f"${total_usd:.6f}\n", style="green")
        if total_diem:
            summary.append("Total DIEM:  ", style="bold")
            summary.append(f"{total_diem:.4f}\n", style="cyan")
        summary.append("Total Units: ", style="bold")
        summary.append(f"{total_units:.2f}\n")

        date_range_parts = []
        if start_date:
            date_range_parts.append(f"from {start_date}")
        if end_date:
            date_range_parts.append(f"to {end_date}")
        title = "📊 Usage Summary"
        if date_range_parts:
            title += f" ({' '.join(date_range_parts)})"

        console.print(Panel(summary, title=title, border_style="blue"))

        # SKU breakdown table
        if sku_summary:
            sku_table = Table(title="Usage by Product SKU", show_lines=False)
            sku_table.add_column("SKU", style="cyan")
            sku_table.add_column("Requests", justify="right")
            sku_table.add_column("Cost", justify="right", style="green")
            sku_table.add_column("Units", justify="right")

            for sku, data in sorted(
                sku_summary.items(), key=lambda x: x[1]["total_cost"], reverse=True
            ):
                sku_table.add_row(
                    sku,
                    str(data["count"]),
                    f"${data['total_cost']:.6f}",
                    f"{data['total_units']:.2f}",
                )
            console.print(sku_table)

        # Recent entries table
        recent_table = Table(title="Recent Entries (last 5)", show_lines=False)
        recent_table.add_column("Timestamp", style="dim", no_wrap=True)
        recent_table.add_column("SKU", style="cyan")
        recent_table.add_column("Amount", justify="right", style="green")
        recent_table.add_column("Currency")
        recent_table.add_column("Units", justify="right")

        for entry in entries[:5]:
            ts = getattr(entry, "timestamp", "") or ""
            sku = getattr(entry, "sku", "") or ""
            amount = getattr(entry, "amount", 0) or 0
            currency = getattr(entry, "currency", "") or ""
            units = getattr(entry, "units", 0) or 0
            recent_table.add_row(
                str(ts)[:24],
                sku,
                f"{amount:.6f}",
                currency,
                f"{units:.2f}",
            )
        console.print(recent_table)

    # usage-history reports no totals; a nextCursor means more pages remain.
    if has_more:
        hint = (
            "More entries are available beyond this page. Narrow the range with "
            "--start-date / --end-date to see the rest."
        )
        if plain:
            click.echo(f"\n{hint}")
        else:
            console.print(f"\n[yellow]{hint}[/yellow]")


# ---------------------------------------------------------------------------
# `venice-py account keys ...` — single-key retrieve + update.
# `account list/create/delete` already live under the top-level `api-keys`
# group; this subgroup adds the missing single-key surface (retrieve/update)
# while keeping the `api-keys` group intact for backward compatibility.
# ---------------------------------------------------------------------------


@account.group("keys")
def keys():
    """Inspect and update individual API keys.

    See ``venice-py api-keys`` for list/create/delete.
    """
    pass


@keys.command("get")
@click.argument("api-key-id")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def keys_get(ctx, api_key_id, output_json):
    """Retrieve a single API key by ID.

    Examples:

      venice-py account keys get key_abc123
      venice-py account keys get key_abc123 --json
    """
    asyncio.run(_keys_get_async(ctx, api_key_id, output_json))


async def _keys_get_async(ctx, api_key_id, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        api_key_obj = await client.api_keys.retrieve(api_key_id=api_key_id)

    _render_api_key(api_key_obj, plain=plain, output_json=output_json, title="API Key")


@keys.command("update")
@click.argument("api-key-id")
@click.option("--description", default=None, help="New description for the key.")
@click.option(
    "--expiry",
    "expires_at",
    default=None,
    help='New expiration date (ISO 8601, e.g. "2026-12-31" or "2026-12-31T23:59:00Z").',
)
@click.option(
    "--limit-usd",
    "limit_usd",
    type=float,
    default=None,
    help="Epoch USD consumption limit.",
)
@click.option(
    "--limit-diem",
    "limit_diem",
    type=float,
    default=None,
    help="Epoch DIEM consumption limit.",
)
@click.option(
    "--limit-vcu",
    "limit_vcu",
    type=float,
    default=None,
    help="Epoch VCU consumption limit (legacy alias for diem).",
)
@click.option(
    "--limit-period",
    "limit_period",
    type=click.Choice(["EPOCH", "MONTH", "LIFETIME"]),
    default=None,
    help="Period over which the consumption limit resets.",
)
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def keys_update(
    ctx,
    api_key_id,
    description,
    expires_at,
    limit_usd,
    limit_diem,
    limit_vcu,
    limit_period,
    output_json,
):
    """Update an existing API key.

    At least one of ``--description``, ``--expiry``, ``--limit-usd``,
    ``--limit-diem``, ``--limit-vcu``, or ``--limit-period`` must be provided.

    Examples:

      venice-py account keys update key_abc123 --description "Production"
      venice-py account keys update key_abc123 --expiry 2026-12-31
      venice-py account keys update key_abc123 --limit-usd 50 --limit-diem 5
      venice-py account keys update key_abc123 --limit-period MONTH
    """
    # ``--limit-vcu`` is the legacy alias for ``--limit-diem``; the request
    # model (ConsumptionLimit) has no ``vcu`` field, so fold it into ``diem``
    # (explicit ``--limit-diem`` wins).
    consumption_limit: dict[str, Any] | None = None
    if limit_usd is not None or limit_diem is not None or limit_vcu is not None:
        consumption_limit = {}
        if limit_usd is not None:
            consumption_limit["usd"] = limit_usd
        diem = limit_diem if limit_diem is not None else limit_vcu
        if diem is not None:
            consumption_limit["diem"] = diem

    if (
        description is None
        and expires_at is None
        and consumption_limit is None
        and limit_period is None
    ):
        raise click.UsageError(
            "Provide at least one of --description, --expiry, --limit-usd, "
            "--limit-diem, --limit-vcu, or --limit-period."
        )

    asyncio.run(
        _keys_update_async(
            ctx,
            api_key_id=api_key_id,
            description=description,
            expires_at=expires_at,
            consumption_limit=consumption_limit,
            limit_period=limit_period,
            output_json=output_json,
        )
    )


async def _keys_update_async(
    ctx,
    *,
    api_key_id: str,
    description: str | None,
    expires_at: str | None,
    consumption_limit: dict[str, Any] | None,
    limit_period: str | None = None,
    output_json: bool,
):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        updated = await client.api_keys.update(
            id=api_key_id,
            description=description,
            expires_at=expires_at,
            consumption_limit=consumption_limit,
            limit_period=limit_period,  # type: ignore[arg-type]
        )

    _render_api_key(updated, plain=plain, output_json=output_json, title="Updated API Key")


def _render_api_key(api_key_obj, *, plain: bool, output_json: bool, title: str) -> None:
    """Render a single ApiKey object as JSON, plain text, or a Rich panel."""
    if output_json:
        if hasattr(api_key_obj, "model_dump"):
            data = api_key_obj.model_dump()
        else:
            data = {
                "id": getattr(api_key_obj, "id", None),
                "description": getattr(api_key_obj, "description", None),
                "apiKeyType": getattr(api_key_obj, "apiKeyType", None),
                "createdAt": getattr(api_key_obj, "createdAt", None),
                "expiresAt": getattr(api_key_obj, "expiresAt", None),
                "lastUsedAt": getattr(api_key_obj, "lastUsedAt", None),
                "last6Chars": getattr(api_key_obj, "last6Chars", None),
                "consumptionLimits": getattr(api_key_obj, "consumptionLimits", None),
                "usage": getattr(api_key_obj, "usage", None),
            }
        click.echo(json.dumps(data, indent=2, default=str))
        return

    key_id = getattr(api_key_obj, "id", None)
    description = getattr(api_key_obj, "description", None) or "(no description)"
    api_key_type = getattr(api_key_obj, "apiKeyType", None) or ""
    created_at = getattr(api_key_obj, "createdAt", None)
    expires_at = getattr(api_key_obj, "expiresAt", None)
    last_used = getattr(api_key_obj, "lastUsedAt", None)
    last6 = getattr(api_key_obj, "last6Chars", None)
    limits = getattr(api_key_obj, "consumptionLimits", None)

    if plain:
        click.echo(f"{title}:")
        click.echo(f"  ID:          {key_id}")
        click.echo(f"  Description: {description}")
        if api_key_type:
            click.echo(f"  Type:        {api_key_type}")
        if last6:
            click.echo(f"  Prefix:      ***{last6}")
        if created_at:
            click.echo(f"  Created:     {created_at}")
        if expires_at:
            click.echo(f"  Expires:     {expires_at}")
        click.echo(f"  Last used:   {last_used or 'Never'}")
        if limits is not None:
            click.echo(
                f"  Limits:      usd={getattr(limits, 'usd', None)} "
                f"diem={getattr(limits, 'diem', None)} "
                f"vcu={getattr(limits, 'vcu', None)}"
            )
        return

    from rich.panel import Panel
    from rich.text import Text

    lines = Text()
    lines.append("ID:          ", style="bold")
    lines.append(f"{key_id}\n", style="cyan")
    lines.append("Description: ", style="bold")
    lines.append(f"{description}\n")
    if api_key_type:
        lines.append("Type:        ", style="bold")
        lines.append(f"{api_key_type}\n", style="yellow")
    if last6:
        lines.append("Prefix:      ", style="bold")
        lines.append(f"***{last6}\n", style="dim")
    if created_at:
        lines.append("Created:     ", style="bold")
        lines.append(f"{created_at}\n", style="dim")
    if expires_at:
        lines.append("Expires:     ", style="bold")
        lines.append(f"{expires_at}\n")
    lines.append("Last used:   ", style="bold")
    lines.append(f"{last_used or 'Never'}\n", style="dim")
    if limits is not None:
        usd = getattr(limits, "usd", None)
        diem = getattr(limits, "diem", None)
        vcu = getattr(limits, "vcu", None)
        lines.append("Limits:      ", style="bold")
        lines.append(f"usd={usd}, diem={diem}" + (f", vcu={vcu}" if vcu is not None else "") + "\n")

    console.print(Panel(lines, title=title, border_style="green"))


# ---------------------------------------------------------------------------
# `venice-py account keys rate-limits` / `rate-limit-logs` — surface
# client.api_keys.get_rate_limits() and get_rate_limit_logs().
# ---------------------------------------------------------------------------


def _dump_response(obj: Any) -> Any:
    """Serialize a response model (or dict) for JSON output."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return obj


@keys.command("rate-limits")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def keys_rate_limits(ctx, output_json):
    """Show current rate limits (per-model RPM/TPM) for your API key.

    Examples:

      venice-py account keys rate-limits
      venice-py account keys rate-limits --json
    """
    asyncio.run(_keys_rate_limits_async(ctx, output_json))


async def _keys_rate_limits_async(ctx, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.api_keys.get_rate_limits()

    data = getattr(response, "data", None)

    if output_json:
        click.echo(json.dumps(_dump_response(response), indent=2, default=str))
        return

    access_permitted = getattr(data, "accessPermitted", None)
    api_tier = getattr(data, "apiTier", None)
    tier_id = getattr(api_tier, "id", None) if api_tier is not None else None
    next_epoch = getattr(data, "nextEpochBegins", None)
    model_limits = getattr(data, "rateLimits", None) or []

    # Build per-model rows. Each ModelRateLimit has a list[RateLimit] keyed by type.
    rows = []
    for ml in model_limits:
        model_id = getattr(ml, "apiModelId", None) or "(default)"
        per_type: dict[str, Any] = {}
        for rl in getattr(ml, "rateLimits", None) or []:
            per_type[getattr(rl, "type", "?")] = getattr(rl, "amount", None)
        rows.append((model_id, per_type))

    if plain:
        if tier_id is not None:
            click.echo(f"API Tier: {tier_id}")
        if access_permitted is not None:
            click.echo(f"Access Permitted: {'yes' if access_permitted else 'no'}")
        if next_epoch:
            click.echo(f"Next Epoch Begins: {next_epoch}")
        click.echo(f"\n{'MODEL':<40} {'RPM':>12} {'RPD':>12} {'TPM':>14}")
        click.echo("-" * 80)
        for model_id, per_type in rows:
            click.echo(
                f"{model_id:<40} {str(per_type.get('RPM', '-')):>12} "
                f"{str(per_type.get('RPD', '-')):>12} {str(per_type.get('TPM', '-')):>14}"
            )
        if not rows:
            click.echo("No per-model rate limits returned.")
        return

    from rich.table import Table

    header_parts = []
    if tier_id is not None:
        header_parts.append(f"[bold]API Tier:[/bold] {tier_id}")
    if access_permitted is not None:
        header_parts.append(f"[bold]Access Permitted:[/bold] {'yes' if access_permitted else 'no'}")
    if next_epoch:
        header_parts.append(f"[bold]Next Epoch Begins:[/bold] {next_epoch}")
    if header_parts:
        console.print("  ".join(header_parts))

    table = Table(title="Rate Limits by Model", show_lines=False)
    table.add_column("Model", style="cyan", no_wrap=False)
    table.add_column("RPM", justify="right", style="green")
    table.add_column("RPD", justify="right", style="green")
    table.add_column("TPM", justify="right", style="yellow")

    for model_id, per_type in rows:
        table.add_row(
            model_id,
            str(per_type.get("RPM", "-")),
            str(per_type.get("RPD", "-")),
            str(per_type.get("TPM", "-")),
        )
    console.print(table)
    if not rows:
        console.print("[yellow]No per-model rate limits returned.[/yellow]")


@keys.command("rate-limit-logs")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def keys_rate_limit_logs(ctx, output_json):
    """Show the last 50 rate-limit violations for your account.

    Examples:

      venice-py account keys rate-limit-logs
      venice-py account keys rate-limit-logs --json
    """
    asyncio.run(_keys_rate_limit_logs_async(ctx, output_json))


async def _keys_rate_limit_logs_async(ctx, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.api_keys.get_rate_limit_logs()

    entries = getattr(response, "data", None) or []

    if output_json:
        click.echo(json.dumps(_dump_response(response), indent=2, default=str))
        return

    if not entries:
        msg = "No rate limit violations found."
        if plain:
            click.echo(msg)
        else:
            console.print(f"[green]{msg}[/green]")
        return

    if plain:
        click.echo(f"{'TIMESTAMP':<26} {'MODEL':<28} {'TYPE':<8} {'TIER':<10}")
        click.echo("-" * 74)
        for e in entries:
            click.echo(
                f"{str(getattr(e, 'timestamp', '') or ''):<26} "
                f"{str(getattr(e, 'modelId', '') or ''):<28} "
                f"{str(getattr(e, 'rateLimitType', '') or ''):<8} "
                f"{str(getattr(e, 'rateLimitTier', '') or ''):<10}"
            )
        click.echo(f"\nTotal: {len(entries)} violation(s)")
        return

    from rich.table import Table

    table = Table(title="Recent Rate Limit Violations", show_lines=False)
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="red")
    table.add_column("Tier", style="yellow")

    for e in entries:
        table.add_row(
            str(getattr(e, "timestamp", "") or ""),
            str(getattr(e, "modelId", "") or ""),
            str(getattr(e, "rateLimitType", "") or ""),
            str(getattr(e, "rateLimitTier", "") or ""),
        )
    console.print(table)
    console.print(f"[dim]Total: {len(entries)} violation(s)[/dim]")
