"""API Keys commands for Venice AI CLI — Manage API keys."""

import asyncio
import json
from typing import TYPE_CHECKING, Literal, cast

import click

from venice_ai.cli.utils.console import console

if TYPE_CHECKING:
    from venice_ai.core.models.common import ConsumptionLimit


@click.group("api-keys")
def api_keys():
    """Manage your Venice AI API keys.

    List, create, and delete API keys for your account.
    """
    pass


@api_keys.command("list")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.pass_context
def list_keys(ctx, output_json):
    """List all API keys for your account.

    Displays key names, prefixes, creation dates, and last used dates.

    Examples:

      venice-py api-keys list

      venice-py api-keys list --json
    """
    asyncio.run(_list_keys_async(ctx, output_json))


async def _list_keys_async(ctx, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        keys = await client.api_keys.list()

    if output_json:
        data = []
        for key in keys:
            if hasattr(key, "model_dump"):
                data.append(key.model_dump())
            elif isinstance(key, dict):
                data.append(key)
            else:
                data.append(
                    {
                        "id": getattr(key, "id", None),
                        "description": getattr(key, "description", None),
                        "apiKeyType": getattr(key, "apiKeyType", None),
                        "createdAt": getattr(key, "createdAt", None),
                        "lastUsedAt": getattr(key, "lastUsedAt", None),
                        "expiresAt": getattr(key, "expiresAt", None),
                        "last6Chars": getattr(key, "last6Chars", None),
                    }
                )
        click.echo(json.dumps(data, indent=2, default=str))
        return

    if not keys:
        msg = "No API keys found."
        if plain:
            click.echo(msg)
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        return

    if plain:
        click.echo(f"{'NAME/DESCRIPTION':<35} {'PREFIX':<12} {'CREATED':<22} {'LAST USED':<22}")
        click.echo("-" * 95)
        for key in keys:
            description = getattr(key, "description", "") or ""
            last6 = getattr(key, "last6Chars", None)
            prefix = f"***{last6}" if last6 else "N/A"
            created_at = str(getattr(key, "createdAt", "") or "")[:20]
            last_used = str(getattr(key, "lastUsedAt", "") or "Never")[:20]
            click.echo(f"{description:<35} {prefix:<12} {created_at:<22} {last_used:<22}")
        click.echo(f"\nTotal: {len(keys)} key(s)")
    else:
        from rich.table import Table

        table = Table(title="Venice AI API Keys", show_lines=False)
        table.add_column("Name / Description", style="bold white", no_wrap=False)
        table.add_column("Prefix", style="cyan", no_wrap=True)
        table.add_column("Type", style="yellow")
        table.add_column("Created", style="dim", no_wrap=True)
        table.add_column("Last Used", style="dim", no_wrap=True)

        for key in keys:
            description = getattr(key, "description", None) or "(no description)"
            last6 = getattr(key, "last6Chars", None)
            prefix = f"***{last6}" if last6 else "N/A"
            key_type = getattr(key, "apiKeyType", None) or ""
            created_at = str(getattr(key, "createdAt", "") or "")[:20]
            last_used_raw = getattr(key, "lastUsedAt", None)
            last_used = str(last_used_raw)[:20] if last_used_raw else "Never"

            table.add_row(description, prefix, key_type, created_at, last_used)

        console.print(table)
        console.print(f"\n[dim]Total: {len(keys)} key(s)[/dim]")


def _build_consumption_limit(limit_usd, limit_diem, limit_vcu) -> "ConsumptionLimit | None":
    """Assemble a ``ConsumptionLimit`` from the CLI limit flags.

    The request model (``ConsumptionLimit``) exposes only ``usd`` and ``diem``
    — there is no ``vcu`` field. ``vcu`` is the documented legacy alias for
    ``diem``, so ``--limit-vcu`` folds into ``diem`` when ``--limit-diem`` is
    not supplied (explicit ``--limit-diem`` wins). Returns ``None`` when no
    limit flag was provided.
    """
    if limit_usd is None and limit_diem is None and limit_vcu is None:
        return None
    from venice_ai.core.models.common import ConsumptionLimit

    # Prefer explicit diem; fall back to the legacy vcu alias.
    diem = limit_diem if limit_diem is not None else limit_vcu
    return ConsumptionLimit(usd=limit_usd, diem=diem)


@api_keys.command("create")
@click.option(
    "--name", "--description", "name", required=True, help="Name/description for the new API key"
)
@click.option(
    "--type",
    "api_key_type",
    type=click.Choice(["INFERENCE", "ADMIN"]),
    default="INFERENCE",
    show_default=True,
    help="API key type.",
)
@click.option(
    "--limit-usd",
    "limit_usd",
    type=float,
    default=None,
    help="USD consumption limit.",
)
@click.option(
    "--limit-diem",
    "limit_diem",
    type=float,
    default=None,
    help="DIEM consumption limit.",
)
@click.option(
    "--limit-vcu",
    "limit_vcu",
    type=float,
    default=None,
    help="VCU consumption limit (legacy alias for --limit-diem).",
)
@click.option(
    "--limit-period",
    "limit_period",
    type=click.Choice(["EPOCH", "MONTH", "LIFETIME"]),
    default=None,
    help="Period over which the consumption limit resets.",
)
@click.option(
    "--model-privacy",
    "model_privacy",
    type=click.Choice(["ALL", "PRIVATE_TEXT", "PRIVATE_ONLY"]),
    default=None,
    help=(
        "Restrict which models this key may call, by privacy tier. "
        "PRIVATE_TEXT requires text and embedding models to be Private, TEE, "
        "or E2EE; PRIVATE_ONLY requires it of every model. "
        "Defaults to the account setting."
    ),
)
@click.option(
    "--expiry",
    "expiry",
    default=None,
    help='Expiration date (ISO 8601, e.g. "2026-12-31" or "2026-12-31T23:59:00Z").',
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.pass_context
def create_key(
    ctx,
    name,
    api_key_type,
    limit_usd,
    limit_diem,
    limit_vcu,
    limit_period,
    model_privacy,
    expiry,
    output_json,
):
    """Create a new API key.

    The secret key is only shown once — save it immediately!

    Examples:

      venice-py api-keys create --name "Production Key"

      venice-py api-keys create --name "Admin" --type ADMIN

      venice-py api-keys create --name "Capped" --limit-usd 50 --limit-period MONTH

      venice-py api-keys create --name "Private" --model-privacy PRIVATE_ONLY

      venice-py api-keys create --name "My App" --json
    """
    asyncio.run(
        _create_key_async(
            ctx,
            name=name,
            api_key_type=api_key_type,
            limit_usd=limit_usd,
            limit_diem=limit_diem,
            limit_vcu=limit_vcu,
            limit_period=limit_period,
            model_privacy=model_privacy,
            expiry=expiry,
            output_json=output_json,
        )
    )


async def _create_key_async(
    ctx,
    *,
    name,
    api_key_type="INFERENCE",
    limit_usd=None,
    limit_diem=None,
    limit_vcu=None,
    limit_period=None,
    model_privacy=None,
    expiry=None,
    output_json=False,
):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.types.api import CreateApiKeyRequest

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    consumption_limit = _build_consumption_limit(limit_usd, limit_diem, limit_vcu)

    request = CreateApiKeyRequest(
        description=name,
        # api_key_type is constrained to these values by the click.Choice option.
        apiKeyType=cast(Literal["INFERENCE", "ADMIN"], api_key_type),
        consumptionLimit=consumption_limit,
        limitPeriod=limit_period,
        modelPrivacy=model_privacy,
        expiresAt=expiry,
    )

    async with VeniceClient(**get_client_kwargs()) as client:
        new_key = await client.api_keys.create(api_key_request=request)

    # Extract fields
    key_id = getattr(new_key, "id", None)
    description = getattr(new_key, "description", None) or name
    api_key_type = getattr(new_key, "apiKeyType", None) or "INFERENCE"
    created_at = getattr(new_key, "createdAt", None)
    secret_key = getattr(new_key, "apiKey", None)

    if output_json:
        data = {
            "id": key_id,
            "description": description,
            "apiKeyType": api_key_type,
            "createdAt": str(created_at) if created_at else None,
            "apiKey": secret_key,
        }
        click.echo(json.dumps(data, indent=2, default=str))
        return

    if plain:
        click.echo("API key created successfully!")
        click.echo(f"ID:          {key_id}")
        click.echo(f"Description: {description}")
        click.echo(f"Type:        {api_key_type}")
        if created_at:
            click.echo(f"Created:     {created_at}")
        if secret_key:
            click.echo("")
            click.echo("WARNING: Save this key now — it will NOT be shown again!")
            click.echo(f"API Key: {secret_key}")
    else:
        from rich.panel import Panel
        from rich.text import Text

        lines = Text()
        lines.append("ID:          ", style="bold")
        lines.append(f"{key_id}\n", style="cyan")
        lines.append("Description: ", style="bold")
        lines.append(f"{description}\n")
        lines.append("Type:        ", style="bold")
        lines.append(f"{api_key_type}\n", style="yellow")
        if created_at:
            lines.append("Created:     ", style="bold")
            lines.append(f"{created_at}\n", style="dim")

        if secret_key:
            lines.append("\n")
            lines.append(
                "⚠️  Save this key immediately — it will NOT be shown again!\n",
                style="bold red",
            )
            lines.append("API Key: ", style="bold")
            lines.append(f"{secret_key}\n", style="bold green")

        panel = Panel(lines, title="✅ API Key Created", border_style="green")
        console.print(panel)


@api_keys.command("delete")
@click.argument("key-id")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt")
@click.pass_context
def delete_key(ctx, key_id, yes):
    """Delete an API key.

    KEY_ID is the unique identifier of the key to delete.
    You will be prompted to confirm before deletion.

    Examples:

      venice-py api-keys delete key_abc123

      venice-py api-keys delete key_abc123 --yes
    """
    asyncio.run(_delete_key_async(ctx, key_id, yes))


async def _delete_key_async(ctx, key_id, yes):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    # Confirm deletion (skipped if --yes / -y flag is provided)
    if not yes:
        if plain:
            confirmed = click.confirm(f"Delete API key '{key_id}'? This cannot be undone.")
        else:
            console.print(f"[yellow]⚠️  About to delete API key:[/yellow] [bold]{key_id}[/bold]")
            confirmed = click.confirm("This action cannot be undone. Continue?")
    else:
        confirmed = True

    if not confirmed:
        if plain:
            click.echo("Deletion cancelled.")
        else:
            console.print("[dim]Deletion cancelled.[/dim]")
        return

    async with VeniceClient(**get_client_kwargs()) as client:
        result = await client.api_keys.delete(api_key_id=key_id)

    # Check result
    success = True
    if hasattr(result, "success"):
        success = result.success
    elif isinstance(result, dict):
        success = result.get("success", True)

    if success:
        if plain:
            click.echo(f"API key '{key_id}' deleted successfully.")
        else:
            console.print(f"[bold green]✅ API key deleted:[/bold green] [cyan]{key_id}[/cyan]")
    else:
        if plain:
            click.echo(f"Failed to delete API key '{key_id}'.")
        else:
            console.print(
                f"[bold red]❌ Failed to delete API key:[/bold red] [cyan]{key_id}[/cyan]"
            )
