"""``venice-py models`` Click group.

Wraps the historical ``venice-py models`` listing command (kept as the
default callback when no subcommand is supplied, so existing flag-based
invocations like ``venice-py models --type text`` keep working) and hangs
new model-discovery subcommands off the same group.

Subcommands:
    - ``venice-py models resolve --type <type> ...`` — wraps
      :meth:`venice_ai.resources.models.Models.resolve` and the eleven
      ``resolve_*`` shortcuts.
    - ``venice-py models get <id>`` — wraps
      :meth:`venice_ai.resources.models.Models.get`.
    - ``venice-py models capabilities <id>`` — wraps
      :meth:`venice_ai.resources.models.Models.get_capabilities`.
"""

from __future__ import annotations

import asyncio

import click

from .capabilities import capabilities as capabilities_command
from .get import get as get_command
from .resolve import resolve as resolve_command


@click.group("models", invoke_without_command=True)
# Display options
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information for all models")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.option(
    "--currency",
    type=click.Choice(["usd", "diem", "both"]),
    default="both",
    help="Currency to display (default: both)",
)
@click.option("--no-legend", is_flag=True, help="Hide capability icon legend")
# Type filtering
@click.option(
    "--type",
    "-t",
    "model_type",
    multiple=True,
    type=click.Choice(
        [
            "text",
            "image",
            "embedding",
            "tts",
            "asr",
            "music",
            "upscale",
            "inpaint",
            "video",
            "decision",
        ]
    ),
    help="Filter by model type (repeatable)",
)
# Capability filtering
@click.option(
    "--function-calling",
    is_flag=True,
    help="Filter models with function calling support",
)
@click.option("--vision", is_flag=True, help="Filter models with vision capabilities")
@click.option("--reasoning", is_flag=True, help="Filter models with reasoning support")
@click.option("--web-search", is_flag=True, help="Filter models with web search")
@click.option("--code", is_flag=True, help="Filter models optimized for code")
@click.option("--response-schema", is_flag=True, help="Filter models supporting response schemas")
# Trait filtering
@click.option(
    "--trait",
    "traits",
    multiple=True,
    help="Filter by trait (can be used multiple times)",
)
# Price filtering
@click.option("--max-input", type=float, help="Maximum input price per 1M tokens (USD)")
@click.option("--max-output", type=float, help="Maximum output price per 1M tokens (USD)")
@click.option("--max-gen", type=float, help="Maximum generation price (for images)")
@click.option("--budget", type=float, help="Maximum average price (input+output)/2")
# Status filtering
@click.option("--beta", is_flag=True, default=None, help="Show only beta models")
@click.option("--no-beta", "no_beta_flag", is_flag=True, help="Exclude beta models")
@click.option("--online", is_flag=True, default=None, help="Show only online models")
# Sorting
@click.option(
    "--sort",
    type=click.Choice(["name", "id", "price-asc", "price-desc", "context", "created"]),
    default="name",
    help="Sort models by criterion (default: name)",
)
# Search & detail
@click.option("--search", help="Search in model names and descriptions")
@click.option("--id", "--detail", "detail_id", help="Show details for specific model")
@click.option("--compare", "compare_ids", help="Compare multiple models (comma-separated IDs)")
@click.pass_context
def models(ctx: click.Context, **kwargs) -> None:
    """List, filter, resolve, and inspect available AI models.

    With no subcommand, lists models with the supplied filter flags.
    Use ``venice-py models <subcommand> --help`` for resolve/get/capabilities.

    Examples:

      # List all models (default)
      venice-py models

      # Show only text models
      venice-py models --type text

      # Filter by capabilities
      venice-py models --vision --function-calling

      # Resolve the cheapest chat model with function calling
      venice-py models resolve --type chat --function-calling

      # Inspect a single model
      venice-py models get llama-3.3-70b

      # Get a typed capability view
      venice-py models capabilities llama-3.3-70b
    """
    if ctx.invoked_subcommand is not None:
        # Subcommand will run instead — drop top-level filter flags.
        return

    from .command import list_models

    # Handle beta flag logic
    beta_value: bool | None = None
    if kwargs.get("beta"):
        beta_value = True
    elif kwargs.get("no_beta_flag"):
        beta_value = False
    kwargs["beta"] = beta_value
    kwargs.pop("no_beta_flag", None)

    asyncio.run(list_models(ctx, **kwargs))


# Register subcommands
models.add_command(resolve_command)
models.add_command(get_command)
models.add_command(capabilities_command)


__all__ = ["models"]
