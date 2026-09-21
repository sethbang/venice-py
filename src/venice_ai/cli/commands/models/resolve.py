"""``venice-py models resolve`` — auto-pick a model by type and capabilities.

Wraps :meth:`venice_ai.resources.models.Models.resolve` and the eleven
``resolve_*`` shortcuts (chat / embedding / image / video / tts / asr /
inpaint / music / decision / video-upscale / cheapest-video) behind a single
``--type`` flag.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import click

from ...utils.console import console

# All supported types, including the two non-resolve helpers.
_TYPE_CHOICES = [
    "chat",
    "embedding",
    "image",
    "video",
    "tts",
    "asr",
    "inpaint",
    "music",
    "decision",
    "video-upscale",
    "cheapest-video",
]


@click.command("resolve")
@click.option(
    "--type",
    "model_type",
    type=click.Choice(_TYPE_CHOICES, case_sensitive=False),
    required=True,
    help="Model category to resolve.",
)
# Chat-only filters
@click.option("--function-calling", is_flag=True, help="Require function-calling support (chat).")
@click.option("--vision", is_flag=True, help="Require vision/image input (chat).")
@click.option("--reasoning", is_flag=True, help="Require reasoning support (chat).")
@click.option("--code", is_flag=True, help="Require code-optimized (chat).")
@click.option("--response-schema", is_flag=True, help="Require structured-output support (chat).")
@click.option("--min-context-tokens", type=int, default=None, help="Minimum context window (chat).")
@click.option("--require-private", is_flag=True, help="Privacy-first only (chat).")
# Video-only filters
@click.option(
    "--video-type",
    type=click.Choice(["text-to-video", "image-to-video"], case_sensitive=False),
    default=None,
    help="Filter video models by direction.",
)
@click.option("--audio", is_flag=True, help="Require audio support (video).")
@click.option("--min-resolution", default=None, help="Minimum video resolution (e.g. 720p).")
@click.option("--min-duration", default=None, help="Minimum video duration (e.g. 5s).")
# cheapest-video extras
@click.option(
    "--duration",
    default="5s",
    help="Quote duration for --type cheapest-video (default: 5s).",
)
@click.option("--resolution", default=None, help="Quote resolution for --type cheapest-video.")
@click.option("--aspect-ratio", default=None, help="Quote aspect ratio for --type cheapest-video.")
# General filters
@click.option(
    "--preferred",
    "preferred_models",
    multiple=True,
    help="Preferred model IDs in priority order (repeatable).",
)
@click.option(
    "--exclude",
    "exclude_models",
    multiple=True,
    help="Model IDs to exclude (repeatable).",
)
@click.option(
    "--include-beta",
    "include_beta",
    is_flag=True,
    help="Include beta models (default: excluded).",
)
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def resolve(
    ctx: click.Context,
    model_type: str,
    function_calling: bool,
    vision: bool,
    reasoning: bool,
    code: bool,
    response_schema: bool,
    min_context_tokens: int | None,
    require_private: bool,
    video_type: str | None,
    audio: bool,
    min_resolution: str | None,
    min_duration: str | None,
    duration: str,
    resolution: str | None,
    aspect_ratio: str | None,
    preferred_models: tuple[str, ...],
    exclude_models: tuple[str, ...],
    include_beta: bool,
    output_json: bool,
) -> None:
    """Resolve a model ID by type and capability requirements.

    Examples:

      venice-py models resolve --type chat
      venice-py models resolve --type chat --function-calling --vision
      venice-py models resolve --type embedding
      venice-py models resolve --type video --audio --min-resolution 720p
      venice-py models resolve --type cheapest-video --duration 5s
      venice-py models resolve --type video-upscale
      venice-py models resolve --type chat --json
    """
    asyncio.run(
        _resolve_async(
            ctx,
            model_type=model_type.lower(),
            function_calling=function_calling,
            vision=vision,
            reasoning=reasoning,
            code=code,
            response_schema=response_schema,
            min_context_tokens=min_context_tokens,
            require_private=require_private,
            video_type=video_type,
            audio=audio,
            min_resolution=min_resolution,
            min_duration=min_duration,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            preferred_models=list(preferred_models) if preferred_models else None,
            exclude_models=list(exclude_models) if exclude_models else None,
            include_beta=include_beta,
            output_json=output_json,
        )
    )


async def _resolve_async(
    ctx: click.Context,
    *,
    model_type: str,
    function_calling: bool,
    vision: bool,
    reasoning: bool,
    code: bool,
    response_schema: bool,
    min_context_tokens: int | None,
    require_private: bool,
    video_type: str | None,
    audio: bool,
    min_resolution: str | None,
    min_duration: str | None,
    duration: str,
    resolution: str | None,
    aspect_ratio: str | None,
    preferred_models: list[str] | None,
    exclude_models: list[str] | None,
    include_beta: bool,
    output_json: bool,
) -> None:
    from venice_ai import VeniceClient

    from ...config import get_client_kwargs

    exclude_beta = not include_beta

    async with VeniceClient(**get_client_kwargs()) as client:
        if model_type == "cheapest-video":
            result = await client.models.resolve_cheapest_video(
                duration=duration,
                video_type=video_type,  # type: ignore[arg-type]
                resolution=resolution,
                audio=audio if audio else None,
                aspect_ratio=aspect_ratio,
                exclude_models=exclude_models,
                exclude_beta=exclude_beta,
            )
            _render_cheapest_video(result, output_json=output_json)
            return

        if model_type == "video-upscale":
            model_id = await client.models.resolve_video_upscale(
                preferred_models=preferred_models,
                exclude_models=exclude_models,
            )
            _render_model_id(model_id, model_type=model_type, output_json=output_json)
            return

        # General resolve() path covers chat / embedding / image / video / tts /
        # asr / inpaint / music / decision. We pass only the kwargs that apply to the
        # requested type — the SDK ignores irrelevant flags but staying tidy
        # makes failures easier to diagnose.
        kwargs: dict[str, Any] = {
            "type": model_type,
            "preferred_models": preferred_models,
            "exclude_models": exclude_models,
            "exclude_beta": exclude_beta,
        }

        if model_type == "chat":
            kwargs.update(
                require_function_calling=function_calling,
                require_vision=vision,
                require_reasoning=reasoning,
                require_code_optimization=code,
                require_response_schema=response_schema,
                min_context_tokens=min_context_tokens,
                require_private=require_private,
            )
        elif model_type == "video":
            kwargs.update(
                video_type=video_type,
                require_audio=audio,
                min_resolution=min_resolution,
                min_duration=min_duration,
            )
        # embedding / image / tts / asr / inpaint / music: only general kwargs.

        try:
            model_id = await client.models.resolve(**kwargs)
        except ValueError as exc:
            if output_json:
                click.echo(json.dumps({"error": str(exc)}))
            else:
                console.print(f"[red]Could not resolve model:[/red] {exc}")
            ctx.exit(1)
            return

        _render_model_id(model_id, model_type=model_type, output_json=output_json)


def _render_model_id(model_id: str, *, model_type: str, output_json: bool) -> None:
    """Render a single resolved model ID to stdout."""
    if output_json:
        click.echo(json.dumps({"type": model_type, "model": model_id}))
        return

    from rich.panel import Panel

    console.print(
        Panel(
            f"[bold cyan]{model_id}[/bold cyan]",
            title=f"Resolved {model_type} model",
            border_style="green",
        )
    )


def _render_cheapest_video(result: Any, *, output_json: bool) -> None:
    """Render a CheapestVideoResult to stdout."""
    # ``is_dataclass()`` is True for both instances and the class itself; we
    # only want the instance branch here. The ``not isinstance(..., type)``
    # guard narrows mypy's view to ``DataclassInstance``.
    if is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)
    elif hasattr(result, "model_dump"):
        payload = result.model_dump()
    else:
        payload = {
            "model": getattr(result, "model", None),
            "quote_usd": getattr(result, "quote_usd", None),
            "all_quotes": getattr(result, "all_quotes", {}) or {},
        }

    if output_json:
        click.echo(json.dumps(payload, default=str))
        return

    from rich.panel import Panel
    from rich.table import Table

    quote = payload.get("quote_usd")
    quote_str = f"${quote:.6f}" if isinstance(quote, int | float) else str(quote)
    panel = Panel(
        f"[bold cyan]{payload.get('model')}[/bold cyan]\nQuote: [green]{quote_str}[/green]",
        title="Cheapest video model",
        border_style="green",
    )
    console.print(panel)

    all_quotes = payload.get("all_quotes") or {}
    if all_quotes:
        table = Table(title="All quotes", show_lines=False)
        table.add_column("Model", style="cyan")
        table.add_column("USD", justify="right", style="green")
        for mid, price in sorted(all_quotes.items(), key=lambda kv: kv[1]):
            try:
                table.add_row(mid, f"${float(price):.6f}")
            except (TypeError, ValueError):
                table.add_row(mid, str(price))
        console.print(table)


__all__ = ["resolve"]
