"""``venice-py models resolve`` — auto-pick a model by type and capabilities.

Wraps :meth:`venice_ai.resources.models.Models.resolve` and the eleven
``resolve_*`` shortcuts (chat / embedding / image / video / tts / asr /
inpaint / music / decision / video-upscale / cheapest-video / cheapest-music)
behind a single ``--type`` flag.
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
    "cheapest-music",
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
@click.option(
    "--prompt-caching",
    is_flag=True,
    help="Require a listed cached-input price (chat). Venice has no caching flag; "
    "a price does not guarantee cache hits.",
)
@click.option("--multiple-images", is_flag=True, help="Require several images per request (chat).")
@click.option("--e2ee", is_flag=True, help="Require end-to-end encryption (chat).")
@click.option(
    "--exclude-reasoning",
    is_flag=True,
    help="Only models that never reason (chat); cannot be combined with --reasoning.",
)
@click.option(
    "--exclude-uncensored", is_flag=True, help="Skip models flagged uncensored (chat/image)."
)
@click.option(
    "--custom-size",
    is_flag=True,
    help="Only models sized by explicit width/height, not aspect_ratio (image).",
)
# Image / inpaint
@click.option(
    "--quality",
    default=None,
    help="Quality tier the model must offer (image/inpaint); also the tier priced by "
    "--prefer cheapest.",
)
# Music-only filters
@click.option(
    "--music-only",
    is_flag=True,
    help="Skip TTS, sound-effect and voice-changer models typed as music (music).",
)
@click.option(
    "--force-instrumental", is_flag=True, help="Require force_instrumental support (music)."
)
# Video-only filters
@click.option(
    "--video-type",
    type=click.Choice(["text-to-video", "image-to-video"], case_sensitive=False),
    default=None,
    help="Filter video models by direction.",
)
@click.option(
    "--input-mode",
    type=click.Choice(
        ["image", "reference", "first_last_frame", "transition", "multi_angle"],
        case_sensitive=False,
    ),
    default=None,
    help="Input an image-to-video model must take (video, cheapest-video); "
    "--video-type image-to-video alone means plain image-to-video.",
)
@click.option(
    "--audio",
    is_flag=True,
    help="Require audio support (video). For cheapest-video, also quote with audio on.",
)
@click.option(
    "--audio-configurable",
    is_flag=True,
    help="Require models that accept an explicit audio on/off (video, cheapest-video).",
)
@click.option("--min-resolution", default=None, help="Minimum video resolution (e.g. 720p).")
@click.option("--min-duration", default=None, help="Minimum video duration (e.g. 5s).")
# cheapest-video extras
@click.option(
    "--duration",
    default=None,
    help="Clip length to quote: for --type cheapest-video a listed duration (default: each "
    "model's shortest); for --type cheapest-music and --type music, seconds.",
)
@click.option(
    "--resolution",
    default=None,
    help="Pin the quote resolution for --type cheapest-video (default: each model's lowest).",
)
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
    "--prefer",
    type=click.Choice(["cheapest"], case_sensitive=False),
    default=None,
    help="Rank matching models by strict price order instead of the catalog default; "
    "ties go to the default-trait model (skips beta and e2ee models unless asked for).",
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
    prompt_caching: bool,
    multiple_images: bool,
    e2ee: bool,
    exclude_reasoning: bool,
    exclude_uncensored: bool,
    custom_size: bool,
    quality: str | None,
    music_only: bool,
    force_instrumental: bool,
    video_type: str | None,
    input_mode: str | None,
    audio: bool,
    audio_configurable: bool,
    min_resolution: str | None,
    min_duration: str | None,
    duration: str | None,
    resolution: str | None,
    aspect_ratio: str | None,
    prefer: str | None,
    preferred_models: tuple[str, ...],
    exclude_models: tuple[str, ...],
    include_beta: bool,
    output_json: bool,
) -> None:
    """Resolve a model ID by type and capability requirements.

    Examples:

      venice-py models resolve --type chat
      venice-py models resolve --type chat --function-calling --vision
      venice-py models resolve --type embedding --prefer cheapest
      venice-py models resolve --type chat --function-calling --prefer cheapest
      venice-py models resolve --type video --audio --min-resolution 720p
      venice-py models resolve --type cheapest-video --video-type text-to-video
      venice-py models resolve --type cheapest-music --duration 10
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
            prompt_caching=prompt_caching,
            multiple_images=multiple_images,
            e2ee=e2ee,
            exclude_reasoning=exclude_reasoning,
            exclude_uncensored=exclude_uncensored,
            custom_size=custom_size,
            quality=quality,
            music_only=music_only,
            force_instrumental=force_instrumental,
            video_type=video_type,
            input_mode=input_mode.lower() if input_mode else None,
            audio=audio,
            audio_configurable=audio_configurable,
            min_resolution=min_resolution,
            min_duration=min_duration,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            prefer=prefer.lower() if prefer else None,
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
    duration: str | None,
    resolution: str | None,
    aspect_ratio: str | None,
    preferred_models: list[str] | None,
    exclude_models: list[str] | None,
    include_beta: bool,
    output_json: bool,
    prefer: str | None = None,
    prompt_caching: bool = False,
    multiple_images: bool = False,
    e2ee: bool = False,
    exclude_reasoning: bool = False,
    exclude_uncensored: bool = False,
    custom_size: bool = False,
    quality: str | None = None,
    music_only: bool = False,
    force_instrumental: bool = False,
    audio_configurable: bool = False,
    input_mode: str | None = None,
) -> None:
    from venice_ai import VeniceClient

    from ...config import get_client_kwargs

    exclude_beta = not include_beta

    async with VeniceClient(**get_client_kwargs()) as client:
        if model_type == "cheapest-video":
            result = await client.models.resolve_cheapest_video(
                duration=duration,
                video_type=video_type,  # type: ignore[arg-type]
                input_mode=input_mode,  # type: ignore[arg-type]
                resolution=resolution,
                audio=audio,
                aspect_ratio=aspect_ratio,
                require_audio_configurable=audio_configurable,
                exclude_models=exclude_models,
                exclude_beta=exclude_beta,
            )
            _render_cheapest_video(result, output_json=output_json)
            return

        if model_type == "cheapest-music":
            music_result = await client.models.resolve_cheapest_music(
                duration_seconds=_music_seconds(duration),
                require_force_instrumental=force_instrumental,
                exclude_models=exclude_models,
                exclude_beta=exclude_beta,
            )
            _render_cheapest_video(
                music_result, output_json=output_json, title="Cheapest music model"
            )
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
            "prefer": prefer,
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
                require_prompt_caching=prompt_caching,
                require_multiple_images=multiple_images,
                require_e2ee=e2ee,
                exclude_reasoning=exclude_reasoning,
                exclude_uncensored=exclude_uncensored,
            )
        elif model_type == "video":
            kwargs.update(
                video_type=video_type,
                input_mode=input_mode,
                require_audio=audio,
                min_resolution=min_resolution,
                min_duration=min_duration,
                require_audio_configurable=audio_configurable,
            )
        elif model_type == "image":
            kwargs.update(
                require_quality=quality,
                require_custom_size=custom_size,
                exclude_uncensored=exclude_uncensored,
            )
        elif model_type == "inpaint":
            kwargs.update(require_quality=quality)
        elif model_type == "music":
            kwargs.update(
                exclude_non_music=music_only,
                require_force_instrumental=force_instrumental,
                music_duration_seconds=_music_seconds(duration),
            )
        # embedding / tts / asr / decision: only general kwargs.

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


def _music_seconds(duration: str | None) -> int | None:
    """Parse ``--duration`` as whole seconds for the music types."""
    if duration is None:
        return None
    try:
        return int(duration)
    except ValueError:
        raise click.BadParameter(
            f"--duration must be a whole number of seconds for music, got {duration!r}"
        ) from None


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


def _render_cheapest_video(
    result: Any, *, output_json: bool, title: str = "Cheapest video model"
) -> None:
    """Render a CheapestVideoResult or CheapestMusicResult to stdout."""
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
            "request_params": getattr(result, "request_params", {}) or {},
            "skipped": getattr(result, "skipped", {}) or {},
        }

    if output_json:
        click.echo(json.dumps(payload, default=str))
        return

    from rich.panel import Panel
    from rich.table import Table

    quote = payload.get("quote_usd")
    quote_str = f"${quote:.6f}" if isinstance(quote, int | float) else str(quote)
    params = payload.get("request_params") or {}
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    body = f"[bold cyan]{payload.get('model')}[/bold cyan]\nQuote: [green]{quote_str}[/green]"
    if params_str:
        body += f"\nQuoted with: {params_str}"
    panel = Panel(body, title=title, border_style="green")
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
