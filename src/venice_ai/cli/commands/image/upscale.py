"""
Image upscaling command.
"""

import asyncio
from pathlib import Path
from typing import Any

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError

from ...config import get_client_kwargs
from ...utils import (
    console,
    is_plain_mode,
    open_file,
    print_error,
    print_info,
    print_success,
)


@click.command(name="upscale")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--scale", type=float, default=None, help="Scale factor (e.g. 2.0 for 2x)")
@click.option(
    "--creativity",
    type=float,
    default=None,
    help="Detail/texture the upscaler adds; server clamps to 0-0.02 (default 0.01)",
)
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--save-dir",
    default=".",
    help="Directory to save the result (default: current dir)",
)
@click.option("--open", "open_image", is_flag=True, default=False, help="Open image after saving")
@click.pass_context
def upscale_image(
    ctx: click.Context,
    input_file: str,
    scale: float | None,
    creativity: float | None,
    output: str | None,
    save_dir: str,
    open_image: bool,
) -> None:
    """Upscale an image using AI

    Examples:

        venice-py image upscale photo.jpg --scale 2

        venice-py image upscale photo.png --scale 4 --creativity 0.02 --save-dir ./upscaled
    """
    asyncio.run(
        _upscale_async(
            ctx,
            input_file,
            scale,
            creativity,
            output,
            save_dir,
            open_image,
        )
    )


async def _upscale_async(
    ctx: click.Context,
    input_file: str,
    scale: float | None,
    creativity: float | None,
    output: str | None,
    save_dir: str,
    open_image: bool,
) -> None:
    """Async implementation of image upscaling"""
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    input_path = Path(input_file)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Build kwargs
    upscale_kwargs: dict[str, Any] = {"image": str(input_path)}
    if scale is not None:
        upscale_kwargs["scale"] = scale
    if creativity is not None:
        upscale_kwargs["creativity"] = creativity

    if plain:
        click.echo(f"Upscaling: {input_path}")
    else:
        print_info(f"Upscaling: {input_path}")
        if scale:
            console.print(f"[cyan]Scale:[/cyan] {scale}x")

    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            if plain:
                click.echo("Sending to API...")
                try:
                    result_bytes = await client.image.upscale(**upscale_kwargs)
                    click.echo("Upscaling complete.")
                except Exception as e:
                    click.echo(f"Upscaling failed: {e}")
                    raise
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Upscaling image...", total=None)
                    try:
                        result_bytes = await client.image.upscale(**upscale_kwargs)
                        progress.update(task, description="[green]Upscaling complete!")
                    except Exception as e:
                        progress.update(task, description=f"[red]Failed: {e}")
                        raise

            # Determine output path
            if output:
                out_file = Path(output)
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = input_path.suffix or ".png"
                out_file = save_path / f"upscaled_{timestamp}{suffix}"

            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(result_bytes)

            if plain:
                click.echo(f"Saved: {out_file}")
                click.echo(f"  Size: {len(result_bytes):,} bytes")
            else:
                print_success(f"Saved: {out_file}")
                console.print(f"  Size: {len(result_bytes):,} bytes")

            if open_image:
                open_file(str(out_file))

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e
