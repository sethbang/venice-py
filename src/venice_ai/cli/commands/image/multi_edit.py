"""
Multi-image edit command (POST /image/multi-edit).
"""

import asyncio
from collections.abc import Sequence
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


@click.command(name="multi-edit")
@click.option("--prompt", "-p", required=True, help="Edit instruction describing the changes")
@click.option(
    "--image",
    "-i",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Input image, repeatable. The first is the base; the rest are layered in order. "
    "The limit is per model.",
)
@click.option(
    "--image-2",
    type=click.Path(exists=True),
    default=None,
    help="Layer image, sent after every --image",
)
@click.option(
    "--image-3",
    type=click.Path(exists=True),
    default=None,
    help="Layer image, sent after every --image and --image-2",
)
@click.option("--model", "-m", default=None, help="Edit model to use")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--save-dir",
    default=".",
    help="Directory to save the result (default: current dir)",
)
@click.option("--aspect-ratio", default=None, help="Aspect ratio for the output (e.g. 1:1, 16:9)")
@click.option(
    "--resolution",
    type=click.Choice(["1K", "2K", "4K"]),
    default=None,
    help="Resolution tier for the output image",
)
@click.option(
    "--output-format",
    type=click.Choice(["jpeg", "png", "webp"]),
    default=None,
    help="Output format for the edited image",
)
@click.option(
    "--quality",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Output quality for quality-aware models",
)
@click.option("--open", "open_image", is_flag=True, default=False, help="Open image after saving")
@click.pass_context
def multi_edit_image(
    ctx: click.Context,
    prompt: str,
    image: tuple[str, ...],
    image_2: str | None,
    image_3: str | None,
    model: str | None,
    output: str | None,
    save_dir: str,
    aspect_ratio: str | None,
    resolution: str | None,
    output_format: str | None,
    quality: str | None,
    open_image: bool,
) -> None:
    """Edit an image using several layered inputs

    The first image is the base; additional images are layered on top. Repeat
    --image for each input. The limit is per model, and the API rejects too many.

    Examples:

        venice-py image multi-edit --prompt "combine these" --image base.png

        venice-py image multi-edit -p "blend" -i base.png --image-2 overlay.png

        venice-py image multi-edit -p "blend" -i a.png -i b.png -i c.png -i d.png
    """
    asyncio.run(
        _multi_edit_async(
            ctx=ctx,
            prompt=prompt,
            image=image,
            image_2=image_2,
            image_3=image_3,
            model=model,
            output=output,
            save_dir=save_dir,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            quality=quality,
            open_image=open_image,
        )
    )


async def _multi_edit_async(
    ctx: click.Context,
    prompt: str,
    image: str | Sequence[str],
    image_2: str | None = None,
    image_3: str | None = None,
    model: str | None = None,
    output: str | None = None,
    save_dir: str = ".",
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    output_format: str | None = None,
    quality: str | None = None,
    open_image: bool = False,
) -> None:
    """Async implementation of multi-image editing"""
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    inputs = [image] if isinstance(image, str) else list(image)
    inputs.extend(str(layer) for layer in (image_2, image_3) if layer is not None)
    base_path = Path(inputs[0])
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if plain:
        click.echo(f"Editing: {base_path}")
        click.echo(f"Prompt: {prompt}")
    else:
        print_info(f"Editing: {base_path}")
        console.print(f"[cyan]Prompt:[/cyan] {prompt}")
        if model:
            console.print(f"[cyan]Model:[/cyan] {model}")

    # Build kwargs, only including provided values
    edit_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "images": inputs,
    }
    if model is not None:
        edit_kwargs["model"] = model
    if aspect_ratio:
        edit_kwargs["aspect_ratio"] = aspect_ratio
    if resolution:
        edit_kwargs["resolution"] = resolution
    if output_format:
        edit_kwargs["output_format"] = output_format
    if quality:
        edit_kwargs["quality"] = quality

    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            if plain:
                click.echo("Sending to API...")
                try:
                    result_bytes = await client.image.multi_edit(**edit_kwargs)
                    click.echo("Edit complete.")
                except Exception as e:
                    click.echo(f"Edit failed: {e}")
                    raise
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Editing images...", total=None)
                    try:
                        result_bytes = await client.image.multi_edit(**edit_kwargs)
                        progress.update(task, description="[green]Edit complete!")
                    except Exception as e:
                        progress.update(task, description=f"[red]Failed: {e}")
                        raise

            # Determine output extension
            ext = f".{output_format}" if output_format else (base_path.suffix or ".png")

            if output:
                out_file = Path(output)
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = save_path / f"multi_edited_{timestamp}{ext}"

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
