"""
Image generation and batch generation commands.
"""

import asyncio
import base64
from pathlib import Path
from typing import Any, cast

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import ImageGenerationResponse

from ..._model_defaults import resolve_default_model
from ...config import get_client_kwargs, load_config
from ...utils import (
    ParameterValidator,
    console,
    is_plain_mode,
    open_file,
    print_error,
    print_info,
    print_success,
)
from ._helpers import _load_preset_config, validate_size


@click.command(name="generate")
@click.argument("prompt", required=False)
@click.option("--model", "-m", help="Image model to use", default=None)
@click.option(
    "--size",
    "-s",
    default="1024x1024",
    callback=validate_size,
    help="Image size in WxH format (e.g., 1024x1024, 1920x1080)",
)
@click.option("--num-images", "-n", type=int, help="Number of images to generate", default=1)
@click.option("--output", "-o", help="Output file path (without extension)", default=None)
@click.option("--save-dir", help="Directory to save images", default=None)
@click.option("--show-timing/--no-show-timing", default=True, help="Show generation timing")
# GENERATION CONTROL PARAMETERS
@click.option("--steps", type=int, help="Inference steps (1-50, higher=better quality)")
@click.option("--cfg-scale", type=float, help="CFG scale (0-20, higher=stricter prompt adherence)")
@click.option("--seed", type=int, help="Random seed for reproducibility (0=random)")
# STYLE & ARTISTIC CONTROL PARAMETERS
@click.option("--style-preset", "-sp", help="Style preset to apply")
@click.option("--lora-strength", type=int, help="LoRA strength 0-100 (if model uses LoRAs)")
# OUTPUT FORMAT & QUALITY PARAMETERS
@click.option("--format", type=click.Choice(["jpeg", "png", "webp"]), help="Image format")
@click.option("--aspect-ratio", help="Aspect ratio for the output (e.g. 1:1, 16:9)")
@click.option(
    "--resolution",
    type=click.Choice(["1K", "2K", "4K"]),
    help="Resolution tier for the output image",
)
@click.option(
    "--quality",
    type=click.Choice(["low", "medium", "high"]),
    help="Output quality for quality-aware models",
)
@click.option(
    "--enable-web-search/--no-enable-web-search",
    default=None,
    help="Allow the image model to incorporate web search results",
)
@click.option("--safe-mode/--no-safe-mode", default=None, help="Blur adult content")
@click.option("--hide-watermark", is_flag=True, help="Hide Venice watermark")
@click.option("--embed-exif", is_flag=True, help="Embed generation metadata in EXIF")
@click.option("--return-binary", is_flag=True, help="Return raw binary data (faster)")
# AUTO-OPEN
@click.option(
    "--open",
    "open_image",
    is_flag=True,
    default=False,
    help="Open generated image after saving",
)
# PRESET SUPPORT
@click.option("--preset", "-p", help="Load configuration from preset (built-in or custom)")
# INTERACTIVE MODE
@click.option("--interactive", "-i", is_flag=True, help="Interactive wizard mode")
@click.pass_context
def generate_image(
    ctx: click.Context,
    prompt: str | None,
    model: str | None,
    size: str,
    num_images: int,
    output: str | None,
    save_dir: str | None,
    show_timing: bool,
    # New parameters
    steps: int | None,
    cfg_scale: float | None,
    seed: int | None,
    style_preset: str | None,
    lora_strength: int | None,
    format: str | None,
    aspect_ratio: str | None,
    resolution: str | None,
    quality: str | None,
    enable_web_search: bool | None,
    safe_mode: bool | None,
    hide_watermark: bool,
    embed_exif: bool,
    return_binary: bool,
    open_image: bool,
    preset: str | None,
    interactive: bool,
) -> None:
    """Generate an image from a text prompt with full parameter control

    Use --interactive flag to launch a wizard that guides you through all parameters.
    Use --preset to load a saved or built-in preset configuration.

    Examples:

        # Basic usage
        venice-py image generate "sunset over mountains"

        # With advanced parameters
        venice-py image generate "cyberpunk city" --steps 30 --cfg-scale 8.5 --style-preset Cinematic

        # Using a preset
        venice-py image generate "portrait photo" --preset photorealistic

        # Interactive mode
        venice-py image generate --interactive
    """
    if interactive:
        # Lazy import to avoid circular dependency (wizard imports _generate_image_async)
        from .wizard import _interactive_image_generation

        asyncio.run(_interactive_image_generation(ctx))
    else:
        if not prompt:
            print_error("Prompt is required when not using --interactive mode")
            ctx.exit(1)

        # Apply preset if specified
        if preset:
            preset_config = _load_preset_config(preset)
            if preset_config:
                # Apply preset values (CLI args override preset values)
                if steps is None and "steps" in preset_config:
                    steps = preset_config["steps"]
                if cfg_scale is None and "cfg_scale" in preset_config:
                    cfg_scale = preset_config["cfg_scale"]
                if seed is None and "seed" in preset_config:
                    seed = preset_config["seed"]
                if style_preset is None and "style_preset" in preset_config:
                    style_preset = preset_config["style_preset"]
                if lora_strength is None and "lora_strength" in preset_config:
                    lora_strength = preset_config["lora_strength"]
                if format is None and "format" in preset_config:
                    format = preset_config["format"]
                if safe_mode is None and "safe_mode" in preset_config:
                    safe_mode = preset_config["safe_mode"]
                if not hide_watermark and preset_config.get("hide_watermark"):
                    hide_watermark = preset_config["hide_watermark"]
                if not embed_exif and preset_config.get("embed_exif"):
                    embed_exif = preset_config["embed_exif"]

                print_info(f"Applied preset: {preset}")
            else:
                print_error(f"Preset not found: {preset}")
                raise SystemExit(1)

        # Use provided parameters
        asyncio.run(
            _generate_image_async(
                ctx,
                prompt,
                model,
                size,
                num_images,
                output,
                save_dir,
                show_timing,
                steps,
                cfg_scale,
                seed,
                style_preset,
                lora_strength,
                format,
                safe_mode,
                hide_watermark,
                embed_exif,
                return_binary,
                open_image,
                aspect_ratio,
                resolution,
                quality,
                enable_web_search,
            )
        )


async def _generate_image_async(
    ctx: click.Context,
    prompt: str,
    model: str | None,
    size: str,
    num_images: int,
    output: str | None,
    save_dir: str | None,
    show_timing: bool,
    # New parameters
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    style_preset: str | None = None,
    lora_strength: int | None = None,
    format: str | None = None,
    safe_mode: bool | None = None,
    hide_watermark: bool = False,
    embed_exif: bool = False,
    return_binary: bool = False,
    open_image: bool = False,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    quality: str | None = None,
    enable_web_search: bool | None = None,
) -> None:
    """Async implementation of image generation with full parameter support"""

    # Get config
    config = ctx.obj.get("config", load_config())
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    # Parse size
    width, height = map(int, size.split("x"))

    # Determine save directory
    save_path = Path(save_dir) if save_dir else Path(config["output"]["images_dir"])

    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine output format for filename
    output_format = format or "png"

    # Initialize Venice client
    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            # Resolve model at runtime (no hardcoded fallback)
            model = await resolve_default_model(client, config, "image", explicit=model)
            validator = ParameterValidator(client)
            is_valid, error_message = await validator.validate_image_parameters(
                model=model,
                width=width,
                height=height,
                prompt=prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                lora_strength=lora_strength,
                num_images=num_images,
            )

            if not is_valid:
                print_error(f"Parameter validation failed: {error_message}")
                raise SystemExit(1)

            # Show generation info
            if plain:
                click.echo(f"Generating {num_images} image(s) with {model}")
                click.echo(f"Prompt: {prompt}")
                click.echo(f"Size: {size}")
                if steps:
                    click.echo(f"Steps: {steps}")
                if cfg_scale:
                    click.echo(f"CFG Scale: {cfg_scale}")
                if style_preset:
                    click.echo(f"Style: {style_preset}")
            else:
                print_info(f"Generating {num_images} image(s) with {model}")
                console.print(f"[cyan]Prompt:[/cyan] {prompt}")
                console.print(f"[cyan]Size:[/cyan] {size}")
                if steps:
                    console.print(f"[cyan]Steps:[/cyan] {steps}")
                if cfg_scale:
                    console.print(f"[cyan]CFG Scale:[/cyan] {cfg_scale}")
                if style_preset:
                    console.print(f"[cyan]Style:[/cyan] {style_preset}")

            # Build kwargs for SDK call, only including non-None values
            generate_kwargs: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_images": num_images,
            }

            # Add optional parameters if provided
            if steps is not None:
                generate_kwargs["steps"] = steps
            if cfg_scale is not None:
                generate_kwargs["cfg_scale"] = cfg_scale
            if seed is not None:
                generate_kwargs["seed"] = seed
            if style_preset:
                generate_kwargs["style_preset"] = style_preset
            if lora_strength is not None:
                generate_kwargs["lora_strength"] = lora_strength
            if format:
                generate_kwargs["format"] = format
            if aspect_ratio:
                generate_kwargs["aspect_ratio"] = aspect_ratio
            if resolution:
                generate_kwargs["resolution"] = resolution
            if quality:
                generate_kwargs["quality"] = quality
            if enable_web_search is not None:
                generate_kwargs["enable_web_search"] = enable_web_search
            if safe_mode is not None:
                generate_kwargs["safe_mode"] = safe_mode
            if hide_watermark:
                generate_kwargs["hide_watermark"] = hide_watermark
            if embed_exif:
                generate_kwargs["embed_exif_metadata"] = embed_exif
            if return_binary:
                generate_kwargs["return_binary"] = return_binary
            else:
                generate_kwargs["return_binary"] = False

            # Generate images (with or without progress indicator)
            if plain:
                click.echo("Generating image...")
                try:
                    response = cast(
                        ImageGenerationResponse,
                        await client.image.create(**generate_kwargs),
                    )
                    click.echo("Generation complete.")
                except Exception as e:
                    click.echo(f"Generation failed: {e}")
                    raise
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Generating image...", total=None)

                    try:
                        # Use **kwargs to pass optional parameters
                        response = cast(
                            ImageGenerationResponse,
                            await client.image.create(**generate_kwargs),
                        )

                        progress.update(task, description="[green]Generation complete!")

                    except Exception as e:
                        progress.update(task, description=f"[red]Generation failed: {e}")
                        raise

            # Process and save generated images
            if response and response.images:
                if plain:
                    click.echo(f"Generated {len(response.images)} image(s)")
                else:
                    print_success(f"Generated {len(response.images)} image(s)")

                for i, image_data in enumerate(response.images):
                    # Decode base64 image data
                    image_bytes = base64.b64decode(image_data)

                    # Determine filename with correct extension
                    if output and num_images == 1:
                        filename = f"{output}.{output_format}"
                    elif output:
                        filename = f"{output}_{i + 1}.{output_format}"
                    else:
                        # Generate timestamp-based filename
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"venice_image_{timestamp}_{i + 1}.{output_format}"

                    # Save the image
                    file_path = save_path / filename
                    with open(file_path, "wb") as f:
                        f.write(image_bytes)

                    if plain:
                        click.echo(f"Saved: {file_path}")
                        click.echo(f"  Size: {len(image_bytes):,} bytes")
                    else:
                        console.print(f"[green]✓[/green] Saved: {file_path}")
                        console.print(f"  Size: {len(image_bytes):,} bytes")

                    # Auto-open the image if requested
                    if open_image:
                        open_file(str(file_path))

                # Show timing information if available
                if show_timing and response.timing:
                    if plain:
                        click.echo(f"Generation time: {response.timing.inferenceDuration}ms")
                    else:
                        console.print(
                            f"\n[dim]Generation time: {response.timing.inferenceDuration}ms[/dim]"
                        )
            else:
                print_error("No images were generated")
                raise SystemExit(1)

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command(name="batch")
@click.option(
    "--prompts-file",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="File containing prompts (one per line)",
)
@click.option("--model", "-m", help="Image model to use", default=None)
@click.option(
    "--size",
    "-s",
    default="1024x1024",
    callback=validate_size,
    help="Image size in WxH format (e.g., 1024x1024, 1920x1080)",
)
@click.option("--save-dir", help="Directory to save images", default=None)
# GENERATION CONTROL PARAMETERS
@click.option("--steps", type=int, help="Inference steps (1-50, higher=better quality)")
@click.option("--cfg-scale", type=float, help="CFG scale (0-20, higher=stricter prompt adherence)")
@click.option("--seed", type=int, help="Random seed for reproducibility (0=random)")
# STYLE & ARTISTIC CONTROL PARAMETERS
@click.option("--style-preset", "-sp", help="Style preset to apply")
# OUTPUT FORMAT & QUALITY PARAMETERS
@click.option("--format", type=click.Choice(["jpeg", "png", "webp"]), help="Image format")
@click.option("--safe-mode/--no-safe-mode", default=None, help="Blur adult content")
@click.option("--hide-watermark", is_flag=True, help="Hide Venice watermark")
@click.option("--embed-exif", is_flag=True, help="Embed generation metadata in EXIF")
@click.pass_context
def batch_generate(
    ctx: click.Context,
    prompts_file: str,
    model: str | None,
    size: str,
    save_dir: str | None,
    steps: int | None,
    cfg_scale: float | None,
    seed: int | None,
    style_preset: str | None,
    format: str | None,
    safe_mode: bool | None,
    hide_watermark: bool,
    embed_exif: bool,
) -> None:
    """Generate multiple images from a file of prompts"""
    asyncio.run(
        _batch_generate_async(
            ctx,
            prompts_file,
            model,
            size,
            save_dir,
            steps,
            cfg_scale,
            seed,
            style_preset,
            format,
            safe_mode,
            hide_watermark,
            embed_exif,
        )
    )


async def _batch_generate_async(
    ctx: click.Context,
    prompts_file: str,
    model: str | None,
    size: str,
    save_dir: str | None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    style_preset: str | None = None,
    format: str | None = None,
    safe_mode: bool | None = None,
    hide_watermark: bool = False,
    embed_exif: bool = False,
) -> None:
    """Async implementation of batch image generation"""

    # Read prompts from file
    prompts_path = Path(prompts_file)
    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print_error("No prompts found in file")
        raise SystemExit(1)

    print_info(f"Found {len(prompts)} prompt(s) to process")

    # Get config
    config = ctx.obj.get("config", load_config())
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    # Parse size
    width, height = map(int, size.split("x"))

    # Determine save directory
    save_path = Path(save_dir) if save_dir else Path(config["output"]["images_dir"])

    save_path.mkdir(parents=True, exist_ok=True)

    # Determine output format for filenames
    output_format = format or "png"

    # Initialize Venice client and process prompts
    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            # Resolve model at runtime (no hardcoded fallback)
            model = await resolve_default_model(client, config, "image", explicit=model)

            # Build base kwargs for SDK calls (prompt added per iteration)
            base_kwargs: dict[str, Any] = {
                "model": model,
                "width": width,
                "height": height,
                "num_images": 1,
                "return_binary": False,
            }
            if steps is not None:
                base_kwargs["steps"] = steps
            if cfg_scale is not None:
                base_kwargs["cfg_scale"] = cfg_scale
            if seed is not None:
                base_kwargs["seed"] = seed
            if style_preset:
                base_kwargs["style_preset"] = style_preset
            if format:
                base_kwargs["format"] = format
            if safe_mode is not None:
                base_kwargs["safe_mode"] = safe_mode
            if hide_watermark:
                base_kwargs["hide_watermark"] = hide_watermark
            if embed_exif:
                base_kwargs["embed_exif_metadata"] = embed_exif

            successful = 0
            failed = 0

            if plain:
                # Plain mode: simple text output without Rich progress
                for idx, prompt in enumerate(prompts, 1):
                    click.echo(f"[{idx}/{len(prompts)}] Generating: {prompt[:50]}...")

                    try:
                        generate_kwargs = {**base_kwargs, "prompt": prompt}
                        response = cast(
                            ImageGenerationResponse,
                            await client.image.create(**generate_kwargs),
                        )

                        if response and response.images:
                            image_bytes = base64.b64decode(response.images[0])
                            from datetime import datetime

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"batch_{idx}_{timestamp}.{output_format}"
                            file_path = save_path / filename

                            with open(file_path, "wb") as f:
                                f.write(image_bytes)

                            click.echo(f"  Saved: {file_path}")
                            successful += 1
                        else:
                            click.echo("  Warning: No image generated")
                            failed += 1

                    except Exception as e:
                        click.echo(f"  Failed: {str(e)[:50]}")
                        failed += 1

                    await asyncio.sleep(0.5)
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    for idx, prompt in enumerate(prompts, 1):
                        task_desc = f"[{idx}/{len(prompts)}] Generating: {prompt[:50]}..."
                        task = progress.add_task(task_desc, total=None)

                        try:
                            generate_kwargs = {**base_kwargs, "prompt": prompt}
                            response = cast(
                                ImageGenerationResponse,
                                await client.image.create(**generate_kwargs),
                            )

                            if response and response.images:
                                # Save the image
                                image_bytes = base64.b64decode(response.images[0])
                                from datetime import datetime

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"batch_{idx}_{timestamp}.{output_format}"
                                file_path = save_path / filename

                                with open(file_path, "wb") as f:
                                    f.write(image_bytes)

                                progress.update(
                                    task,
                                    description=f"[green]✓[/green] {prompt[:50]}... → {filename}",
                                )
                                successful += 1
                            else:
                                progress.update(
                                    task,
                                    description=f"[yellow]⚠[/yellow] {prompt[:50]}... - No image",
                                )
                                failed += 1

                        except Exception as e:
                            progress.update(
                                task,
                                description=f"[red]✗[/red] {prompt[:50]}... - {str(e)[:30]}",
                            )
                            failed += 1

                        # Small delay between requests to be respectful
                        await asyncio.sleep(0.5)

            # Summary
            if plain:
                click.echo("\nBatch Generation Complete")
                click.echo(f"Successful: {successful}")
                if failed > 0:
                    click.echo(f"Failed: {failed}")
                click.echo(f"Images saved to: {save_path}")
            else:
                console.print("\n[bold]Batch Generation Complete[/bold]")
                console.print(f"✅ Successful: {successful}")
                if failed > 0:
                    console.print(f"❌ Failed: {failed}")
                console.print(f"📁 Images saved to: {save_path}")

            if failed > 0:
                raise SystemExit(1)

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e
