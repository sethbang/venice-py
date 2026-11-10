"""
Interactive image generation wizard.
"""

import asyncio
from pathlib import Path
from typing import Any

import click
import questionary
from rich.panel import Panel
from rich.table import Table

from venice_ai import VeniceClient

from ...config import get_client_kwargs, load_config
from ...presets import save_preset
from ...utils import (
    console,
    print_error,
    print_info,
)


async def _interactive_image_generation(ctx: click.Context) -> None:
    """Interactive wizard for image generation with all parameter options"""

    console.print(
        Panel(
            "[bold cyan]🎨 Interactive Image Generation Wizard[/bold cyan]\n\n"
            "This wizard will guide you through all available image generation parameters.",
            border_style="cyan",
        )
    )

    # Get config
    config = ctx.obj.get("config", load_config())

    # Initialize Venice client to fetch available options
    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            # Fetch available models and styles
            print_info("Fetching available models and styles...")

            try:
                # Use the SDK's type parameter to get only image models
                models_response = await client.models.list(type="image")
                available_models = [m.id for m in models_response.data if m.id]

                if not available_models:
                    print_error("No image models found in API response.")
                    print_info("Please check your API connection and try again.")
                    raise SystemExit(1)
            except Exception as e:
                print_error(f"Could not fetch models from API: {e}")
                print_info("Please check your API key and connection.")
                raise SystemExit(1) from e

            try:
                styles_response = await client.image.list_styles()
                available_styles = (
                    styles_response.data if styles_response and styles_response.data else []
                )
            except Exception:
                available_styles = []

            # === REQUIRED PARAMETERS ===
            console.print("\n[bold]📝 Required Parameters[/bold]")

            # Prompt
            prompt = await asyncio.to_thread(
                lambda: questionary.text(
                    "Enter your image prompt (what you want to generate):", qmark="✨"
                ).ask()
            )

            if not prompt:
                print_error("Prompt is required!")
                raise SystemExit(1)

            # Model selection
            available = available_models or []
            default_model = config.get("defaults", {}).get("image_model") or (
                available[0] if available else None
            )
            if default_model not in available_models:
                default_model = available[0] if available else None

            model = await asyncio.to_thread(
                lambda: questionary.select(
                    "Select image model:",
                    choices=available_models,
                    default=default_model,
                    qmark="🤖",
                ).ask()
            )

            # === IMAGE DIMENSIONS ===
            console.print("\n[bold]📐 Image Dimensions[/bold]")

            size_presets = {
                "1024x1024 (Square - Default)": (1024, 1024),
                "1280x720 (16:9 Landscape)": (1280, 720),
                "720x1280 (9:16 Portrait)": (720, 1280),
                "1280x960 (4:3 Landscape)": (1280, 960),
                "960x1280 (3:4 Portrait)": (960, 1280),
                "512x512 (Small Square)": (512, 512),
                "Custom dimensions": None,
            }

            size_choice = await asyncio.to_thread(
                lambda: questionary.select(
                    "Select image size:",
                    choices=list(size_presets.keys()),
                    default="1024x1024 (Square - Default)",
                    qmark="📏",
                ).ask()
            )

            preset_dims = size_presets[size_choice]
            if preset_dims is None:
                # Custom dimensions
                width_str = await asyncio.to_thread(
                    lambda: questionary.text(
                        "Enter width (must be divisible by 8):", default="1024"
                    ).ask()
                )
                height_str = await asyncio.to_thread(
                    lambda: questionary.text(
                        "Enter height (must be divisible by 8):", default="1024"
                    ).ask()
                )
                width, height = int(width_str), int(height_str)
            else:
                width, height = preset_dims

            # Number of images
            num_images_str = await asyncio.to_thread(
                lambda: questionary.text(
                    "Number of images to generate (1-4):", default="1", qmark="🔢"
                ).ask()
            )
            num_images = int(num_images_str) if num_images_str else 1

            # === GENERATION PARAMETERS ===
            console.print("\n[bold]⚙️  Generation Parameters[/bold]")

            configure_advanced = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "Configure advanced generation parameters (steps, CFG scale, seed)?",
                    default=False,
                ).ask()
            )

            steps = None
            cfg_scale = None
            seed = None

            if configure_advanced:
                steps_str = await asyncio.to_thread(
                    lambda: questionary.text(
                        "Inference steps (1-50, higher=better quality, leave empty for default):",
                        default="",
                    ).ask()
                )
                if steps_str:
                    steps = int(steps_str)

                cfg_str = await asyncio.to_thread(
                    lambda: questionary.text(
                        "CFG scale (0-20, higher=stricter prompt adherence, leave empty for default):",
                        default="",
                    ).ask()
                )
                if cfg_str:
                    cfg_scale = float(cfg_str)

                use_seed = await asyncio.to_thread(
                    lambda: questionary.confirm(
                        "Use a specific seed for reproducibility?", default=False
                    ).ask()
                )

                if use_seed:
                    seed_str = await asyncio.to_thread(
                        lambda: questionary.text(
                            "Enter seed (integer, or leave empty for random):",
                            default="",
                        ).ask()
                    )
                    if seed_str:
                        seed = int(seed_str)

            # === STYLE PARAMETERS ===
            console.print("\n[bold]🎨 Style Parameters[/bold]")

            configure_style = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "Configure style parameters (style preset, LoRA)?",
                    default=False,
                ).ask()
            )

            style_preset = None
            lora_strength = None

            if configure_style:
                # Style preset
                if available_styles:
                    use_style = await asyncio.to_thread(
                        lambda: questionary.confirm(
                            f"Apply a style preset? ({len(available_styles)} available)",
                            default=False,
                        ).ask()
                    )

                    if use_style:
                        style_choices = ["None"] + available_styles
                        style_preset = await asyncio.to_thread(
                            lambda: questionary.select(
                                "Select style preset:",
                                choices=style_choices,
                                qmark="🎭",
                            ).ask()
                        )
                        if style_preset == "None":
                            style_preset = None

                # LoRA strength
                use_lora = await asyncio.to_thread(
                    lambda: questionary.confirm(
                        "Configure LoRA strength (if model uses LoRAs)?", default=False
                    ).ask()
                )

                if use_lora:
                    lora_str = await asyncio.to_thread(
                        lambda: questionary.text("LoRA strength (0-100):", default="50").ask()
                    )
                    if lora_str:
                        lora_strength = int(lora_str)

            # === OUTPUT PARAMETERS ===
            console.print("\n[bold]💾 Output Parameters[/bold]")

            # Format
            format_choice = await asyncio.to_thread(
                lambda: questionary.select(
                    "Output format:",
                    choices=[
                        "webp (Recommended - Best compression)",
                        "png (Highest quality)",
                        "jpeg (Most compatible)",
                    ],
                    default="webp (Recommended - Best compression)",
                    qmark="📄",
                ).ask()
            )
            output_format = format_choice.split()[0]  # Extract format name

            # Safe mode
            safe_mode = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "Enable safe mode (blur adult content)?", default=True
                ).ask()
            )

            # Watermark
            hide_watermark = await asyncio.to_thread(
                lambda: questionary.confirm("Hide Venice watermark?", default=False).ask()
            )

            # EXIF metadata
            embed_exif = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "Embed generation metadata in EXIF?", default=False
                ).ask()
            )

            # Return binary
            return_binary = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "Use binary mode (faster, but only for single images)?",
                    default=False,
                ).ask()
            )

            # === OUTPUT LOCATION ===
            save_dir = Path(
                config.get("output", {}).get("images_dir", str(Path.home() / "Pictures" / "venice"))
            )

            custom_location = await asyncio.to_thread(
                lambda: questionary.confirm(
                    f"Save to custom location? (default: {save_dir})", default=False
                ).ask()
            )

            if custom_location:
                save_dir_str = await asyncio.to_thread(
                    lambda: questionary.path("Enter save directory:", default=str(save_dir)).ask()
                )
                if save_dir_str:
                    save_dir = Path(save_dir_str)

            # === SUMMARY ===
            console.print("\n[bold cyan]📋 Generation Summary[/bold cyan]")

            summary_table = Table(show_header=False, box=None, padding=(0, 2))
            summary_table.add_column("Parameter", style="cyan")
            summary_table.add_column("Value", style="white")

            summary_table.add_row("Prompt", prompt[:50] + "..." if len(prompt) > 50 else prompt)
            summary_table.add_row("Model", model)
            summary_table.add_row("Size", f"{width}x{height}")
            summary_table.add_row("Images", str(num_images))
            if steps:
                summary_table.add_row("Steps", str(steps))
            if cfg_scale:
                summary_table.add_row("CFG Scale", str(cfg_scale))
            if seed:
                summary_table.add_row("Seed", str(seed))
            if style_preset:
                summary_table.add_row("Style", style_preset)
            if lora_strength is not None:
                summary_table.add_row("LoRA Strength", str(lora_strength))
            summary_table.add_row("Format", output_format)
            summary_table.add_row("Safe Mode", "Yes" if safe_mode else "No")
            if hide_watermark:
                summary_table.add_row("Hide Watermark", "Yes")
            if embed_exif:
                summary_table.add_row("EXIF Metadata", "Yes")
            summary_table.add_row("Save Directory", str(save_dir))

            console.print(summary_table)

            # Confirm and generate
            proceed = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "\nProceed with image generation?", default=True, qmark="🚀"
                ).ask()
            )

            if not proceed:
                print_info("Image generation cancelled.")
                return

            # Ask to save as preset
            save_as_preset = await asyncio.to_thread(
                lambda: questionary.confirm(
                    "Save this configuration as a preset?", default=False
                ).ask()
            )

            if save_as_preset:
                preset_name = await asyncio.to_thread(
                    lambda: questionary.text("Preset name:", qmark="💾").ask()
                )

                if preset_name:
                    preset_config: dict[str, Any] = {}
                    if steps:
                        preset_config["steps"] = steps
                    if cfg_scale is not None:
                        preset_config["cfg_scale"] = float(cfg_scale)
                    if seed:
                        preset_config["seed"] = seed
                    if style_preset:
                        preset_config["style_preset"] = style_preset
                    if lora_strength is not None:
                        preset_config["lora_strength"] = lora_strength
                    preset_config["format"] = output_format
                    preset_config["safe_mode"] = safe_mode
                    if hide_watermark:
                        preset_config["hide_watermark"] = hide_watermark
                    if embed_exif:
                        preset_config["embed_exif"] = embed_exif

                    save_preset(preset_name, preset_config)

            # Call the actual generation function
            from .generate import _generate_image_async

            await _generate_image_async(
                ctx=ctx,
                prompt=prompt,
                model=model,
                size=f"{width}x{height}",
                num_images=num_images,
                output=None,
                save_dir=str(save_dir),
                show_timing=True,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                style_preset=style_preset,
                lora_strength=lora_strength,
                format=output_format,
                safe_mode=safe_mode,
                hide_watermark=hide_watermark,
                embed_exif=embed_exif,
                return_binary=return_binary,
            )

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e
