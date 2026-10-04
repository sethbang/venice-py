#!/usr/bin/env python3
"""
Venice AI SDK - Text-to-Image Generation
========================================

This example demonstrates how to generate images from text prompts using the Venice AI SDK.
Learn how to create compelling visual content with AI image generation models.

The model is the cheapest one sized by pixel dimensions,
``resolve_image(prefer="cheapest", require_custom_size=True)``: models that list
aspect ratios size by ``aspect_ratio`` and ignore ``width`` / ``height``.

Every image is generated with ``hide_watermark=True`` and saved under
``examples/results/`` with a ``t2i_`` prefix; the pixel size of each returned
image is read back from its header and checked against the request, because
the catalog cannot tell whether a model returns exactly the size asked for.
Each run makes seven paid generations.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError
from venice_ai.types.api import ImageGenerationResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import SKIPPED, image_dimensions  # noqa: E402

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


async def _generate(
    client: VeniceClient,
    model: str,
    prompt: str,
    stem: str,
    written: list[Path],
    *,
    width: int = 512,
    height: int = 512,
) -> bool:
    """Generate one image, save it, and check its size matches the request."""
    try:
        response: ImageGenerationResponse = await client.image.create(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            num_images=1,
            hide_watermark=True,
            return_binary=False,  # Get structured response
        )
    except VeniceError as e:
        print(f"❌ Generation failed: {e}")
        return False

    data = response.bytes(0)
    actual_w, actual_h = image_dimensions(data)
    # save() sniffs the format from the image bytes when the path has no suffix.
    path = response.save(RESULTS_DIR / stem, overwrite=True)
    written.append(path)
    print(f"✅ Saved: {path} ({actual_w}x{actual_h}, {len(data)} bytes)")
    if response.timing:
        print(f"⏱️ Generation time: {response.timing.inferenceDuration}ms")
    if (actual_w, actual_h) != (width, height):
        print(f"❌ Requested {width}x{height} but received {actual_w}x{actual_h}")
        return False
    return True


async def basic_image_generation(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """Generate a simple image from text."""
    print("🎨 Basic Image Generation")
    print("-" * 30)
    return await _generate(
        client,
        model,
        "A serene mountain landscape at sunset with a crystal clear lake",
        "t2i_landscape",
        written,
    )


async def batch_image_generation(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """Generate several images from different prompts.

    Returns ``True`` only if every image generated.
    """
    print("\n🖼️ Batch Image Generation")
    print("-" * 30)

    prompts = [
        "A futuristic city with flying cars",
        "A peaceful forest with magical creatures",
    ]

    ok = True
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎨 Generating image {i}: {prompt[:50]}...")
        ok &= await _generate(client, model, prompt, f"t2i_batch_{i}", written)
    return ok


async def style_variations(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """Generate the same subject with different style modifiers in the prompt.

    How closely each style is followed depends on the model; see
    ``style_variants.py`` for a larger sweep.
    """
    print("\n🎭 Style Variations")
    print("-" * 30)

    base_prompt = "A majestic lion in its natural habitat"
    styles = ["photorealistic", "cartoon animation"]

    ok = True
    for style in styles:
        print(f"\n🖌️ Generating {style} version...")
        ok &= await _generate(
            client,
            model,
            f"{base_prompt}, {style} style",
            f"t2i_lion_{style.replace(' ', '_')}",
            written,
        )
    return ok


async def parameter_exploration(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """Generate the same prompt in landscape and portrait sizes.

    The basic demo above already covers a square 512x512 image.
    """
    print("\n⚙️ Parameter Exploration")
    print("-" * 30)

    base_prompt = "A beautiful garden with colorful flowers"
    sizes = [
        (768, 512, "Landscape format"),
        (512, 768, "Portrait format"),
    ]

    ok = True
    for width, height, description in sizes:
        print(f"\n📐 Generating {description} ({width}x{height})...")
        ok &= await _generate(
            client,
            model,
            base_prompt,
            f"t2i_garden_{width}x{height}",
            written,
            width=width,
            height=height,
        )
    return ok


async def main() -> int:
    """Run all image generation examples.

    Returns ``0`` only if every demo succeeded, ``1`` if any failed, and ``77``
    if no catalog image model is sized by width/height.
    """
    print("🚀 Venice AI Image Generation Examples")
    print("=" * 50)

    written: list[Path] = []
    async with VeniceClient() as client:
        # Resolve the model once; every demo below reuses it.
        try:
            image_model = await client.models.resolve_image(
                prefer="cheapest", require_custom_size=True
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        print(f"📍 Using image model: {image_model}\n")

        results: list[tuple[str, bool]] = [
            ("basic_image_generation", await basic_image_generation(client, image_model, written)),
            ("batch_image_generation", await batch_image_generation(client, image_model, written)),
            ("style_variations", await style_variations(client, image_model, written)),
            ("parameter_exploration", await parameter_exploration(client, image_model, written)),
        ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Image generation examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Basic text-to-image generation")
        print("   - Generating several prompts in sequence")
        print("   - Style modifiers in the prompt")
        print("   - Output dimensions, checked against the returned image")
        print("   - Dynamic model selection")

    print("\n📁 Files written by this run:")
    for path in written:
        print(f"   - {path}")
    if not written:
        print("   (none)")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
