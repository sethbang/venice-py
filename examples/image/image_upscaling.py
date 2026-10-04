#!/usr/bin/env python3
"""
Venice AI SDK - Image Upscaling
================================

This example demonstrates how to upscale images with ``client.image.upscale()``.

- ``scale`` accepts any value from 2 to 4; the output is the input size times
  the scale. The demos use 2 and 4, the two scales the catalog prices.
- ``creativity`` sets how much detail and texture the upscaler adds. The
  server clamps it to 0-0.02, so its effect is subtle.
- The image can be a file path, a ``Path``, raw bytes or a file-like object.
  The demos pass a ``Path``, a path string and raw bytes.

The example generates one small source image and reuses it across the demos,
so every output can be compared with the same original. Each output's pixel
size is read from its header and checked against the expected size. A quote
of the run's catalog price is printed first. Outputs are written to
``examples/results/``.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    SKIPPED,
    catalog_price_usd,
    generate_base_image,
    image_dimensions,
    save_image,
)

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BASE_SIZE = 256

# Upscales usually take 10-20s but can take over a minute under load, so every
# call passes an explicit per-call timeout rather than relying on the default.
UPSCALE_TIMEOUT = 180.0

# The paid calls the demos below make, for the quote printed before they run:
# upscales per scale factor, and generated source images.
PLANNED_UPSCALES = {2: 3, 4: 1}
SOURCE_IMAGES = 1


async def _upscale_and_check(
    client: VeniceClient,
    image: str | bytes | Path,
    source_size: tuple[int, int],
    stem: str,
    written: list[Path],
    *,
    scale: float = 2,
    creativity: float | None = None,
    timeout: float = UPSCALE_TIMEOUT,
) -> bool:
    """Upscale ``image``, save the result, and check its size is ``source * scale``.

    ``timeout`` (seconds, or an ``aiohttp.ClientTimeout``) caps each request.
    Large sources and high scale factors take longer, so a generous per-call
    timeout avoids a premature client-side abort. ``image.edit()`` takes the
    same parameter (see image_editing.py).
    """
    try:
        upscaled = await client.image.upscale(
            image=image, scale=scale, creativity=creativity, timeout=timeout
        )
    except VeniceError as e:
        print(f"   ❌ Upscale failed: {e}")
        return False

    path = save_image(RESULTS_DIR / stem, upscaled)
    written.append(path)
    width, height = image_dimensions(upscaled)
    expected = (round(source_size[0] * scale), round(source_size[1] * scale))
    print(
        f"   ✅ {path.name}: {source_size[0]}x{source_size[1]} → {width}x{height} "
        f"({len(upscaled)} bytes)"
    )
    if (width, height) != expected:
        print(f"   ❌ Expected {expected[0]}x{expected[1]} for scale={scale}")
        return False
    return True


async def basic_upscaling(
    client: VeniceClient, base_path: Path, base_size: tuple[int, int], written: list[Path]
) -> bool:
    """Upscale a file on disk, passed as a ``Path``, by 2x."""
    print("\n🔍 Basic Image Upscaling")
    print("-" * 40)
    return await _upscale_and_check(
        client, base_path, base_size, "upscaled_2x_basic", written, scale=2
    )


async def creativity_sweep(
    client: VeniceClient,
    base_path: Path,
    base_bytes: bytes,
    base_size: tuple[int, int],
    written: list[Path],
) -> bool:
    """Upscale the same source at the lowest and highest ``creativity``.

    The first call passes the image as a path string, the second as raw bytes
    held in memory, so both input types are exercised.
    """
    print("\n🎭 Creativity Sweep")
    print("-" * 40)
    print("   The server clamps creativity to 0-0.02 (default 0.01). The extra texture")
    print("   at 0.02 is subtle and can be hard to tell from run-to-run variation.")

    ok = True
    sources: list[tuple[str, float, str | bytes, str]] = [
        ("faithful", 0.0, str(base_path), "path string"),
        ("detailed", 0.02, base_bytes, "raw bytes"),
    ]
    for name, creativity, image, input_type in sources:
        print(f"   creativity={creativity} ({name}), image as {input_type}")
        ok &= await _upscale_and_check(
            client,
            image,
            base_size,
            f"upscaled_creativity_{name}",
            written,
            scale=2,
            creativity=creativity,
        )
    return ok


async def different_scale_factors(
    client: VeniceClient, base_path: Path, base_size: tuple[int, int], written: list[Path]
) -> bool:
    """Upscale the same source at 4x (the basic demo covers 2x)."""
    print("\n📐 Different Scale Factors")
    print("-" * 40)
    print("   scale=4")
    return await _upscale_and_check(client, base_path, base_size, "upscaled_4x", written, scale=4)


async def _print_quote(client: VeniceClient, image_model: str) -> None:
    """Print the catalog price of every paid call this run makes."""
    upscalers = await client.models.list(type="upscale")
    tiers: dict[str, float] = {}
    for entry in upscalers.data:
        pricing = entry.model_spec.pricing
        listed = pricing.model_dump().get("upscale", {}) if pricing else {}
        tiers = {name: float(tier["usd"]) for name, tier in listed.items()}
        break

    print("💰 Catalog quote for this run:")
    total = 0.0
    unpriced: list[str] = []
    for scale, count in PLANNED_UPSCALES.items():
        price = tiers.get(f"x{scale}")
        if price is None:
            unpriced.append(f"scale={scale}")
            print(f"   {count} upscale(s) at scale={scale}: no catalog tier listed")
            continue
        total += count * price
        print(f"   {count} upscale(s) at scale={scale}: {count} x ${price:.2f}")
    gen_price = catalog_price_usd((await client.models.get(image_model)).model_spec)
    if gen_price is None:
        unpriced.append("source images")
        print(f"   {SOURCE_IMAGES} source image(s) with {image_model}: price not listed")
    else:
        total += SOURCE_IMAGES * gen_price
        print(
            f"   {SOURCE_IMAGES} source image(s) with {image_model}: {SOURCE_IMAGES} x ${gen_price:.2f}"
        )
    note = f", plus the unpriced {', '.join(unpriced)}" if unpriced else ""
    print(f"   Total: ${total:.2f}{note}")


async def main() -> int:
    """Run all image upscaling examples.

    Returns ``0`` only if every demo succeeded, ``1`` if any failed, and ``77``
    if no catalog image model is sized by width/height.
    """
    print("🚀 Venice AI Image Upscaling Examples")
    print("=" * 50)

    written: list[Path] = []
    async with VeniceClient() as client:
        # Pixel sizes are requested below, so pick among models sized by width/height.
        try:
            image_model = await client.models.resolve_image(
                prefer="cheapest", require_custom_size=True
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        await _print_quote(client, image_model)

        print(f"\n🎨 Generating one {BASE_SIZE}x{BASE_SIZE} source image for the demos...")
        try:
            base_bytes = await generate_base_image(
                client,
                "A detailed fantasy landscape with mountains, forest, lake, and castle "
                "at sunset, rich colors and intricate details",
                model=image_model,
                width=BASE_SIZE,
                height=BASE_SIZE,
            )
        except VeniceError as e:
            print(f"❌ Could not generate the source image: {e}")
            return 1
        base_path = save_image(RESULTS_DIR / f"upscale_source_{BASE_SIZE}", base_bytes)
        written.append(base_path)
        base_size = image_dimensions(base_bytes)
        print(f"   💾 {base_path.name}: {base_size[0]}x{base_size[1]}")

        results: list[tuple[str, bool]] = [
            (
                "basic_upscaling",
                await basic_upscaling(client, base_path, base_size, written),
            ),
            (
                "creativity_sweep",
                await creativity_sweep(client, base_path, base_bytes, base_size, written),
            ),
            (
                "different_scale_factors",
                await different_scale_factors(client, base_path, base_size, written),
            ),
        ]

    failed = [name for name, ok in results if not ok]
    passed = len(results) - len(failed)

    if failed:
        print(f"\n⚠️ {passed}/{len(results)} demos completed; failed: {', '.join(failed)}")
    else:
        print(f"\n✨ Image upscaling examples completed! ({passed}/{len(results)})")
        print("\n💡 Key concepts demonstrated:")
        print("   - Upscaling with a generous per-call timeout")
        print("   - Image inputs as a Path, a path string and raw bytes")
        print("   - The creativity setting (0-0.02)")
        print("   - Scale factors 2 and 4, checked against the output size")

    print(f"\n📁 Files written by this run ({len(written)}):")
    for path in written:
        print(f"   - {path.name}")

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
