#!/usr/bin/env python3
"""
Venice AI SDK - Background Removal
====================================

This example demonstrates how to remove backgrounds from images using the Venice AI SDK.
Learn how to isolate subjects, create transparent PNGs, and build compositing workflows
with AI-powered background removal.

One source image is generated without the Venice watermark, so no watermark
fragment survives into the cutouts, and every demo reuses it. Each cutout's
pixels are decoded (this needs Pillow) and checked: at least
``MIN_CLEAR_FRACTION`` of the image must be see-through, so a background was
removed, and at least ``MIN_SOLID_FRACTION`` must stay solid, so the subject
was kept. A cutout must also keep its source's pixel size. Without Pillow the
example exits with code 77 before making any paid call.

Each run makes one paid generation and five background removals. Outputs are
written to ``examples/results/``.
"""

import asyncio
import sys
from io import BytesIO
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    SKIPPED,
    alpha_coverage,
    generate_base_image,
    image_dimensions,
    pillow_available,
    save_image,
)

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A publicly reachable, direct image link for the URL input demo.
SAMPLE_IMAGE_URL = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400"

# A cutout must be at least this see-through (background gone) and this solid
# (subject kept), as fractions of its pixels.
MIN_CLEAR_FRACTION = 0.10
MIN_SOLID_FRACTION = 0.05


def _save_cutout(
    cutout: bytes, stem: str, written: list[Path], source_size: tuple[int, int] | None
) -> bool:
    """Save a cutout and check its transparency (and the source's size)."""
    path = save_image(RESULTS_DIR / stem, cutout)
    written.append(path)
    width, height = image_dimensions(cutout)
    print(f"   💾 {path.name}: {width}x{height}, {len(cutout)} bytes")
    coverage = alpha_coverage(cutout)
    if coverage is None:
        raise RuntimeError("Pillow is required to check cutouts")
    clear, solid = coverage
    print(f"   🔍 {clear:.1%} see-through, {solid:.1%} solid")
    if clear < MIN_CLEAR_FRACTION:
        print(f"   ❌ Under {MIN_CLEAR_FRACTION:.0%} see-through, so no background was removed")
        return False
    if solid < MIN_SOLID_FRACTION:
        print(f"   ❌ Under {MIN_SOLID_FRACTION:.0%} solid, so the subject was removed too")
        return False
    if source_size is not None and (width, height) != source_size:
        print(f"   ❌ Size changed from {source_size[0]}x{source_size[1]}")
        return False
    return True


async def basic_background_removal(
    client: VeniceClient, original: bytes, written: list[Path]
) -> bool:
    """Remove the background from image bytes held in memory."""
    print("✂️ Basic Background Removal")
    print("-" * 40)

    try:
        print("✂️ Removing background...")
        cutout = await client.image.background_remove(image=original)
    except VeniceError as e:
        print(f"❌ Error during background removal: {e}")
        return False

    return _save_cutout(cutout, "bg_removal_transparent", written, image_dimensions(original))


async def background_removal_from_url(client: VeniceClient, written: list[Path]) -> bool:
    """Remove the background from an image the server downloads from a URL."""
    print("\n🌐 Background Removal from URL")
    print("-" * 40)
    print(f"🔗 Image URL: {SAMPLE_IMAGE_URL}")

    try:
        cutout = await client.image.background_remove(image_url=SAMPLE_IMAGE_URL)
    except VeniceError as e:
        print(f"❌ Error removing background from URL: {e}")
        print("💡 The URL must be a publicly reachable, direct image link")
        return False

    return _save_cutout(cutout, "bg_removal_from_url", written, None)


async def different_input_methods(
    client: VeniceClient, sample: bytes, sample_path: Path, written: list[Path]
) -> bool:
    """Pass the image as a path string, a file-like object and a Path.

    The basic demo above already sends raw bytes.
    """
    print("\n📂 Different Input Methods")
    print("-" * 40)
    sample_size = image_dimensions(sample)

    methods: list[tuple[str, str, str | BytesIO | Path]] = [
        ("📁 Method 1: File path (string)", "path", str(sample_path)),
        ("📖 Method 2: File-like object (BinaryIO)", "fileobj", BytesIO(sample)),
        ("🗂️ Method 3: Path object", "pathobj", sample_path),
    ]

    ok = True
    for label, tag, image in methods:
        print(f"\n{label}")
        try:
            cutout = await client.image.background_remove(image=image)
        except VeniceError as e:
            print(f"   ❌ Failed: {e}")
            ok = False
            continue
        ok &= _save_cutout(cutout, f"bg_removal_from_{tag}", written, sample_size)
    return ok


async def main() -> int:
    """Run all background removal examples.

    Returns ``0`` only if every demo succeeded, ``1`` if any failed, and ``77``
    if Pillow is missing or no catalog image model is sized by width/height.
    """
    print("🚀 Venice AI Background Removal Examples")
    print("=" * 50)

    if not pillow_available():
        print(
            "SKIPPED: Pillow is not installed; it is needed to check cutouts (pip install Pillow)"
        )
        return SKIPPED

    written: list[Path] = []
    async with VeniceClient() as client:
        try:
            print("🎨 Generating image of a red sports car on a city street...")
            original = await generate_base_image(
                client, "A red sports car parked on a city street, clear subject, studio lighting"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        except VeniceError as e:
            print(f"❌ Could not generate the source image: {e}")
            return 1
        original_path = save_image(RESULTS_DIR / "bg_removal_original", original)
        written.append(original_path)
        print(f"   💾 Original: {original_path.name} ({len(original)} bytes)\n")

        results: list[tuple[str, bool]] = [
            ("basic_background_removal", await basic_background_removal(client, original, written)),
            ("background_removal_from_url", await background_removal_from_url(client, written)),
            (
                "different_input_methods",
                await different_input_methods(client, original, original_path, written),
            ),
        ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Background removal examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Basic background removal from image bytes")
        print("   - Background removal from URL (image_url parameter)")
        print("   - Multiple input methods (bytes, path, BinaryIO, Path)")
        print("   - Checking how much of each cutout is see-through and how much is solid")

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
