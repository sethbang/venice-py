#!/usr/bin/env python3
"""
Venice AI SDK - Multi-Image Editing
===================================

This example demonstrates ``client.image.multi_edit()``. Unlike the
single-image ``edit()`` method, ``multi_edit()`` accepts up to **three input
images** (``image``, ``image_2``, ``image_3``) plus a text prompt. The first
image is the base; the others are references the model draws on. It is
prompt-driven editing, not pixel compositing: the model decides how to
combine the inputs from the instruction and may invent new detail.

Only edit models whose ``constraints.combineImages`` is true accept more than
one image, so the multi-image demos select their model with
``client.models.resolve_inpaint(require_combine_images=True, prefer="cheapest")``,
the lowest-priced model that qualifies.

Three input images are generated once (a desert, a night sky and a set of
lanterns) and reused by every demo.

``multi_edit()`` returns raw image **bytes**; the examples save them with the
extension detected from the bytes. Outputs are written to
``examples/results/``.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError
from venice_ai.types.api import InpaintModelSpec

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


def _save(data: bytes, stem: str, written: list[Path]) -> None:
    path = save_image(RESULTS_DIR / stem, data)
    written.append(path)
    width, height = image_dimensions(data)
    print(f"   💾 {path.name}: {width}x{height}, {len(data)} bytes")


async def _quote(client: VeniceClient, model: str, input_images: int) -> None:
    spec = (await client.models.get(model)).model_spec
    price = catalog_price_usd(spec, input_images=input_images)
    if price is not None:
        print(f"💰 Catalog price with {input_images} input image(s): ${price:.2f}")


# ---------------------------------------------------------------------------
# 1. Basic multi-edit with a single image
# ---------------------------------------------------------------------------


async def basic_multi_edit(client: VeniceClient, desert: bytes, written: list[Path]) -> bool:
    """Use multi_edit() with a single image and prompt (simplest case)."""
    print("🔀 Basic Multi-Edit (Single Image)")
    print("-" * 30)

    model = await client.models.resolve_inpaint(prefer="cheapest")
    print(f"📍 Using edit model: {model}")
    await _quote(client, model, 1)

    try:
        print("🔀 Applying multi-edit: a camel caravan …")
        edited = await client.image.multi_edit(
            prompt="Add a camel caravan walking along the crest of the dunes",
            model=model,
            image=desert,
        )
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    _save(edited, "multi_edit_caravan", written)
    return True


# ---------------------------------------------------------------------------
# 2. Two input images
# ---------------------------------------------------------------------------


async def two_image_editing(
    client: VeniceClient, model: str, desert: bytes, sky: bytes, written: list[Path]
) -> bool:
    """Combine two images with one prompt."""
    print("\n🖼️  Two-Image Editing")
    print("-" * 30)
    print(f"📍 Using multi-image edit model: {model}")
    await _quote(client, model, 2)

    try:
        print("🔀 Combining the two images …")
        combined = await client.image.multi_edit(
            prompt=(
                "Put the starry night sky from the second image above the desert "
                "dunes of the first image, creating a twilight scene"
            ),
            model=model,
            image=desert,
            image_2=sky,
        )
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    _save(combined, "multi_edit_two_images", written)
    return True


# ---------------------------------------------------------------------------
# 3. Three input images
# ---------------------------------------------------------------------------


async def three_image_composition(
    client: VeniceClient,
    model: str,
    desert: bytes,
    sky: bytes,
    lanterns: bytes,
    written: list[Path],
) -> bool:
    """Combine three images into one scene."""
    print("\n🎭 Three-Image Composition")
    print("-" * 30)
    print(f"📍 Using multi-image edit model: {model}")
    await _quote(client, model, 3)

    try:
        print("🔀 Combining three images into a single scene …")
        final = await client.image.multi_edit(
            prompt=(
                "Merge these three images into one night scene: the desert dunes "
                "from the first image, the starry sky from the second above them, "
                "and the glowing lanterns from the third floating over the sand"
            ),
            model=model,
            image=desert,
            image_2=sky,
            image_3=lanterns,
        )
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    _save(final, "multi_edit_three_images", written)
    print("\n📋 Inputs:")
    print("   • image   → desert dunes (base)")
    print("   • image_2 → starry night sky")
    print("   • image_3 → floating lanterns")
    return True


# ---------------------------------------------------------------------------
# 4. Model selection for multi-edit
# ---------------------------------------------------------------------------


async def model_selection(client: VeniceClient) -> bool:
    """Show which edit models accept several images and which one is selected.

    This demo only reads the catalog; the two- and three-image demos make the
    paid edits with the model selected here.
    """
    print("\n🔍 Model Selection for Multi-Edit")
    print("-" * 30)

    listing = await client.models.list(type="inpaint")
    multi = []
    single = []
    for entry in listing.data:
        spec = entry.model_spec
        if isinstance(spec, InpaintModelSpec) and spec.constraints is not None:
            (multi if spec.constraints.combineImages else single).append(entry.id)
    print(f"🎨 Edit models that accept several images ({len(multi)}):")
    for model_id in multi:
        print(f"   • {model_id}")
    print(f"🚫 Single-image only ({len(single)}): {', '.join(single) or 'none'}")

    try:
        catalog_pick = await client.models.resolve_inpaint(require_combine_images=True)
        cheapest = await client.models.resolve_inpaint(
            require_combine_images=True, prefer="cheapest"
        )
    except NoMatchingModelError as e:
        print(f"❌ No multi-image edit model is available: {e}")
        return False
    print(f"\n📍 require_combine_images=True → {catalog_pick} (catalog order)")
    print(f"📍 require_combine_images=True, prefer='cheapest' → {cheapest}")
    await _quote(client, cheapest, 2)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all multi-edit examples.

    Returns ``0`` only if every demo succeeded, ``1`` if any failed, and ``77``
    if no catalog edit model combines several images or no image model is
    sized by width/height.
    """
    print("🚀 Venice AI Multi-Image Editing Examples")
    print("=" * 50)

    written: list[Path] = []
    async with VeniceClient() as client:
        try:
            multi_model = await client.models.resolve_inpaint(
                require_combine_images=True, prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no edit model in the catalog combines several images ({e})")
            return SKIPPED

        inputs: dict[str, bytes] = {}
        try:
            for name, prompt in [
                ("desert", "A dramatic desert landscape with sand dunes at golden hour"),
                ("sky", "A starry night sky with the Milky Way and shooting stars"),
                ("lanterns", "Glowing paper lanterns floating in a dark scene"),
            ]:
                print(f"🎨 Generating input image: {name} …")
                inputs[name] = await generate_base_image(client, prompt)
                _save(inputs[name], f"multi_edit_input_{name}", written)
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        except VeniceError as e:
            print(f"❌ Could not generate the input images: {e}")
            return 1
        desert, sky, lanterns = inputs["desert"], inputs["sky"], inputs["lanterns"]
        print()

        results: list[tuple[str, bool]] = [
            ("basic_multi_edit", await basic_multi_edit(client, desert, written)),
            (
                "two_image_editing",
                await two_image_editing(client, multi_model, desert, sky, written),
            ),
            (
                "three_image_composition",
                await three_image_composition(client, multi_model, desert, sky, lanterns, written),
            ),
            ("model_selection", await model_selection(client)),
        ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 50)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✨ Multi-edit examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Single-image multi-edit (simplest case)")
        print("   - Two input images (image + image_2)")
        print("   - Three input images (image + image_2 + image_3)")
        print("   - resolve_inpaint(require_combine_images=True, prefer='cheapest')")
        print("   - Self-contained generate → multi-edit workflow")

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
