#!/usr/bin/env python3
"""
Venice AI SDK - Image Editing
=============================

This example demonstrates how to edit and modify existing images using
the Venice AI SDK's image editing capabilities.  Learn how to use
``client.image.edit()`` to apply AI-powered edits such as adding objects,
changing scenes, removing elements, and prompt-based targeted editing.

The ``edit()`` method returns raw image **bytes**; the examples save them with
the extension detected from the bytes.

Edit models differ in what they accept, so each demo selects its model from
catalog traits with ``client.models.resolve_inpaint()`` rather than by ID,
adding ``prefer="cheapest"`` to take the lowest-priced model that qualifies:

- no filter for the default edit model the basic and input-method demos use
- ``require_resolution="2K"`` for a model that accepts a ``resolution`` tier
  (models without a ``resolutions`` constraint reject the parameter)
- ``require_uncensored=True`` for a model that honours ``safe_mode=False``
- ``require_quality`` / ``require_combine_images`` for quality tiers and
  multi-image input (see ``quality_control.py`` and ``multi_edit.py``)

One base image is generated and reused by every demo. Each run makes that
generation plus six paid edits. Outputs are written to ``examples/results/``.
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
    format_price,
    generate_base_image,
    image_dimensions,
    save_image,
)

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A publicly reachable, direct image link for the URL input demo.
SAMPLE_IMAGE_URL = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400"

# A 2K-tier output has a long side of about 2048px, whatever the source size.
TWO_K_MIN_LONG_SIDE = 1920


def _save(data: bytes, stem: str, written: list[Path]) -> Path:
    path = save_image(RESULTS_DIR / stem, data)
    written.append(path)
    width, height = image_dimensions(data)
    print(f"   💾 {path.name}: {width}x{height}, {len(data)} bytes")
    return path


# ---------------------------------------------------------------------------
# 1. Basic image editing
# ---------------------------------------------------------------------------


async def basic_image_editing(
    client: VeniceClient, edit_model: str, base: bytes, written: list[Path]
) -> bool:
    """Edit the base image, passed as raw bytes, with a text prompt.

    The edit endpoint has no mask parameter: the model decides what to change
    from the instruction, so name the region or subject explicitly.
    """
    print("✏️  Basic Image Editing")
    print("-" * 30)
    print("ℹ️  Be specific about the region/subject in the instruction")
    print("   (e.g. 'add a rainbow across the sky').\n")

    try:
        print(f"✏️  Editing with {edit_model}: adding a vibrant rainbow …")
        edited = await client.image.edit(
            prompt="Add a vivid rainbow arching across the sky",
            model=edit_model,
            image=base,  # pass raw bytes directly
        )
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    _save(edited, "edit_with_rainbow", written)
    return True


# ---------------------------------------------------------------------------
# 2. Different input methods
# ---------------------------------------------------------------------------


async def different_input_methods(
    client: VeniceClient, edit_model: str, base_path: Path, written: list[Path]
) -> bool:
    """Pass the image to ``edit()`` as a path string, a file object and a URL.

    The basic demo above already passes raw bytes.
    """
    print("\n🖼️  Different Input Methods")
    print("-" * 30)
    prompt = "Add a small wooden cabin at the edge of the meadow"

    ok = True
    for label, tag, source in [
        ("📂 Method A — file path string", "filepath", str(base_path)),
        ("📄 Method B — file-like object", "fileobj", None),
        ("🌐 Method C — URL (sent as-is in the JSON body)", "url", SAMPLE_IMAGE_URL),
    ]:
        print(f"\n{label} …")
        try:
            if source is None:
                with open(base_path, "rb") as img_file:
                    edited = await client.image.edit(
                        prompt=prompt, model=edit_model, image=img_file
                    )
            else:
                edited = await client.image.edit(
                    prompt="Add a small red scarf around the dog's neck"
                    if tag == "url"
                    else prompt,
                    model=edit_model,
                    image=source,
                )
        except VeniceError as e:
            print(f"   ❌ Failed: {e}")
            ok = False
            continue
        _save(edited, f"edit_input_{tag}", written)
    return ok


# ---------------------------------------------------------------------------
# 3. Model selection for image editing
# ---------------------------------------------------------------------------


async def model_selection(client: VeniceClient) -> bool:
    """Show the edit catalog's traits and select models by capability and price.

    This demo only reads the catalog; the other demos make the paid edits with
    the models selected here. A capability no catalog model offers is reported
    as "no match" (``NoMatchingModelError``), which is the resolver's answer
    about the catalog rather than a failure, so the demo still returns ``True``.
    """
    print("\n🔍 Model Selection for Image Editing")
    print("-" * 30)

    listing = await client.models.list(type="inpaint")
    print(f"🎨 Edit models in the catalog ({len(listing.data)}):")
    for entry in listing.data:
        spec = entry.model_spec
        if not isinstance(spec, InpaintModelSpec) or spec.constraints is None:
            continue
        c = spec.constraints
        traits = [
            "multi-image" if c.combineImages else "single-image",
            f"resolutions={c.resolutions}" if c.resolutions else None,
            f"qualities={c.qualities}" if c.qualities else None,
            "uncensored" if spec.uncensored else None,
        ]
        price = catalog_price_usd(spec, quality=c.defaultQuality)
        shown = f"${price:.2f}" if price is not None else "n/a"
        if c.qualities and c.defaultQuality:
            low = catalog_price_usd(spec, quality=c.qualities[0])
            shown += f" at quality='{c.defaultQuality}'"
            if low is not None and low != price:
                shown += f", ${low:.2f} at quality='{c.qualities[0]}'"
        print(f"   • {entry.id}: {shown}, {', '.join(t for t in traits if t)}")

    print("\n🎯 Selecting by capability:")
    selections = [
        ("default", {}),
        ("require_combine_images=True", {"require_combine_images": True}),
        ("require_resolution='2K'", {"require_resolution": "2K"}),
        ("require_quality='high'", {"require_quality": "high"}),
        ("require_uncensored=True", {"require_uncensored": True}),
    ]
    for label, kwargs in selections:
        try:
            catalog_pick = await client.models.resolve_inpaint(**kwargs)
            cheapest = await client.models.resolve_inpaint(**kwargs, prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"   {label:30} → no match ({e})")
            continue
        print(f"   {label:30} → {catalog_pick} (catalog order), {cheapest} (prefer='cheapest')")
    return True


# ---------------------------------------------------------------------------
# 4. Disabling safe_mode
# ---------------------------------------------------------------------------


async def disable_safe_mode(client: VeniceClient, base: bytes, written: list[Path]) -> bool | None:
    """Pair ``safe_mode=False`` with an edit model that honours it.

    ``safe_mode`` (on by default) blurs output the server classifies as adult
    content. Only models Venice flags as ``uncensored`` return such content
    unblurred, so the model is selected with ``require_uncensored=True``. For
    a non-adult prompt like the one below, the result looks the same either
    way; the point is the pairing of flag and model.
    """
    print("\n🔓 Disabling safe_mode")
    print("-" * 30)

    try:
        model = await client.models.resolve_inpaint(require_uncensored=True, prefer="cheapest")
    except NoMatchingModelError as e:
        print(f"Section skipped: no uncensored edit model is available ({e})")
        return None
    print(f"📍 Using uncensored edit model: {model}")

    try:
        edited = await client.image.edit(
            prompt="Add a dramatic cinematic lighting effect",
            model=model,
            image=base,
            safe_mode=False,
        )
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    _save(edited, "edit_safe_mode_off", written)
    return True


# ---------------------------------------------------------------------------
# 5. Resolution tier + request timeout
# ---------------------------------------------------------------------------


async def edit_with_resolution_and_timeout(
    client: VeniceClient, base: bytes, written: list[Path]
) -> bool | None:
    """Request a ``resolution`` tier with a generous per-call ``timeout``.

    - ``resolution`` (``"1K"`` / ``"2K"`` / ``"4K"``) is accepted only by edit
      models that list it in their ``resolutions`` constraint; others reject
      the request with a 400. ``resolve_inpaint(require_resolution="2K")``
      selects a model that accepts it.
    - ``timeout`` caps the request. Higher-resolution edits take longer, so a
      generous per-call timeout (seconds, or an ``aiohttp.ClientTimeout``)
      avoids a premature client-side abort. ``upscale()`` takes the same
      parameter (see image_upscaling.py).

    ``multi_edit()`` accepts ``resolution`` too, with the same rules.
    """
    print("\n📐 Resolution tier + timeout")
    print("-" * 30)

    try:
        model = await client.models.resolve_inpaint(require_resolution="2K", prefer="cheapest")
    except NoMatchingModelError as e:
        print(f"Section skipped: no edit model accepts resolution='2K' ({e})")
        return None
    spec = (await client.models.get(model)).model_spec
    price = catalog_price_usd(spec, resolution="2K")
    print(f"📍 Using edit model: {model} (catalog {format_price(price)} at 2K)")

    base_w, base_h = image_dimensions(base)
    try:
        print("✏️  Editing at resolution='2K' with a 180s timeout …")
        edited = await client.image.edit(
            prompt="Add dramatic storm clouds gathering on the horizon",
            model=model,
            image=base,
            resolution="2K",
            timeout=180.0,
        )
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    _save(edited, "edit_resolution_2k", written)
    width, height = image_dimensions(edited)
    if max(width, height) < TWO_K_MIN_LONG_SIDE:
        print(
            f"❌ Expected a 2K-tier output (long side about 2048px) from a "
            f"{base_w}x{base_h} source, got {width}x{height}"
        )
        return False
    print(f"✅ 2K-tier output: {width}x{height} from a {base_w}x{base_h} source")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all image editing examples.

    Returns ``0`` if every demo that ran succeeded (a demo with no suitable
    catalog model is skipped), ``1`` if any failed, and ``77`` if the catalog
    has no edit model at all, or no image model sized by width/height to
    generate the base image.
    """
    print("🚀 Venice AI Image Editing Examples")
    print("=" * 50)

    written: list[Path] = []
    async with VeniceClient() as client:
        try:
            edit_model = await client.models.resolve_inpaint(prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no image edit model ({e})")
            return SKIPPED
        spec = (await client.models.get(edit_model)).model_spec
        price = catalog_price_usd(spec)
        print(f"📍 Default edit model: {edit_model} (catalog {format_price(price)} per edit)\n")

        try:
            print("🎨 Generating the base image every demo edits …")
            base = await generate_base_image(
                client, "A serene mountain landscape with a clear blue sky and green meadows"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        except VeniceError as e:
            print(f"❌ Could not generate the base image: {e}")
            return 1
        base_path = _save(base, "edit_base_landscape", written)
        print()

        results: list[tuple[str, bool | None]] = [
            ("basic_image_editing", await basic_image_editing(client, edit_model, base, written)),
            (
                "different_input_methods",
                await different_input_methods(client, edit_model, base_path, written),
            ),
            ("model_selection", await model_selection(client)),
            ("disable_safe_mode", await disable_safe_mode(client, base, written)),
            (
                "edit_with_resolution_and_timeout",
                await edit_with_resolution_and_timeout(client, base, written),
            ),
        ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    print("\n" + "=" * 50)
    if skipped:
        print(f"⏭️ Skipped for lack of a catalog model: {', '.join(skipped)}")
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✨ Image editing examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Generate → edit workflow (self-contained)")
        print("   - Prompt-based targeted editing")
        print("   - Input methods: bytes, file path, file object, URL")
        print("   - Selecting edit models by catalog traits and price with resolve_inpaint()")
        if "disable_safe_mode" not in skipped:
            print("   - safe_mode=False on an uncensored edit model")
        if "edit_with_resolution_and_timeout" not in skipped:
            print("   - resolution='2K' on a resolution-aware model, with a per-call timeout")

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
