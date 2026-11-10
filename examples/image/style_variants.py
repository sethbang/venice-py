#!/usr/bin/env python3
"""
Venice AI SDK - Image Style Variations
=======================================

This example demonstrates how to generate images in different artistic styles.
Learn how to control the aesthetic and artistic direction of AI-generated images.

Each section renders one subject with two contrasting style modifiers. Both
renders in a section share a seed, so the style modifier is the only input
that changes between them. How closely a modifier is followed depends on the
model.

The API also has named presets. ``client.image.list_styles()`` lists them for free
on every run, and a paid run renders one with ``style_preset``. The catalog has no
flag for which models apply a preset, so the check is that the server echoes the
one sent; whether it changed the picture is for you to judge.

Each returned image's format and pixel size are read from its bytes and
checked against the 512x512 request.

Generation costs real money, so by default this example only resolves the model
(free), prints what it would run and exits 77 ("skipped"). Opt in to make the
11 paid generations, about $0.11 at the cheapest listed $0.01 per image, and write
the outputs to ``examples/results/`` with a ``style_`` prefix::

    VENICE_RUN_PAID_IMAGES=1
"""

import asyncio
import os
import sys
from pathlib import Path

from _helpers import SKIPPED, catalog_price_usd, format_price, image_dimensions

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

WIDTH = HEIGHT = 512

RUN_PAID = os.environ.get("VENICE_RUN_PAID_IMAGES") == "1"

SECTIONS: list[tuple[str, str, str, int, list[tuple[str, str]]]] = [
    (
        "Artistic styles",
        "lion",
        "a majestic lion",
        1001,
        [
            ("photorealistic", "professional wildlife photography, 8k, detailed"),
            ("pixel_art", "retro pixel art, 16-bit game style"),
        ],
    ),
    (
        "Mediums",
        "garden",
        "a serene Japanese garden with cherry blossoms",
        1002,
        [
            ("pencil_sketch", "detailed pencil sketch, graphite drawing"),
            ("acrylic", "vibrant acrylic painting, bold colors"),
        ],
    ),
    (
        "Art movements",
        "reading",
        "a woman reading a book by candlelight",
        1003,
        [
            ("baroque", "Baroque style, Rembrandt lighting, dramatic chiaroscuro"),
            ("pop_art", "Pop Art style, Warhol inspired, bold colors"),
        ],
    ),
    (
        "Moods",
        "forest",
        "a forest path",
        1004,
        [
            ("cheerful", "bright sunny day, vibrant colors, joyful atmosphere"),
            ("mysterious", "foggy evening, mysterious atmosphere, ethereal"),
        ],
    ),
    (
        "Color palettes",
        "city",
        "a cityscape at dusk",
        1005,
        [
            ("warm", "warm color palette, oranges and reds, sunset tones"),
            ("monochrome", "black and white, high contrast, noir style"),
        ],
    ),
]


async def render_section(
    client: VeniceClient,
    model: str,
    title: str,
    prefix: str,
    subject: str,
    seed: int,
    variants: list[tuple[str, str]],
    written: list[Path],
) -> list[tuple[str, bool, str | None]]:
    """Render ``subject`` once per variant, all with the same seed."""
    print(f"\n🎨 {title}: '{subject}' in {len(variants)} variants (seed {seed})")
    print("-" * 40)

    results: list[tuple[str, bool, str | None]] = []
    for name, modifier in variants:
        print(f"\n   {name}")
        try:
            response: ImageGenerationResponse = await client.image.create(
                model=model,
                prompt=f"{subject}, {modifier}",
                width=WIDTH,
                height=HEIGHT,
                num_images=1,
                seed=seed,
                hide_watermark=True,
                return_binary=False,
            )
        except VeniceError as e:
            print(f"   ❌ Failed: {e}")
            results.append((name, False, str(e)))
            continue

        data = response.bytes(0)
        try:
            size = image_dimensions(data)
        except ValueError as e:
            print(f"   ❌ Not a recognised image: {e}")
            results.append((name, False, str(e)))
            continue
        path = response.save(RESULTS_DIR / f"style_{prefix}_{name}", overwrite=True)
        written.append(path)
        timing = f", {response.timing.inferenceDuration}ms" if response.timing else ""
        print(f"   ✅ {path.name} ({size[0]}x{size[1]}, {len(data)} bytes{timing})")
        if size != (WIDTH, HEIGHT):
            error = f"requested {WIDTH}x{HEIGHT}, received {size[0]}x{size[1]}"
            print(f"   ❌ {error}")
            results.append((name, False, error))
            continue
        results.append((name, True, None))

    succeeded = sum(1 for _, ok, _ in results if ok)
    print(f"\n📊 {title}: {succeeded}/{len(results)} generated")
    return results


async def list_presets(client: VeniceClient) -> list[str]:
    """List the named style presets (a free call) and print the first few."""
    print("\n🗂️  Style presets from client.image.list_styles() (free)")
    styles = (await client.image.list_styles()).data
    shown = ", ".join(styles[:8])
    print(f"   {len(styles)} presets: {shown}{', …' if len(styles) > 8 else ''}")
    return styles


async def render_preset(
    client: VeniceClient, model: str, preset: str, written: list[Path]
) -> tuple[str, bool, str | None]:
    """Render with a named ``style_preset`` and check the server echoes it."""
    print(f"\n🎨 style_preset={preset!r}")
    try:
        response = await client.image.create(
            model=model,
            prompt="a quiet harbor at sunrise",
            width=WIDTH,
            height=HEIGHT,
            style_preset=preset,
            hide_watermark=True,
            return_binary=False,
        )
    except VeniceError as e:
        print(f"   ❌ Failed: {e}")
        return preset, False, str(e)
    data = response.bytes(0)
    path = response.save(RESULTS_DIR / "style_preset", overwrite=True)
    written.append(path)
    size = image_dimensions(data)
    print(f"   ✅ {path.name} ({size[0]}x{size[1]}, {len(data)} bytes)")
    echo = response.request.get("data") if isinstance(response.request, dict) else None
    echoed = echo.get("style_preset") if isinstance(echo, dict) else None
    if echoed != preset:
        print(f"   ❌ The server echoed style_preset={echoed!r}")
        return preset, False, f"echoed {echoed!r}"
    print("   ✔ The server echoed the preset")
    return preset, True, None


async def main() -> int:
    """Run all style variation examples.

    Returns ``0`` only if every render succeeded, ``1`` if any failed, and
    ``77`` if ``VENICE_RUN_PAID_IMAGES`` is unset or no catalog image model is
    sized by width/height.
    """
    print("🚀 Venice AI Image Style Variations Examples")
    print("=" * 50)
    total_paid = sum(len(section[4]) for section in SECTIONS) + 1
    if RUN_PAID:
        print(f"💸 VENICE_RUN_PAID_IMAGES=1 — this run makes {total_paid} paid generations.")
    else:
        print("🧪 Preview only (default). Set VENICE_RUN_PAID_IMAGES=1 to generate images.")

    written: list[Path] = []
    sections: list[tuple[str, list[tuple[str, bool, str | None]]]] = []
    async with VeniceClient() as client:
        # Pixel sizes are requested below, so pick among models sized by width/height.
        try:
            image_model = await client.models.resolve_image(
                prefer="cheapest", require_custom_size=True
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        print(f"📍 Using image model: {image_model}")
        try:
            presets = await list_presets(client)
        except VeniceError as e:
            print(f"❌ Could not list style presets: {e}")
            return 1
        if not presets:
            print("❌ client.image.list_styles() returned no presets")
            return 1

        if not RUN_PAID:
            spec = (await client.models.get(image_model)).model_spec
            print(f"\n🧾 This run would make {total_paid} paid generations:")
            for title, _prefix, subject, _seed, variants in SECTIONS:
                names = ", ".join(name for name, _modifier in variants)
                print(f"   - {title}: '{subject}' as {names}")
            print(f"   - style_preset: {presets[0]!r}")
            print(f"💰 Catalog price per image: {format_price(catalog_price_usd(spec))}")
            print("⏭️  Not generated: set VENICE_RUN_PAID_IMAGES=1 to generate (incurs this cost).")
            print("SKIPPED: VENICE_RUN_PAID_IMAGES is not set, so no image was generated")
            return SKIPPED

        for title, prefix, subject, seed, variants in SECTIONS:
            results = await render_section(
                client, image_model, title, prefix, subject, seed, variants, written
            )
            sections.append((title, results))
        sections.append(
            ("style_preset", [await render_preset(client, image_model, presets[0], written)])
        )

    total = sum(len(r) for _, r in sections)
    succeeded = sum(1 for _, r in sections for _, ok, _ in r if ok)
    failed = total - succeeded

    print("\n" + "=" * 50)
    print(f"📊 Overall: {succeeded}/{total} generations succeeded")

    if failed:
        print(f"\n❌ {failed} generation(s) failed:")
        for title, results in sections:
            for name, ok, err in results:
                if not ok:
                    print(f"   [{title}] {name}: {err}")
    else:
        print("\n✨ Style variation examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Artistic style variations (photorealistic, pixel art)")
        print("   - Different artistic mediums (pencil, acrylic)")
        print("   - Art movements and eras (Baroque, Pop Art)")
        print("   - Mood and atmosphere control")
        print("   - Color palette manipulation")
        print("   - A shared seed per section, so only the style modifier varies")
        print("   - Named presets: client.image.list_styles() and style_preset")

    print("\n📁 Files written by this run:")
    for path in written:
        print(f"   - {path.name}")
    if not written:
        print("   (none)")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
