#!/usr/bin/env python3
"""
Venice AI SDK - Image Generation Options
========================================

Shows the ``client.image`` options the other image examples leave out:

- ``format``, ``embed_exif_metadata``, ``cfg_scale``, ``safe_mode`` and the
  content-safety response headers (``response.content_safety_info``).
- ``enhance_prompt``, with the rewrite read back from ``response.enhanced_prompt``.
- ``disable_prompt_optimization_thinking`` and ``style_references``, each on a
  model whose catalog entry sets the matching flag.
- ``transparent_background`` on generation and on ``multi_edit``. No model may
  offer it, so ``NoMatchingModelError`` skips that section.
- ``client.image.submit()``, which builds an ``ImageJob``. The request fires in
  ``job.wait()``, so ``client.gather`` can run several with one concurrency cap.
- ``client.image.simple_generate()``, the OpenAI-compatible
  ``POST /images/generations``. Venice supports ``n=1`` only. ``quality`` is left
  unset here: the API spec calls it a compatibility field, while the SDK documents
  it for quality-aware models. Only a live comparison on such a model would settle
  which is right.

``lora_strength`` is not sent, because it applies only to models that use LoRAs and
the catalog does not say which do. ``steps`` is not sent either.

The request the server echoes in ``response.request["data"]`` is how a request
option is checked where the image cannot show it.

Generation costs real money, so by default this example resolves each section's
model, prints its catalog price (free) and exits 77 ("skipped"). Prompt enhancement
has no quote path: the guide states a flat $0.04 per applied rewrite, on top of the
image, charged only when a rewrite happens. Opt in to run all nine calls, writing
the outputs to ``examples/results/`` with an ``options_`` prefix::

    VENICE_RUN_PAID_IMAGES=1

Cost with the opt-in: the printed catalog prices plus that $0.04. On the prices listed
when this was written, about $0.25 to $0.40; the two ``transparent_background`` calls
depend on which models offer it.
"""

import asyncio
import os
import sys
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Any

from _helpers import (
    SKIPPED,
    alpha_coverage,
    catalog_price_usd,
    format_price,
    image_dimensions,
    save_and_report,
    spec_flag,
)

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError, detect_image_format
from venice_ai.types.api import ImageGenerationResponse

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RUN_PAID = os.environ.get("VENICE_RUN_PAID_IMAGES") == "1"

#: Paid calls each section makes.
PAID_CALLS = {
    "output_options": 1,
    "prompt_enhancement": 1,
    "skip_prompt_thinking": 1,
    "style_references": 1,
    "transparent_background": 2,
    "submit_jobs": 2,
    "openai_compatible": 1,
}

#: The guide's flat charge for an applied prompt rewrite. There is no quote for it.
ENHANCE_PROMPT_USD = 0.04

#: How much of a cutout must be see-through to count as having a transparent background.
MIN_CLEAR_FRACTION = 0.01


class NoModel(Exception):
    """No catalog model offers what a section needs."""


def echoed(response: ImageGenerationResponse) -> dict[str, Any]:
    """The request the server echoed back, or an empty dict if it sent none."""
    echo = response.request.get("data") if isinstance(response.request, dict) else None
    return echo if isinstance(echo, dict) else {}


def report_safety(response: ImageGenerationResponse) -> None:
    """Print the content-safety headers. Their absence is normal, not a failure."""
    info = response.content_safety_info
    if info is None:
        print("   ℹ️ No content-safety headers on this response")
        return
    print(f"   🛡️ blurred={info.is_blurred}, content violation={info.is_content_violation}")


async def cheapest_with_flag(client: VeniceClient, flag: str) -> str:
    """The cheapest image model whose catalog entry sets ``flag``.

    :raises NoModel: If no image model sets it.
    """
    specs = {m.id: m.model_spec for m in (await client.models.list(type="image")).data}
    tried: list[str] = []
    while True:
        try:
            model = await client.models.resolve_image(prefer="cheapest", exclude_models=tried)
        except NoMatchingModelError as e:
            raise NoModel(f"no image model sets {flag}") from e
        if spec_flag(specs.get(model), flag):
            return model
        tried.append(model)


async def output_options(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """``format``, ``embed_exif_metadata``, ``cfg_scale`` and ``safe_mode`` in one request."""
    response = await client.image.create(
        model=model,
        prompt="A red bicycle leaning against a white wall",
        width=512,
        height=512,
        format="png",
        embed_exif_metadata=True,
        cfg_scale=6.0,
        safe_mode=True,
        hide_watermark=True,
        return_binary=False,
    )
    data = response.bytes(0)
    save_and_report(data, RESULTS_DIR / "options_output", written)
    report_safety(response)
    has_exif = b"eXIf" in data or b"Exif" in data
    print(f"   ℹ️ EXIF marker in the bytes: {has_exif} (a model may not write one)")
    ok = True
    fmt = detect_image_format(data)[0]
    if fmt != "png":
        print(f"   ❌ Asked for format='png' and received {fmt}")
        ok = False
    echo = echoed(response)
    if echo.get("cfg_scale") != 6.0:
        print(f"   ❌ The server echoed cfg_scale={echo.get('cfg_scale')!r}")
        ok = False
    else:
        print("   ✔ The server echoed cfg_scale=6.0")
    return ok


async def prompt_enhancement(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """``enhance_prompt``: read the rewrite back from the response.

    Enhancement fails open: with no rewrite, the original prompt is used and the
    surcharge is not billed, so an absent rewrite is reported rather than failed.
    """
    response = await client.image.create(
        model=model,
        prompt="a cabin in the woods",
        width=512,
        height=512,
        enhance_prompt=True,
        hide_watermark=True,
        return_binary=False,
        timeout=180.0,  # The rewrite adds up to about 30s before generation starts.
    )
    save_and_report(response.bytes(0), RESULTS_DIR / "options_enhanced", written)
    rewrite = response.enhanced_prompt
    if rewrite:
        print(f"   ✅ Rewritten prompt: {rewrite}")
    else:
        print("   ℹ️ No rewrite was applied, so the original prompt was used and not surcharged")
    return True


async def skip_prompt_thinking(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """``disable_prompt_optimization_thinking`` on a model that advertises it."""
    response = await client.image.create(
        model=model,
        prompt="A paper boat on a still pond",
        disable_prompt_optimization_thinking=True,
        hide_watermark=True,
        return_binary=False,
    )
    save_and_report(response.bytes(0), RESULTS_DIR / "options_no_thinking", written)
    echo = echoed(response)
    if "disable_prompt_optimization_thinking" not in echo:
        print("   ℹ️ The server did not echo the flag, so whether it applied is not shown")
        return True
    if echo["disable_prompt_optimization_thinking"] is not True:
        print(
            "   ❌ The server echoed disable_prompt_optimization_thinking="
            f"{echo['disable_prompt_optimization_thinking']!r}"
        )
        return False
    print("   ✔ The server echoed the flag")
    return True


async def style_references(
    client: VeniceClient, model: str, reference: bytes, written: list[Path]
) -> bool:
    """``style_references``: guide a render's look with a base64 reference image."""
    spec = (await client.models.get(model)).model_spec
    entry: dict[str, Any] = {"image": b64encode(reference).decode("ascii")}
    constraints = getattr(spec, "constraints", None)
    if spec_flag(constraints, "supportsStyleReferenceStrength"):
        entry["strength"] = 0.5
    response = await client.image.create(
        model=model,
        prompt="A paper boat on a still pond",
        style_references=[entry],
        hide_watermark=True,
        return_binary=False,
    )
    save_and_report(response.bytes(0), RESULTS_DIR / "options_style_reference", written)
    print("   ℹ️ Nothing in the response measures style: compare it with the reference")
    return True


async def transparent_background(
    client: VeniceClient, image_model: str, edit_model: str, written: list[Path]
) -> bool:
    """``transparent_background`` on a generation and on a multi-image edit.

    Pixels are checked when Pillow is installed, since a header can declare an
    alpha channel on a fully opaque image.
    """
    response = await client.image.create(
        model=image_model,
        prompt="A single green apple, centered, on a plain background",
        format="png",
        transparent_background=True,
        hide_watermark=True,
        return_binary=False,
    )
    data = response.bytes(0)
    save_and_report(data, RESULTS_DIR / "options_transparent", written)
    edited = await client.image.multi_edit(
        prompt="Add a small green leaf to the apple, keeping the background transparent",
        model=edit_model,
        image=data,
        output_format="png",
        transparent_background=True,
    )
    save_and_report(edited, RESULTS_DIR / "options_transparent_edit", written)

    ok = True
    for label, image in (("generation", data), ("edit", edited)):
        coverage = alpha_coverage(image)
        if coverage is None:
            print(f"   ℹ️ {label}: Pillow is not installed, so the pixels were not checked")
            continue
        print(f"   🔍 {label}: {coverage[0]:.1%} see-through")
        if coverage[0] < MIN_CLEAR_FRACTION:
            print(f"   ❌ {label}: the background is not transparent")
            ok = False
    return ok


async def submit_jobs(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """``client.image.submit()``: two ``ImageJob`` objects run under one concurrency cap."""

    async def render(prompt: str, name: str) -> Path:
        async with await client.image.submit(
            model=model,
            prompt=prompt,
            width=512,
            height=512,
            format="png",
            hide_watermark=True,
        ) as job:
            path = RESULTS_DIR / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            return await job.download(path)

    outcomes = await client.gather(
        [
            render("A yellow kite over a green hill", "options_job_kite"),
            render("A blue teapot on a wooden shelf", "options_job_teapot"),
        ],
        max_concurrency=2,
    )
    ok = True
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            print(f"   ❌ {type(outcome).__name__}: {outcome}")
            ok = False
            continue
        written.append(outcome)
        data = outcome.read_bytes()
        print(f"   💾 {outcome.name}: {image_dimensions(data)}, {len(data)} bytes")
        if detect_image_format(data)[0] != "png":
            print(f"   ❌ {outcome.name} is not a PNG")
            ok = False
    return ok


async def openai_compatible(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """``simple_generate``: the OpenAI-compatible endpoint, which returns one image."""
    response = await client.image.simple_generate(
        prompt="A clean isometric illustration of a small workshop",
        model=model,
        n=1,
        size="512x512",
        response_format="b64_json",
        output_format="png",
        moderation="auto",
    )
    if len(response.data) != 1 or not response.data[0].b64_json:
        print(f"   ❌ Expected one base64 image, received {len(response.data)} entries")
        return False
    data = b64decode(response.data[0].b64_json)
    save_and_report(data, RESULTS_DIR / "options_openai_compatible", written)
    if detect_image_format(data)[0] != "png":
        print("   ❌ Asked for output_format='png'")
        return False
    return True


async def resolve_models(client: VeniceClient) -> dict[str, str | None]:
    """The model behind each role, ``None`` where the catalog has none. All free."""
    pixel_model = await client.models.resolve_image(prefer="cheapest", require_custom_size=True)
    models: dict[str, str | None] = {"pixel": pixel_model}
    try:
        models["thinking"] = await cheapest_with_flag(client, "supportsOptimizePromptThinking")
    except NoModel:
        models["thinking"] = None
    try:
        models["style"] = await cheapest_with_flag(client, "supportsStyleReferences")
    except NoModel:
        models["style"] = None
    try:
        models["transparent"] = await client.models.resolve_image(
            require_transparent_background=True, prefer="cheapest"
        )
        models["transparent_edit"] = await client.models.resolve_inpaint(
            require_combine_images=True, require_transparent_background=True, prefer="cheapest"
        )
    except NoMatchingModelError:
        models["transparent"] = models["transparent_edit"] = None
    return models


async def print_plan(client: VeniceClient, models: dict[str, str | None]) -> None:
    """Print each role's model and catalog price, then the guide's enhancement charge."""
    print("🧾 Paid calls by section:")
    for name, count in PAID_CALLS.items():
        print(f"   - {name}: {count}")
    for role, model in models.items():
        if model is None:
            print(f"💰 {role}: no model in the catalog")
            continue
        spec = (await client.models.get(model)).model_spec
        print(f"💰 {role}: {model}, catalog {format_price(catalog_price_usd(spec))}")
    print(f"💰 enhance_prompt adds ${ENHANCE_PROMPT_USD:.2f} per applied rewrite (no quote path)")


async def main() -> int:
    """Run the option demos.

    Returns ``0`` if every section that ran succeeded, ``1`` if any failed, and ``77``
    if ``VENICE_RUN_PAID_IMAGES`` is unset or no image model is sized by width/height.
    """
    print("🚀 Venice AI Image Generation Options")
    print("=" * 50)

    written: list[Path] = []
    results: list[tuple[str, bool]] = []
    async with VeniceClient() as client:
        try:
            models = await resolve_models(client)
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        await print_plan(client, models)
        if not RUN_PAID:
            print("SKIPPED: VENICE_RUN_PAID_IMAGES is not set, so no image was generated")
            return SKIPPED

        pixel = models["pixel"] or ""

        async def run(name: str, coro: Any) -> None:
            print(f"\n🎨 {name}")
            print("-" * 40)
            try:
                results.append((name, await coro))
            except (VeniceError, ValueError) as e:
                print(f"   ❌ {type(e).__name__}: {e}")
                results.append((name, False))

        await run("output_options", output_options(client, pixel, written))
        await run("prompt_enhancement", prompt_enhancement(client, pixel, written))
        if models["thinking"]:
            await run(
                "skip_prompt_thinking", skip_prompt_thinking(client, models["thinking"], written)
            )
        else:
            print("Section skipped: skip_prompt_thinking: no image model advertises it")
        if models["style"]:
            if written:
                # The first image this run wrote is the reference, so it costs nothing extra.
                reference = written[0].read_bytes()
                await run(
                    "style_references",
                    style_references(client, models["style"], reference, written),
                )
            else:
                print("Section skipped: style_references: no earlier image to use as a reference")
        else:
            print("Section skipped: style_references: no image model advertises them")
        if models["transparent"] and models["transparent_edit"]:
            await run(
                "transparent_background",
                transparent_background(
                    client, models["transparent"], models["transparent_edit"], written
                ),
            )
        else:
            print("Section skipped: transparent_background: the catalog has no model for it")
        await run("submit_jobs", submit_jobs(client, pixel, written))
        await run("openai_compatible", openai_compatible(client, pixel, written))

    failed = [name for name, ok in results if not ok]
    print(f"\n📁 Files written by this run ({len(written)}):")
    for path in written:
        print(f"   - {path.name}")
    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1
    print("\n✨ Image generation options completed!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
