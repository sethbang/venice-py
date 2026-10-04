#!/usr/bin/env python3
"""
Venice AI SDK - Vision / Multimodal Chat Completions
=====================================================

This example demonstrates how to send images to vision-capable models using
the Venice AI SDK. Every image is encoded as a base64 ``data:`` URI before
sending (see "Why we pre-fetch and resize the source images" below) — learn
how to analyze a single image, send image bytes you produced yourself, compare
multiple images, and use different analysis prompt strategies.

Why we pre-fetch and resize the source images
---------------------------------------------
Some Venice vision models accept smaller maximum input dimensions than others
and answer a multi-megapixel image with HTTP 500 ``"Inference processing
failed"``. To make this example work with every vision-capable model, we
download the source images once and run them through
``fit_image_bytes(max_dim=512)`` before encoding as base64. That keeps every
image within the tighter limits without excluding any model.

Each section checks that the answer is complete (``finish_reason``, and a
completion count below the cap, since some models report a cut-off answer as
``"stop"``) and that it mentions what is actually in the image, so a model that
ignored the image or was cut off counts as a failure. Requests leave Venice's
own system prompt out; the image and the question are all the model needs.
The script exits 77 if the catalog has no vision model, and a section whose
model is missing (several images in one message) prints ``Section skipped:``.

Requirements:
    - Pillow, for ``fit_image_bytes()`` and the locally drawn image
      (``pip install 'venice-py[cli]'`` or ``pip install Pillow``). Without it
      the script prints ``SKIPPED:`` and exits 77.
"""

import asyncio
import base64
import importlib.util
import io
import sys

import aiohttp

from venice_ai import (
    NoMatchingModelError,
    VeniceClient,
    VeniceError,
    detect_image_format,
    fit_image_bytes,
)
from venice_ai.types.api import SystemMessage, UserMessage
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import VeniceParameters

# ---------------------------------------------------------------------------
# Public image URLs (Wikimedia Commons — stable, direct, no signed redirects).
# ---------------------------------------------------------------------------
# Wikimedia originals are full-resolution photographs (multi-megabyte / multi-
# megapixel). We download once and run them through fit_image_bytes() before
# sending so every Venice vision model, including those with a tighter
# image-size limit, accepts them.
#
# IMAGE_1: tabby cat on a stone wall  → expect "cat", "tabby", "feline"
# IMAGE_2: tri-color beagle on a leash → expect "dog", "beagle", "canine"
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
SAMPLE_IMAGE_URL_2 = "https://upload.wikimedia.org/wikipedia/commons/5/55/Beagle_600.jpg"


async def _fetch_and_fit_data_uri(
    session: aiohttp.ClientSession, url: str, *, max_dim: int = 512
) -> str:
    """Fetch *url*, fit to ``max_dim``, return a base64 ``data:`` URI.

    Sets a friendly User-Agent — Wikimedia and many other CDNs reject the
    default aiohttp UA.
    """
    headers = {"User-Agent": "venice-py-sdk-examples/1.0 (vision example)"}
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        raw = await resp.read()
    fitted = fit_image_bytes(raw, max_dim=max_dim)
    _ext, mime = detect_image_format(fitted)
    return f"data:{mime};base64,{base64.b64encode(fitted).decode('ascii')}"


# Room for a full description. The sections resolve with
# exclude_reasoning=True, so none of this budget goes to hidden reasoning.
MAX_COMPLETION_TOKENS = 1024

NO_VENICE_PROMPT = VeniceParameters(include_venice_system_prompt=False)


class VisionCheckError(Exception):
    """A vision answer was truncated, empty, or didn't describe the image."""


def _check_complete(response: ChatCompletionResponse, label: str) -> str:
    """Return the answer text, raising if it is empty or was cut off."""
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    content = response.text or ""
    print(f"   🏁 Finish reason: {finish_reason} ({used} of {MAX_COMPLETION_TOKENS} tokens)")
    if finish_reason == "length" or used is None or used >= MAX_COMPLETION_TOKENS:
        raise VisionCheckError(f"{label}: the answer may be cut off by max_completion_tokens")
    if not content.strip():
        raise VisionCheckError(f"{label}: the model returned no text")
    return content


def _check_keywords(content: str, keywords: list[str], label: str) -> None:
    """Raise if the answer mentions none of the expected keywords."""
    lowered = content.lower()
    if not any(kw.lower() in lowered for kw in keywords):
        raise VisionCheckError(f"{label}: the answer mentions none of {keywords}")


# ============================================================================
# 1. Basic Image Analysis
# ============================================================================


async def basic_image_analysis(cat_uri: str):
    print("👁️  Basic Image Analysis")
    print("-" * 40)

    async with VeniceClient() as client:
        vision_model = await client.models.resolve_chat(
            require_vision=True, exclude_reasoning=True, prefer="cheapest"
        )
        print(f"📍 Using vision model: {vision_model}")

        # Pattern 1: fluent builder — most readable for incremental construction.
        # cat_uri is a fitted base64 data URI prepared at startup.
        message = (
            UserMessage.builder()
            .text("What do you see in this image? Describe it in one paragraph.")
            .image(cat_uri)
            .build()
        )

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[message],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.5,
        )

        content = _check_complete(response, "basic_image_analysis")
        print(f"🤖 Assistant: {content}")

        if response.usage:
            usage = response.usage
            print("\n📊 Token Usage:")
            print(f"   Input tokens:  {usage.prompt_tokens}")
            print(f"   Output tokens: {usage.completion_tokens}")
            print(f"   Total tokens:  {usage.total_tokens}")

        # SAMPLE_IMAGE_URL is a tabby cat on a stone wall
        _check_keywords(content, ["cat", "tabby", "feline", "kitten"], "basic_image_analysis")


# ============================================================================
# 2. Base64 Image Input (self-contained)
# ============================================================================

# Text drawn into the local image; only a model that read the pixels can repeat it.
DRAWN_TEXT = "VENICE 2718"


def _draw_sample_png() -> bytes:
    """Draw a small PNG locally: a red circle on a blue background, plus a caption."""
    from PIL import Image, ImageDraw, ImageFont  # optional extra; main() checks for it

    image = Image.new("RGB", (512, 512), (40, 90, 200))
    draw = ImageDraw.Draw(image)
    draw.ellipse((156, 60, 356, 260), fill=(220, 30, 30))
    draw.text(
        (256, 380),
        DRAWN_TEXT,
        fill=(255, 255, 255),
        anchor="mm",
        font=ImageFont.load_default(size=56),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def base64_image_input():
    print("\n🖼️  Base64 Image Input (drawn locally → analysed)")
    print("-" * 40)

    # --- Step 1: Make image bytes in-process so the demo is self-contained ---
    png_bytes = _draw_sample_png()
    print(f"🎨 Drew a 512x512 PNG locally ({len(png_bytes)} bytes)")

    async with VeniceClient() as client:
        # --- Step 2: Send the bytes as a base64 data URI ---
        vision_model = await client.models.resolve_chat(
            require_vision=True, exclude_reasoning=True, prefer="cheapest"
        )
        print(f"📍 Analysing with vision model: {vision_model}")

        # detect_image_format() reads the magic bytes, so the data URI carries
        # the right MIME type whatever produced the image.
        _ext, mime_type = detect_image_format(png_bytes)
        data_uri = f"data:{mime_type};base64,{base64.b64encode(png_bytes).decode('ascii')}"
        message = (
            UserMessage.builder()
            .text(
                "Describe this image in two sentences: its shapes and colors, and the exact "
                "text written on it."
            )
            .image(data_uri)
            .build()
        )

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[message],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.2,
        )

        content = _check_complete(response, "base64_image_input")
        print(f"🤖 Assistant: {content}")

        # The image is a red circle on blue with a caption only the pixels carry.
        _check_keywords(content, ["circle", "circular", "round"], "base64_image_input (shape)")
        _check_keywords(content, ["red"], "base64_image_input (color)")
        _check_keywords(content, ["2718"], "base64_image_input (text)")


# ============================================================================
# 3. Image Comparison
# ============================================================================


async def image_comparison(cat_uri: str, dog_uri: str):
    print("\n🔍 Image Comparison (two images)")
    print("-" * 40)

    async with VeniceClient() as client:
        # Two images in one message need a model that accepts several images.
        vision_model = await client.models.resolve_chat(
            require_vision=True,
            require_multiple_images=True,
            exclude_reasoning=True,
            prefer="cheapest",
        )
        print(f"📍 Using vision model: {vision_model}")

        # Builder makes multi-image messages especially compact
        message = (
            UserMessage.builder()
            .text(
                "I'm showing you two images. "
                "Please compare them in one short paragraph: what are the main "
                "differences and similarities?"
            )
            .image(cat_uri)
            .image(dog_uri)
            .build()
        )

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[
                SystemMessage(
                    content="You are an observant image analyst. Be concise but thorough.",
                ),
                message,
            ],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.4,
        )

        content = _check_complete(response, "image_comparison")
        print(f"🤖 Comparison: {content}")

        # A real comparison mentions what is in BOTH images: the cat from
        # image 1 and the dog from image 2.
        _check_keywords(content, ["cat", "feline", "tabby", "kitten"], "image_comparison (1)")
        _check_keywords(content, ["dog", "beagle", "canine", "puppy"], "image_comparison (2)")


# ============================================================================
# 4. Detailed Analysis Prompts
# ============================================================================


async def detailed_analysis_prompts(cat_uri: str):
    print("\n📝 Detailed Analysis Prompts")
    print("-" * 40)

    # Various prompt strategies for the same image
    strategies = [
        ("Description", "Describe exactly what you see in this image in 3-4 sentences."),
        (
            "Spatial reasoning",
            "Describe the spatial layout of the image: foreground, middle-ground, and "
            "background, one sentence each.",
        ),
    ]

    async with VeniceClient() as client:
        vision_model = await client.models.resolve_chat(
            require_vision=True, exclude_reasoning=True, prefer="cheapest"
        )
        print(f"📍 Using vision model: {vision_model}")

        # Aggregate descriptions for a single keyword check across strategies.
        all_text_parts: list[str] = []

        for label, prompt_text in strategies:
            print(f"\n🔎 Strategy: {label}")
            message = UserMessage.builder().text(prompt_text).image(cat_uri).build()

            response = await client.chat.completions.create(
                model=vision_model,
                messages=[message],
                venice_parameters=NO_VENICE_PROMPT,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.3,
            )

            content = _check_complete(response, f"detailed_analysis_prompts ({label})")
            print(f"   🤖 {content}")
            all_text_parts.append(content)

        # Verify at least one strategy mentioned something cat-like.
        _check_keywords(
            "\n".join(all_text_parts),
            ["cat", "tabby", "feline", "kitten"],
            "detailed_analysis_prompts",
        )


# ============================================================================
# 5. Model Discovery — list vision-capable models
# ============================================================================


async def model_discovery():
    print("\n🗺️  Vision Model Discovery")
    print("-" * 40)

    from venice_ai.types.api import TextModelSpec

    async with VeniceClient() as client:
        # Vision is a chat/text capability — restrict the listing to text
        # models so ``capabilities`` is typed.
        all_models = await client.models.list(type="text")

        vision_models = []
        for model in all_models.data:
            spec = model.model_spec
            if not isinstance(spec, TextModelSpec):
                continue
            caps = spec.capabilities
            if caps is not None and caps.supportsVision:
                vision_models.append(model.id)

        print(f"✅ Found {len(vision_models)} vision-capable model(s):")
        for m in sorted(vision_models):
            print(f"   • {m}")

        # prefer="cheapest" ranks strictly by price, so the cheapest vision
        # model may be a reasoning model. The sections above add
        # exclude_reasoning=True so the whole token budget goes to the answer.
        cheapest_vision = await client.models.resolve_chat(require_vision=True, prefer="cheapest")
        cheapest_direct = await client.models.resolve_chat(
            require_vision=True, exclude_reasoning=True, prefer="cheapest"
        )
        print(f"\n⭐ Cheapest vision model (via resolve_chat): {cheapest_vision}")
        print(f"⭐ Cheapest vision model that answers directly: {cheapest_direct}")


# ============================================================================
# Main
# ============================================================================


async def main() -> int:
    print("🚀 Venice AI Vision / Multimodal Examples")
    print("=" * 50)

    if importlib.util.find_spec("PIL") is None:
        # fit_image_bytes() and the locally drawn image both need Pillow.
        print(
            "SKIPPED: Pillow is not installed (pip install 'venice-py[cli]' or pip install Pillow)"
        )
        return 77

    # Pre-fetch and fit the source images once. fit_image_bytes(max_dim=512)
    # ensures the resulting data URIs are accepted by every Venice vision
    # model, including those with tighter input size limits.
    print("\n⬇️  Pre-fetching source images and fitting to 512 px...")
    async with aiohttp.ClientSession() as session:
        cat_uri, dog_uri = await asyncio.gather(
            _fetch_and_fit_data_uri(session, SAMPLE_IMAGE_URL),
            _fetch_and_fit_data_uri(session, SAMPLE_IMAGE_URL_2),
        )
    print(f"   ✅ cat ({len(cat_uri)} chars), dog ({len(dog_uri)} chars)")

    # Every section needs a vision model; without one there is nothing to show.
    async with VeniceClient() as client:
        try:
            await client.models.resolve_chat(require_vision=True, exclude_reasoning=True)
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no non-reasoning vision model ({e})")
            return 77

    sub_examples = [
        ("basic_image_analysis", basic_image_analysis(cat_uri)),
        ("base64_image_input", base64_image_input()),
        ("image_comparison", image_comparison(cat_uri, dog_uri)),
        ("detailed_analysis_prompts", detailed_analysis_prompts(cat_uri)),
        ("model_discovery", model_discovery()),
    ]

    results: list[tuple[str, bool | None]] = []
    for name, coro in sub_examples:
        try:
            await coro
            results.append((name, True))
        except NoMatchingModelError as e:
            # The catalog has no model for this section (checked before VeniceError,
            # which it subclasses): a skip, not a failure.
            print(f"\nSection skipped: '{name}' has no matching model ({e})")
            results.append((name, None))
        except (VeniceError, VisionCheckError) as e:
            print(f"\n❌ Sub-example '{name}' failed: {e}")
            results.append((name, False))

    # Summary
    succeeded = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if ok is False)
    total = len(results)
    print("\n" + "=" * 50)
    print(f"{'❌' if failed else '✨'} {succeeded}/{total} vision examples completed")
    for name, ok in results:
        marker = {True: "✅", False: "❌", None: "⏭️ "}[ok]
        print(f"   {marker} {name}")
    if failed:
        return 1

    print("\n💡 Key concepts demonstrated:")
    print("   - fit_image_bytes() to keep images within all models' size limits")
    print("   - Sending images as base64 data URIs, including locally made bytes")
    print("   - Comparing multiple images in one message")
    print("   - Different analysis prompt strategies")
    print("   - Discovering vision-capable models")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)
