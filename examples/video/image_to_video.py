#!/usr/bin/env python3
"""
Venice AI SDK - Image-to-Video Generation
==========================================

This example demonstrates how to animate still images into videos using the
Venice AI SDK's **image-to-video (I2V)** capabilities. You supply an
``image_url`` pointing to a publicly accessible reference image (or a
``data:`` URI):

    1. ``client.models.resolve_cheapest_video(video_type="image-to-video")``
       — quote every plain I2V model (free) at its own cheapest valid request
       and return the lowest, with the ``request_params`` it was quoted with.
       Reference-to-video, transition, first/last-frame and multi-angle
       models share ``model_type="image-to-video"`` but ignore or need more
       than one start image; the resolver leaves them out unless you pass
       ``input_mode=`` (``video_input_mode()`` classifies a model).
       ``client.models.get(model)`` exposes any model's ``constraints``.
    2. ``client.video.quote(...)`` — estimate the cost.
    3. ``client.video.run(..., image_url=...)`` — submit and receive a
       :class:`venice_ai.VideoJob`.
    4. ``async with job:`` — releases the finished job's stored media on a clean
       exit; an error raised through the block keeps it.
    5. ``await job.wait(on_progress=...)`` — polls until completion (raises
       :class:`VideoGenerationError` on failure or ``TimeoutError`` on
       poll exhaustion).
    6. ``await job.download(path, status)`` — saves to disk via the SDK's
       managed HTTP session.

I2V models have their own constraints, separate from the text-to-video
models: durations and resolutions vary per model, and most list no aspect
ratios, leaving the output shape to the model (it may differ from the source
image's). Build each request from the resolved model's constraints rather
than hardcoding values.

The ``prompt`` describes the **desired motion or animation** — not the image
content itself. For example, given a photo of a mountain lake you might
prompt *"gentle ripples spread across the water as a breeze picks up"*.

When the catalog has no model a section asks for, the resolver raises
:class:`venice_ai.NoMatchingModelError` and the section is skipped; when
models exist but every quote fails it raises
:class:`venice_ai.ModelQuotesUnavailableError`, and the section fails. A model
whose quote fails while others succeed is left out of the comparison; it is
listed, and a paid run declines to queue, since the pick may not be the cheapest.

Generation costs real money, so by default this example only quotes (free),
skips the generation and exits 77 ("skipped"). Opt in to queue, wait for and
download one short clip on the cheapest I2V model at its shortest duration
and lowest resolution, and check the clip against the request::

    VENICE_RUN_PAID_VIDEO=1

A job quoted above ``VIDEO_CEILING_USD`` ($0.50) is refused. To spend more, raise it
with ``VENICE_VIDEO_MAX_USD``.

A queue call that ends in a 5xx, a read timeout or a dropped connection may
still have queued and billed a job, and the example says so. A file that is not
a video keeps the job's stored copy. A clip that is the wrong length or size is
still a delivered video, so it is saved and released, and the example exits 1.
"""

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from _helpers import (
    NOT_PAID,
    Outcome,
    ceiling_refusal,
    duration_seconds,
    exit_code,
    read_video_listing,
    report_error,
    report_uncertain_queue,
    resolution_height,
    unrecognised_durations,
    url_serves,
    video_constraints,
    wait_and_check,
)

from venice_ai import (
    NoMatchingModelError,
    VeniceClient,
    cheapest_video_params,
    video_input_mode,
)
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.models import VideoModelConstraints

RUN_PAID = os.environ.get("VENICE_RUN_PAID_VIDEO") == "1"

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/video/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# A publicly-accessible 800x600 sample image (Picsum Photos CDN: a forest on
# a rocky shore). Uses the direct fastly CDN URL to avoid 302 redirects that
# the Venice API may not follow.
SAMPLE_IMAGE_URL = (
    "https://fastly.picsum.photos/id/10/800/600.jpg"
    "?hmac=9u_ZYBasFb_VEVrBgjTZor_IfBxtpq9zl_CjKJr7-cs"
)

# Files written by this run, listed at the end.
WRITTEN: list[Path] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _generate(
    client: VeniceClient,
    model: str,
    params: dict[str, Any],
    prompt: str,
    output_path: Path,
) -> Outcome:
    """Submit one I2V job and wait for it; any failure returns ``False``.

    Returns the skip reason without queueing when ``VENICE_RUN_PAID_VIDEO``
    is unset.
    """
    if not RUN_PAID:
        print("⏭️  Not queued: set VENICE_RUN_PAID_VIDEO=1 to generate (incurs the quoted cost).")
        return NOT_PAID
    if not await asyncio.to_thread(url_serves, SAMPLE_IMAGE_URL, "image/"):
        return False
    try:
        try:
            job = await client.video.run(
                model=model,
                prompt=prompt,
                image_url=SAMPLE_IMAGE_URL,
                **params,
            )
        except (VeniceError, ValueError) as e:
            if not report_uncertain_queue(e):
                report_error(e)
            return False
        print(f"📨 Queued — queue_id: {job.queue_id}")
        # Most I2V models list no aspect ratio and follow the source image's. The
        # check covers what was sent: duration and resolution, and aspect ratio when
        # the model lists one.
        return await wait_and_check(
            job, output_path, params, WRITTEN, expect_audio=params.get("audio")
        )
    except (VeniceError, ValueError) as e:
        report_error(e)
        return False


# ---------------------------------------------------------------------------
# Section 1 — Basic image-to-video generation
# ---------------------------------------------------------------------------


async def basic_image_to_video(client: VeniceClient) -> Outcome:
    """Animate a reference image into a short video clip.

    Returns ``True`` on success, ``False`` if the quotes, generation or the
    save step failed, and the reason when skipped (``VENICE_RUN_PAID_VIDEO``
    unset, or no plain image-to-video model listed).
    """
    print("🖼️  Basic Image-to-Video Generation")
    print("-" * 40)

    try:
        # Quotes every plain image-to-video model (one start image; no
        # reference, transition, first/last-frame or multi-angle models) at
        # its own shortest duration and lowest resolution, audio off where
        # configurable, and returns the lowest together with the parameters
        # that price was quoted for.
        pick = await client.models.resolve_cheapest_video(video_type="image-to-video")
    except NoMatchingModelError as e:
        return f"no plain image-to-video model to generate with ({e})"
    except (VeniceError, ValueError) as e:
        report_error(e)
        return False

    prompt = (
        "The scene slowly comes to life — colours gently shift and a "
        "soft breeze seems to move through the frame"
    )
    print(f"📍 Cheapest of {len(pick.all_quotes)} quoted models: {pick.model}")
    print(f"📐 Request parameters it was quoted with: {pick.request_params}")
    print(f"🖼️  Image URL: {SAMPLE_IMAGE_URL[:80]}…")
    print(f"📝 Prompt (motion): {prompt}")
    print(f"💵 Quoted cost: ${pick.quote_usd:.4f}")
    print("ℹ️  image_url must be publicly accessible (http://, https://, or data: URI)")
    for model_id, error in pick.quote_failures.items():
        print(f"⚠️  Not compared: {model_id} (quote failed: {error})")
    if RUN_PAID:
        refusal = ceiling_refusal(pick.quote_usd)
        if refusal is not None:
            return refusal
    if pick.quote_failures and RUN_PAID:
        print(
            f"❌ Not queued: {len(pick.quote_failures)} models could not be priced, and any of them "
            "may be cheaper than this pick."
        )
        return False

    return await _generate(
        client, pick.model, pick.request_params, prompt, RESULTS_DIR / "i2v_basic.mp4"
    )


# ---------------------------------------------------------------------------
# Section 2 — Cost estimation for I2V
# ---------------------------------------------------------------------------


async def cost_estimation(client: VeniceClient) -> Outcome:
    """Get price quotes for image-to-video generation (free).

    Every duration and resolution quoted comes from the model's own
    constraints, so each quote is expected to succeed; any failure fails the
    section.
    """
    print("\n💰 I2V Cost Estimation with quote()")
    print("-" * 40)

    try:
        model = await client.models.resolve_video(video_type="image-to-video")
    except NoMatchingModelError as e:
        return f"no image-to-video model to quote ({e})"
    try:
        constraints = await video_constraints(client, model)
        base = cheapest_video_params(constraints)
    except (VeniceError, ValueError) as e:
        report_error(e)
        return False

    print(f"📍 Model: {model}")
    print(f"   durations={constraints.durations} resolutions={constraints.resolutions}")

    ok = True

    async def quote(label: str, **overrides: object) -> None:
        nonlocal ok
        try:
            resp = await client.video.quote(model=model, **{**base, **overrides})
            print(f"   💵 {label}: ${float(resp.quote):.4f}")
        except VeniceError as e:
            print(f"   ❌ {label}: {e}")
            ok = False

    print("\n📊 By duration (lowest resolution):")
    for duration in constraints.durations:
        if duration_seconds(duration) is not None:
            await quote(f"{duration:>4}", duration_seconds=duration)

    print(f"\n📊 By resolution ({base['duration_seconds']}):")
    for resolution in sorted(constraints.resolutions, key=resolution_height):
        await quote(f"{resolution:>5}", resolution=resolution)

    return ok


# ---------------------------------------------------------------------------
# Section 3 — Model discovery for I2V
# ---------------------------------------------------------------------------


async def model_discovery(client: VeniceClient) -> Outcome:
    """Browse image-to-video models and narrow them with resolver filters (free)."""
    print("\n🔍 I2V Model Discovery")
    print("-" * 40)

    try:
        listing = read_video_listing((await client.models.list(type="video")).data)
        if listing.unreadable:
            # A listed model whose constraints cannot be read is a parsing
            # failure; dropping it could make a kind look absent.
            print(f"❌ {len(listing.unreadable)} listed video models could not be read:")
            for line in listing.unreadable[:5]:
                print(f"   - {line}")
            return False
        i2v: list[tuple[str, VideoModelConstraints]] = []
        modes: Counter[str] = Counter()
        for model_id, spec, constraints in listing.models:
            if spec.offline or constraints.model_type != "image-to-video":
                continue
            # The catalog types every image-conditioned model "image-to-video";
            # video_input_mode() tells the plain ones from reference,
            # transition, first/last-frame and multi-angle models.
            mode = video_input_mode({"id": model_id, "name": spec.name or ""})
            modes[mode] += 1
            if mode == "image":
                i2v.append((model_id, constraints))
        if not i2v:
            return (
                "the catalog lists no online image-to-video model that takes a single start image"
            )
        if odd := [f"{m}: {d}" for m, c in i2v if (d := unrecognised_durations(c))]:
            print(f"❌ Durations that cannot be read: {'; '.join(odd[:5])}")
            return False
        breakdown = ", ".join(f"{mode}={count}" for mode, count in modes.most_common())
        print(f"📚 model_type='image-to-video' models online by input mode: {breakdown}")
        print(f"   {len(i2v)} take a single start image. First five:")
        for model_id, c in i2v[:5]:
            ratios = c.aspect_ratios or "chosen by the model"
            print(
                f"   - {model_id}: durations={c.durations} resolutions={c.resolutions} "
                f"aspect_ratios={ratios}"
            )

        # Filter on a duration/resolution pair the listing actually offers:
        # the one the most plain I2V models list.
        pairs = Counter(
            (d, r)
            for _, c in i2v
            for d in c.durations
            if duration_seconds(d) is not None
            for r in c.resolutions
        )
        if not pairs:
            # Every duration was read, so this is the catalog, not parsing:
            # the plain I2V models choose their own length or list no
            # resolution tiers.
            return (
                "no plain image-to-video model lists both a fixed duration and a "
                "resolution to filter on"
            )
        (duration, resolution), count = pairs.most_common(1)[0]

        print("\n🎯 Resolver filters narrow the choice:")
        default = await client.models.resolve_video(video_type="image-to-video")
        print(f"   resolve_video(video_type='image-to-video') -> {default}")
        print(f"   ({duration} at {resolution} is listed by {count} plain I2V models)")
        filtered = await client.models.resolve_video(
            video_type="image-to-video",
            require_duration=duration,
            require_resolution=resolution,
        )
        print(
            f"   ... require_duration={duration!r}, require_resolution={resolution!r} -> {filtered}"
        )
        quote = await client.video.quote(
            model=filtered, duration_seconds=duration, resolution=resolution
        )
        print(f"   💵 {duration} at {resolution} on {filtered}: ${float(quote.quote):.4f}")
        return True

    except (VeniceError, ValueError) as e:
        report_error(e)
        return False


def _print_prompt_guidance() -> None:
    """Prompt-writing guidance (no API call).

    The prompt describes the *motion*, so the same image animates very
    differently under different prompts. Each variation is the basic
    section's ``run()`` call with another ``prompt``; queueing them would
    cost one more clip each, so they are shown, not generated.
    """
    print("\n🎨 Prompt guidance: same image, different motion")
    for label, prompt in (
        ("Gentle", "A gentle breeze blows through the scene, creating soft, calming motion"),
        (
            "Dramatic",
            "A dramatic storm builds — wind gusts intensify, trees sway violently, "
            "waves crash and the light shifts rapidly",
        ),
    ):
        print(f"   {label}: client.video.run(..., image_url=..., prompt={prompt!r})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all image-to-video examples.

    Returns ``1`` when a section failed, ``77`` (skipped) when the basic
    generation did not run, and ``0`` otherwise; any other skipped section is
    listed with its reason.
    """
    print("🚀 Venice AI Image-to-Video Examples")
    print("=" * 60)
    if RUN_PAID:
        print("💸 VENICE_RUN_PAID_VIDEO=1 — generation sections will queue paid jobs.")
    else:
        print("🧪 Quote only (default). Set VENICE_RUN_PAID_VIDEO=1 to generate videos.")

    async with VeniceClient() as client:
        results: list[tuple[str, Outcome]] = [
            ("basic_image_to_video", await basic_image_to_video(client)),
            ("cost_estimation", await cost_estimation(client)),
            ("model_discovery", await model_discovery(client)),
        ]

    if WRITTEN:
        print(f"\n📁 Files written by this run in {RESULTS_DIR}/:")
        for path in WRITTEN:
            print(f"   - {path.name}")

    _print_prompt_guidance()
    # Generation is this example's core feature; quotes alone do not show it.
    code = exit_code(results, core="basic_image_to_video")
    if code:
        return code

    print("\n✨ Image-to-video examples completed!")
    # Reaching this line means the basic generation passed; the rest may have been skipped.
    passed = {name for name, outcome in results if outcome is True}
    print("\n💡 Key concepts demonstrated:")
    print("   - image_url switches run() to I2V mode")
    print("   - resolve_cheapest_video(): cheapest plain I2V model plus its quoted parameters")
    if "model_discovery" in passed:
        print("   - video_input_mode(): plain I2V vs reference / transition models")
    print("   - Prompt describes desired motion, not image content")
    print("   - Clip checked against its request: duration and resolution")
    if "cost_estimation" in passed:
        print("   - Cost estimation with quote()")
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
