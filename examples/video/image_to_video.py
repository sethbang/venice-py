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
    4. ``async with job:`` — releases the finished job's stored media on exit.
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
:class:`venice_ai.ModelQuotesUnavailableError`, and the section fails.

Generation costs real money, so by default this example only quotes (free),
skips the generation and exits 77 ("skipped"). Opt in to queue, wait for and
download one short clip on the cheapest I2V model at its shortest duration
and lowest resolution, and check the clip against the request::

    VENICE_RUN_PAID_VIDEO=1
"""

import asyncio
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from venice_ai import (
    APITimeoutError,
    ModelQuotesUnavailableError,
    NoMatchingModelError,
    VeniceClient,
    VideoGenerationError,
    VideoJob,
    cheapest_video_params,
    video_input_mode,
)
from venice_ai.exceptions import AuthenticationError, PermissionDeniedError, VeniceError
from venice_ai.types.api.models import VideoModelConstraints, VideoModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    NOT_PAID,
    Outcome,
    check_video,
    duration_seconds,
    exit_code,
    print_progress,
    read_video_listing,
    unrecognised_durations,
)

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


def _resolution_height(value: str) -> int:
    """Sort key for resolution tiers: ``"720p"`` -> 720, ``"4k"`` -> 2160."""
    tier = value.strip().lower()
    if tier.endswith("k") and tier[:-1].isdigit():
        return {2: 1440, 4: 2160, 8: 4320}.get(int(tier[:-1]), 0)
    digits = tier.removesuffix("p")
    return int(digits) if digits.isdigit() else 0


async def _constraints(client: VeniceClient, model: str) -> VideoModelConstraints:
    """Return the video constraints the catalog lists for ``model``."""
    entry = await client.models.get(model)
    spec = entry.model_spec
    if not isinstance(spec, VideoModelSpec) or spec.constraints is None:
        raise ValueError(f"{model} has no video constraints in the catalog")
    return spec.constraints


async def _wait_and_save(job: VideoJob, output_path: Path, request: dict[str, Any]) -> bool:
    """Wait for completion, save to disk and check the clip against ``request``."""
    print(f"⏳ Polling for completion (queue_id={job.queue_id}) ...")
    try:
        status = await job.wait(on_progress=print_progress)
        saved = await job.download(output_path, status)
    except VideoGenerationError as e:
        code = f" (code={e.error_code})" if e.error_code else ""
        print(f"❌ Generation failed: {e}{code}")
        return False
    except TimeoutError as e:
        # wait() ran out of polls; the job is still queued or running.
        print(f"❌ {e} (queue_id={job.queue_id})")
        return False
    except APITimeoutError as e:
        # One status or download request timed out; the job itself may still
        # finish and be billed, so keep its queue_id.
        print(f"❌ A status or download request timed out (queue_id={job.queue_id}): {e}")
        return False

    data = saved.read_bytes()
    WRITTEN.append(saved)
    print(f"💾 Saved to {saved} ({len(data):,} bytes)")
    # Most I2V models list no aspect ratio and follow the source image's, so
    # the check covers the duration and resolution that were sent.
    if not check_video(data, request, expect_audio=request.get("audio")):
        print("❌ The clip does not match the request")
        return False
    print("✅ Video ready and matches the request")
    return True


def _report_error(e: Exception) -> None:
    if isinstance(e, ModelQuotesUnavailableError):
        # Models matched but none could be priced: an outage, a rate limit or
        # a request Venice rejects. Each model's own error says which.
        print(f"❌ No usable quote: {len(e.failures)} quotes failed")
        for model_id, error in list(e.failures.items())[:3]:
            print(f"   {model_id}: {type(error).__name__}: {error}")
        return
    print(f"❌ Error: {e}")
    if isinstance(e, (AuthenticationError, PermissionDeniedError)):
        print("💡 Check VENICE_API_KEY: video generation needs a valid key with API access")


def _image_reachable(url: str) -> bool:
    """Free pre-flight: confirm the reference image still serves an image.

    The server fetches ``image_url`` only after the job is queued, so a dead
    link would otherwise surface as a failed paid job.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "venice-py-example"})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - fixed URL
            content_type = resp.headers.get("Content-Type", "")
    except OSError as e:
        print(f"❌ Reference image unreachable: {e}")
        return False
    if not content_type.startswith("image/"):
        print(f"❌ Reference image URL returned {content_type!r}, not an image")
        return False
    return True


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
    if not await asyncio.to_thread(_image_reachable, SAMPLE_IMAGE_URL):
        return False
    try:
        async with await client.video.run(
            model=model,
            prompt=prompt,
            image_url=SAMPLE_IMAGE_URL,
            **params,
        ) as job:
            print(f"📨 Queued — queue_id: {job.queue_id}")
            return await _wait_and_save(job, output_path, params)
    except (VeniceError, ValueError) as e:
        _report_error(e)
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
        _report_error(e)
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
        constraints = await _constraints(client, model)
        base = cheapest_video_params(constraints)
    except (VeniceError, ValueError) as e:
        _report_error(e)
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
    for resolution in sorted(constraints.resolutions, key=_resolution_height):
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
        _report_error(e)
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
    print("\n💡 Key concepts demonstrated:")
    print("   - image_url switches run() to I2V mode")
    print("   - resolve_cheapest_video(): cheapest plain I2V model plus its quoted parameters")
    print("   - video_input_mode(): plain I2V vs reference / transition models")
    print("   - Prompt describes desired motion, not image content")
    print("   - Clip checked against its request: duration and resolution")
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
