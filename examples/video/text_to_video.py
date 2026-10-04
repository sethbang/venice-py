#!/usr/bin/env python3
"""
Venice AI SDK - Text-to-Video Generation
=========================================

This example demonstrates how to generate videos from text prompts using the
Venice AI SDK. Video generation is **queue-based** and asynchronous; the SDK
exposes the lifecycle through :class:`venice_ai.VideoJob`:

    1. ``client.models.resolve_cheapest_video(...)`` — quote every matching
       model (free) at its own cheapest valid request and return the lowest,
       with the ``request_params`` it was quoted with.
    2. ``client.models.get(model)`` — read a model's ``constraints``
       (durations, resolutions, aspect ratios, audio) and build a request
       from values the model actually accepts.
    3. ``client.video.quote(...)`` — get an estimated cost in USD.
    4. ``client.video.run(...)`` — submit and receive a ``VideoJob``.
    5. ``async with job:`` — releases the finished job's stored media on exit.
    6. ``await job.wait(on_progress=...)`` — poll until ``VideoCompletedStatus``
       (raises :class:`VideoGenerationError` on failure or ``TimeoutError`` on
       poll exhaustion).
    7. ``await job.download(path, status)`` — write the video to disk using
       the SDK's managed HTTP session (no separate ``aiohttp`` session needed).

Models differ in what they accept: one lists 4/6/8/10s, another 5/10s; some
let you switch audio off and others reject the ``audio`` field outright.
Never hardcode these values — read them from the resolved model's
constraints, or ask the resolver for a model that has what you need
(``require_duration=``, ``require_resolution=``,
``require_audio_configurable=True``).

When the catalog has no model a section asks for, the resolver raises
:class:`venice_ai.NoMatchingModelError` and the section is skipped; when
models exist but every quote fails it raises
:class:`venice_ai.ModelQuotesUnavailableError`, and the section fails.

Generation costs real money, so by default this example only quotes (free),
skips the two generation sections and exits 77 ("skipped"). Opt in to queue,
wait for and download two short clips, each on the cheapest matching model at
its shortest duration and lowest resolution, and check each clip against its
request::

    VENICE_RUN_PAID_VIDEO=1
"""

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from venice_ai import (
    APITimeoutError,
    CheapestVideoResult,
    ModelQuotesUnavailableError,
    NoMatchingModelError,
    VeniceClient,
    VideoGenerationError,
    VideoJob,
    cheapest_video_params,
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

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/video/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_PAID = os.environ.get("VENICE_RUN_PAID_VIDEO") == "1"

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


async def _wait_and_save(
    job: VideoJob,
    output_path: Path,
    request: dict[str, Any],
    *,
    expect_audio: bool | None = None,
) -> bool:
    """Wait for completion, save to disk and check the clip against ``request``.

    The duration, aspect ratio and resolution that were sent must show up in
    the file. ``expect_audio`` (when not ``None``) also checks that the file
    has, or lacks, a sound track, so a request whose audio setting was
    ignored fails.
    """
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
    if not check_video(data, request, expect_audio=expect_audio):
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


def _print_pick(pick: CheapestVideoResult) -> None:
    """Show the resolver's choice and how many models it compared."""
    print(f"📍 Cheapest of {len(pick.all_quotes)} quoted models: {pick.model}")
    print(f"📐 Request parameters it was quoted with: {pick.request_params}")
    print(f"💵 Quoted cost: ${pick.quote_usd:.4f}")


# ---------------------------------------------------------------------------
# Section 1 — Basic text-to-video generation
# ---------------------------------------------------------------------------


async def basic_text_to_video(client: VeniceClient) -> Outcome:
    """Generate a video from a simple text prompt.

    Returns ``True`` on success, ``False`` if the quotes, submission,
    generation or the save step failed, and the reason when skipped
    (``VENICE_RUN_PAID_VIDEO`` unset, or no text-to-video model listed).
    """
    print("🎬 Basic Text-to-Video Generation")
    print("-" * 40)

    try:
        # Quotes every text-to-video model at its own shortest duration and
        # lowest resolution (audio off where configurable) and returns the
        # lowest, together with the parameters that price was quoted for.
        try:
            pick = await client.models.resolve_cheapest_video(video_type="text-to-video")
        except NoMatchingModelError as e:
            return f"no text-to-video model to generate with ({e})"
        prompt = "A serene mountain landscape at sunrise with golden light spilling over snow-capped peaks"

        _print_pick(pick)
        print(f"📝 Prompt: {prompt}")
        if not RUN_PAID:
            print(
                "⏭️  Not queued: set VENICE_RUN_PAID_VIDEO=1 to generate (incurs the quoted cost)."
            )
            return NOT_PAID

        # run() returns a VideoJob; using it as an async context manager
        # releases the job's stored media on exit once the job has finished.
        # _wait_and_save() handles its own errors, so a job that timed out
        # is left running and reported with a WARNING naming its queue_id.
        async with await client.video.run(
            model=pick.model, prompt=prompt, **pick.request_params
        ) as job:
            print(f"📨 Queued — queue_id: {job.queue_id}")
            return await _wait_and_save(job, RESULTS_DIR / "basic_video.mp4", pick.request_params)

    except (VeniceError, ValueError) as e:
        _report_error(e)
        return False


# ---------------------------------------------------------------------------
# Section 2 — Cost estimation with quote()
# ---------------------------------------------------------------------------


async def cost_estimation(client: VeniceClient) -> Outcome:
    """Get price quotes before committing to generation (free).

    Every duration and resolution quoted comes from the model's own
    constraints, so each quote is expected to succeed; any failure fails the
    section.
    """
    print("\n💰 Cost Estimation with quote()")
    print("-" * 40)

    try:
        model = await client.models.resolve_video(video_type="text-to-video")
    except NoMatchingModelError as e:
        return f"no text-to-video model to quote ({e})"
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
# Section 3 — Advanced options
# ---------------------------------------------------------------------------


async def advanced_options(client: VeniceClient) -> Outcome:
    """Demonstrate ``negative_prompt`` and switching audio off.

    ``audio=False`` is only accepted by models whose constraints say
    ``audio_configurable``; ask the resolver for one instead of sending the
    field to a model that rejects it. ``resolve_cheapest_video`` already
    quotes those models with ``audio=False``, so its ``request_params``
    carry the flag. ``negative_prompt`` is a hint: the catalog does not say
    which models honor it, and some ignore it.
    """
    print("\n⚙️  Advanced Options")
    print("-" * 40)

    try:
        try:
            pick = await client.models.resolve_cheapest_video(
                video_type="text-to-video",
                require_audio_configurable=True,
                audio=False,
            )
        except NoMatchingModelError as e:
            return f"no text-to-video model lets audio be switched off ({e})"
        prompt = (
            "A slow-motion closeup of ocean waves crashing on a rocky shore, cinematic lighting"
        )
        negative_prompt = "blurry, low quality, text, watermark"

        print("🔇 Only audio-configurable models, quoted with audio=False")
        _print_pick(pick)
        print(f"📝 Prompt: {prompt}")
        print(f"🚫 Negative (a hint; not every model honors it): {negative_prompt}")
        if not RUN_PAID:
            print(
                "⏭️  Not queued: set VENICE_RUN_PAID_VIDEO=1 to generate (incurs the quoted cost)."
            )
            return NOT_PAID

        async with await client.video.run(
            model=pick.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **pick.request_params,
        ) as job:
            print(f"📨 Queued — queue_id: {job.queue_id}")
            return await _wait_and_save(
                job, RESULTS_DIR / "advanced_video.mp4", pick.request_params, expect_audio=False
            )

    except (VeniceError, ValueError) as e:
        _report_error(e)
        return False


# ---------------------------------------------------------------------------
# Section 4 — Model discovery
# ---------------------------------------------------------------------------


async def model_discovery(client: VeniceClient) -> Outcome:
    """Browse text-to-video models and narrow them with resolver filters (free)."""
    print("\n🔍 Model Discovery")
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
        t2v = [
            (model_id, c)
            for model_id, spec, c in listing.models
            if c.model_type == "text-to-video" and not spec.offline
        ]
        if not t2v:
            return "the catalog lists no online text-to-video model"
        if odd := [f"{m}: {d}" for m, c in t2v if (d := unrecognised_durations(c))]:
            print(f"❌ Durations that cannot be read: {'; '.join(odd[:5])}")
            return False
        print(f"📚 {len(t2v)} text-to-video models online. First five:")
        for model_id, c in t2v[:5]:
            audio = (
                "on/off" if c.audio_configurable else ("always" if c.audio else "not configurable")
            )
            print(
                f"   - {model_id}: durations={c.durations} resolutions={c.resolutions} audio={audio}"
            )

        # Filter on a duration/resolution pair the listing actually offers:
        # the one the most audio-configurable models list.
        pairs = Counter(
            (d, r)
            for _, c in t2v
            if c.audio_configurable
            for d in c.durations
            if duration_seconds(d) is not None
            for r in c.resolutions
        )
        if not any(c.audio_configurable for _, c in t2v):
            return "no online text-to-video model lets audio be switched off"
        if not pairs:
            # Every duration was read, so this is the catalog, not parsing:
            # the audio-configurable models choose their own length or list
            # no resolution tiers.
            return (
                "no audio-configurable text-to-video model lists both a fixed duration "
                "and a resolution to filter on"
            )
        (duration, resolution), count = pairs.most_common(1)[0]

        print("\n🎯 Resolver filters narrow the choice:")
        default = await client.models.resolve_video(video_type="text-to-video")
        print(f"   resolve_video(video_type='text-to-video') -> {default}")
        print(f"   ({duration} at {resolution} is listed by {count} audio-configurable models)")
        filtered = await client.models.resolve_video(
            video_type="text-to-video",
            require_duration=duration,
            require_resolution=resolution,
            require_audio_configurable=True,
        )
        print(
            f"   ... require_duration={duration!r}, require_resolution={resolution!r}, "
            f"require_audio_configurable=True -> {filtered}"
        )
        quote = await client.video.quote(
            model=filtered, duration_seconds=duration, resolution=resolution, audio=False
        )
        print(
            f"   💵 {duration} at {resolution} without audio on {filtered}: "
            f"${float(quote.quote):.4f}"
        )
        return True

    except (VeniceError, ValueError) as e:
        _report_error(e)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all text-to-video examples.

    Returns ``1`` when a section failed, ``77`` (skipped) when the basic
    generation did not run, and ``0`` otherwise; any other skipped section is
    listed with its reason.
    """
    print("🚀 Venice AI Text-to-Video Examples")
    print("=" * 60)
    if RUN_PAID:
        print("💸 VENICE_RUN_PAID_VIDEO=1 — generation sections will queue paid jobs.")
    else:
        print("🧪 Quote only (default). Set VENICE_RUN_PAID_VIDEO=1 to generate videos.")

    async with VeniceClient() as client:
        results: list[tuple[str, Outcome]] = [
            ("basic_text_to_video", await basic_text_to_video(client)),
            ("cost_estimation", await cost_estimation(client)),
            ("advanced_options", await advanced_options(client)),
            ("model_discovery", await model_discovery(client)),
        ]

    if WRITTEN:
        print(f"\n📁 Files written by this run in {RESULTS_DIR}/:")
        for path in WRITTEN:
            print(f"   - {path.name}")

    # Generation is this example's core feature; quotes alone do not show it.
    code = exit_code(results, core="basic_text_to_video")
    if code:
        return code

    print("\n✨ Text-to-video examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - resolve_cheapest_video(): cheapest model plus the parameters it was quoted at")
    print("   - Each clip checked against its request: duration, aspect, resolution, audio")
    print("   - Quote matrix built from one model's listed durations and resolutions")
    print(
        "   - Resolver filters: require_duration / require_resolution / require_audio_configurable"
    )
    print("   - Cost estimation with quote() before committing")
    print("   - VideoJob: wait(on_progress=...) and download(path, status)")
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
