#!/usr/bin/env python3
"""
Venice AI SDK - Video: Upscale
==============================

Demonstrates the video-upscale flow on ``client.video``. Upscaling is
distinct from generation:

- Provide ``video_url`` (the source video) and no ``prompt``.
- Use ``upscale_factor`` instead of ``resolution``. The factors a model
  accepts are listed in its ``constraints.resolutions`` (e.g. ``"2x"``,
  ``"4x"``); each factor multiplies **each dimension**, so ``2`` turns a
  640x360 clip into 1280x720.
- Leave ``duration_seconds`` out. Upscale models list only ``"Auto"`` as
  their duration: the server reads the length and frame rate from the
  source file, both for the quote and for the job.

The quote honours the factor: measured on a 10-second 640x360 sample,
``upscale_factor=2`` quoted $0.13 and ``upscale_factor=4`` quoted $1.00.

The lifecycle is the usual one: ``quote()`` for a price, ``run()`` to queue
and get a :class:`venice_ai.VideoJob`, ``wait()`` to poll to completion and
``download()`` to save the result. Leaving the ``async with`` block after the
job finished releases its stored media on the server.

Model resolution: ``client.models.list(type="upscale")`` returns the *image*
upscaler, not the video upscaler. Use the dedicated shortcut
``client.models.resolve_video_upscale()`` instead; it looks for
``type="video"`` models with ``model_type="video"`` + ``video_input=True``.

Enhancement
-----------
Models that publish ``constraints.topaz.models`` take an ``enhancement_model``
and tuning fields such as ``noise`` and ``recover_detail``. ``enhancement_model``
is quotable, so its price is printed for free. A paid run upscales once more with
it. The tuning fields have no quote parameter, so that price is the base.

Quote only by default
---------------------
The quote is free and always runs. Upscaling costs real money, so the job
is only queued when you opt in; without it the example exits 77
("skipped")::

    VENICE_RUN_PAID_VIDEO=1

Each job must also quote at or under ``VIDEO_CEILING_USD`` ($0.50). Above it the
job is refused, because price grows with the source's length, frame rate and the
factor, and a long source can reach many dollars. Raise the ceiling for one run
when you mean to spend more::

    VENICE_UPSCALE_MAX_USD=2.00

Cost with the opt-in: the upscale ($0.13 on the default sample) and, where the
model publishes enhancement models, one enhanced upscale at its printed quote.

If the catalog lists no video upscaler, ``resolve_video_upscale()`` raises
:class:`venice_ai.NoMatchingModelError` and the example exits 77 as well.

Set ``VENICE_UPSCALE_SOURCE_URL`` to upscale your own clip (an HTTPS URL or
a ``data:`` URI; data URIs get large fast, so prefer a signed URL).
"""

import asyncio
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Literal

import aiohttp
from _helpers import (
    NOT_PAID,
    SKIPPED,
    Outcome,
    UnplayableVideoError,
    ceiling_refusal,
    display_url,
    download_video,
    exit_code,
    mp4_tracks,
    print_progress,
    print_recovery,
    read_playable_video,
    report_uncertain_queue,
    url_serves,
    video_constraints,
)

from venice_ai import NoMatchingModelError, VeniceClient, VideoGenerationError
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.models import VideoModelConstraints, VideoModelSpec

RUN_PAID = os.environ.get("VENICE_RUN_PAID_VIDEO") == "1"

# A small sample clip (Big Buck Bunny, CC BY 3.0, Blender Foundation; 640x360,
# 10s) so this example produces a real quote out of the box.
SOURCE_VIDEO_URL = os.environ.get(
    "VENICE_UPSCALE_SOURCE_URL",
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
)

# Resolve results dir relative to this file's location.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: Set this to spend more than the ceiling on one run.
CEILING_ENV_VAR = "VENICE_UPSCALE_MAX_USD"


def _fetch_bytes(url: str) -> bytes:
    """Read a URL (``https:`` or ``data:``) into memory."""
    # Some hosts refuse urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "venice-py-example"})
    with urllib.request.urlopen(request, timeout=60) as resp:  # noqa: S310 - fixed or user-set URL
        return resp.read()


def _mp4_frame_size(data: bytes) -> tuple[int, int] | None:
    """Width and height of the MP4's video track, or ``None`` if it has none."""
    video = next((t for t in mp4_tracks(data) if t.handler == "vide" and t.width), None)
    return (video.width, video.height) if video else None


# The upscale factors video.run() / quote() accept, by catalog label.
UpscaleFactor = Literal[1, 2, 4]
FACTORS: dict[str, UpscaleFactor] = {"1x": 1, "2x": 2, "4x": 4}


async def _smallest_factor(client: VeniceClient, model: str) -> UpscaleFactor:
    """Smallest enlarging upscale factor the model lists (``"2x"`` -> ``2``).

    ``1x`` is skipped: it enhances quality at the same size, so it would not
    show the dimensions growing.
    """
    entry = await client.models.get(model)
    spec = entry.model_spec
    listed = (
        spec.constraints.resolutions
        if isinstance(spec, VideoModelSpec) and spec.constraints
        else []
    )
    factors: list[UpscaleFactor] = [
        FACTORS[r.lower()] for r in listed if r.lower() in FACTORS and FACTORS[r.lower()] > 1
    ]
    if not factors:
        raise ValueError(f"{model} lists no upscale factor above 1x (resolutions={listed})")
    return min(factors)


async def quote_upscale(client: VeniceClient, model: str, factor: UpscaleFactor) -> float | None:
    """Get a cost estimate for upscaling the source video (free).

    Returns the quoted USD, or ``None`` if the quote failed.
    """
    print("\n💰 Quote an Upscale Job")
    print("-" * 30)
    print(f"🎞️  Source: {display_url(SOURCE_VIDEO_URL)}")
    print(f"🔍 upscale_factor={factor} (smallest enlarging factor the model lists)")

    try:
        quote = await client.video.quote(
            model=model,
            video_url=SOURCE_VIDEO_URL,
            upscale_factor=factor,
        )
    except VeniceError as e:
        # The server fetches and inspects the source to price it, so a 4xx
        # here usually means the URL is unreachable or not a supported video.
        print(f"❌ Quote failed: {e}")
        print("   Ensure the source is a fetchable video the server can inspect")
        print("   (a public HTTPS URL or a data: URI).")
        return None

    print(f"💵 Estimated cost: ${float(quote.quote):.4f}")
    return float(quote.quote)


def enhancement_models(constraints: VideoModelConstraints) -> list[str]:
    """The ``enhancement_model`` values a model publishes under ``constraints.topaz.models``.

    Each entry is an object with a ``display`` name and the ``value`` to send.
    """
    topaz = (constraints.model_extra or {}).get("topaz")
    models = topaz.get("models") if isinstance(topaz, dict) else None
    if not isinstance(models, list):
        return []
    return [str(m["value"]) for m in models if isinstance(m, dict) and m.get("value")]


async def enhanced_upscale(
    client: VeniceClient, model: str, factor: UpscaleFactor, base_usd: float
) -> Outcome:
    """Quote an upscale with an ``enhancement_model`` and, on a paid run, run it."""
    print("\n✨ Enhancement: enhancement_model and tuning fields")
    print("-" * 30)
    try:
        listed = enhancement_models(await video_constraints(client, model))
    except (ValueError, VeniceError) as e:
        print(f"❌ Could not read {model}'s enhancement models: {e}")
        return False
    if not listed:
        return f"{model} publishes no constraints.topaz.models"
    enhancement = listed[0]
    print(f"🧪 {len(listed)} listed: {', '.join(listed)}; using {enhancement}")
    try:
        quote = await client.video.quote(
            model=model,
            video_url=SOURCE_VIDEO_URL,
            upscale_factor=factor,
            enhancement_model=enhancement,
        )
    except VeniceError as e:
        print(f"❌ Quote failed: {e}")
        return False
    usd = float(quote.quote)
    print(f"💵 ${usd:.4f} with it, ${base_usd:.4f} without; tuning fields are not priced by quotes")
    return await upscale_job(
        client,
        model,
        factor,
        usd,
        output_name=f"upscaled_x{factor}_enhanced.mp4",
        enhancement_model=enhancement,
        noise=0.2,
        recover_detail=0.5,
    )


async def upscale_job(
    client: VeniceClient,
    model: str,
    factor: UpscaleFactor,
    quoted_usd: float,
    *,
    output_name: str | None = None,
    **fields: Any,
) -> Outcome:
    """Queue the upscale, wait for it and download the result (paid).

    Returns the skip reason when ``VENICE_RUN_PAID_VIDEO`` is unset or the quote
    is over the spending ceiling. ``fields`` are extra ``video.run`` arguments.
    """
    print("\n🚀 Run an Upscale Job")
    print("-" * 30)
    if not RUN_PAID:
        print("⏭️  Not queued: set VENICE_RUN_PAID_VIDEO=1 to run it (incurs the quoted cost).")
        return NOT_PAID
    refusal = ceiling_refusal(quoted_usd, CEILING_ENV_VAR)
    if refusal is not None:
        return refusal
    if not await asyncio.to_thread(url_serves, SOURCE_VIDEO_URL, "video/"):
        return False

    output_path = RESULTS_DIR / (output_name or f"upscaled_x{factor}.mp4")
    try:
        job = await client.video.run(
            model=model,
            video_url=SOURCE_VIDEO_URL,
            upscale_factor=factor,
            **fields,
        )
    except (VeniceError, ValueError) as e:
        if not report_uncertain_queue(e):
            print(f"❌ Upscale request rejected: {e}")
        return False
    print(f"📨 Queued — queue_id: {job.queue_id}")

    # The handlers sit outside the block: leaving it cleanly releases a finished
    # job's stored video, so a failed download has to raise through it.
    try:
        async with job:
            status = await job.wait(on_progress=print_progress)
            saved = await download_video(job, output_path, status)
            data = read_playable_video(saved)
    except UnplayableVideoError as e:
        print(f"❌ {output_path} is not a usable video: {e} (queue_id={job.queue_id})")
        print_recovery(job)
        return False
    except VideoGenerationError as e:
        code = f" (code={e.error_code})" if e.error_code else ""
        print(f"❌ Upscale failed: {e}{code}")
        return False
    except TimeoutError as e:
        # wait() ran out of polls; the job is still queued or running.
        print(f"❌ {e} (queue_id={job.queue_id})")
        print_recovery(job)
        return False
    except aiohttp.ClientResponseError as e:
        # The error shows the storage host at most, never the link.
        print(f"❌ The download answered HTTP {e.status} (queue_id={job.queue_id})")
        print_recovery(job)
        return False
    except VeniceError as e:
        # A status request failed, or every download attempt did. That does not
        # cancel the job: it may still finish, and it is billed either way.
        print(f"❌ Upscale request failed (queue_id={job.queue_id}): {e}")
        print_recovery(job)
        return False

    print(f"💾 Saved {saved} ({len(data):,} bytes)")

    # An upscale that silently returned the source size would still be a
    # valid MP4, so compare the frame size against the source's.
    out_size = _mp4_frame_size(data)
    try:
        source = await asyncio.to_thread(_fetch_bytes, SOURCE_VIDEO_URL)
    except OSError as e:
        print(f"❌ Could not re-read the source to compare sizes: {e}")
        return False
    src_size = _mp4_frame_size(source)
    if out_size is None or src_size is None:
        print(f"❌ Could not read frame sizes (source={src_size}, output={out_size})")
        return False
    expected = (src_size[0] * factor, src_size[1] * factor)
    print(f"📐 Source {src_size[0]}x{src_size[1]} -> output {out_size[0]}x{out_size[1]}")
    if out_size != expected:
        print(f"❌ Expected {expected[0]}x{expected[1]} for upscale_factor={factor}")
        return False
    print(f"✅ Each dimension is {factor}x the source's")
    return True


async def main() -> int:
    print("🚀 Venice AI Video — Upscale Example")
    print("=" * 50)
    if RUN_PAID:
        print("💸 VENICE_RUN_PAID_VIDEO=1 — the upscale job will be queued (paid).")
    else:
        print("🧪 Quote only (default). Set VENICE_RUN_PAID_VIDEO=1 to run the upscale.")

    async with VeniceClient() as client:
        # Pick the upscale model dynamically — never hardcode.
        try:
            model = await client.models.resolve_video_upscale()
        except NoMatchingModelError as e:
            print(f"\nSKIPPED: no video-upscale model is listed ({e})")
            return SKIPPED
        except VeniceError as e:
            print(f"❌ Could not resolve a video-upscale model: {e}")
            return 1
        try:
            factor = await _smallest_factor(client, model)
        except (ValueError, VeniceError) as e:
            print(f"❌ Could not read {model}'s upscale factors: {e}")
            return 1
        print(f"📍 Model: {model}")

        quoted = await quote_upscale(client, model, factor)
        results: dict[str, Outcome] = {"quote": quoted is not None}
        if quoted is not None:
            results["upscale"] = await upscale_job(client, model, factor, quoted)
            # Nothing else is queued after a failure, which may have been billed.
            if results["upscale"] is not False:
                results["enhanced_upscale"] = await enhanced_upscale(client, model, factor, quoted)

    print("\n📊 Summary:")
    for name, outcome in results.items():
        label = (
            "✅ passed" if outcome is True else "❌ failed" if outcome is False else "⏭️  skipped"
        )
        print(f"   {label}: {name}")

    # The upscale itself is this example's core feature; a quote alone does not
    # show it.
    code = exit_code(list(results.items()), core="upscale")
    if code:
        return code

    print("\n💡 Tips:")
    print("   - upscale_factor=1 (where listed) is quality-only enhancement at the same size.")
    print("   - upscale_factor=4 multiplies each dimension by 4 (16x the pixels).")
    print("     It quoted about 8x the price of upscale_factor=2 on the sample clip.")
    print("   - No duration_seconds for upscaling: the length comes from the source file.")
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
