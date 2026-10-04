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

The lifecycle is the usual one: ``quote()`` for a price, ``run()`` to queue
and get a :class:`venice_ai.VideoJob`, ``wait()`` to poll to completion and
``download()`` to save the result. Leaving the ``async with`` block after the
job finished releases its stored media on the server.

Model resolution: ``client.models.list(type="upscale")`` returns the *image*
upscaler, not the video upscaler. Use the dedicated shortcut
``client.models.resolve_video_upscale()`` instead; it looks for
``type="video"`` models with ``model_type="video"`` + ``video_input=True``.

Quote only by default
---------------------
The quote is free and always runs. Upscaling costs real money, so the job
is only queued when you opt in; without it the example exits 77
("skipped")::

    VENICE_RUN_PAID_VIDEO=1

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
from typing import Literal

from venice_ai import NoMatchingModelError, VeniceClient, VideoGenerationError
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.models import VideoModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import SKIPPED, mp4_tracks, print_progress  # noqa: E402

RUN_PAID = os.environ.get("VENICE_RUN_PAID_VIDEO") == "1"

# A small public-domain sample clip (Big Buck Bunny, 640x360, 10s) so this
# example produces a real quote out of the box.
SOURCE_VIDEO_URL = os.environ.get(
    "VENICE_UPSCALE_SOURCE_URL",
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
)

# Resolve results dir relative to this file's location.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


async def quote_upscale(client: VeniceClient, model: str, factor: UpscaleFactor) -> bool:
    """Get a cost estimate for upscaling the source video (free)."""
    print("\n💰 Quote an Upscale Job")
    print("-" * 30)
    print(f"🎞️  Source: {SOURCE_VIDEO_URL}")
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
        return False

    print(f"💵 Estimated cost: ${float(quote.quote):.4f}")
    return True


async def upscale_job(client: VeniceClient, model: str, factor: UpscaleFactor) -> bool | None:
    """Queue the upscale, wait for it and download the result (paid).

    Returns ``None`` when skipped because ``VENICE_RUN_PAID_VIDEO`` is unset.
    """
    print("\n🚀 Run an Upscale Job")
    print("-" * 30)
    if not RUN_PAID:
        print("⏭️  Not queued: set VENICE_RUN_PAID_VIDEO=1 to run it (incurs the quoted cost).")
        return None

    output_path = RESULTS_DIR / f"upscaled_x{factor}.mp4"
    try:
        async with await client.video.run(
            model=model,
            video_url=SOURCE_VIDEO_URL,
            upscale_factor=factor,
        ) as job:
            print(f"📨 Queued — queue_id: {job.queue_id}")
            status = await job.wait(on_progress=print_progress)
            saved = await job.download(output_path, status)
    except VideoGenerationError as e:
        code = f" (code={e.error_code})" if e.error_code else ""
        print(f"❌ Upscale failed: {e}{code}")
        return False
    except TimeoutError as e:
        print(f"❌ {e}")
        return False
    except VeniceError as e:
        # Queueing was rejected, or a status/download request failed after
        # the job was queued (a timeout there does not cancel the job).
        print(f"❌ Upscale request failed: {e}")
        return False

    data = saved.read_bytes()
    if data[4:8] != b"ftyp":
        print(f"❌ {saved} is not an MP4 file (header {data[:12]!r})")
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

        results: dict[str, bool | None] = {"quote": await quote_upscale(client, model, factor)}
        if results["quote"]:
            results["upscale"] = await upscale_job(client, model, factor)

    print("\n📊 Summary:")
    labels = {True: "✅ passed", False: "❌ failed", None: "⏭️  skipped (not paid)"}
    for name, outcome in results.items():
        print(f"   {labels[outcome]}: {name}")

    failed = [name for name, outcome in results.items() if outcome is False]
    if failed:
        print(f"\n❌ Failed steps: {failed}")
        return 1
    if results.get("upscale") is None:
        # The upscale itself is this example's core feature; a quote alone
        # does not show it.
        print("\nSKIPPED: VENICE_RUN_PAID_VIDEO is not set, so nothing was upscaled")
        return SKIPPED

    print("\n💡 Tips:")
    print("   - upscale_factor=1 (where listed) is quality-only enhancement at the same size.")
    print("   - upscale_factor=4 multiplies each dimension by 4 (16x the pixels); expensive.")
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
