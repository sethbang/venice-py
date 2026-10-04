#!/usr/bin/env python3
"""
Venice AI SDK - Video: Advanced Body Fields
===========================================

Demonstrates the advanced fields on ``client.video.run()`` / ``submit()``
that go beyond a basic text-to-video or image-to-video request:

- ``reference_image_urls`` — references for character / style consistency.
- ``end_image_url`` — the final-frame reference for models that support
  transitions (paired with ``image_url`` as the first frame).
- ``reference_audio_urls`` — donor clips (2–15s, WAV/MP3) for vocals, sound
  effects or music. They must be paired with at least one reference image or
  video; audio-only reference requests are rejected.
- ``reference_video_urls`` — donor clips (2–15s, MP4/MOV) whose subject
  motion, camera movement and style the generation inherits. Quote them with
  ``reference_video_total_duration`` so the price matches the queue charge.
- ``elements`` + ``scene_image_urls`` — structured characters / objects and
  scene references for element-aware models, addressed in the prompt as
  ``@Element1`` … and ``@Image1`` …. Each element carries either images
  (``frontal_image_url`` plus optional ``reference_image_urls``) or a
  ``video_url``, never both.

Choosing a model
----------------
The catalog types reference-to-video, transition and plain image-to-video
models all as ``model_type="image-to-video"``. The SDK tells them apart with
``video_input_mode()``, and ``resolve_cheapest_video(input_mode=...)`` quotes
only the models of one kind (free) and returns the cheapest together with the
``request_params`` it was quoted at. Each section asks for the aspect ratio of
its sample media and narrows the candidates with what the catalog does say:

- ``reference_image_urls``: ``input_mode="reference"``.
- ``end_image_url``: ``input_mode="transition"``.
- ``reference_audio_urls``: ``input_mode="reference"`` and ``audio=True``,
  excluding models whose constraints do not list ``audio_input``.
- ``reference_video_urls``: the catalog has no reliable flag for this
  (``video_input`` is false on every reference model, including the ones that
  take donor clips). The quote is the signal instead: a model that takes a
  donor clip prices its length, so its quote rises when
  ``reference_video_total_duration`` is added. The section quotes reference
  models from cheapest up and keeps the cheapest whose price rises.
- ``elements``: neither the catalog nor the quote says which models accept
  elements, so this section takes its model from
  ``VENICE_VIDEO_ELEMENTS_MODEL`` (an element-aware reference-to-video
  model; the Venice reference-to-video guide lists them) and is skipped
  when the variable is unset.

Models whose reference-image limits (minimum short side, aspect bounds) the
800x600 sample photos break are excluded before quoting.

When the catalog has no model a section can use, the resolver raises
:class:`venice_ai.NoMatchingModelError` and that section is skipped. When
models exist but every quote fails, it raises
:class:`venice_ai.ModelQuotesUnavailableError` and the section fails: that
is an outage or a rejected request, not a gap in the catalog.

Quote only by default
---------------------
Without the opt-in below, each section resolves its model, prints the request
and its quote, and stops; the example then exits 77 ("skipped"), because
nothing was generated or verified. The quote prices duration, resolution and
reference-video length; it takes no reference images, audio or elements, so
treat it as the base price. To queue the jobs, wait for them and download the
results to ``examples/results/`` (incurs at least the quoted cost), set::

    VENICE_RUN_PAID_VIDEO=1

A paid run also reads the key's spendable balance
(``client.api_keys.get_rate_limits()``) before and after each job and prints
the drop next to the base quote. Anything else spending from the same key at
the same time is included in that drop.

The example exits 0 when at least one section generated a clip that matched
its request and none failed; sections skipped for want of a model are listed
as such. It exits 1 when any section failed and 77 when nothing was
generated.

The reference media are public sample files without people: Picsum photos
(800x600) and a 5s CC0 clip of a flower from MDN. The reference audio is a
short synthesized melody built inline as a ``data:`` URI.

Each saved clip is checked against its request: duration, aspect ratio,
resolution tier, and a sound track present or absent as requested. A clip
that passes shows the job honored the request; it does not show that the
model used every advanced field, so watch the clips to judge the references.
"""

import asyncio
import base64
import io
import math
import os
import struct
import sys
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from venice_ai import (
    APITimeoutError,
    ModelQuotesUnavailableError,
    NoMatchingModelError,
    VeniceClient,
    VideoGenerationError,
    VideoInputMode,
    cheapest_video_params,
    video_input_mode,
)
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.models import VideoModelConstraints, VideoModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import NOT_PAID, Outcome, check_video, exit_code, print_progress  # noqa: E402

RUN_PAID = os.environ.get("VENICE_RUN_PAID_VIDEO") == "1"
ELEMENTS_ENV_VAR = "VENICE_VIDEO_ELEMENTS_MODEL"

# Resolve results dir relative to this file's location.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Public 800x600 sample photos (Picsum Photos CDN, direct fastly URLs so no
# redirect has to be followed). Sections that animate them ask for the same
# 4:3 aspect ratio.
IMAGE_SIZE = (800, 600)
IMAGE_ASPECT_RATIO = "4:3"
FOREST_SHORE_URL = (
    "https://fastly.picsum.photos/id/10/800/600.jpg"
    "?hmac=9u_ZYBasFb_VEVrBgjTZor_IfBxtpq9zl_CjKJr7-cs"
)
SNOWY_PEAKS_URL = (
    "https://fastly.picsum.photos/id/29/800/600.jpg"
    "?hmac=KbDC2qhBvoFM4XpOZrAnybO4JNiXS9mox0PORy6NCJA"
)
PUPPY_URL = (
    "https://fastly.picsum.photos/id/237/800/600.jpg"
    "?hmac=KK6IftaxyI6scjoxeWUHr2AWr-7RQuExLH9vY4hSR6w"
)

# A short public CC0 clip (a flower, 960x540 H.264, 5s, 1 MB) for the video
# reference field. Reference-video models bill by the donor's length, so a
# short clip keeps the job cheap.
SAMPLE_VIDEO_URL = "https://developer.mozilla.org/shared-assets/videos/flower.mp4"
SAMPLE_VIDEO_ASPECT_RATIO = "16:9"
SAMPLE_VIDEO_SECONDS = 5.055

# The video models the catalog lists, by id (filled once in main()).
Catalog = dict[str, VideoModelConstraints]


def _melody_wav_data_url(sample_rate: int = 16000) -> str:
    """A 3-second rising C-major arpeggio as an inline WAV ``data:`` URI."""
    notes_hz = (261.63, 329.63, 392.00, 523.25, 392.00, 329.63)
    note_len = sample_rate // 2  # half a second per note
    frames = bytearray()
    for freq in notes_hz:
        for i in range(note_len):
            envelope = min(1.0, i / 400, (note_len - i) / 400)  # avoid clicks
            sample = 0.4 * envelope * math.sin(2 * math.pi * freq * i / sample_rate)
            frames += struct.pack("<h", int(sample * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class SectionSkip(Exception):
    """A section has no model to run on: none can serve it, or none was named.

    Handled like the resolver's :class:`venice_ai.NoMatchingModelError`: the
    section is skipped, not failed.
    """


@dataclass
class Pick:
    """The model a section will use and the request it was quoted at."""

    model: str
    params: dict[str, Any]
    quote_usd: float
    how: str  # why this model was chosen


async def _load_catalog(client: VeniceClient) -> tuple[Catalog, dict[str, str]]:
    """Constraints and display names of every online video model."""
    listing = await client.models.list(type="video")
    catalog: Catalog = {}
    names: dict[str, str] = {}
    for m in listing.data:
        spec = m.model_spec
        if isinstance(spec, VideoModelSpec) and spec.constraints is not None and not spec.offline:
            catalog[m.id] = spec.constraints
            names[m.id] = spec.name or ""
    return catalog, names


def _input_mode(model_id: str, names: dict[str, str]) -> VideoInputMode:
    return video_input_mode({"id": model_id, "name": names.get(model_id, "")})


def _breaks_image_limits(constraints: VideoModelConstraints) -> bool:
    """True if the 800x600 sample photos break the model's reference-image limits."""
    extra = constraints.model_extra or {}
    width, height = IMAGE_SIZE
    min_short = extra.get("reference_image_min_short_side_pixels")
    if min_short is not None and min(width, height) < min_short:
        return True
    ratio = width / height
    low = extra.get("reference_image_min_aspect_ratio")
    high = extra.get("reference_image_max_aspect_ratio")
    return (low is not None and ratio < low) or (high is not None and ratio > high)


def _lists_audio_input(constraints: VideoModelConstraints) -> bool:
    """True if the model's constraints say it accepts reference audio."""
    return bool((constraints.model_extra or {}).get("audio_input"))


async def _cheapest(
    client: VeniceClient,
    catalog: Catalog,
    *,
    mode: VideoInputMode,
    what: str,
    aspect_ratio: str,
    audio: bool = False,
    sends_images: bool = True,
    needs_audio_input: bool = False,
) -> Pick:
    """``resolve_cheapest_video`` for one input mode, minus models that cannot take the inputs.

    Raises :class:`venice_ai.NoMatchingModelError` when no model of ``mode``
    is left after the exclusions.
    """
    exclude = [
        mid
        for mid, c in catalog.items()
        if (sends_images and _breaks_image_limits(c))
        or (needs_audio_input and not _lists_audio_input(c))
    ]
    pick = await client.models.resolve_cheapest_video(
        input_mode=mode,
        aspect_ratio=aspect_ratio,
        audio=audio,
        exclude_models=exclude,
    )
    return Pick(
        pick.model,
        pick.request_params,
        pick.quote_usd,
        f"cheapest of {len(pick.all_quotes)} {what} models quoted",
    )


async def _cheapest_reference_video(client: VeniceClient, catalog: Catalog) -> Pick:
    """The cheapest reference model whose quote prices a donor clip.

    Reference models are quoted at their cheapest request (free), then
    re-quoted from cheapest up with ``reference_video_total_duration``. A
    model whose price rises bills for the donor clip, so it takes one. A donor
    never lowers a quote, so the search stops once the next model's base price
    reaches the best donor-inclusive price found.

    Raises :class:`SectionSkip` only when every donor quote succeeded and none
    rose. If no model qualified and some donor quotes failed, the answer is
    unknown, so it raises :class:`venice_ai.ModelQuotesUnavailableError`.
    """
    base = await client.models.resolve_cheapest_video(
        input_mode="reference", aspect_ratio=SAMPLE_VIDEO_ASPECT_RATIO
    )
    best: Pick | None = None
    checked = 0
    failures: dict[str, Exception] = {}
    for model_id, base_usd in sorted(base.all_quotes.items(), key=lambda item: item[1]):
        if best is not None and base_usd >= best.quote_usd:
            break
        params = cheapest_video_params(catalog[model_id], aspect_ratio=SAMPLE_VIDEO_ASPECT_RATIO)
        try:
            quote = await client.video.quote(
                model=model_id, **params, reference_video_total_duration=SAMPLE_VIDEO_SECONDS
            )
        except VeniceError as e:
            print(f"   {model_id}: donor quote failed ({e})")
            failures[model_id] = e
            continue
        checked += 1
        with_donor = float(quote.quote)
        takes_donor = with_donor > base_usd
        print(
            f"   {model_id}: ${base_usd:.4f} -> ${with_donor:.4f} with the donor "
            f"({'prices it' if takes_donor else 'ignores it'})"
        )
        if takes_donor and (best is None or with_donor < best.quote_usd):
            best = Pick(model_id, params, with_donor, how="")
    if best is None:
        if failures:
            raise ModelQuotesUnavailableError(
                f"no reference model's donor quote rose, and {len(failures)} donor quotes "
                "failed, so whether any model takes a reference video is unknown",
                resource_type="video",
                failures=failures,
            )
        raise SectionSkip(f"none of the {checked} reference models' quotes prices a donor clip")
    best.how = (
        f"cheapest of {checked} reference models re-quoted with the donor whose price rises with it"
    )
    if failures:
        best.how += f"; {len(failures)} donor quotes failed and were not compared"
    return best


async def _configured_elements_model(
    client: VeniceClient, catalog: Catalog, names: dict[str, str]
) -> Pick:
    """The element-aware model named by ``VENICE_VIDEO_ELEMENTS_MODEL``, quoted."""
    model = os.environ.get(ELEMENTS_ENV_VAR)
    if not model:
        raise SectionSkip(
            f"{ELEMENTS_ENV_VAR} is not set (no catalog field says which models accept elements)"
        )
    constraints = catalog.get(model)
    if constraints is None:
        raise ValueError(f"{ELEMENTS_ENV_VAR}={model} is not an online video model")
    mode = _input_mode(model, names)
    if constraints.model_type != "image-to-video" or mode != "reference":
        raise ValueError(
            f"{ELEMENTS_ENV_VAR}={model} has model_type={constraints.model_type!r} and "
            f"input mode {mode!r}; elements need a reference-to-video model"
        )
    if _breaks_image_limits(constraints):
        raise ValueError(f"{model}: the 800x600 sample images break its reference-image limits")
    # Elements models list few aspect ratios: use 4:3 when listed; otherwise
    # cheapest_video_params() picks 16:9 when listed, else the first listed.
    ratio = IMAGE_ASPECT_RATIO if IMAGE_ASPECT_RATIO in constraints.aspect_ratios else None
    params = cheapest_video_params(constraints, aspect_ratio=ratio)
    quote = await client.video.quote(model=model, **params)
    return Pick(model, params, float(quote.quote), f"set by {ELEMENTS_ENV_VAR}")


# ---------------------------------------------------------------------------
# Running a section
# ---------------------------------------------------------------------------


async def _headroom_usd(client: VeniceClient) -> float | None:
    """USD this API key can still spend, or ``None`` if it cannot be read.

    The figure is the lesser of the account balance and the key's remaining
    spend limit. The drop across a job is what the job cost, plus anything
    else spent from the key in the meantime.
    """
    try:
        limits = await client.api_keys.get_rate_limits()
    except VeniceError as e:
        print(f"   (spendable balance not readable, so the job's cost is not measured: {e})")
        return None
    usd = limits.data.balances.USD
    return float(usd) if usd is not None else None


async def _run_section(
    client: VeniceClient,
    *,
    title: str,
    fields: str,
    select: Callable[[], Awaitable[Pick]],
    build: Callable[[dict[str, Any]], dict[str, Any]],
    output_name: str,
) -> Outcome:
    """Pick a model, quote the request and optionally run and verify it.

    ``build(params)`` returns the full ``run()`` payload for the quoted
    ``params``. The saved clip is checked against those params. Returns
    ``True`` when the clip matched, ``False`` on any failure, and the skip
    reason when no model fits or paid runs are not enabled.
    """
    print(f"\n{title}")
    print("-" * 40)

    try:
        pick = await select()
    except (SectionSkip, NoMatchingModelError) as e:
        print(f"⏭️  No model for {fields}: {e}")
        return str(e)
    except ModelQuotesUnavailableError as e:
        print(f"❌ No usable quote, so no model could be chosen: {len(e.failures)} quotes failed")
        for model_id, error in list(e.failures.items())[:3]:
            print(f"   {model_id}: {type(error).__name__}: {error}")
        return False
    except (VeniceError, ValueError) as e:
        print(f"❌ Model selection failed: {e}")
        return False

    payload = {"model": pick.model, **build(pick.params)}
    print(f"📍 Model: {pick.model} ({pick.how})")
    print(f"📐 Request parameters it was quoted at: {pick.params}")
    print(f"🧩 Advanced fields: {fields}")
    print(f"📝 Prompt: {payload['prompt']}")
    unquoted = [
        key
        for key in ("reference_image_urls", "reference_audio_urls", "elements", "scene_image_urls")
        if key in payload
    ]
    note = f" (not priced by the quote: {', '.join(unquoted)})" if unquoted else ""
    print(f"💵 Base quote: ${pick.quote_usd:.4f}{note}")

    if not RUN_PAID:
        print("🧪 Quoted only, not submitted: set VENICE_RUN_PAID_VIDEO=1 to generate it.")
        return NOT_PAID

    output_path = RESULTS_DIR / output_name
    headroom_before = await _headroom_usd(client)
    try:
        job = await client.video.run(**payload)
    except (VeniceError, ValueError) as e:
        print(f"❌ Queue failed: {e}")
        return False

    async with job:
        print(f"📨 Queued — queue_id: {job.queue_id}")
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
            # One HTTP request timed out. The job itself may still finish and
            # be billed.
            print(f"❌ A status or download request timed out (queue_id={job.queue_id}): {e}")
            return False
        except VeniceError as e:
            # The server can also reject a queued job while it is polled, for
            # example when the provider's content policy refuses the inputs.
            print(f"❌ Rejected after queueing (queue_id={job.queue_id}): {e}")
            return False

    data = saved.read_bytes()
    print(f"💾 Saved {saved} ({len(data):,} bytes)")
    headroom_after = await _headroom_usd(client)
    if headroom_before is not None and headroom_after is not None:
        print(
            f"💳 Key balance dropped ${headroom_before - headroom_after:.4f} over this job "
            f"(base quote ${pick.quote_usd:.4f}; includes any other use of this key meanwhile)"
        )
    # ``audio`` is in the params only for models whose audio is configurable.
    if not check_video(data, pick.params, expect_audio=pick.params.get("audio")):
        print("❌ The clip does not match the request")
        return False
    print("✅ The clip matches the request; watch it to judge the references")
    return True


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


async def reference_images(client: VeniceClient, catalog: Catalog) -> Outcome:
    """``reference_image_urls``: keep a subject consistent with a reference photo."""
    return await _run_section(
        client,
        title="🖼️  reference_image_urls",
        fields="reference_image_urls",
        select=lambda: _cheapest(
            client,
            catalog,
            mode="reference",
            what="reference-to-video",
            aspect_ratio=IMAGE_ASPECT_RATIO,
        ),
        build=lambda params: {
            **params,
            "prompt": "The black puppy from the reference image bounds across a sunny meadow",
            "reference_image_urls": [PUPPY_URL],
        },
        output_name="advanced_reference_images.mp4",
    )


async def end_image(client: VeniceClient, catalog: Catalog) -> Outcome:
    """``image_url`` + ``end_image_url``: transition between two frames."""
    return await _run_section(
        client,
        title="🎬 image_url + end_image_url (transition)",
        fields="image_url, end_image_url",
        select=lambda: _cheapest(
            client,
            catalog,
            mode="transition",
            what="transition",
            aspect_ratio=IMAGE_ASPECT_RATIO,
        ),
        build=lambda params: {
            **params,
            "prompt": (
                "The camera rises from the forested shore and flies toward snowy mountain peaks"
            ),
            "image_url": FOREST_SHORE_URL,
            "end_image_url": SNOWY_PEAKS_URL,
        },
        output_name="advanced_end_image.mp4",
    )


async def reference_audio(client: VeniceClient, catalog: Catalog) -> Outcome:
    """``reference_audio_urls`` paired with a reference image."""
    return await _run_section(
        client,
        title="🔊 reference_audio_urls (+ reference image)",
        fields="reference_audio_urls, reference_image_urls",
        select=lambda: _cheapest(
            client,
            catalog,
            mode="reference",
            what="reference-to-video",
            aspect_ratio=IMAGE_ASPECT_RATIO,
            audio=True,
            needs_audio_input=True,
        ),
        build=lambda params: {
            **params,
            # <Image 1> / <Audio 1> address the first reference image and
            # audio clip, the tag syntax reference-to-video prompts use.
            "prompt": (
                "Refer to the snowy mountain peaks in <Image 1> to generate a slow "
                "aerial shot drifting past the same peaks. Use the melody in <Audio 1> "
                "as the background music."
            ),
            "reference_image_urls": [SNOWY_PEAKS_URL],
            "reference_audio_urls": [_melody_wav_data_url()],
        },
        output_name="advanced_reference_audio.mp4",
    )


async def reference_video(client: VeniceClient, catalog: Catalog) -> Outcome:
    """``reference_video_urls``: inherit a donor clip's motion and style.

    The prompt refers to the donor clip as ``<Video 1>``, the first (and
    only) reference video.
    """
    return await _run_section(
        client,
        title="🎞️  reference_video_urls",
        fields="reference_video_urls",
        select=lambda: _cheapest_reference_video(client, catalog),
        build=lambda params: {
            **params,
            "prompt": (
                "Refer to the slow drifting camera movement in <Video 1> to generate a "
                "new clip of glowing lanterns floating over a calm lake at dusk"
            ),
            "reference_video_urls": [SAMPLE_VIDEO_URL],
        },
        output_name="advanced_reference_video.mp4",
    )


async def elements_and_scenes(
    client: VeniceClient, catalog: Catalog, names: dict[str, str]
) -> Outcome:
    """``elements`` + ``scene_image_urls`` for element-aware models."""
    return await _run_section(
        client,
        title="🎭 elements + scene_image_urls",
        fields="elements, scene_image_urls",
        select=lambda: _configured_elements_model(client, catalog, names),
        build=lambda params: {
            **params,
            "prompt": "@Element1 trots along a trail in @Image1, cinematic tracking shot",
            # Image-only element: the frontal image doubles as the reference
            # when no extra angles are given.
            "elements": [{"frontal_image_url": PUPPY_URL}],
            "scene_image_urls": [SNOWY_PEAKS_URL],
        },
        output_name="advanced_elements.mp4",
    )


async def main() -> int:
    print("🚀 Venice AI Video — Advanced Fields Example")
    print("=" * 50)
    if RUN_PAID:
        print("💸 VENICE_RUN_PAID_VIDEO=1 — each section will submit a paid job.")
    else:
        print("🧪 Quote only (default). Set VENICE_RUN_PAID_VIDEO=1 to submit.")

    async with VeniceClient() as client:
        try:
            catalog, names = await _load_catalog(client)
        except VeniceError as e:
            print(f"❌ Could not list video models: {e}")
            return 1
        results: dict[str, Outcome] = {
            "reference_image_urls": await reference_images(client, catalog),
            "end_image_url": await end_image(client, catalog),
            "reference_audio_urls": await reference_audio(client, catalog),
            "reference_video_urls": await reference_video(client, catalog),
            "elements_and_scenes": await elements_and_scenes(client, catalog, names),
        }

    print("\n📊 Summary:")
    for name, outcome in results.items():
        if outcome is True:
            label = "✅ generated and verified"
        elif outcome is False:
            label = "❌ failed"
        elif outcome == NOT_PAID:
            label = "🧪 quoted only"
        else:
            label = "⏭️  skipped, no model"
        print(f"   {label}: {name}")

    # The sections are independent demos, so the example has shown its
    # feature when any one of them generated a clip that matched its request.
    return exit_code(list(results.items()), core=None)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
