"""
Shared helpers for Venice AI video examples.

- ``mp4_tracks`` walks an MP4's box tree (``moov`` -> ``trak`` -> ``tkhd`` /
  ``mdia`` -> ``hdlr`` / ``mdhd``) and returns each track's kind, frame size
  and duration, so an example can check what the server returned instead of
  trusting that a file arrived.
- ``check_video`` compares a downloaded clip against the request that
  produced it: duration, aspect ratio, resolution (at least the requested
  tier) and whether a sound track is present. It prints one line per check
  and returns ``False`` when any check fails.
- ``read_video_listing`` splits a ``models.list(type="video")`` listing into
  the entries whose constraints could be read and the IDs of those that
  could not, and ``unrecognised_durations`` lists the duration tiers of one
  model that cannot be read, so a parsing regression fails an example
  instead of looking like a model kind the catalog lacks.
  ``duration_seconds`` reads a duration tier (``"5s"`` -> ``5``).
- ``print_progress`` renders ``VideoJob.wait`` progress updates.
- ``wait_and_check`` waits for a job, saves its video and checks the clip.
- ``download_video`` saves a finished video, retrying a dropped transfer, and
  ``print_recovery`` says how to reach a job that was not released.
- ``read_playable_video`` raises ``UnplayableVideoError`` for a saved file that
  is not an MP4 with a video track. Call it inside the job's ``async with``
  block, so that a bad delivery keeps the stored video instead of releasing it.
- ``report_uncertain_queue`` tells a queue call the server may have accepted
  from one it refused or never received, and says what that means for the bill.
- ``resolution_height`` sorts resolution tiers, ``video_constraints`` reads a
  model's catalog constraints, and ``report_error`` prints a video API failure.
- ``url_serves`` is a free pre-flight: it confirms a media URL still answers with
  the expected kind of content, before a paid job depends on it.
- ``display_url`` gives a URL in a form safe to print: a signed link's query can
  grant access on its own.
- ``spending_ceiling`` reads the most one job may quote, and ``ceiling_refusal``
  turns a quote above it into a skip.
- ``SKIPPED`` is the exit code an example returns when an optional
  prerequisite is missing (paid generation not enabled, no matching model);
  it prints a line starting with ``SKIPPED:`` first.
- ``exit_code`` turns per-section outcomes into the example's exit code:
  ``1`` if any section failed, ``SKIPPED`` if nothing the example exists to
  show was verified, else ``0``. That is the ``core`` section when the
  example has one, or any section when its sections are independent demos.
  Other skipped sections print a ``Section skipped:`` line.
"""

import asyncio
import os
import struct
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from venice_ai import ModelQuotesUnavailableError, VeniceClient, VideoGenerationError
from venice_ai.exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    VeniceError,
)
from venice_ai.resources.video import VideoJob
from venice_ai.types.api.models import ModelResponse, VideoModelConstraints, VideoModelSpec
from venice_ai.types.api.video import VideoCompletedStatus, VideoProcessingStatus

#: Exit code for "skipped": a missing optional prerequisite, not a failure.
SKIPPED = 77

#: The most one paid job may quote before it is refused, in USD.
VIDEO_CEILING_USD = 0.50
#: Set this to spend more than the ceiling on one generation.
VIDEO_CEILING_ENV_VAR = "VENICE_VIDEO_MAX_USD"

#: A section's result: ``True`` passed, ``False`` failed, a string is the
#: reason it was skipped.
Outcome = bool | str

#: Skip reason for generation sections when paid runs are not enabled.
NOT_PAID = "VENICE_RUN_PAID_VIDEO is not set, so no video was generated"

#: Skip reason for paid sections after a queue call whose outcome is unknown.
QUEUE_UNCERTAIN = "an earlier queue call may have been billed, so no further job was queued"

#: Tries a finished video's download gets before the last error is raised.
DOWNLOAD_ATTEMPTS = 3

# Allowed gap between the requested and the delivered clip length, in
# seconds. Encoders round to whole frames, so a 4s request comes back as
# about 4.04s.
DURATION_TOLERANCE_S = 0.5
# Allowed relative gap between the requested and the delivered aspect ratio.
# Encoders round frame sizes to multiples of 8 or 16 (16:9 at 480p is often
# 864x496, about 2% off).
ASPECT_TOLERANCE = 0.05
# A resolution tier such as "480p" names the frame's short side: 848x480 at
# 16:9, 640x480 at 4:3, 480x848 at 9:16. Encoders round each dimension to
# whole 16-pixel macroblocks, so a short side up to this many pixels below the
# tier still counts (a "360p" frame can be 352 lines).
MACROBLOCK_PX = 16


@dataclass
class Mp4Track:
    """One track of an MP4 file."""

    handler: str  # "vide" for video, "soun" for audio
    width: int  # 0 for audio tracks
    height: int
    seconds: float | None


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(type, payload_start, box_end)`` for the boxes in ``data[start:end]``."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1:  # 64-bit size follows the type
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:  # box runs to the end of its parent
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def mp4_tracks(data: bytes) -> list[Mp4Track]:
    """List the tracks declared in an MP4's ``moov`` box."""
    tracks: list[Mp4Track] = []
    for kind, start, end in _boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for trak, t_start, t_end in _boxes(data, start, end):
            if trak != b"trak":
                continue
            handler, width, height, seconds = "", 0, 0, None
            for box, b_start, b_end in _boxes(data, t_start, t_end):
                if box == b"tkhd":
                    # Width and height are the box's last two 16.16 fields.
                    width, height = (v >> 16 for v in struct.unpack(">II", data[b_end - 8 : b_end]))
                elif box == b"mdia":
                    for sub, s_start, _ in _boxes(data, b_start, b_end):
                        if sub == b"hdlr":
                            # version/flags (4) + pre_defined (4) + handler type
                            handler = data[s_start + 8 : s_start + 12].decode("latin-1")
                        elif sub == b"mdhd":
                            if data[s_start] == 1:  # version 1: 64-bit times
                                scale, duration = struct.unpack(
                                    ">IQ", data[s_start + 20 : s_start + 32]
                                )
                            else:
                                scale, duration = struct.unpack(
                                    ">II", data[s_start + 12 : s_start + 20]
                                )
                            seconds = duration / scale if scale and duration else None
            tracks.append(Mp4Track(handler, width, height, seconds))
    return tracks


def _tier_height(value: str) -> int | None:
    """``"480p"`` / ``"768P"`` -> 480 / 768; ``"2k"`` -> 1440, ``"4k"`` -> 2160."""
    tier = value.strip().lower()
    if tier.endswith("k") and tier[:-1].isdigit():
        return {2: 1440, 4: 2160, 8: 4320}.get(int(tier[:-1]))
    digits = tier.removesuffix("p")
    return int(digits) if digits.isdigit() else None


def resolution_height(value: str) -> int:
    """Sort key for resolution tiers: ``"720p"`` -> 720, ``"4k"`` -> 2160."""
    return _tier_height(value) or 0


async def video_constraints(client: VeniceClient, model: str) -> VideoModelConstraints:
    """Return the video constraints the catalog lists for ``model``."""
    entry = await client.models.get(model)
    spec = entry.model_spec
    if not isinstance(spec, VideoModelSpec) or spec.constraints is None:
        raise ValueError(f"{model} has no video constraints in the catalog")
    return spec.constraints


def report_error(e: Exception) -> None:
    """Print a failed video API call, with a hint where one helps."""
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


def url_serves(url: str, expected_prefix: str) -> bool:
    """Free pre-flight: whether ``url`` still answers with ``expected_prefix`` content.

    ``expected_prefix`` is the start of the media type, such as ``"image/"`` or
    ``"video/"``. The server fetches a media URL only after the job is queued, so
    a dead link would otherwise surface as a failed paid job. A ``data:`` URI
    carries its own content and is not fetched. Only the headers are read.
    """
    if url.startswith("data:"):
        return True
    request = urllib.request.Request(url, headers={"User-Agent": "venice-py-example"})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - fixed or user-set URL
            content_type = resp.headers.get("Content-Type", "")
    except OSError as e:
        print(f"❌ {display_url(url)} is unreachable: {e}")
        return False
    if not content_type.startswith(expected_prefix):
        print(f"❌ {display_url(url)} returned {content_type!r}, not {expected_prefix}*")
        return False
    return True


def display_url(url: str) -> str:
    """A URL in a form safe to print, without its query or fragment.

    A signed link's query can grant access on its own, and a ``data:`` URI is the
    whole file.
    """
    if url.startswith("data:"):
        return "a data: URI"
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def spending_ceiling(env_var: str = VIDEO_CEILING_ENV_VAR) -> float:
    """The most one job may quote: *env_var*, or ``VIDEO_CEILING_USD``.

    Raises:
        ValueError: If the variable is set to something other than a number.
    """
    raw = os.environ.get(env_var)
    return float(raw) if raw else VIDEO_CEILING_USD


def ceiling_refusal(quote_usd: float, env_var: str = VIDEO_CEILING_ENV_VAR) -> Outcome | None:
    """Refuse a job quoted above the spending ceiling.

    A quote leaves out reference images, audio, elements and keyframes, so a job
    using them can cost more than its quote and more than the ceiling.

    Returns:
        ``None`` if the job may go ahead, the reason to skip it if its quote is over
        the ceiling, or ``False`` if the variable is not a number.
    """
    try:
        ceiling = spending_ceiling(env_var)
    except ValueError as e:
        print(f"❌ {env_var} is not a number: {e}")
        return False
    if quote_usd <= ceiling:
        return None
    reason = (
        f"the ${quote_usd:.4f} quote is over the ${ceiling:.2f} ceiling; "
        f"set {env_var} to spend more"
    )
    print(f"⏭️  Not queued: {reason}.")
    return reason


def duration_seconds(value: Any) -> int | None:
    """``"4s"`` -> ``4``; ``None`` for non-numeric tiers such as ``"Auto"``."""
    digits = str(value).strip().lower().removesuffix("s")
    return int(digits) if digits.isdigit() else None


#: The duration tier of models that choose the clip length themselves.
MODEL_CHOSEN_DURATION = "auto"


@dataclass
class VideoListing:
    """A video catalog listing split by whether each entry could be read."""

    #: ``(model_id, spec, constraints)`` for every entry that was read.
    models: list[tuple[str, VideoModelSpec, VideoModelConstraints]]
    #: ``"model_id: reason"`` for every entry that could not be read.
    unreadable: list[str]


def read_video_listing(entries: Iterable[ModelResponse]) -> VideoListing:
    """Read the constraints of every entry in a ``models.list(type="video")`` listing.

    An entry is read when its ``model_spec`` is a ``VideoModelSpec`` with
    constraints and a ``model_type``. Any other entry is reported as
    unreadable rather than dropped: a model the catalog lists but the SDK
    cannot parse could be of any kind, so it is a failure, not a missing
    model kind. Check the durations of the models you use with
    ``unrecognised_durations``. Offline models are read too; filter on
    ``spec.offline`` afterwards.
    """
    listing = VideoListing(models=[], unreadable=[])
    for entry in entries:
        spec = entry.model_spec
        if not isinstance(spec, VideoModelSpec):
            listing.unreadable.append(f"{entry.id}: spec is {type(spec).__name__}")
            continue
        constraints = spec.constraints
        if constraints is None:
            listing.unreadable.append(f"{entry.id}: no constraints")
            continue
        if not constraints.model_type:
            listing.unreadable.append(f"{entry.id}: no model_type")
            continue
        listing.models.append((entry.id, spec, constraints))
    return listing


def unrecognised_durations(constraints: VideoModelConstraints) -> list[str]:
    """The listed duration tiers that are neither ``"<n>s"`` nor ``"Auto"``.

    Every tier is a whole number of seconds or the model-chosen ``"Auto"``,
    so anything else means the listing changed shape and the model's
    durations cannot be filtered on.
    """
    return [
        d
        for d in constraints.durations
        if duration_seconds(d) is None and d.strip().lower() != MODEL_CHOSEN_DURATION
    ]


def meets_tier(width: int, height: int, tier_height: int) -> bool:
    """True if a ``width`` x ``height`` frame is at least the requested tier.

    The frame passes when its short side reaches the tier (``640x480`` for
    ``480p`` at 4:3) or, for providers that keep the pixel count rather than
    the line count at wide aspect ratios such as 21:9, when it holds as many
    pixels as the tier's 16:9 frame. Both allow one macroblock of encoder
    rounding. A frame from a lower tier fails both: ``640x360`` for ``480p``,
    ``960x540`` for ``720p``. A larger frame than requested passes.
    """
    lines = tier_height - MACROBLOCK_PX
    return min(width, height) >= lines or width * height >= lines * lines * 16 / 9


class UnplayableVideoError(Exception):
    """A delivered file is not an MP4 with a video track."""


def _require_video_track(data: bytes) -> tuple[list[Mp4Track], Mp4Track]:
    """Return an MP4's tracks and its video track.

    Raises :class:`UnplayableVideoError` when ``data`` has neither.
    """
    if data[4:8] != b"ftyp":
        raise UnplayableVideoError(f"Not an MP4 file (header {data[:12]!r})")
    try:
        tracks = mp4_tracks(data)
    except (struct.error, IndexError) as e:
        raise UnplayableVideoError(f"Malformed MP4 ({e})") from e
    video = next((t for t in tracks if t.handler == "vide" and t.width and t.height), None)
    if video is None:
        raise UnplayableVideoError(f"No video track found (tracks: {tracks})")
    return tracks, video


def read_playable_video(path: Path) -> bytes:
    """Read a saved video, raising :class:`UnplayableVideoError` if it cannot be one.

    A download can succeed with the wrong bytes, such as an HTML page answering
    200. Call this inside the job's ``async with`` block: leaving the block
    cleanly releases the stored video, and this raises through it to keep it.
    """
    data = path.read_bytes()
    _require_video_track(data)
    return data


def check_video(data: bytes, request: Mapping[str, Any], *, expect_audio: bool | None) -> bool:
    """Check a downloaded clip against the request that produced it.

    ``request`` holds the ``duration_seconds`` / ``resolution`` /
    ``aspect_ratio`` that were sent; a missing key is not checked.
    ``expect_audio`` says whether a sound track must be present (``True``),
    absent (``False``) or is not checked (``None``).
    """
    try:
        tracks, video = _require_video_track(data)
    except UnplayableVideoError as e:
        print(f"❌ {e}")
        return False
    has_audio = any(t.handler == "soun" for t in tracks)
    ok = True

    def report(passed: bool, label: str) -> None:
        nonlocal ok
        print(f"   {'✅' if passed else '❌'} {label}")
        ok = ok and passed

    length = f"{video.seconds:.2f}s" if video.seconds is not None else "length unknown"
    print(f"🔎 Delivered: {video.width}x{video.height}, {length}, audio={has_audio}")

    wanted_s = duration_seconds(request.get("duration_seconds"))
    if wanted_s is not None:
        if video.seconds is None:
            report(False, f"duration for a {wanted_s}s request: not recorded in the track header")
        else:
            report(
                abs(video.seconds - wanted_s) <= DURATION_TOLERANCE_S,
                f"duration {video.seconds:.2f}s for a {wanted_s}s request",
            )

    ratio = request.get("aspect_ratio")
    if isinstance(ratio, str) and ":" in ratio:
        w, h = (float(part) for part in ratio.split(":", 1))
        actual = video.width / video.height
        report(
            abs(actual - w / h) / (w / h) <= ASPECT_TOLERANCE,
            f"aspect {actual:.3f} for a {ratio} request",
        )

    tier = request.get("resolution")
    height = _tier_height(tier) if isinstance(tier, str) else None
    if height is not None:
        report(
            meets_tier(video.width, video.height, height),
            f"{video.width}x{video.height} for a {tier} request "
            f"(needs a short side of at least {height - MACROBLOCK_PX} px or as many pixels "
            f"as a 16:9 {tier} frame; larger is accepted)",
        )

    if expect_audio is not None:
        report(has_audio == expect_audio, f"sound track present={has_audio}, wanted {expect_audio}")
    return ok


def print_progress(status: VideoProcessingStatus) -> None:
    """Render a one-line progress update suitable for ``VideoJob.wait``."""
    remaining_s = status.estimated_remaining_ms / 1000 if status.estimated_remaining_ms else 0
    print(f"⏳ Processing… {status.progress_percent:.0f}% (~{remaining_s:.0f}s remaining)")


def exit_code(results: list[tuple[str, Outcome]], core: str | None) -> int:
    """Exit code for an example, from its sections' outcomes.

    ``core`` names the section that holds the example's main feature. Pass
    ``None`` for an example whose sections are independent demos with no
    single main one: it has then verified its feature when any section passed.

    Prints the failed sections, a ``Section skipped:`` line per skipped
    section other than ``core``, and a ``SKIPPED:`` line when nothing the
    example exists to show was verified: ``core`` was skipped or, without a
    ``core``, no section passed. A failure always wins over a skip.
    """
    outcomes = dict(results)
    for name, outcome in results:
        if isinstance(outcome, str) and name != core:
            print(f"Section skipped: {name}: {outcome}")
    failed = [name for name, outcome in results if outcome is False]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1
    if core is not None:
        core_outcome = outcomes[core]
        if isinstance(core_outcome, str):
            print(f"\nSKIPPED: {core_outcome}")
            return SKIPPED
        return 0
    if any(outcome is True for outcome in outcomes.values()):
        return 0
    reasons = list(dict.fromkeys(o for o in outcomes.values() if isinstance(o, str)))
    reason = (
        reasons[0]
        if len(reasons) == 1
        else "no section generated a verified clip: " + "; ".join(reasons)
    )
    print(f"\nSKIPPED: {reason}")
    return SKIPPED


async def download_video(job: VideoJob, output_path: Path, status: VideoCompletedStatus) -> Path:
    """Save a finished video, retrying a dropped transfer.

    A model that returns a ``download_url`` hands the link out once, at queue
    time. Unless ``job.download_url`` was saved, this process is the one chance
    to save the billed video. The link is short-lived but survives a few
    retries, and fetching it costs nothing. The last failure is raised.
    """
    for attempt in range(1, DOWNLOAD_ATTEMPTS):
        try:
            return await job.download(output_path, status)
        except (APITimeoutError, APIConnectionError) as e:
            print(f"⚠️  Download attempt {attempt} of {DOWNLOAD_ATTEMPTS} failed: {e}")
            await asyncio.sleep(2 * attempt)
    return await job.download(output_path, status)


async def wait_and_check(
    job: VideoJob,
    output_path: Path,
    request: Mapping[str, Any],
    written: list[Path],
    *,
    expect_audio: bool | None = None,
) -> bool:
    """Wait for a job, save its video and check the clip against ``request``.

    Each saved file is added to ``written``. The handlers sit outside the block:
    leaving it cleanly releases a finished job's stored video, so a failed
    download has to raise through it. Any failure prints its reason and returns
    ``False``.
    """
    print(f"⏳ Polling for completion (queue_id={job.queue_id}) ...")
    try:
        async with job:
            status = await job.wait(on_progress=print_progress)
            saved = await download_video(job, output_path, status)
            data = read_playable_video(saved)
    except UnplayableVideoError as e:
        written.append(output_path)
        print(f"❌ {output_path} is not a usable video: {e} (queue_id={job.queue_id})")
        print_recovery(job)
        return False
    except VideoGenerationError as e:
        code = f" (code={e.error_code})" if e.error_code else ""
        print(f"❌ Generation failed: {e}{code}")
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
        print(f"❌ Video request failed (queue_id={job.queue_id}): {e}")
        print_recovery(job)
        return False

    written.append(saved)
    print(f"💾 Saved to {saved} ({len(data):,} bytes)")
    if not check_video(data, request, expect_audio=expect_audio):
        print("❌ The clip does not match the request")
        return False
    print("✅ Video ready and matches the request")
    return True


def queue_outcome_unknown(e: Exception) -> bool:
    """True if a failed ``video.run()`` may still have queued, and billed, a job.

    A job is billed when it is queued, and the SDK does not resend a queue
    request the server may have processed. A 5xx, a read timeout, a dropped
    connection or an unreadable success body can all come after the job was
    accepted. A ``ValueError`` is raised before anything is sent, a 4xx is the
    server refusing the job, and a connection that was never made sent nothing.
    """
    if not isinstance(e, VeniceError):
        return False
    if isinstance(e, APIError) and e.status_code is not None and 400 <= e.status_code < 500:
        return False
    return not (
        isinstance(e, APIConnectionError | APITimeoutError)
        and isinstance(
            e.original_error, aiohttp.ClientConnectorError | aiohttp.ConnectionTimeoutError
        )
    )


def report_uncertain_queue(e: Exception) -> bool:
    """Say so if a failed queue call may have been billed; ``True`` if it was printed."""
    if not queue_outcome_unknown(e):
        return False
    print(f"⚠️  {type(e).__name__}: {e}")
    print("   Outcome unknown: the job may be queued and billed. Check your usage before")
    print("   re-running this example, because a re-run pays for a new job.")
    return True


def print_recovery(job: VideoJob) -> None:
    """Say how to reach a job that left its block with an error, so was not released."""
    print(
        "   Its stored video was not released. Fetch it with "
        f"client.video.retrieve(model={job.model!r}, queue_id={job.queue_id!r}); a model "
        "that gives its download link only at queue time cannot be fetched once this "
        "process exits, unless job.download_url was saved."
    )
