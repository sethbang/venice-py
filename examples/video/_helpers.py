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
- ``SKIPPED`` is the exit code an example returns when an optional
  prerequisite is missing (paid generation not enabled, no matching model);
  it prints a line starting with ``SKIPPED:`` first.
- ``exit_code`` turns per-section outcomes into the example's exit code:
  ``1`` if any section failed, ``SKIPPED`` if nothing the example exists to
  show was verified, else ``0``. That is the ``core`` section when the
  example has one, or any section when its sections are independent demos.
  Other skipped sections print a ``Section skipped:`` line.
"""

import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from venice_ai.types.api.models import ModelResponse, VideoModelConstraints, VideoModelSpec
from venice_ai.types.api.video import VideoProcessingStatus

#: Exit code for "skipped": a missing optional prerequisite, not a failure.
SKIPPED = 77

#: A section's result: ``True`` passed, ``False`` failed, a string is the
#: reason it was skipped.
Outcome = bool | str

#: Skip reason for generation sections when paid runs are not enabled.
NOT_PAID = "VENICE_RUN_PAID_VIDEO is not set, so no video was generated"

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


def check_video(data: bytes, request: Mapping[str, Any], *, expect_audio: bool | None) -> bool:
    """Check a downloaded clip against the request that produced it.

    ``request`` holds the ``duration_seconds`` / ``resolution`` /
    ``aspect_ratio`` that were sent; a missing key is not checked.
    ``expect_audio`` says whether a sound track must be present (``True``),
    absent (``False``) or is not checked (``None``).
    """
    if data[4:8] != b"ftyp":
        print(f"❌ Not an MP4 file (header {data[:12]!r})")
        return False
    tracks = mp4_tracks(data)
    video = next((t for t in tracks if t.handler == "vide" and t.width and t.height), None)
    if video is None:
        print(f"❌ No video track found (tracks: {tracks})")
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
