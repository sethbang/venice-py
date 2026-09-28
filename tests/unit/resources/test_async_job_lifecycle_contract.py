"""Lifecycle contracts shared by the queued media jobs.

``VideoJob``, ``MusicJob`` and ``VoiceChangerJob`` wrap a queue → poll →
download → release flow. These tests pin the behaviour a caller relies on:

* ``download()`` either writes the file it returns or raises; it never hands
  back a path to a file that does not exist.
* Leaving the context manager before the job reached a terminal status is
  surfaced at WARNING: the release endpoint only frees stored media, it does
  not stop generation, so the job keeps running and is billed.
* The release docstrings describe storage cleanup, not cancellation.
* A job the server rejects during validation surfaces from ``wait()`` as a
  generation failure, the same as a ``FAILED`` status.
"""

from __future__ import annotations

import inspect
import logging
import re
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from venice_ai.exceptions import (
    InvalidRequestError,
    VeniceError,
    VideoGenerationError,
    _make_status_error,
)
from venice_ai.resources.music import Music, MusicJob
from venice_ai.resources.video import Video, VideoJob
from venice_ai.resources.voice_changer import VoiceChanger, VoiceChangerJob
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQueueResponse,
)
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
)
from venice_ai.types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerQueueResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Resp400:
    status = 400
    headers: dict[str, str] = {}


def _client() -> Mock:
    client = Mock()
    for name in ("video", "music", "voice_changer"):
        resource = Mock()
        resource.retrieve = AsyncMock()
        resource.cancel = AsyncMock()
        setattr(client, name, resource)
    client.video.cancel.return_value = VideoCompleteResponse(success=True)
    client.music.cancel.return_value = MusicCompleteResponse(success=True)
    client.fetch_external = AsyncMock(return_value=b"BYTES")
    return client


async def _passthrough_to_thread(fn, *args, **kwargs):
    return fn(*args, **kwargs)


async def _no_sleep(_seconds: float) -> None:
    return None


def _video_job(client: Mock) -> VideoJob:
    return VideoJob(client, VideoQueueResponse(model="wan-2.6-text-to-video", queue_id="q-video"))


def _music_job(client: Mock) -> MusicJob:
    return MusicJob(client, MusicQueueResponse(model="some-music-model", queue_id="q-music"))


def _voice_job(client: Mock) -> VoiceChangerJob:
    return VoiceChangerJob(
        client,
        VoiceChangerQueueResponse(
            model="some-vc-model", queue_id="q-vc", status="QUEUED", duration_seconds=3.0
        ),
    )


# job factory, empty completed status, in-progress status
JOB_KINDS = {
    "video": (
        _video_job,
        lambda: VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None),
        lambda: VideoProcessingStatus(
            status="PROCESSING", average_execution_time=1000.0, execution_duration=100.0
        ),
    ),
    "music": (
        _music_job,
        lambda: MusicCompletedStatus(status="COMPLETED", url=None),
        lambda: MusicProcessingStatus(
            status="PROCESSING", average_execution_time=1000.0, execution_duration=100.0
        ),
    ),
    "voice_changer": (
        _voice_job,
        lambda: VoiceChangerCompletedStatus(status="COMPLETED"),
        None,
    ),
}

# Jobs whose release endpoint is a storage delete that runs alongside a
# generation that cannot be stopped from the client.
STORAGE_RELEASE_JOBS = ["video", "music"]


# ---------------------------------------------------------------------------
# download(): never return a path that was not written
# ---------------------------------------------------------------------------


def test_download_matrix_is_not_empty():
    assert len(JOB_KINDS) >= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(JOB_KINDS))
async def test_download_without_any_source_raises(kind, tmp_path, monkeypatch):
    make_job, empty_completed, _ = JOB_KINDS[kind]
    client = _client()
    job = make_job(client)
    monkeypatch.setattr(
        f"venice_ai.resources.{type(job).__module__.rsplit('.', 1)[-1]}.asyncio.to_thread",
        _passthrough_to_thread,
    )
    target = tmp_path / "out.bin"
    try:
        returned = await job.download(target, empty_completed())
    except (VeniceError, ValueError):
        assert not target.exists()
        return
    pytest.fail(
        f"{type(job).__name__}.download() returned {returned!s} for a completed status "
        f"with no inline data, no url and no queue-time download url; "
        f"file exists={target.exists()}"
    )


@pytest.mark.asyncio
async def test_video_download_without_source_raises_generation_error(tmp_path, monkeypatch):
    monkeypatch.setattr("venice_ai.resources.video.asyncio.to_thread", _passthrough_to_thread)
    job = _video_job(_client())
    empty = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)
    with pytest.raises(VideoGenerationError):
        await job.download(tmp_path / "out.mp4", empty)
    assert not (tmp_path / "out.mp4").exists()


# ---------------------------------------------------------------------------
# __aexit__: warn when the job is left running
# ---------------------------------------------------------------------------


def _warnings_for(caplog, queue_id: str) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.levelno >= logging.WARNING and queue_id in r.getMessage()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STORAGE_RELEASE_JOBS)
@pytest.mark.parametrize("state", ["never_polled", "processing"])
@pytest.mark.parametrize("release", ["succeeds", "rejects_400"])
async def test_exit_before_terminal_status_warns(kind, state, release, caplog):
    make_job, _, processing = JOB_KINDS[kind]
    client = _client()
    job = make_job(client)
    if release == "rejects_400":
        getattr(client, kind).cancel.side_effect = InvalidRequestError(
            "Request ID is invalid", response=_Resp400()
        )
    with caplog.at_level(logging.DEBUG):
        async with job:
            if state == "processing":
                job._status = processing()

    warnings = _warnings_for(caplog, job.queue_id)
    assert warnings, (
        f"{type(job).__name__} left its context while {state} (release {release}) "
        f"without a WARNING; the job keeps running and is billed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STORAGE_RELEASE_JOBS)
async def test_exit_after_wait_timeout_warns(kind, caplog):
    make_job, _, processing = JOB_KINDS[kind]
    client = _client()
    job = make_job(client)
    getattr(client, kind).cancel.side_effect = InvalidRequestError(
        "Request ID is invalid", response=_Resp400()
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(TimeoutError):
        async with job:
            job._status = processing()
            raise TimeoutError("generation did not complete")

    assert _warnings_for(caplog, job.queue_id), (
        f"{type(job).__name__} left its context on a timeout while still processing "
        f"without a WARNING; the job keeps running and is billed"
    )


# terminal status per job kind
TERMINAL_STATUSES = {
    ("video", "completed"): lambda: VideoCompletedStatus(
        status="COMPLETED", url="https://x", expires_at=None
    ),
    ("video", "failed"): lambda: VideoFailedStatus(status="FAILED", error="boom"),
    ("music", "completed"): lambda: MusicCompletedStatus(status="COMPLETED", url="https://x"),
    ("music", "failed"): lambda: MusicFailedStatus(status="FAILED", error="boom"),
}


# A release rejected with 400 after a terminal status is the benign
# "already released" case observed for video jobs.
TERMINAL_EXIT_CASES = [
    (kind, terminal, release)
    for kind, terminal in sorted(TERMINAL_STATUSES)
    for release in ("succeeds", "rejects_400")
    if release == "succeeds" or kind == "video"
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "terminal", "release"), TERMINAL_EXIT_CASES)
async def test_exit_after_terminal_status_does_not_warn(kind, terminal, release, caplog):
    make_job, _, _ = JOB_KINDS[kind]
    client = _client()
    job = make_job(client)
    if release == "rejects_400":
        getattr(client, kind).cancel.side_effect = InvalidRequestError(
            "Request ID is invalid", response=_Resp400()
        )
    with caplog.at_level(logging.DEBUG):
        async with job:
            job._status = TERMINAL_STATUSES[(kind, terminal)]()
    assert not _warnings_for(caplog, job.queue_id)


# ---------------------------------------------------------------------------
# Release docstrings describe storage cleanup, not cancellation
# ---------------------------------------------------------------------------

_CANCEL_CLAIM = re.compile(
    r"(cancel|abort|stop)\w*\s+(an?\s+)?in[- ]progress"
    r"|regardless of whether (the job|generation) has finished",
    re.IGNORECASE,
)

RELEASE_METHODS = [
    VideoJob.cancel,
    VideoJob.__aexit__,
    Video.cancel,
    MusicJob.cancel,
    MusicJob.__aexit__,
    Music.cancel,
    VoiceChangerJob.cancel,
    VoiceChanger.cancel,
]


def test_cancel_claim_pattern_is_live():
    assert len(RELEASE_METHODS) > 0
    assert _CANCEL_CLAIM.search("Release server-side storage / cancel an in-progress job.")


@pytest.mark.parametrize("method", RELEASE_METHODS, ids=lambda m: m.__qualname__)
def test_release_docstring_does_not_promise_cancellation(method):
    doc = inspect.getdoc(method) or ""
    match = _CANCEL_CLAIM.search(doc)
    assert match is None, (
        f"{method.__qualname__} docstring promises cancellation ({match.group(0)!r}), "
        "but the release endpoint only deletes stored media"
    )


# ---------------------------------------------------------------------------
# wait(): validation rejections are generation failures
# ---------------------------------------------------------------------------

# Bodies of HTTP 400 responses /video/retrieve returned for queued jobs that
# failed server-side validation.
REJECTION_BODIES = [
    {
        "error": "elements: Value error, Cannot provide both image URLs and video URL "
        "for the same element"
    },
    {
        "error": "The uploaded image dimensions are 1x1; width and height must each "
        "be at least 32 pixels"
    },
    {"error": "Invalid request parameters"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", REJECTION_BODIES, ids=["element", "image-size", "opaque"])
async def test_wait_maps_retrieve_rejection_to_generation_error(body, monkeypatch):
    monkeypatch.setattr("venice_ai.resources.video.asyncio.sleep", _no_sleep)
    client = _client()
    job = _video_job(client)
    rejection = _make_status_error(
        "API request failed with status 400", body=body, response=_Resp400()
    )
    assert isinstance(rejection, InvalidRequestError)
    client.video.retrieve.side_effect = rejection
    with pytest.raises(VideoGenerationError) as exc_info:
        await job.wait()
    assert body["error"] in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, InvalidRequestError)


# ---------------------------------------------------------------------------
# submit(): prompt is optional on the wire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upscale_submit_without_prompt_omits_it_from_body():
    client = MagicMock()
    client.models.get = AsyncMock(side_effect=LookupError("offline"))
    client.post = AsyncMock(
        return_value=VideoQueueResponse(model="topaz-video-upscale", queue_id="q-up")
    )
    video = Video(client)
    await video.submit(  # type: ignore[call-arg]
        model="topaz-video-upscale",
        duration_seconds="Auto",
        video_url="https://example.com/source.mp4",
        upscale_factor=2,
    )
    body = client.post.await_args.kwargs["json_data"]
    assert "prompt" not in body
    assert body["video_url"] == "https://example.com/source.mp4"
    assert body["upscale_factor"] == 2
