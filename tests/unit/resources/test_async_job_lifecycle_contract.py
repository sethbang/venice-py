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
    MusicGenerationError,
    RateLimitError,
    UnprocessableEntityError,
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


# The resource method each job calls to release its stored media.
_RELEASE_METHOD = {"video": "cancel", "music": "release", "voice_changer": "cancel"}


def _client() -> Mock:
    client = Mock()
    for name, release in _RELEASE_METHOD.items():
        resource = Mock()
        resource.retrieve = AsyncMock()
        setattr(resource, release, AsyncMock())
        setattr(client, name, resource)
    client.video.cancel.return_value = VideoCompleteResponse(success=True)
    client.music.release.return_value = MusicCompleteResponse(success=True)
    client.fetch_external = AsyncMock(return_value=b"BYTES")
    return client


def _release_mock(client: Mock, kind: str) -> AsyncMock:
    mock: AsyncMock = getattr(getattr(client, kind), _RELEASE_METHOD[kind])
    return mock


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
        _release_mock(client, kind).side_effect = InvalidRequestError(
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
    _release_mock(client, kind).side_effect = InvalidRequestError(
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
        _release_mock(client, kind).side_effect = InvalidRequestError(
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
    MusicJob.release,
    MusicJob.__aexit__,
    Music.release,
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
    await video.submit(
        model="topaz-video-upscale",
        video_url="https://example.com/source.mp4",
        upscale_factor=2,
    )
    body = client.post.await_args.kwargs["json_data"]
    assert "prompt" not in body
    assert "duration" not in body
    assert body["video_url"] == "https://example.com/source.mp4"
    assert body["upscale_factor"] == 2


# ---------------------------------------------------------------------------
# wait(): terminal retrieve-time rejections, for every queued media job
# ---------------------------------------------------------------------------


class _Resp422:
    status = 422
    headers: dict[str, str] = {}


class _Resp429:
    status = 429
    headers: dict[str, str] = {}


# Body of the HTTP 422 /video/retrieve returned when the provider refused a
# queued job on content policy.
REFUNDED_REJECTION = {
    "error": "The request was rejected by the provider's content policy. "
    "Credits have been refunded."
}

GENERATION_ERRORS = {"video": VideoGenerationError, "music": MusicGenerationError}
REJECTING_JOBS = sorted(GENERATION_ERRORS)


def _job_and_resource(kind: str, monkeypatch) -> tuple[VideoJob | MusicJob, Mock]:
    make_job, _, _ = JOB_KINDS[kind]
    client = _client()
    monkeypatch.setattr(f"venice_ai.resources.{kind}.asyncio.sleep", _no_sleep)
    return make_job(client), getattr(client, kind)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", REJECTING_JOBS)
async def test_wait_maps_retrieve_422_to_generation_error(kind, monkeypatch):
    job, resource = _job_and_resource(kind, monkeypatch)
    rejection = _make_status_error(
        "API request failed with status 422", body=REFUNDED_REJECTION, response=_Resp422()
    )
    assert isinstance(rejection, UnprocessableEntityError)
    resource.retrieve.side_effect = rejection
    with pytest.raises(GENERATION_ERRORS[kind]) as exc_info:
        await job.wait()
    assert "Credits have been refunded" in str(exc_info.value)
    assert exc_info.value.__cause__ is rejection


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", REJECTING_JOBS)
@pytest.mark.parametrize("status", [400, 422])
async def test_retrieve_rejection_is_terminal_and_not_reported_as_billed(
    kind, status, monkeypatch, caplog
):
    job, resource = _job_and_resource(kind, monkeypatch)
    resource.retrieve.side_effect = _make_status_error(
        f"API request failed with status {status}",
        body=REFUNDED_REJECTION,
        response=_Resp400() if status == 400 else _Resp422(),
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(GENERATION_ERRORS[kind]):
        async with job:
            await job.wait()
    assert not _warnings_for(caplog, job.queue_id), (
        f"{type(job).__name__} was rejected at retrieve time (HTTP {status}) but "
        "still warned that the job keeps running and is billed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", REJECTING_JOBS)
async def test_wait_maps_retrieve_400_to_generation_error(kind, monkeypatch):
    job, resource = _job_and_resource(kind, monkeypatch)
    rejection = _make_status_error(
        "API request failed with status 400",
        body={"error": "Invalid request parameters"},
        response=_Resp400(),
    )
    resource.retrieve.side_effect = rejection
    with pytest.raises(GENERATION_ERRORS[kind]) as exc_info:
        await job.wait()
    assert exc_info.value.__cause__ is rejection


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", REJECTING_JOBS)
async def test_unknown_request_id_is_not_a_generation_failure(kind, monkeypatch, caplog):
    job, resource = _job_and_resource(kind, monkeypatch)
    resource.retrieve.side_effect = _make_status_error(
        "API request failed with status 400",
        body={"error": "Request ID is invalid."},
        response=_Resp400(),
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(InvalidRequestError) as exc_info:
        async with job:
            await job.wait()
    assert not isinstance(exc_info.value, tuple(GENERATION_ERRORS.values()))
    assert _warnings_for(caplog, job.queue_id), "an unknown queue id is not a job outcome"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", REJECTING_JOBS)
async def test_rate_limit_at_retrieve_is_not_terminal(kind, monkeypatch, caplog):
    job, resource = _job_and_resource(kind, monkeypatch)
    resource.retrieve.side_effect = _make_status_error(
        "API request failed with status 429",
        body={"error": "Too many requests"},
        response=_Resp429(),
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(RateLimitError):
        async with job:
            await job.wait()
    assert _warnings_for(caplog, job.queue_id), "a rate-limited poll leaves the job running"


# ---------------------------------------------------------------------------
# submit() / run() / quote(): duration is optional for source-timed models
# ---------------------------------------------------------------------------


def _video_catalog_entry(model_id: str, durations: list[str]):
    from venice_ai.types.api.models import (
        ModelResponse,
        VideoModelConstraints,
        VideoModelSpec,
    )

    spec = VideoModelSpec(
        name=model_id,
        constraints=VideoModelConstraints(model_type="video", durations=durations),
    )
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "type": "video",
            "object": "model",
            "owned_by": "venice.ai",
            "model_spec": spec.model_dump(),
        }
    )


def _video_resource(durations: list[str] | None, response) -> tuple[Video, MagicMock]:
    client = MagicMock()
    if durations is None:
        client.models.get = AsyncMock(side_effect=LookupError("offline"))
    else:
        client.models.get = AsyncMock(return_value=_video_catalog_entry("m", durations))
    client.post = AsyncMock(return_value=response)
    return Video(client), client


UPSCALE_QUEUED = VideoQueueResponse(model="topaz-video-upscale", queue_id="q-up")


@pytest.mark.asyncio
async def test_upscale_run_without_duration_queues_without_it():
    video, client = _video_resource(["Auto"], UPSCALE_QUEUED)
    job = await video.run(
        model="topaz-video-upscale",
        video_url="https://example.com/source.mp4",
        upscale_factor=2,
    )
    assert isinstance(job, VideoJob)
    body = client.post.await_args.kwargs["json_data"]
    assert "duration" not in body


@pytest.mark.asyncio
async def test_upscale_quote_without_duration_omits_it():
    from venice_ai.types.api.video import VideoQuoteResponse

    video, client = _video_resource(["Auto"], VideoQuoteResponse(quote=1.1))
    quote = await video.quote(
        model="topaz-video-upscale", video_url="https://example.com/source.mp4"
    )
    assert quote.quote == 1.1
    body = client.post.await_args.kwargs["json_data"]
    assert "duration" not in body
    assert body["video_url"] == "https://example.com/source.mp4"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["submit", "run", "quote"])
async def test_generation_model_without_duration_fails_before_the_request(method):
    video, client = _video_resource(["5s", "10s"], UPSCALE_QUEUED)
    kwargs = {"model": "wan-2-7-text-to-video"}
    if method != "quote":
        kwargs["prompt"] = "a lighthouse at dusk"
    with pytest.raises(ValueError, match=r"duration_seconds is required.*5s"):
        await getattr(video, method)(**kwargs)
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_duration_with_unknown_catalog_is_left_to_the_server():
    video, client = _video_resource(None, UPSCALE_QUEUED)
    await video.submit(model="topaz-video-upscale", video_url="https://example.com/s.mp4")
    body = client.post.await_args.kwargs["json_data"]
    assert "duration" not in body


@pytest.mark.asyncio
async def test_provided_duration_is_still_checked_against_the_catalog():
    video, client = _video_resource(["5s", "10s"], UPSCALE_QUEUED)
    with pytest.raises(ValueError, match="not supported"):
        await video.submit(model="wan-2-7-text-to-video", prompt="x", duration_seconds=7)
    client.post.assert_not_awaited()


def test_duration_is_optional_on_every_video_entry_point():
    for method in (Video.submit, Video.run, Video.quote):
        param = inspect.signature(method).parameters["duration_seconds"]
        assert param.default is None, f"Video.{method.__name__} still requires duration_seconds"
