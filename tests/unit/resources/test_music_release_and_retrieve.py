"""Music job release semantics, retrieve format detection and duration preflight.

Venice's ``/audio/complete`` deletes stored media; it does not stop a running
job, and generation is billed when the job is queued. These tests pin the
behavior that follows from that: a job still generating is not "released"
on exit, a ``success: false`` cleanup is surfaced, and a completed retrieve is
recognised by its ``Content-Type`` rather than by "anything that is not JSON".
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.exceptions import APIResponseProcessingError, InvalidRequestError, NotFoundError
from venice_ai.resources.music import Music, MusicJob, _preflight_validate_music_duration
from venice_ai.types.api.models import ModelResponse, MusicModelSpec
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQueueResponse,
)


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers: dict[str, str] = {}


_PROCESSING = MusicProcessingStatus(
    status="PROCESSING", average_execution_time=1000.0, execution_duration=900.0
)


@pytest.fixture
def client() -> Mock:
    client = Mock()
    client.music = Mock()
    client.music.retrieve = AsyncMock(return_value=_PROCESSING)
    client.music.release = AsyncMock(return_value=MusicCompleteResponse(success=True))
    return client


@pytest.fixture
def job(client: Mock) -> MusicJob:
    return MusicJob(client, MusicQueueResponse(model="ace-step-15", queue_id="q-1"))


# ---------------------------------------------------------------------------
# Leaving the context
# ---------------------------------------------------------------------------


async def test_exit_while_still_generating_does_not_release(
    job: MusicJob, client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        async with job:
            pass
    client.music.retrieve.assert_awaited_once()
    client.music.release.assert_not_awaited()
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "q-1" in message
    assert "still generating" in message


async def test_exit_after_final_poll_completes_releases(job: MusicJob, client: Mock) -> None:
    client.music.retrieve.return_value = MusicCompletedStatus(status="COMPLETED")
    async with job:
        pass
    client.music.release.assert_awaited_once_with(model="ace-step-15", queue_id="q-1")


async def test_exit_with_an_exception_keeps_completed_audio(
    job: MusicJob, client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    client.music.retrieve.return_value = MusicCompletedStatus(status="COMPLETED")
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="disk full"):
        async with job:
            await job.wait(poll_interval=0)
            raise OSError("disk full")
    client.music.release.assert_not_awaited()
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "q-1" in message
    assert "not released" in message


async def test_exit_with_an_exception_releases_a_failed_job(job: MusicJob, client: Mock) -> None:
    job._status = MusicFailedStatus(status="FAILED", error="boom")
    with pytest.raises(RuntimeError):
        async with job:
            raise RuntimeError("caller gave up")
    client.music.release.assert_awaited_once()


async def test_explicit_release_inside_a_failing_block_is_honored(
    job: MusicJob, client: Mock
) -> None:
    with pytest.raises(RuntimeError):
        async with job:
            await job.release()
            raise RuntimeError("discarded on purpose")
    client.music.release.assert_awaited_once()


async def test_exit_when_media_is_already_gone_neither_releases_nor_warns(
    job: MusicJob, client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    client.music.retrieve.side_effect = NotFoundError(
        "Media could not be found. Request may may be invalid, expired, or deleted.",
        response=_Resp(404),
    )
    with caplog.at_level(logging.WARNING):
        async with job:
            pass
    client.music.release.assert_not_awaited()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_exit_for_an_unknown_queue_id_neither_releases_nor_warns(
    job: MusicJob, client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    client.music.retrieve.side_effect = InvalidRequestError(
        "Request ID is invalid.", response=_Resp(400), body={"error": "Request ID is invalid."}
    )
    with caplog.at_level(logging.WARNING):
        async with job:
            pass
    client.music.release.assert_not_awaited()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_wait_lets_a_media_gone_404_through(job: MusicJob, client: Mock) -> None:
    client.music.retrieve.side_effect = NotFoundError(
        "Media could not be found.", response=_Resp(404)
    )
    with pytest.raises(NotFoundError):
        await job.wait(poll_interval=0)


async def test_exit_after_wait_completed_skips_the_extra_poll(job: MusicJob, client: Mock) -> None:
    client.music.retrieve.return_value = MusicCompletedStatus(status="COMPLETED")
    async with job:
        await job.wait(poll_interval=0)
    assert client.music.retrieve.await_count == 1
    client.music.release.assert_awaited_once()


async def test_unsuccessful_release_is_logged_once(
    job: MusicJob, client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    stub = _PostOnlyClient()
    stub.post.return_value = MusicCompleteResponse(success=False)
    music = Music(stub)  # type: ignore[arg-type]
    music.retrieve = AsyncMock(return_value=MusicCompletedStatus(status="COMPLETED"))  # type: ignore[method-assign]
    client.music = music
    with caplog.at_level(logging.WARNING):
        async with job:
            pass
    assert sum("did not complete" in r.getMessage() for r in caplog.records) == 1


async def test_wait_can_resume_after_a_timeout(job: MusicJob, client: Mock) -> None:
    with pytest.raises(TimeoutError):
        await job.wait(poll_interval=0, max_polls=2)
    client.music.retrieve.return_value = MusicCompletedStatus(status="COMPLETED")
    status = await job.wait(poll_interval=0)
    assert isinstance(status, MusicCompletedStatus)


# ---------------------------------------------------------------------------
# Music.release
# ---------------------------------------------------------------------------


class _PostOnlyClient:
    def __init__(self) -> None:
        self._api_key = "test-key"
        self.post = AsyncMock(return_value=MusicCompleteResponse(success=True))


async def test_resource_release_posts_to_audio_complete() -> None:
    stub = _PostOnlyClient()
    music = Music(stub)  # type: ignore[arg-type]
    await music.release(model="ace-step-15", queue_id="q-1")
    stub.post.assert_awaited_once()
    assert stub.post.await_args.args[0] == "audio/complete"
    assert stub.post.await_args.kwargs["json_data"] == {
        "model": "ace-step-15",
        "queue_id": "q-1",
    }


def test_music_has_no_cancel_alias() -> None:
    # ``/audio/complete`` releases media; it cancels nothing, so the only
    # name for it is ``release``.
    assert not hasattr(Music, "cancel")
    assert not hasattr(MusicJob, "cancel")


async def test_resource_release_logs_unsuccessful_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stub = _PostOnlyClient()
    stub.post.return_value = MusicCompleteResponse(success=False)
    music = Music(stub)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        response = await music.release(model="ace-step-15", queue_id="q-1")
    assert response.success is False
    assert any("did not complete" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Retrieve: format from Content-Type
# ---------------------------------------------------------------------------


def _raw(body: bytes, content_type: str) -> Mock:
    resp = Mock()
    resp.headers = {"content-type": content_type}
    resp.status = 200
    resp.content_length = len(body)
    resp.read = AsyncMock(return_value=body)
    resp.text = AsyncMock(return_value=body.decode("utf-8", errors="replace"))
    resp.close = Mock()
    return resp


async def _retrieve(body: bytes, content_type: str) -> MusicCompletedStatus:
    stub = _PostOnlyClient()
    stub.post = AsyncMock(return_value=_raw(body, content_type))
    result = await Music(stub).retrieve(model="m", queue_id="q")  # type: ignore[arg-type]
    assert isinstance(result, MusicCompletedStatus)
    return result


@pytest.mark.parametrize(
    ("content_type", "audio_format"),
    [
        ("audio/flac", "flac"),
        ("audio/mpeg", "mp3"),
        ("audio/wav", "wav"),
        ("audio/x-wav", "wav"),
        ("audio/mp4", "m4a"),
        ("audio/x-m4a", "m4a"),
        ("audio/mpeg; charset=binary", "mp3"),
    ],
)
async def test_audio_content_types_are_completed(content_type: str, audio_format: str) -> None:
    result = await _retrieve(b"\x00audio-bytes", content_type)
    assert result.content_type == content_type.split(";")[0]
    assert result.audio_format == audio_format
    assert result.data == b"\x00audio-bytes"


async def test_octet_stream_is_sniffed_past_an_id3_tag() -> None:
    # An ID3v2 header (10 bytes, syncsafe size 4) precedes the FLAC marker.
    body = b"ID3\x04\x00\x00\x00\x00\x00\x04" + b"TAGS" + b"fLaC" + b"\x00" * 8
    result = await _retrieve(body, "application/octet-stream")
    assert result.audio_format == "flac"


async def test_unrecognised_octet_stream_is_rejected() -> None:
    with pytest.raises(APIResponseProcessingError, match="octet-stream"):
        await _retrieve(b"not audio at all", "application/octet-stream")


@pytest.mark.parametrize("content_type", ["text/html", "text/plain", ""])
async def test_non_audio_bodies_are_rejected(content_type: str) -> None:
    with pytest.raises(APIResponseProcessingError, match="preview"):
        await _retrieve(b"<html>captive portal</html>", content_type)


async def test_empty_audio_body_is_rejected() -> None:
    with pytest.raises(APIResponseProcessingError, match="empty"):
        await _retrieve(b"", "audio/mpeg")


def test_json_completed_status_has_no_format() -> None:
    status = MusicCompletedStatus.model_validate({"status": "COMPLETED", "url": "https://x/y"})
    assert status.content_type is None
    assert status.audio_format is None


# ---------------------------------------------------------------------------
# Duration preflight
# ---------------------------------------------------------------------------


def _client_with(spec: MusicModelSpec) -> Mock:
    entry = ModelResponse.model_validate(
        {
            "id": "lyria-3-pro",
            "type": "music",
            "object": "model",
            "owned_by": "venice.ai",
            "model_spec": spec.model_dump(),
        }
    )
    stub = Mock()
    stub.models = Mock()
    stub.models.get = AsyncMock(return_value=entry)
    return stub


async def test_duration_on_a_model_without_duration_metadata_raises() -> None:
    stub = _client_with(MusicModelSpec(name="Lyria"))
    with pytest.raises(ValueError, match="does not take duration_seconds"):
        await _preflight_validate_music_duration(stub, "lyria-3-pro", 10)
