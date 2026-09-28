"""Unit tests for VideoJob lifecycle management."""

import logging
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.exceptions import InvalidRequestError, VideoGenerationError
from venice_ai.resources.video import Video, VideoJob
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
)


@pytest.fixture
def queue_response():
    return VideoQueueResponse(model="wan-2.6-text-to-video", queue_id="q-1")


@pytest.fixture
def mock_client():
    client = Mock()
    client.video = Mock()
    client.video.retrieve = AsyncMock()
    client.video.cancel = AsyncMock(return_value=VideoCompleteResponse(success=True))
    client.fetch_external = AsyncMock(return_value=b"VIDEO_BYTES")
    return client


@pytest.fixture
def job(mock_client, queue_response):
    return VideoJob(mock_client, queue_response)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


def test_init_sets_fields(job, queue_response):
    assert job.model == queue_response.model
    assert job.queue_id == queue_response.queue_id
    assert job.status is None
    assert job.is_complete is False
    assert job.is_failed is False
    assert job.progress is None


def test_progress_with_processing_status(job):
    job._status = VideoProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=250.0,
    )
    assert job.progress == pytest.approx(0.25)


def test_is_complete_true_after_completed_status(job):
    job._status = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    assert job.is_complete is True
    assert job.is_failed is False


def test_is_failed_true_after_failed_status(job):
    job._status = VideoFailedStatus(status="FAILED", error="boom", error_code="E1")
    assert job.is_failed is True
    assert job.is_complete is False


# ---------------------------------------------------------------------------
# poll() / wait()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_updates_status(job, mock_client):
    expected = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.video.retrieve.return_value = expected
    result = await job.poll()
    assert result is expected
    assert job.status is expected
    mock_client.video.retrieve.assert_awaited_once_with(model=job.model, queue_id=job.queue_id)


@pytest.mark.asyncio
async def test_wait_returns_completed(job, mock_client, monkeypatch):
    completed = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.video.retrieve.return_value = completed

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("venice_ai.resources.video.asyncio.sleep", _fake_sleep)
    result = await job.wait(poll_interval=0.5)
    assert result is completed
    # First poll already returned COMPLETED, so no sleeps should have occurred.
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_wait_invokes_progress_callback_then_completes(job, mock_client, monkeypatch):
    processing = VideoProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=300.0,
    )
    completed = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.video.retrieve.side_effect = [processing, completed]

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("venice_ai.resources.video.asyncio.sleep", _no_sleep)
    seen: list[VideoProcessingStatus] = []
    result = await job.wait(on_progress=seen.append)
    assert result is completed
    assert seen == [processing]


@pytest.mark.asyncio
async def test_wait_raises_video_generation_error_on_failed(job, mock_client, monkeypatch):
    failed = VideoFailedStatus(status="FAILED", error="boom", error_code="E1")
    mock_client.video.retrieve.return_value = failed

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("venice_ai.resources.video.asyncio.sleep", _no_sleep)
    with pytest.raises(VideoGenerationError) as exc:
        await job.wait()
    assert exc.value.error_code == "E1"


@pytest.mark.asyncio
async def test_wait_raises_timeout_after_max_polls(job, mock_client, monkeypatch):
    processing = VideoProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=0.0,
    )
    mock_client.video.retrieve.return_value = processing

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("venice_ai.resources.video.asyncio.sleep", _no_sleep)
    with pytest.raises(TimeoutError):
        await job.wait(max_polls=3)


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_inline_data_writes_via_to_thread(job, tmp_path, monkeypatch):
    completed = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)
    completed._set_data(b"INLINE_DATA")

    to_thread_calls: list[str] = []

    async def _record_to_thread(fn, *args, **kwargs):
        to_thread_calls.append(fn.__name__)
        return fn(*args, **kwargs)

    monkeypatch.setattr("venice_ai.resources.video.asyncio.to_thread", _record_to_thread)
    target = tmp_path / "out.mp4"
    result = await job.download(target, completed)
    assert result == target
    assert target.read_bytes() == b"INLINE_DATA"
    # Both the mkdir AND the write_bytes must go through to_thread.
    assert "write_bytes" in to_thread_calls
    assert "mkdir" in to_thread_calls


@pytest.mark.asyncio
async def test_download_falls_back_to_queue_download_url(mock_client, tmp_path, monkeypatch):
    """For VPS models the file URL comes from the queue-time
    download_url; retrieve returns JSON status with no url/data. download()
    must fall back to the stored download_url."""
    queue_resp = VideoQueueResponse(
        model="some-vps-model",
        queue_id="q-vps",
        download_url="https://cdn.example.com/queued.mp4",
    )
    job = VideoJob(mock_client, queue_resp)
    # Status carries neither inline data nor a url.
    completed = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)

    async def _passthrough_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("venice_ai.resources.video.asyncio.to_thread", _passthrough_to_thread)
    target = tmp_path / "out.mp4"
    result = await job.download(target, completed)
    assert result == target
    assert target.read_bytes() == b"VIDEO_BYTES"
    mock_client.fetch_external.assert_awaited_once_with("https://cdn.example.com/queued.mp4")


@pytest.mark.asyncio
async def test_download_url_uses_client_fetch_external(job, mock_client, tmp_path, monkeypatch):
    completed = VideoCompletedStatus(
        status="COMPLETED", url="https://cdn.example.com/v.mp4", expires_at=None
    )

    async def _passthrough_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("venice_ai.resources.video.asyncio.to_thread", _passthrough_to_thread)
    target = tmp_path / "out.mp4"
    result = await job.download(target, completed)
    assert result == target
    assert target.read_bytes() == b"VIDEO_BYTES"
    mock_client.fetch_external.assert_awaited_once_with("https://cdn.example.com/v.mp4")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aexit_calls_complete_on_normal_exit(mock_client, queue_response):
    async with VideoJob(mock_client, queue_response) as job:
        assert job is not None
    mock_client.video.cancel.assert_awaited_once_with(
        model=queue_response.model,
        queue_id=queue_response.queue_id,
    )


@pytest.mark.asyncio
async def test_aexit_propagates_user_exception_when_cleanup_fails(
    mock_client,
    queue_response,
    caplog,
):
    mock_client.video.cancel.side_effect = RuntimeError("cleanup-failed")
    with pytest.raises(ValueError, match="user-error"):
        async with VideoJob(mock_client, queue_response):
            raise ValueError("user-error")
    # Cleanup failure was logged, not raised.
    assert any("cleanup failed" in rec.getMessage().lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_aexit_logs_when_cleanup_fails_on_normal_exit(
    mock_client,
    queue_response,
    caplog,
):
    mock_client.video.cancel.side_effect = RuntimeError("oh no")
    # No user exception → cleanup failure should log a warning, not raise.
    async with VideoJob(mock_client, queue_response):
        pass
    assert any("cleanup failed" in rec.getMessage().lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_aexit_benign_invalid_request_id_not_warned(
    mock_client,
    queue_response,
    caplog,
):
    """A 400 'Request ID is invalid' on cleanup after the job finished is
    benign (the queue entry is already gone), so a normal queue→complete exit
    should not emit a WARNING. It is still noted at DEBUG for diagnosability."""

    class _Resp:
        status = 400
        headers: dict[str, str] = {}

    mock_client.video.cancel.side_effect = InvalidRequestError(
        "Request ID is invalid", response=_Resp()
    )
    with caplog.at_level(logging.DEBUG, logger="venice_ai.resources.video"):
        async with VideoJob(mock_client, queue_response) as job:
            job._status = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)

    cleanup_recs = [r for r in caplog.records if "cleanup" in r.getMessage().lower()]
    assert cleanup_recs, "cleanup outcome should still be logged"
    assert all(r.levelno < logging.WARNING for r in cleanup_recs), (
        "a benign 400 'Request ID is invalid' cleanup must not log at WARNING"
    )


# ---------------------------------------------------------------------------
# Video.generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_video_job(mock_client):
    """Video.generate() should queue and wrap the response in a VideoJob."""
    queue_resp = VideoQueueResponse(model="wan-2.6-text-to-video", queue_id="q-2")
    mock_client.post = AsyncMock(return_value=queue_resp)
    video = Video(mock_client)
    job = await video.run(
        model="wan-2.6-text-to-video",
        prompt="hi",
        duration_seconds="5s",
    )
    assert isinstance(job, VideoJob)
    assert job.model == "wan-2.6-text-to-video"
    assert job.queue_id == "q-2"
