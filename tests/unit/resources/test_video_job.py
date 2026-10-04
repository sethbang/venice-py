"""Unit tests for VideoJob lifecycle management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    InvalidRequestError,
    PermissionDeniedError,
    ServiceUnavailableError,
    VideoGenerationError,
)
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


_COMPLETED = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)


@pytest.mark.asyncio
async def test_aexit_calls_complete_on_normal_exit(mock_client, queue_response):
    async with VideoJob(mock_client, queue_response) as job:
        job._status = _COMPLETED
    mock_client.video.cancel.assert_awaited_once_with(
        model=queue_response.model,
        queue_id=queue_response.queue_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["never_polled", "processing"])
async def test_aexit_before_terminal_status_sends_nothing_and_warns(
    mock_client, queue_response, caplog, state
):
    """Releasing a job that is still generating deletes nothing, and a final
    poll of an inline-video model would stream the whole MP4: a clean exit
    before a terminal status sends no request and says how to finish up."""
    with caplog.at_level(logging.WARNING, logger="venice_ai.resources.video"):
        async with VideoJob(mock_client, queue_response) as job:
            if state == "processing":
                job._status = VideoProcessingStatus(
                    status="PROCESSING", average_execution_time=1000, execution_duration=100
                )
    mock_client.video.cancel.assert_not_awaited()
    mock_client.video.retrieve.assert_not_awaited()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert queue_response.queue_id in warnings[0]
    assert queue_response.model in warnings[0]
    assert "keeps running" in warnings[0]
    assert "wait()" in warnings[0] and "cancel()" in warnings[0]


@pytest.mark.asyncio
async def test_aexit_on_user_exception_keeps_the_media_and_propagates(
    mock_client,
    queue_response,
    caplog,
):
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="user-error"):
        async with VideoJob(mock_client, queue_response) as job:
            job._status = VideoCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
            raise ValueError("user-error")
    # A completed video is not deleted when saving it failed: it can be retried.
    mock_client.video.cancel.assert_not_awaited()
    message = " ".join(r.getMessage() for r in caplog.records)
    assert queue_response.queue_id in message
    assert "not released" in message
    assert "client.video.retrieve(" in message


@pytest.mark.asyncio
async def test_aexit_on_user_exception_still_releases_a_failed_job(mock_client, queue_response):
    with pytest.raises(ValueError, match="user-error"):
        async with VideoJob(mock_client, queue_response) as job:
            job._status = VideoFailedStatus(status="FAILED", error="boom")
            raise ValueError("user-error")
    mock_client.video.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_aexit_logs_when_cleanup_fails_on_normal_exit(
    mock_client,
    queue_response,
    caplog,
):
    mock_client.video.cancel.side_effect = RuntimeError("oh no")
    # No user exception → cleanup failure should log a warning, not raise.
    async with VideoJob(mock_client, queue_response) as job:
        job._status = _COMPLETED
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
# cancel(): jobs with a queue-time download_url
# ---------------------------------------------------------------------------

# The download_url grants access on its own; it must never reach a log line
# or an exception message.
_LINK_SECRET = "SECRET-TOKEN-7f3a9c"


class _Resp400:
    status = 400
    headers: dict[str, str] = {}


def _unknown_request_id() -> InvalidRequestError:
    return InvalidRequestError("Request ID is invalid", response=_Resp400())


@asynccontextmanager
async def _link_server(status: int) -> AsyncIterator[tuple[str, list[web.Request]]]:
    """Serve the download_url host, answering every request with *status*."""
    seen: list[web.Request] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(request)
        if status == 204:
            return web.Response(status=204)
        return web.json_response({"error": "object not found"}, status=status)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    # The server's access log records the path; only the SDK's logs are checked.
    runner = web.AppRunner(app, shutdown_timeout=0.1, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", seen
    finally:
        await runner.cleanup()


def _link_job(client: VeniceClient, base: str) -> VideoJob:
    return VideoJob(
        client,
        VideoQueueResponse(
            model="grok-imagine-1-5-lite-text-to-video",
            queue_id="q-link",
            download_url=f"{base}/videos/{_LINK_SECRET}.mp4",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("link_status", [204, 404])
@pytest.mark.parametrize("api_origin", ["same", "other"])
async def test_cancel_deletes_download_url_without_credentials(
    link_status, api_origin, monkeypatch, caplog
):
    """The link is deleted with a bare DELETE (no API key, even on the API's
    own origin); 204 and 404 both mean the file is gone, and the 400 that
    /video/complete answers for such jobs is expected."""
    async with _link_server(link_status) as (base, seen):
        base_url = f"{base}/api/v1" if api_origin == "same" else None
        async with VeniceClient(api_key="sk-test-secret", base_url=base_url) as client:
            complete = AsyncMock(side_effect=_unknown_request_id())
            monkeypatch.setattr(client.video, "cancel", complete)
            job = _link_job(client, base)
            job._status = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)
            with caplog.at_level(logging.DEBUG):
                result = await job.cancel()

    assert result.success is True
    assert job._released is True
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "DELETE"
    assert request.path == f"/videos/{_LINK_SECRET}.mp4"
    assert "Authorization" not in request.headers
    assert "X-Sign-In-With-X" not in request.headers
    assert "sk-test-secret" not in str(dict(request.headers))
    complete.assert_awaited_once_with(model=job.model, queue_id=job.queue_id)
    assert _LINK_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_cancel_returns_complete_response_when_complete_accepts(monkeypatch):
    async with _link_server(204) as (base, seen), VeniceClient(api_key="sk-test") as client:
        complete = AsyncMock(return_value=VideoCompleteResponse(success=True))
        monkeypatch.setattr(client.video, "cancel", complete)
        result = await _link_job(client, base).cancel()
    assert result is complete.return_value
    assert [r.method for r in seen] == ["DELETE"]


@pytest.mark.asyncio
async def test_cancel_link_error_status_raises_without_the_url(monkeypatch, caplog):
    async with _link_server(403) as (base, _seen), VeniceClient(api_key="sk-test") as client:
        complete = AsyncMock(return_value=VideoCompleteResponse(success=True))
        monkeypatch.setattr(client.video, "cancel", complete)
        job = _link_job(client, base)
        with caplog.at_level(logging.DEBUG), pytest.raises(PermissionDeniedError) as info:
            await job.cancel()
    assert info.value.status_code == 403
    assert _LINK_SECRET not in str(info.value)
    assert _LINK_SECRET not in caplog.text
    complete.assert_not_awaited()
    assert job._released is False


@pytest.mark.asyncio
async def test_cancel_link_delete_is_sent_once_and_never_logged_on_5xx(monkeypatch, caplog):
    """A retry would log the request URL, so the link DELETE is not retried."""
    async with _link_server(503) as (base, seen), VeniceClient(api_key="sk-test") as client:
        monkeypatch.setattr(client.video, "cancel", AsyncMock())
        job = _link_job(client, base)
        with caplog.at_level(logging.DEBUG), pytest.raises(ServiceUnavailableError) as info:
            await job.cancel()
    assert len(seen) == 1
    assert _LINK_SECRET not in str(info.value)
    assert _LINK_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_aexit_warns_when_the_link_delete_is_rejected(monkeypatch, caplog):
    """A 400 from the storage host is not the benign "Request ID is invalid":
    the file is still stored, so the exit says so at WARNING."""
    async with _link_server(400) as (base, _seen), VeniceClient(api_key="sk-test") as client:
        monkeypatch.setattr(client.video, "cancel", AsyncMock())
        with caplog.at_level(logging.DEBUG):
            async with _link_job(client, base) as job:
                job._status = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "cleanup failed" in warnings[0].getMessage()
    assert _LINK_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_cancel_after_link_delete_propagates_other_complete_errors(monkeypatch):
    async with _link_server(204) as (base, _seen), VeniceClient(api_key="sk-test") as client:
        other = InvalidRequestError("model: unknown model", response=_Resp400())
        monkeypatch.setattr(client.video, "cancel", AsyncMock(side_effect=other))
        with pytest.raises(InvalidRequestError, match="unknown model"):
            await _link_job(client, base).cancel()


@pytest.mark.asyncio
async def test_cancel_without_download_url_only_calls_complete(mock_client, queue_response):
    mock_client.video.cancel.side_effect = _unknown_request_id()
    job = VideoJob(mock_client, queue_response)
    with pytest.raises(InvalidRequestError, match="Request ID is invalid"):
        await job.cancel()
    mock_client.video.cancel.assert_awaited_once_with(
        model=queue_response.model, queue_id=queue_response.queue_id
    )
    assert job._released is False


@pytest.mark.asyncio
async def test_aexit_after_completion_deletes_the_link_and_never_logs_it(monkeypatch, caplog):
    async with _link_server(204) as (base, seen), VeniceClient(api_key="sk-test") as client:
        complete = AsyncMock(side_effect=_unknown_request_id())
        monkeypatch.setattr(client.video, "cancel", complete)
        with caplog.at_level(logging.DEBUG):
            async with _link_job(client, base) as job:
                job._status = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)
    assert [r.method for r in seen] == ["DELETE"]
    complete.assert_awaited_once()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert _LINK_SECRET not in caplog.text


@pytest.mark.asyncio
async def test_aexit_on_exception_with_link_warns_without_the_url(monkeypatch, caplog):
    async with _link_server(204) as (base, seen), VeniceClient(api_key="sk-test") as client:
        complete = AsyncMock(return_value=VideoCompleteResponse(success=True))
        monkeypatch.setattr(client.video, "cancel", complete)
        with caplog.at_level(logging.DEBUG), pytest.raises(OSError):
            async with _link_job(client, base) as job:
                job._status = VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)
                raise OSError("disk full")
    assert seen == []
    complete.assert_not_awaited()
    assert "q-link" in caplog.text
    assert "not released" in caplog.text
    assert _LINK_SECRET not in caplog.text


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
