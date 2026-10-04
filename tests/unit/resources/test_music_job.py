"""Unit tests for the MusicJob lifecycle + Music.submit / release methods."""

from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.exceptions import MusicGenerationError
from venice_ai.resources.music import Music, MusicJob
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQueueResponse,
)


@pytest.fixture
def queue_response() -> MusicQueueResponse:
    return MusicQueueResponse(model="elevenlabs-music", queue_id="q-music-1")


@pytest.fixture
def mock_client():
    client = Mock()
    client.music = Mock()
    client.music.retrieve = AsyncMock()
    client.music.release = AsyncMock(return_value=MusicCompleteResponse(success=True))
    client.fetch_external = AsyncMock(return_value=b"MUSIC_BYTES")
    return client


@pytest.fixture
def job(mock_client, queue_response) -> MusicJob:
    return MusicJob(mock_client, queue_response)


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_init_sets_fields(job: MusicJob, queue_response: MusicQueueResponse) -> None:
    assert job.model == queue_response.model
    assert job.queue_id == queue_response.queue_id
    assert job.status is None
    assert job.is_complete is False
    assert job.is_failed is False
    assert job.progress is None


def test_progress_with_processing_status(job: MusicJob) -> None:
    job._status = MusicProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=250.0,
    )
    assert job.progress == pytest.approx(0.25)


def test_is_complete_after_completed(job: MusicJob) -> None:
    job._status = MusicCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    assert job.is_complete is True
    assert job.is_failed is False


def test_is_failed_after_failed(job: MusicJob) -> None:
    job._status = MusicFailedStatus(status="FAILED", error="boom", error_code="E1")
    assert job.is_failed is True
    assert job.is_complete is False


# ---------------------------------------------------------------------------
# poll() / wait()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_updates_status(job: MusicJob, mock_client) -> None:
    expected = MusicCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.music.retrieve.return_value = expected
    result = await job.poll()
    assert result is expected
    assert job.status is expected
    mock_client.music.retrieve.assert_awaited_once_with(model=job.model, queue_id=job.queue_id)


@pytest.mark.asyncio
async def test_wait_returns_completed_immediately(job: MusicJob, mock_client, monkeypatch) -> None:
    completed = MusicCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.music.retrieve.return_value = completed

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("venice_ai.resources.music.asyncio.sleep", _fake_sleep)
    result = await job.wait(poll_interval=0.5)
    assert result is completed
    # First poll was already COMPLETED, so no sleep.
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_wait_invokes_progress_then_completes(
    job: MusicJob, mock_client, monkeypatch
) -> None:
    processing = MusicProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=300.0,
    )
    completed = MusicCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.music.retrieve.side_effect = [processing, completed]

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("venice_ai.resources.music.asyncio.sleep", _no_sleep)
    seen: list[MusicProcessingStatus] = []
    result = await job.wait(on_progress=seen.append)
    assert result is completed
    assert seen == [processing]


@pytest.mark.asyncio
async def test_wait_raises_on_failure(job: MusicJob, mock_client) -> None:
    failed = MusicFailedStatus(status="FAILED", error="nope", error_code="X1")
    mock_client.music.retrieve.return_value = failed
    with pytest.raises(MusicGenerationError) as exc:
        await job.wait()
    assert exc.value.error_code == "X1"


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_writes_inline_bytes(job: MusicJob, tmp_path) -> None:
    completed = MusicCompletedStatus(status="COMPLETED", url=None, expires_at=None)
    completed._set_data(b"INLINE_MUSIC")
    target = tmp_path / "out.mp3"
    result = await job.download(target, completed)
    assert result == target
    assert target.read_bytes() == b"INLINE_MUSIC"


@pytest.mark.asyncio
async def test_download_fetches_url_when_no_inline_bytes(
    job: MusicJob, mock_client, tmp_path
) -> None:
    completed = MusicCompletedStatus(
        status="COMPLETED", url="https://example.com/a.mp3", expires_at=None
    )
    target = tmp_path / "out.mp3"
    await job.download(target, completed)
    mock_client.fetch_external.assert_awaited_once_with("https://example.com/a.mp3")
    assert target.read_bytes() == b"MUSIC_BYTES"


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_context_manager_releases_a_finished_job(job: MusicJob, mock_client) -> None:
    mock_client.music.retrieve.return_value = MusicCompletedStatus(status="COMPLETED")
    async with job:
        pass
    mock_client.music.release.assert_awaited_once_with(model=job.model, queue_id=job.queue_id)


# ---------------------------------------------------------------------------
# Music.{submit,quote,retrieve,release} — direct method tests
# ---------------------------------------------------------------------------


class _PostOnlyClient:
    """Minimal client stub — only the surface ``Music`` needs for its methods."""

    def __init__(self) -> None:
        self._api_key = "test-key"
        self.post = AsyncMock()


@pytest.fixture
def music_resource() -> tuple[Music, _PostOnlyClient]:
    client = _PostOnlyClient()
    return Music(client), client  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_submit_sends_expected_body(music_resource) -> None:
    resource, client = music_resource
    client.post.return_value = MusicQueueResponse(
        model="elevenlabs-music", queue_id="q1", status="QUEUED"
    )

    await resource.submit(
        model="elevenlabs-music",
        prompt="Gentle piano intro",
        duration_seconds=30,
        force_instrumental=True,
    )

    client.post.assert_awaited_once()
    args, kwargs = client.post.call_args
    assert args[0] == "audio/queue"
    body = kwargs["json_data"]
    assert body == {
        "model": "elevenlabs-music",
        "prompt": "Gentle piano intro",
        "duration_seconds": 30,
        "force_instrumental": True,
    }


@pytest.mark.asyncio
async def test_release_sends_expected_body(music_resource) -> None:
    resource, client = music_resource
    client.post.return_value = MusicCompleteResponse(success=True)

    result = await resource.release(model="elevenlabs-music", queue_id="q1")

    assert result.success is True
    args, kwargs = client.post.call_args
    assert args[0] == "audio/complete"
    assert kwargs["json_data"] == {"model": "elevenlabs-music", "queue_id": "q1"}
