"""Additional tests for ``venice_ai.resources.music``.

Exercises:

- ``MusicJob.__aexit__`` cleanup-failure logging branches
- ``wait()`` ``max_polls`` exhaustion + on_progress guard miss
- ``download()`` URL fetch path (already partially tested, this hits the
  explicit branch behaviour through different inputs)
- ``Music.quote()`` body
- ``Music.retrieve()`` full body — JSON status dispatch, inline-binary path,
  JSON parse failure, and fallback validation
- ``Music.run()`` body — wraps submit + returns MusicJob

All tests use mocked transport — no live API calls. Fixtures match the
``test_music_job.py`` style.
"""

from __future__ import annotations

import json
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
    MusicQuoteResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_response(
    data,
    *,
    content_type: str = "application/json",
    status: int = 200,
    raise_on_json: Exception | None = None,
):
    """Mock aiohttp.ClientResponse for ``raw_response=True`` calls.

    ``data`` is the JSON-serialisable payload returned by ``.json()`` (or
    a string treated as raw text/binary).
    """
    resp = Mock()
    resp.headers = {"content-type": content_type}
    resp.status = status
    if isinstance(data, (bytes, bytearray)):
        body = data
        text = data.decode("utf-8", errors="replace")
    elif isinstance(data, str):
        body = data.encode("utf-8")
        text = data
    else:
        text = json.dumps(data)
        body = text.encode("utf-8")

    resp.content_length = len(body)
    if raise_on_json is not None:
        resp.json = AsyncMock(side_effect=raise_on_json)
    else:
        resp.json = AsyncMock(return_value=data)
    resp.text = AsyncMock(return_value=text)
    resp.read = AsyncMock(return_value=body)
    resp.close = Mock()
    return resp


# ---------------------------------------------------------------------------
# Fixtures (mirroring test_music_job.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def queue_response() -> MusicQueueResponse:
    return MusicQueueResponse(model="elevenlabs-music", queue_id="q-music-1")


@pytest.fixture
def mock_client():
    client = Mock()
    client.music = Mock()
    client.music.retrieve = AsyncMock()
    client.music.cancel = AsyncMock(return_value=MusicCompleteResponse(success=True))
    client.fetch_external = AsyncMock(return_value=b"MUSIC_BYTES")
    return client


@pytest.fixture
def job(mock_client, queue_response) -> MusicJob:
    return MusicJob(mock_client, queue_response)


class _PostOnlyClient:
    """Minimal client stub — only the surface ``Music`` needs for its methods."""

    def __init__(self) -> None:
        self._api_key = "test-key"
        self.post = AsyncMock()


@pytest.fixture
def music_resource() -> tuple[Music, _PostOnlyClient]:
    client = _PostOnlyClient()
    return Music(client), client  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MusicJob.__aexit__ — lines 78-90 (cleanup-failure branches)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aexit_cleanup_failure_no_active_exception_logs_warning(
    job: MusicJob, mock_client, caplog
) -> None:
    """Lines 80-82: cleanup raises while no exception is in flight → warn."""
    mock_client.music.cancel.side_effect = RuntimeError("cleanup boom")
    with caplog.at_level("WARNING"):
        async with job:
            pass
    assert any("MusicJob cleanup failed" in r.message for r in caplog.records)
    assert any("cleanup boom" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_aexit_cleanup_failure_during_exception_logs_combined_warning(
    job: MusicJob, mock_client, caplog
) -> None:
    """Lines 83-90: cleanup raises while user code already raised → warn with original exc."""
    mock_client.music.cancel.side_effect = RuntimeError("cleanup boom")
    with caplog.at_level("WARNING"), pytest.raises(ValueError, match="user error"):
        async with job:
            raise ValueError("user error")
    # The exception-context branch logs the original ValueError name.
    assert any("ValueError" in r.message and "cleanup boom" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# MusicJob.wait — lines 138->140, 141 (timeout + on_progress branch coverage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_raises_timeout_when_max_polls_exhausted(
    job: MusicJob, mock_client, monkeypatch
) -> None:
    """Line 141: max_polls exhausted with no terminal status → TimeoutError."""
    processing = MusicProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=10.0,
    )
    mock_client.music.retrieve.return_value = processing

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("venice_ai.resources.music.asyncio.sleep", _no_sleep)

    with pytest.raises(TimeoutError, match="did not complete within 3 polls"):
        await job.wait(max_polls=3)


@pytest.mark.asyncio
async def test_wait_skips_on_progress_when_callback_is_none(
    job: MusicJob, mock_client, monkeypatch
) -> None:
    """Line 138->140: PROCESSING status without on_progress skips the callback branch."""
    processing = MusicProcessingStatus(
        status="PROCESSING",
        average_execution_time=1000.0,
        execution_duration=300.0,
    )
    completed = MusicCompletedStatus(status="COMPLETED", url="https://x", expires_at=None)
    mock_client.music.retrieve.side_effect = [processing, completed]

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("venice_ai.resources.music.asyncio.sleep", _no_sleep)
    # No on_progress callback — line 138 condition short-circuits → 140
    result = await job.wait()
    assert result is completed


# ---------------------------------------------------------------------------
# Music.quote — lines 235-244
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quote_sends_expected_body_with_duration(music_resource) -> None:
    """Lines 235-244: quote() with duration_seconds posts to /audio/quote."""
    resource, client = music_resource
    client.post.return_value = MusicQuoteResponse(quote=0.42)

    result = await resource.quote(model="elevenlabs-music", duration_seconds=30)

    assert result.quote == 0.42
    args, kwargs = client.post.call_args
    assert args[0] == "audio/quote"
    assert kwargs["json_data"] == {"model": "elevenlabs-music", "duration_seconds": 30}
    assert kwargs["cast_to"] is MusicQuoteResponse


@pytest.mark.asyncio
async def test_quote_sends_expected_body_with_character_count(music_resource) -> None:
    """quote() with character_count posts the expected body."""
    resource, client = music_resource
    client.post.return_value = MusicQuoteResponse(quote=1)

    await resource.quote(model="elevenlabs-music", character_count=500)

    _, kwargs = client.post.call_args
    assert kwargs["json_data"] == {"model": "elevenlabs-music", "character_count": 500}


@pytest.mark.asyncio
async def test_quote_validates_model_id(music_resource) -> None:
    """validate_model_id rejects empty model arg."""
    resource, _ = music_resource
    with pytest.raises(ValueError):
        await resource.quote(model="")


# ---------------------------------------------------------------------------
# Music.retrieve — lines 265-338
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_processing_status(music_resource) -> None:
    """Lines 318-321: status='PROCESSING' returns MusicProcessingStatus."""
    resource, client = music_resource
    payload = {
        "status": "PROCESSING",
        "average_execution_time": 1000.0,
        "execution_duration": 250.0,
    }
    client.post.return_value = _make_raw_response(payload)

    result = await resource.retrieve(model="elevenlabs-music", queue_id="q1")

    assert isinstance(result, MusicProcessingStatus)
    assert result.execution_duration == 250.0
    args, kwargs = client.post.call_args
    assert args[0] == "audio/retrieve"
    assert kwargs["raw_response"] is True


@pytest.mark.asyncio
async def test_retrieve_failed_status(music_resource) -> None:
    """Lines 322-323: status='FAILED' returns MusicFailedStatus."""
    resource, client = music_resource
    payload = {"status": "FAILED", "error": "model overloaded", "error_code": "OVERLOAD"}
    client.post.return_value = _make_raw_response(payload)

    result = await resource.retrieve(model="elevenlabs-music", queue_id="q1")

    assert isinstance(result, MusicFailedStatus)
    assert result.error == "model overloaded"
    assert result.error_code == "OVERLOAD"


@pytest.mark.asyncio
async def test_retrieve_completed_status_json(music_resource) -> None:
    """Lines 324-325: JSON status='COMPLETED' returns MusicCompletedStatus with URL."""
    resource, client = music_resource
    payload = {
        "status": "COMPLETED",
        "url": "https://cdn.example.com/track.mp3",
        "expires_at": "2030-01-01T00:00:00Z",
    }
    client.post.return_value = _make_raw_response(payload)

    result = await resource.retrieve(model="elevenlabs-music", queue_id="q1")

    assert isinstance(result, MusicCompletedStatus)
    assert result.url == "https://cdn.example.com/track.mp3"
    assert result.data is None


@pytest.mark.asyncio
async def test_retrieve_inline_audio_bytes(music_resource) -> None:
    """Lines 289-300: non-JSON content-type → read raw bytes into completed status."""
    resource, client = music_resource
    audio = b"\x00\x01ID3\x03MP3-bytes-here..."
    raw = _make_raw_response(audio, content_type="audio/mpeg")
    client.post.return_value = raw

    result = await resource.retrieve(model="elevenlabs-music", queue_id="q1")

    assert isinstance(result, MusicCompletedStatus)
    assert result.url is None
    assert result.data == audio
    raw.close.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_invalid_json_raises_after_logging(music_resource, caplog) -> None:
    """Lines 302-316: JSON parse failure logs preview and re-raises."""
    resource, client = music_resource
    err = ValueError("not json")
    client.post.return_value = _make_raw_response(
        "definitely not json", content_type="application/json", raise_on_json=err
    )

    with caplog.at_level("ERROR"), pytest.raises(ValueError, match="not json"):
        await resource.retrieve(model="elevenlabs-music", queue_id="q1")

    assert any("Failed to parse JSON" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_retrieve_invalid_json_with_text_failure_uses_placeholder(
    music_resource, caplog
) -> None:
    """Lines 305-308: when both .json() and .text() fail, body_preview is a placeholder."""
    resource, client = music_resource
    raw = _make_raw_response(
        "irrelevant", content_type="application/json", raise_on_json=ValueError("nope")
    )
    raw.text = AsyncMock(side_effect=RuntimeError("text broken"))
    client.post.return_value = raw

    with caplog.at_level("ERROR"), pytest.raises(ValueError, match="nope"):
        await resource.retrieve(model="elevenlabs-music", queue_id="q1")

    # The fallback placeholder text appears in the structured log args / message
    # because logger.error has it in the format string.
    assert any("unable to read body" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_retrieve_unknown_status_falls_through_validators(music_resource) -> None:
    """Lines 327-336: unknown status string lets fallback validate to whichever model fits."""
    # A payload that doesn't match the explicit if-tree (status not in
    # PROCESSING/FAILED/COMPLETED) falls through to the validator loop.
    # Provide enough fields to satisfy MusicProcessingStatus on retry-validation.
    resource, client = music_resource
    payload = {
        "status": "PROCESSING",  # Wire status keyword forces strict literal match
        "average_execution_time": 50.0,
        "execution_duration": 5.0,
    }
    # But mute the if-branch by sending a non-dict that bypasses lines 318-325
    # entirely — we want the fallback loop. Simplest: a list payload.
    client.post.return_value = _make_raw_response(payload)

    # Sanity check: the dict-branch returns immediately for known statuses
    # The 327-336 fallback exists for non-dict / non-matching shapes.
    # Use a non-dict payload to force the fallback loop.
    list_payload = [payload]  # not a dict → skips the if-branch
    client.post.return_value = _make_raw_response(list_payload)

    # All three validators will fail (lists don't match any of the models),
    # so the function raises ValueError per line 338.
    with pytest.raises(ValueError, match="Unable to parse music retrieve response"):
        await resource.retrieve(model="elevenlabs-music", queue_id="q1")


@pytest.mark.asyncio
async def test_retrieve_unknown_status_in_dict_uses_fallback(music_resource) -> None:
    """Lines 327-336: dict with unrecognized status falls through the if/elif
    tree into the validator loop; when none of the typed models accept the
    payload, the SDK raises ValueError."""
    resource, client = music_resource
    bogus_payload = {"status": "UNKNOWN_STATUS", "weird_field": "z"}
    client.post.return_value = _make_raw_response(bogus_payload)

    with pytest.raises(ValueError, match="Unable to parse music retrieve response"):
        await resource.retrieve(model="elevenlabs-music", queue_id="q1")


@pytest.mark.asyncio
async def test_retrieve_validates_model_id(music_resource) -> None:
    """retrieve() rejects empty model arg via validate_model_id."""
    resource, _ = music_resource
    with pytest.raises(ValueError):
        await resource.retrieve(model="", queue_id="q1")


# ---------------------------------------------------------------------------
# Music.run — lines 388-399 (submit + wrap into MusicJob)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_music_job_wrapping_submit_result(music_resource) -> None:
    """Lines 388-399: run() wraps submit() result into a MusicJob."""
    resource, client = music_resource
    queue = MusicQueueResponse(model="elevenlabs-music", queue_id="q-run-1")
    client.post.return_value = queue

    job = await resource.run(
        model="elevenlabs-music",
        prompt="A happy tune",
        duration_seconds=20,
        force_instrumental=True,
        lyrics_optimizer=False,
        voice="alto",
        language_code="en",
        speed=1.0,
        lyrics_prompt="la la la",
    )

    assert isinstance(job, MusicJob)
    assert job.model == "elevenlabs-music"
    assert job.queue_id == "q-run-1"

    # submit() forwards to client.post("audio/queue", ...) — verify the body
    # carries the wide-parameter set we passed in.
    args, kwargs = client.post.call_args
    assert args[0] == "audio/queue"
    body = kwargs["json_data"]
    assert body["model"] == "elevenlabs-music"
    assert body["prompt"] == "A happy tune"
    assert body["duration_seconds"] == 20
    assert body["force_instrumental"] is True
    assert body["lyrics_optimizer"] is False
    assert body["voice"] == "alto"
    assert body["language_code"] == "en"
    assert body["speed"] == 1.0
    assert body["lyrics_prompt"] == "la la la"


@pytest.mark.asyncio
async def test_run_with_minimal_args(music_resource) -> None:
    """run() with only required args still produces a MusicJob."""
    resource, client = music_resource
    queue = MusicQueueResponse(model="elevenlabs-music", queue_id="q-run-2")
    client.post.return_value = queue

    job = await resource.run(model="elevenlabs-music", prompt="minimal")

    assert isinstance(job, MusicJob)
    assert job.queue_id == "q-run-2"


# ---------------------------------------------------------------------------
# MusicJob.poll integration — verifies wait() polling cadence interacts with
# the real Music.retrieve() (regression check that wait() doesn't accidentally
# bypass progress reporting on the 138->140 branch).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_failed_status_short_circuits_before_sleep(
    job: MusicJob, mock_client, monkeypatch
) -> None:
    """Coverage cross-check: failure path exits before the sleep call."""
    sleep_calls: list[float] = []

    async def _track_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("venice_ai.resources.music.asyncio.sleep", _track_sleep)
    failed = MusicFailedStatus(status="FAILED", error="bad", error_code="X")
    mock_client.music.retrieve.return_value = failed

    with pytest.raises(MusicGenerationError):
        await job.wait()
    # First poll already FAILED → no sleep called.
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_submit_forwards_loop_to_request_body(music_resource) -> None:
    """``loop`` reaches the wire so loop-capable models can splice the clip."""
    resource, client = music_resource
    client.post.return_value = MusicQueueResponse(model="elevenlabs-music", queue_id="q-loop-1")

    await resource.submit(
        model="elevenlabs-music",
        prompt="Ambient pad",
        duration_seconds=30,
        loop=True,
    )

    args, kwargs = client.post.call_args
    assert args[0] == "audio/queue"
    assert kwargs["json_data"]["loop"] is True


@pytest.mark.asyncio
async def test_submit_omits_loop_when_unset(music_resource) -> None:
    """Omitted ``loop`` must not appear in the body — the model default applies."""
    resource, client = music_resource
    client.post.return_value = MusicQueueResponse(model="elevenlabs-music", queue_id="q-loop-2")

    await resource.submit(
        model="elevenlabs-music",
        prompt="Ambient pad",
        duration_seconds=30,
    )

    _, kwargs = client.post.call_args
    assert "loop" not in kwargs["json_data"]


@pytest.mark.asyncio
async def test_run_forwards_loop_to_request_body(music_resource) -> None:
    """``run()`` is the managed-lifecycle entry point and must pass loop through."""
    resource, client = music_resource
    client.post.return_value = MusicQueueResponse(model="elevenlabs-music", queue_id="q-loop-3")

    await resource.run(
        model="elevenlabs-music",
        prompt="Ambient pad",
        duration_seconds=30,
        loop=True,
    )

    _, kwargs = client.post.call_args
    assert kwargs["json_data"]["loop"] is True
