"""Tests for ``venice_ai.resources.voice_changer``.

**The happy path cannot be live-verified.** The
``/audio/voice-changer/*`` endpoints are deployed — a quote against a music
model returns the server's own contract error — but the live catalog contains
zero models reporting ``voice_changer: true``, and the spec's example id
``elevenlabs-voice-changer`` 404s. So there is no cassette and no end-to-end
run behind these tests; they are written against the spec.

That makes the shape assertions matter more than usual, because a mock written
against a wrong assumption passes just as happily as a correct one. The three
places this family diverges from music/video — and where reasoning by analogy
produces a plausible, wrong implementation — are each covered directly:

1. ``duration_seconds`` is a bare count of seconds, **not** video's ``"60s"``.
2. ``retrieve`` has **two** arms, not three. There is no ``FAILED`` status and
   no JSON completed arm with a download URL.
3. ``submit`` dispatches on file-vs-URL: multipart for one, JSON for the other.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from venice_ai.resources.voice_changer import VoiceChanger, VoiceChangerJob
from venice_ai.types.api.requests.voice_changer import (
    QueueVoiceChangerRequest,
    QuoteVoiceChangerRequest,
    RetrieveVoiceChangerRequest,
)
from venice_ai.types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerCompleteResponse,
    VoiceChangerProcessingStatus,
    VoiceChangerQueueResponse,
    VoiceChangerQuoteResponse,
)

MODEL = "some-voice-changer"
AUDIO = b"ID3\x04fake-mp3-bytes"


def _make_raw_response(data, *, content_type="application/json", status=200):
    """Mock aiohttp.ClientResponse for ``raw_response=True`` calls."""
    resp = Mock()
    resp.headers = {"content-type": content_type}
    resp.status = status
    if isinstance(data, (bytes, bytearray)):
        body, text = bytes(data), bytes(data).decode("utf-8", errors="replace")
    else:
        text = json.dumps(data)
        body = text.encode()
    resp.content_length = len(body)
    resp.json = AsyncMock(return_value=data)
    resp.text = AsyncMock(return_value=text)
    resp.read = AsyncMock(return_value=body)
    resp.close = Mock()
    return resp


@pytest.fixture
def queue_response() -> VoiceChangerQueueResponse:
    return VoiceChangerQueueResponse(
        model=MODEL, queue_id="q-vc-1", status="QUEUED", duration_seconds=52.0
    )


@pytest.fixture
def client():
    c = Mock()
    c.post = AsyncMock()
    c.voice_changer = Mock()
    return c


# ---------------------------------------------------------------------------
# 1. duration_seconds is a bare number, not video's "60s"
# ---------------------------------------------------------------------------


class TestQuoteDurationIsBareSeconds:
    @pytest.mark.parametrize("value", [60, 1, 3600, "60", "1"])
    def test_accepts_positive_whole_seconds(self, value):
        assert QuoteVoiceChangerRequest(model=MODEL, duration_seconds=value).duration_seconds

    @pytest.mark.parametrize("value", ["60s", "1m", "PT60S", "60.5", "sixty", ""])
    def test_rejects_the_video_style_duration_forms(self, value):
        """``_format_video_duration`` output must not be accepted here."""
        with pytest.raises(ValidationError):
            QuoteVoiceChangerRequest(model=MODEL, duration_seconds=value)

    @pytest.mark.parametrize("value", [0, -1, "0"])
    def test_rejects_non_positive(self, value):
        with pytest.raises(ValidationError):
            QuoteVoiceChangerRequest(model=MODEL, duration_seconds=value)

    @pytest.mark.asyncio
    async def test_quote_body(self, client):
        client.post.return_value = VoiceChangerQuoteResponse(quote=0.35, duration_seconds=60)
        await VoiceChanger(client).quote(model=MODEL, duration_seconds=60)
        kwargs = client.post.call_args.kwargs
        assert kwargs["json_data"] == {"model": MODEL, "duration_seconds": 60}
        assert client.post.call_args.args[0] == "audio/voice-changer/quote"


# ---------------------------------------------------------------------------
# 2. retrieve has exactly two arms
# ---------------------------------------------------------------------------


class TestRetrieveHasTwoArms:
    @pytest.mark.asyncio
    async def test_json_processing_arm(self, client):
        client.post.return_value = _make_raw_response(
            {
                "status": "PROCESSING",
                "average_execution_time": 10000,
                "execution_duration": 4200,
            }
        )
        result = await VoiceChanger(client).retrieve(model=MODEL, queue_id="q-vc-1")
        assert isinstance(result, VoiceChangerProcessingStatus)
        assert result.execution_duration == 4200
        assert client.post.call_args.kwargs["raw_response"] is True

    @pytest.mark.asyncio
    async def test_audio_arm_is_discriminated_by_content_type(self, client):
        """Completion is signalled by the content type, not a status field."""
        client.post.return_value = _make_raw_response(AUDIO, content_type="audio/mpeg")
        result = await VoiceChanger(client).retrieve(model=MODEL, queue_id="q-vc-1")
        assert isinstance(result, VoiceChangerCompletedStatus)
        assert result.data == AUDIO
        assert result.status == "COMPLETED"

    def test_the_union_has_no_failed_arm(self):
        """The spec declares no FAILED status; inventing one models a state
        the server never sends."""
        from venice_ai.types.api import voice_changer as vc_types

        assert not [n for n in vc_types.__all__ if "Failed" in n]

    def test_completed_status_has_no_url_field(self):
        """Unlike video, the audio is inline — there is nothing to download from."""
        assert "url" not in VoiceChangerCompletedStatus.model_fields
        assert "expires_at" not in VoiceChangerCompletedStatus.model_fields

    @pytest.mark.asyncio
    async def test_delete_media_on_completion_is_sent(self, client):
        client.post.return_value = _make_raw_response(AUDIO, content_type="audio/mpeg")
        await VoiceChanger(client).retrieve(
            model=MODEL, queue_id="q-vc-1", delete_media_on_completion=True
        )
        assert client.post.call_args.kwargs["json_data"]["delete_media_on_completion"] is True

    def test_delete_media_defaults_to_false(self):
        req = RetrieveVoiceChangerRequest(model=MODEL, queue_id="q")
        assert req.delete_media_on_completion is False


# ---------------------------------------------------------------------------
# 3. submit dispatches on file vs URL
# ---------------------------------------------------------------------------


class TestSourceDispatch:
    def test_request_rejects_both_sources(self):
        with pytest.raises(ValidationError, match="not both"):
            QueueVoiceChangerRequest(model=MODEL, file=b"x", audio_url="https://e.com/a.mp3")

    def test_request_rejects_neither_source(self):
        with pytest.raises(ValidationError, match="source recording is required"):
            QueueVoiceChangerRequest(model=MODEL)

    def test_request_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            QueueVoiceChangerRequest(model=MODEL, audio_url="https://e.com/a.mp3", nope=1)

    @pytest.mark.asyncio
    async def test_url_source_goes_as_json_without_a_file_key(self, client):
        client.post.return_value = VoiceChangerQueueResponse(
            model=MODEL, queue_id="q", status="QUEUED", duration_seconds=10
        )
        await VoiceChanger(client).submit(model=MODEL, audio_url="https://e.com/a.mp3")
        body = client.post.call_args.kwargs["json_data"]
        assert body["audio_url"] == "https://e.com/a.mp3"
        assert "file" not in body

    @pytest.mark.asyncio
    async def test_file_source_goes_as_multipart(self, client, tmp_path):
        src = tmp_path / "source.mp3"
        src.write_bytes(AUDIO)
        vc = VoiceChanger(client)
        vc._request_multipart = AsyncMock(
            return_value={
                "model": MODEL,
                "queue_id": "q",
                "status": "QUEUED",
                "duration_seconds": 12.5,
            }
        )
        result = await vc.submit(model=MODEL, file=str(src), voice="Aria")

        assert client.post.await_count == 0, "a file source must not go as JSON"
        kwargs = vc._request_multipart.call_args.kwargs
        assert kwargs["path"] == "audio/voice-changer/queue"
        filename, content, content_type = kwargs["files"]["file"]
        assert filename == "source.mp3"
        assert content == AUDIO
        assert content_type == "audio/mpeg"
        # The binary goes in `files`, never duplicated into the form fields.
        assert "file" not in kwargs["data"]
        assert kwargs["data"]["voice"] == "Aria"
        assert result.duration_seconds == 12.5

    @pytest.mark.asyncio
    async def test_submit_rejects_both_before_reading_any_file(self, client):
        with pytest.raises(ValidationError):
            await VoiceChanger(client).submit(
                model=MODEL, file=b"x", audio_url="https://e.com/a.mp3"
            )


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


class TestVoiceChangerJob:
    def test_carries_the_billed_duration(self, client, queue_response):
        job = VoiceChangerJob(client, queue_response)
        assert job.duration_seconds == 52.0
        assert job.queue_id == "q-vc-1"

    @pytest.mark.asyncio
    async def test_wait_returns_on_the_audio_arm(self, client, queue_response):
        processing = VoiceChangerProcessingStatus(
            status="PROCESSING", average_execution_time=1000, execution_duration=100
        )
        completed = VoiceChangerCompletedStatus()
        completed._set_data(AUDIO)
        client.voice_changer.retrieve = AsyncMock(side_effect=[processing, completed])

        seen: list[VoiceChangerProcessingStatus] = []
        job = VoiceChangerJob(client, queue_response)
        result = await job.wait(poll_interval=0, on_progress=seen.append)

        assert result is completed
        assert len(seen) == 1
        assert job.is_complete

    @pytest.mark.asyncio
    async def test_wait_times_out(self, client, queue_response):
        processing = VoiceChangerProcessingStatus(
            status="PROCESSING", average_execution_time=1000, execution_duration=100
        )
        client.voice_changer.retrieve = AsyncMock(return_value=processing)
        job = VoiceChangerJob(client, queue_response)
        with pytest.raises(TimeoutError):
            await job.wait(poll_interval=0, max_polls=3)

    @pytest.mark.asyncio
    async def test_download_writes_the_inline_bytes(self, client, queue_response, tmp_path):
        completed = VoiceChangerCompletedStatus()
        completed._set_data(AUDIO)
        out = tmp_path / "nested" / "converted.mp3"
        saved = await VoiceChangerJob(client, queue_response).download(out, completed)
        assert saved.read_bytes() == AUDIO

    @pytest.mark.asyncio
    async def test_download_without_bytes_is_an_error_not_a_url_fetch(
        self, client, queue_response, tmp_path
    ):
        """There is no URL fallback on this endpoint, so empty means broken."""
        with pytest.raises(ValueError, match="no audio bytes"):
            await VoiceChangerJob(client, queue_response).download(
                tmp_path / "x.mp3", VoiceChangerCompletedStatus()
            )

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up(self, client, queue_response):
        client.voice_changer.cancel = AsyncMock(
            return_value=VoiceChangerCompleteResponse(success=True)
        )
        async with VoiceChangerJob(client, queue_response) as job:
            assert job.queue_id == "q-vc-1"
        client.voice_changer.cancel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_progress_is_none_once_complete(self, client, queue_response):
        completed = VoiceChangerCompletedStatus()
        client.voice_changer.retrieve = AsyncMock(return_value=completed)
        job = VoiceChangerJob(client, queue_response)
        await job.poll()
        assert job.progress is None

    @pytest.mark.asyncio
    async def test_progress_while_processing(self, client, queue_response):
        client.voice_changer.retrieve = AsyncMock(
            return_value=VoiceChangerProcessingStatus(
                status="PROCESSING", average_execution_time=10000, execution_duration=2500
            )
        )
        job = VoiceChangerJob(client, queue_response)
        await job.poll()
        assert job.progress == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_progress_clamps_past_the_average(self, client, queue_response):
        """A slower-than-average run must not report 340%."""
        client.voice_changer.retrieve = AsyncMock(
            return_value=VoiceChangerProcessingStatus(
                status="PROCESSING", average_execution_time=1000, execution_duration=3400
            )
        )
        job = VoiceChangerJob(client, queue_response)
        await job.poll()
        assert job.progress == 1.0


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_body(self, client):
        client.post.return_value = VoiceChangerCompleteResponse(success=True)
        await VoiceChanger(client).cancel(model=MODEL, queue_id="q-vc-1")
        assert client.post.call_args.args[0] == "audio/voice-changer/complete"
        assert client.post.call_args.kwargs["json_data"] == {
            "model": MODEL,
            "queue_id": "q-vc-1",
        }


class TestRun:
    @pytest.mark.asyncio
    async def test_run_wraps_submit_in_a_job(self, client):
        client.post.return_value = VoiceChangerQueueResponse(
            model=MODEL, queue_id="q-run", status="QUEUED", duration_seconds=7
        )
        job = await VoiceChanger(client).run(model=MODEL, audio_url="https://e.com/a.mp3")
        assert isinstance(job, VoiceChangerJob)
        assert job.queue_id == "q-run"
        assert job.duration_seconds == 7


class TestResolveVoiceChanger:
    """Voice changers are ``type="music"`` + ``voice_changer=True``.

    Resolving by type alone hands back a music *generator*, which the
    voice-changer endpoints reject with a 400.
    """

    @staticmethod
    def _catalog(*specs):
        from venice_ai.types.api import ModelsListResponse

        return ModelsListResponse.model_validate(
            {
                "object": "list",
                "type": "music",
                "data": [
                    {
                        "id": mid,
                        "object": "model",
                        "created": 1771804800.0,
                        "owned_by": "venice.ai",
                        "type": "music",
                        "model_spec": spec,
                    }
                    for mid, spec in specs
                ],
            }
        )

    @staticmethod
    def _models(catalog):
        from venice_ai.resources.models import Models

        client = Mock()
        client.get = AsyncMock(return_value=catalog)
        return Models(client)

    @pytest.mark.asyncio
    async def test_picks_only_the_voice_changer(self):
        models = self._models(
            self._catalog(
                ("music-gen", {"name": "Generator"}),
                ("vc-model", {"name": "Changer", "voice_changer": True}),
            )
        )
        assert await models.resolve_voice_changer() == "vc-model"

    @pytest.mark.asyncio
    async def test_raises_clearly_when_none_exist(self):
        """The live catalog has none today, so this is the common path."""
        models = self._models(self._catalog(("music-gen", {"name": "Generator"})))
        with pytest.raises(ValueError, match="No available voice-changer models found"):
            await models.resolve_voice_changer()

    @pytest.mark.asyncio
    async def test_honours_preferred_models(self):
        models = self._models(
            self._catalog(
                ("vc-a", {"name": "A", "voice_changer": True}),
                ("vc-b", {"name": "B", "voice_changer": True}),
            )
        )
        assert await models.resolve_voice_changer(preferred_models=["vc-b"]) == "vc-b"

    @pytest.mark.asyncio
    async def test_honours_exclusions(self):
        models = self._models(
            self._catalog(
                ("vc-a", {"name": "A", "voice_changer": True}),
                ("vc-b", {"name": "B", "voice_changer": True}),
            )
        )
        assert await models.resolve_voice_changer(exclude_models=["vc-a"]) == "vc-b"

    def test_absent_flag_means_no_not_undeclared(self):
        """The API sends these only when true, following ``uncensored``."""
        from venice_ai.types.api.models import MusicModelSpec

        spec = MusicModelSpec(name="Generator")
        assert spec.voice_changer is False
        assert spec.supports_seed is False
        assert spec.supports_background_noise_removal is False
        assert spec.supports_custom_voice_id is False
