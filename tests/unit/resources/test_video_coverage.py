"""
Comprehensive test coverage for venice_ai.resources.video module.

Covers all four async methods of the Video resource:
- queue (text-to-video and image-to-video, all optional params)
- quote (all optional params, both T2V and I2V paths)
- retrieve (PROCESSING, FAILED, COMPLETED statuses + fallback logic)
- complete
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.video import Video
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
    VideoQuoteResponse,
    VideoTranscriptionResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_response(
    data,
    *,
    content_type: str = "application/json",
    status: int = 200,
):
    """Build a mock aiohttp.ClientResponse for ``raw_response=True`` calls.

    ``data`` is the JSON-serialisable payload returned by ``.json()``.
    """
    resp = Mock()
    resp.headers = {"content-type": content_type}
    resp.status = status
    text = json.dumps(data) if not isinstance(data, str) else data
    resp.content_length = len(text)
    resp.json = AsyncMock(return_value=data)
    resp.text = AsyncMock(return_value=text)
    resp.read = AsyncMock(return_value=text.encode() if isinstance(text, str) else text)
    resp.close = Mock()
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def video_resource():
    """Create a Video resource backed by a mock client."""
    mock_client = AsyncMock()
    resource = Video(mock_client)
    return resource


# ---------------------------------------------------------------------------
# queue() – text-to-video
# ---------------------------------------------------------------------------


class TestVideoQueueTextToVideo:
    """Cover queue() when no image_url is provided (T2V path)."""

    @pytest.mark.asyncio
    async def test_queue_minimal_t2v(self, video_resource):
        """Lines 174-197: minimal T2V request (only required params)."""
        expected = VideoQueueResponse(model="wan-2.6-text-to-video", queue_id="q-1")
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.submit(
            model="wan-2.6-text-to-video",
            prompt="A sunset over the ocean",
            duration_seconds="5s",
        )

        assert result.model == "wan-2.6-text-to-video"
        assert result.queue_id == "q-1"
        video_resource._client.post.assert_called_once()
        call_kwargs = video_resource._client.post.call_args
        body = call_kwargs.kwargs["json_data"]
        assert body["model"] == "wan-2.6-text-to-video"
        assert body["prompt"] == "A sunset over the ocean"
        assert body["duration"] == "5s"
        # Optional keys should not be present (unless pydantic defaults)
        assert "image_url" not in body

    @pytest.mark.asyncio
    async def test_queue_t2v_all_optional_params(self, video_resource):
        """Lines 179-186: every optional param branch (negative_prompt,
        resolution, audio, aspect_ratio) with non-None values."""
        expected = VideoQueueResponse(model="wan-2.6-text-to-video", queue_id="q-2")
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.submit(
            model="wan-2.6-text-to-video",
            prompt="A cat playing piano",
            duration_seconds="10s",
            negative_prompt="blurry, low quality",
            resolution="1080p",
            audio=True,
            aspect_ratio="16:9",
        )

        assert result.queue_id == "q-2"
        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert body["negative_prompt"] == "blurry, low quality"
        assert body["resolution"] == "1080p"
        assert body["audio"] is True
        assert body["aspect_ratio"] == "16:9"


# ---------------------------------------------------------------------------
# queue() – image-to-video
# ---------------------------------------------------------------------------


class TestVideoQueueImageToVideo:
    """Cover queue() when image_url is provided (I2V path)."""

    @pytest.mark.asyncio
    async def test_queue_i2v_minimal(self, video_resource):
        """Lines 189-191: image_url triggers VideoImageToVideoRequest."""
        expected = VideoQueueResponse(model="wan-2.6-image-to-video", queue_id="q-3")
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.submit(
            model="wan-2.6-image-to-video",
            prompt="Animate this photo",
            duration_seconds="5s",
            image_url="https://example.com/photo.jpg",
        )

        assert result.queue_id == "q-3"
        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert body["image_url"] == "https://example.com/photo.jpg"

    @pytest.mark.asyncio
    async def test_queue_i2v_with_all_optional_params(self, video_resource):
        """Lines 179-191: I2V path exercising all optional branches."""
        expected = VideoQueueResponse(model="wan-2.6-image-to-video", queue_id="q-4")
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.submit(
            model="wan-2.6-image-to-video",
            prompt="Subtle motion",
            duration_seconds="5s",
            negative_prompt="blurry",
            resolution="720p",
            audio=False,
            aspect_ratio="9:16",
            image_url="data:image/png;base64,abc123",
        )

        assert result.queue_id == "q-4"
        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert body["image_url"] == "data:image/png;base64,abc123"
        assert body["negative_prompt"] == "blurry"
        assert body["audio"] is False

    @pytest.mark.asyncio
    async def test_queue_i2v_http_url(self, video_resource):
        """image_url starting with http:// (non-TLS)."""
        expected = VideoQueueResponse(model="model-i2v", queue_id="q-5")
        video_resource._client.post = AsyncMock(return_value=expected)

        await video_resource.submit(
            model="model-i2v",
            prompt="Go",
            duration_seconds="5s",
            image_url="http://example.com/img.jpg",
        )

        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert body["image_url"] == "http://example.com/img.jpg"


# ---------------------------------------------------------------------------
# quote()
# ---------------------------------------------------------------------------


class TestVideoQuote:
    """Cover all branches of the quote() method (lines 253-272)."""

    @pytest.mark.asyncio
    async def test_quote_minimal(self, video_resource):
        """Minimal quote: only the required fields per the API spec."""
        expected = VideoQuoteResponse(quote=0.05)
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.quote(
            model="wan-2-7-text-to-video",
            duration_seconds="5s",
        )

        assert result.quote == 0.05
        video_resource._client.post.assert_called_once()
        call_kwargs = video_resource._client.post.call_args
        assert call_kwargs.kwargs["cast_to"] is VideoQuoteResponse

    @pytest.mark.asyncio
    async def test_quote_all_optional_params(self, video_resource):
        """Every documented optional param forwards correctly."""
        expected = VideoQuoteResponse(quote=0.10)
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.quote(
            model="wan-2-7-text-to-video",
            duration_seconds="10s",
            resolution="1080p",
            audio=True,
            aspect_ratio="16:9",
            upscale_factor=2,
            video_url="https://example.com/source.mp4",
        )

        assert result.quote == 0.10
        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert body["resolution"] == "1080p"
        assert body["audio"] is True
        assert body["aspect_ratio"] == "16:9"
        assert body["upscale_factor"] == 2
        assert body["video_url"] == "https://example.com/source.mp4"

    @pytest.mark.asyncio
    async def test_quote_without_optional_params(self, video_resource):
        """Unset optionals must be excluded from the serialized body."""
        expected = VideoQuoteResponse(quote=1)
        video_resource._client.post = AsyncMock(return_value=expected)

        await video_resource.quote(
            model="wan-2-7-text-to-video",
            duration_seconds="5s",
        )

        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert "video_url" not in body
        assert "audio" not in body
        assert "aspect_ratio" not in body
        assert "resolution" not in body
        assert "upscale_factor" not in body


# ---------------------------------------------------------------------------
# retrieve() – PROCESSING status
# ---------------------------------------------------------------------------


class TestVideoRetrieveProcessing:
    """Cover retrieve() returning PROCESSING status (lines 332-357)."""

    @pytest.mark.asyncio
    async def test_retrieve_processing(self, video_resource):
        """Lines 354-357: dict response with status == PROCESSING."""
        payload = {
            "status": "PROCESSING",
            "average_execution_time": 30000.0,
            "execution_duration": 15000.0,
        }
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(payload),
        )

        result = await video_resource.retrieve(
            model="wan-2.6-text-to-video",
            queue_id="q-abc",
        )

        assert isinstance(result, VideoProcessingStatus)
        assert result.status == "PROCESSING"
        assert result.average_execution_time == 30000.0
        assert result.execution_duration == 15000.0


# ---------------------------------------------------------------------------
# retrieve() – FAILED status
# ---------------------------------------------------------------------------


class TestVideoRetrieveFailed:
    """Cover retrieve() returning FAILED status (lines 358-359)."""

    @pytest.mark.asyncio
    async def test_retrieve_failed(self, video_resource):
        """Lines 358-359: dict response with status == FAILED."""
        payload = {
            "status": "FAILED",
            "error": "Model overloaded",
            "error_code": "OVERLOADED",
        }
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(payload),
        )

        result = await video_resource.retrieve(
            model="wan-2.6-text-to-video",
            queue_id="q-def",
        )

        assert isinstance(result, VideoFailedStatus)
        assert result.status == "FAILED"
        assert result.error == "Model overloaded"


# ---------------------------------------------------------------------------
# retrieve() – COMPLETED status
# ---------------------------------------------------------------------------


class TestVideoRetrieveCompleted:
    """Cover retrieve() returning COMPLETED status (lines 360-361)."""

    @pytest.mark.asyncio
    async def test_retrieve_completed(self, video_resource):
        """Lines 360-361: dict response with status == COMPLETED."""
        payload = {
            "status": "COMPLETED",
            "url": "https://storage.example.com/video.mp4",
            "expires_at": "2025-12-31T23:59:59Z",
        }
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(payload),
        )

        result = await video_resource.retrieve(
            model="wan-2.6-text-to-video",
            queue_id="q-ghi",
        )

        assert isinstance(result, VideoCompletedStatus)
        assert result.status == "COMPLETED"
        assert result.url == "https://storage.example.com/video.mp4"

    @pytest.mark.asyncio
    async def test_retrieve_with_delete_media(self, video_resource):
        """Lines 332-337: delete_media_on_completion param is forwarded."""
        payload = {
            "status": "COMPLETED",
            "url": "https://storage.example.com/video.mp4",
        }
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(payload),
        )

        result = await video_resource.retrieve(
            model="wan-2.6-text-to-video",
            queue_id="q-jkl",
            delete_media_on_completion=True,
        )

        assert isinstance(result, VideoCompletedStatus)
        body = video_resource._client.post.call_args.kwargs["json_data"]
        assert body["delete_media_on_completion"] is True


# ---------------------------------------------------------------------------
# retrieve() – fallback parsing (lines 364-371)
# ---------------------------------------------------------------------------


class TestVideoRetrieveFallback:
    """Cover retrieve() fallback parsing when response is not a dict
    or has an unknown status field."""

    @pytest.mark.asyncio
    async def test_retrieve_fallback_non_dict_processing(self, video_resource):
        """Lines 409-413: non-dict JSON response that can be parsed as
        VideoProcessingStatus via the fallback loop."""
        # .json() returns a list (non-dict), so the isinstance(dict) check
        # at line 399 falls through to the fallback loop.  The loop tries
        # VideoProcessingStatus.model_validate on the raw data — lists are
        # not valid for any of the status types, but we can just verify
        # that the fallback path is exercised and raises ValueError.
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(
                [{"status": "PROCESSING"}],
            ),
        )

        with pytest.raises(ValueError, match="Unable to parse video retrieve response"):
            await video_resource.retrieve(
                model="wan-2.6-text-to-video",
                queue_id="q-mno",
            )

    @pytest.mark.asyncio
    async def test_retrieve_unknown_status_is_treated_as_in_progress(self, video_resource, caplog):
        """A status the SDK does not know is kept and treated as still in
        progress, so polling continues instead of failing on the first poll."""
        payload = {"status": "UNKNOWN_STATUS"}
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(payload),
        )

        result = await video_resource.retrieve(
            model="wan-2.6-text-to-video",
            queue_id="q-pqr",
        )

        assert isinstance(result, VideoProcessingStatus)
        assert result.status == "UNKNOWN_STATUS"
        assert result.progress_percent == 0.0
        assert "unknown status 'UNKNOWN_STATUS'" in caplog.text

    @pytest.mark.asyncio
    async def test_retrieve_fallback_non_parseable(self, video_resource):
        """Lines 409-417: JSON response that cannot be parsed into any status type."""
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response("not-a-dict"),
        )

        with pytest.raises(ValueError, match="Unable to parse video retrieve response"):
            await video_resource.retrieve(
                model="wan-2.6-text-to-video",
                queue_id="q-stu",
            )

    @pytest.mark.asyncio
    async def test_retrieve_fallback_dict_no_status_key(self, video_resource):
        """Lines 399-401, 409: dict without 'status' key triggers fallback."""
        # A dict without 'status' will go through the if-elif chain with
        # status=None, falling through to the fallback loop.
        payload = {"url": "https://example.com/video.mp4"}
        video_resource._client.post = AsyncMock(
            return_value=_make_raw_response(payload),
        )

        # The fallback loop will try each type and all will fail
        with pytest.raises(ValueError, match="Unable to parse video retrieve response"):
            await video_resource.retrieve(
                model="wan-2.6-text-to-video",
                queue_id="q-vwx",
            )


# ---------------------------------------------------------------------------
# retrieve() – defensive response-handling paths
# ---------------------------------------------------------------------------


class TestVideoRetrieveDefensivePaths:
    """Cover defensive paths for non-JSON responses and JSON parse failures
    in retrieve()."""

    @pytest.mark.asyncio
    async def test_retrieve_binary_response(self, video_resource):
        """Lines 362-382: non-JSON content-type returns synthetic COMPLETED
        status without reading the body."""
        resp = _make_raw_response(
            "binary-video-data",
            content_type="video/mp4",
        )
        video_resource._client.post = AsyncMock(return_value=resp)

        result = await video_resource.retrieve(
            model="wan-2.6-text-to-video",
            queue_id="q-binary",
        )

        assert isinstance(result, VideoCompletedStatus)
        assert result.status == "COMPLETED"
        assert result.url is None
        resp.close.assert_called_once()
        # .json() should NOT have been called for a non-JSON response
        resp.json.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_json_parse_failure(self, video_resource):
        """Lines 385-399: JSON parse failure re-raises after logging."""
        resp = _make_raw_response(
            "{invalid-json",
            content_type="application/json",
        )
        resp.json = AsyncMock(side_effect=ValueError("bad json"))
        video_resource._client.post = AsyncMock(return_value=resp)

        with pytest.raises(ValueError, match="bad json"):
            await video_resource.retrieve(
                model="wan-2.6-text-to-video",
                queue_id="q-badjson",
            )

    @pytest.mark.asyncio
    async def test_retrieve_json_parse_failure_body_unreadable(self, video_resource):
        """Lines 388-391: when .text() also fails during error logging,
        the original JSON error is still re-raised."""
        resp = _make_raw_response(
            "irrelevant",
            content_type="application/json",
        )
        resp.json = AsyncMock(side_effect=ValueError("bad json"))
        resp.text = AsyncMock(side_effect=RuntimeError("connection reset"))
        video_resource._client.post = AsyncMock(return_value=resp)

        with pytest.raises(ValueError, match="bad json"):
            await video_resource.retrieve(
                model="wan-2.6-text-to-video",
                queue_id="q-unreadable",
            )


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestVideoComplete:
    """Cover the complete() method (lines 407-413)."""

    @pytest.mark.asyncio
    async def test_complete_success(self, video_resource):
        """Lines 407-417: successful complete call."""
        expected = VideoCompleteResponse(success=True)
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.cancel(
            model="wan-2.6-text-to-video",
            queue_id="q-xyz",
        )

        assert result.success is True
        video_resource._client.post.assert_called_once()
        call_kwargs = video_resource._client.post.call_args
        assert call_kwargs.args[0] == "video/complete"
        body = call_kwargs.kwargs["json_data"]
        assert body["model"] == "wan-2.6-text-to-video"
        assert body["queue_id"] == "q-xyz"
        assert call_kwargs.kwargs["cast_to"] is VideoCompleteResponse

    @pytest.mark.asyncio
    async def test_complete_failure(self, video_resource):
        """complete() with success=False."""
        expected = VideoCompleteResponse(success=False)
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.cancel(
            model="wan-2.6-text-to-video",
            queue_id="q-fail",
        )

        assert result.success is False


class TestVideoTranscribe:
    """Cover Video.transcribe() for both response_format paths."""

    @pytest.mark.asyncio
    async def test_transcribe_json_default(self, video_resource):
        expected = VideoTranscriptionResponse(transcript="hello world", lang="en")
        video_resource._client.post = AsyncMock(return_value=expected)

        result = await video_resource.transcribe(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

        assert result.transcript == "hello world"
        assert result.lang == "en"
        video_resource._client.post.assert_called_once_with(
            "video/transcriptions",
            json_data={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "response_format": "json",
            },
            cast_to=VideoTranscriptionResponse,
        )

    @pytest.mark.asyncio
    async def test_transcribe_text_format(self, video_resource):
        raw = _make_raw_response("plain transcript", content_type="text/plain")
        video_resource._client.post = AsyncMock(return_value=raw)

        result = await video_resource.transcribe(
            "https://www.youtube.com/watch?v=abc",
            response_format="text",
        )

        assert result == "plain transcript"
        video_resource._client.post.assert_called_once_with(
            "video/transcriptions",
            json_data={
                "url": "https://www.youtube.com/watch?v=abc",
                "response_format": "text",
            },
            raw_response=True,
        )
        raw.text.assert_awaited_once()
        raw.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_rejects_non_http_url(self, video_resource):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await video_resource.transcribe("file:///local/path.mp4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
