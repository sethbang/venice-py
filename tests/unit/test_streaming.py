"""
Comprehensive tests for the streaming module.

Tests cover Stream and BytesResponse classes with focus on error handling,
resource cleanup, and state management.
"""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from venice_ai.exceptions import (
    APIConnectionError,
    APITimeoutError,
    StreamConsumedError,
    VeniceError,
)
from venice_ai.streaming import BytesResponse, Stream


class TestStream:
    """Test Stream wrapper."""

    def test_init(self):
        """Test Stream initialization."""
        mock_iterator = AsyncMock()
        mock_client = Mock()

        stream = Stream(mock_iterator, client=mock_client)

        assert stream._iterator is mock_iterator
        assert stream._client is mock_client
        assert stream._consumed is False

    def test_aiter_returns_self(self):
        """Test that __aiter__ returns self."""
        mock_iterator = AsyncMock()
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        result = stream.__aiter__()

        assert result is stream

    def test_aiter_raises_when_consumed(self):
        """Test that __aiter__ raises StreamConsumedError when already consumed."""
        mock_iterator = AsyncMock()
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)
        stream._consumed = True

        with pytest.raises(StreamConsumedError, match="Cannot iterate over a consumed stream"):
            stream.__aiter__()

    @pytest.mark.asyncio
    async def test_anext_normal_iteration(self):
        """Test normal iteration through Stream."""
        mock_iterator = AsyncMock()
        mock_iterator.__anext__ = AsyncMock(side_effect=[1, 2, 3, StopAsyncIteration])
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        assert await stream.__anext__() == 1
        assert await stream.__anext__() == 2
        assert await stream.__anext__() == 3
        assert stream._consumed is False  # Not consumed until StopAsyncIteration

    @pytest.mark.asyncio
    async def test_anext_stop_async_iteration_marks_consumed_and_closes(self):
        """Test that StopAsyncIteration marks stream as consumed and calls close."""
        mock_iterator = AsyncMock()
        mock_iterator.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        with (
            patch.object(stream, "close", new_callable=AsyncMock) as mock_close,
            pytest.raises(StopAsyncIteration),
        ):
            await stream.__anext__()

            assert stream._consumed is True
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_anext_timeout_error_handling(self):
        """Test TimeoutError handling in __anext__."""
        mock_iterator = AsyncMock()
        mock_iterator.__anext__ = AsyncMock(side_effect=TimeoutError("Test timeout"))
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        with (
            patch.object(stream, "close", new_callable=AsyncMock) as mock_close,
            pytest.raises(APITimeoutError, match="Stream timed out before it finished"),
        ):
            await stream.__anext__()

            assert stream._consumed is True
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_anext_client_error_handling(self):
        """Test aiohttp.ClientError handling in __anext__."""
        mock_iterator = AsyncMock()
        mock_iterator.__anext__ = AsyncMock(side_effect=aiohttp.ClientError("Test client error"))
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        with (
            patch.object(stream, "close", new_callable=AsyncMock) as mock_close,
            pytest.raises(
                APIConnectionError, match="connection failed while streaming the response"
            ),
        ):
            await stream.__anext__()

            assert stream._consumed is True
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_anext_api_error_re_raised(self):
        """Test that APIError subclasses are re-raised directly."""
        from venice_ai.exceptions import APIError

        mock_iterator = AsyncMock()
        # APIError requires a response parameter
        mock_response = Mock()
        api_error = APIError("Test API error", response=mock_response)
        mock_iterator.__anext__ = AsyncMock(side_effect=api_error)
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        with (
            patch.object(stream, "close", new_callable=AsyncMock) as mock_close,
            pytest.raises(APIError, match="Test API error"),
        ):
            await stream.__anext__()

            assert stream._consumed is True
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_anext_runtime_error_re_raised(self):
        """Test that RuntimeError subclasses are re-raised directly."""
        mock_iterator = AsyncMock()
        runtime_error = RuntimeError("Test runtime error")
        mock_iterator.__anext__ = AsyncMock(side_effect=runtime_error)
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        with (
            patch.object(stream, "close", new_callable=AsyncMock) as mock_close,
            pytest.raises(RuntimeError, match="Test runtime error"),
        ):
            await stream.__anext__()

            assert stream._consumed is True
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_anext_unexpected_exception_wrapped(self):
        """Test that unexpected exceptions are wrapped in VeniceError."""

        mock_iterator = AsyncMock()
        unexpected_error = ValueError("Unexpected error")
        mock_iterator.__anext__ = AsyncMock(side_effect=unexpected_error)
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        with (
            patch.object(stream, "close", new_callable=AsyncMock) as mock_close,
            patch("venice_ai.streaming.logger") as mock_log,
        ):
            with pytest.raises(VeniceError, match="Stream error: Unexpected error"):
                await stream.__anext__()

            assert stream._consumed is True
            mock_close.assert_called_once()
            mock_log.exception.assert_called_once()
            assert "Unexpected exception in Stream" in mock_log.exception.call_args[0][0]

    @pytest.mark.asyncio
    async def test_close_with_aclose_method(self):
        """Test close() when iterator has an aclose method."""
        mock_iterator = AsyncMock()
        mock_iterator.aclose = AsyncMock()
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        await stream.close()

        mock_iterator.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_aclose_method(self):
        """Test close() when iterator doesn't have an aclose method."""
        mock_iterator = Mock()  # No aclose method
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        # Should not raise an exception
        await stream.close()

    @pytest.mark.asyncio
    async def test_close_with_exception_in_iterator_aclose(self):
        """Test close() handles exceptions from iterator.aclose gracefully."""
        mock_iterator = AsyncMock()
        mock_iterator.aclose = AsyncMock(side_effect=Exception("Close failed"))
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        # Should not raise an exception - errors are silently handled
        await stream.close()

        mock_iterator.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_async_iteration_lifecycle(self):
        """Test complete iteration lifecycle."""
        data = [1, 2, 3]

        async def mock_async_iter():
            for item in data:
                yield item

        mock_iterator = mock_async_iter()
        mock_client = Mock()
        stream = Stream(mock_iterator, client=mock_client)

        results = []
        with patch.object(stream, "close", new_callable=AsyncMock) as mock_close:
            async for item in stream:
                results.append(item)

        assert results == data
        # close() should be called when StopAsyncIteration is reached
        mock_close.assert_called_once()
        assert stream._consumed is True


class TestBytesResponse:
    """Test BytesResponse wrapper."""

    def test_init(self):
        """Test BytesResponse initialization."""
        content = b"test content"
        response = Mock()

        bytes_response = BytesResponse(content, response)

        assert bytes_response.content == content
        assert bytes_response.response is response

    def test_attributes_accessible(self):
        """Test that content and response attributes are accessible."""
        content = b"hello world"
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/octet-stream"}

        bytes_response = BytesResponse(content, mock_response)

        assert bytes_response.content == content
        assert bytes_response.response.status == 200
        assert bytes_response.response.headers["content-type"] == "application/octet-stream"


class TestStreamIntegration:
    """Integration tests for Stream class."""

    @pytest.mark.asyncio
    async def test_stream_with_real_async_iterator(self):
        """Test Stream with a real async iterator."""
        data = ["chunk1", "chunk2", "chunk3"]

        async def async_data_generator():
            for item in data:
                yield item

        iterator = async_data_generator()
        mock_client = Mock()

        stream = Stream(iterator, client=mock_client)

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert collected == data
        assert stream._consumed is True


class TestStreamErrorScenarios:
    """Test various error scenarios for streams."""

    @pytest.mark.asyncio
    async def test_stream_iterator_exception_during_iteration(self):
        """Test Stream handling of iterator exceptions during normal iteration."""

        async def failing_async_iterator():
            yield 1
            yield 2
            raise ValueError("Iterator failed")

        mock_client = Mock()
        stream = Stream(failing_async_iterator(), client=mock_client)

        assert await stream.__anext__() == 1
        assert await stream.__anext__() == 2

        with (
            patch.object(stream, "close", new_callable=AsyncMock),
            pytest.raises(VeniceError, match="Stream error: Iterator failed"),
        ):
            await stream.__anext__()


class TestChatCompletionChunkUsage:
    """The streaming chunk usage model must surface the detail/cache
    breakdown the wire sends and must not silently drop unmodeled usage keys.
    """

    def test_chunk_usage_preserves_prompt_tokens_details_cached_tokens(self):
        """Non-regression: the exact 2026-06-04 wire capture round-trips."""
        from venice_ai.types.api.streaming import ChatCompletionChunk

        # Streamed final chunk captured 2026-06-04.
        wire = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": {
                "prompt_tokens": 1657,
                "completion_tokens": 85,
                "total_tokens": 1742,
                "prompt_tokens_details": {"cached_tokens": 576},
            },
        }
        chunk = ChatCompletionChunk.model_validate(wire)
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 1657
        assert chunk.usage.completion_tokens == 85
        assert chunk.usage.total_tokens == 1742
        assert chunk.usage.prompt_tokens_details is not None
        assert chunk.usage.prompt_tokens_details.cached_tokens == 576

    def test_chunk_usage_models_detail_and_cache_fields(self):
        """The detail/cache breakdown must be typed and accessible, not dropped."""
        from venice_ai.types.api.streaming import ChatCompletionChunk

        wire = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": {
                "prompt_tokens": 1657,
                "completion_tokens": 85,
                "total_tokens": 1742,
                "prompt_tokens_details": {"cached_tokens": 576},
                "completion_tokens_details": {"reasoning_tokens": 40},
                "cache_read_input_tokens": 576,
                "cache_creation_input_tokens": 12,
            },
        }
        usage = ChatCompletionChunk.model_validate(wire).usage
        assert usage is not None

        # Typed completion-token breakdown survives and is accessible.
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 40

        # Top-level cache counters survive.
        assert usage.cache_read_input_tokens == 576
        assert usage.cache_creation_input_tokens == 12

        # And they round-trip through model_dump (i.e. were not dropped on parse).
        dumped = usage.model_dump(exclude_none=True)
        assert dumped.get("completion_tokens_details") == {"reasoning_tokens": 40}
        assert dumped.get("cache_read_input_tokens") == 576
        assert dumped.get("cache_creation_input_tokens") == 12

    def test_chunk_usage_does_not_drop_unmodeled_keys(self):
        """A wholly-unmodeled usage key must survive (extra='allow' policy)."""
        from venice_ai.types.api.streaming import ChatCompletionChunk

        wire = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "future_usage_field": 99,
            },
        }
        usage = ChatCompletionChunk.model_validate(wire).usage
        assert usage is not None
        assert usage.model_extra is not None
        assert usage.model_extra.get("future_usage_field") == 99


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers: dict[str, str] = {}


class TestStreamErrorLogging:
    """An error handed to the caller is not also logged as a failure."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (401, {"error": "Authentication failed"}),
            (
                404,
                {"error": "Specified model not found: nope. Did you mean: model-a, model-b?"},
            ),
        ],
        ids=["401", "404"],
    )
    async def test_classified_api_error_is_raised_without_error_log(self, status, body, caplog):
        import logging

        from venice_ai.exceptions import APIError, _make_status_error

        error = _make_status_error(
            f"API request failed with status {status}", body=body, response=_Resp(status)
        )
        mock_iterator = AsyncMock()
        mock_iterator.__anext__ = AsyncMock(side_effect=error)
        stream = Stream(mock_iterator, client=Mock())

        with (
            caplog.at_level(logging.DEBUG, logger="venice_ai.streaming"),
            pytest.raises(APIError) as exc_info,
        ):
            await stream.__anext__()

        assert exc_info.value is error
        assert stream._consumed is True
        loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not loud, [r.getMessage() for r in loud]
        assert not any("Did you mean" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unexpected_exception_is_still_logged_at_error(self, caplog):
        import logging

        mock_iterator = AsyncMock()
        mock_iterator.__anext__ = AsyncMock(side_effect=ValueError("boom"))
        stream = Stream(mock_iterator, client=Mock())

        with (
            caplog.at_level(logging.DEBUG, logger="venice_ai.streaming"),
            pytest.raises(VeniceError),
        ):
            await stream.__anext__()

        assert any(r.levelno >= logging.ERROR for r in caplog.records)
