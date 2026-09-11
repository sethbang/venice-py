"""
Streaming Response Handler
=========================

This module provides streaming response handling for the Venice AI SDK,
enabling real-time processing of API responses as they arrive from the server.

The streaming system is designed to handle various types of streaming responses
including chat completions, audio generation, and other real-time API outputs.
It provides proper resource management, error handling, and type safety.

Key Features:
    * **Real-time Processing**: Stream responses as they arrive
    * **Resource Management**: Automatic cleanup of network resources
    * **Error Handling**: Comprehensive error handling for network issues
    * **Type Safety**: Generic typing for different chunk types
    * **Consumption Tracking**: Prevents multiple iterations over consumed streams

Streaming Types:
    * **Chat Completions**: Real-time text generation
    * **Audio Responses**: Streaming audio data
    * **Image Generation**: Progressive image data (future)
    * **Custom Responses**: Extensible for new streaming endpoints

Example:
    >>> from venice_ai import VeniceClient
    >>>
    >>> client = VeniceClient(api_key="your-api-key")
    >>>
    >>> # Create a streaming chat completion. Model IDs change; resolve one
    >>> # from the live catalog rather than hardcoding.
    >>> stream = await client.chat.completions.create(
    ...     model=await client.models.resolve_chat(),
    ...     messages=[{"role": "user", "content": "Write a story"}],
    ...     stream=True
    ... )
    >>>
    >>> # Process chunks as they arrive
    >>> async for chunk in stream:
    ...     if chunk.choices[0].delta.content:
    ...         print(chunk.choices[0].delta.content, end="")
    >>>
    >>> # Stream is automatically closed when iteration completes
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    TypeVar,
)

if TYPE_CHECKING:
    from ._client import VeniceClient
    from .types.api.chat import ChatCompletionResponse
    from .types.api.streaming import ChatCompletionChunk  # noqa: F401

import asyncio

import aiohttp

from .exceptions import (
    APIConnectionError,
    APITimeoutError,
    StreamConsumedError,
)

logger = logging.getLogger(__name__)

#: Covariant type variable for async iterator protocol
T_co = TypeVar("T_co", covariant=True)


class AsyncClosableIterator(Protocol[T_co]):
    """
    Protocol for async iterators that support resource cleanup.

    This protocol extends the standard AsyncIterator to include the aclose()
    method that many async generators provide for proper resource cleanup.
    This ensures that network connections, file handles, and other resources
    are properly released when streaming is complete.

    The aclose() method is particularly important for streaming HTTP responses
    where the underlying connection needs to be closed to free up resources
    and prevent connection leaks.

    Methods:
        aclose(): Async method to close the iterator and release resources
    """

    def __aiter__(self) -> AsyncIterator[T_co]: ...

    async def __anext__(self) -> T_co: ...

    async def aclose(self) -> None:
        """
        Close the async iterator and release any associated resources.

        This method should be called to ensure proper cleanup of resources
        such as network connections, file handles, or other system resources
        associated with the iterator.
        """
        ...


class Stream[ChunkType]:
    """
    Wrapper for handling streaming responses from the Venice AI API.

    This class provides a robust interface for iterating over streaming API
    responses, managing the underlying iterator lifecycle, and ensuring proper
    resource cleanup. It handles various error conditions and provides
    consumption tracking to prevent multiple iterations.

    The Stream class is generic and can handle different types of streaming
    responses by parameterizing the ChunkType. It automatically handles
    network errors, timeouts, and other streaming-related issues.

    Type Parameters:
        ChunkType: The type of chunks yielded by the stream (e.g., Dict[str, Any])

    Attributes:
        _iterator: The underlying async iterator that yields response chunks
        _client: Reference to the Venice client for context and resource management
        _consumed: Flag indicating whether the stream has been fully consumed

    Error Handling:
        * StreamConsumedError: Raised when attempting to iterate a consumed stream
        * APITimeoutError: Raised when streaming operations timeout
        * APIConnectionError: Raised for network connectivity issues
        * VeniceError: Raised for other unexpected streaming errors

    Example:
        >>> # Basic streaming usage
        >>> stream = Stream(response_iterator, client=client)
        >>> async for chunk in stream:
        ...     process_chunk(chunk)
        >>>
        >>> # Manual cleanup (optional, done automatically)
        >>> await stream.close()

        >>> # Error handling
        >>> try:
        ...     async for chunk in stream:
        ...         print(chunk)
        ... except StreamConsumedError:
        ...     print("Stream was already consumed")
        ... except APITimeoutError:
        ...     print("Streaming timed out")
    """

    def __init__(self, iterator: AsyncIterator[ChunkType], *, client: VeniceClient):
        """
        Initialize a new Stream wrapper.

        Args:
            iterator: The underlying async iterator that yields response chunks.
                     This is typically created by the HTTP client when making
                     streaming API requests.
            client: The Venice client instance for context and resource management.
                   Used for logging, metrics, and potential resource cleanup.

        Note:
            The stream starts in an unconsumed state and can be iterated over
            exactly once. After consumption, attempts to iterate will raise
            StreamConsumedError.
        """
        self._iterator = iterator
        self._client = client  # Retain client for potential context or resource management
        self._consumed = False  # Track if the stream has been consumed
        self._start_time: float | None = None  # Track stream duration
        self._bytes_streamed = 0  # Track total bytes

        # Track custom stream creation
        try:
            from .observability.metrics import get_enhanced_metrics

            metrics = get_enhanced_metrics()
            if metrics._enabled:
                stream_type = type(self).__name__
                metrics.custom_stream_created_total.labels(stream_type=stream_type).inc()
        except Exception:
            pass  # nosec B110

    async def _handle_stream_exhausted(self) -> None:
        """Handle normal stream exhaustion."""
        self._consumed = True
        await self.close()

    async def _handle_stream_error(self, error: Exception) -> None:
        """Handle known stream errors (timeout, client errors)."""
        self._consumed = True
        await self.close()

    async def _handle_unexpected_error(self, error: Exception) -> None:
        """Handle unexpected errors during streaming."""
        logger.error(f"Unexpected streaming error: {error}")
        self._consumed = True
        await self.close()

    def _convert_to_api_error(self, error: Exception) -> Exception:
        """
        Convert known errors to appropriate API error types.

        Uses standardized exception hierarchy - more specific exceptions
        are checked first before general ones.
        """
        if isinstance(error, aiohttp.ServerTimeoutError):
            # Server-side timeout - more specific, check first
            return APITimeoutError("Server timeout during streaming", original_error=error)
        elif isinstance(error, TimeoutError):
            # General timeout - includes client-side timeouts
            return APITimeoutError("Stream request timed out", original_error=error)
        elif isinstance(error, aiohttp.ClientError):
            return APIConnectionError("Connection error during streaming", original_error=error)
        return error

    async def __anext__(self) -> ChunkType:
        """
        Get the next chunk from the streaming response.

        This method implements the async iterator protocol, yielding chunks
        as they become available from the underlying stream. It handles various
        error conditions and ensures proper resource cleanup on stream completion.

        Returns:
            The next chunk from the streaming response. The exact type depends
            on the ChunkType parameter but is typically a dictionary containing
            the parsed response data.

        Raises:
            StopAsyncIteration: When the stream is exhausted (normal completion)
            StreamConsumedError: When attempting to iterate a consumed stream
            APITimeoutError: When streaming operations timeout
            APIConnectionError: When network connectivity issues occur
            VeniceError: For other unexpected streaming errors

        Error Handling:
            * Timeouts are converted to APITimeoutError with original exception context
            * Network errors are converted to APIConnectionError
            * Unexpected errors are logged and wrapped in VeniceError
            * CancelledError is always re-raised for graceful shutdown
            * Stream is marked as consumed and cleaned up on any error

        Note:
            This method automatically marks the stream as consumed and closes
            resources when the stream is exhausted or encounters an error.
        """
        # Start timing on first chunk
        if self._start_time is None:
            self._start_time = time.time()

        try:
            # Simply pass through the item from the raw iterator
            chunk = await self._iterator.__anext__()

            # Track bytes if chunk is bytes
            if isinstance(chunk, bytes):
                self._bytes_streamed += len(chunk)

            return chunk
        except StopAsyncIteration:
            await self._handle_stream_exhausted()
            raise
        except (TimeoutError, aiohttp.ClientError) as e:
            await self._handle_stream_error(e)
            raise self._convert_to_api_error(e) from e
        except asyncio.CancelledError:
            raise  # Always re-raise for graceful shutdown
        except Exception as e:
            from .exceptions import APIError, VeniceError

            # APIError subclasses are re-raised unchanged to preserve detailed API
            # error information (status codes, headers, retry-after, etc.)
            if isinstance(e, APIError):
                await self._handle_unexpected_error(e)
                raise

            # RuntimeError indicates programmer errors (assertions, invalid states)
            # that should be visible for debugging. Re-raise without wrapping.
            if isinstance(e, RuntimeError):
                logger.error(f"Runtime error in stream: {e}")
                await self._handle_unexpected_error(e)
                raise

            # Wrap all other unexpected exceptions with context for better debugging
            logger.exception(f"Unexpected exception in Stream: {type(e).__name__}: {e}")
            await self._handle_unexpected_error(e)
            raise VeniceError(f"Stream error: {str(e)}") from e

    def __aiter__(self) -> AsyncIterator[ChunkType]:
        """
        Return the async iterator object for the stream.

        This method implements the async iterable protocol, allowing the stream
        to be used in async for loops. It includes consumption checking to
        prevent multiple iterations over the same stream.

        Returns:
            The stream iterator (self) for use in async iteration

        Raises:
            StreamConsumedError: If the stream has already been consumed by
                                a previous iteration

        Example:
            >>> stream = Stream(iterator, client=client)
            >>> async for chunk in stream:  # This calls __aiter__()
            ...     process_chunk(chunk)
            >>>
            >>> # Attempting to iterate again will raise an error
            >>> async for chunk in stream:  # StreamConsumedError
            ...     pass
        """
        if self._consumed:
            raise StreamConsumedError("Cannot iterate over a consumed stream.")
        return self

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager, closing the stream."""
        await self.close()

    def get_iterator(self) -> AsyncIterator[ChunkType]:
        """Return the underlying async iterator and mark this stream as consumed.

        The caller takes ownership of the iterator. This stream instance should
        not be iterated after calling this method.
        """
        self._consumed = True
        return self._iterator

    async def close(self) -> None:
        """
        Close the stream and release any underlying resources.

        This method ensures proper cleanup of the underlying iterator and any
        associated network resources. It's designed to be idempotent and safe
        to call multiple times. The method is automatically called when stream
        iteration completes or encounters an error.

        Resource Cleanup:
            * Calls aclose() on the underlying iterator if supported
            * Silently handles any cleanup errors to prevent resource leaks
            * Ensures network connections are properly closed
            * Frees any associated memory or file handles

        Example:
            >>> stream = Stream(iterator, client=client)
            >>> try:
            ...     async for chunk in stream:
            ...         process_chunk(chunk)
            ... finally:
            ...     await stream.close()  # Optional, done automatically

        Note:
            This method never raises exceptions. Any errors during cleanup
            are silently handled to ensure the application can continue
            running even if resource cleanup fails.
        """
        # Record metrics before closing
        if self._start_time is not None:
            try:
                from .observability.metrics import get_enhanced_metrics

                metrics = get_enhanced_metrics()
                if metrics._enabled:
                    stream_type = type(self).__name__
                    duration = time.time() - self._start_time

                    # Record stream duration
                    metrics.custom_stream_duration_seconds.labels(stream_type=stream_type).observe(
                        duration
                    )

                    # Record bytes streamed if any
                    if self._bytes_streamed > 0:
                        metrics.custom_stream_bytes_total.labels(stream_type=stream_type).inc(
                            self._bytes_streamed
                        )
            except Exception as e:
                # Metrics recording failures must not prevent stream closure.
                logger.debug(
                    f"Non-critical error during stream metrics recording: {type(e).__name__}",
                    extra={
                        "error_type": type(e).__name__,
                        "stream_type": type(self).__name__,
                    },
                )

        # Clean up resources if the iterator supports aclose method
        # Many async generators provide aclose() for proper resource cleanup
        if hasattr(self._iterator, "aclose"):
            with contextlib.suppress(Exception):  # nosec B110
                await self._iterator.aclose()  # pyright: ignore[reportAttributeAccessIssue]


def _assemble_reasoning(summary_parts: list[str], encrypted_parts: list[str]) -> str | None:
    """Join reasoning deltas so the assembled field survives a round trip.

    An encrypted reasoning block is unterminated: the server reads it from its
    header to the end of ``reasoning_content``. Joining deltas in arrival order
    therefore welds any summary the model emitted *after* the block onto its
    token, and echoing that back is rejected. Summary text *before* it is
    tolerated, so putting the block last preserves every delta and keeps the
    field valid — the same shape the non-streaming endpoint returns.

    This holds for the one block per response observed in practice. Because the
    server parses to end-of-string, a flat text field cannot carry two blocks
    under any ordering; should a response ever contain more, only the last would
    be recoverable and the API would need a structured field to express it.
    """
    if not summary_parts and not encrypted_parts:
        return None
    return "".join(summary_parts) + "".join(encrypted_parts)


class ChatStream(Stream["ChatCompletionChunk"]):
    """Enhanced stream for chat completions with convenience accessors.

    Wraps a ``Stream[ChatCompletionChunk]`` and adds:

    * :meth:`text_deltas` — yields only text content strings (display-only)
    * :meth:`collect` — silently consumes the stream and returns a full
      :class:`~venice_ai.types.api.chat.ChatCompletionResponse`
    * :meth:`collect_with_deltas` — yields text deltas live AND populates
      :attr:`final_response` once iteration completes; one pass, both signals

    Example — display deltas live and use the final aggregated response::

        async with await client.chat.completions.stream(
            model=model, messages=messages,
        ) as s:
            async for text in s.collect_with_deltas():
                print(text, end="", flush=True)
            print()
            assert s.final_response is not None
            print("usage:", s.final_response.usage)
    """

    def __init__(
        self,
        iterator: AsyncIterator[ChatCompletionChunk],
        *,
        client: VeniceClient,
    ):
        super().__init__(iterator, client=client)
        self._final_response: ChatCompletionResponse | None = None
        self._reasoning_summary: str | None = None

    @property
    def final_response(self) -> ChatCompletionResponse | None:
        """Assembled response after :meth:`collect` or :meth:`collect_with_deltas`.

        ``None`` until one of those two methods has finished consuming the
        stream. :meth:`text_deltas` does not populate this — use
        :meth:`collect_with_deltas` if you need both live deltas and the
        final aggregated response.
        """
        return self._final_response

    @property
    def reasoning_summary(self) -> str | None:
        """The plain-language reasoning summary, without any encrypted block.

        ``None`` until :meth:`collect` or :meth:`collect_with_deltas` has
        finished consuming the stream, and for models that emit no reasoning.

        Use this to display a model's thinking. The assembled
        ``message.reasoning_content`` deliberately still carries the encrypted
        reasoning block, because the API requires it back verbatim on the next
        turn; rendering that field shows users several KB of opaque token.
        """
        return self._reasoning_summary

    async def text_deltas(self) -> AsyncIterator[str]:
        """Yield only text content deltas, filtering empty/None.

        Does not populate :attr:`final_response`. If you want both the
        deltas and the final aggregated response in one pass, use
        :meth:`collect_with_deltas` instead.
        """
        async for chunk in self:
            for choice in chunk.choices:
                if choice.delta and choice.delta.content:
                    yield choice.delta.content

    async def collect(self) -> ChatCompletionResponse:
        """Consume the entire stream and return a ``ChatCompletionResponse``.

        Accumulates text, reasoning content, and tool calls from all chunks
        into a single non-streaming response object.

        .. note::
            Only the first choice (index 0) is tracked. For ``n > 1`` requests,
            use the raw ``Stream`` iterator and accumulate per-choice yourself.
            A one-time warning is emitted if non-zero choices are observed.

        :raises ValueError: If the stream completes without a ``finish_reason``
            on choice 0 (typically indicates the stream was interrupted).
        """
        import warnings

        from .types.api.chat import ChatCompletionResponse

        text_parts: list[str] = []
        reasoning_summary_parts: list[str] = []
        reasoning_encrypted_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        model = ""
        stream_id = ""
        created = 0
        usage = None
        finish_reason: str | None = None
        warned_multi_choice = False

        async for chunk in self:
            model = chunk.model or model
            if chunk.id:
                stream_id = chunk.id
            if chunk.created:
                created = chunk.created
            for choice in chunk.choices:
                if choice.index != 0:
                    if not warned_multi_choice:
                        warnings.warn(
                            "ChatStream.collect() only tracks choice 0; non-zero "
                            "indices ignored. Use the raw Stream iterator for n>1.",
                            stacklevel=2,
                        )
                        warned_multi_choice = True
                    continue
                if choice.delta:
                    if choice.delta.content:
                        text_parts.append(choice.delta.content)
                    if choice.delta.reasoning_content:
                        if choice.delta.reasoning_encrypted:
                            reasoning_encrypted_parts.append(choice.delta.reasoning_content)
                        else:
                            reasoning_summary_parts.append(choice.delta.reasoning_content)
                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index or 0
                            if idx not in tool_calls_by_index:
                                tool_calls_by_index[idx] = {
                                    "id": tc.id or "",
                                    "type": tc.type or "function",
                                    "name": "",
                                    "arguments": "",
                                }
                            entry = tool_calls_by_index[idx]
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    entry["name"] += tc.function.name
                                if tc.function.arguments:
                                    entry["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            if chunk.usage:
                usage = chunk.usage

        if finish_reason is None:
            raise ValueError(
                "Stream completed without a finish_reason — likely interrupted "
                "before the final chunk arrived."
            )

        message: dict[str, object] = {
            "role": "assistant",
            "content": "".join(text_parts) if text_parts else None,
            "reasoning_content": _assemble_reasoning(
                reasoning_summary_parts, reasoning_encrypted_parts
            ),
            "tool_calls": [
                {
                    "id": entry["id"],
                    "type": "function",
                    "function": {"name": entry["name"], "arguments": entry["arguments"]},
                }
                for _, entry in sorted(tool_calls_by_index.items())
            ]
            if tool_calls_by_index
            else None,
        }
        # Flag the message the way the non-streaming endpoint does — but only when
        # there is a block, so responses without reasoning gain no stray key.
        if reasoning_encrypted_parts:
            message["reasoning_encrypted"] = True

        payload: dict[str, object] = {
            "id": stream_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        # Only include usage when the API sent one; otherwise leave it None.
        # Stream chunks carry ChatUsage; convert to dict so pydantic re-validates
        # as ChatCompletionResponse.usage (ChatUsage) rather than rejecting the
        # cross-model instance under strict mode.
        if usage is not None:
            payload["usage"] = usage.model_dump()
        response = ChatCompletionResponse.model_validate(payload)
        self._reasoning_summary = (
            "".join(reasoning_summary_parts) if reasoning_summary_parts else None
        )
        self._final_response = response
        return response

    async def collect_with_deltas(self) -> AsyncIterator[str]:
        """Yield each text delta and assemble the final response in one pass.

        Solves the "two streams to display deltas live AND get the final
        aggregated response" pattern by doing both jobs from a single
        consumption: each ``choice.delta.content`` is yielded as it arrives,
        and once iteration completes :attr:`final_response` holds the
        assembled :class:`ChatCompletionResponse` (with usage, tool_calls,
        reasoning, finish_reason).

        Tool-call and reasoning deltas are accumulated into the final
        response but not yielded — only text deltas are yielded.

        Example::

            async with await client.chat.completions.stream(
                model=model, messages=messages,
            ) as s:
                async for text in s.collect_with_deltas():
                    print(text, end="", flush=True)
                print()
                if s.final_response and s.final_response.usage:
                    print("tokens:", s.final_response.usage.total_tokens)

        .. note::
            Only choice 0 is tracked (same constraint as :meth:`collect`).
            A one-time warning is emitted if multi-choice (``n > 1``) chunks
            are observed.

        :raises ValueError: If the stream completes without a
            ``finish_reason`` on choice 0 (typically indicates the stream
            was interrupted before the final chunk arrived).
        """
        import warnings

        from .types.api.chat import ChatCompletionResponse

        text_parts: list[str] = []
        reasoning_summary_parts: list[str] = []
        reasoning_encrypted_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        model = ""
        stream_id = ""
        created = 0
        usage = None
        finish_reason: str | None = None
        warned_multi_choice = False

        async for chunk in self:
            model = chunk.model or model
            if chunk.id:
                stream_id = chunk.id
            if chunk.created:
                created = chunk.created
            for choice in chunk.choices:
                if choice.index != 0:
                    if not warned_multi_choice:
                        warnings.warn(
                            "ChatStream.collect_with_deltas() only tracks choice 0; "
                            "non-zero indices ignored. Use the raw Stream iterator for n>1.",
                            stacklevel=2,
                        )
                        warned_multi_choice = True
                    continue
                if choice.delta:
                    if choice.delta.content:
                        text_parts.append(choice.delta.content)
                        yield choice.delta.content
                    if choice.delta.reasoning_content:
                        if choice.delta.reasoning_encrypted:
                            reasoning_encrypted_parts.append(choice.delta.reasoning_content)
                        else:
                            reasoning_summary_parts.append(choice.delta.reasoning_content)
                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index or 0
                            if idx not in tool_calls_by_index:
                                tool_calls_by_index[idx] = {
                                    "id": tc.id or "",
                                    "type": tc.type or "function",
                                    "name": "",
                                    "arguments": "",
                                }
                            entry = tool_calls_by_index[idx]
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    entry["name"] += tc.function.name
                                if tc.function.arguments:
                                    entry["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            if chunk.usage:
                usage = chunk.usage

        if finish_reason is None:
            raise ValueError(
                "Stream completed without a finish_reason — likely interrupted "
                "before the final chunk arrived."
            )

        message: dict[str, object] = {
            "role": "assistant",
            "content": "".join(text_parts) if text_parts else None,
            "reasoning_content": _assemble_reasoning(
                reasoning_summary_parts, reasoning_encrypted_parts
            ),
            "tool_calls": [
                {
                    "id": entry["id"],
                    "type": "function",
                    "function": {"name": entry["name"], "arguments": entry["arguments"]},
                }
                for _, entry in sorted(tool_calls_by_index.items())
            ]
            if tool_calls_by_index
            else None,
        }
        # Flag the message the way the non-streaming endpoint does — but only when
        # there is a block, so responses without reasoning gain no stray key.
        if reasoning_encrypted_parts:
            message["reasoning_encrypted"] = True

        payload: dict[str, object] = {
            "id": stream_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage.model_dump()
        self._reasoning_summary = (
            "".join(reasoning_summary_parts) if reasoning_summary_parts else None
        )
        self._final_response = ChatCompletionResponse.model_validate(payload)


class BytesResponse:
    """
    A wrapper for raw byte responses that includes the original aiohttp.ClientResponse object.

    This class provides a way to access the raw byte content of a response while
    also retaining the original response object for accessing headers and other metadata.

    :param content: The raw byte content of the response
    :type content: bytes
    :param response: The original aiohttp.ClientResponse object
    :type response: Any
    """

    def __init__(self, content: bytes, response: Any):
        self.content = content
        self.response = response
