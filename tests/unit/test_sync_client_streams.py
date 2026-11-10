"""The sync stream proxy offers what the async stream it wraps offers."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import re
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

import venice_ai.resources
from venice_ai import SyncVeniceClient
from venice_ai.exceptions import APIError, StreamConsumedError
from venice_ai.streaming import ChatStream, Stream
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.streaming import ChatCompletionChunk

UNREACHABLE = "http://127.0.0.1:9/api/v1"


def _chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> ChatCompletionChunk:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "reasoning_content": reasoning_content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


class _Source:
    """An async chunk source that records where it ran and whether it was closed."""

    def __init__(self) -> None:
        self.threads: list[threading.Thread] = []
        self.closed_on: list[threading.Thread] = []
        self.produced = 0

    async def chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        try:
            for chunk in (
                _chunk(reasoning_content="thinking "),
                _chunk(content="Hel"),
                _chunk(content="lo"),
                _chunk(
                    finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                ),
            ):
                self.threads.append(threading.current_thread())
                self.produced += 1
                yield chunk
        finally:
            self.closed_on.append(threading.current_thread())


@pytest.fixture
def client() -> Iterator[SyncVeniceClient]:
    sync_client = SyncVeniceClient(api_key="fake-key", base_url=UNREACHABLE)
    yield sync_client
    sync_client.close()


def _open_chat_stream(client: SyncVeniceClient, source: _Source) -> Any:
    """Open a real ``ChatStream`` through ``client.chat.completions.stream``."""
    stream = ChatStream(source.chunks(), client=client._async_client)
    client._async_client.chat.completions.stream = AsyncMock(return_value=stream)  # type: ignore[method-assign]
    return client.chat.completions.stream(model="stub-model", messages=[])


def test_collect_blocks_and_returns_the_assembled_response(client: SyncVeniceClient) -> None:
    source = _Source()

    response = _open_chat_stream(client, source).collect()

    assert isinstance(response, ChatCompletionResponse)
    assert response.choices[0].message.content == "Hello"
    assert response.usage is not None
    assert response.usage.total_tokens == 3
    assert set(source.threads) == {client._thread}


def test_final_response_and_reasoning_summary_follow_collect(client: SyncVeniceClient) -> None:
    stream = _open_chat_stream(client, _Source())
    assert stream.final_response is None
    assert stream.reasoning_summary is None

    response = stream.collect()

    assert stream.final_response is response
    assert stream.reasoning_summary == "thinking "


def test_text_deltas_iterate_lazily_on_the_loop_thread(client: SyncVeniceClient) -> None:
    source = _Source()
    stream = _open_chat_stream(client, source)

    deltas = stream.text_deltas()
    assert not inspect.isasyncgen(deltas)
    assert source.produced == 0

    assert next(deltas) == "Hel"
    assert source.produced == 2

    assert list(deltas) == ["lo"]
    assert stream.final_response is None
    assert set(source.threads) == {client._thread}


def test_collect_with_deltas_yields_text_then_fills_final_response(
    client: SyncVeniceClient,
) -> None:
    stream = _open_chat_stream(client, _Source())

    assert list(stream.collect_with_deltas()) == ["Hel", "lo"]

    assert stream.final_response is not None
    assert stream.final_response.choices[0].message.content == "Hello"
    assert stream.reasoning_summary == "thinking "


def test_context_manager_and_iteration_still_work(client: SyncVeniceClient) -> None:
    source = _Source()

    with _open_chat_stream(client, source) as stream:
        chunks = list(stream)

    assert len(chunks) == 4
    assert source.closed_on == [client._thread]


def test_with_block_closes_a_stream_left_after_a_text_delta_break(
    client: SyncVeniceClient,
) -> None:
    source = _Source()

    with _open_chat_stream(client, source) as stream:
        for _text in stream.text_deltas():
            break
        assert source.closed_on == []

    assert source.closed_on == [client._thread]


def test_get_iterator_gives_a_blocking_iterator_and_consumes_the_stream(
    client: SyncVeniceClient,
) -> None:
    stream = _open_chat_stream(client, _Source())

    assert len(list(stream.get_iterator())) == 4

    with pytest.raises(StreamConsumedError):
        iter(stream)


def test_iterating_a_consumed_stream_raises_like_the_async_one(client: SyncVeniceClient) -> None:
    stream = _open_chat_stream(client, _Source())
    assert len(list(stream)) == 4

    with pytest.raises(StreamConsumedError):
        list(stream)


def test_collect_after_consuming_raises_like_the_async_one(client: SyncVeniceClient) -> None:
    stream = _open_chat_stream(client, _Source())
    list(stream)

    with pytest.raises(StreamConsumedError):
        stream.collect()


def test_plain_stream_has_no_chat_helpers(client: SyncVeniceClient) -> None:
    """``create(stream=True)`` returns a plain ``Stream``, as it does when async."""
    source = _Source()
    plain = Stream(source.chunks(), client=client._async_client)
    client._async_client.chat.completions.create = AsyncMock(return_value=plain)  # type: ignore[method-assign]

    stream = client.chat.completions.create(model="stub-model", messages=[], stream=True)

    assert not hasattr(stream, "collect")
    with pytest.raises(AttributeError):
        stream.final_response  # noqa: B018
    assert len(list(stream)) == 4


def test_every_public_chat_stream_member_is_usable_without_await(
    client: SyncVeniceClient,
) -> None:
    """A member added to ``ChatStream`` cannot silently stay async here."""
    proxy = _open_chat_stream(client, _Source())
    public = [name for name in dir(ChatStream) if not name.startswith("_")]
    assert {"collect", "text_deltas", "final_response"} <= set(public)

    for name in public:
        member = getattr(proxy, name)
        assert not inspect.iscoroutinefunction(member), name
        assert not inspect.isasyncgenfunction(member), name
    proxy.close()


# ---------------------------------------------------------------------------
# Every method that hands back an async iterable, over real HTTP on loopback
# ---------------------------------------------------------------------------

AUDIO_CHUNKS = [b"audio-one-", b"audio-two-", b"audio-three"]


def _sse(*payloads: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    return "".join([*lines, "data: [DONE]\n\n"]).encode()


class _StreamServer:
    """A loopback API on a thread of its own, recording what it was asked for."""

    def __init__(self) -> None:
        self.speech_inputs: list[str] = []
        self.dropped_by_client = threading.Event()
        self.base_url = ""

    async def speech(self, request: web.Request) -> web.StreamResponse:
        text = (await request.json())["input"]
        self.speech_inputs.append(text)
        if text == "rejected":
            return web.json_response({"error": "nope"}, status=400)
        response = web.StreamResponse(headers={"Content-Type": "audio/mpeg"})
        await response.prepare(request)
        if text == "endless":
            try:
                while True:
                    await response.write(b"x" * 8192)
                    await asyncio.sleep(0.01)
            except (ConnectionResetError, asyncio.CancelledError):
                self.dropped_by_client.set()
                raise
        for chunk in AUDIO_CHUNKS:
            await response.write(chunk)
        await response.write_eof()
        return response

    async def chat(self, request: web.Request) -> web.Response:
        chunk = _chunk(content="hi").model_dump(exclude_none=True)
        return web.Response(body=_sse(chunk), content_type="text/event-stream")

    async def responses(self, request: web.Request) -> web.Response:
        event = {"type": "response.output_text.delta", "delta": "hi"}
        return web.Response(body=_sse(event), content_type="text/event-stream")


@pytest.fixture
def stream_server() -> Iterator[_StreamServer]:
    server = _StreamServer()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def start() -> web.AppRunner:
        app = web.Application()
        app.router.add_post("/api/v1/audio/speech", server.speech)
        app.router.add_post("/api/v1/chat/completions", server.chat)
        app.router.add_post("/api/v1/responses", server.responses)
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        server.base_url = f"http://127.0.0.1:{port}/api/v1"
        return runner

    runner = asyncio.run_coroutine_threadsafe(start(), loop).result(10)
    yield server
    asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


@pytest.fixture
def http_client(stream_server: _StreamServer) -> Iterator[SyncVeniceClient]:
    with SyncVeniceClient(api_key="test-key", base_url=stream_server.base_url) as client:
        yield client


SPEECH = {"model": "stub-tts", "voice": "af_heart"}


def test_create_speech_stream_iterates_with_a_plain_for_loop(
    http_client: SyncVeniceClient,
) -> None:
    chunks = http_client.audio.create_speech(input="hello", stream=True, **SPEECH)

    assert isinstance(chunks, Iterator)
    assert b"".join(chunks) == b"".join(AUDIO_CHUNKS)


def test_create_speech_stream_sends_nothing_until_iterated(
    http_client: SyncVeniceClient, stream_server: _StreamServer
) -> None:
    chunks = http_client.audio.create_speech(input="hello", stream=True, **SPEECH)

    assert stream_server.speech_inputs == []
    next(chunks)
    assert stream_server.speech_inputs == ["hello"]
    chunks.close()


def test_create_speech_stream_error_reaches_the_for_loop(
    http_client: SyncVeniceClient,
) -> None:
    chunks = http_client.audio.create_speech(input="rejected", stream=True, **SPEECH)

    with pytest.raises(APIError):
        next(chunks)


def test_leaving_a_speech_stream_early_drops_the_connection(
    http_client: SyncVeniceClient, stream_server: _StreamServer
) -> None:
    for _chunk_bytes in http_client.audio.create_speech(input="endless", stream=True, **SPEECH):
        break

    assert stream_server.dropped_by_client.wait(5)


def test_stream_long_text_single_segment_iterates_with_a_plain_for_loop(
    http_client: SyncVeniceClient,
) -> None:
    chunks = http_client.audio.stream_long_text(input="hello", **SPEECH)

    assert isinstance(chunks, Iterator)
    assert b"".join(chunks) == b"".join(AUDIO_CHUNKS)


def test_stream_long_text_many_segments_iterates_in_input_order(
    http_client: SyncVeniceClient, stream_server: _StreamServer
) -> None:
    chunks = http_client.audio.stream_long_text(
        input="one. two. three.", max_words_per_segment=1, **SPEECH
    )

    assert b"".join(chunks) == b"".join(AUDIO_CHUNKS) * 3
    assert sorted(stream_server.speech_inputs) == ["one.", "three.", "two."]


def test_chat_create_stream_iterates_with_a_plain_for_loop(http_client: SyncVeniceClient) -> None:
    stream = http_client.chat.completions.create(
        model="stub", messages=[{"role": "user", "content": "x"}], stream=True
    )

    assert [chunk.choices[0].delta.content for chunk in stream] == ["hi"]


def test_chat_stream_helper_iterates_with_a_plain_for_loop(http_client: SyncVeniceClient) -> None:
    with http_client.chat.completions.stream(
        model="stub", messages=[{"role": "user", "content": "x"}]
    ) as stream:
        assert list(stream.text_deltas()) == ["hi"]


def test_responses_create_stream_iterates_with_a_plain_for_loop(
    http_client: SyncVeniceClient,
) -> None:
    stream = http_client.responses.create(model="stub", input="x", stream=True)

    assert [event.type for event in stream] == ["response.output_text.delta"]


def test_a_coroutine_returning_an_async_generator_comes_back_blocking(
    client: SyncVeniceClient,
) -> None:
    async def walk() -> AsyncIterator[int]:
        yield 1
        yield 2

    client._async_client.numbers = AsyncMock(return_value=walk())  # type: ignore[attr-defined]

    assert list(client.numbers()) == [1, 2]


def test_a_coroutine_returning_a_custom_async_iterable_comes_back_unchanged(
    client: SyncVeniceClient,
) -> None:
    class Numbers:
        def __aiter__(self) -> AsyncIterator[int]:
            raise AssertionError("not iterated by the client")

        def total(self) -> int:
            return 3

    numbers = Numbers()
    client._async_client.numbers = AsyncMock(return_value=numbers)  # type: ignore[attr-defined]

    assert client.numbers() is numbers


# Public resource methods annotated as returning an async iterable, each driven
# without ``await`` by a test above.
_ASYNC_ITERABLE_METHODS = {
    "Audio.create_speech",
    "Audio.stream_long_text",
    "ChatCompletions.create",
    "Responses.create",
}


def test_every_method_returning_an_async_iterable_is_listed_above() -> None:
    """A new one fails here until a blocking test for it joins the list.

    Read from the source, as annotations cannot be evaluated: a method named
    ``list`` shadows the builtin for the annotations of the methods after it.
    """
    found = set()
    for source in Path(venice_ai.resources.__file__).parent.rglob("*.py"):
        for cls in ast.walk(ast.parse(source.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            for method in cls.body:
                if (
                    isinstance(method, ast.AsyncFunctionDef | ast.FunctionDef)
                    and not method.name.startswith("_")
                    and method.returns is not None
                    and re.search(r"\bAsync(Iterator|Iterable)\b", ast.unparse(method.returns))
                ):
                    found.add(f"{cls.name}.{method.name}")

    assert found == _ASYNC_ITERABLE_METHODS
