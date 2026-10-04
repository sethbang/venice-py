"""A response body that stalls surfaces as ``APITimeoutError`` on every read path.

A local server answers each request with a status line, headers and the first
bytes of a body, then stalls. Each case calls one SDK method whose body is read
after the status arrived, through a different code path, and expects the
client timeout to surface as :class:`~venice_ai.exceptions.APITimeoutError`
(never a bare :class:`TimeoutError`, which ``APITimeoutError`` does not
subclass).
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.exceptions import APIResponseProcessingError, APITimeoutError, BillingTimeoutError
from venice_ai.types.enums import BillingFormatEnum

STALL_SECONDS = 5.0
CLIENT_TIMEOUT = 0.3

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_B64 = base64.b64encode(PNG).decode()

# Content-Type the stalled body claims, per endpoint path.
CONTENT_TYPES = {
    "/api/v1/image/generate": "image/png",
    "/api/v1/image/edit": "image/png",
    "/api/v1/image/multi-edit": "image/png",
    "/api/v1/image/background-remove": "image/png",
    "/api/v1/image/upscale": "image/png",
    "/api/v1/audio/speech": "audio/mpeg",
    "/api/v1/audio/retrieve": "audio/flac",
    "/api/v1/video/retrieve": "video/mp4",
    "/api/v1/video/transcriptions": "text/plain",
    "/api/v1/audio/voice-changer/retrieve": "audio/mpeg",
    "/api/v1/crypto/rpc/ethereum-mainnet": "application/json",
    "/api/v1/billing/usage-history": "text/csv",
    "/api/v1/chat/completions": "application/json",
    "/asset.bin": "application/octet-stream",
}


async def _stalling_handler(request: web.Request) -> web.StreamResponse:
    await request.read()
    content_type = CONTENT_TYPES.get(request.path, "application/json")
    response = web.StreamResponse(status=200, headers={"Content-Type": content_type})
    response.content_length = 10_000
    await response.prepare(request)
    await response.write(b"[" if "json" in content_type else b"\x00" * 16)
    await asyncio.sleep(STALL_SECONDS)
    return response


async def _json_status_stall(request: web.Request) -> web.StreamResponse:
    """A JSON status body (music, video, voice changer retrieve) that stalls."""
    await request.read()
    response = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
    response.content_length = 10_000
    await response.prepare(request)
    await response.write(b'{"status": "PROC')
    await asyncio.sleep(STALL_SECONDS)
    return response


async def _sse_stall(request: web.Request) -> web.StreamResponse:
    """A chat stream that sends one chunk, then stalls."""
    await request.read()
    response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    await response.write(
        b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
        b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    )
    await asyncio.sleep(STALL_SECONDS)
    return response


@asynccontextmanager
async def _server() -> AsyncIterator[str]:
    """Serve the stalling handlers on the running loop and yield the base URL."""
    app = web.Application()
    app.router.add_route("*", "/json-status/{tail:.*}", _json_status_stall)
    app.router.add_route("*", "/sse/{tail:.*}", _sse_stall)
    app.router.add_route("*", "/{tail:.*}", _stalling_handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _client(base: str) -> VeniceClient:
    return VeniceClient(api_key="sk-test", base_url=f"{base}/api/v1", timeout=CLIENT_TIMEOUT)


async def _batch_rpc(client: VeniceClient) -> Any:
    return await client.crypto.batch_rpc(
        network="ethereum-mainnet",
        requests=[{"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber"}],
    )


Call = Callable[[VeniceClient, str], Awaitable[Any]]

BINARY_CASES: dict[str, Call] = {
    "image.create(return_binary=True)": lambda c, _: c.image.create(
        model="m", prompt="p", return_binary=True
    ),
    "image.edit": lambda c, _: c.image.edit(prompt="p", image=PNG, model="m"),
    "image.multi_edit": lambda c, _: c.image.multi_edit(prompt="p", image=PNG, model="m"),
    "image.background_remove": lambda c, _: c.image.background_remove(
        image_url="https://example.com/a.png"
    ),
    "image.upscale": lambda c, _: c.image.upscale(image=PNG),
    "audio.create_speech": lambda c, _: c.audio.create_speech(input="hi", model="m", voice="v"),
    "music.retrieve (inline audio)": lambda c, _: c.music.retrieve(model="m", queue_id="q"),
    "video.retrieve (inline video)": lambda c, _: c.video.retrieve(model="m", queue_id="q"),
    "video.transcribe(text)": lambda c, _: c.video.transcribe(
        "https://example.com/v", response_format="text"
    ),
    "voice_changer.retrieve (inline audio)": lambda c, _: c.voice_changer.retrieve(
        model="m", queue_id="q"
    ),
    "crypto.batch_rpc": lambda c, _: _batch_rpc(c),
    "fetch_external": lambda c, base: c.fetch_external(f"{base}/asset.bin"),
    "_request JSON body": lambda c, _: c.get("models"),
}


BODY_TIMEOUT = "Timed out reading the response body"


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


@pytest.mark.parametrize("call", BINARY_CASES.values(), ids=BINARY_CASES.keys())
async def test_stalled_body_raises_api_timeout_error(call: Call) -> None:
    async with _server() as base, _client(base) as client:
        with pytest.raises(APITimeoutError, match=BODY_TIMEOUT) as excinfo:
            await asyncio.wait_for(call(client, base), timeout=STALL_SECONDS / 2)
    assert isinstance(excinfo.value.original_error, TimeoutError)


async def test_stalled_csv_usage_history_is_a_billing_timeout() -> None:
    async with _server() as base, _client(base) as client:
        with pytest.raises(BillingTimeoutError) as excinfo:
            await asyncio.wait_for(
                client.billing.get_usage_history(format=BillingFormatEnum.CSV),
                timeout=STALL_SECONDS / 2,
            )
    assert isinstance(excinfo.value.original_error, TimeoutError)


@pytest.mark.parametrize("resource", ["music", "video", "voice_changer"])
async def test_stalled_json_status_raises_api_timeout_error(resource: str) -> None:
    async with _server() as base, _client(f"{base}/json-status") as client:
        with pytest.raises(APITimeoutError, match=BODY_TIMEOUT):
            await asyncio.wait_for(
                getattr(client, resource).retrieve(model="m", queue_id="q"),
                timeout=STALL_SECONDS / 2,
            )


async def test_stalled_audio_stream_raises_api_timeout_error() -> None:
    async with _server() as base, _client(base) as client:
        stream = await client.audio.create_speech(input="hi", model="m", voice="v", stream=True)
        with pytest.raises(APITimeoutError, match="Stream timed out before it finished"):
            await asyncio.wait_for(_drain(stream), timeout=STALL_SECONDS / 2)


async def test_stalled_chat_stream_raises_api_timeout_error() -> None:
    async with _server() as base, _client(f"{base}/sse") as client:
        stream = await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(APITimeoutError, match="Stream timed out before it finished"):
            await asyncio.wait_for(_drain(stream), timeout=STALL_SECONDS / 2)


async def test_a_body_that_is_not_json_is_not_a_connection_error() -> None:
    """A non-JSON body is a processing error, never a transport error."""

    async def html(request: web.Request) -> web.Response:
        return web.Response(text="<html>proxy error</html>", content_type="text/html")

    app = web.Application()
    app.router.add_get("/api/v1/models", html)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        async with VeniceClient(api_key="k", base_url=f"http://127.0.0.1:{port}/api/v1") as c:
            with pytest.raises(APIResponseProcessingError):
                await c.get("models")
    finally:
        await runner.cleanup()
