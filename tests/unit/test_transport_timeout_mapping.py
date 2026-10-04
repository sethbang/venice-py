"""A request timeout surfaces as ``APITimeoutError`` on every request path.

The SDK sends a request either directly or through a rate limiter, and reads
its body either at once or as a stream. Each combination is exercised against a
local server that stalls, so no request leaves the machine. A timeout raised by
a rate limiter itself, before any request is sent, must stay distinguishable
from one raised by the HTTP request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web

from venice_ai import APITimeoutError, UserMessage, VeniceClient
from venice_ai.exceptions import InternalServerError
from venice_ai.rate_limiting import SimpleRateLimiter

_TIMEOUT = 0.3
_STALL = 5.0


async def _stall_before_headers(request: web.Request) -> web.StreamResponse:
    await asyncio.sleep(_STALL)
    return web.json_response({})


async def _stall_in_body(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        headers={"Content-Type": "application/json", "Content-Length": "100"}
    )
    await response.prepare(request)
    await response.write(b'{"id":')
    await asyncio.sleep(_STALL)
    return response


async def _error_status_stall_in_body(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=500, headers={"Content-Type": "application/json", "Content-Length": "100"}
    )
    await response.prepare(request)
    await response.write(b'{"error":')
    await asyncio.sleep(_STALL)
    return response


@asynccontextmanager
async def _server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", handler)
    app.router.add_post("/api/v1/responses", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}/api/v1"
    finally:
        await runner.cleanup()


async def _chat(client: VeniceClient, *, stream: bool) -> None:
    messages = [UserMessage(content="hi")]
    if stream:
        chunks = await client.chat.completions.create(model="m", messages=messages, stream=True)
        async for _ in chunks:
            pass
    else:
        await client.chat.completions.create(model="m", messages=messages)


def _assert_request_timeout(exc: BaseException, prefix: str = "Request timed out.") -> None:
    assert type(exc) is APITimeoutError
    assert str(exc).startswith(prefix)
    assert isinstance(exc.original_error, TimeoutError)
    assert exc.__cause__ is exc.original_error


class _FutureLimiter:
    """A scheduler that runs the request in its own task and hands back a future.

    Like a scheduler with a deadline of its own, it turns any ``TimeoutError``
    the request raises into its own error. A request timeout must therefore
    already be an ``APITimeoutError`` by the time the limiter sees it.
    """

    def __init__(self) -> None:
        self.saw_bare_timeout = False

    def is_running(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def submit_request(
        self,
        metadata: Any,
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Any = None,
    ) -> Any:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        async def run() -> None:
            try:
                future.set_result(await request_func())
            except TimeoutError:
                self.saw_bare_timeout = True
                future.set_exception(RuntimeError("limiter deadline"))
            except Exception as e:
                future.set_exception(e)

        asyncio.get_running_loop().create_task(run())
        return SimpleNamespace(request=SimpleNamespace(future=future))


class _LimiterWithOwnDeadline(_FutureLimiter):
    """A limiter whose own wait for capacity times out before sending."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = False

    async def submit_request(
        self,
        metadata: Any,
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Any = None,
    ) -> Any:
        async with asyncio.timeout(0.01):
            await asyncio.sleep(1)
        self.sent = True
        return await request_func()


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("limited", [False, True], ids=["direct", "simple-limiter"])
async def test_timeout_before_headers_is_api_timeout(limited: bool, stream: bool) -> None:
    async with _server(_stall_before_headers) as base_url:
        limiter = SimpleRateLimiter() if limited else None
        async with VeniceClient(
            api_key="k", base_url=base_url, timeout=_TIMEOUT, rate_limiter=limiter
        ) as client:
            with pytest.raises(APITimeoutError) as info:
                await _chat(client, stream=stream)
    _assert_request_timeout(info.value)


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("limited", [False, True], ids=["direct", "simple-limiter"])
async def test_responses_timeout_is_api_timeout(limited: bool, stream: bool) -> None:
    async with _server(_stall_before_headers) as base_url:
        limiter = SimpleRateLimiter() if limited else None
        async with VeniceClient(
            api_key="k", base_url=base_url, timeout=_TIMEOUT, rate_limiter=limiter
        ) as client:
            with pytest.raises(APITimeoutError) as info:
                if stream:
                    events = await client.responses.create(model="m", input="hi", stream=True)
                    async for _ in events:
                        pass
                else:
                    await client.responses.create(model="m", input="hi")
    _assert_request_timeout(info.value)


@pytest.mark.parametrize("limited", [False, True], ids=["direct", "simple-limiter"])
async def test_timeout_while_reading_body_is_api_timeout(limited: bool) -> None:
    async with _server(_stall_in_body) as base_url:
        limiter = SimpleRateLimiter() if limited else None
        async with VeniceClient(
            api_key="k", base_url=base_url, timeout=_TIMEOUT, rate_limiter=limiter
        ) as client:
            with pytest.raises(APITimeoutError) as info:
                await _chat(client, stream=False)
    # The status line had arrived, so the message says the request was processed.
    _assert_request_timeout(info.value, prefix="Timed out reading the response body.")


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
async def test_scheduler_never_sees_a_bare_request_timeout(stream: bool) -> None:
    limiter = _FutureLimiter()
    async with (
        _server(_stall_before_headers) as base_url,
        VeniceClient(
            api_key="k", base_url=base_url, timeout=_TIMEOUT, rate_limiter=limiter
        ) as client,
    ):
        with pytest.raises(APITimeoutError) as info:
            await _chat(client, stream=stream)
    _assert_request_timeout(info.value)
    assert not limiter.saw_bare_timeout


async def test_limiter_timeout_is_not_reported_as_request_timeout() -> None:
    limiter = _LimiterWithOwnDeadline()
    async with (
        _server(_stall_before_headers) as base_url,
        VeniceClient(
            api_key="k", base_url=base_url, timeout=_TIMEOUT, rate_limiter=limiter
        ) as client,
    ):
        with pytest.raises(TimeoutError) as info:
            await _chat(client, stream=False)
    assert not isinstance(info.value, APITimeoutError)
    assert not limiter.sent


async def test_error_status_with_stalled_body_raises_the_status_error() -> None:
    async with (
        _server(_error_status_stall_in_body) as base_url,
        VeniceClient(api_key="k", base_url=base_url, timeout=_TIMEOUT, max_retries=0) as client,
    ):
        with pytest.raises(InternalServerError) as info:
            await _chat(client, stream=False)
    assert info.value.status_code == 500
    assert "Failed to read response body: TimeoutError" in str(info.value.body)
