"""An API key must never be reachable from an exception the SDK raises or logs.

Every aiohttp ``ClientResponseError``, ``ContentTypeError`` and ``TooManyRedirects``
keeps a ``RequestInfo`` whose headers hold the ``Authorization`` the request was sent
with, and its ``repr`` prints them. These tests send real requests over loopback,
through a session the SDK owns and through one the caller supplies, and walk what
each failure leaves behind: ``str``, ``repr``, ``args``, ``original_error``,
``validation_error``, the ``__cause__`` / ``__context__`` chain and every log record.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import logging
import socket
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import venice_ai
from venice_ai import VeniceClient
from venice_ai.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseProcessingError,
    APIResponseValidationError,
    VeniceError,
)
from venice_ai.middleware.retry import RetryOptions, create_retry_middleware
from venice_ai.types.api.video import VideoQuoteResponse
from venice_ai.utils.errors import map_aiohttp_error

API_KEY = "sk-SENTINELAPIKEY"
FAST_RETRIES = RetryOptions(max_attempts=1, base_delay=0.001, max_delay=0.002)

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]
Call = Callable[[VeniceClient], Awaitable[Any]]


@asynccontextmanager
async def _server(handler: Handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _server_answering_garbage() -> AsyncIterator[str]:
    """Answers every request with bytes that are not HTTP."""

    async def answer(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"this is not http\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


@asynccontextmanager
async def _clients(api: str, kind: str) -> AsyncIterator[VeniceClient]:
    """A client on a session the SDK builds, or on one the caller supplies."""
    base_url = f"{api}/api/v1"
    if kind == "sdk-session":
        async with VeniceClient(
            api_key=API_KEY, base_url=base_url, retry_options=FAST_RETRIES
        ) as client:
            yield client
        return
    async with (
        aiohttp.ClientSession() as session,
        VeniceClient(api_key=API_KEY, base_url=base_url, http_client=session) as client,
    ):
        yield client


def _chain(error: BaseException | None) -> Iterator[BaseException]:
    seen: set[int] = set()
    stack = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack += [current.__cause__, current.__context__]
        stack += getattr(current, "exceptions", ())
        for name in ("original_error", "validation_error"):
            kept = getattr(current, name, None)
            if isinstance(kept, BaseException):
                stack.append(kept)


def _what_an_exception_shows(error: BaseException) -> Iterator[str]:
    yield f"{error!s} {error!r} {error.args!r}"
    yield repr(getattr(error, "response_data", None))
    yield repr(getattr(error, "response", None))
    yield repr(getattr(error, "request_info", None))
    for response in getattr(error, "history", ()):
        yield f"{response!r} {response.request_info!r}"


def _assert_key_is_absent(error: BaseException) -> None:
    for link in _chain(error):
        text = " ".join(_what_an_exception_shows(link))
        assert API_KEY not in text, f"{type(link).__name__} carries the API key: {text}"


def _assert_logs_are_clean(caplog: pytest.LogCaptureFixture) -> None:
    formatter = logging.Formatter()
    for record in caplog.records:
        text = f"{record.__dict__} {formatter.format(record)}"
        assert API_KEY not in text, f"log record carries the API key: {text}"


@pytest.fixture
def all_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


# ---------------------------------------------------------------------------
# What the server answers
# ---------------------------------------------------------------------------


async def _html_200(_request: web.Request) -> web.Response:
    return web.Response(text="<html>a proxy page</html>", content_type="text/html")


async def _html_404(_request: web.Request) -> web.Response:
    return web.Response(status=404, text="<html>not here</html>", content_type="text/html")


async def _html_502(_request: web.Request) -> web.Response:
    return web.Response(status=502, text="<html>bad gateway</html>", content_type="text/html")


async def _redirect_loop(request: web.Request) -> web.Response:
    return web.Response(status=302, headers={"Location": str(request.rel_url)})


async def _dropped_midway(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse()
    response.content_length = 1_000
    await response.prepare(request)
    await response.write(b"short")
    assert request.transport is not None
    request.transport.close()
    return response


FAILURES: dict[str, Handler] = {
    "html-200": _html_200,
    "html-404": _html_404,
    "html-502": _html_502,
    "redirect-loop": _redirect_loop,
    "dropped-midway": _dropped_midway,
}


# ---------------------------------------------------------------------------
# What the SDK was asked to do
# ---------------------------------------------------------------------------


async def _post_typed(client: VeniceClient) -> Any:
    return await client.post("video/quote", json_data={}, cast_to=VideoQuoteResponse)


async def _get_untyped(client: VeniceClient) -> Any:
    return await client.get("thing")


async def _multipart(client: VeniceClient) -> Any:
    return await client.video._request_multipart(
        "POST", "thing", files={"f": ("f.txt", b"x")}, data={"a": "b"}
    )


async def _crypto_batch(client: VeniceClient) -> Any:
    request = {"jsonrpc": "2.0", "id": 1, "method": "m", "params": []}
    return await client.crypto.batch_rpc(network="eth-mainnet", requests=[request])


async def _video_retrieve(client: VeniceClient) -> Any:
    return await client.video.retrieve(model="m", queue_id="q")


async def _music_retrieve(client: VeniceClient) -> Any:
    return await client.music.retrieve(model="m", queue_id="q")


async def _voice_changer_retrieve(client: VeniceClient) -> Any:
    return await client.voice_changer.retrieve(model="m", queue_id="q")


async def _billing_csv(client: VeniceClient) -> Any:
    return await client.billing.get_usage_history(format="csv")


async def _speech_stream(client: VeniceClient) -> Any:
    stream = await client.audio.create_speech(model="m", input="hi", voice="v", stream=True)
    return [chunk async for chunk in stream]


async def _chat_stream(client: VeniceClient) -> Any:
    stream = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    return [chunk async for chunk in stream]


CALLS: dict[str, Call] = {
    "post-typed": _post_typed,
    "get-untyped": _get_untyped,
    "multipart": _multipart,
    "crypto-batch": _crypto_batch,
    "video-retrieve": _video_retrieve,
    "music-retrieve": _music_retrieve,
    "voice-changer-retrieve": _voice_changer_retrieve,
    "billing-csv": _billing_csv,
    "speech-stream": _speech_stream,
    "chat-stream": _chat_stream,
}

# Calls that take a 200 with any body as their result, so nothing fails.
RETURNS_BODY = {
    ("billing-csv", "html-200"),
    ("speech-stream", "html-200"),
    ("multipart", "html-200"),
    ("chat-stream", "html-200"),
}


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("call", CALLS)
async def test_failed_call_keeps_the_key_out_of_error_and_logs(
    all_logs: pytest.LogCaptureFixture, call: str, failure: str, kind: str
) -> None:
    error: VeniceError | aiohttp.ClientError | None = None
    async with _server(FAILURES[failure]) as api, _clients(api, kind) as client:
        try:
            await asyncio.wait_for(CALLS[call](client), timeout=10)
        except (VeniceError, aiohttp.ClientError) as e:
            error = e

    assert (error is None) == ((call, failure) in RETURNS_BODY), repr(error)
    if error is not None:
        _assert_key_is_absent(error)
    _assert_logs_are_clean(all_logs)


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
@pytest.mark.parametrize("call", ["post-typed", "get-untyped", "crypto-batch", "chat-stream"])
async def test_answer_that_is_not_http_keeps_the_key_out(
    all_logs: pytest.LogCaptureFixture, call: str, kind: str
) -> None:
    async with _server_answering_garbage() as api, _clients(api, kind) as client:
        with pytest.raises((VeniceError, aiohttp.ClientError)) as excinfo:
            await asyncio.wait_for(CALLS[call](client), timeout=10)

    _assert_key_is_absent(excinfo.value)
    _assert_logs_are_clean(all_logs)


# ---------------------------------------------------------------------------
# The exception types callers rely on, with what they keep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
async def test_html_on_a_typed_call_is_a_validation_error_holding_the_aiohttp_error(
    kind: str,
) -> None:
    async with _server(_html_200) as api, _clients(api, kind) as client:
        with pytest.raises(APIResponseValidationError) as excinfo:
            await _post_typed(client)

    original = excinfo.value.validation_error
    assert isinstance(original, aiohttp.ContentTypeError)
    assert excinfo.value.__cause__ is original
    assert original.status == 200
    assert original.message
    assert URL(str(original.request_info.real_url)).path == "/api/v1/video/quote"
    assert len(original.request_info.headers) == 0
    _assert_key_is_absent(excinfo.value)


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
async def test_html_on_an_untyped_call_is_a_processing_error_holding_the_aiohttp_error(
    kind: str,
) -> None:
    async with _server(_html_200) as api, _clients(api, kind) as client:
        with pytest.raises(APIResponseProcessingError) as excinfo:
            await _get_untyped(client)

    assert isinstance(excinfo.value.original_error, aiohttp.ContentTypeError)
    assert excinfo.value.__cause__ is excinfo.value.original_error
    assert excinfo.value.original_error.status == 200
    _assert_key_is_absent(excinfo.value)


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
async def test_redirect_loop_is_a_connection_error_holding_the_aiohttp_error(kind: str) -> None:
    async with _server(_redirect_loop) as api, _clients(api, kind) as client:
        with pytest.raises(APIConnectionError) as excinfo:
            await _get_untyped(client)

    assert isinstance(excinfo.value.original_error, aiohttp.TooManyRedirects)
    assert excinfo.value.original_error_type is aiohttp.TooManyRedirects
    assert excinfo.value.__cause__ is excinfo.value.original_error
    assert excinfo.value.original_error.history == ()
    _assert_key_is_absent(excinfo.value)


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
async def test_crypto_batch_on_html_raises_a_content_type_error_without_the_key(kind: str) -> None:
    async with _server(_html_200) as api, _clients(api, kind) as client:
        with pytest.raises(aiohttp.ContentTypeError) as excinfo:
            await _crypto_batch(client)

    _assert_key_is_absent(excinfo.value)


@pytest.mark.parametrize("failure", ["html-404", "html-502"])
async def test_status_error_keeps_its_response_and_the_sdk_session_keeps_it_key_free(
    failure: str,
) -> None:
    async with _server(FAILURES[failure]) as api, _clients(api, "sdk-session") as client:
        with pytest.raises(APIError) as excinfo:
            await _get_untyped(client)

    response = excinfo.value.response
    assert isinstance(response, aiohttp.ClientResponse)
    assert API_KEY not in repr(response.request_info)
    assert len(response.request_info.headers) == 0


async def test_response_handed_to_the_caller_holds_no_key() -> None:
    async with _server(_html_200) as api, _clients(api, "sdk-session") as client:
        response = await client._request("GET", "thing", raw_response=True)
        try:
            assert isinstance(response, aiohttp.ClientResponse)
            assert API_KEY not in repr(response.request_info)
            with pytest.raises(aiohttp.ContentTypeError) as excinfo:
                await response.json()
            assert API_KEY not in repr(excinfo.value)
        finally:
            response.release()


# ---------------------------------------------------------------------------
# Retry hooks
# ---------------------------------------------------------------------------


async def test_retry_hook_on_an_api_request_never_gets_the_key() -> None:
    seen: list[Exception | None] = []
    options = RetryOptions(
        max_attempts=1,
        base_delay=0.001,
        max_delay=0.002,
        retry_exceptions=(aiohttp.ClientResponseError,),
        on_retry=lambda _attempt, _delay, error: seen.append(error),
    )
    async with _server_answering_garbage() as api:
        middleware = create_retry_middleware(options, api_origin=URL(api))
        async with aiohttp.ClientSession(middlewares=[middleware]) as session:
            with pytest.raises(aiohttp.ClientResponseError) as excinfo:
                await session.get(f"{api}/x", headers={"Authorization": f"Bearer {API_KEY}"})

    assert len(seen) == 1
    assert isinstance(seen[0], Exception)
    _assert_key_is_absent(seen[0])
    _assert_key_is_absent(excinfo.value)


# ---------------------------------------------------------------------------
# forget_request_headers
# ---------------------------------------------------------------------------


def _forget(error: BaseException) -> None:
    from venice_ai.utils.errors import forget_request_headers

    forget_request_headers(error)


def _error_holding_the_key(**kwargs: Any) -> aiohttp.ClientResponseError:
    headers = CIMultiDictProxy(CIMultiDict({"Authorization": f"Bearer {API_KEY}"}))
    url = URL("https://api.example/x")
    info = aiohttp.RequestInfo(url, "GET", headers, url)
    return aiohttp.ClientResponseError(info, (), status=400, message="Bad", **kwargs)


def test_forget_request_headers_keeps_everything_but_the_headers() -> None:
    error = _error_holding_the_key(headers=CIMultiDict({"Retry-After": "7"}))

    _forget(error)

    assert error.status == 400
    assert error.message == "Bad"
    assert error.headers is not None and error.headers["Retry-After"] == "7"
    assert error.request_info.method == "GET"
    assert str(error.request_info.real_url) == "https://api.example/x"
    assert error.args == (error.request_info, ())
    assert API_KEY not in f"{error!r} {error.args!r}"


def test_forget_request_headers_walks_the_chain() -> None:
    inner = _error_holding_the_key()
    middle = ValueError("wrapping")
    middle.__context__ = inner
    outer = VeniceError("outermost")
    outer.__cause__ = middle

    _forget(outer)

    assert API_KEY not in repr(inner)


@pytest.mark.parametrize("request_info", [None, "not request info"])
def test_forget_request_headers_leaves_other_request_info_alone(request_info: Any) -> None:
    error = aiohttp.ClientResponseError(request_info, ())

    _forget(error)

    assert error.request_info is request_info


def test_map_aiohttp_error_forgets_the_headers_of_what_it_keeps() -> None:
    error = _error_holding_the_key()

    mapped = map_aiohttp_error(error)

    assert isinstance(mapped, APIConnectionError)
    assert mapped.original_error is error
    _assert_key_is_absent(mapped)


def test_every_json_parse_in_the_sdk_goes_through_parse_json() -> None:
    """A bare ``response.json()`` on a caller's session raises an error holding the key."""
    package = Path(venice_ai.__file__).parent
    # errors.py is the helper; the RPC session never holds the key; the limiter discards
    # the error.
    allowed = {"utils/errors.py", "auth/x402_solana.py", "rate_limiting/simple.py"}
    found = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "await " in path.read_text() and _awaits_json_method(path)
    ]
    assert found and set(found) <= allowed, found


def _awaits_json_method(path: Path) -> bool:
    return any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "json"
        for node in ast.walk(ast.parse(path.read_text()))
    )


# ---------------------------------------------------------------------------
# DEBUG request-header logging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "authorization",
        "AUTHORIZATION",
        "x-api-key",
        "X-API-KEY",
        "x_api_key",
        "api-key",
        "apikey",
        "proxy-authorization",
        "Proxy-Authorization",
        "cookie",
        "Cookie",
        "x-sign-in-with-x",
        "x-402-payment",
        "x-auth-token",
        "x-tenant-secret",
    ],
)
async def test_debug_log_redacts_sensitive_header_names_in_any_case(
    all_logs: pytest.LogCaptureFixture, name: str
) -> None:
    async def ok(_request: web.Request) -> web.Response:
        return web.json_response({})

    async with (
        _server(ok) as api,
        VeniceClient(api_key="sk-other", base_url=f"{api}/api/v1") as client,
    ):
        await client.get("e", headers={name: API_KEY})

    assert any("Request headers" in record.getMessage() for record in all_logs.records)
    _assert_logs_are_clean(all_logs)


async def test_debug_log_still_shows_other_headers(all_logs: pytest.LogCaptureFixture) -> None:
    async def ok(_request: web.Request) -> web.Response:
        return web.json_response({})

    async with (
        _server(ok) as api,
        VeniceClient(api_key=API_KEY, base_url=f"{api}/api/v1") as client,
    ):
        await client.get("e", headers={"X-Request-Tag": "visible-tag"})

    assert any("visible-tag" in record.getMessage() for record in all_logs.records)
    _assert_logs_are_clean(all_logs)


# ---------------------------------------------------------------------------
# A proxy's password
# ---------------------------------------------------------------------------

PROXY_PASSWORD = "SENTINELPROXYPASSWORD"
PROXY_NEEDLES = (PROXY_PASSWORD, base64.b64encode(f"user:{PROXY_PASSWORD}".encode()).decode())
NO_RETRIES = RetryOptions(max_attempts=0)


def _assert_proxy_password_is_absent(error: BaseException) -> None:
    links = list(_chain(error))
    assert len(links) > 1, "the aiohttp error should stay chained"
    for link in links:
        text = " ".join(_what_an_exception_shows(link)) + "".join(traceback.format_exception(link))
        for needle in PROXY_NEEDLES:
            assert needle not in text, f"{type(link).__name__} carries it: {text}"


def _assert_no_proxy_password_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    formatter = logging.Formatter()
    for record in caplog.records:
        text = f"{record.__dict__} {formatter.format(record)}"
        for needle in PROXY_NEEDLES:
            assert needle not in text, f"log record carries it: {text}"


@asynccontextmanager
async def _proxy_answering_407() -> AsyncIterator[str]:
    seen: list[bytes] = []

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://user:{PROXY_PASSWORD}@127.0.0.1:{port}"
    finally:
        assert seen, "the proxy was never reached"
        server.close()
        await server.wait_closed()


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
async def test_connect_407_keeps_the_proxy_password_out(
    all_logs: pytest.LogCaptureFixture, kind: str
) -> None:
    hook_errors: list[Exception | None] = []
    retries = RetryOptions(max_attempts=0, on_retry=lambda _i, _d, e: hook_errors.append(e))
    async with _proxy_answering_407() as proxy:
        # https, so aiohttp sends CONNECT, the only request a proxy can refuse this way.
        base_url = "https://127.0.0.1:1/api/v1"
        if kind == "sdk-session":
            async with VeniceClient(
                api_key=API_KEY, base_url=base_url, proxy=proxy, retry_options=retries
            ) as client:
                with pytest.raises(APIConnectionError) as exc_info:
                    await client.get("e")
        else:
            async with (
                aiohttp.ClientSession(proxy=proxy) as session,
                VeniceClient(api_key=API_KEY, base_url=base_url, http_client=session) as client,
            ):
                with pytest.raises(APIConnectionError) as exc_info:
                    await client.get("e")

    assert isinstance(exc_info.value.__cause__, aiohttp.ClientHttpProxyError)
    assert exc_info.value.__cause__.status == 407
    _assert_proxy_password_is_absent(exc_info.value)
    _assert_no_proxy_password_in_logs(all_logs)


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
async def test_unreachable_proxy_keeps_the_proxy_password_out(
    all_logs: pytest.LogCaptureFixture, kind: str
) -> None:
    proxy = f"http://user:{PROXY_PASSWORD}@127.0.0.1:{_unused_port()}"
    base_url = "http://127.0.0.1:1/api/v1"
    if kind == "sdk-session":
        async with VeniceClient(
            api_key=API_KEY, base_url=base_url, proxy=proxy, retry_options=NO_RETRIES
        ) as client:
            with pytest.raises(APIConnectionError) as exc_info:
                await client.get("e")
    else:
        async with (
            aiohttp.ClientSession(proxy=proxy) as session,
            VeniceClient(api_key=API_KEY, base_url=base_url, http_client=session) as client,
        ):
            with pytest.raises(APIConnectionError) as exc_info:
                await client.get("e")

    assert isinstance(exc_info.value.__cause__, aiohttp.ClientProxyConnectionError)
    _assert_proxy_password_is_absent(exc_info.value)
    _assert_no_proxy_password_in_logs(all_logs)


async def test_proxy_password_stays_out_of_an_on_retry_hook(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    hook_errors: list[Exception | None] = []
    retries = RetryOptions(
        max_attempts=1, base_delay=0.001, on_retry=lambda _i, _d, e: hook_errors.append(e)
    )
    proxy = f"http://user:{PROXY_PASSWORD}@127.0.0.1:{_unused_port()}"
    async with VeniceClient(
        api_key=API_KEY, base_url="http://127.0.0.1:1/api/v1", proxy=proxy, retry_options=retries
    ) as client:
        with pytest.raises(APIConnectionError):
            await client.get("e")

    assert hook_errors and hook_errors[0] is not None
    for error in hook_errors:
        assert error is not None
        text = f"{error!s} {error!r} {error.args!r}"
        for needle in PROXY_NEEDLES:
            assert needle not in text
    _assert_no_proxy_password_in_logs(all_logs)
