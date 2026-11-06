"""A download link grants access on its own, so nothing the SDK writes may contain it.

``VideoJob.download_url`` promises the SDK never logs the link, and music jobs
fetch the same kind of link. These tests hold the SDK to that over real loopback
requests: every log record at DEBUG, including its ``extra`` fields and
traceback, and every exception a failed fetch raises, including its ``repr``, the
attributes that keep the response, and the whole ``__cause__`` / ``__context__``
chain. A hook given to ``RetryOptions.on_retry`` is held to the same standard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web
from pydantic import ValidationError
from yarl import URL

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    PermissionDeniedError,
    RateLimitError,
    VeniceError,
)
from venice_ai.middleware.retry import RetryOptions, create_retry_middleware
from venice_ai.resources.music import MusicJob
from venice_ai.resources.video import VideoJob
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQueueResponse,
)
from venice_ai.types.api.video import VideoCompletedStatus, VideoQueueResponse

IN_PATH = "PATHSECRET"
IN_QUERY = "QUERYSECRET"
IN_FRAGMENT = "FRAGMENTSECRET"
SECRETS = (IN_PATH, IN_QUERY, IN_FRAGMENT)
LINK_TAIL = f"/{IN_PATH}/video.mp4?sig={IN_QUERY}"
API_KEY = "sk-TESTKEYSECRET"
VIDEO_FILE = b"\x00\x00\x00\x18ftypmp42VIDEO"
FAST_RETRIES = RetryOptions(max_attempts=2, base_delay=0.001, max_delay=0.002)

Handler = Callable[[web.Request], Any]


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


def _serve_after_503s(failures: int) -> Handler:
    remaining = [failures]

    async def handler(_request: web.Request) -> web.Response:
        if remaining[0]:
            remaining[0] -= 1
            return web.Response(status=503, reason="Busy")
        return web.Response(body=VIDEO_FILE, content_type="video/mp4")

    return handler


def _client(api: str) -> VeniceClient:
    return VeniceClient(api_key=API_KEY, base_url=f"{api}/api/v1", retry_options=FAST_RETRIES)


def _job(client: VeniceClient, link: str) -> VideoJob:
    return VideoJob(
        client,
        VideoQueueResponse(model="m", queue_id="q", download_url=link),
    )


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


def _what_an_exception_keeps(error: BaseException) -> Iterator[str]:
    """The text of an exception and of every attribute that can quote what it failed on."""
    yield f"{error!s} {error!r}"
    for name in ("original_error", "validation_error", "response_data", "doc"):
        kept = getattr(error, name, None)
        yield repr(kept)
        if isinstance(kept, ValidationError):
            yield f"{kept.errors()} {kept.json()}"
    if isinstance(error, ValidationError):
        yield f"{error.errors()} {error.json()}"
    response = getattr(error, "response", None)
    if response is not None:
        yield repr(response)
        for name in ("url", "real_url", "headers", "history", "request_info", "_body"):
            yield repr(getattr(response, name, None))


def _assert_error_is_clean(error: BaseException, *secrets: str) -> None:
    for link in _chain(error):
        text = " ".join(_what_an_exception_keeps(link))
        for secret in secrets or (*SECRETS, API_KEY):
            assert secret not in text, f"{type(link).__name__} carries {secret}: {text}"


def _assert_logs_are_clean(caplog: pytest.LogCaptureFixture, *secrets: str) -> None:
    formatter = logging.Formatter()
    for record in caplog.records:
        text = f"{record.__dict__} {formatter.format(record)}"
        for secret in secrets or SECRETS:
            assert secret not in text, f"log record carries {secret}: {text}"


@pytest.fixture
def all_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


# ---------------------------------------------------------------------------
# Logging while a download retries
# ---------------------------------------------------------------------------


async def test_download_retry_logs_only_scheme_and_host(
    all_logs: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """The claim on VideoJob.download_url: this SDK never logs the link."""
    async with (
        _server(_serve_after_503s(1)) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        saved = await _job(client, f"{storage}{LINK_TAIL}").download(
            tmp_path / "v.mp4", VideoCompletedStatus(status="COMPLETED")
        )

    assert saved.read_bytes() == VIDEO_FILE
    retrying = [r.getMessage() for r in all_logs.records if "Retrying" in r.getMessage()]
    assert len(retrying) == 1
    assert f"GET {URL(storage).scheme}://127.0.0.1 " in retrying[0]
    _assert_logs_are_clean(all_logs)


async def test_refused_retry_after_wait_logs_only_scheme_and_host(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async def too_long(_request: web.Request) -> web.Response:
        return web.Response(status=503, headers={"Retry-After": "99999"})

    async with (
        _server(too_long) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        with pytest.raises(aiohttp.ClientResponseError):
            await client.fetch_external(f"{storage}{LINK_TAIL}")

    refused = [r.getMessage() for r in all_logs.records if "Not retrying" in r.getMessage()]
    assert len(refused) == 1
    assert "GET http://127.0.0.1:" in refused[0]
    _assert_logs_are_clean(all_logs)


async def test_api_request_retry_logs_the_path_but_no_query(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async with _server(_serve_after_503s(1)) as api, _client(api) as client:
        await client.fetch_external(f"{api}/api/v1/thing?sig={IN_QUERY}#{IN_FRAGMENT}")

    retrying = [r.getMessage() for r in all_logs.records if "Retrying" in r.getMessage()]
    assert len(retrying) == 1
    assert f"GET {api}/api/v1/thing " in retrying[0]
    _assert_logs_are_clean(all_logs, IN_QUERY, IN_FRAGMENT)


def _stub_request(url: str) -> MagicMock:
    request = MagicMock()
    request.method = "GET"
    request.url = URL(url)
    request.headers = {}
    return request


@pytest.mark.parametrize(
    "failure",
    [
        aiohttp.ConnectionTimeoutError(f"Connection timeout to host http://h{LINK_TAIL}"),
        aiohttp.ClientConnectionError(f"failed on http://h{LINK_TAIL}"),
    ],
    ids=["connection-timeout", "connection-error"],
)
async def test_retry_log_leaves_out_the_failure_text_which_can_quote_the_url(
    all_logs: pytest.LogCaptureFixture, failure: Exception
) -> None:
    calls = [failure, None]

    async def handler(_request: Any) -> Any:
        outcome = calls.pop(0)
        if outcome is not None:
            raise outcome
        response = MagicMock()
        response.status = 200
        return response

    middleware = create_retry_middleware(FAST_RETRIES, api_origin=URL("https://api.example"))
    await middleware(_stub_request(f"http://h{LINK_TAIL}#{IN_FRAGMENT}"), handler)

    assert any("Retrying" in r.getMessage() for r in all_logs.records)
    _assert_logs_are_clean(all_logs)


async def test_middleware_built_without_an_api_origin_logs_only_scheme_and_host(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    calls = [aiohttp.ClientConnectionError("boom"), None]

    async def handler(_request: Any) -> Any:
        outcome = calls.pop(0)
        if outcome is not None:
            raise outcome
        response = MagicMock()
        response.status = 200
        return response

    middleware = create_retry_middleware(FAST_RETRIES)
    await middleware(_stub_request(f"https://user:{IN_QUERY}@h.example{LINK_TAIL}"), handler)

    assert any("GET https://h.example " in r.getMessage() for r in all_logs.records)
    _assert_logs_are_clean(all_logs)


# ---------------------------------------------------------------------------
# What a failed fetch raises
# ---------------------------------------------------------------------------


async def test_error_status_raises_client_response_error_without_the_link(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async def gone(_request: web.Request) -> web.Response:
        return web.Response(status=410, reason="Gone", headers={"Retry-After": "7"})

    async with (
        _server(gone) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await client.fetch_external(f"{storage}{LINK_TAIL}")

    error = excinfo.value
    assert error.status == 410
    assert error.message == "Gone"
    assert error.headers is not None and error.headers["Retry-After"] == "7"
    assert str(error.request_info.real_url) == "http://127.0.0.1"
    _assert_error_is_clean(error)
    _assert_logs_are_clean(all_logs)


async def test_error_on_a_redirect_target_drops_the_redirect_history(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        if request.path.startswith(f"/{IN_PATH}/video"):
            return web.Response(status=302, headers={"Location": f"/{IN_PATH}/moved"})
        return web.Response(
            status=404,
            headers={"Location": f"/{IN_PATH}/x", "Content-Location": f"/{IN_PATH}/x"},
        )

    async with (
        _server(handler) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await client.fetch_external(f"{storage}{LINK_TAIL}")

    assert excinfo.value.history == ()
    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


async def test_error_on_the_api_host_does_not_show_the_api_key() -> None:
    async def denied(_request: web.Request) -> web.Response:
        return web.Response(status=403)

    async with _server(denied) as api, _client(api) as client:
        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await client.fetch_external(f"{api}{LINK_TAIL}")

    _assert_error_is_clean(excinfo.value)


async def test_redirect_loop_raises_without_the_link(all_logs: pytest.LogCaptureFixture) -> None:
    async def loop(_request: web.Request) -> web.Response:
        return web.Response(status=302, headers={"Location": LINK_TAIL})

    async with (
        _server(loop) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        with pytest.raises(APIConnectionError) as excinfo:
            await client.fetch_external(f"{storage}{LINK_TAIL}")

    assert excinfo.value.original_error is None
    assert "TooManyRedirects" in str(excinfo.value)
    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


async def test_unreachable_host_raises_without_the_link(all_logs: pytest.LogCaptureFixture) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async with _server(_serve_after_503s(0)) as api, _client(api) as client:
        with pytest.raises(APIConnectionError) as excinfo:
            await client.fetch_external(f"http://127.0.0.1:{port}{LINK_TAIL}")

    assert excinfo.value.original_error is None
    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


async def test_body_dropped_midway_raises_without_the_link() -> None:
    async def dropped(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        response.content_length = 1_000
        await response.prepare(request)
        await response.write(b"short")
        assert request.transport is not None
        request.transport.close()
        return response

    async with (
        _server(dropped) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        with pytest.raises(APIConnectionError) as excinfo:
            await client.fetch_external(f"{storage}{LINK_TAIL}")

    _assert_error_is_clean(excinfo.value)


# ---------------------------------------------------------------------------
# Deleting through the link
# ---------------------------------------------------------------------------


async def test_failed_delete_on_the_link_stays_out_of_error_and_logs(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async def denied(_request: web.Request) -> web.Response:
        return web.Response(status=403)

    async with (
        _server(denied) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        job = _job(client, f"{storage}{LINK_TAIL}")
        with pytest.raises(PermissionDeniedError) as excinfo:
            await job.cancel()

    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


@pytest.mark.parametrize(
    "body",
    [
        f"<Error><Code>SignatureDoesNotMatch</Code><Resource>{LINK_TAIL}</Resource></Error>",
        f'{{"error": "cannot delete {LINK_TAIL}"}}',
    ],
    ids=["storage-xml", "json"],
)
async def test_storage_error_body_quoting_the_link_stays_out_of_error_and_logs(
    all_logs: pytest.LogCaptureFixture, body: str
) -> None:
    """Storage errors often echo the request, and leaving the block logs the error's text."""

    async def denied(_request: web.Request) -> web.Response:
        return web.Response(status=403, text=body)

    async with _server(denied) as storage, _server(_serve_after_503s(0)) as api, _client(api) as c:
        job = _job(c, f"{storage}{LINK_TAIL}")
        job._status = VideoCompletedStatus(status="COMPLETED")
        with pytest.raises(PermissionDeniedError) as excinfo:
            await job.cancel()
        _assert_error_is_clean(excinfo.value)

        async with job:
            pass

    assert any("VideoJob cleanup failed" in r.getMessage() for r in all_logs.records)
    _assert_logs_are_clean(all_logs)


async def test_unreachable_link_on_delete_raises_without_the_link(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async with _server(_serve_after_503s(0)) as api, _client(api) as client:
        job = _job(client, f"http://127.0.0.1:{port}{LINK_TAIL}")
        with pytest.raises(APIConnectionError) as excinfo:
            await job.cancel()

    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


# ---------------------------------------------------------------------------
# The link inside a response the SDK cannot parse
# ---------------------------------------------------------------------------


def _json_answer(body: Any) -> Handler:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response(body)

    return handler


async def test_unparseable_queue_response_does_not_log_the_link(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    link = f"https://storage{LINK_TAIL}"

    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith("/video/queue"):
            return web.json_response({"model": "m", "download_url": link})
        return web.json_response({"object": "list", "data": []})

    async with _server(handler) as api, _client(api) as client:
        with pytest.raises(APIResponseValidationError) as excinfo:
            await client.video.submit(model="m", prompt="p", duration_seconds=5)

    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)
    assert any(r.levelno == logging.ERROR for r in all_logs.records)


@pytest.mark.parametrize(
    "body",
    [
        {"status": "COMPLETED", "url": [f"https://storage{LINK_TAIL}"]},
        {"url": f"https://storage{LINK_TAIL}"},
        [f"https://storage{LINK_TAIL}"],
    ],
    ids=["invalid-field", "no-status", "not-an-object"],
)
async def test_unparseable_retrieve_response_keeps_the_link_out(
    all_logs: pytest.LogCaptureFixture, body: Any
) -> None:
    async with _server(_json_answer(body)) as api, _client(api) as client:
        with pytest.raises((VeniceError, ValueError)) as excinfo:
            await client.video.retrieve(model="m", queue_id="q")

    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


async def test_malformed_retrieve_body_preview_stays_out_of_logs(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async def truncated(_request: web.Request) -> web.Response:
        return web.Response(
            text=f'{{"status": "COMPLETED", "url": "https://storage{LINK_TAIL}',
            content_type="application/json",
        )

    async with _server(truncated) as api, _client(api) as client:
        with pytest.raises(ValueError) as excinfo:
            await client.video.retrieve(model="m", queue_id="q")

    _assert_logs_are_clean(all_logs)
    _assert_error_is_clean(excinfo.value)


async def test_unparseable_queue_body_stays_out_of_the_error() -> None:
    async def truncated(_request: web.Request) -> web.Response:
        return web.Response(
            text=f'{{"model": "m", "download_url": "https://storage{LINK_TAIL}',
            content_type="application/json",
        )

    async with _server(truncated) as api, _client(api) as client:
        with pytest.raises(APIResponseValidationError) as excinfo:
            await client.post("video/queue", json_data={}, cast_to=VideoQueueResponse)

    assert excinfo.value.response_data is None
    _assert_error_is_clean(excinfo.value)


async def test_validation_error_on_a_hidden_model_keeps_its_useful_attributes() -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response(
            {"model": "m", "download_url": f"https://storage{LINK_TAIL}"},
            headers={"CF-RAY": "ray-1"},
        )

    async with _server(handler) as api, _client(api) as client:
        with pytest.raises(APIResponseValidationError) as excinfo:
            await client.post("video/queue", json_data={}, cast_to=VideoQueueResponse)

    error = excinfo.value
    assert error.response_data is None
    assert error.model_name == "VideoQueueResponse"
    assert isinstance(error.validation_error, ValidationError)
    assert [(e["type"], e["loc"]) for e in error.validation_error.errors()] == [
        ("missing", ("queue_id",))
    ]
    assert error.response is not None
    assert error.response.status == 200
    assert error.response.headers["CF-RAY"] == "ray-1"
    _assert_error_is_clean(error)


async def test_validation_error_on_a_model_that_shows_input_keeps_the_body() -> None:
    from venice_ai.types.api.video import VideoQuoteResponse

    async with (
        _server(_json_answer({"unexpected": "value"})) as api,
        _client(api) as client,
    ):
        with pytest.raises(APIResponseValidationError) as excinfo:
            await client.post("video/quote", json_data={}, cast_to=VideoQuoteResponse)

    assert excinfo.value.response_data == {"unexpected": "value"}
    assert isinstance(excinfo.value.response, aiohttp.ClientResponse)


# ---------------------------------------------------------------------------
# The response kept on a failed DELETE
# ---------------------------------------------------------------------------


async def test_failed_delete_keeps_status_and_retry_after_but_not_the_link() -> None:
    async def denied(_request: web.Request) -> web.Response:
        return web.Response(
            status=429,
            reason="Slow Down",
            headers={"Retry-After": "7", "Location": f"/{IN_PATH}/x", "x-amz-echo": LINK_TAIL},
        )

    async with (
        _server(denied) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        job = _job(client, f"{storage}{LINK_TAIL}")
        with pytest.raises(RateLimitError) as excinfo:
            await job.cancel()

    error = excinfo.value
    assert error.status_code == 429
    assert error.retry_after_seconds == 7
    response = error.response
    assert response is not None
    assert not isinstance(response, aiohttp.ClientResponse)
    assert response.status == 429
    assert response.reason == "Slow Down"
    assert response.method == "DELETE"
    assert response.headers["Retry-After"] == "7"
    assert "Location" not in response.headers
    assert str(response.url) == "http://127.0.0.1"
    _assert_error_is_clean(error)


# ---------------------------------------------------------------------------
# on_retry hooks
# ---------------------------------------------------------------------------


def _hook_recording(seen: list[Exception | None], **options: Any) -> RetryOptions:
    return RetryOptions(
        max_attempts=1,
        base_delay=0.001,
        max_delay=0.002,
        on_retry=lambda _attempt, _delay, error: seen.append(error),
        **options,
    )


def _drop_first_connection() -> Handler:
    calls = [0]

    async def handler(request: web.Request) -> web.Response:
        calls[0] += 1
        if calls[0] == 1:
            assert request.transport is not None
            request.transport.close()
        return web.Response(body=VIDEO_FILE, content_type="video/mp4")

    return handler


@asynccontextmanager
async def _session_stuck_on_a_full_pool(
    server: str, seen: list[Exception | None], api_origin: URL
) -> AsyncIterator[aiohttp.ClientSession]:
    """A session whose next request times out waiting to connect, which aiohttp reports
    with an error that quotes the whole URL."""
    middleware = create_retry_middleware(_hook_recording(seen), api_origin=api_origin)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=1),
        middlewares=[middleware],
        timeout=aiohttp.ClientTimeout(connect=0.05),
    ) as session:
        holder = asyncio.create_task(session.get(f"{server}/hold"))
        await asyncio.sleep(0.02)
        try:
            yield session
        finally:
            holder.cancel()
            await asyncio.gather(holder, return_exceptions=True)


def _hold_until_cancelled() -> Handler:
    async def handler(request: web.Request) -> web.Response:
        if request.path == "/hold":
            await asyncio.sleep(30)
        return web.Response(body=b"VIDEO")

    return handler


async def test_retry_hook_on_another_host_gets_an_error_without_the_link() -> None:
    seen: list[Exception | None] = []
    async with (
        _server(_hold_until_cancelled()) as storage,
        _session_stuck_on_a_full_pool(storage, seen, URL("https://api.example")) as session,
    ):
        with pytest.raises(aiohttp.ConnectionTimeoutError):
            await session.get(f"{storage}{LINK_TAIL}#{IN_FRAGMENT}")

    assert len(seen) == 1
    error = seen[0]
    assert isinstance(error, APITimeoutError)
    assert error.original_error_type is aiohttp.ConnectionTimeoutError
    assert "ConnectionTimeoutError" in str(error)
    assert error.__cause__ is None and error.__context__ is None
    _assert_error_is_clean(error)


async def test_retry_hook_on_fetch_external_gets_an_error_without_the_link() -> None:
    seen: list[Exception | None] = []
    async with (
        _server(_drop_first_connection()) as storage,
        _server(_serve_after_503s(0)) as api,
        VeniceClient(
            api_key=API_KEY, base_url=f"{api}/api/v1", retry_options=_hook_recording(seen)
        ) as client,
    ):
        assert await client.fetch_external(f"{storage}{LINK_TAIL}") == VIDEO_FILE

    assert len(seen) == 1
    error = seen[0]
    assert isinstance(error, APIConnectionError)
    assert error.original_error_type is aiohttp.ServerDisconnectedError
    assert "ServerDisconnectedError" in str(error)
    _assert_error_is_clean(error)


async def test_retry_hook_on_an_api_request_gets_the_error_itself() -> None:
    seen: list[Exception | None] = []
    async with (
        _server(_drop_first_connection()) as api,
        VeniceClient(
            api_key=API_KEY, base_url=f"{api}/api/v1", retry_options=_hook_recording(seen)
        ) as client,
    ):
        await client.fetch_external(f"{api}/api/v1/thing")

    assert len(seen) == 1
    assert isinstance(seen[0], aiohttp.ServerDisconnectedError)


async def test_retry_hook_on_an_api_request_never_sees_its_query_string() -> None:
    seen: list[Exception | None] = []
    async with (
        _server(_hold_until_cancelled()) as api,
        _session_stuck_on_a_full_pool(api, seen, URL(api)) as session,
    ):
        with pytest.raises(aiohttp.ConnectionTimeoutError):
            await session.get(f"{api}/api/v1/thing?sig={IN_QUERY}")

    assert len(seen) == 1
    assert isinstance(seen[0], APITimeoutError)
    _assert_error_is_clean(seen[0])


async def test_retry_hook_handles_a_failure_that_is_not_a_transport_error() -> None:
    seen: list[Exception | None] = []
    calls = [OSError(f"cannot open http://h{LINK_TAIL}"), None]

    async def handler(_request: Any) -> Any:
        outcome = calls.pop(0)
        if outcome is not None:
            raise outcome
        response = MagicMock()
        response.status = 200
        return response

    middleware = create_retry_middleware(
        _hook_recording(seen, retry_exceptions=(OSError,)), api_origin=URL("https://api.example")
    )
    await middleware(_stub_request(f"http://h{LINK_TAIL}"), handler)

    assert isinstance(seen[0], APIConnectionError)
    assert seen[0].original_error_type is OSError
    _assert_error_is_clean(seen[0])


async def test_retry_hook_that_raises_does_not_chain_the_original_error() -> None:
    def hook(_attempt: int, _delay: float, _error: Exception | None) -> None:
        raise RuntimeError("hook failed")

    async def handler(_request: Any) -> Any:
        raise aiohttp.ConnectionTimeoutError(f"Connection timeout to host http://h{LINK_TAIL}")

    middleware = create_retry_middleware(
        RetryOptions(max_attempts=1, base_delay=0.001, on_retry=hook),
        api_origin=URL("https://api.example"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        await middleware(_stub_request(f"http://h{LINK_TAIL}"), handler)

    _assert_error_is_clean(excinfo.value)


async def test_delete_on_the_link_is_not_retried_so_no_hook_runs() -> None:
    seen: list[Exception | None] = []
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async with (
        _server(_serve_after_503s(0)) as api,
        VeniceClient(
            api_key=API_KEY, base_url=f"{api}/api/v1", retry_options=_hook_recording(seen)
        ) as client,
    ):
        job = _job(client, f"http://127.0.0.1:{port}{LINK_TAIL}")
        with pytest.raises(APIConnectionError):
            await job.cancel()

    assert seen == []


# ---------------------------------------------------------------------------
# Music jobs fetch the same kind of link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [MusicQueueResponse, MusicProcessingStatus, MusicFailedStatus, MusicCompletedStatus],
)
def test_music_models_hide_their_input_in_validation_errors(model: type[Any]) -> None:
    assert model.model_config.get("hide_input_in_errors") is True


async def test_unparseable_music_queue_response_keeps_the_link_out(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    body = {"model": "m", "url": f"https://storage{LINK_TAIL}"}
    async with _server(_json_answer(body)) as api, _client(api) as client:
        with pytest.raises(APIResponseValidationError) as excinfo:
            await client.post("audio/queue", json_data={}, cast_to=MusicQueueResponse)

    assert excinfo.value.response_data is None
    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


@pytest.mark.parametrize(
    "body",
    [
        {"status": "COMPLETED", "url": [f"https://storage{LINK_TAIL}"]},
        {"url": f"https://storage{LINK_TAIL}"},
        [f"https://storage{LINK_TAIL}"],
    ],
    ids=["invalid-field", "no-status", "not-an-object"],
)
async def test_unparseable_music_retrieve_response_keeps_the_link_out(
    all_logs: pytest.LogCaptureFixture, body: Any
) -> None:
    async with _server(_json_answer(body)) as api, _client(api) as client:
        with pytest.raises((VeniceError, ValueError)) as excinfo:
            await client.music.retrieve(model="m", queue_id="q")

    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


async def test_malformed_music_retrieve_body_stays_out_of_logs_and_error(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    async def truncated(_request: web.Request) -> web.Response:
        return web.Response(
            text=f'{{"status": "COMPLETED", "url": "https://storage{LINK_TAIL}',
            content_type="application/json",
        )

    async with _server(truncated) as api, _client(api) as client:
        with pytest.raises(json.JSONDecodeError) as excinfo:
            await client.music.retrieve(model="m", queue_id="q")

    _assert_logs_are_clean(all_logs)
    _assert_error_is_clean(excinfo.value)
    assert any(r.levelno == logging.ERROR for r in all_logs.records)


async def test_music_download_failure_and_retry_logs_keep_the_link_out(
    all_logs: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    calls = [0]

    async def handler(_request: web.Request) -> web.Response:
        calls[0] += 1
        return web.Response(status=503 if calls[0] == 1 else 410, reason="Gone")

    async with (
        _server(handler) as storage,
        _server(_serve_after_503s(0)) as api,
        _client(api) as client,
    ):
        job = MusicJob(client, MusicQueueResponse(model="m", queue_id="q"))
        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await job.download(
                tmp_path / "a.mp3",
                MusicCompletedStatus(status="COMPLETED", url=f"{storage}{LINK_TAIL}"),
            )

    assert excinfo.value.status == 410
    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


def test_validate_response_model_on_a_hidden_model_keeps_the_link_out(
    all_logs: pytest.LogCaptureFixture,
) -> None:
    from venice_ai._validation import validate_response_model

    with pytest.raises(APIResponseValidationError) as excinfo:
        validate_response_model(
            {"model": "m", "download_url": f"https://storage{LINK_TAIL}"},
            VideoQueueResponse,
        )

    assert excinfo.value.response_data is None
    _assert_error_is_clean(excinfo.value)
    _assert_logs_are_clean(all_logs)


async def test_html_answer_for_a_hidden_model_keeps_body_and_key_out_of_the_error() -> None:
    """aiohttp's ContentTypeError holds the request, whose headers carry the API key."""

    async def html(_request: web.Request) -> web.Response:
        return web.Response(text=f"<a href='https://storage{LINK_TAIL}'>", content_type="text/html")

    async with _server(html) as api, _client(api) as client:
        with pytest.raises(APIResponseValidationError) as excinfo:
            await client.post("video/queue", json_data={}, cast_to=VideoQueueResponse)

    error = excinfo.value
    assert error.__cause__ is None and error.__context__ is None
    assert error.response_data is None
    assert "ContentTypeError" in str(error)
    _assert_error_is_clean(error)


# ---------------------------------------------------------------------------
# What the samples teach
# ---------------------------------------------------------------------------


def test_no_sample_prints_a_download_link() -> None:
    root = Path(__file__).parents[2]
    sources = [
        *(root / "src/venice_ai/resources").glob("*.py"),
        *(root / "src/venice_ai/cli").rglob("*.py"),
        *(root / "src/venice_ai/skills").rglob("*.md"),
        *(root / "website/docs/guides").rglob("*.md"),
        *(root / "examples/video").rglob("*.py"),
        *(root / "examples/music").rglob("*.py"),
        root / "README.md",
    ]
    assert len(sources) > 20
    printed_link = re.compile(
        r"(print|echo|info|debug)\(.*\b(status|result|job|queued|completed|queue_resp)"
        r"\.(url|download_url)\b"
    )
    offenders = [
        f"{path.relative_to(root)}: {line.strip()}"
        for path in sources
        for line in path.read_text(encoding="utf-8").splitlines()
        if printed_link.search(line)
    ]
    assert offenders == []
