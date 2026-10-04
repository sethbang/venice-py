"""The billing-aware retry policy, observed on the wire.

Each test drives a real aiohttp session (or a real ``VeniceClient``) against a
local server that counts the requests it receives. The count is what a paid
endpoint would bill, so "a paid POST is sent exactly once" is asserted as the
server's hit count, not as a mock's call count.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.core.config import HttpClientConfig
from venice_ai.middleware.retry import (
    RETRY_COUNT_HEADER,
    RetryClass,
    RetryOptions,
    classify_request,
    create_retry_middleware,
)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "headers", "expected"),
    [
        ("GET", "/api/v1/models", {}, RetryClass.IDEMPOTENT),
        ("DELETE", "/api/v1/api_keys", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/video/quote", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/audio/quote", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/video/retrieve", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/audio/complete", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/audio/voice-changer/retrieve", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/billing/usage-analytics", {}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/crypto/rpc/base", {"Idempotency-Key": "k"}, RetryClass.IDEMPOTENT),
        ("POST", "/api/v1/chat/completions", {}, RetryClass.INFERENCE),
        ("POST", "/api/v1/responses", {}, RetryClass.INFERENCE),
        ("POST", "/api/v1/embeddings", {}, RetryClass.INFERENCE),
        ("POST", "/api/v1/image/generate", {}, RetryClass.PAID),
        ("POST", "/api/v1/video/queue", {}, RetryClass.PAID),
        ("POST", "/api/v1/audio/queue", {}, RetryClass.PAID),
        ("POST", "/api/v1/audio/speech", {}, RetryClass.PAID),
        ("POST", "/api/v1/api_keys", {}, RetryClass.PAID),
        ("POST", "/api/v1/x402/top-up", {}, RetryClass.PAID),
        ("POST", "/api/v1/crypto/rpc/base", {}, RetryClass.PAID),
        ("POST", "/api/v1/something/new", {}, RetryClass.PAID),
    ],
)
def test_classify_request(
    method: str, path: str, headers: dict[str, str], expected: RetryClass
) -> None:
    assert classify_request(method, path, headers) is expected


def test_defaults() -> None:
    options = RetryOptions()
    assert options.max_attempts == 2
    assert options.base_delay == 0.5
    assert options.max_delay == 8.0
    assert options.max_retry_after == 60.0
    assert options.max_inference_500_retries == 1
    assert 429 not in options.retry_status_codes
    assert HttpClientConfig().max_retries == 2


def test_classify_request_is_public() -> None:
    import venice_ai

    assert venice_ai.classify_request is classify_request
    assert venice_ai.RetryClass is RetryClass
    assert {"classify_request", "RetryClass", "RetryOptions"} <= set(venice_ai.__all__)
    assert venice_ai.classify_request("POST", "/api/v1/video/queue", {}) is RetryClass.PAID


@pytest.mark.parametrize("codes", [{429}, {429, 503}, frozenset({429})])
def test_429_in_retry_status_codes_is_rejected(codes: Any) -> None:
    with pytest.raises(ValueError, match=r"cannot contain 429.*SimpleRateLimiter\(max_retries"):
        RetryOptions(retry_status_codes=codes)


def test_429_cannot_be_added_through_replace() -> None:
    with pytest.raises(ValueError, match="cannot contain 429"):
        dataclasses.replace(RetryOptions(), retry_status_codes={429, 500})


def test_retry_status_codes_are_frozen() -> None:
    options = RetryOptions(retry_status_codes={408, 503})
    assert options.retry_status_codes == frozenset({408, 503})
    assert isinstance(options.retry_status_codes, frozenset)
    assert isinstance(RetryOptions().retry_status_codes, frozenset)


@pytest.mark.parametrize("codes", [[500, 503], (500,), "500", 503])
def test_retry_status_codes_must_be_a_set(codes: Any) -> None:
    with pytest.raises(TypeError, match="retry_status_codes must be a set"):
        RetryOptions(retry_status_codes=codes)


def test_429_cannot_be_assigned_after_construction() -> None:
    options = RetryOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.retry_status_codes = frozenset({429})  # type: ignore[misc]
    assert 429 not in options.retry_status_codes


def test_collections_are_immutable_and_not_shared_mutably() -> None:
    methods = {"GET"}
    exceptions = [TimeoutError]
    options = RetryOptions(idempotent_methods=methods, retry_exceptions=exceptions)
    methods.add("POST")
    exceptions.append(ValueError)
    assert options.idempotent_methods == frozenset({"GET"})
    assert options.retry_exceptions == (TimeoutError,)
    with pytest.raises(TypeError, match="idempotent_methods must be a set"):
        RetryOptions(idempotent_methods=["GET"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retry_exceptions must be a sequence"):
        RetryOptions(retry_exceptions={TimeoutError})  # type: ignore[arg-type]


def test_client_retry_options_copy_cannot_change_status_codes() -> None:
    client = VeniceClient(api_key="k", retry_options=RetryOptions(retry_status_codes={503}))
    copy = client.retry_options
    assert copy is not None
    assert not hasattr(copy.retry_status_codes, "add")
    assert client.retry_options == copy


# ---------------------------------------------------------------------------
# Local server
# ---------------------------------------------------------------------------


class _Server:
    """Answers every request with a configured behavior and counts hits."""

    def __init__(self) -> None:
        self.status = 500
        self.headers: dict[str, str] = {}
        self.mode = "status"  # "status" | "slow" | "disconnect"
        self.hits = 0
        self.retry_counts: list[str | None] = []
        self.idempotency_keys: list[str | None] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.hits += 1
        self.retry_counts.append(request.headers.get(RETRY_COUNT_HEADER))
        self.idempotency_keys.append(request.headers.get("Idempotency-Key"))
        await request.read()
        if self.mode == "slow":
            await asyncio.sleep(1.0)
        if self.mode == "disconnect":
            assert request.transport is not None
            request.transport.close()
            await asyncio.sleep(0.05)
        return web.json_response({"error": "x"}, status=self.status, headers=self.headers)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@asynccontextmanager
async def _server() -> AsyncIterator[_Server]:
    srv = _Server()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", srv._handle)
    srv._runner = web.AppRunner(app)
    await srv._runner.setup()
    site = web.TCPSite(srv._runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    assert sockets is not None
    srv.port = sockets[0].getsockname()[1]
    try:
        yield srv
    finally:
        await srv._runner.cleanup()


_FAST = dataclasses.replace(RetryOptions(), base_delay=0.0, jitter_factor=0.0)


async def _send(
    srv: _Server,
    method: str,
    path: str,
    *,
    options: RetryOptions = _FAST,
    timeout: aiohttp.ClientTimeout | None = None,
    headers: dict[str, str] | None = None,
) -> int | str:
    """Send one request through the retry middleware; return status or error name."""
    async with aiohttp.ClientSession(
        middlewares=[create_retry_middleware(options)],
        timeout=timeout or aiohttp.ClientTimeout(total=10),
    ) as session:
        try:
            async with session.request(method, srv.url + path, json={}, headers=headers) as r:
                return r.status
        except (aiohttp.ClientError, TimeoutError) as exc:
            return type(exc).__name__


# (method, path, status) -> expected hits with max_attempts=2
_STATUS_MATRIX = [
    ("GET", "/api/v1/models", 500, 3),
    ("GET", "/api/v1/models", 504, 3),
    ("POST", "/api/v1/video/quote", 500, 3),
    ("POST", "/api/v1/chat/completions", 500, 2),
    ("POST", "/api/v1/chat/completions", 502, 3),
    ("POST", "/api/v1/chat/completions", 503, 3),
    ("POST", "/api/v1/chat/completions", 504, 1),
    ("POST", "/api/v1/image/generate", 500, 1),
    ("POST", "/api/v1/image/generate", 502, 1),
    ("POST", "/api/v1/image/generate", 504, 1),
    ("POST", "/api/v1/video/queue", 500, 1),
    ("POST", "/api/v1/video/queue", 503, 1),
    ("POST", "/api/v1/x402/top-up", 503, 1),
    ("POST", "/api/v1/api_keys", 503, 1),
    ("POST", "/api/v1/image/generate", 503, 3),
    ("POST", "/api/v1/audio/queue", 503, 3),
    ("POST", "/api/v1/audio/queue", 500, 1),
    ("GET", "/api/v1/models", 429, 1),
    ("POST", "/api/v1/chat/completions", 400, 1),
]


@pytest.mark.parametrize(("method", "path", "status", "hits"), _STATUS_MATRIX)
async def test_status_matrix(method: str, path: str, status: int, hits: int) -> None:
    async with _server() as srv:
        srv.status = status
        assert await _send(srv, method, path) == status
        assert srv.hits == hits


async def test_retry_count_header_marks_resent_attempts() -> None:
    async with _server() as srv:
        srv.status = 503
        await _send(srv, "GET", "/api/v1/models")
        assert srv.retry_counts == [None, "1", "2"]


async def test_read_timeout_never_resends_a_paid_post() -> None:
    timeout = aiohttp.ClientTimeout(total=10, sock_read=0.2)
    async with _server() as srv:
        srv.mode = "slow"
        assert await _send(srv, "POST", "/api/v1/video/queue", timeout=timeout) == (
            "SocketTimeoutError"
        )
        assert srv.hits == 1


async def test_read_timeout_never_resends_chat() -> None:
    timeout = aiohttp.ClientTimeout(total=10, sock_read=0.2)
    async with _server() as srv:
        srv.mode = "slow"
        await _send(srv, "POST", "/api/v1/chat/completions", timeout=timeout)
        assert srv.hits == 1


async def test_read_timeout_resends_a_get() -> None:
    timeout = aiohttp.ClientTimeout(total=10, sock_read=0.2)
    async with _server() as srv:
        srv.mode = "slow"
        await _send(srv, "GET", "/api/v1/models", timeout=timeout)
        assert srv.hits == 3


async def test_server_disconnect_never_resends_a_paid_post() -> None:
    async with _server() as srv:
        srv.mode = "disconnect"
        result = await _send(srv, "POST", "/api/v1/image/generate")
        assert result == "ServerDisconnectedError"
        assert srv.hits == 1


async def test_server_disconnect_resends_a_get() -> None:
    retries: list[Exception | None] = []
    options = dataclasses.replace(_FAST, on_retry=lambda _i, _d, e: retries.append(e))
    async with _server() as srv:
        srv.mode = "disconnect"
        await _send(srv, "GET", "/api/v1/models", options=options)
        # aiohttp itself reruns the whole middleware chain once for an
        # idempotent request whose pooled connection dropped, so the counts are
        # lower bounds here; the POST case above shows aiohttp adds no resend.
        assert len(retries) >= 2
        assert srv.hits >= 3


async def test_connect_failure_is_retried_for_paid_posts() -> None:
    attempts: list[Exception | None] = []
    options = dataclasses.replace(_FAST, on_retry=lambda _i, _d, e: attempts.append(e))
    async with _server() as srv:
        port = srv.port
    # The server is gone: every attempt is refused before anything is sent.
    srv.port = port
    result = await _send(srv, "POST", "/api/v1/video/queue", options=options)
    assert result == "ClientConnectorError"
    assert len(attempts) == 2


async def test_retry_after_within_cap_is_used() -> None:
    delays: list[float] = []
    options = dataclasses.replace(_FAST, on_retry=lambda _i, d, _e: delays.append(d))
    async with _server() as srv:
        srv.status = 503
        srv.headers = {"retry-after-ms": "20"}
        await _send(srv, "POST", "/api/v1/video/queue", options=options)
        assert srv.hits == 3
        assert delays == [pytest.approx(0.02), pytest.approx(0.02)]


async def test_retry_after_above_cap_is_surfaced() -> None:
    async with _server() as srv:
        srv.status = 503
        srv.headers = {"Retry-After": "120"}
        assert await _send(srv, "GET", "/api/v1/models") == 503
        assert srv.hits == 1


async def test_idempotency_key_makes_a_post_safe_to_resend() -> None:
    async with _server() as srv:
        srv.status = 500
        await _send(srv, "POST", "/api/v1/crypto/rpc/base", headers={"Idempotency-Key": "abc"})
        assert srv.hits == 3
        assert srv.idempotency_keys == ["abc", "abc", "abc"]


async def test_custom_classifier_overrides_the_default() -> None:
    options = dataclasses.replace(_FAST, classifier=lambda *_: RetryClass.IDEMPOTENT)
    async with _server() as srv:
        srv.status = 500
        await _send(srv, "POST", "/api/v1/image/generate", options=options)
        assert srv.hits == 3


async def test_retry_non_idempotent_false_disables_post_retries() -> None:
    options = dataclasses.replace(_FAST, retry_non_idempotent=False)
    async with _server() as srv:
        srv.status = 503
        await _send(srv, "POST", "/api/v1/chat/completions", options=options)
        assert srv.hits == 1


# ---------------------------------------------------------------------------
# Through VeniceClient
# ---------------------------------------------------------------------------


async def _client_call(srv: _Server, **client_kwargs: Any) -> None:
    client = VeniceClient(api_key="sk-test", base_url=srv.url + "/api/v1", **client_kwargs)
    async with client:
        with pytest.raises(Exception):  # noqa: B017 - the hit count is under test
            await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}]
            )


async def test_client_chat_500_is_retried_once() -> None:
    async with _server() as srv:
        srv.status = 500
        await _client_call(srv, retry_options=_FAST)
        assert srv.hits == 2


async def test_client_max_retries_keyword_keeps_the_billing_rules() -> None:
    async with _server() as srv:
        srv.status = 500
        await _client_call(srv, max_retries=5)
        assert srv.hits == 2


async def test_crypto_rpc_sends_a_stable_idempotency_key() -> None:
    async with _server() as srv:
        srv.status = 500
        client = VeniceClient(api_key="sk-test", base_url=srv.url + "/api/v1", retry_options=_FAST)
        async with client:
            with pytest.raises(Exception):  # noqa: B017 - the hit count is under test
                await client.crypto.rpc(network="base-mainnet", method="eth_chainId")
        assert srv.hits == 3
        keys = set(srv.idempotency_keys)
        assert len(keys) == 1
        assert None not in keys
