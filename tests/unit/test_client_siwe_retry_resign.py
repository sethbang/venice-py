"""Every retry attempt must carry its own freshly signed SIWE envelope.

Venice treats a SIWE nonce as single-use. The retry middleware is an aiohttp
*client* middleware, so it re-awaits ``handler(request)`` on the **same**
``ClientRequest`` object — headers included. Without an explicit re-sign the
retried attempt replays the envelope the first attempt already sent, and the
server answers ``401 This nonce has already been used`` instead of serving the
retry. The rate limiter is a second, independent path: it retries a ``429`` by
re-invoking the request callable, so anything signed once and captured in it
is resent.

These tests drive a real :class:`VeniceClient` against a local aiohttp server
that fails and then succeeds, and assert on the **decoded nonce** of each
attempt the server actually saw. Asserting that the envelopes merely differ
would be vacuous: ECDSA signing here is deterministic (RFC 6979), so envelope
inequality only proves the signed *message* changed, which a static nonce
paired with a ticking ``Issued At`` would satisfy.

Skips if the ``[x402]`` extra (``eth_account`` / ``siwe``) is not installed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

pytest.importorskip("eth_account", reason="x402 extra not installed")
pytest.importorskip("siwe", reason="x402 extra not installed")

from aiohttp import web  # noqa: E402

from venice_ai import VeniceClient  # noqa: E402
from venice_ai.auth.x402 import X402Auth  # noqa: E402
from venice_ai.middleware.retry import RetryOptions  # noqa: E402

pytestmark = pytest.mark.asyncio

# Throwaway keys — never funded, never reused.
_KEY_A = "0x" + "a" * 63 + "b"
_KEY_B = "0x" + "c" * 63 + "d"

# Retry fast and deterministically; the wall clock is not under test.
_FAST_RETRIES = RetryOptions(
    max_attempts=2,
    base_delay=0.0,
    jitter_factor=0.0,
    respect_retry_after=False,
)


def _decode(header: str) -> dict[str, Any]:
    """Decode an ``X-Sign-In-With-X`` value into its JSON envelope."""
    return json.loads(base64.b64decode(header))  # type: ignore[no-any-return]


def _nonce_of(header: str) -> str:
    """Pull the SIWE nonce out of an envelope's signed message."""
    message = _decode(header)["message"]
    found = re.search(r"^Nonce: (\S+)$", message, re.M)
    assert found is not None, f"no nonce in SIWE message: {message!r}"
    return found.group(1)


class _FlakyServer:
    """Local server that fails the first ``fail_times`` requests, then succeeds.

    Records the ``X-Sign-In-With-X`` header of every attempt it sees, so the
    tests can assert on what actually went out on the wire rather than on what
    the client believed it sent.
    """

    def __init__(self, fail_times: int = 2, *, sse: bool = False, fail_status: int = 503) -> None:
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.sse = sse
        self.envelopes: list[str | None] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    async def _handle(self, request: web.Request) -> web.Response:
        self.envelopes.append(request.headers.get("X-Sign-In-With-X"))
        if len(self.envelopes) <= self.fail_times:
            return web.json_response({"error": "transient"}, status=self.fail_status)
        if self.sse:
            return web.Response(
                body=b"data: [DONE]\n\n",
                content_type="text/event-stream",
            )
        return web.json_response({"ok": True})

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@asynccontextmanager
async def flaky_server(*, sse: bool = False, fail_status: int = 503) -> AsyncIterator[_FlakyServer]:
    """Run a :class:`_FlakyServer` for the duration of the block.

    Deliberately a context manager rather than a fixture: this suite's
    ``event_loop`` fixture is module-scoped, so an async fixture can end up on a
    different loop than the test body, leaving the client waiting on a server
    that never runs.
    """
    srv = _FlakyServer(sse=sse, fail_status=fail_status)
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


class _SequentialWorkerLimiter:
    """A rate limiter that runs every request in ONE long-lived worker task.

    Deliberately adversarial for the re-sign machinery: the worker task is
    created before any request is submitted, so a signer published outside the
    submitted callable would never reach the middleware. A scheduler is free to
    work this way, so the client must not depend on it not doing so.
    """

    classifier = None

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[Callable[[], Awaitable[Any]], Any]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def is_running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        if not self.is_running():
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def _run(self) -> None:
        while True:
            func, future = await self._queue.get()
            try:
                future.set_result(await func())
            except Exception as exc:  # relayed to the caller, like a real scheduler
                future.set_exception(exc)

    async def submit_request(
        self,
        metadata: Any,
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Any = None,
    ) -> Any:
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((request_func, future))
        return await future


class _RetryOn429Limiter:
    """Models the 429 retry loop in ``SimpleRateLimiter.submit_request``.

    That loop re-invokes ``request_func()`` for each attempt rather than
    replaying a prepared request, so anything a request closure captured once —
    such as a per-call SIWE envelope passed in ``headers`` — is resent verbatim.
    This is a second replay path, independent of the retry middleware, and the
    stub reproduces its shape.
    """

    classifier = None

    def __init__(self, attempts: int = 3) -> None:
        self.attempts = attempts

    def is_running(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def submit_request(
        self,
        metadata: Any,
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Any = None,
    ) -> Any:
        response = None
        for _ in range(self.attempts):
            response = await request_func()
            if getattr(response, "status", 200) != 429:
                return response
        return response


async def test_each_retry_attempt_carries_a_fresh_nonce() -> None:
    """The direct path: two 503s then a 200, three distinct nonces on the wire."""
    async with flaky_server() as server:
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                auth=X402Auth(private_key=_KEY_A),
                base_url=server.base_url,
                retry_options=_FAST_RETRIES,
            )
        async with client:
            await client.post("chat/completions", json_data={"model": "x", "messages": []})

        assert len(server.envelopes) == 3, "expected the initial attempt plus two retries"
        assert all(e is not None for e in server.envelopes)
        nonces = [_nonce_of(e) for e in server.envelopes if e is not None]
        assert len(set(nonces)) == 3, f"a nonce was replayed across retries: {nonces}"


async def test_each_retry_attempt_carries_a_fresh_nonce_via_scheduler() -> None:
    """The scheduler path must re-sign too, even from a pre-existing worker task."""
    async with flaky_server() as server:
        limiter = _SequentialWorkerLimiter()
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                auth=X402Auth(private_key=_KEY_A),
                base_url=server.base_url,
                retry_options=_FAST_RETRIES,
                rate_limiter=limiter,
            )
        try:
            async with client:
                await client.post("chat/completions", json_data={"model": "x", "messages": []})
        finally:
            await limiter.stop()

        nonces = [_nonce_of(e) for e in server.envelopes if e is not None]
        assert len(nonces) == 3
        assert len(set(nonces)) == 3, f"a nonce was replayed across retries: {nonces}"


async def test_retry_resigns_with_the_per_call_wallet_not_the_client_default() -> None:
    """A per-call ``auth=`` wallet must sign every attempt — including retries.

    ``client.x402.balance(auth=...)`` may pass a different wallet than the one
    the client was constructed with. Re-signing with the client's own wallet
    would put a signature from wallet A on a URL naming wallet B — a worse
    failure than a replayed nonce.
    """
    wallet_a = X402Auth(private_key=_KEY_A)
    wallet_b = X402Auth(private_key=_KEY_B)
    assert wallet_a.wallet_address != wallet_b.wallet_address

    async with flaky_server() as server:
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                auth=wallet_a,
                base_url=server.base_url,
                retry_options=_FAST_RETRIES,
            )
        async with client:
            # The stub server answers with a body the response model rejects;
            # only the headers it recorded are under test here.
            with pytest.raises(Exception):  # noqa: B017 - body shape is not under test
                await client.x402.balance(auth=wallet_b)

        assert len(server.envelopes) == 3
        addresses = {_decode(e)["address"] for e in server.envelopes if e is not None}
        assert addresses == {wallet_b.wallet_address}, (
            f"retry signed with the wrong wallet: {addresses}"
        )
        nonces = [_nonce_of(e) for e in server.envelopes if e is not None]
        assert len(set(nonces)) == 3, f"a nonce was replayed across retries: {nonces}"


async def test_streaming_requests_also_resign_on_retry() -> None:
    """Streaming inference re-signs on retry like the unary path.

    Streaming chat is the main mode-2 use case. The retry fires on the status
    line, before any of the event stream is read.
    """

    class _Chunk(BaseModel):
        pass

    async with flaky_server(sse=True) as server:
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                auth=X402Auth(private_key=_KEY_A),
                base_url=server.base_url,
                retry_options=_FAST_RETRIES,
            )
        async with client:
            stream = client._stream_request(
                "POST",
                "chat/completions",
                json_data={"model": "x", "messages": [], "stream": True},
                cast_to=_Chunk,
            )
            [chunk async for chunk in stream]

        nonces = [_nonce_of(e) for e in server.envelopes if e is not None]
        assert len(nonces) == 3
        assert len(set(nonces)) == 3, f"a nonce was replayed across retries: {nonces}"


async def test_limiter_429_retries_resign_a_per_call_wallet() -> None:
    """A 429 retry must not resend the envelope the previous attempt spent."""
    # 429 is deliberately absent from RetryOptions.retry_status_codes, so the
    # middleware ignores it and only the limiter's loop drives these attempts.
    async with flaky_server(fail_status=429) as server:
        wallet_b = X402Auth(private_key=_KEY_B)
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                auth=X402Auth(private_key=_KEY_A),
                base_url=server.base_url,
                retry_options=_FAST_RETRIES,
                rate_limiter=_RetryOn429Limiter(attempts=3),
            )
        async with client:
            with pytest.raises(Exception):  # noqa: B017 - body shape is not under test
                await client.x402.balance(auth=wallet_b)

        assert len(server.envelopes) == 3, "expected two 429s and then a success"
        addresses = {_decode(e)["address"] for e in server.envelopes if e is not None}
        assert addresses == {wallet_b.wallet_address}
        nonces = [_nonce_of(e) for e in server.envelopes if e is not None]
        assert len(set(nonces)) == 3, f"a nonce was replayed across limiter retries: {nonces}"


async def test_a_hand_rolled_envelope_is_left_untouched() -> None:
    """A header the caller built is the caller's to refresh.

    Re-signing it would substitute an address the caller did not choose, so the
    SDK leaves it exactly as passed — which is the contract the x402 skill
    reference documents. Nothing in ``src/`` reaches this branch any more, so
    without this test it would be live but unexercised.
    """
    wallet_b = X402Auth(private_key=_KEY_B)
    mine = wallet_b.build_header()

    async with flaky_server() as server:
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                auth=X402Auth(private_key=_KEY_A),
                base_url=server.base_url,
                retry_options=_FAST_RETRIES,
            )
        async with client:
            await client.post(
                "chat/completions",
                json_data={"model": "x", "messages": []},
                headers={"X-Sign-In-With-X": mine},
            )

        assert server.envelopes == [mine, mine, mine], (
            "the SDK rewrote an envelope the caller supplied"
        )


async def test_bearer_wins_when_a_wallet_is_also_configured() -> None:
    """api_key + auth: Bearer is the default auth, and no envelope is injected.

    This is the case where a wallet *is* available to sign with, so it exercises
    the guards rather than passing for want of a wallet.
    """
    async with flaky_server() as server:
        client = VeniceClient(
            api_key="vn_test_123456789",
            auth=X402Auth(private_key=_KEY_A),
            base_url=server.base_url,
            retry_options=_FAST_RETRIES,
        )
        async with client:
            await client.post("chat/completions", json_data={"model": "x", "messages": []})

        assert server.envelopes == [None, None, None], (
            "a wallet envelope was attached to a Bearer-authenticated request"
        )
