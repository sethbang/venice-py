"""RPC transport failures in :func:`fetch_solana_tx_context` hide the RPC URL.

Hosted Solana RPC endpoints commonly carry an API key in the query string or
path, so the URL is a secret. aiohttp prints the URL in the text of its response
errors. These tests drive a real aiohttp session against a local server to get
real aiohttp exceptions, and assert the secret is absent from everything an
application could log.
"""

from __future__ import annotations

import socket
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

pytest.importorskip("solders", reason="x402-solana extra not installed")

from solders.hash import Hash  # noqa: E402
from solders.keypair import Keypair  # noqa: E402

from venice_ai.auth.x402_solana import (  # noqa: E402
    USDC_SOLANA_MAINNET,
    SolanaX402Auth,
    fetch_solana_tx_context,
)
from venice_ai.exceptions import PaymentRequiredError  # noqa: E402
from venice_ai.resources.x402 import X402  # noqa: E402

_SECRET = "SECRETrpcTOKEN123"
_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _assert_no_secret(err: BaseException) -> None:
    """The secret must be absent from the error's text and everything chained to it."""
    assert _SECRET not in str(err)
    assert _SECRET not in repr(err)
    assert _SECRET not in "".join(traceback.format_exception(err))
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest_asyncio.fixture
async def rpc_server() -> AsyncIterator[Callable[[Handler], Awaitable[str]]]:
    """Start a loopback server for one handler; returns its secret-bearing URL."""
    runners: list[web.AppRunner] = []

    async def start(handler: Handler) -> str:
        app = web.Application()
        app.router.add_post("/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        return f"http://127.0.0.1:{port}/v2/{_SECRET}/?api-key={_SECRET}"

    yield start
    for runner in runners:
        await runner.cleanup()


async def _fetch(url: str) -> tuple[str, int, str]:
    async with aiohttp.ClientSession() as session:
        return await fetch_solana_tx_context(url, USDC_SOLANA_MAINNET, http=session)


def _refused_url() -> str:
    """A loopback URL nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v2/{_SECRET}/?api-key={_SECRET}"


@pytest.mark.parametrize("status", [401, 429, 500])
@pytest.mark.asyncio
async def test_http_status_error_becomes_runtime_error_without_url(
    rpc_server: Any, status: int
) -> None:
    async def handler(_: web.Request) -> web.Response:
        return web.Response(status=status, text="nope")

    url = await rpc_server(handler)

    with pytest.raises(RuntimeError, match=f"getLatestBlockhash.*HTTP {status}") as exc_info:
        await _fetch(url)

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_non_json_content_type_becomes_runtime_error_without_url(rpc_server: Any) -> None:
    async def handler(_: web.Request) -> web.Response:
        return web.Response(status=200, text="<html>gateway</html>", content_type="text/html")

    url = await rpc_server(handler)

    with pytest.raises(RuntimeError, match="getLatestBlockhash.*ContentTypeError") as exc_info:
        await _fetch(url)

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_undecodable_json_body_becomes_runtime_error(rpc_server: Any) -> None:
    async def handler(_: web.Request) -> web.Response:
        return web.Response(status=200, text="not json", content_type="application/json")

    url = await rpc_server(handler)

    with pytest.raises(RuntimeError, match="getLatestBlockhash.*JSONDecodeError") as exc_info:
        await _fetch(url)

    _assert_no_secret(exc_info.value)


@pytest.mark.parametrize("body", ["[]", "null", '"text"'])
@pytest.mark.asyncio
async def test_json_body_that_is_not_an_object_becomes_runtime_error(
    rpc_server: Any, body: str
) -> None:
    async def handler(_: web.Request) -> web.Response:
        return web.Response(status=200, text=body, content_type="application/json")

    url = await rpc_server(handler)

    with pytest.raises(RuntimeError, match="getLatestBlockhash unexpected result") as exc_info:
        await _fetch(url)

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_connection_refused_becomes_runtime_error_without_url() -> None:
    with pytest.raises(RuntimeError, match="getLatestBlockhash.*Client") as exc_info:
        await _fetch(_refused_url())

    _assert_no_secret(exc_info.value)
    assert "127.0.0.1" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_url_becomes_runtime_error_without_url() -> None:
    """``InvalidURL``'s own text is the URL."""
    with pytest.raises(RuntimeError, match="getLatestBlockhash") as exc_info:
        await _fetch(f"not-a-url-{_SECRET}")

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_becomes_runtime_error_without_url() -> None:
    http = MagicMock()
    http.post.side_effect = TimeoutError(f"timed out reaching {_SECRET}")

    with pytest.raises(RuntimeError, match="getLatestBlockhash.*TimeoutError") as exc_info:
        await fetch_solana_tx_context(_SECRET, USDC_SOLANA_MAINNET, http=http)

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_failure_on_second_call_names_that_method(rpc_server: Any) -> None:
    """The mint lookup fails after the blockhash read succeeded."""

    async def handler(request: web.Request) -> web.Response:
        rpc = await request.json()
        if rpc["method"] == "getLatestBlockhash":
            return web.json_response({"result": {"value": {"blockhash": str(Hash.default())}}})
        return web.Response(status=503)

    url = await rpc_server(handler)

    with pytest.raises(RuntimeError, match="getAccountInfo.*HTTP 503") as exc_info:
        await _fetch(url)

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_rpc_error_member_still_raises_runtime_error(rpc_server: Any) -> None:
    async def handler(_: web.Request) -> web.Response:
        return web.json_response({"error": {"code": -32005, "message": "rate limited"}})

    url = await rpc_server(handler)

    with pytest.raises(RuntimeError, match="getLatestBlockhash error.*rate limited") as exc_info:
        await _fetch(url)

    _assert_no_secret(exc_info.value)


@pytest.mark.asyncio
async def test_top_up_with_solana_surfaces_runtime_error_and_does_not_settle(
    rpc_server: Any,
) -> None:
    """Through the resource: the documented RuntimeError, no signature, no second POST."""

    async def handler(_: web.Request) -> web.Response:
        return web.Response(status=401)

    url = await rpc_server(handler)
    requirement = {
        "network": "solana",
        "asset": USDC_SOLANA_MAINNET,
        "amount": "5000000",
        "payTo": "8qUL23aSj7mDWdoLMXGHFvnVCT9wd7jXcysiekroADEL",
        "extra": {"feePayer": "BFK9TLC3edb13K6v4YyH3DwPb5DSUpkWvb7XnqCL9b4F"},
    }
    client = MagicMock()
    resource = X402(client)
    probe = PaymentRequiredError(
        "Payment required",
        response=MagicMock(status_code=402),
        body={"x402Version": 2, "accepts": [requirement]},
    )
    resource.top_up = AsyncMock(side_effect=[probe])  # type: ignore[method-assign]
    auth = SolanaX402Auth(private_key=str(Keypair()))

    async with aiohttp.ClientSession() as session:
        client._get_session = AsyncMock(return_value=session)
        with pytest.raises(RuntimeError, match="HTTP 401") as exc_info:
            await resource.top_up_with_solana(auth=auth, amount_usdc=5.0, rpc_url=url)

    _assert_no_secret(exc_info.value)
    assert resource.top_up.call_count == 1


@pytest.mark.asyncio
async def test_successful_rpc_returns_the_context(rpc_server: Any) -> None:
    """The happy path over a real session is unchanged."""
    import base64

    mint_data = bytes(44) + bytes([6]) + bytes(37)

    async def handler(request: web.Request) -> web.Response:
        rpc = await request.json()
        if rpc["method"] == "getLatestBlockhash":
            return web.json_response({"result": {"value": {"blockhash": str(Hash.default())}}})
        return web.json_response(
            {
                "result": {
                    "value": {
                        "owner": _TOKEN_PROGRAM,
                        "data": [base64.b64encode(mint_data).decode("ascii"), "base64"],
                    }
                }
            }
        )

    url = await rpc_server(handler)

    assert await _fetch(url) == (str(Hash.default()), 6, _TOKEN_PROGRAM)
