"""Headers meant for the API stay on the API.

Every test drives real SDK call sites through a real aiohttp session against
loopback servers on separate ports, which are separate origins. Each setup
configures a secret a different way, and the same assertions apply to all:

* a host other than the API receives ``Accept``, ``User-Agent`` and what the
  transport itself needs, and nothing else;
* the API still receives the secret, so a pass cannot come from dropping it
  everywhere.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from multidict import CIMultiDict
from yarl import URL

from venice_ai import VeniceClient
from venice_ai.middleware.retry import RetryOptions
from venice_ai.resources.video import VideoJob
from venice_ai.types.api.video import VideoQueueResponse

pytest.importorskip("solders", reason="x402-solana extra not installed")

from solders.hash import Hash  # noqa: E402
from solders.keypair import Keypair  # noqa: E402

from venice_ai.auth.x402_solana import USDC_SOLANA_MAINNET, SolanaX402Auth  # noqa: E402

KEY = "sk-sentinel-key"
SECRET = "sentinel-tenant-secret"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TRANSPORT_HEADERS = {"host", "content-type", "content-length", "accept-encoding"}
ONLY_ACCEPT_AND_USER_AGENT = {"accept", "user-agent"}

REQUIREMENT = {
    "network": "solana",
    "asset": USDC_SOLANA_MAINNET,
    "amount": "5000000",
    "payTo": "8qUL23aSj7mDWdoLMXGHFvnVCT9wd7jXcysiekroADEL",
    "extra": {"feePayer": "BFK9TLC3edb13K6v4YyH3DwPb5DSUpkWvb7XnqCL9b4F"},
}


@dataclass
class Seen:
    """Requests one server received, as ``(method, path, headers)``."""

    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)

    def record(self, request: web.Request) -> None:
        self.requests.append((request.method, request.path, dict(request.headers)))

    def last_names(self) -> set[str]:
        return {name.lower() for name in self.requests[-1][2]} - TRANSPORT_HEADERS

    def assert_each_request_carried_only_accept_and_user_agent(self) -> None:
        assert self.requests, "the host was never reached"
        for method, path, headers in self.requests:
            names = {name.lower() for name in headers} - TRANSPORT_HEADERS
            assert names == ONLY_ACCEPT_AND_USER_AGENT, f"{method} {path} carried {headers}"
            assert SECRET not in str(headers) and KEY not in str(headers)


@asynccontextmanager
async def _serve(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
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


async def _third_party(request: web.Request, seen: Seen) -> web.StreamResponse:
    seen.record(request)
    if request.method == "POST":
        rpc = await request.json()
        if rpc.get("method") == "getLatestBlockhash":
            return web.json_response({"result": {"value": {"blockhash": str(Hash.default())}}})
        mint = base64.b64encode(bytes(44) + bytes([6]) + bytes(37)).decode()
        return web.json_response(
            {"result": {"value": {"owner": TOKEN_PROGRAM, "data": [mint, "base64"]}}}
        )
    if request.method == "DELETE":
        return web.Response(status=204)
    return web.Response(body=b"VIDEO")


async def _api(request: web.Request, seen: Seen, third_party: str) -> web.StreamResponse:
    seen.record(request)
    tail = request.path.removeprefix("/api/v1/")
    if tail == "x402/top-up":
        if "X-402-Payment" in request.headers:
            return web.json_response(
                {
                    "success": True,
                    "data": {
                        "walletAddress": "w",
                        "amountCredited": 5.0,
                        "newBalance": 5.0,
                        "paymentId": "p",
                    },
                }
            )
        return web.json_response({"x402Version": 2, "accepts": [REQUIREMENT]}, status=402)
    if tail == "cross":
        raise web.HTTPFound(f"{third_party}/landed")
    if tail == "cross-keep-method":
        raise web.HTTPTemporaryRedirect(f"{third_party}/landed")
    if tail == "same":
        raise web.HTTPFound("/api/v1/echo")
    if tail == "file":
        return web.Response(body=b"ON THE API")
    if tail == "chat/completions":
        chunk = {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        await stream.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
        await stream.write_eof()
        return stream
    if tail == "audio/speech":
        return web.Response(body=b"AUDIO", content_type="audio/mpeg")
    if tail == "billing/balance":
        return web.json_response({"canConsume": True})
    if tail == "multipart":
        await request.post()
    return web.json_response({"ok": True})


@dataclass
class World:
    """The API, a third-party host, and a client configured by one setup."""

    client: VeniceClient
    api: Seen
    host: Seen
    api_url: str
    host_url: str
    secret_headers_on_api: dict[str, str]


Setup = Callable[[str], Any]


@asynccontextmanager
async def _client_headers(base_url: str) -> AsyncIterator[tuple[VeniceClient, dict[str, str]]]:
    async with VeniceClient(
        api_key=KEY, base_url=base_url, headers={"X-Tenant-Secret": SECRET}
    ) as c:
        yield c, {"X-Tenant-Secret": SECRET, "Authorization": f"Bearer {KEY}"}


@asynccontextmanager
async def _authorization_in_client_headers(
    base_url: str,
) -> AsyncIterator[tuple[VeniceClient, dict[str, str]]]:
    auth = {"Authorization": f"Bearer {SECRET}"}
    async with VeniceClient(api_key="placeholder", base_url=base_url, headers=auth) as c:
        yield c, auth


@asynccontextmanager
async def _caller_session_defaults(
    base_url: str,
) -> AsyncIterator[tuple[VeniceClient, dict[str, str]]]:
    defaults = {"Authorization": f"Bearer {SECRET}", "X-Gateway-Token": SECRET}
    async with (
        aiohttp.ClientSession(headers=defaults) as session,
        VeniceClient(api_key=KEY, base_url=base_url, http_client=session) as c,
    ):
        yield c, {"X-Gateway-Token": SECRET, "Authorization": f"Bearer {KEY}"}


@asynccontextmanager
async def _client_headers_on_caller_session(
    base_url: str,
) -> AsyncIterator[tuple[VeniceClient, dict[str, str]]]:
    async with (
        aiohttp.ClientSession() as session,
        VeniceClient(
            api_key=KEY, base_url=base_url, http_client=session, headers={"X-Tenant-Secret": SECRET}
        ) as c,
    ):
        yield c, {"X-Tenant-Secret": SECRET, "Authorization": f"Bearer {KEY}"}


SETUPS = {
    "client-headers": _client_headers,
    "authorization-in-client-headers": _authorization_in_client_headers,
    "caller-session-defaults": _caller_session_defaults,
    "client-headers-on-caller-session": _client_headers_on_caller_session,
}


@pytest_asyncio.fixture(loop_scope="function", params=sorted(SETUPS))
async def world(request: pytest.FixtureRequest) -> AsyncIterator[World]:
    api = Seen()
    host = Seen()

    async def host_handler(r: web.Request) -> web.StreamResponse:
        return await _third_party(r, host)

    async with _serve(host_handler) as host_url:

        async def api_handler(r: web.Request) -> web.StreamResponse:
            return await _api(r, api, host_url)

        async with (
            _serve(api_handler) as api_url,
            SETUPS[request.param](f"{api_url}/api/v1") as (client, secrets),
        ):
            yield World(client, api, host, api_url, host_url, secrets)


def _assert_api_received_the_secret(world: World) -> None:
    assert world.api.requests, "the API was never reached"
    headers = world.api.requests[-1][2]
    for name, value in world.secret_headers_on_api.items():
        assert headers.get(name) == value, f"{name} missing from {headers}"


# ---------------------------------------------------------------------------
# The three calls the SDK makes to other hosts
# ---------------------------------------------------------------------------


async def test_top_up_rpc_call_carries_no_secret(world: World) -> None:
    auth = SolanaX402Auth(private_key=str(Keypair()))

    result = await world.client.x402.top_up_with_solana(
        auth=auth, amount_usdc=5.0, rpc_url=f"{world.host_url}/rpc"
    )

    assert result.success is True
    assert [r[0] for r in world.host.requests] == ["POST", "POST"]
    world.host.assert_each_request_carried_only_accept_and_user_agent()
    _assert_api_received_the_secret(world)


async def test_fetch_external_on_another_host_carries_no_secret(world: World) -> None:
    body = await world.client.fetch_external(f"{world.host_url}/link?sig=1")

    assert body == b"VIDEO"
    world.host.assert_each_request_carried_only_accept_and_user_agent()


async def test_download_link_delete_carries_no_secret(world: World) -> None:
    url = f"{world.host_url}/link?sig=1"
    job = VideoJob(world.client, VideoQueueResponse(model="m", queue_id="q", download_url=url))

    assert await job._delete_download_url(url) is True

    assert world.host.requests[-1][0] == "DELETE"
    world.host.assert_each_request_carried_only_accept_and_user_agent()


# ---------------------------------------------------------------------------
# Redirects away from the API
# ---------------------------------------------------------------------------


async def test_cross_origin_redirect_from_api_request_carries_no_secret(world: World) -> None:
    await world.client.get("cross", raw_response=True)

    _assert_api_received_the_secret(world)
    world.host.assert_each_request_carried_only_accept_and_user_agent()


async def test_cross_origin_redirect_from_api_post_carries_no_secret(world: World) -> None:
    response = await world.client.post("cross-keep-method", raw_response=True, json_data={"a": 1})
    response.release()

    _assert_api_received_the_secret(world)
    # This host answers any POST as JSON-RPC; only the headers matter here.
    world.host.assert_each_request_carried_only_accept_and_user_agent()


async def test_fetch_external_redirected_off_the_api_carries_no_secret(world: World) -> None:
    body = await world.client.fetch_external(f"{world.api_url}/api/v1/cross")

    assert body == b"VIDEO"
    _assert_api_received_the_secret(world)
    world.host.assert_each_request_carried_only_accept_and_user_agent()


async def test_same_origin_redirect_keeps_the_secret(world: World) -> None:
    await world.client.get("same")

    assert world.api.requests[-1][1] == "/api/v1/echo"
    _assert_api_received_the_secret(world)


async def test_fetch_external_on_the_api_keeps_the_secret(world: World) -> None:
    assert await world.client.fetch_external(f"{world.api_url}/api/v1/file") == b"ON THE API"

    _assert_api_received_the_secret(world)


# ---------------------------------------------------------------------------
# Every path that sends an API request still authenticates it
# ---------------------------------------------------------------------------


async def test_json_request_keeps_the_secret(world: World) -> None:
    await world.client.get("echo")

    _assert_api_received_the_secret(world)


async def test_stream_keeps_the_secret(world: World) -> None:
    stream = await world.client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    chunks = [chunk async for chunk in stream]

    assert chunks[0].choices[0].delta.content == "hi"
    _assert_api_received_the_secret(world)


async def test_multipart_keeps_the_secret(world: World) -> None:
    await world.client.audio._request_multipart(
        "POST", "multipart", files={"f": ("a.bin", b"x", "application/octet-stream")}
    )

    _assert_api_received_the_secret(world)


async def test_billing_keeps_the_secret(world: World) -> None:
    await world.client.billing.get_balance()

    _assert_api_received_the_secret(world)


async def test_streamed_audio_keeps_the_secret(world: World) -> None:
    stream = await world.client.audio.create_speech(model="m", input="hi", voice="v", stream=True)
    assert b"".join([chunk async for chunk in stream]) == b"AUDIO"

    _assert_api_received_the_secret(world)


# ---------------------------------------------------------------------------
# Details of what is removed and what stays
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _pair() -> AsyncIterator[tuple[Seen, Seen, str, str]]:
    api = Seen()
    host = Seen()

    async def host_handler(r: web.Request) -> web.StreamResponse:
        return await _third_party(r, host)

    async with _serve(host_handler) as host_url:

        async def api_handler(r: web.Request) -> web.StreamResponse:
            return await _api(r, api, host_url)

        async with _serve(api_handler) as api_url:
            yield api, host, api_url, host_url


async def test_sdk_session_holds_no_default_custom_header() -> None:
    async with VeniceClient(
        api_key=KEY, base_url="http://127.0.0.1:1/api/v1", headers={"X-Tenant-Secret": SECRET}
    ) as client:
        session = await client._get_session()

        assert "X-Tenant-Secret" not in session.headers
        assert "Authorization" not in session.headers


async def test_headers_given_as_a_multidict_reach_the_api() -> None:
    async with _pair() as (api, _host, api_url, _host_url):
        headers: Any = CIMultiDict({"X-Tenant-Secret": SECRET})
        async with VeniceClient(api_key=KEY, base_url=f"{api_url}/api/v1", headers=headers) as c:
            await c.get("echo")

        assert api.requests[-1][2]["X-Tenant-Secret"] == SECRET


async def test_wallet_signature_does_not_follow_a_redirect_off_the_api() -> None:
    wallet = SolanaX402Auth(private_key=str(Keypair()))
    async with _pair() as (api, host, api_url, _host_url):
        async with VeniceClient(api_key="", auth=wallet, base_url=f"{api_url}/api/v1") as c:
            response = await c.get("cross", raw_response=True)
            response.release()

        assert "X-Sign-In-With-X" in api.requests[-1][2]
        host.assert_each_request_carried_only_accept_and_user_agent()


async def test_retried_request_to_another_host_carries_no_retry_counter() -> None:
    attempts = {"n": 0}
    host = Seen()

    async def flaky(request: web.Request) -> web.Response:
        host.record(request)
        attempts["n"] += 1
        return web.Response(status=503) if attempts["n"] == 1 else web.Response(body=b"VIDEO")

    async with (
        _serve(flaky) as host_url,
        VeniceClient(
            api_key=KEY,
            base_url="http://127.0.0.1:1/api/v1",
            retry_options=RetryOptions(max_attempts=1, base_delay=0.001, max_delay=0.002),
        ) as client,
    ):
        assert await client.fetch_external(f"{host_url}/link") == b"VIDEO"

    assert attempts["n"] == 2
    host.assert_each_request_carried_only_accept_and_user_agent()


@pytest.mark.parametrize("kind", ["sdk-session", "caller-session"])
@pytest.mark.parametrize("call", ["top_up_rpc", "fetch_external", "link_delete"])
async def test_credentials_in_a_url_still_reach_that_host(kind: str, call: str) -> None:
    """A URL's own credentials were put there for that host, unlike a session default."""
    async with _pair() as (_api, host, api_url, host_url):
        url = str(URL(host_url).with_user("rpcuser").with_password("rpcpass") / "rpc")
        base_url = f"{api_url}/api/v1"

        async def send(client: VeniceClient) -> None:
            if call == "fetch_external":
                await client.fetch_external(url)
            elif call == "link_delete":
                job = VideoJob(
                    client, VideoQueueResponse(model="m", queue_id="q", download_url=url)
                )
                await job._delete_download_url(url)
            else:
                wallet = SolanaX402Auth(private_key=str(Keypair()))
                await client.x402.top_up_with_solana(auth=wallet, amount_usdc=5.0, rpc_url=url)

        if kind == "sdk-session":
            async with VeniceClient(
                api_key=KEY, base_url=base_url, headers={"X-Tenant-Secret": SECRET}
            ) as client:
                await send(client)
        else:
            async with (
                aiohttp.ClientSession(headers={"X-Gateway-Token": SECRET}) as session,
                VeniceClient(api_key=KEY, base_url=base_url, http_client=session) as client,
            ):
                await send(client)

        headers = host.requests[-1][2]
        expected = base64.b64encode(b"rpcuser:rpcpass").decode()
        assert headers["Authorization"] == f"Basic {expected}"
        assert SECRET not in str(headers)


async def test_shared_cookie_on_a_caller_session_stays_off_another_host() -> None:
    async with _pair() as (api, host, api_url, host_url):
        async with (
            aiohttp.ClientSession(cookies={"sid": SECRET}) as session,
            VeniceClient(api_key=KEY, base_url=f"{api_url}/api/v1", http_client=session) as client,
        ):
            await client.get("echo")
            await client.fetch_external(f"{host_url}/link")

        assert SECRET in api.requests[-1][2]["Cookie"]
        host.assert_each_request_carried_only_accept_and_user_agent()


async def test_caller_session_middleware_still_runs_and_cannot_add_to_a_third_party_call() -> None:
    ran: list[str] = []

    async def caller_middleware(request: Any, handler: Any) -> Any:
        ran.append(str(request.url.path))
        request.headers["X-Signed-By-Caller"] = SECRET
        return await handler(request)

    async with _pair() as (api, host, api_url, host_url):
        async with (
            aiohttp.ClientSession(middlewares=(caller_middleware,)) as session,
            VeniceClient(api_key=KEY, base_url=f"{api_url}/api/v1", http_client=session) as client,
        ):
            await client.get("echo")
            await client.fetch_external(f"{host_url}/link")

        assert ran == ["/api/v1/echo", "/link"]
        assert api.requests[-1][2]["X-Signed-By-Caller"] == SECRET
        host.assert_each_request_carried_only_accept_and_user_agent()


async def test_cookie_a_host_set_for_itself_is_not_sent_back_to_it() -> None:
    host = Seen()

    async def sets_a_cookie(request: web.Request) -> web.Response:
        host.record(request)
        response = web.Response(body=b"VIDEO")
        response.set_cookie("own", "value")
        return response

    jar = aiohttp.CookieJar(unsafe=True)
    async with (
        _serve(sets_a_cookie) as host_url,
        VeniceClient(api_key=KEY, base_url="http://127.0.0.1:1/api/v1", cookie_jar=jar) as client,
    ):
        await client.fetch_external(f"{host_url}/link")
        await client.fetch_external(f"{host_url}/link")

    assert len(jar) == 1
    host.assert_each_request_carried_only_accept_and_user_agent()


async def test_netrc_credentials_are_not_sent_to_another_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login user password netrc-password\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    host = Seen()

    async def serve_file(request: web.Request) -> web.Response:
        host.record(request)
        return web.Response(body=b"VIDEO")

    async with (
        _serve(serve_file) as host_url,
        VeniceClient(api_key=KEY, base_url="http://127.0.0.2:1/api/v1", trust_env=True) as client,
    ):
        assert await client.fetch_external(f"{host_url}/link") == b"VIDEO"

    host.assert_each_request_carried_only_accept_and_user_agent()


async def test_proxy_credentials_still_reach_the_proxy() -> None:
    """The proxy's own header is added after the filter, for a request to any host."""
    seen: list[bytes] = []

    async def forward_proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nVIDEO")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(forward_proxy, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with VeniceClient(
            api_key=KEY,
            base_url="http://127.0.0.1:1/api/v1",
            headers={"X-Tenant-Secret": SECRET},
            proxy=f"http://user:pass@127.0.0.1:{port}",
        ) as client:
            assert await client.fetch_external("http://127.0.0.1:2/link") == b"VIDEO"
    finally:
        server.close()

    expected = base64.b64encode(b"user:pass")
    assert b"Proxy-Authorization: Basic " + expected in seen[0]
    assert SECRET.encode() not in seen[0]
    assert KEY.encode() not in seen[0]


async def test_caller_session_is_left_unchanged() -> None:
    async def caller_middleware(request: Any, handler: Any) -> Any:
        return await handler(request)

    async with (
        _pair() as (_api, _host, api_url, host_url),
        aiohttp.ClientSession(
            headers={"X-Gateway-Token": SECRET}, middlewares=(caller_middleware,)
        ) as session,
        VeniceClient(api_key=KEY, base_url=f"{api_url}/api/v1", http_client=session) as client,
    ):
        await client.fetch_external(f"{host_url}/link")

        assert session.headers["X-Gateway-Token"] == SECRET
        assert session._middlewares == (caller_middleware,)
