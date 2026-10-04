"""One API-key precedence, and authentication on every request path.

The key resolves in one order across every entry point: an explicit
``api_key=``, then ``config.api_key``, then ``VENICE_API_KEY``. Whatever
session the client sends through -- its own, or one the caller passed as
``http_client`` -- each API request carries the resolved credentials, and
absolute media URLs fetched through that session never do.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from venice_ai import SyncVeniceClient, VeniceClient
from venice_ai.core.auth import resolve_api_key
from venice_ai.core.config import VeniceAIConfig
from venice_ai.factory import VeniceClientFactory


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VENICE_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


def test_config_api_key_authenticates_the_client() -> None:
    client = VeniceClient(config=VeniceAIConfig(api_key="sk-from-config"))
    assert client._api_key == "sk-from-config"


def test_factory_uses_config_api_key() -> None:
    client = VeniceClientFactory.create_client(config=VeniceAIConfig(api_key="sk-from-config"))
    assert client._api_key == "sk-from-config"


def test_sync_client_uses_config_api_key() -> None:
    client = SyncVeniceClient(config=VeniceAIConfig(api_key="sk-from-config"))
    try:
        assert client._async_client._api_key == "sk-from-config"
    finally:
        client.close()


def test_explicit_api_key_beats_config_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VENICE_API_KEY", "sk-from-env")
    config = VeniceAIConfig(api_key="sk-from-config")
    assert VeniceClient(api_key="sk-explicit", config=config)._api_key == "sk-explicit"


def test_config_api_key_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = VeniceAIConfig(api_key="sk-from-config")
    monkeypatch.setenv("VENICE_API_KEY", "sk-from-env")
    assert VeniceClient(config=config)._api_key == "sk-from-config"


def test_env_is_the_fallback_when_config_has_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    config = VeniceAIConfig()
    assert config.api_key is None
    monkeypatch.setenv("VENICE_API_KEY", "sk-from-env")
    assert VeniceClient(config=config)._api_key == "sk-from-env"


def test_explicit_empty_key_stops_the_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VENICE_API_KEY", "sk-from-env")
    assert resolve_api_key("", VeniceAIConfig(api_key="sk-from-config")) == ""


def test_no_key_anywhere_is_rejected() -> None:
    with pytest.raises(ValueError, match="config"):
        VeniceClient(config=VeniceAIConfig())


# ---------------------------------------------------------------------------
# Per-request authentication over a local server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _recording_server() -> AsyncIterator[tuple[str, list[web.Request]]]:
    seen: list[web.Request] = []

    async def record(request: web.Request) -> web.Response:
        await request.read()
        seen.append(request)
        if request.path.endswith("/audio/speech"):
            return web.Response(body=b"\x00" * 32, content_type="audio/mpeg")
        if request.path.endswith(".bin"):
            return web.Response(body=b"asset", content_type="application/octet-stream")
        return web.json_response({"object": "list", "data": []})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", record)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", seen
    finally:
        await runner.cleanup()


async def _exercise(client: VeniceClient, api_host: str, cdn_host: str) -> None:
    """Send a JSON request, a multipart upload, a streamed speech request and two media fetches."""
    await client.get("models")
    await client.image._request_multipart(
        method="POST", path="image/upscale", files={"image": ("a.png", b"\x89PNG", "image/png")}
    )
    async for _ in client.audio._stream_audio_bytes("POST", "audio/speech", json_data={}):
        pass
    await client.fetch_external(f"{api_host}/media/same-origin.bin")
    await client.fetch_external(f"{cdn_host}/cdn.bin")


def _api_requests(seen: list[web.Request]) -> list[web.Request]:
    return [r for r in seen if r.path.startswith("/api/v1/") or r.path.startswith("/media/")]


@pytest.mark.parametrize("own_session", [False, True], ids=["sdk-session", "caller-session"])
async def test_every_api_request_carries_the_key_and_other_hosts_do_not(
    own_session: bool,
) -> None:
    async with _recording_server() as (api_host, seen), _recording_server() as (cdn_host, cdn):
        session = aiohttp.ClientSession() if own_session else None
        try:
            async with VeniceClient(
                api_key="sk-test",
                base_url=f"{api_host}/api/v1",
                http_client=session,
                headers={"X-Client-Tag": "t1"},
            ) as client:
                await _exercise(client, api_host, cdn_host)
        finally:
            if session is not None:
                assert "Authorization" not in session.headers
                await session.close()

    assert [r.path for r in _api_requests(seen)] == [
        "/api/v1/models",
        "/api/v1/image/upscale",
        "/api/v1/audio/speech",
        "/media/same-origin.bin",
    ]
    for request in _api_requests(seen):
        assert request.headers.getall("Authorization") == ["Bearer sk-test"], request.path
        assert request.headers.get("X-Client-Tag") == "t1", request.path
    (cdn_request,) = cdn
    assert "Authorization" not in cdn_request.headers


class _Wallet:
    """A wallet auth double: each call signs a fresh envelope, as a real one does."""

    def __init__(self) -> None:
        self.signed = 0

    def build_header(self) -> str:
        self.signed += 1
        return f"siwe-envelope-{self.signed}"


@pytest.mark.parametrize("own_session", [False, True], ids=["sdk-session", "caller-session"])
async def test_wallet_only_client_signs_every_api_request(
    own_session: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VENICE_API_KEY", "sk-must-not-be-used")
    wallet = _Wallet()
    async with _recording_server() as (api_host, seen), _recording_server() as (cdn_host, _):
        session = aiohttp.ClientSession() if own_session else None
        try:
            async with VeniceClient(
                api_key="",
                auth=wallet,  # type: ignore[arg-type]
                base_url=f"{api_host}/api/v1",
                http_client=session,
            ) as client:
                await _exercise(client, api_host, cdn_host)
        finally:
            if session is not None:
                await session.close()

    envelopes = []
    for request in _api_requests(seen):
        assert "Authorization" not in request.headers, request.path
        envelopes.append(request.headers["X-Sign-In-With-X"])
    assert len(set(envelopes)) == len(envelopes) == 4


async def test_per_call_authorization_header_overrides_the_client_key() -> None:
    async with (
        _recording_server() as (base, seen),
        VeniceClient(api_key="sk-test", base_url=f"{base}/api/v1") as client,
    ):
        await client.get("models", headers={"authorization": "Bearer sk-other"})
    assert seen[0].headers.getall("Authorization") == ["Bearer sk-other"]
