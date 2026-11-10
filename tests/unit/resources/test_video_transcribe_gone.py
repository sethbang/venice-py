"""``video.transcribe`` against the 410 the spec says the endpoint always answers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.exceptions import ModelGoneError


@asynccontextmanager
async def _gone_server() -> AsyncIterator[tuple[str, list[web.Request]]]:
    seen: list[web.Request] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(request)
        return web.json_response({"error": "This endpoint is no longer available"}, status=410)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}/api/v1", seen
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["json", "text"])
async def test_transcribe_raises_model_gone_once(response_format) -> None:
    async with (
        _gone_server() as (base, seen),
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        with pytest.raises(ModelGoneError):
            await client.video.transcribe(
                "https://example.com/watch?v=x", response_format=response_format
            )
    assert [r.path for r in seen] == ["/api/v1/video/transcriptions"]
