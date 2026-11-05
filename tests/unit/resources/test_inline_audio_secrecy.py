"""An error about an inline music or voice-changer body shows none of that body.

A status body can hold a download link, which grants access on its own, whatever
Content-Type it is served with.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.exceptions import APIResponseProcessingError
from venice_ai.utils.errors import ResponseSummary

LINK_SECRET = "SENTINELLINKTOKEN"
BODY = f'{{"status":"COMPLETED","url":"https://cdn.test/a?sig={LINK_SECRET}"}}'.encode()


@asynccontextmanager
async def _client(content_type: str, body: bytes) -> AsyncIterator[VeniceClient]:
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(body=body, headers={"Content-Type": content_type})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        async with VeniceClient(api_key="sk-fake", base_url=f"http://127.0.0.1:{port}/api/v1") as c:
            yield c
    finally:
        await runner.cleanup()


async def _retrieve(client: VeniceClient, resource: str) -> Any:
    if resource == "music":
        return await client.music.retrieve(model="m", queue_id="q")
    return await client.voice_changer.retrieve(model="m", queue_id="q")


@pytest.mark.parametrize("resource", ["music", "voice_changer"])
@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("text/plain", BODY),
        ("application/octet-stream", BODY),
        ("audio/mpeg", b""),
    ],
    ids=["text", "octet-stream", "empty-audio"],
)
async def test_processing_error_shows_no_body_and_keeps_none(
    resource: str, content_type: str, body: bytes
) -> None:
    async with _client(content_type, body) as client:
        with pytest.raises(APIResponseProcessingError) as exc_info:
            await _retrieve(client, resource)

    error = exc_info.value
    assert LINK_SECRET not in f"{error!s} {error!r} {error.args!r}"
    assert "preview" not in str(error)
    assert isinstance(error.response, ResponseSummary)
    assert not hasattr(error.response, "read")
    assert error.response.status == 200


async def test_voice_changer_parse_failure_keeps_the_body_out_of_logs_and_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    malformed = f'{{"status":"COMPLETED","url":"https://cdn.test/a?sig={LINK_SECRET}"'.encode()

    async with _client("application/json", malformed) as client:
        with pytest.raises(ValueError) as exc_info:
            await client.voice_changer.retrieve(model="m", queue_id="q")

    error = exc_info.value
    assert LINK_SECRET not in f"{error!s} {error!r} {getattr(error, 'doc', '')}"
    for record in caplog.records:
        assert LINK_SECRET not in record.getMessage()
