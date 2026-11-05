"""``video.retrieve`` reports a finished video only for a body that is one.

Real loopback HTTP, so the Content-Type and body are what aiohttp delivers. A
gateway's HTML page or a status body served as text must not read as a finished
video, or ``job.wait()`` returns and ``job.download()`` writes the page to disk.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web
from click.testing import CliRunner

from venice_ai import VeniceClient
from venice_ai.cli.cli import cli
from venice_ai.exceptions import APIResponseProcessingError
from venice_ai.resources.video import VideoJob
from venice_ai.types.api.video import VideoCompletedStatus, VideoQueueResponse
from venice_ai.utils.errors import ResponseSummary

LINK_SECRET = "SENTINELLINKTOKEN"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16
MOV = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 16
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 16
HTML = f"<html>login {LINK_SECRET}</html>".encode()
STATUS_JSON = f'{{"status":"COMPLETED","url":"https://cdn.test/v?sig={LINK_SECRET}"}}'.encode()

REJECTED = {
    "html": ("text/html", HTML),
    "json-as-text": ("text/plain", STATUS_JSON),
    "empty-video": ("video/mp4", b""),
    "empty-octet-stream": ("application/octet-stream", b""),
    "html-as-octet-stream": ("application/octet-stream", HTML),
    "json-as-octet-stream": ("application/octet-stream", STATUS_JSON),
    "audio-as-octet-stream": ("application/octet-stream", b"fLaC" + b"\x00" * 16),
    "image": ("image/png", b"\x89PNG\r\n"),
}
ACCEPTED = {
    "mp4": ("video/mp4", MP4),
    "webm": ("video/webm", WEBM),
    "with-parameters": ("video/mp4; codecs=avc1", MP4),
    "upper-case": ("VIDEO/MP4", MP4),
    "mp4-as-octet-stream": ("application/octet-stream", MP4),
    "mov-as-octet-stream": ("application/octet-stream", MOV),
    "webm-as-octet-stream": ("application/octet-stream", WEBM),
    "fragmented-as-octet-stream": (
        "application/octet-stream",
        b"\x00\x00\x00\x18styp" + b"\x00" * 16,
    ),
    "s3-default-type": ("binary/octet-stream", MP4),
    "application-mp4": ("application/mp4", MP4),
    "no-type": ("", MP4),
}


def _retrieve_handler(content_type: str, body: bytes) -> Any:
    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith("/video/queue"):
            return web.json_response({"model": "m", "queue_id": "q"})
        if request.path.endswith("/video/complete"):
            return web.json_response({"success": True})
        return web.Response(body=body, headers={"Content-Type": content_type})

    return handler


@asynccontextmanager
async def _client(content_type: str, body: bytes) -> AsyncIterator[VeniceClient]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", _retrieve_handler(content_type, body))
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


@pytest.mark.parametrize("case", sorted(ACCEPTED))
async def test_a_video_body_is_the_finished_video(case: str) -> None:
    content_type, body = ACCEPTED[case]
    async with _client(content_type, body) as client:
        status = await client.video.retrieve(model="m", queue_id="q")

    assert isinstance(status, VideoCompletedStatus)
    assert status.data == body


@pytest.mark.parametrize("case", sorted(REJECTED))
async def test_any_other_body_is_an_error_that_shows_none_of_it(case: str) -> None:
    content_type, body = REJECTED[case]
    async with _client(content_type, body) as client:
        with pytest.raises(APIResponseProcessingError) as exc_info:
            await client.video.retrieve(model="m", queue_id="q")

    error = exc_info.value
    assert content_type.split(";")[0] in str(error)
    assert LINK_SECRET not in f"{error!s} {error!r} {error.args!r}"
    assert isinstance(error.response, ResponseSummary)
    assert error.response.status == 200
    assert error.__cause__ is None
    assert error.__context__ is None


async def _retrieve_with_no_content_type(body: bytes) -> Any:
    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        await reader.read(1)
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with VeniceClient(api_key="sk-fake", base_url=f"http://127.0.0.1:{port}/api/v1") as c:
            return await c.video.retrieve(model="m", queue_id="q")
    finally:
        server.close()


async def test_a_body_with_no_content_type_that_is_not_video_is_an_error() -> None:
    with pytest.raises(APIResponseProcessingError, match="Content-Type"):
        await _retrieve_with_no_content_type(HTML)


async def test_a_video_with_no_content_type_is_accepted() -> None:
    status = await _retrieve_with_no_content_type(MP4)
    assert status.data == MP4


async def test_job_wait_reports_the_error_and_nothing_is_written(tmp_path: Path) -> None:
    async with _client("text/html", HTML) as client:
        job = VideoJob(client, VideoQueueResponse(model="m", queue_id="q"))

        with pytest.raises(APIResponseProcessingError):
            await job.wait(poll_interval=0, max_polls=2)

    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def api_in_a_thread() -> Iterator[Any]:
    """Serve from another thread, because ``CliRunner`` runs its own event loop."""
    loop = asyncio.new_event_loop()
    runners: list[web.AppRunner] = []

    def start(content_type: str, body: bytes) -> str:
        async def serve() -> str:
            app = web.Application()
            app.router.add_route("*", "/{tail:.*}", _retrieve_handler(content_type, body))
            runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
            await runner.setup()
            runners.append(runner)
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            return f"http://127.0.0.1:{port}/api/v1"

        return asyncio.run_coroutine_threadsafe(serve(), loop).result(timeout=5)

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield start
    for runner in runners:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.mark.parametrize(
    ("content_type", "body", "expected_exit", "saved"),
    [
        ("text/html", HTML, 1, False),
        ("text/plain", STATUS_JSON, 1, False),
        ("video/mp4", b"", 1, False),
        ("video/mp4", MP4, 0, True),
    ],
    ids=["html", "json-as-text", "empty", "video"],
)
def test_video_generate_saves_only_a_video(
    tmp_path: Path,
    api_in_a_thread: Any,
    content_type: str,
    body: bytes,
    expected_exit: int,
    saved: bool,
) -> None:
    base_url = api_in_a_thread(content_type, body)
    out = tmp_path / "out.mp4"

    with (
        patch(
            "venice_ai.cli.commands.video.get_client_kwargs",
            return_value={"api_key": "sk-fake", "base_url": base_url},
            create=True,
        ),
        patch(
            "venice_ai.cli.config.get_client_kwargs",
            return_value={"api_key": "sk-fake", "base_url": base_url},
        ),
    ):
        result = CliRunner().invoke(
            cli, ["video", "generate", "a prompt", "-m", "m", "-o", str(out)]
        )

    assert result.exit_code == expected_exit, result.output
    assert LINK_SECRET not in result.output
    assert ("Video saved" in result.output) is saved
    assert out.exists() is saved
