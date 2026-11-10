"""Job objects returned through ``SyncVeniceClient`` run without ``await``.

Coroutine methods block, properties pass through, and ``with job:`` keeps the
release rule of the async context manager: a clean exit releases the job's
stored media, and an exception in the block reaches ``__aexit__`` and releases
nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import inspect
import logging
import pkgutil
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web

import venice_ai.resources
from venice_ai import SyncVeniceClient
from venice_ai._sync_client import _wrap_result
from venice_ai.exceptions import VideoGenerationError
from venice_ai.resources.image import ImageJob
from venice_ai.resources.music import MusicJob
from venice_ai.resources.video import VideoJob
from venice_ai.resources.voice_changer import VoiceChangerJob
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicProcessingStatus,
    MusicQueueResponse,
)
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
)
from venice_ai.types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerCompleteResponse,
    VoiceChangerProcessingStatus,
    VoiceChangerQueueResponse,
)

UNREACHABLE = "http://127.0.0.1:9/api/v1"
MP4 = b"\x00\x00\x00\x18ftypmp42"

_RELEASE_METHOD = {"video": "cancel", "music": "release", "voice_changer": "cancel"}


def _backend() -> Mock:
    """What a job calls: its resource's retrieve and release methods."""
    backend = Mock()
    for name, release in _RELEASE_METHOD.items():
        resource = Mock()
        resource.retrieve = AsyncMock()
        setattr(resource, release, AsyncMock())
        setattr(backend, name, resource)
    backend.video.cancel.return_value = VideoCompleteResponse(success=True)
    backend.music.release.return_value = MusicCompleteResponse(success=True)
    backend.voice_changer.cancel.return_value = VoiceChangerCompleteResponse(success=True)
    return backend


def _with_body(status: Any) -> Any:
    status._set_data(MP4)
    return status


# job factory, processing status, completed status carrying a body
JOBS: dict[str, tuple[Callable[[Mock], Any], Any, Any]] = {
    "video": (
        lambda b: VideoJob(b, VideoQueueResponse(model="stub-model", queue_id="q-video")),
        VideoProcessingStatus(
            status="PROCESSING", average_execution_time=1000.0, execution_duration=250.0
        ),
        _with_body(VideoCompletedStatus(status="COMPLETED", url=None, expires_at=None)),
    ),
    "music": (
        lambda b: MusicJob(b, MusicQueueResponse(model="stub-model", queue_id="q-music")),
        MusicProcessingStatus(
            status="PROCESSING", average_execution_time=1000.0, execution_duration=250.0
        ),
        _with_body(MusicCompletedStatus(status="COMPLETED", url=None)),
    ),
    "voice_changer": (
        lambda b: VoiceChangerJob(
            b,
            VoiceChangerQueueResponse(
                model="stub-model", queue_id="q-voice", status="QUEUED", duration_seconds=3.0
            ),
        ),
        VoiceChangerProcessingStatus(status="PROCESSING"),
        _with_body(VoiceChangerCompletedStatus(status="COMPLETED")),
    ),
}


@pytest.fixture
def client() -> Iterator[SyncVeniceClient]:
    sync_client = SyncVeniceClient(api_key="fake-key", base_url=UNREACHABLE)
    yield sync_client
    sync_client.close()


def _start(client: SyncVeniceClient, kind: str) -> tuple[Any, Mock]:
    """Return a job through ``client.<kind>.run``, and the backend it calls."""
    backend = _backend()
    make, processing, completed = JOBS[kind]
    getattr(backend, kind).retrieve.side_effect = [processing, completed]
    resource = getattr(client._async_client, kind)
    resource.run = AsyncMock(return_value=make(backend))
    return getattr(client, kind).run(model="stub-model"), backend


def _release(backend: Mock, kind: str) -> AsyncMock:
    mock: AsyncMock = getattr(getattr(backend, kind), _RELEASE_METHOD[kind])
    return mock


# ---------------------------------------------------------------------------
# Driving a job
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", JOBS)
def test_job_runs_to_a_saved_file_and_releases_on_clean_exit(
    client: SyncVeniceClient, kind: str, tmp_path: Path
) -> None:
    job, backend = _start(client, kind)

    with job as entered:
        assert entered is job
        status = job.wait(poll_interval=0)
        saved = job.download(tmp_path / "out.bin", status)
        assert job.is_complete is True
        _release(backend, kind).assert_not_awaited()

    assert saved.read_bytes() == MP4
    _release(backend, kind).assert_awaited_once()


@pytest.mark.parametrize("kind", JOBS)
def test_properties_and_fields_pass_through_live(client: SyncVeniceClient, kind: str) -> None:
    job, _backend_mock = _start(client, kind)
    _make, processing, _completed = JOBS[kind]

    assert job.model == "stub-model"
    assert job.queue_id == f"q-{'voice' if kind == 'voice_changer' else kind}"
    assert job.status is None
    assert job.is_complete is False

    assert job.poll() == processing

    assert job.status == processing
    assert job.progress is not None


@pytest.mark.parametrize("kind", JOBS)
def test_progress_callback_runs_on_the_loop_thread(client: SyncVeniceClient, kind: str) -> None:
    job, _backend_mock = _start(client, kind)
    threads: list[threading.Thread] = []

    job.wait(
        poll_interval=0, on_progress=lambda _status: threads.append(threading.current_thread())
    )

    assert threads == [client._thread]


@pytest.mark.timeout(20)
def test_calling_the_client_from_a_progress_callback_raises_instead_of_hanging(
    client: SyncVeniceClient,
) -> None:
    job, _backend_mock = _start(client, "video")

    with pytest.raises(RuntimeError, match="loop thread"):
        job.wait(poll_interval=0, on_progress=lambda _status: job.poll())


def test_image_job_renders_once_and_saves(client: SyncVeniceClient, tmp_path: Path) -> None:
    backend = Mock()
    backend.image.create = AsyncMock(return_value=b"PNG-BYTES")
    client._async_client.image.submit = AsyncMock(  # type: ignore[method-assign]
        return_value=ImageJob(backend, {"model": "stub-model", "prompt": "x"})
    )

    with client.image.submit(model="stub-model", prompt="x") as job:
        assert job.is_complete is False
        assert job.wait() == b"PNG-BYTES"
        saved = job.download(tmp_path / "out.png")
        assert job.is_complete is True

    assert saved.read_bytes() == b"PNG-BYTES"
    backend.image.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Leaving the block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", JOBS)
def test_exception_in_block_propagates_and_releases_nothing(
    client: SyncVeniceClient, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    job, backend = _start(client, kind)

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="disk full"), job:
        job.wait(poll_interval=0)
        raise ValueError("disk full")

    _release(backend, kind).assert_not_awaited()
    assert job.queue_id in caplog.text
    assert "ValueError" in caplog.text


@pytest.mark.parametrize("kind", JOBS)
def test_keyboard_interrupt_in_block_releases_nothing(client: SyncVeniceClient, kind: str) -> None:
    job, backend = _start(client, kind)

    with pytest.raises(KeyboardInterrupt), job:
        job.wait(poll_interval=0)
        raise KeyboardInterrupt

    _release(backend, kind).assert_not_awaited()


def test_failed_job_is_released_even_though_the_block_raised(client: SyncVeniceClient) -> None:
    """The rule for a job with nothing to keep carries over from the async block."""
    job, backend = _start(client, "video")
    backend.video.retrieve.side_effect = [
        VideoFailedStatus(status="FAILED", error="refused", error_code="E")
    ]

    with pytest.raises(VideoGenerationError), job:
        job.wait(poll_interval=0)

    backend.video.cancel.assert_awaited_once()


def test_exit_forwards_the_exception_triple_and_the_aexit_result(
    client: SyncVeniceClient,
) -> None:
    received: list[tuple[Any, Any, Any]] = []

    class Suppressing(VideoJob):
        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:  # type: ignore[override]
            received.append((exc_type, exc_val, exc_tb))
            return True

    backend = _backend()
    client._async_client.video.run = AsyncMock(  # type: ignore[method-assign]
        return_value=Suppressing(backend, VideoQueueResponse(model="m", queue_id="q"))
    )
    boom = ValueError("boom")

    with client.video.run(model="m"):
        raise boom

    exc_type, exc_val, exc_tb = received[0]
    assert exc_type is ValueError
    assert exc_val is boom
    assert exc_tb is boom.__traceback__
    assert exc_tb is not None


# ---------------------------------------------------------------------------
# Coverage of every job class
# ---------------------------------------------------------------------------


def _job_classes() -> list[type]:
    found: set[type] = set()
    for module_info in pkgutil.walk_packages(
        venice_ai.resources.__path__, venice_ai.resources.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        for name, obj in vars(module).items():
            if name.endswith("Job") and inspect.isclass(obj) and obj.__module__ == module_info.name:
                found.add(obj)
    return sorted(found, key=lambda cls: cls.__name__)


def test_job_class_scan_finds_the_known_jobs() -> None:
    assert {ImageJob, MusicJob, VideoJob, VoiceChangerJob} <= set(_job_classes())


@pytest.mark.parametrize("job_class", _job_classes(), ids=lambda cls: cls.__name__)
def test_every_job_class_is_wrapped_with_no_coroutine_left_exposed(
    client: SyncVeniceClient, job_class: type
) -> None:
    from venice_ai._sync_client import _wrap_result

    job = object.__new__(job_class)
    wrapped = _wrap_result(job, client._run)

    assert wrapped is not job
    assert hasattr(wrapped, "__enter__")
    assert hasattr(wrapped, "__exit__")
    coroutine_methods = [
        name
        for name, member in inspect.getmembers(job_class)
        if not name.startswith("_") and inspect.iscoroutinefunction(member)
    ]
    assert coroutine_methods
    for name in coroutine_methods:
        assert not inspect.iscoroutinefunction(getattr(wrapped, name)), name


# ---------------------------------------------------------------------------
# Through real HTTP, against a loopback server
# ---------------------------------------------------------------------------


@pytest.fixture
def video_server() -> Iterator[tuple[str, list[str]]]:
    """Answer the three video endpoints from a thread of its own, logging each path."""
    seen: list[str] = []
    polls = {"count": 0}

    async def queue(request: web.Request) -> web.Response:
        seen.append(request.path)
        return web.json_response({"model": "stub-model", "queue_id": "q-http"})

    async def retrieve(request: web.Request) -> web.Response:
        seen.append(request.path)
        polls["count"] += 1
        if polls["count"] == 1:
            return web.json_response(
                {"status": "PROCESSING", "average_execution_time": 100, "execution_duration": 50}
            )
        return web.Response(body=MP4, content_type="video/mp4")

    async def complete(request: web.Request) -> web.Response:
        seen.append(request.path)
        return web.json_response({"success": True})

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def start() -> tuple[web.AppRunner, str]:
        app = web.Application()
        app.router.add_post("/api/v1/video/queue", queue)
        app.router.add_post("/api/v1/video/retrieve", retrieve)
        app.router.add_post("/api/v1/video/complete", complete)
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return runner, f"http://127.0.0.1:{port}/api/v1"

    runner, base_url = asyncio.run_coroutine_threadsafe(start(), loop).result(10)
    yield base_url, seen
    asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_video_run_over_http_finishes_and_releases(
    video_server: tuple[str, list[str]], tmp_path: Path
) -> None:
    base_url, seen = video_server

    with (
        SyncVeniceClient(api_key="fake-key", base_url=base_url) as client,
        client.video.run(model="stub-model", prompt="x", duration_seconds=5) as job,
    ):
        status = job.wait(poll_interval=0)
        saved = job.download(tmp_path / "out.mp4", status)
        assert seen[-1] == "/api/v1/video/retrieve"

    assert saved.read_bytes() == MP4
    assert seen == [
        "/api/v1/video/queue",
        "/api/v1/video/retrieve",
        "/api/v1/video/retrieve",
        "/api/v1/video/complete",
    ]


def test_video_run_over_http_keeps_the_media_when_the_block_raises(
    video_server: tuple[str, list[str]],
) -> None:
    base_url, seen = video_server

    with (
        SyncVeniceClient(api_key="fake-key", base_url=base_url) as client,
        pytest.raises(OSError, match="disk full"),
        client.video.run(model="stub-model", prompt="x", duration_seconds=5) as job,
    ):
        job.wait(poll_interval=0)
        raise OSError("disk full")

    assert "/api/v1/video/complete" not in seen


# ---------------------------------------------------------------------------
# close() with a call still blocked on another thread
# ---------------------------------------------------------------------------


def _block_on_thread(call: Callable[[], Any]) -> tuple[threading.Thread, dict[str, Any]]:
    """Run *call* on a daemon thread, keeping what it raises for the test to read."""
    outcome: dict[str, Any] = {}

    def body() -> None:
        try:
            outcome["returned"] = call()
        except BaseException as e:  # noqa: BLE001 - the error is what the test reads
            outcome["raised"] = e

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread, outcome


def _assert_released_by_close(thread: threading.Thread, outcome: dict[str, Any]) -> None:
    thread.join(timeout=5)
    assert not thread.is_alive(), "a blocked caller was stranded by close()"
    assert isinstance(outcome.get("raised"), RuntimeError), outcome
    assert "closed" in str(outcome["raised"])


def test_close_releases_a_job_wait_blocked_on_another_thread() -> None:
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    backend = _backend()
    polled = threading.Event()

    async def retrieve(**_: Any) -> Any:
        polled.set()
        return JOBS["video"][1]

    backend.video.retrieve = retrieve
    job = _wrap_result(
        VideoJob(backend, VideoQueueResponse(model="stub-model", queue_id="q")), client._run
    )
    thread, outcome = _block_on_thread(lambda: job.wait(poll_interval=3600))
    assert polled.wait(timeout=5)

    client.close()

    _assert_released_by_close(thread, outcome)


def test_close_waits_for_a_cancelled_call_to_finish_unwinding() -> None:
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    started = threading.Event()
    unwound = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            await asyncio.sleep(0)
            unwound.set()

    thread, outcome = _block_on_thread(lambda: client._run(blocked()))
    assert started.wait(timeout=5)

    client.close()

    assert unwound.is_set()
    _assert_released_by_close(thread, outcome)


def test_close_releases_a_caller_whose_task_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("venice_ai._sync_client._CANCEL_WAIT_SECONDS", 0.05)
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    started = threading.Event()

    async def stubborn() -> None:
        started.set()
        while True:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(3600)

    thread, outcome = _block_on_thread(lambda: client._run(stubborn()))
    assert started.wait(timeout=5)

    client.close()

    _assert_released_by_close(thread, outcome)


def test_close_reports_a_task_it_gave_up_on_and_asyncio_never_logs_it_later(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Otherwise asyncio logs it at whatever moment the garbage collector finds the task."""
    monkeypatch.setattr("venice_ai._sync_client._CANCEL_WAIT_SECONDS", 0.05)

    def close_on_a_stubborn_task() -> None:
        client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
        started = threading.Event()

        async def stubborn() -> None:
            started.set()
            while True:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(3600)

        thread, outcome = _block_on_thread(lambda: client._run(stubborn()))
        assert started.wait(timeout=5)
        client.close()
        _assert_released_by_close(thread, outcome)

    with caplog.at_level(logging.WARNING):
        close_on_a_stubborn_task()
        gc.collect()

    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []
    warnings = [r.getMessage() for r in caplog.records if r.name == "venice_ai._sync_client"]
    assert len(warnings) == 1
    assert "ignored cancellation" in warnings[0]


def test_close_releases_a_walk_blocked_on_another_thread() -> None:
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    started = threading.Event()

    async def source() -> Any:
        started.set()
        await asyncio.sleep(3600)
        yield 1

    walk = _wrap_result(source(), client._run)
    thread, outcome = _block_on_thread(lambda: next(walk))
    assert started.wait(timeout=5)

    client.close()

    _assert_released_by_close(thread, outcome)


def test_close_logs_no_asyncio_error_for_a_walk_blocked_on_another_thread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    started = threading.Event()

    async def source() -> Any:
        started.set()
        await asyncio.sleep(3600)
        yield 1

    walk = _wrap_result(source(), client._run)
    thread, outcome = _block_on_thread(lambda: next(walk))
    assert started.wait(timeout=5)

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        client.close()

    _assert_released_by_close(thread, outcome)
    assert not [r for r in caplog.records if r.name == "asyncio"]


def test_close_releases_a_request_blocked_on_another_thread() -> None:
    """A real in-flight request, whose session close() is about to close."""
    arrived = threading.Event()
    loop = asyncio.new_event_loop()
    server_thread = threading.Thread(target=loop.run_forever, daemon=True)
    server_thread.start()

    async def hang(_: web.Request) -> web.Response:
        arrived.set()
        await asyncio.Event().wait()
        return web.Response()

    async def start() -> tuple[web.AppRunner, str]:
        app = web.Application()
        app.router.add_get("/api/v1/models", hang)
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return runner, f"http://127.0.0.1:{port}/api/v1"

    runner, base_url = asyncio.run_coroutine_threadsafe(start(), loop).result(10)
    try:
        client = SyncVeniceClient(api_key="k", base_url=base_url)
        thread, outcome = _block_on_thread(client.models.list)
        assert arrived.wait(timeout=5)

        client.close()

        _assert_released_by_close(thread, outcome)
    finally:

        async def stop() -> None:
            # The handler is still hanging: nothing else ends it before the loop closes.
            await runner.cleanup()
            others = asyncio.all_tasks() - {asyncio.current_task()}
            for task in others:
                task.cancel()
            await asyncio.gather(*others, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        server_thread.join(5)
        loop.close()


def test_close_cancels_other_tasks_only_after_the_async_client_has_closed() -> None:
    """A task no call owns may be the limiter's, whose stop() can wait for it."""
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    unwound = threading.Event()
    alive_during_close: list[bool] = []
    background: list[asyncio.Task[None]] = []

    async def worker() -> None:
        try:
            await asyncio.sleep(3600)
        finally:
            unwound.set()

    async def spawn() -> None:
        background.append(asyncio.create_task(worker()))
        await asyncio.sleep(0)

    client._run(spawn())

    async def close() -> None:
        alive_during_close.append(not background[0].done() and not unwound.is_set())

    client._async_client.close = close

    client.close()

    assert alive_during_close == [True]
    assert unwound.is_set()


def test_a_call_made_while_closing_raises_instead_of_starting() -> None:
    client = SyncVeniceClient(api_key="k", base_url=UNREACHABLE)
    outcome: dict[str, Any] = {}
    finished = threading.Event()

    async def stop() -> None:
        ran = threading.Event()

        async def late() -> None:
            ran.set()

        def call() -> None:
            try:
                client._run(late())
            except RuntimeError as e:
                outcome["raised"] = e
            finished.set()

        threading.Thread(target=call, daemon=True).start()
        await asyncio.to_thread(finished.wait, 5)
        outcome["ran"] = ran.is_set()

    client._async_client.close = stop

    client.close()

    assert isinstance(outcome.get("raised"), RuntimeError), outcome
    assert outcome["ran"] is False


def test_video_job_download_url_is_readable_without_await(client: SyncVeniceClient) -> None:
    queued = VideoQueueResponse(model="m", queue_id="q", download_url="https://cdn.example.com/v")
    client._async_client.video.run = AsyncMock(return_value=VideoJob(_backend(), queued))
    job = client.video.run(model="m")
    assert job.download_url == "https://cdn.example.com/v"
