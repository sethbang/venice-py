"""``iter_*`` paginators under ``SyncVeniceClient``.

A paginator comes back as a plain iterable. Its pages are fetched one at a
time, on the client's loop thread, only when the walk reaches them, and leaving
the walk early leaves nothing suspended or running on the loop.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import signal
import subprocess
import sys
import textwrap
import threading
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web

from venice_ai import SyncVeniceClient
from venice_ai import _sync_client as sync_client_module
from venice_ai._pagination import Paginator, _PageResult
from venice_ai.middleware.retry import RetryOptions, _active_retry_options
from venice_ai.types.api.characters import Character

UNREACHABLE = "http://127.0.0.1:9/api/v1"

# How long an interrupted fetch gets to see its cancellation. It stays well under
# the per-request timeout in tests/unit/conftest.py, which cancels an orphaned
# fetch on its own and would otherwise pass for the SDK doing it.
CANCEL_WAIT_S = 2


@pytest.fixture
def client() -> Iterator[SyncVeniceClient]:
    sync_client = SyncVeniceClient(api_key="fake-key", base_url=UNREACHABLE)
    yield sync_client
    sync_client.close()


def _item(name: str) -> SimpleNamespace:
    """An item carrying both identities the paginators key on."""
    return SimpleNamespace(id=name, slug=name)


_A, _B, _C = _item("a"), _item("b"), _item("c")


def _single_page(*items: Any, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(data=list(items), **fields)


def _reviews_page(page: int, total: int, *items: Any) -> SimpleNamespace:
    return _single_page(*items, pagination=SimpleNamespace(page=page, totalPages=total))


def _transactions_page(has_more: bool, *items: Any) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(transactions=list(items), pagination=SimpleNamespace(hasMore=has_more))
    )


# resource, its page method, the pages it answers, the call under test, the items walked.
ENDPOINTS = {
    "characters.iter_all": (
        "characters",
        "list",
        [_single_page(_A, _B), _single_page(_C)],
        lambda c: c.characters.iter_all(page_size=2),
        [_A, _B, _C],
    ),
    "characters.iter_reviews": (
        "characters",
        "reviews",
        [_reviews_page(1, 2, _A, _B), _reviews_page(2, 2, _C)],
        lambda c: c.characters.iter_reviews("some-slug", page_size=2),
        [_A, _B, _C],
    ),
    "billing.iter_usage_history": (
        "billing",
        "get_usage_history",
        [_single_page("a", "b", nextCursor="next"), _single_page("c", nextCursor=None)],
        lambda c: c.billing.iter_usage_history(),
        ["a", "b", "c"],
    ),
    "x402.iter_transactions": (
        "x402",
        "transactions",
        [_transactions_page(True, _A, _B), _transactions_page(False, _C)],
        lambda c: c.x402.iter_transactions(auth=Mock(), page_size=2),
        [_A, _B, _C],
    ),
}
_CSV_FIRST = SimpleNamespace(nextCursor="next")
_CSV_LAST = SimpleNamespace(nextCursor=None)
ENDPOINTS["billing.iter_usage_history_csv"] = (
    "billing",
    "get_usage_history",
    [_CSV_FIRST, _CSV_LAST],
    lambda c: c.billing.iter_usage_history_csv(),
    [_CSV_FIRST, _CSV_LAST],
)


def _stub_pages(
    client: SyncVeniceClient, resource: str, method: str, pages: list[Any]
) -> tuple[AsyncMock, list[threading.Thread]]:
    """Answer *resource*'s page method with *pages*, recording the fetching thread."""
    threads: list[threading.Thread] = []
    remaining = iter(pages)

    async def answer(*_args: Any, **_kwargs: Any) -> Any:
        threads.append(threading.current_thread())
        return next(remaining)

    stub = AsyncMock(side_effect=answer)
    setattr(getattr(client._async_client, resource), method, stub)
    return stub, threads


class _TrackedPaginator(Paginator[int]):
    """A paginator that records where and whether its async generator was closed."""

    def __init__(self, pages: list[list[int]]) -> None:
        self.fetched: list[int] = []
        self.closed_on: list[threading.Thread] = []
        self.fetch_threads: list[threading.Thread] = []

        async def fetch(index: int) -> _PageResult[int]:
            self.fetched.append(index)
            self.fetch_threads.append(threading.current_thread())
            return _PageResult(items=pages[index], has_more=index + 1 < len(pages))

        super().__init__(fetch, page_size=2)

    async def __aiter__(self) -> AsyncIterator[int]:  # type: ignore[override]
        try:
            async for item in super().__aiter__():
                yield item
        finally:
            self.closed_on.append(threading.current_thread())


def _tracked(client: SyncVeniceClient, pages: list[list[int]]) -> _TrackedPaginator:
    """Serve a tracked paginator from ``client.characters.iter_all``."""
    paginator = _TrackedPaginator(pages)
    client._async_client.characters.iter_all = lambda **_: paginator  # type: ignore[method-assign]
    return paginator


# ---------------------------------------------------------------------------
# Every endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ENDPOINTS)
def test_paginator_walks_every_page_with_a_plain_for_loop(
    client: SyncVeniceClient, name: str
) -> None:
    resource, method, pages, call, expected = ENDPOINTS[name]
    stub, threads = _stub_pages(client, resource, method, pages)

    walked = []
    for item in call(client):
        walked.append(item)

    assert walked == expected
    assert stub.await_count == 2
    assert threads == [client._thread, client._thread]


@pytest.mark.parametrize("name", ENDPOINTS)
def test_building_a_paginator_fetches_nothing(client: SyncVeniceClient, name: str) -> None:
    resource, method, pages, call, _ = ENDPOINTS[name]
    stub, _threads = _stub_pages(client, resource, method, pages)

    iter(call(client))

    stub.assert_not_called()


# ---------------------------------------------------------------------------
# Laziness and iteration semantics
# ---------------------------------------------------------------------------


def test_pages_are_fetched_only_as_the_walk_reaches_them(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1, 2], [3]])

    walk = iter(client.characters.iter_all())
    assert paginator.fetched == []

    assert next(walk) == 1
    assert paginator.fetched == [0]

    assert next(walk) == 2
    assert paginator.fetched == [0]

    assert next(walk) == 3
    assert paginator.fetched == [0, 1]

    assert list(walk) == []
    assert paginator.fetch_threads == [client._thread, client._thread]


def test_max_items_stops_without_fetching_a_discarded_page(client: SyncVeniceClient) -> None:
    stub, _threads = _stub_pages(
        client,
        "characters",
        "list",
        [_single_page(_A, _B), _single_page(_C, _item("d")), _single_page(_item("not-fetched"))],
    )

    assert list(client.characters.iter_all(page_size=2, max_items=3)) == [_A, _B, _C]
    assert stub.await_count == 2


def test_iterating_again_restarts_from_the_first_page(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1, 2], [3]])

    pages = client.characters.iter_all()

    assert list(pages) == [1, 2, 3]
    assert list(pages) == [1, 2, 3]
    assert paginator.fetched == [0, 1, 0, 1]


def test_paginator_attributes_pass_through(client: SyncVeniceClient) -> None:
    assert client.api_keys.iter_all(page_size=7).page_size == 7


def test_page_error_reaches_the_caller_and_ends_the_walk(client: SyncVeniceClient) -> None:
    calls: list[int] = []

    async def fetch(index: int) -> _PageResult[int]:
        calls.append(index)
        if index == 1:
            raise ValueError("page failed")
        return _PageResult(items=[1], has_more=True)

    client._async_client.characters.iter_all = lambda **_: Paginator(fetch)  # type: ignore[method-assign]

    walk = iter(client.characters.iter_all())
    assert next(walk) == 1
    with pytest.raises(ValueError, match="page failed"):
        next(walk)
    assert list(walk) == []
    assert calls == [0, 1]


def test_retry_override_applies_to_each_page_and_the_walk_still_ends(
    client: SyncVeniceClient,
) -> None:
    options = RetryOptions(max_attempts=7)
    seen: list[RetryOptions | None] = []

    async def fetch(index: int) -> _PageResult[int]:
        seen.append(_active_retry_options.get())
        return _PageResult(items=[index], has_more=index == 0)

    client._async_client.characters.iter_all = lambda **_: Paginator(fetch)  # type: ignore[method-assign]

    with client.with_retries(options):
        assert list(client.characters.iter_all()) == [0, 1]

    assert seen == [options, options]


# ---------------------------------------------------------------------------
# Leaving early
# ---------------------------------------------------------------------------


def _pending_tasks(client: SyncVeniceClient) -> set[asyncio.Task[Any]]:
    async def others() -> set[asyncio.Task[Any]]:
        return asyncio.all_tasks() - {asyncio.current_task()}  # type: ignore[arg-type]

    return client._run(others())


def test_break_closes_the_walk_on_the_loop_thread(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1, 2], [3]])

    for _item in client.characters.iter_all():
        break

    assert paginator.closed_on == [client._thread]
    assert paginator.fetched == [0]
    assert _pending_tasks(client) == set()


def test_exception_in_the_loop_body_closes_the_walk(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1, 2], [3]])

    with pytest.raises(ValueError, match="in the body"):
        for _item in client.characters.iter_all():
            raise ValueError("in the body")

    assert paginator.closed_on == [client._thread]
    assert paginator.fetched == [0]


def test_a_walk_never_started_runs_nothing(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1]])

    walk = iter(client.characters.iter_all())
    walk.close()  # type: ignore[attr-defined]

    assert paginator.fetched == []
    assert paginator.closed_on == []


def test_closing_the_client_closes_a_walk_still_held(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1, 2], [3]])

    walk = iter(client.characters.iter_all())
    next(walk)
    client.close()

    assert paginator.closed_on == [client._thread]
    # Dropping the walk afterwards has nothing left to do and raises nothing.
    walk.close()  # type: ignore[attr-defined]


def test_closing_inside_a_retry_block_logs_no_asyncgen_error(
    client: SyncVeniceClient, caplog: pytest.LogCaptureFixture
) -> None:
    paginator = _tracked(client, [[1, 2], [3]])
    walk = iter(client.characters.iter_all())
    next(walk)

    with caplog.at_level(logging.ERROR), client.with_retries(RetryOptions(max_attempts=2)):
        client.close()

    assert paginator.closed_on == [client._thread]
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    walk.close()  # type: ignore[attr-defined]


def test_stepping_a_walk_after_close_raises_without_leaking_a_coroutine(
    client: SyncVeniceClient,
) -> None:
    _tracked(client, [[1, 2], [3]])

    walk = iter(client.characters.iter_all())
    next(walk)
    client.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="closed"):
            next(walk)
        gc.collect()

    assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []


@pytest.mark.timeout(20)
def test_calling_the_client_from_its_own_loop_thread_raises_instead_of_hanging(
    client: SyncVeniceClient,
) -> None:
    """A progress callback runs on the loop thread, where blocking would deadlock."""

    async def reenter() -> None:
        inner = asyncio.sleep(0)
        client._run(inner)

    with pytest.raises(RuntimeError, match="loop thread"):
        client._run(reenter())


@pytest.mark.timeout(20)
def test_a_walk_collected_on_the_loop_thread_does_not_deadlock(client: SyncVeniceClient) -> None:
    paginator = _tracked(client, [[1, 2], [3]])
    walk = iter(client.characters.iter_all())
    next(walk)

    async def collect_here() -> None:
        walk.close()  # type: ignore[attr-defined]

    client._run(collect_here())

    # The walk could not block here; the loop still finishes closing it.
    async def settle() -> None:
        for _ in range(10):
            await asyncio.sleep(0)

    client._run(settle())
    assert paginator.closed_on == [client._thread]


@pytest.mark.timeout(20)
@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs signal.pthread_kill")
def test_interrupting_a_blocked_fetch_cancels_it(client: SyncVeniceClient) -> None:
    """Ctrl-C during a page fetch stops the fetch rather than orphaning it."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("KeyboardInterrupt is only delivered to the main thread")

    started = threading.Event()
    cancelled = threading.Event()

    async def fetch(_index: int) -> _PageResult[int]:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _PageResult(items=[], has_more=False)

    client._async_client.characters.iter_all = lambda **_: Paginator(fetch)  # type: ignore[method-assign]

    def interrupt_once_started() -> None:
        # A real signal: only that wakes a thread blocked on a lock. Never sent
        # unless the fetch is under way, when pytest.raises below is waiting for it.
        if started.wait(10):
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)  # type: ignore[arg-type]

    threading.Thread(target=interrupt_once_started, daemon=True).start()

    with pytest.raises(KeyboardInterrupt):
        for _item in client.characters.iter_all():
            pass

    assert cancelled.wait(CANCEL_WAIT_S)


def test_an_interrupt_just_after_scheduling_still_cancels_the_call(
    client: SyncVeniceClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C between scheduling a call and waiting on it cancels the call too.

    A signal can land on any bytecode, including the ones right after the call
    is handed to the loop. Raising the interrupt as the hand-off returns puts it
    there every time, which a real signal does only now and then.
    """
    started = threading.Event()
    cancelled = threading.Event()

    async def fetch(_index: int) -> _PageResult[int]:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _PageResult(items=[], has_more=False)

    real_start_on_loop = sync_client_module._start_on_loop
    armed = [True]

    def start_then_interrupt(*args: Any) -> None:
        real_start_on_loop(*args)
        if armed[0]:
            armed[0] = False
            assert started.wait(5)
            raise KeyboardInterrupt

    client._async_client.characters.iter_all = lambda **_: Paginator(fetch)  # type: ignore[method-assign]
    monkeypatch.setattr(sync_client_module, "_start_on_loop", start_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        for _item in client.characters.iter_all():
            pass

    assert cancelled.wait(CANCEL_WAIT_S)


def test_an_interrupt_while_handing_the_call_to_the_loop_still_cancels_it(
    client: SyncVeniceClient,
) -> None:
    """Ctrl-C inside the hand-off to the loop, after the call is queued, cancels it.

    Waking the loop releases the GIL, so the loop can start the call while the
    caller is still inside ``call_soon_threadsafe``: a signal handled there
    escapes before the caller holds any handle on the call.
    """
    started = threading.Event()
    cancelled = threading.Event()

    async def fetch(_index: int) -> _PageResult[int]:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _PageResult(items=[], has_more=False)

    loop = client._loop
    real_call_soon_threadsafe = loop.call_soon_threadsafe
    armed = [True]

    def interrupting_call_soon_threadsafe(callback: Any, *args: Any, **kwargs: Any) -> Any:
        handle = real_call_soon_threadsafe(callback, *args, **kwargs)
        if armed[0] and threading.current_thread() is threading.main_thread():
            armed[0] = False
            assert started.wait(5)
            raise KeyboardInterrupt
        return handle

    client._async_client.characters.iter_all = lambda **_: Paginator(fetch)  # type: ignore[method-assign]
    loop.call_soon_threadsafe = interrupting_call_soon_threadsafe  # type: ignore[method-assign]
    try:
        with pytest.raises(KeyboardInterrupt):
            for _item in client.characters.iter_all():
                pass
    finally:
        loop.call_soon_threadsafe = real_call_soon_threadsafe  # type: ignore[method-assign]

    assert cancelled.wait(CANCEL_WAIT_S)


def test_interpreter_exit_with_a_walk_still_held_does_not_hang() -> None:
    script = textwrap.dedent(
        """
        from venice_ai import SyncVeniceClient
        from venice_ai._pagination import Paginator, _PageResult

        async def fetch(index):
            return _PageResult(items=[1, 2], has_more=True)

        client = SyncVeniceClient(api_key="fake-key", base_url="http://127.0.0.1:9/api/v1")
        client._async_client.characters.iter_all = lambda: Paginator(fetch)
        walk = iter(client.characters.iter_all())
        print(next(walk))
        """
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


# ---------------------------------------------------------------------------
# Through real HTTP, against a loopback server
# ---------------------------------------------------------------------------


def _character(slug: str) -> dict[str, Any]:
    return {
        "slug": slug,
        "name": slug,
        "description": None,
        "adult": False,
        "webEnabled": True,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "tags": [],
        "stats": {"imports": 0},
        "modelId": "stub-model",
    }


@pytest.fixture
def loopback() -> Iterator[Callable[[Any], tuple[str, list[dict[str, str]]]]]:
    """Serve ``GET /api/v1/characters`` from a thread of its own."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runners: list[web.AppRunner] = []

    def serve(pages: list[list[str]]) -> tuple[str, list[dict[str, str]]]:
        seen: list[dict[str, str]] = []

        async def handler(request: web.Request) -> web.Response:
            seen.append(dict(request.query))
            index = len(seen) - 1
            slugs = pages[index] if index < len(pages) else []
            return web.json_response(
                {"object": "list", "data": [_character(slug) for slug in slugs]}
            )

        async def start() -> str:
            app = web.Application()
            app.router.add_get("/api/v1/characters", handler)
            runner = web.AppRunner(app, shutdown_timeout=0.1)
            await runner.setup()
            runners.append(runner)
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            return f"http://127.0.0.1:{port}/api/v1"

        return asyncio.run_coroutine_threadsafe(start(), loop).result(10), seen

    yield serve

    for runner in runners:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_walk_over_http_is_typed_and_lazy(loopback: Any) -> None:
    base_url, seen = loopback([["a", "b"], ["c"]])

    with SyncVeniceClient(api_key="fake-key", base_url=base_url) as client:
        pages = client.characters.iter_all(page_size=2)
        assert seen == []

        walk = iter(pages)
        first = next(walk)
        assert isinstance(first, Character)
        assert first.slug == "a"
        assert len(seen) == 1

        assert [c.slug for c in walk] == ["b", "c"]
        assert [request["offset"] for request in seen] == ["0", "2"]


def test_break_over_http_stops_requesting_pages(loopback: Any) -> None:
    base_url, seen = loopback([["a", "b"], ["c"]])

    with SyncVeniceClient(api_key="fake-key", base_url=base_url) as client:
        for _character_item in client.characters.iter_all(page_size=2):
            break

    assert len(seen) == 1
