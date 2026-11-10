"""``VeniceClient.close`` stops its rate limiter once, however the limiter arrived.

Fakes are SimpleNamespace or spec'd mocks: a bare MagicMock would invent
``cleanup_backend`` and make the ownership checks vacuous.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from venice_ai import VeniceClient
from venice_ai.factory import AdaptiveSchedulerAdapter


def _limiter(backend: SimpleNamespace | None = None) -> SimpleNamespace:
    """A limiter shaped like the factory's adapter around an adaptive scheduler."""
    scheduler = SimpleNamespace(state_manager=SimpleNamespace(backend=backend))
    return SimpleNamespace(
        stop=AsyncMock(),
        start=AsyncMock(),
        is_running=lambda: True,
        cleanup_backend=AdaptiveSchedulerAdapter(scheduler).cleanup_backend,
    )


def _backend() -> SimpleNamespace:
    return SimpleNamespace(cleanup=AsyncMock())


def _client(**kwargs) -> VeniceClient:
    return VeniceClient(api_key="test-key", base_url="http://127.0.0.1:9/api/v1", **kwargs)


async def test_close_stops_an_injected_limiter() -> None:
    client = _client()
    limiter = _limiter()
    client._inject_rate_limiter(limiter)

    await client.close()

    limiter.stop.assert_awaited_once()


async def test_async_with_stops_an_injected_limiter() -> None:
    client = _client()
    limiter = _limiter()
    client._inject_rate_limiter(limiter)

    async with client:
        pass

    limiter.stop.assert_awaited_once()


async def test_close_stops_a_constructor_limiter_once() -> None:
    limiter = _limiter()
    client = _client(rate_limiter=limiter)

    await client.close()

    limiter.stop.assert_awaited_once()


async def test_close_twice_stops_once() -> None:
    client = _client()
    limiter = _limiter()
    client._inject_rate_limiter(limiter)

    await client.close()
    await client.close()

    limiter.stop.assert_awaited_once()


async def test_concurrent_close_stops_once() -> None:
    client = _client()
    limiter = _limiter()

    async def slow_stop() -> None:
        await asyncio.sleep(0)

    limiter.stop = AsyncMock(side_effect=slow_stop)
    client._inject_rate_limiter(limiter)

    await asyncio.gather(client.close(), client.close())

    limiter.stop.assert_awaited_once()


async def test_a_limiter_assigned_to_the_attribute_is_stopped() -> None:
    client = _client()
    limiter = _limiter()
    client.rate_limiter = limiter

    await client.close()

    limiter.stop.assert_awaited_once()


async def test_two_distinct_limiters_are_each_stopped() -> None:
    constructed = _limiter()
    replacement = _limiter()
    client = _client(rate_limiter=constructed)
    client.rate_limiter = replacement

    await client.close()

    constructed.stop.assert_awaited_once()
    replacement.stop.assert_awaited_once()


async def test_close_cleans_up_the_backend_of_an_injected_limiter() -> None:
    client = _client()
    backend = _backend()
    client._inject_rate_limiter(_limiter(backend))

    await client.close()
    await client.close()

    backend.cleanup.assert_awaited_once()


async def test_close_leaves_the_backend_of_a_constructor_limiter_alone() -> None:
    backend = _backend()
    limiter = _limiter(backend)
    client = _client(rate_limiter=limiter)

    await client.close()

    limiter.stop.assert_awaited_once()
    backend.cleanup.assert_not_awaited()


async def test_cleanup_follows_stop() -> None:
    calls: list[str] = []
    backend = SimpleNamespace(cleanup=AsyncMock(side_effect=lambda: calls.append("cleanup")))
    limiter = _limiter(backend)
    limiter.stop = AsyncMock(side_effect=lambda: calls.append("stop"))
    client = _client()
    client._inject_rate_limiter(limiter)

    await client.close()

    assert calls == ["stop", "cleanup"]


async def test_limiter_stops_before_the_http_client_closes() -> None:
    calls: list[str] = []
    limiter = _limiter()
    limiter.stop = AsyncMock(side_effect=lambda: calls.append("limiter"))
    client = _client()
    client._inject_rate_limiter(limiter)
    client._venice_http_client = SimpleNamespace(
        close=AsyncMock(side_effect=lambda: calls.append("http"))
    )

    await client.close()

    assert calls == ["limiter", "http"]


@pytest.mark.parametrize("error", [RuntimeError("boom"), ValueError("boom")])
async def test_a_failing_stop_still_closes_the_http_client(error: Exception) -> None:
    limiter = _limiter()
    limiter.stop = AsyncMock(side_effect=error)
    client = _client()
    client._inject_rate_limiter(limiter)
    http = SimpleNamespace(close=AsyncMock())
    client._venice_http_client = http

    await client.close()

    http.close.assert_awaited_once()
    assert client._is_closed is True


async def test_a_failing_cleanup_does_not_escape_close() -> None:
    backend = SimpleNamespace(cleanup=AsyncMock(side_effect=OSError("redis gone")))
    client = _client()
    client._inject_rate_limiter(_limiter(backend))

    await client.close()

    assert client._is_closed is True


async def test_an_injected_limiter_without_a_backend_closes_cleanly() -> None:
    limiter = SimpleNamespace(stop=AsyncMock(), is_running=lambda: True)
    client = _client()
    client._inject_rate_limiter(limiter)

    await client.close()

    limiter.stop.assert_awaited_once()


async def test_adapter_cleans_up_the_backend_of_its_scheduler() -> None:
    backend = _backend()
    scheduler = SimpleNamespace(state_manager=SimpleNamespace(backend=backend))

    await AdaptiveSchedulerAdapter(scheduler).cleanup_backend()

    backend.cleanup.assert_awaited_once()


@pytest.mark.parametrize(
    "scheduler",
    [
        SimpleNamespace(),
        SimpleNamespace(state_manager=SimpleNamespace()),
        SimpleNamespace(state_manager=SimpleNamespace(backend=SimpleNamespace())),
    ],
    ids=["no-state-manager", "no-backend", "backend-without-cleanup"],
)
async def test_adapter_cleanup_is_a_no_op_without_a_backend_to_clean(
    scheduler: SimpleNamespace,
) -> None:
    await AdaptiveSchedulerAdapter(scheduler).cleanup_backend()


async def test_a_cancelled_stop_still_closes_the_http_client() -> None:
    limiter = _limiter()
    limiter.stop = AsyncMock(side_effect=asyncio.CancelledError())
    client = _client()
    client._inject_rate_limiter(limiter)
    http = SimpleNamespace(close=AsyncMock())
    client._venice_http_client = http

    with pytest.raises(asyncio.CancelledError):
        await client.close()

    http.close.assert_awaited_once()


class _Blocked:
    """An awaitable step that blocks until released, recording that it started."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self) -> None:
        self.entered.set()
        await self.release.wait()


def _closable(limiter: Any, http_close: Any = None) -> tuple[VeniceClient, Any, Any]:
    """A client owning *limiter*'s backend, with fake HTTP resources to observe."""
    client = _client()
    client._inject_rate_limiter(limiter)
    client._owns_rate_limiter_backend = True
    http: Any = SimpleNamespace(close=http_close or AsyncMock())
    session: Any = SimpleNamespace(close=AsyncMock())
    client._venice_http_client = http
    client._session = session
    client._should_close_session = True
    return client, http, session


async def test_cancel_during_stop_still_cleans_the_backend_and_closes_http() -> None:
    backend = _backend()
    limiter = _limiter(backend)
    blocked = _Blocked()
    limiter.stop = blocked
    client, http, session = _closable(limiter)

    async with asyncio.timeout(1):
        close = asyncio.create_task(client.close())
        await blocked.entered.wait()
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close

    backend.cleanup.assert_awaited_once()
    http.close.assert_awaited_once()
    session.close.assert_awaited_once()


async def test_cancel_during_the_first_limiters_stop_still_stops_the_second() -> None:
    backend = _backend()
    scheduler = _limiter(backend)
    replacement: Any = _limiter()
    blocked = _Blocked()
    replacement.stop = blocked
    client, _http, _session = _closable(scheduler)
    client.rate_limiter = replacement

    async with asyncio.timeout(1):
        close = asyncio.create_task(client.close())
        await blocked.entered.wait()
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close

    scheduler.stop.assert_awaited_once()
    backend.cleanup.assert_awaited_once()


async def test_cancel_during_the_second_limiters_stop_still_cleans_the_backend() -> None:
    backend = _backend()
    scheduler = _limiter(backend)
    blocked = _Blocked()
    scheduler.stop = blocked
    replacement: Any = _limiter()
    client, _http, _session = _closable(scheduler)
    client.rate_limiter = replacement

    async with asyncio.timeout(1):
        close = asyncio.create_task(client.close())
        await blocked.entered.wait()
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close

    backend.cleanup.assert_awaited_once()


async def test_cancel_during_http_close_still_closes_the_session() -> None:
    blocked = _Blocked()
    client, _http, session = _closable(_limiter(), http_close=blocked)

    async with asyncio.timeout(1):
        close = asyncio.create_task(client.close())
        await blocked.entered.wait()
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close

    session.close.assert_awaited_once()


async def test_a_retry_after_a_cancelled_close_repeats_nothing() -> None:
    backend = _backend()
    limiter = _limiter(backend)
    blocked = _Blocked()
    limiter.stop = blocked
    client, http, session = _closable(limiter)

    async with asyncio.timeout(1):
        close = asyncio.create_task(client.close())
        await blocked.entered.wait()
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        await client.close()

    backend.cleanup.assert_awaited_once()
    http.close.assert_awaited_once()
    session.close.assert_awaited_once()


async def test_a_concurrent_close_returns_only_after_the_first_finishes() -> None:
    limiter = _limiter()
    blocked = _Blocked()
    limiter.stop = blocked
    client, _http, session = _closable(limiter)

    async with asyncio.timeout(1):
        first = asyncio.create_task(client.close())
        await blocked.entered.wait()
        second = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not second.done()

        blocked.release.set()
        await second
        session.close.assert_awaited_once()
        await first


async def test_a_concurrent_close_is_not_cancelled_by_the_first_being_cancelled() -> None:
    limiter = _limiter()
    blocked = _Blocked()
    limiter.stop = blocked
    client, _http, session = _closable(limiter)

    async with asyncio.timeout(1):
        first = asyncio.create_task(client.close())
        await blocked.entered.wait()
        second = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second

    session.close.assert_awaited_once()


async def test_cancelling_a_waiting_close_leaves_the_first_running() -> None:
    limiter = _limiter()
    blocked = _Blocked()
    limiter.stop = blocked
    client, _http, session = _closable(limiter)

    async with asyncio.timeout(1):
        first = asyncio.create_task(client.close())
        await blocked.entered.wait()
        second = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

        assert not first.done()
        blocked.release.set()
        await first

    session.close.assert_awaited_once()


async def test_close_called_from_inside_its_own_stop_does_not_deadlock() -> None:
    limiter = _limiter()
    client, _http, session = _closable(limiter)

    async def stop() -> None:
        await client.close()

    limiter.stop = stop

    async with asyncio.timeout(1):
        await client.close()

    session.close.assert_awaited_once()


async def test_close_from_a_task_that_stop_spawns_and_awaits_does_not_deadlock() -> None:
    limiter = _limiter()
    client, _http, session = _closable(limiter)

    async def stop() -> None:
        await asyncio.create_task(client.close())

    limiter.stop = stop

    async with asyncio.timeout(1):
        await client.close()

    session.close.assert_awaited_once()


async def test_close_from_an_older_task_that_stop_awaits_gives_up_waiting_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A worker from before the close is invisible to it; only a time limit can break this."""
    monkeypatch.setattr("venice_ai._client._CLOSE_JOIN_TIMEOUT_SECONDS", 0.01)
    limiter = _limiter()
    client, _http, session = _closable(limiter)
    closing = asyncio.Event()

    async def worker() -> None:
        await closing.wait()
        await client.close()

    worker_task = asyncio.create_task(worker())

    async def stop() -> None:
        closing.set()
        await worker_task

    limiter.stop = stop

    async with asyncio.timeout(1):
        await client.close()

    session.close.assert_awaited_once()
    assert "still running" in caplog.text


async def test_close_from_an_older_task_that_stop_cancels_unwinds_at_once() -> None:
    limiter = _limiter()
    client, _http, session = _closable(limiter)
    closing = asyncio.Event()
    waiting = asyncio.Event()

    async def worker() -> None:
        await closing.wait()
        waiting.set()
        await client.close()

    worker_task = asyncio.create_task(worker())

    async def stop() -> None:
        closing.set()
        await waiting.wait()
        await asyncio.sleep(0)
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    limiter.stop = stop

    async with asyncio.timeout(1):
        await client.close()

    session.close.assert_awaited_once()


async def test_close_from_another_event_loop_returns_instead_of_touching_this_one() -> None:
    limiter = _limiter()
    blocked = _Blocked()
    limiter.stop = blocked
    client, _http, session = _closable(limiter)

    async with asyncio.timeout(1):
        first = asyncio.create_task(client.close())
        await blocked.entered.wait()
        await asyncio.wait_for(asyncio.to_thread(asyncio.run, client.close()), timeout=0.5)
        assert not first.done()
        blocked.release.set()
        await first

    session.close.assert_awaited_once()


async def test_closing_one_client_does_not_shortcut_closing_another() -> None:
    other_limiter = _limiter()
    other_blocked = _Blocked()
    other_limiter.stop = other_blocked
    other, _http, other_session = _closable(other_limiter)
    limiter = _limiter()
    client, _http2, _session = _closable(limiter)

    closes_seen_by_waiter: list[int] = []

    async def stop() -> None:
        # Joins a close already running on another client, from inside this client's close.
        waiter = asyncio.create_task(other.close())
        await asyncio.sleep(0)
        other_blocked.release.set()
        await waiter
        closes_seen_by_waiter.append(other_session.close.await_count)

    limiter.stop = stop

    async with asyncio.timeout(1):
        first = asyncio.create_task(other.close())
        await other_blocked.entered.wait()
        await client.close()
        await first

    assert closes_seen_by_waiter == [1]
