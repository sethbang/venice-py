"""Unit tests for SyncVeniceClient and its proxy classes."""

import concurrent.futures
import inspect
import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from venice_ai._sync_client import SyncVeniceClient, _SyncProxy, _SyncStreamProxy, _wrap_result
from venice_ai.streaming import Stream


@pytest.fixture
def mock_async_client():
    """Build a Mock VeniceClient with `chat.completions` as APIResource-like."""
    from venice_ai._resource import APIResource

    completions = Mock()
    # APIResource detection is via isinstance — make completions look like one.
    chat = Mock(spec=APIResource)
    chat.completions = completions
    client = Mock()
    client.chat = chat
    client.base_url = "https://api.venice.ai/api/v1/"
    client.close = AsyncMock()
    return client


@pytest.fixture
def sync_client(mock_async_client):
    """Build a SyncVeniceClient backed by the mock async client."""
    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient(api_key="test-key")
    yield client
    if not client._is_closed:
        client.close()


# ---------------------------------------------------------------------------
# Construction / lifecycle
# ---------------------------------------------------------------------------


def test_init_starts_background_thread(sync_client):
    assert sync_client._thread.is_alive()
    assert sync_client._is_closed is False


def test_failed_construction_leaves_no_thread():
    """A client that fails to build must not leave its loop thread behind."""
    before = set(threading.enumerate())

    with (
        patch("venice_ai._sync_client.VeniceClient", side_effect=ValueError("no key")),
        pytest.raises(ValueError, match="no key"),
    ):
        SyncVeniceClient()

    assert set(threading.enumerate()) - before == set()


def test_rejected_key_leaves_no_thread(monkeypatch):
    """The same holds for the real validation in ``VeniceClient``."""
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    before = set(threading.enumerate())

    with pytest.raises(ValueError, match="No authentication provided"):
        SyncVeniceClient()

    assert set(threading.enumerate()) - before == set()


def test_half_built_client_finalizes_quietly():
    """``__del__`` and attribute lookup work on an instance ``__init__`` never finished."""
    half_built = SyncVeniceClient.__new__(SyncVeniceClient)

    half_built.__del__()

    assert not hasattr(half_built, "chat")


def test_context_manager_calls_close(mock_async_client):
    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        with SyncVeniceClient() as client:
            assert client._is_closed is False
        assert client._is_closed is True
        assert not client._thread.is_alive()


def test_close_is_idempotent(mock_async_client):
    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient()
    client.close()
    # Second call must not raise (loop already stopped).
    client.close()
    assert client._is_closed is True


def test_unclosed_client_warns_on_del(mock_async_client):
    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient()
    # Trigger __del__ explicitly via a deliberately leaked instance.
    with pytest.warns(ResourceWarning, match="Unclosed SyncVeniceClient"):
        client.__del__()
    # Best-effort cleanup so the test doesn't actually leak.
    client.close()


def test_an_unclosed_client_is_collected_and_warns(mock_async_client):
    """The loop thread holds no reference to the client, so dropping it runs ``__del__``."""
    import gc
    import weakref

    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient()
    loop, thread = client._loop, client._thread
    ref = weakref.ref(client)
    with pytest.warns(ResourceWarning, match="Unclosed SyncVeniceClient"):
        del client
        gc.collect()
    assert ref() is None
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_closed_client_no_warning_on_del(mock_async_client):
    import warnings

    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient()
    client.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn warnings into errors
        client.__del__()  # should NOT warn — already closed


# ---------------------------------------------------------------------------
# __getattr__ proxying / caching
# ---------------------------------------------------------------------------


def test_resource_attribute_returns_cached_proxy(sync_client):
    proxy_a = sync_client.chat
    proxy_b = sync_client.chat
    assert proxy_a is proxy_b
    assert isinstance(proxy_a, _SyncProxy)


def test_sub_resource_returns_cached_proxy(sync_client):
    chat_proxy = sync_client.chat
    completions_a = chat_proxy.completions
    completions_b = chat_proxy.completions
    assert completions_a is completions_b


def test_async_method_wraps_to_sync(sync_client, mock_async_client):
    # Attach an async method on the mock client; verify it's wrapped.
    mock_async_client.fetch_external = AsyncMock(return_value=b"ok")
    result = sync_client.fetch_external("https://example.com/x")
    assert result == b"ok"


def test_sync_proxy_invokes_async_method_on_subresource(sync_client, mock_async_client):
    """Calling an async method through ``client.chat.foo()`` must route through
    ``_SyncProxy.__getattr__`` (the sub-resource proxy), not just
    ``SyncVeniceClient.__getattr__``. This exercises the wrap-and-call branch
    of the proxy that powers every ``client.<resource>.<method>(...)`` chain.
    """
    # mock_async_client.chat is Mock(spec=APIResource) → it gets wrapped in _SyncProxy.
    # Attaching an AsyncMock here makes inspect.iscoroutinefunction(attr) → True.
    mock_async_client.chat.fetch = AsyncMock(return_value="fetched")

    chat_proxy = sync_client.chat
    assert isinstance(chat_proxy, _SyncProxy)
    result = chat_proxy.fetch("payload", key="value")
    assert result == "fetched"
    mock_async_client.chat.fetch.assert_awaited_once_with("payload", key="value")


def test_sync_proxy_passes_through_non_async_non_resource_attrs(sync_client, mock_async_client):
    """Plain values on a sub-resource (non-coroutine, non-APIResource) are
    returned as-is by ``_SyncProxy.__getattr__``."""
    mock_async_client.chat.some_value = 42
    chat_proxy = sync_client.chat
    assert chat_proxy.some_value == 42


# ---------------------------------------------------------------------------
# _SyncStreamProxy
# ---------------------------------------------------------------------------


def test_sync_stream_proxy_iteration():
    """_SyncStreamProxy iterates an async stream synchronously."""
    from venice_ai.streaming import Stream

    async def _agen():
        for i in (1, 2, 3):
            yield i

    real_stream = Stream(_agen(), client=Mock())

    # Build a one-off SyncVeniceClient just to get a working _run.
    with patch("venice_ai._sync_client.VeniceClient", return_value=Mock(close=AsyncMock())):
        client = SyncVeniceClient()
    try:
        proxy = _SyncStreamProxy(real_stream, client._run)
        assert list(proxy) == [1, 2, 3]
    finally:
        client.close()


def test_sync_stream_proxy_exposes_e2ee():
    """The sync proxy surfaces ``ChatStream.e2ee`` (None for a plain Stream)."""
    from venice_ai.streaming import ChatStream, Stream
    from venice_ai.tee.types import TeeTrust

    async def _agen():
        return
        yield

    trust = TeeTrust(
        model="e2ee-x",
        verification="baseline",
        verifier=None,
        server_verified=True,
        nonce="cd" * 32,
        model_public_key="04" + "ab" * 64,
        signing_address="0x" + "ab" * 20,
    )
    with patch("venice_ai._sync_client.VeniceClient", return_value=Mock(close=AsyncMock())):
        client = SyncVeniceClient()
    try:
        chat_stream = ChatStream(_agen(), client=Mock(), e2ee=trust)
        assert _SyncStreamProxy(chat_stream, client._run).e2ee is trust
        assert _SyncStreamProxy(Stream(_agen(), client=Mock()), client._run).e2ee is None
    finally:
        client.close()


def test_sync_stream_proxy_close_invokes_underlying_close():
    """``_SyncStreamProxy.close()`` must run the underlying async stream's close()."""
    closed = []

    class FakeStream:
        async def close(self) -> None:
            closed.append(True)

    def _run(coro):
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    proxy = _SyncStreamProxy(FakeStream(), _run)  # type: ignore[arg-type]
    proxy.close()
    assert closed == [True]


def test_wrap_result_returns_non_stream_value_unchanged():
    """``_wrap_result`` returns non-Stream values as-is (no proxy wrapping)."""
    from venice_ai._sync_client import _wrap_result

    sentinel = {"any": "object"}
    assert _wrap_result(sentinel, run=lambda c: c) is sentinel


def test_init_forwards_explicit_optional_params_to_async_client(mock_async_client):
    """When optional params are passed explicitly, they're forwarded to ``VeniceClient(...)``.
    This covers the ``if x is not None: init_kwargs["x"] = x`` branches."""
    captured: dict = {}

    def fake_venice_client(**kwargs):
        captured.update(kwargs)
        return mock_async_client

    with patch("venice_ai._sync_client.VeniceClient", side_effect=fake_venice_client):
        client = SyncVeniceClient(
            api_key="test-key",
            base_url="https://example.test/api/v1",
            timeout=42.5,
            max_retries=7,
            rate_limiter_config={"x": 1},
            rate_limiter_config_path="/tmp/cfg.yaml",
        )
    try:
        assert captured["api_key"] == "test-key"
        assert captured["base_url"] == "https://example.test/api/v1"
        assert captured["timeout"] == 42.5
        assert captured["max_retries"] == 7
        assert captured["rate_limiter_config"] == {"x": 1}
        assert captured["rate_limiter_config_path"] == "/tmp/cfg.yaml"
    finally:
        client.close()


def test_sync_stream_proxy_context_manager_forwards_args():
    """__exit__ must forward exception info to the underlying stream."""
    captured = []

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            captured.append((exc_type, exc_val, exc_tb))

    stream = FakeStream()

    def _run(coro):
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    proxy = _SyncStreamProxy(stream, _run)  # type: ignore[arg-type]
    proxy.__enter__()
    err = ValueError("boom")
    proxy.__exit__(type(err), err, None)
    assert captured[0][0] is ValueError
    assert captured[0][1] is err


# ---------------------------------------------------------------------------
# _wrap_result: what becomes a blocking iterator
# ---------------------------------------------------------------------------


class _CustomStream:
    """A ``stream_cls`` that is not a ``Stream``, with a surface of its own."""

    def __init__(self, iterator, client) -> None:
        self.iterator = iterator
        self.client = client

    def __aiter__(self):
        return self.iterator

    def custom_method(self) -> str:
        return "custom"


@pytest.mark.parametrize("make", [MagicMock, AsyncMock], ids=["MagicMock", "AsyncMock"])
def test_wrap_result_leaves_a_mock_alone(make):
    mock = make()
    assert _wrap_result(mock, None) is mock


def test_a_mocked_resource_method_returns_its_mock_through_the_client():
    with SyncVeniceClient(api_key="k", base_url="http://127.0.0.1:9/api/v1") as client:
        listing = MagicMock(data=[1, 2])
        client._async_client.models.list = AsyncMock(return_value=listing)
        assert client.models.list() is listing


def test_wrap_result_wraps_an_async_generator():
    async def agen():
        yield 1

    with SyncVeniceClient(api_key="k", base_url="http://127.0.0.1:9/api/v1") as client:
        wrapped = _wrap_result(agen(), client._run)
        assert inspect.isgenerator(wrapped)
        assert list(wrapped) == [1]


def test_a_custom_stream_cls_that_is_not_a_stream_comes_back_unchanged():
    """It keeps its own surface and stays async-only: nothing here can know what it offers."""
    with SyncVeniceClient(api_key="k", base_url="http://127.0.0.1:9/api/v1") as client:
        stream = client.chat.completions.create(
            model="stub-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_cls=_CustomStream,
        )
        assert isinstance(stream, _CustomStream)
        assert stream.custom_method() == "custom"
        assert stream.client is client._async_client
        client._run(stream.iterator.aclose())


def test_a_custom_stream_cls_that_is_a_stream_gets_the_blocking_proxy():
    class _MyStream(Stream):
        pass

    with SyncVeniceClient(api_key="k", base_url="http://127.0.0.1:9/api/v1") as client:
        stream = client.chat.completions.create(
            model="stub-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_cls=_MyStream,
        )
        assert isinstance(stream, _SyncStreamProxy)
        assert isinstance(stream._stream, _MyStream)
        stream.close()


# ---------------------------------------------------------------------------
# A call that stops the loop thread
# ---------------------------------------------------------------------------


def _in_thread(fn, timeout: float = 5) -> dict:
    """Run *fn* on a daemon thread; report what it returned or raised, or that it hung."""
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["returned"] = fn()
        except BaseException as e:  # noqa: BLE001 - the test inspects it
            outcome["raised"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    outcome["hung"] = thread.is_alive()
    return outcome


@pytest.mark.parametrize(
    "stop", [SystemExit(3), KeyboardInterrupt()], ids=["SystemExit", "KeyboardInterrupt"]
)
def test_a_call_that_raises_systemexit_on_the_loop_gets_it_back(mock_async_client, stop):
    """SystemExit or KeyboardInterrupt raised on the loop thread, as by ``sys.exit()``
    in an ``on_progress`` callback, reaches the caller as it would under
    ``asyncio.run``. The loop keeps running: other calls, later calls and
    ``close()`` work, even after a second such call."""
    import asyncio

    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient(api_key="test-key")

    async def stop_the_loop() -> None:
        await asyncio.sleep(0)
        raise stop

    async def answer_later() -> str:
        await asyncio.sleep(0.2)
        return "answered"

    bystander: dict = {}
    bystander_thread = threading.Thread(
        target=lambda: bystander.update(_in_thread(lambda: client._run(answer_later()))),
        daemon=True,
    )
    bystander_thread.start()

    for _ in range(2):
        caller = _in_thread(lambda: client._run(stop_the_loop()))
        assert not caller["hung"], "the caller waited forever after its call stopped the loop"
        assert caller.get("raised") is stop

    bystander_thread.join(5)
    assert bystander.get("returned") == "answered"

    later = _in_thread(lambda: client._run(asyncio.sleep(0, result="still works")))
    assert later.get("returned") == "still works"

    closing = _in_thread(client.close, timeout=10)
    assert not closing["hung"], "close() waited forever"
    assert "raised" not in closing
    mock_async_client.close.assert_awaited_once()
    assert client._loop.is_closed()


def test_close_finishes_when_a_cancelled_call_raises_systemexit(mock_async_client):
    """A call that answers close()'s cancellation with sys.exit() does not hang close()."""
    import asyncio

    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient(api_key="test-key")
    started = threading.Event()

    async def exit_when_cancelled() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise SystemExit(4) from None

    caller = threading.Thread(
        target=lambda: _in_thread(lambda: client._run(exit_when_cancelled()), timeout=10),
        daemon=True,
    )
    caller.start()
    assert started.wait(5)

    closing = _in_thread(client.close, timeout=10)
    assert not closing["hung"], "close() waited forever"
    assert "raised" not in closing
    mock_async_client.close.assert_awaited_once()
    assert client._loop.is_closed()


def test_close_finishes_when_a_task_raises_systemexit_in_the_same_pass_as_stop(
    mock_async_client,
):
    """SystemExit raised in the loop pass that runs close()'s ``stop()`` ends the
    loop thread instead of starting the loop again."""
    import time

    with patch("venice_ai._sync_client.VeniceClient", return_value=mock_async_client):
        client = SyncVeniceClient(api_key="test-key")
    loop = client._loop
    real_stop = loop.stop

    def stop_then_exit() -> None:
        real_stop()
        raise SystemExit(5)

    # close() schedules self._loop.stop; this makes that one callback both stop
    # the loop and raise, so the two land in the same pass.
    loop.stop = stop_then_exit  # type: ignore[method-assign]

    started = time.monotonic()
    closing = _in_thread(client.close, timeout=10)
    elapsed = time.monotonic() - started

    assert not closing["hung"], "close() waited forever"
    assert "raised" not in closing, closing.get("raised")
    assert elapsed < 3, f"close() took {elapsed:.1f}s"
    assert not client._thread.is_alive()
    assert loop.is_closed()


@pytest.mark.parametrize("settle", ["result", "exception", "cancel"])
def test_a_call_whose_future_is_done_before_it_starts_never_runs(settle):
    """A call settled before the loop reaches it is closed, not run, and not wrapped."""
    import asyncio

    from venice_ai._sync_client import _start_on_loop

    ran = threading.Event()
    wrap = Mock(side_effect=lambda coro: coro)

    async def body() -> str:
        ran.set()
        return "ran"

    coro = body()
    future: concurrent.futures.Future = concurrent.futures.Future()
    loop = asyncio.new_event_loop()
    try:
        _start_on_loop(loop, coro, future, wrap)
        if settle == "result":
            future.set_result("settled elsewhere")
        elif settle == "exception":
            future.set_exception(RuntimeError("settled elsewhere"))
        else:
            future.cancel()
        loop.run_until_complete(asyncio.sleep(0.01))
    finally:
        loop.close()

    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
    assert not ran.is_set()
    wrap.assert_not_called()
