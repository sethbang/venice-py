"""Unit tests for SyncVeniceClient and its proxy classes."""

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
