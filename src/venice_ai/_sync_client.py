"""
Synchronous wrapper for VeniceClient.

Provides ``SyncVeniceClient`` — a drop-in synchronous interface that
delegates to the async ``VeniceClient`` via a background event loop thread.

Example::

    from venice_ai import SyncVeniceClient

    with SyncVeniceClient() as client:
        response = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import threading
from collections.abc import Iterator
from typing import Any

from ._client import VeniceClient
from ._resource import APIResource
from .core.http_client import ConnectionLimits
from .middleware.retry import RetryOptions
from .streaming import Stream

logger = logging.getLogger(__name__)


class _SyncStreamProxy:
    """Wraps an async ``Stream`` so it can be iterated synchronously.

    Supports both ``for chunk in stream`` and ``with stream`` usage::

        stream = client.chat.completions.create(model=..., messages=[...], stream=True)
        for chunk in stream:
            print(chunk)

        # Or with context manager:
        with client.chat.completions.create(..., stream=True) as stream:
            for chunk in stream:
                ...
    """

    __slots__ = ("_stream", "_run")

    def __init__(self, stream: Stream[Any], run: Any) -> None:
        self._stream = stream
        self._run = run

    def __iter__(self) -> _SyncStreamProxy:
        return self

    def __next__(self) -> Any:
        try:
            return self._run(self._stream.__anext__())
        except StopAsyncIteration:
            raise StopIteration from None

    def __enter__(self) -> _SyncStreamProxy:
        self._run(self._stream.__aenter__())
        return self

    def __exit__(self, *args: object) -> None:
        self._run(self._stream.__aexit__(*args))

    def close(self) -> None:
        """Close the underlying async stream."""
        self._run(self._stream.close())


def _should_proxy(obj: Any) -> bool:
    """Return True if *obj* is an API resource or namespace that should be proxied."""
    return isinstance(obj, APIResource) or (
        hasattr(obj, "__dict__")
        and any(isinstance(getattr(obj, name, None), APIResource) for name in vars(obj))
    )


def _wrap_result(result: Any, run: Any) -> Any:
    """Wrap async return values (e.g. Stream) for synchronous use."""
    if isinstance(result, Stream):
        return _SyncStreamProxy(result, run)
    return result


class _SyncProxy:
    """Generic proxy that converts async method calls into synchronous ones.

    Wraps an async object so that:
    - Attribute access returns either a ``_SyncProxy`` (for sub-resources)
      or a sync-wrapped callable (for async methods).
    - Non-async attributes are returned as-is.

    Sub-resource proxies are cached after first access to avoid re-wrapping.
    """

    __slots__ = ("_target", "_run", "_cache")

    def __init__(self, target: Any, run: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_run", run)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Any:
        # Return cached proxy if available
        cache: dict[str, Any] = self._cache
        if name in cache:
            return cache[name]

        attr = getattr(self._target, name)

        # Async method → wrap in synchronous caller, then wrap result
        if inspect.iscoroutinefunction(attr):
            run = self._run

            @functools.wraps(attr)
            def sync_method(*args: Any, **kwargs: Any) -> Any:
                return _wrap_result(run(attr(*args, **kwargs)), run)

            return sync_method

        # Sub-resource or namespace containing resources → wrap and cache
        if _should_proxy(attr):
            proxy = _SyncProxy(attr, self._run)
            cache[name] = proxy
            return proxy

        return attr

    def __repr__(self) -> str:
        return f"_SyncProxy({self._target!r})"


class SyncVeniceClient:
    """Synchronous wrapper around :class:`~venice_ai.VeniceClient`.

    Accepts the same constructor arguments as ``VeniceClient``. All async
    resource methods (``client.chat.completions.create(...)``, etc.) are
    available as regular synchronous calls.

    Uses a dedicated background thread with its own event loop so it works
    regardless of whether the caller already has a running loop (e.g. Jupyter).

    Must be used as a context manager (``with``) or explicitly closed via
    :meth:`close`::

        with SyncVeniceClient(api_key="...") as client:
            models = client.models.list()

        # Or without context manager:
        client = SyncVeniceClient(api_key="...")
        try:
            models = client.models.list()
        finally:
            client.close()
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        rate_limiter_config: dict[str, Any] | None = None,
        rate_limiter_config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a synchronous Venice AI client.

        Accepts the same parameters as :class:`~venice_ai.VeniceClient`.
        The most common ones are surfaced explicitly for IDE autocompletion;
        additional keyword arguments (``http_client``, ``proxy``,
        ``connector_limit``, ``headers``, ``retry_options``, etc.) are
        forwarded directly to the underlying async client.

        :param api_key: API key. When ``None``, falls back to ``config.api_key``
            (if a ``config`` is passed), then the ``VENICE_API_KEY`` env var.
        :param base_url: Override the default API base URL.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum retry attempts for failed requests.
        :param rate_limiter_config: Dict-based rate limiter configuration.
        :param rate_limiter_config_path: Path to a rate limiter config file.
        :param kwargs: All other arguments accepted by
            :class:`~venice_ai.VeniceClient`.
        """
        # Build the kwargs dict, omitting None so VeniceClient uses its defaults
        init_kwargs: dict[str, Any] = {**kwargs}
        if api_key is not None:
            init_kwargs["api_key"] = api_key
        if base_url is not None:
            init_kwargs["base_url"] = base_url
        if timeout is not None:
            init_kwargs["timeout"] = timeout
        if max_retries is not None:
            init_kwargs["max_retries"] = max_retries
        if rate_limiter_config is not None:
            init_kwargs["rate_limiter_config"] = rate_limiter_config
        if rate_limiter_config_path is not None:
            init_kwargs["rate_limiter_config_path"] = rate_limiter_config_path

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._async_client = VeniceClient(**init_kwargs)
        self._is_closed = False
        # Active per-block RetryOptions override set by with_retries(). When
        # not None, _run wraps each submitted coroutine in the async
        # client's with_retries(options) context manager so the ContextVar
        # mechanism that powers the async path also applies here.
        self._retry_override: RetryOptions | None = None

    def _run(self, coro: Any) -> Any:
        """Submit a coroutine to the background loop and block until done.

        If a :meth:`with_retries` block is active, the coroutine is wrapped
        in :meth:`VeniceClient.with_retries` first so the override applies.
        """
        if self._retry_override is not None:
            coro = self._wrap_with_retries(coro, self._retry_override)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _wrap_with_retries(self, coro: Any, options: RetryOptions) -> Any:
        async with self._async_client.with_retries(options):
            return await coro

    @contextlib.contextmanager
    def with_retries(self, options: RetryOptions) -> Iterator[None]:
        """Synchronous parallel of :meth:`VeniceClient.with_retries`.

        Within the ``with`` block, every method call on this client runs
        with *options* as its retry policy; on exit the previous override
        (or the construction-time default) is restored. Blocks may be
        nested.

        Example::

            with client.with_retries(RetryOptions(max_attempts=5)):
                response = client.chat.completions.create(...)

        :param options: The :class:`RetryOptions` to use inside the block.
        """
        prev = self._retry_override
        self._retry_override = options
        try:
            yield
        finally:
            self._retry_override = prev

    @property
    def retry_options(self) -> RetryOptions | None:
        """The retry policy calls made now are sent with (frozen).

        Inside a :meth:`with_retries` block this is the block's override;
        otherwise the async client's construction-time policy. ``None`` when
        the client wraps a caller-supplied ``http_client`` session.
        """
        if self._retry_override is not None:
            return self._retry_override
        return self._async_client.retry_options

    @property
    def connection_limits(self) -> ConnectionLimits:
        """The connection-pool limits of the underlying HTTP session."""
        return self._async_client.connection_limits

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._async_client, name)

        if inspect.iscoroutinefunction(attr):
            run = self._run

            @functools.wraps(attr)
            def sync_method(*args: Any, **kwargs: Any) -> Any:
                return _wrap_result(run(attr(*args, **kwargs)), run)

            return sync_method

        if _should_proxy(attr):
            proxy = _SyncProxy(attr, self._run)
            # Cache so repeated access (e.g. client.chat.completions) reuses the proxy
            object.__setattr__(self, name, proxy)
            return proxy

        return attr

    def close(self) -> None:
        """Close the underlying async client and shut down the event loop thread.

        Idempotent — safe to call multiple times.
        """
        if self._is_closed:
            return
        self._is_closed = True
        try:
            self._run(self._async_client.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("SyncVeniceClient background thread did not shut down within 5s")
            self._loop.close()

    def __enter__(self) -> SyncVeniceClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Warn (and best-effort cleanup) if the user forgot to call close().
        # Mirrors aiohttp.ClientSession's ResourceWarning convention.
        if not getattr(self, "_is_closed", True):
            import warnings

            warnings.warn(
                f"Unclosed SyncVeniceClient {self!r} — use a `with` block or call .close()",
                ResourceWarning,
                stacklevel=2,
            )

    def __repr__(self) -> str:
        return f"SyncVeniceClient(base_url={self._async_client.base_url!r})"
