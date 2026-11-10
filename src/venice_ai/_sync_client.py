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
import concurrent.futures
import contextlib
import functools
import inspect
import logging
import sys
import threading
from collections.abc import AsyncIterable, Callable, Collection, Coroutine, Iterator
from types import TracebackType
from typing import Any

from ._client import VeniceClient
from ._pagination import Paginator
from ._resource import APIResource
from .core.http_client import ConnectionLimits
from .middleware.retry import RetryOptions
from .resources.image import ImageJob
from .resources.music import MusicJob
from .resources.video import VideoJob
from .resources.voice_changer import VoiceChangerJob
from .streaming import Stream

logger = logging.getLogger(__name__)

_JOB_TYPES = (ImageJob, MusicJob, VideoJob, VoiceChangerJob)

#: How long close() gives calls still running to unwind once cancelled.
_CANCEL_WAIT_SECONDS = 5.0


async def _cancel_and_wait(tasks: Collection[asyncio.Task[Any]]) -> None:
    """Cancel *tasks*, leaving out the current one, and give them time to unwind."""
    others = set(tasks) - {asyncio.current_task()}
    for task in others:
        task.cancel()
    if others:
        await asyncio.wait(others, timeout=_CANCEL_WAIT_SECONDS)


async def _cancel_remaining_tasks() -> None:
    await _cancel_and_wait(asyncio.all_tasks())


def _give_up_on_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Report tasks that outlived cancellation, on a loop about to close for good.

    They can never finish, so asyncio would log each one as destroyed while pending
    at whatever later moment the garbage collector reached it. Reporting them here
    ties the report to the ``close()`` that caused it.
    """
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task._log_destroy_pending = False  # type: ignore[attr-defined]
    if pending:
        logger.warning(
            "SyncVeniceClient closed with %d task(s) still running: they ignored cancellation",
            len(pending),
        )


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

    Whatever else the wrapped stream offers is available without ``await``, so
    a :class:`~venice_ai.streaming.ChatStream` from
    ``client.chat.completions.stream(...)`` has ``collect()``, ``text_deltas()``,
    ``collect_with_deltas()``, ``final_response`` and ``reasoning_summary``::

        with client.chat.completions.stream(model=..., messages=[...]) as stream:
            for text in stream.collect_with_deltas():
                print(text, end="", flush=True)
            print(stream.final_response.usage)

    A plain ``Stream``, which is what ``create(..., stream=True)`` returns, has
    none of those, the same as when used asynchronously.
    """

    __slots__ = ("_stream", "_run")

    def __init__(self, stream: Stream[Any], run: Any) -> None:
        self._stream = stream
        self._run = run

    def __iter__(self) -> _SyncStreamProxy:
        # Raises StreamConsumedError for a consumed stream, as ``async for`` does.
        self._stream.__aiter__()
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

    def get_iterator(self) -> Iterator[Any]:
        """Return a blocking iterator over the raw chunks, consuming this stream."""
        return _iterate(self._stream.get_iterator(), self._run)

    def __getattr__(self, name: str) -> Any:
        return _sync_attribute(getattr(self._stream, name), self._run)


def _start_on_loop(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    future: concurrent.futures.Future[Any],
    wrap: Callable[[Coroutine[Any, Any, Any]], Coroutine[Any, Any, Any]] | None = None,
) -> None:
    """Run *coro* as a task on *loop* and report its outcome through *future*.

    Like :func:`asyncio.run_coroutine_threadsafe`, except that the caller creates
    *future* first, so it holds a handle on the call before the call can start.
    Cancelling *future* cancels the task, or stops it from starting at all.

    *wrap*, if given, builds the coroutine the task runs around *coro*. It is
    applied only when the call starts: a call cancelled first closes *coro*, and
    no wrapper exists to be left unawaited.
    """

    def start() -> None:
        if future.cancelled():
            coro.close()
            return
        runnable = coro if wrap is None else wrap(coro)
        try:
            task = loop.create_task(runnable)
        except BaseException as e:
            runnable.close()
            coro.close()
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                future.set_exception(e)
            return

        def report(done: asyncio.Task[Any]) -> None:
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                if done.cancelled():
                    future.cancel()
                elif (error := done.exception()) is not None:
                    future.set_exception(error)
                else:
                    future.set_result(done.result())

        def cancel_task(cancelled: concurrent.futures.Future[Any]) -> None:
            if cancelled.cancelled():
                # The loop may already be closed; its tasks are gone then.
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(task.cancel)

        task.add_done_callback(report)
        future.add_done_callback(cancel_task)

    loop.call_soon_threadsafe(start)


def _iterate(source: AsyncIterable[Any], run: Any) -> Iterator[Any]:
    """Drive an async iterable from blocking code, one item at a time.

    Each step runs on the client's loop thread, and nothing runs between steps.
    Closing the returned generator (a ``break``, an exception in the loop body,
    or dropping it) closes the async side on the loop thread too, so a
    suspended async generator does not outlive the walk.

    The steps are coroutines defined here rather than ``__anext__()`` called
    directly, because an async generator binds to the event loop of the thread
    that creates its first step.
    """
    iterator = source.__aiter__()

    async def step() -> Any:
        return await iterator.__anext__()

    async def close() -> None:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()

    try:
        while True:
            try:
                item = run(step())
            except StopAsyncIteration:
                return
            yield item
    finally:
        try:
            run(close())
        except RuntimeError as e:
            # No blocking call is possible here (the client is closed, or the
            # walk was dropped on the loop thread). The loop closes a generator
            # it still holds, either on its own or from SyncVeniceClient.close().
            logger.debug("Sync iterator left closing to the event loop: %s", e)


class _SyncPaginatorProxy:
    """Wraps a :class:`~venice_ai._pagination.Paginator` for a plain ``for`` loop.

    Building it fetches nothing. As with the async paginator, each ``for`` loop
    is a new walk from the first page, and pages are fetched only as the walk
    reaches them.
    """

    __slots__ = ("_paginator", "_run")

    def __init__(self, paginator: Paginator[Any], run: Any) -> None:
        self._paginator = paginator
        self._run = run

    def __iter__(self) -> Iterator[Any]:
        return _iterate(self._paginator, self._run)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._paginator, name)

    def __repr__(self) -> str:
        return f"_SyncPaginatorProxy({self._paginator!r})"


class _SyncJobProxy:
    """Wraps a job object (``VideoJob`` and the like) for blocking code.

    Coroutine methods block until done, and properties and fields read straight
    through to the job. ``with job:`` is the job's ``async with``, so it releases
    the stored media on the same terms: an exception raised in the block reaches
    ``__aexit__``, which keeps the media of a billed job for resuming::

        with client.video.run(model=model, prompt="...", duration_seconds=5) as job:
            status = job.wait()
            job.download("output.mp4", status)

    An ``on_progress`` callback passed to ``wait()`` runs on the client's loop
    thread: it should return quickly and cannot call the client, which raises
    ``RuntimeError`` there.
    """

    __slots__ = ("_job", "_run")

    def __init__(self, job: Any, run: Any) -> None:
        self._job = job
        self._run = run

    def __enter__(self) -> _SyncJobProxy:
        self._run(self._job.__aenter__())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        result: bool | None = self._run(self._job.__aexit__(exc_type, exc_val, exc_tb))
        return result

    def __getattr__(self, name: str) -> Any:
        return _sync_attribute(getattr(self._job, name), self._run)

    def __repr__(self) -> str:
        return f"_SyncJobProxy({self._job!r})"


def _should_proxy(obj: Any) -> bool:
    """Return True if *obj* is an API resource or namespace that should be proxied."""
    return isinstance(obj, APIResource) or (
        hasattr(obj, "__dict__")
        and any(isinstance(getattr(obj, name, None), APIResource) for name in vars(obj))
    )


def _wrap_result(result: Any, run: Any) -> Any:
    """Wrap what the SDK returns for asynchronous use so that blocking code can use it.

    Only the SDK's own async objects are wrapped: streams, paginators, jobs and
    async generators. Anything else comes back unchanged, whatever it supports. A
    ``MagicMock`` looks async-iterable and is not one, and a custom object has a
    surface this module cannot know. That includes a ``stream_cls`` that does not
    subclass :class:`~venice_ai.streaming.Stream`: it stays an async object bound to
    the client's loop, so subclass ``Stream`` to iterate one from blocking code.
    """
    if isinstance(result, Stream):
        return _SyncStreamProxy(result, run)
    if isinstance(result, Paginator):
        return _SyncPaginatorProxy(result, run)
    if isinstance(result, _JOB_TYPES):
        return _SyncJobProxy(result, run)
    if inspect.isasyncgen(result):
        return _iterate(result, run)
    return result


def _sync_attribute(attr: Any, run: Any) -> Any:
    """Return the form of *attr* that blocking code calls.

    Coroutine functions block until done, and async generator functions become
    blocking iterators. Plain functions run as they are, with their result
    wrapped, since some build an async object without awaiting anything
    (``iter_*`` returning a ``Paginator``). Anything else, including a
    property's value, is returned unchanged.
    """
    if inspect.iscoroutinefunction(attr):

        @functools.wraps(attr)
        def blocking_method(*args: Any, **kwargs: Any) -> Any:
            return _wrap_result(run(attr(*args, **kwargs)), run)

        return blocking_method

    if inspect.isasyncgenfunction(attr):

        @functools.wraps(attr)
        def iterating_method(*args: Any, **kwargs: Any) -> Iterator[Any]:
            return _iterate(attr(*args, **kwargs), run)

        return iterating_method

    if inspect.ismethod(attr) or inspect.isfunction(attr):

        @functools.wraps(attr)
        def plain_method(*args: Any, **kwargs: Any) -> Any:
            return _wrap_result(attr(*args, **kwargs), run)

        return plain_method

    return attr


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

        wrapped = _sync_attribute(attr, self._run)
        if wrapped is not attr:
            return wrapped

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
    available as regular synchronous calls. What they return works without
    ``await`` too: ``iter_*`` paginators and audio byte streams
    (``audio.create_speech(stream=True)``, ``audio.stream_long_text``) in a plain
    ``for`` loop, streams in a ``for`` or ``with``, and jobs (``video.run``,
    ``music.run``, ``voice_changer.run``, ``image.submit``) with blocking methods
    and ``with``. A custom ``stream_cls`` works this way only if it subclasses
    :class:`~venice_ai.streaming.Stream`; any other comes back as the async object
    it is.

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

    Paginators fetch each page only when the walk reaches it::

        for character in client.characters.iter_all():
            print(character.slug)
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

        # Validate the arguments before anything needs cleaning up: a client
        # that fails to build leaves no event loop or thread behind.
        self._async_client = VeniceClient(**init_kwargs)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._is_closed = False
        # Calls blocked in _submit, for close() to release. The lock orders
        # submitting a call against closing the loop.
        self._in_flight: set[concurrent.futures.Future[Any]] = set()
        self._submit_lock = threading.Lock()
        # The tasks running those calls. Only the loop thread touches this.
        self._call_tasks: set[asyncio.Task[Any]] = set()
        # Active per-block RetryOptions override set by with_retries(). When
        # not None, _run wraps each submitted coroutine in the async
        # client's with_retries(options) context manager so the ContextVar
        # mechanism that powers the async path also applies here.
        self._retry_override: RetryOptions | None = None

    def _run(self, coro: Any) -> Any:
        """Submit a coroutine to the background loop and block until done.

        If a :meth:`with_retries` block is active, the coroutine is wrapped
        in :meth:`VeniceClient.with_retries` first so the override applies.

        An interrupt while blocked (Ctrl-C) cancels the coroutine, so nothing
        keeps running on the loop after the caller has gone.

        :raises RuntimeError: If called from the loop thread, where blocking
            would deadlock (an ``on_progress`` callback runs there), once
            :meth:`close` has been called, or if :meth:`close` is called while
            this call is blocked.
        """
        if self._is_closed:
            coro.close()
            raise RuntimeError("SyncVeniceClient is closed.")
        return self._submit(coro, self._retry_override)

    def _submit(self, coro: Any, retry_override: RetryOptions | None = None) -> Any:
        """Run *coro* on the background loop and block until it is done."""
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError(
                "SyncVeniceClient cannot be called from its own loop thread, where "
                "callbacks such as on_progress run: it would wait on itself forever."
            )
        # The future exists before the call reaches the loop, and the hand-off sits
        # inside the try: an interrupt can land on any bytecode once the call is
        # queued (waking the loop releases the GIL), and must still cancel it.
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            with self._submit_lock:
                if self._loop.is_closed() or sys.is_finalizing():
                    coro.close()
                    raise RuntimeError("SyncVeniceClient is closed.")
                self._in_flight.add(future)
                _start_on_loop(
                    self._loop,
                    coro,
                    future,
                    functools.partial(self._wrap_call, retry_override=retry_override),
                )
            return future.result()
        except concurrent.futures.CancelledError as e:
            if not self._is_closed:
                raise
            raise RuntimeError("SyncVeniceClient was closed during this call.") from e
        except BaseException:
            future.cancel()
            raise
        finally:
            with self._submit_lock:
                self._in_flight.discard(future)

    def _wrap_call(
        self, coro: Coroutine[Any, Any, Any], *, retry_override: RetryOptions | None
    ) -> Coroutine[Any, Any, Any]:
        """Build the coroutine a call's task runs: the retry override, if any, then tracking."""
        if retry_override is not None:
            coro = self._wrap_with_retries(coro, retry_override)
        return self._as_call_task(coro)

    async def _as_call_task(self, coro: Any) -> Any:
        task = asyncio.current_task()
        if task is None:
            return await coro
        self._call_tasks.add(task)
        try:
            return await coro
        finally:
            self._call_tasks.discard(task)

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
        # Only reached for this name when __init__ did not finish.
        if name == "_async_client":
            raise AttributeError(name)

        attr = getattr(self._async_client, name)

        wrapped = _sync_attribute(attr, self._run)
        if wrapped is not attr:
            return wrapped

        if _should_proxy(attr):
            proxy = _SyncProxy(attr, self._run)
            # Cache so repeated access (e.g. client.chat.completions) reuses the proxy
            object.__setattr__(self, name, proxy)
            return proxy

        return attr

    def close(self) -> None:
        """Close the underlying async client and shut down the event loop thread.

        A call still blocked on another thread, such as ``job.wait()``, is cancelled
        and raises :class:`RuntimeError` there. New calls raise it too.

        Idempotent — safe to call multiple times.
        """
        if self._is_closed:
            return
        self._is_closed = True
        try:
            # Calls still blocked stop first, so that none is running a generator
            # that is about to be shut down or a request on a session about to close.
            # Other tasks can belong to the limiter, whose stop() may wait for them.
            self._submit(_cancel_and_wait(self._call_tasks))
            # A walk still held by the caller is closed before the session it reads.
            # _submit rather than _run: closing is not retried, and shutting down
            # async generators must not run inside the async generator that a retry
            # override would wrap it in.
            self._submit(self._loop.shutdown_asyncgens())
            self._submit(self._async_client.close())
        finally:
            try:
                self._submit(_cancel_remaining_tasks())
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    logger.warning("SyncVeniceClient background thread did not shut down within 5s")
                with self._submit_lock:
                    # A task that ignored its cancellation will never finish now.
                    for future in self._in_flight:
                        future.cancel()
                    _give_up_on_pending_tasks(self._loop)
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
