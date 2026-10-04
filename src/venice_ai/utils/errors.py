"""
Shared error translation for aiohttp transport failures.

Every place the SDK sends a request or reads a response body maps transport
failures through :func:`map_aiohttp_error`, so a timeout always surfaces as
:class:`~venice_ai.exceptions.APITimeoutError` and a connection failure as
:class:`~venice_ai.exceptions.APIConnectionError`, never as a bare
:class:`TimeoutError` or ``aiohttp.ClientError``:

* :func:`wrap_aiohttp_errors` applies the mapping to a block of code (sending
  a request, or reading a body together with the request that produced it).
* :func:`read_body` reads a response body that a caller received unread
  (``raw_response=True``), and is what resources use for binary, text and JSON
  bodies alike. Parse the body after reading it, outside the mapping, so a
  response that is not the expected content type is reported as such rather
  than as a connection failure (``aiohttp.ContentTypeError`` is an
  ``aiohttp.ClientError``).
* :class:`~venice_ai.streaming.Stream` maps errors raised mid-iteration with
  ``phase="stream"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import aiohttp

Phase = Literal["request", "body", "stream"]

_MAY_HAVE_BEEN_PROCESSED = (
    "If the request reached the server it may have been processed and billed; "
    "the SDK does not resend paid requests after a timeout."
)
_WAS_PROCESSED = (
    "The server had already answered with a status, so the request was processed "
    "and may have been billed; the SDK does not resend paid requests."
)

# (client-side timeout, server timeout, what that means for billing) per phase.
_TIMEOUT_MESSAGES: dict[str, tuple[str, str, str]] = {
    "request": ("Request timed out", "Server timeout during request", _MAY_HAVE_BEEN_PROCESSED),
    "body": (
        "Timed out reading the response body",
        "Server timeout while reading the response body",
        _WAS_PROCESSED,
    ),
    "stream": (
        "Stream timed out before it finished",
        "Server timeout while streaming the response",
        _WAS_PROCESSED,
    ),
}
_CONNECTION_MESSAGES: dict[str, str] = {
    "request": "A connection error occurred",
    "body": "The connection failed while reading the response body",
    "stream": "The connection failed while streaming the response",
}


def map_aiohttp_error(error: BaseException, *, phase: Phase = "request") -> Exception | None:
    """Return the SDK exception for an aiohttp / asyncio transport error.

    Returns ``None`` when *error* is not a transport error, so the caller
    re-raises it unchanged. More specific types are checked before their
    base classes:

    1. ``aiohttp.ConnectionTimeoutError`` (nothing was sent) and
       ``aiohttp.ServerTimeoutError`` → :class:`APITimeoutError`
    2. ``TimeoutError`` (includes ``asyncio.TimeoutError``) → :class:`APITimeoutError`
    3. ``aiohttp.ClientConnectorError`` → :class:`APIConnectionError`
    4. ``aiohttp.ClientError`` (catch-all) → :class:`APIConnectionError`

    Args:
        error: The exception to translate.
        phase: Where it happened. ``"request"`` is sending the request and
            waiting for the status line, ``"body"`` is reading a body after the
            status arrived, and ``"stream"`` is iterating a streamed response.
            Only the message differs: after the status line the server has
            processed the request.
    """
    # Lazy import to avoid a circular import at module load time.
    from venice_ai.exceptions import APIConnectionError, APITimeoutError

    if isinstance(error, aiohttp.ConnectionTimeoutError):
        return APITimeoutError(
            "Timed out connecting to the API; the request was not sent", original_error=error
        )
    timed_out, server_timeout, billing = _TIMEOUT_MESSAGES[phase]
    if isinstance(error, aiohttp.ServerTimeoutError):
        return APITimeoutError(f"{server_timeout}. {billing}", original_error=error)
    if isinstance(error, TimeoutError):
        return APITimeoutError(f"{timed_out}. {billing}", original_error=error)
    if isinstance(error, aiohttp.ClientConnectorError):
        return APIConnectionError("Connection failed", original_error=error)
    if isinstance(error, aiohttp.ClientError):
        return APIConnectionError(_CONNECTION_MESSAGES[phase], original_error=error)
    return None


@asynccontextmanager
async def wrap_aiohttp_errors(*, phase: Phase = "request") -> AsyncIterator[None]:
    """Translate aiohttp / asyncio errors raised in the block into SDK exceptions.

    Usage::

        from venice_ai.utils.errors import wrap_aiohttp_errors

        async with wrap_aiohttp_errors():
            response = await session.request(**kwargs)

    See :func:`map_aiohttp_error` for the mapping and *phase*. Exceptions that
    are not transport errors pass through unchanged.
    """
    try:
        yield
    except (TimeoutError, aiohttp.ClientError) as e:
        mapped = map_aiohttp_error(e, phase=phase)
        if mapped is None:  # pragma: no cover - both caught types always map
            raise
        raise mapped from e


def resolve_timeout(
    timeout: float | aiohttp.ClientTimeout | None, default: aiohttp.ClientTimeout
) -> aiohttp.ClientTimeout:
    """The timeout one request is sent with.

    A per-call *timeout* (seconds or an ``aiohttp.ClientTimeout``) wins;
    ``None`` means *default*, the client's own timeout. Every path that sends
    a request (JSON, multipart, raw audio streams, external downloads)
    resolves its timeout here, so the client's timeout applies to all of them.
    """
    if timeout is None:
        return default
    if isinstance(timeout, aiohttp.ClientTimeout):
        return timeout
    return aiohttp.ClientTimeout(total=float(timeout))


async def read_body(response: Any) -> bytes:
    """Read a response body, mapping a stalled or dropped transfer to SDK errors.

    The body is read once and cached by aiohttp, so a later
    ``response.json()`` or ``response.text()`` parses it without another read.
    Use this for every body a caller receives unread (``raw_response=True``).

    Raises:
        APITimeoutError: If the body does not arrive within the request's
            timeout.
        APIConnectionError: If the connection fails while the body is read.
    """
    async with wrap_aiohttp_errors(phase="body"):
        body: bytes = await response.read()
    return body


__all__ = [
    "map_aiohttp_error",
    "read_body",
    "resolve_timeout",
    "wrap_aiohttp_errors",
]
