"""
Shared error translation for aiohttp transport failures.

Every place the SDK sends a request or reads a response body maps transport
failures through :func:`map_aiohttp_error`, so a timeout always surfaces as
:class:`~venice_ai.exceptions.APITimeoutError` and a connection failure as
:class:`~venice_ai.exceptions.APIConnectionError`, never as a bare
:class:`TimeoutError` or ``aiohttp.ClientError``.

An ``aiohttp.ClientResponseError`` keeps the request's headers, which hold the API
key, and its ``repr`` prints them. :class:`HeaderFreeResponse` keeps them out of
every response and error on a session the SDK builds. :func:`forget_request_headers`
removes them from an error raised on a session the caller supplied, which the mapping
and :func:`parse_json` both apply.

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
* :func:`url_free_transport_error` and :func:`url_free_status_error` build the
  errors for a URL that grants access on its own, such as a download link.
* :class:`ResponseSummary`, :func:`input_free_validation_error`,
  :func:`validate_hiding_input` and :func:`forget_parsed_text` keep such a URL,
  or a body that holds one, out of what an error retains.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError
from yarl import URL

if TYPE_CHECKING:
    from venice_ai.exceptions import APIConnectionError, APITimeoutError

Phase = Literal["request", "body", "stream"]

_ModelT = TypeVar("_ModelT", bound=BaseModel)

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


def _without_credentials(url: URL) -> URL:
    if url.raw_user is None and url.raw_password is None:
        return url
    return url.with_user(None)


def _without_headers(info: aiohttp.RequestInfo) -> aiohttp.RequestInfo:
    """Return *info* without its headers, or a password in its URL.

    A ``ClientHttpProxyError`` holds the request sent to the proxy, whose URL is the
    proxy's own, password included.
    """
    return aiohttp.RequestInfo(
        _without_credentials(info.url),
        info.method,
        CIMultiDictProxy(CIMultiDict()),
        _without_credentials(info.real_url),
    )


class HeaderFreeResponse(aiohttp.ClientResponse):
    """A response whose ``request_info`` has no headers.

    aiohttp builds every ``ClientResponseError``, ``ContentTypeError`` and
    ``TooManyRedirects`` from a response's ``request_info``, so none raised on a
    session using this class holds the request's headers, whichever code raises it.
    """

    def __init__(
        self, method: str, url: URL, *, request_info: aiohttp.RequestInfo, **kwargs: Any
    ) -> None:
        super().__init__(method, url, request_info=_without_headers(request_info), **kwargs)


def forget_request_headers(error: BaseException | None) -> None:
    """Remove the request's headers from each aiohttp error in *error*'s chain, in place.

    A ``ClientResponseError`` found through ``__cause__`` or ``__context__`` keeps its
    type, ``status``, ``message``, URL and response ``headers``. Its ``request_info``
    loses the request headers and any user and password in the URL, and its ``history``
    is emptied, since each response in the history holds them too.
    """
    seen: set[int] = set()
    stack = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        stack += [current.__cause__, current.__context__]
        if isinstance(current, aiohttp.ClientResponseError) and isinstance(
            current.request_info, aiohttp.RequestInfo
        ):
            current.request_info = _without_headers(current.request_info)
            current.history = ()
            current.args = (current.request_info, ())


async def parse_json(response: aiohttp.ClientResponse) -> Any:
    """``await response.json()``, raising an error that does not hold the request's headers."""
    try:
        return await response.json()
    except aiohttp.ClientResponseError as e:
        forget_request_headers(e)
        raise


def map_aiohttp_error(
    error: BaseException, *, phase: Phase = "request"
) -> APIConnectionError | APITimeoutError | None:
    """Return the SDK exception for an aiohttp / asyncio transport error.

    Returns ``None`` when *error* is not a transport error, so the caller
    re-raises it unchanged. The result holds *error* as ``original_error``, without
    the request's headers (see :func:`forget_request_headers`). More specific types
    are checked before their base classes:

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

    forget_request_headers(error)
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


def url_free_transport_error(error: Exception, *, phase: Phase = "request") -> Exception:
    """Return the SDK exception for a transport error on a URL that must stay secret.

    The result is what :func:`map_aiohttp_error` gives, naming the original only
    by type, in its message and in ``original_error_type``: aiohttp's error text
    and ``repr`` can quote the URL, so the original is not attached. Raise the
    result after the ``except`` block has ended, or Python records the original
    as its ``__context__``.

    Args:
        error: The failure. One that :func:`map_aiohttp_error` does not map
            comes back as :class:`~venice_ai.exceptions.APIConnectionError`.
        phase: As for :func:`map_aiohttp_error`.
    """
    from venice_ai.exceptions import APIConnectionError

    mapped = map_aiohttp_error(error, phase=phase) or APIConnectionError(
        _CONNECTION_MESSAGES[phase]
    )
    return type(mapped)(f"{mapped} ({type(error).__name__})", original_error_type=type(error))


def url_free_status_error(response: aiohttp.ClientResponse) -> aiohttp.ClientResponseError:
    """Build the error for a failed response to a URL that must stay secret.

    It is the ``aiohttp.ClientResponseError`` that ``raise_for_status()`` would
    raise, without what its text and ``repr`` take from the request: the URL is
    reduced to scheme and host, and the request headers (which can hold the API
    key) and the redirect history are empty. ``Retry-After`` is the only response
    header kept, since others can repeat the URL.
    """
    summary = ResponseSummary.url_free(response)
    request_info = aiohttp.RequestInfo(
        summary.url, response.method, CIMultiDictProxy(CIMultiDict()), summary.url
    )
    return aiohttp.ClientResponseError(
        request_info,
        (),
        status=response.status,
        message=response.reason or "",
        headers=summary.headers,
    )


@dataclass(frozen=True)
class ResponseSummary:
    """What an error keeps of a response that holds a secret.

    An ``aiohttp.ClientResponse`` keeps its URL, its redirect history, its request
    and its body, so an error that must not expose those holds this instead. It
    has the same ``status``, ``reason``, ``method``, ``url``, ``real_url``, ``ok``
    and ``headers``, and nothing to read a body from.
    """

    status: int
    reason: str | None
    method: str
    url: URL
    headers: CIMultiDictProxy[str]

    @property
    def real_url(self) -> URL:
        return self.url

    @property
    def ok(self) -> bool:
        return self.status < 400

    @classmethod
    def url_free(cls, response: aiohttp.ClientResponse) -> ResponseSummary:
        """For a URL that must stay secret: the URL is reduced to scheme and host, and
        ``Retry-After`` is the only header kept, since others can repeat the URL."""
        headers = CIMultiDict(
            (name, value)
            for name, value in response.headers.items()
            if name.lower() == "retry-after"
        )
        return cls(
            status=response.status,
            reason=response.reason,
            method=response.method,
            url=URL.build(scheme=response.url.scheme, host=response.url.host or ""),
            headers=CIMultiDictProxy(headers),
        )

    @classmethod
    def without_body(cls, response: aiohttp.ClientResponse) -> ResponseSummary:
        """For a body that must stay secret, on a URL and headers that may be shown."""
        return cls(
            status=response.status,
            reason=response.reason,
            method=response.method,
            url=response.url,
            headers=response.headers,
        )


def input_free_validation_error(error: ValidationError) -> ValidationError:
    """Return a copy of *error* that no longer holds the input that failed.

    ``hide_input_in_errors`` hides the input from the error's text only:
    ``errors()`` and ``json()`` still return it. The copy keeps each error's
    ``type``, ``loc`` and ``msg``, with ``input`` set to ``None``. Raise it after
    the ``except`` block has ended, or Python records the original as its
    ``__context__``.
    """
    details: list[InitErrorDetails] = [
        {
            "type": PydanticCustomError(item["type"], item["msg"]),  # pyright: ignore[reportArgumentType]
            "loc": item["loc"],
            "input": None,
        }
        for item in error.errors(include_input=False, include_url=False)
    ]
    return ValidationError.from_exception_data(error.title, details, hide_input=True)


def validate_hiding_input(model: type[_ModelT], data: Any) -> _ModelT:
    """``model.model_validate(data)``, raising an error that does not hold *data*."""
    failure: ValidationError
    try:
        return model.model_validate(data)
    except ValidationError as e:
        failure = input_free_validation_error(e)
    raise failure


def forget_parsed_text(error: Exception) -> None:
    """Drop the text a ``json.JSONDecodeError`` keeps in ``doc``, the whole body."""
    if isinstance(error, json.JSONDecodeError):
        error.doc = ""


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
    "HeaderFreeResponse",
    "ResponseSummary",
    "forget_parsed_text",
    "forget_request_headers",
    "input_free_validation_error",
    "map_aiohttp_error",
    "parse_json",
    "read_body",
    "resolve_timeout",
    "url_free_status_error",
    "url_free_transport_error",
    "validate_hiding_input",
    "wrap_aiohttp_errors",
]
