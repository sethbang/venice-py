"""
Billing-aware retry middleware for the Venice AI client's aiohttp session.

Every request the client sends passes through :func:`create_retry_middleware`.
The middleware decides whether a failure may be retried from two facts: what
kind of request it was, and how far the request got before it failed.

## Why billing matters

Venice bills generation when a job is queued or an inference runs, and it does
not accept an ``Idempotency-Key`` on paid endpoints. Resending a request that
the server already accepted can therefore charge twice: a second video job, a
second image, a second music clip. A 504, a read timeout or a dropped
connection can arrive after the work was done and billed. The policy only
resends a request when it can tell the first attempt was not processed, or
when processing it twice costs nothing.

## Request classes

:func:`classify_request` assigns each request a :class:`RetryClass` from its
method, path and headers:

- ``IDEMPOTENT``: ``GET``/``HEAD``/``OPTIONS``/``PUT``/``DELETE``, free
  control-plane ``POST`` calls (``*/quote``, ``*/retrieve``, ``*/complete``,
  ``billing/*``), and any ``POST`` that carries an ``Idempotency-Key`` header.
- ``INFERENCE``: ``chat/completions``, ``responses`` and ``embeddings``. A 500
  from these endpoints is not billed, but it can be deterministic (a model
  that cannot honor the request fails the same way every time).
- ``PAID``: every other ``POST``, including image, video, music, speech and
  voice generation, API key creation and x402 top-ups. Unknown paths land here.

## What is retried

=====================================  ==========  =========  ====
Failure                                IDEMPOTENT  INFERENCE  PAID
=====================================  ==========  =========  ====
Connection never established           yes         yes        yes
(DNS, refused, connect timeout)
503                                    yes         yes        see below
502                                    yes         yes        no
500                                    yes         once       no
504, read timeout, server disconnect   yes         no         no
=====================================  ==========  =========  ====

A 503 on a ``PAID`` request is retried only when it cannot have followed
processing: on the generation endpoints Venice documents a 503 for, where it
means the model is at capacity and the request was turned away (``image/*``,
``images/generations``, ``audio/speech``, ``audio/queue``, ``audio/voices``),
or when the response carries a ``Retry-After`` header. Elsewhere (for example
``video/queue``, ``x402/top-up``, ``api_keys``) a 503 may come from a gateway
after the work was done, so it is surfaced.

429 is never retried here: the client's rate limiter owns it (see
:class:`~venice_ai.rate_limiting.SimpleRateLimiter`), and
:class:`RetryOptions` rejects 429 in ``retry_status_codes``. A
``Retry-After`` (or ``retry-after-ms``) header sets the delay when it is
between 0 and :attr:`RetryOptions.max_retry_after` seconds; a longer one means
the failure is surfaced instead of waited out.

A total request timeout (``aiohttp.ClientTimeout(total=...)``) covers every
attempt and every backoff sleep, so retries never extend it.
"""

import asyncio
import contextvars
import logging
import random
import re
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

from aiohttp import (
    ClientConnectorError,
    ClientError,
    ClientMiddlewareType,
    ClientResponse,
    ClientSSLError,
    ConnectionTimeoutError,
    ServerTimeoutError,
)

logger = logging.getLogger(__name__)

#: Header sent on every resent attempt with the number of the retry (1, 2, ...).
RETRY_COUNT_HEADER = "x-venice-sdk-retry-count"


class RetryClass(StrEnum):
    """How safe a request is to resend; see the module docstring."""

    IDEMPOTENT = "idempotent"
    INFERENCE = "inference"
    PAID = "paid"


_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"})

# Free POST endpoints: quotes, status reads, media release and billing reads.
_IDEMPOTENT_POST_PATH = re.compile(
    r"(?:^|/)(?:"
    r"(?:video|audio|audio/voice-changer)/(?:quote|retrieve|complete)"
    r"|billing/[\w./-]+"
    r")/?$"
)
# Paid endpoints whose documented 503 means "the model is at capacity": the
# request was turned away before any work was done.
_CAPACITY_503_POST_PATH = re.compile(
    r"(?:^|/)(?:"
    r"image/(?:generate|upscale|edit|multi-edit|background-remove)"
    r"|images/generations"
    r"|audio/(?:speech|queue|voices)"
    r")/?$"
)
# Token-billed inference endpoints whose 500 responses are not billed.
_INFERENCE_POST_PATH = re.compile(r"(?:^|/)(?:chat/completions|responses|embeddings)/?$")


def _request_path(url: Any) -> str:
    path = getattr(url, "path", None)
    if isinstance(path, str):
        return path
    text = str(url)
    text = text.split("?", 1)[0]
    if "://" in text:
        text = text.split("://", 1)[1]
        text = "/" + text.split("/", 1)[1] if "/" in text else "/"
    return text


def classify_request(
    method: str,
    path: str,
    headers: Mapping[str, str],
    *,
    idempotent_methods: AbstractSet[str] = _IDEMPOTENT_METHODS,
) -> RetryClass:
    """Return the :class:`RetryClass` for a request.

    Args:
        method: The HTTP method.
        path: The URL path (``/api/v1/chat/completions``) or a relative path.
        headers: The request headers. A ``POST`` that carries an
            ``Idempotency-Key`` is ``IDEMPOTENT``: the server deduplicates it.
        idempotent_methods: Methods treated as ``IDEMPOTENT`` whatever the path.
    """
    verb = method.upper()
    if verb in idempotent_methods:
        return RetryClass.IDEMPOTENT
    if any(str(key).lower() == "idempotency-key" for key in headers):
        return RetryClass.IDEMPOTENT
    if _IDEMPOTENT_POST_PATH.search(path):
        return RetryClass.IDEMPOTENT
    if _INFERENCE_POST_PATH.search(path):
        return RetryClass.INFERENCE
    return RetryClass.PAID


def _is_connect_failure(error: BaseException) -> bool:
    """``True`` when the request never reached the server.

    TLS certificate and handshake errors are excluded: they fail the same way
    on every attempt.
    """
    if isinstance(error, ClientSSLError):
        return False
    return isinstance(error, ClientConnectorError | ConnectionTimeoutError)


@dataclass(frozen=True)
class RetryOptions:
    """
    Retry configuration for the client's HTTP session.

    The defaults follow the billing-aware policy in
    :mod:`venice_ai.middleware.retry`: a request is resent only when the first
    attempt was not processed or when processing it twice is free.

    Instances are frozen, and their collections are stored immutably, so a
    policy cannot change after it is validated. Derive a variant with
    ``dataclasses.replace(options, max_attempts=...)``, which validates the
    result again.

    Attributes:
        max_attempts: Retries after the initial request (``2`` means up to three
            attempts). ``0`` disables retries.
        retry_status_codes: Statuses that can be retried, subject to the
            request's class (see the module docstring). Statuses added here
            beyond 500/502/503/504 (for example 408) are retried for
            ``IDEMPOTENT`` requests only. Takes a set (``set`` or
            ``frozenset``) and stores a ``frozenset``; any other type raises
            ``TypeError``. 429 is
            rejected with ``ValueError``: the client's rate limiter retries
            it (``SimpleRateLimiter(max_retries=...)``, or
            ``RateLimiterConfig.max_retries`` for a factory-built client), and
            retrying it here as well would resend each request twice over.
        retry_exceptions: Exception types that can be retried, as a sequence
            (stored as a ``tuple``). A failure to connect is retried for every
            class; any other listed exception (read timeout, server
            disconnect) only for ``IDEMPOTENT`` requests.
        base_delay: Delay before the first retry, in seconds.
        max_delay: Upper bound on any computed backoff delay, in seconds.
        exponential_base: Multiplier applied per retry (``2.0`` doubles).
        jitter_factor: Fraction (0.0-1.0) by which each delay is randomly
            shortened, so clients that failed together do not retry together.
        respect_retry_after: Use a server ``Retry-After`` / ``retry-after-ms``
            value as the delay.
        max_retry_after: Longest server-requested delay the client waits, in
            seconds. A longer ``Retry-After`` surfaces the failure instead.
        max_inference_500_retries: How many times a 500 from an ``INFERENCE``
            request (chat completions, responses, embeddings) is retried.
            These 500s are not billed but can be deterministic.
        retry_non_idempotent: When ``False``, a ``POST`` that is not
            ``IDEMPOTENT`` is never retried, not even after a connection
            failure.
        classifier: Replaces :func:`classify_request`. Receives
            ``(method, path, headers)`` and returns a :class:`RetryClass`, for
            callers who know an endpoint is safer (or riskier) to resend than
            the default classification assumes.
        on_retry: Called with ``(retry_index, delay_seconds, exception_or_none)``
            before each retry sleep.
        idempotent_methods: HTTP methods treated as ``IDEMPOTENT``, as a set
            (stored as a ``frozenset``).
    """

    max_attempts: int = 2

    retry_status_codes: AbstractSet[int] = field(
        default_factory=lambda: frozenset({500, 502, 503, 504})
    )

    retry_exceptions: Sequence[type[Exception]] = field(
        default_factory=lambda: (
            TimeoutError,
            ServerTimeoutError,
            ClientError,
        )
    )

    base_delay: float = 0.5

    max_delay: float = 8.0

    exponential_base: float = 2.0

    jitter_factor: float = 0.25

    respect_retry_after: bool = True

    max_retry_after: float = 60.0

    max_inference_500_retries: int = 1

    idempotent_methods: AbstractSet[str] = field(default_factory=lambda: _IDEMPOTENT_METHODS)

    retry_non_idempotent: bool = True

    classifier: Callable[[str, str, Mapping[str, str]], RetryClass] | None = None

    on_retry: Callable[[int, float, Exception | None], None] | None = None

    def __post_init__(self) -> None:
        for name, expected, kind in (
            ("retry_status_codes", AbstractSet, "a set of status codes, such as {500, 503}"),
            ("idempotent_methods", AbstractSet, 'a set of HTTP methods, such as {"GET"}'),
            ("retry_exceptions", Sequence, "a sequence of exception types"),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected) or isinstance(value, str):
                raise TypeError(f"RetryOptions.{name} must be {kind}, not {type(value).__name__}")
        if 429 in self.retry_status_codes:
            raise ValueError(
                "RetryOptions.retry_status_codes cannot contain 429. Rate-limit "
                "responses are retried by the client's rate limiter, which "
                "honors Retry-After and the rate-limit headers: use "
                "VeniceClient(rate_limiter=SimpleRateLimiter(max_retries=...)), or "
                "RateLimiterConfig(max_retries=...) with VeniceClientFactory."
            )
        # Immutable collections: a copy (``client.retry_options``,
        # ``dataclasses.replace``) shares them with the policy it came from.
        object.__setattr__(self, "retry_status_codes", frozenset(self.retry_status_codes))
        object.__setattr__(self, "idempotent_methods", frozenset(self.idempotent_methods))
        object.__setattr__(self, "retry_exceptions", tuple(self.retry_exceptions))


# Per-task scope override for the active RetryOptions. Set by
# :meth:`VeniceClient.with_retries` for the duration of a context-managed
# block; resolved by :func:`create_retry_middleware` at the start of every
# request so concurrent calls inside a ``with_retries`` block all see the
# override (asyncio.create_task() propagates ContextVar values into child
# tasks). Calls outside the block — including any raced from outside —
# see ``None`` and fall back to the client's construction-time default.
_active_retry_options: contextvars.ContextVar["RetryOptions | None"] = contextvars.ContextVar(
    "venice_active_retry_options", default=None
)


# The Venice SIWE/SIWX header. Named here because the retry loop has to
# recognise the one header it is allowed to rewrite between attempts.
_SIWE_HEADER = "X-Sign-In-With-X"

# Per-request signer for that header, published by
# :meth:`VeniceClient._prepare_and_send_request` for the duration of a single
# ``session.request()`` call and reset immediately afterwards. It is scoped
# per request rather than per session because the wallet doing the signing can
# differ between requests on one client — ``client.x402.balance(auth=...)``
# takes a per-call wallet that is not the client's own.
_active_siwe_resigner: contextvars.ContextVar["Callable[[], str] | None"] = contextvars.ContextVar(
    "venice_active_siwe_resigner", default=None
)


def _resign_siwe_header(request: Any) -> None:
    """Give a retried request a freshly signed SIWE envelope.

    Venice accepts a SIWE nonce exactly once, so a retry that replays the
    envelope the previous attempt already sent is rejected with a 401 instead
    of being served. aiohttp hands
    the middleware the same mutable :class:`~aiohttp.ClientRequest` on every
    attempt, so the header is rewritten in place.

    Two conditions both have to hold before anything is rewritten: a signer
    must be published for this request, and the request must already carry the
    header. Requests authenticated with a Bearer key — and the ``X-402-Payment``
    header, which is a signed payment payload rather than a SIWE envelope — are
    therefore never touched.
    """
    resign = _active_siwe_resigner.get()
    if resign is None or _SIWE_HEADER not in request.headers:
        return

    try:
        request.headers[_SIWE_HEADER] = resign()
    # A signing failure must not replace the retryable failure with a new one;
    # the stale envelope goes out and the server's rejection is what surfaces.
    except Exception:
        logger.warning(
            "Could not re-sign the %s header before a retry; the previous "
            "envelope will be replayed and is likely to be rejected as a "
            "reused nonce.",
            _SIWE_HEADER,
            exc_info=True,
        )


def calculate_backoff_delay(
    attempt: int,
    base_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter_factor: float,
) -> float:
    """
    Return the delay before retry number ``attempt + 1``.

    The delay is ``base_delay * exponential_base ** attempt``, capped at
    ``max_delay``, then shortened by a random fraction of up to
    ``jitter_factor``. Jitter only ever shortens the delay, so ``max_delay``
    stays a true upper bound.

    Args:
        attempt: Zero-based retry index (``0`` before the first retry).
        base_delay: Delay before the first retry, in seconds.
        exponential_base: Multiplier applied per retry.
        max_delay: Upper bound on the delay, in seconds.
        jitter_factor: Largest fraction (0.0-1.0) removed at random.

    Returns:
        A delay in seconds, never negative.

    Example:
        >>> calculate_backoff_delay(0, 0.5, 2.0, 8.0, 0.0)
        0.5
        >>> calculate_backoff_delay(10, 0.5, 2.0, 8.0, 0.0)
        8.0
    """
    capped_delay = min(base_delay * (exponential_base**attempt), max_delay)
    if jitter_factor > 0:
        capped_delay *= 1.0 - random.uniform(0.0, min(jitter_factor, 1.0))  # nosec B311
    return max(0.0, capped_delay)


def parse_retry_after_header(response: ClientResponse) -> float | None:
    """
    Return the server-requested retry delay in seconds, or ``None``.

    Reads ``retry-after-ms`` (milliseconds) first, then ``Retry-After`` as
    either a number of seconds or an HTTP date (RFC 9110). Unparseable values
    return ``None``; a date in the past returns ``0.0``.

    Args:
        response: The response whose headers are read.
    """
    headers = response.headers
    retry_after_ms = headers.get("retry-after-ms")
    if isinstance(retry_after_ms, str) and retry_after_ms:
        try:
            return float(retry_after_ms) / 1000.0
        except ValueError:
            logger.warning("Could not parse retry-after-ms header %r", retry_after_ms)

    retry_after = headers.get("Retry-After")
    if not isinstance(retry_after, str) or not retry_after:
        return None

    try:
        return float(retry_after)
    except ValueError:
        try:
            retry_date = parsedate_to_datetime(retry_after)
            if retry_date.tzinfo is None:
                retry_date = retry_date.replace(tzinfo=UTC)
            return max(0.0, (retry_date - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(
                f"Could not parse Retry-After header '{retry_after}' as seconds or HTTP date: {e}"
            )
            return None


def _status_is_retryable(
    options: RetryOptions,
    request_class: RetryClass,
    status: int,
    inference_500_retries: int,
    *,
    path: str = "",
    has_retry_after: bool = False,
) -> bool:
    """Apply the status column of the policy table to one response."""
    if status not in options.retry_status_codes:
        return False
    if status == 503 and request_class is RetryClass.PAID:
        return has_retry_after or bool(_CAPACITY_503_POST_PATH.search(path))
    if status == 503:
        return True
    if request_class is RetryClass.IDEMPOTENT:
        return True
    if request_class is RetryClass.INFERENCE:
        if status == 502:
            return True
        if status == 500:
            return inference_500_retries < options.max_inference_500_retries
    return False


def _exception_is_retryable(
    options: RetryOptions, request_class: RetryClass, error: BaseException
) -> bool:
    """Apply the exception rows of the policy table to one failure."""
    if not any(isinstance(error, exc_type) for exc_type in options.retry_exceptions):
        return False
    return _is_connect_failure(error) or request_class is RetryClass.IDEMPOTENT


def create_retry_middleware(options: RetryOptions | None = None) -> ClientMiddlewareType:
    """
    Create the aiohttp middleware that applies the client's retry policy.

    The policy is described in the module docstring: each request is
    classified with :func:`classify_request` (or ``options.classifier``) and a
    failure is retried only when that class allows it. Paid generation
    requests are never resent once the server may have received them.

    Before each retry the failed response is released back to the connection
    pool, ``options.on_retry`` is called, and the next attempt carries the
    ``x-venice-sdk-retry-count`` header.

    Args:
        options: The construction-time policy. ``None`` uses
            :class:`RetryOptions` defaults. A ``client.with_retries(...)`` block
            overrides it for the requests made inside the block.

    Returns:
        An aiohttp client middleware.

    Example:
        >>> retry_middleware = create_retry_middleware(RetryOptions(max_attempts=1))
        >>> session = ClientSession(middlewares=[retry_middleware])
    """
    default_options = options if options is not None else RetryOptions()

    async def retry_middleware(request: Any, handler: Any) -> Any:
        active_override = _active_retry_options.get()
        options = active_override if active_override is not None else default_options

        method = request.method.upper()
        raw_headers = getattr(request, "headers", None)
        headers: Mapping[str, str] = raw_headers if isinstance(raw_headers, Mapping) else {}
        path = _request_path(request.url)
        if options.classifier is not None:
            request_class = options.classifier(method, path, headers)
        else:
            request_class = classify_request(
                method, path, headers, idempotent_methods=options.idempotent_methods
            )

        if request_class is not RetryClass.IDEMPOTENT and not options.retry_non_idempotent:
            return await handler(request)

        inference_500_retries = 0
        attempt = 0
        while True:
            if attempt:
                _resign_siwe_header(request)
                if isinstance(raw_headers, Mapping):
                    request.headers[RETRY_COUNT_HEADER] = str(attempt)

            try:
                response = await handler(request)
            except asyncio.CancelledError:
                raise
            except (ClientError, ServerTimeoutError, ValueError, OSError, TypeError) as e:
                if attempt >= options.max_attempts or not _exception_is_retryable(
                    options, request_class, e
                ):
                    raise
                delay = calculate_backoff_delay(
                    attempt,
                    options.base_delay,
                    options.exponential_base,
                    options.max_delay,
                    options.jitter_factor,
                )
                logger.info(
                    f"Retrying {request_class.value} request {method} {request.url} "
                    f"(attempt {attempt + 2}/{options.max_attempts + 1}) "
                    f"after {delay:.2f}s due to {type(e).__name__}: {e}"
                )
                if options.on_retry:
                    options.on_retry(attempt, delay, e)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            if attempt >= options.max_attempts or response.status not in (
                options.retry_status_codes
            ):
                return response
            retry_after = (
                parse_retry_after_header(response) if options.respect_retry_after else None
            )
            if not _status_is_retryable(
                options,
                request_class,
                response.status,
                inference_500_retries,
                path=path,
                has_retry_after=retry_after is not None,
            ):
                return response

            delay = calculate_backoff_delay(
                attempt,
                options.base_delay,
                options.exponential_base,
                options.max_delay,
                options.jitter_factor,
            )
            if retry_after is not None:
                if retry_after > options.max_retry_after:
                    logger.info(
                        f"Not retrying {method} {request.url}: status {response.status} "
                        f"asked for a {retry_after:.0f}s wait, above max_retry_after="
                        f"{options.max_retry_after:.0f}s"
                    )
                    return response
                if retry_after > 0:
                    delay = retry_after

            if response.status == 500:
                inference_500_retries += 1
            logger.info(
                f"Retrying {request_class.value} request {method} {request.url} "
                f"(attempt {attempt + 2}/{options.max_attempts + 1}) "
                f"after {delay:.2f}s due to status {response.status}"
            )
            release = getattr(response, "release", None)
            if callable(release):
                release()
            if options.on_retry:
                options.on_retry(attempt, delay, None)
            await asyncio.sleep(delay)
            attempt += 1

    return retry_middleware


# Export the main components
__all__ = [
    "RETRY_COUNT_HEADER",
    "RetryClass",
    "RetryOptions",
    "classify_request",
    "create_retry_middleware",
    "calculate_backoff_delay",
    "parse_retry_after_header",
    "_active_retry_options",
    "_active_siwe_resigner",
    "_SIWE_HEADER",
]
