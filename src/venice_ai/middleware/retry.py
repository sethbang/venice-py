"""
Advanced retry middleware for aiohttp with intelligent exponential backoff and jitter.

This module provides a sophisticated retry mechanism specifically designed for the Venice AI
client's HTTP communication layer. It integrates seamlessly with aiohttp's middleware system
to handle transient failures, rate limiting, and network issues with intelligent retry strategies.

## Key Features

- **Exponential Backoff**: Implements exponential backoff with configurable base delay and multiplier
- **Jitter Support**: Adds randomization to prevent thundering herd problems
- **Smart Retry Logic**: Differentiates between idempotent and non-idempotent HTTP methods
- **Rate Limit Awareness**: Respects Retry-After headers from API responses
- **Configurable Exception Handling**: Customizable set of exceptions that trigger retries
- **Comprehensive Logging**: Detailed logging for monitoring and debugging retry behavior

## Integration with Venice AI Client

The retry middleware is automatically integrated into the Venice AI client's HTTP session
during initialization. It operates transparently at the transport layer, intercepting
failed requests and applying retry logic before propagating failures to higher-level code.

## Retry Strategy

The module implements a sophisticated retry strategy that considers:
1. **HTTP Method Safety**: Only retries idempotent methods (GET, PUT, DELETE, etc.) by default
2. **Status Code Analysis**: Retries on specific HTTP status codes (5xx errors by default)
3. **Exception Type Filtering**: Retries on network timeouts and connection errors
4. **Exponential Backoff**: Increases delay between attempts to reduce server load
5. **Jitter Application**: Adds randomness to prevent synchronized retry storms

## Performance Considerations

The retry mechanism is designed to be efficient and respectful of server resources:
- Caps maximum delay to prevent excessive wait times
- Uses jitter to distribute retry attempts across time
- Respects server-provided Retry-After headers
- Logs retry attempts for monitoring and debugging
"""

import asyncio
import contextvars
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from aiohttp import ClientError, ClientResponse, ServerTimeoutError
from aiohttp.typedefs import Middleware

logger = logging.getLogger(__name__)


@dataclass
class RetryOptions:
    """
    Comprehensive configuration for retry behavior and exponential backoff strategies.

    This class provides fine-grained control over how the retry middleware handles
    failed requests, including timing strategies, condition filtering, and monitoring hooks.

    Attributes:
        max_attempts: Maximum number of retry attempts (excluding the initial request).
            For example, max_attempts=3 means up to 4 total attempts (1 initial + 3 retries).
            Higher values increase resilience but may delay error propagation.

        retry_status_codes: Set of HTTP status codes that should trigger a retry attempt.
            Default includes server errors (5xx). Rate limiting (429) is intentionally excluded
            because SimpleRateLimiter handles 429 retries with per-model state tracking,
            exponential backoff, and Retry-After header support. Common additions might
            include 408 (Request Timeout) or 413 (Payload Too Large) depending on use case.

        retry_exceptions: List of exception types that should trigger retry attempts.
            Focuses on transient network issues and timeouts that are likely to resolve
            on subsequent attempts. Does not include programming errors or authentication failures.

        base_delay: Base delay in seconds for exponential backoff calculation.
            This is the starting delay for the first retry attempt. Subsequent attempts
            use exponential_base^attempt * base_delay. Lower values provide faster retries
            but may overwhelm struggling servers.

        max_delay: Maximum delay in seconds to cap exponential growth.
            Prevents exponential backoff from creating excessively long delays.
            Helps maintain reasonable response times even after multiple failures.

        exponential_base: Base multiplier for exponential backoff calculation.
            Determines how quickly delays increase. 2.0 doubles delay each attempt,
            while 1.5 provides more gradual increases. Higher values back off more aggressively.

        jitter_factor: Randomization factor (0.0 to 1.0) to prevent thundering herd problems.
            Adds random variation to delays to prevent multiple clients from retrying
            simultaneously. 0.1 means ±10% random variation. Higher values increase
            randomization but may make retry timing less predictable.

        respect_retry_after: Whether to honor Retry-After headers from server responses.
            When True, server-provided retry delays override calculated exponential backoff.
            Recommended for APIs that provide intelligent rate limiting guidance.

        max_retry_after: Maximum seconds to wait for server-provided Retry-After values.
            Prevents malicious or misconfigured servers from forcing excessive delays.
            Acts as a safety cap on server-directed retry timing.

        idempotent_methods: Set of HTTP methods considered safe to retry automatically.
            These methods should not have side effects when repeated. POST is notably
            excluded by default since it typically creates or modifies resources.

        retry_non_idempotent: Whether to retry non-idempotent methods like POST.
            Default is True because Venice API endpoints (chat completions, embeddings,
            image generation) are effectively idempotent and safe to retry on transient
            errors. Set to False if your use case involves non-idempotent operations.

        on_retry: Optional callback function for monitoring or logging retry attempts.
            Called with (attempt_number, delay_seconds, exception_or_none) for each retry.
            Useful for metrics collection, alerting, or debugging retry behavior.
    """

    # Maximum number of retry attempts (not including the initial request)
    max_attempts: int = 3

    # HTTP status codes that should trigger a retry
    retry_status_codes: set[int] = field(default_factory=lambda: {500, 502, 503, 504})

    # Exception types that should trigger a retry
    retry_exceptions: list[type[Exception]] = field(
        default_factory=lambda: [
            TimeoutError,
            ServerTimeoutError,
            ClientError,
        ]
    )

    # Base delay in seconds for exponential backoff
    base_delay: float = 1.0

    # Maximum delay in seconds (caps exponential growth)
    max_delay: float = 60.0

    # Exponential base for backoff calculation
    exponential_base: float = 2.0

    # Jitter factor (0.0 to 1.0) - adds randomness to prevent thundering herd
    jitter_factor: float = 0.1  # Using 0.1 for backward compatibility with existing configs

    # Whether to respect Retry-After headers
    respect_retry_after: bool = True

    # Maximum seconds to wait for Retry-After (to prevent abuse)
    max_retry_after: float = 120.0

    # HTTP methods that are considered idempotent and safe to retry
    idempotent_methods: set[str] = field(
        default_factory=lambda: {"GET", "PUT", "DELETE", "HEAD", "OPTIONS", "TRACE"}
    )

    # Whether to retry non-idempotent methods (like POST)
    # Default True because Venice API endpoints (chat completions, embeddings,
    # image generation) are effectively idempotent and safe to retry on transient errors.
    retry_non_idempotent: bool = True

    # Callback for logging or monitoring retries
    on_retry: Callable[[int, float, Exception | None], None] | None = None


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
    Calculate intelligent retry delay using exponential backoff with jitter.

    This function implements a sophisticated backoff strategy that balances quick recovery
    from transient issues with respectful behavior toward struggling servers. The algorithm
    combines exponential backoff (to reduce load on failing services) with jitter
    (to prevent thundering herd problems when multiple clients retry simultaneously).

    The calculation process:
    1. Compute exponential delay: base_delay * (exponential_base ^ attempt)
    2. Cap the result at max_delay to prevent excessive waits
    3. Apply jitter as random variation: ±(jitter_factor * delay)
    4. Ensure the final delay is never negative

    Args:
        attempt: The current attempt number (0-based indexing).
            attempt=0 for first retry, attempt=1 for second retry, etc.
        base_delay: Base delay in seconds for the exponential calculation.
            This is the delay used for the first retry attempt before exponential growth.
        exponential_base: Multiplicative base for exponential backoff.
            Common values are 2.0 (doubling) or 1.5 (50% increase per attempt).
        max_delay: Maximum delay in seconds to cap exponential growth.
            Prevents extremely long delays that could impact user experience.
        jitter_factor: Randomization factor between 0.0 and 1.0.
            0.0 = no randomness, 1.0 = up to 100% variation in either direction.

    Returns:
        Calculated delay in seconds before the next retry attempt.
        Always returns a non-negative value, even with maximum jitter applied.

    Example:
        >>> calculate_backoff_delay(0, 1.0, 2.0, 60.0, 0.1)
        # First retry: ~1.0 seconds ± 10% jitter
        >>> calculate_backoff_delay(2, 1.0, 2.0, 60.0, 0.1)
        # Third retry: ~4.0 seconds ± 10% jitter
    """
    # Calculate exponential backoff
    exponential_delay = base_delay * (exponential_base**attempt)

    # Cap at maximum delay
    capped_delay = min(exponential_delay, max_delay)

    # Add jitter to prevent thundering herd problem
    # Jitter adds randomness in the range [-jitter_factor * delay, +jitter_factor * delay]
    if jitter_factor > 0:
        jitter_range = capped_delay * jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)  # nosec B311
        final_delay = max(0, capped_delay + jitter)
    else:
        final_delay = capped_delay

    return final_delay


def parse_retry_after_header(response: ClientResponse) -> float | None:
    """
    Parse the Retry-After header from an HTTP response to determine server-suggested delay.

    The Retry-After header is commonly used by APIs to indicate when a client should
    retry a request, particularly for rate limiting (429) and temporary service
    unavailability (503) responses. This function handles both formats specified
    in RFC 7231.

    The header can contain either:
    - An integer number of seconds to wait (e.g., "Retry-After: 120")
    - An HTTP-date timestamp indicating when to retry (e.g., "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT")

    Args:
        response: The aiohttp ClientResponse object containing the HTTP headers.
            Must be a valid response object with accessible headers.

    Returns:
        Number of seconds to wait before retrying, or None if:
        - The Retry-After header is not present in the response
        - The header value cannot be parsed as a valid delay
        - The header contains an unrecognized date format

    Note:
        HTTP-date parsing uses email.utils.parsedate_to_datetime() which follows
        RFC 2822 and RFC 5322 date formats commonly used in HTTP headers.
    """
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        # First, try to parse as integer seconds
        return float(retry_after)
    except ValueError:
        # If that fails, try to parse as HTTP date
        try:
            retry_date = parsedate_to_datetime(retry_after)
            # Calculate seconds until the retry date
            now = datetime.now(UTC)
            # Ensure retry_date is timezone-aware
            if retry_date.tzinfo is None:
                retry_date = retry_date.replace(tzinfo=UTC)
            delay = (retry_date - now).total_seconds()
            # Return delay only if it's positive (in the future)
            return max(0.0, delay)
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(
                f"Could not parse Retry-After header '{retry_after}' as seconds or HTTP date: {e}"
            )
            return None


def create_retry_middleware(options: RetryOptions | None = None) -> Middleware:
    """
    Create an intelligent aiohttp middleware that implements advanced retry logic.

    This function returns a middleware component that integrates into aiohttp's request
    pipeline to automatically handle transient failures with sophisticated retry strategies.
    The middleware operates transparently, intercepting failed requests and applying
    configurable retry logic before either succeeding or propagating the final failure.

    The middleware implements several layers of intelligence:

    **Request Analysis**: Determines whether a request is safe to retry based on:
    - HTTP method idempotency (GET, PUT, DELETE are safe; POST typically isn't)
    - Configuration settings for non-idempotent method handling

    **Failure Detection**: Identifies retryable failures through:
    - HTTP status code analysis (rate limiting, server errors)
    - Exception type filtering (timeouts, network errors)
    - Exclusion of permanent failures (authentication, client errors)

    **Retry Strategy**: Applies intelligent backoff using:
    - Exponential backoff to reduce load on struggling servers
    - Jitter to prevent thundering herd problems
    - Server-provided Retry-After header respect
    - Configurable maximum delays and attempt limits

    **Monitoring Integration**: Provides visibility through:
    - Detailed logging of retry attempts and decisions
    - Optional callback hooks for metrics collection
    - Exception preservation for proper error propagation

    Args:
        options: Optional RetryOptions instance for customizing retry behavior.
            If None, uses default retry configuration suitable for most API interactions.
            Default behavior retries up to 3 times with exponential backoff starting
            at 1 second, only for idempotent methods and common transient failures.

    Returns:
        An aiohttp middleware function that can be added to ClientSession middleware list.
        The middleware function signature matches aiohttp's middleware protocol:
        async def middleware(request, handler) -> response

    Example:
        >>> retry_middleware = create_retry_middleware(
        ...     RetryOptions(max_attempts=5, base_delay=0.5)
        ... )
        >>> session = ClientSession(middlewares=[retry_middleware])

    Note:
        The middleware preserves the original request semantics - successful requests
        pass through unchanged, and final failures raise the original exception with
        full context about retry attempts logged.
    """
    default_options = options if options is not None else RetryOptions()

    async def retry_middleware(request: Any, handler: Any) -> Any:
        """
        Middleware implementation that wraps requests with retry logic.

        Each request resolves its effective :class:`RetryOptions` at entry,
        consulting :data:`_active_retry_options` first (set by a
        ``with_retries()`` context manager on the client) and falling back
        to the construction-time *default_options* otherwise.
        """
        # Resolve effective options for this request. ContextVar.get()
        # honors per-task scoping established by client.with_retries().
        active_override = _active_retry_options.get()
        options = active_override if active_override is not None else default_options

        method = request.method.upper()

        # Check if we should retry this method
        if not options.retry_non_idempotent and method not in options.idempotent_methods:
            # Non-idempotent method and we're not configured to retry them
            return await handler(request)

        last_exception = None

        for attempt in range(options.max_attempts + 1):  # +1 for the initial attempt
            # aiohttp replays the same request object, so a wallet-authenticated
            # retry would resend a nonce the server has already seen. Re-signing
            # here — after the backoff sleep — keeps the envelope fresh at the
            # moment it is actually sent. No-op for every other request.
            if attempt:
                _resign_siwe_header(request)

            try:
                # Make the request
                response = await handler(request)

                # Check if we should retry based on status code
                if response.status in options.retry_status_codes and attempt < options.max_attempts:
                    # Calculate delay
                    delay = calculate_backoff_delay(
                        attempt,
                        options.base_delay,
                        options.exponential_base,
                        options.max_delay,
                        options.jitter_factor,
                    )

                    # Check for Retry-After header
                    if options.respect_retry_after:
                        retry_after = parse_retry_after_header(response)
                        if retry_after is not None:
                            # Use Retry-After delay, but cap it at max_retry_after
                            delay = min(retry_after, options.max_retry_after)

                    logger.info(
                        f"Retrying request {method} {request.url} "
                        f"(attempt {attempt + 1}/{options.max_attempts + 1}) "
                        f"after {delay:.2f}s due to status {response.status}"
                    )

                    # Call the retry callback if provided
                    if options.on_retry:
                        options.on_retry(attempt, delay, None)

                    # Wait before retrying
                    await asyncio.sleep(delay)
                    continue

                # Success or non-retryable status code
                return response

            except asyncio.CancelledError:
                raise  # Always re-raise for graceful shutdown
            except (
                ClientError,
                ServerTimeoutError,
                ValueError,
                OSError,
                TypeError,
            ) as e:
                last_exception = e

                # Check if this exception type should trigger a retry
                should_retry = any(isinstance(e, exc_type) for exc_type in options.retry_exceptions)

                if should_retry and attempt < options.max_attempts:
                    # Calculate delay
                    delay = calculate_backoff_delay(
                        attempt,
                        options.base_delay,
                        options.exponential_base,
                        options.max_delay,
                        options.jitter_factor,
                    )

                    logger.info(
                        f"Retrying request {method} {request.url} "
                        f"(attempt {attempt + 1}/{options.max_attempts + 1}) "
                        f"after {delay:.2f}s due to {type(e).__name__}: {e}"
                    )

                    # Call the retry callback if provided
                    if options.on_retry:
                        options.on_retry(attempt, delay, e)

                    # Wait before retrying
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable exception or max attempts reached
                raise

        # If we get here, we've exhausted all retries
        if last_exception:
            raise last_exception

        # This shouldn't happen, but just in case
        raise RuntimeError(
            f"Max retries ({options.max_attempts}) exceeded for {method} {request.url}"
        )

    return retry_middleware


# Export the main components
__all__ = [
    "RetryOptions",
    "create_retry_middleware",
    "calculate_backoff_delay",
    "_active_retry_options",
    "_active_siwe_resigner",
    "_SIWE_HEADER",
]
