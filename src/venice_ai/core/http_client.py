"""
Central HTTP Client for Venice AI

This module provides a single, centralized HTTP client that manages aiohttp.ClientSession
with consistent configuration, retry logic, and proper resource management across the entire SDK.

Key features:
- Single aiohttp.ClientSession for entire application
- Standardized headers (User-Agent, Auth)
- Consistent timeouts (30s default, configurable)
- Retry logic with exponential backoff
- Proper session cleanup on exit
- Connection pooling configuration
- No resource leaks
- Rate limit header extraction for distributed backend support
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import aiohttp

from ..middleware.retry import RetryOptions, create_retry_middleware
from .auth import create_auth_headers
from .config import VeniceAIConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _extract_rate_limit_headers(response: aiohttp.ClientResponse) -> dict[str, str]:
    """
    Extract all rate limit headers from response.

    This is called before status checking to ensure headers are available
    for the cached_rate_limit_headers attribute on RateLimitError.

    Note: aiohttp response.headers is a CIMultiDictProxy populated when
    the response status/headers are received. It's synchronously accessible
    and doesn't require the response body to be consumed.

    Args:
        response: The aiohttp ClientResponse to extract headers from.

    Returns:
        Dict with lowercase header keys and their values.
        Only includes headers that pass basic sanity checks.
    """
    headers: dict[str, str] = {}
    try:
        for key, value in response.headers.items():
            key_lower = key.lower()
            if (
                (key_lower.startswith("x-ratelimit-") or key_lower == "retry-after")
                and value
                and value.strip()
            ):
                headers[key_lower] = value.strip()
    except Exception as e:
        # Response may be in a bad state — log and return what we accumulated.
        logger.debug("rate-limit header extraction failed: %s", e)
    return headers


class VeniceHTTPClient:
    """
    A centralized HTTP client for the Venice AI SDK, designed to manage a
    single `aiohttp.ClientSession` for the entire application lifecycle.

    This class ensures that all HTTP requests are made with a consistent
    configuration, including standardized headers, timeouts, and retry logic.
    By managing a single session, it also provides proper resource management,
    including connection pooling and graceful cleanup.

    Key Features:
    - Manages a single `aiohttp.ClientSession` to avoid resource leaks.
    - Standardized headers, including `User-Agent` and `Authorization`.
    - Configurable timeouts and retry logic with exponential backoff.
    - Centralized connection pooling configuration.
    """

    def __init__(
        self,
        config: VeniceAIConfig,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        trust_env: bool | None = None,
        connector_limit: int | None = None,
        connector_limit_per_host: int | None = None,
        auto_decompress: bool | None = None,
        cookie_jar: aiohttp.CookieJar | None = None,
        skip_auto_headers: list | None = None,
        http_transport_options: dict[str, Any] | None = None,
        retry_options: RetryOptions | None = None,
        proxy: str | None = None,
    ):
        """
        Initialize the central HTTP client.

        Args:
            config: Venice AI configuration
            api_key: API key for authentication
            base_url: Base URL for API requests
            headers: Additional headers to include
            trust_env: Whether to trust environment variables for proxy config
            connector_limit: Global connection pool limit
            connector_limit_per_host: Per-host connection limit
            auto_decompress: Whether to auto-decompress responses
            cookie_jar: A custom aiohttp.CookieJar for managing cookies
            skip_auto_headers: Headers to skip from auto-generation
            http_transport_options: Additional transport options
            retry_options: Retry configuration
            proxy: URL of an HTTP proxy every request of the session is sent through
        """
        self._config = config
        self._api_key = api_key
        self._base_url = base_url or config.api_base_url
        self._custom_headers = headers or {}
        self._trust_env = trust_env
        self._connector_limit = connector_limit
        self._connector_limit_per_host = connector_limit_per_host
        self._auto_decompress = auto_decompress
        self._cookie_jar = cookie_jar
        self._skip_auto_headers = skip_auto_headers
        self._http_transport_options = http_transport_options or {}
        self._retry_options = retry_options
        self._proxy = proxy

        # Session management
        self._session: aiohttp.ClientSession | None = None
        self._is_closed = False
        self._session_lock = asyncio.Lock()

        # Default timeout from config
        self._default_timeout = aiohttp.ClientTimeout(total=config.http_client.timeout)

        # Pre-compute static headers for performance (header marshalling optimization)
        self._static_headers = self._build_static_headers()

    async def __aenter__(self) -> "VeniceHTTPClient":
        """Async context manager entry."""
        await self.get_session()
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb) -> None:
        """Async context manager exit with proper cleanup."""
        await self.close()

    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get or create the aiohttp session with lazy initialization.

        Returns:
            The configured aiohttp ClientSession

        Raises:
            RuntimeError: If client has been closed
        """
        if self._is_closed:
            raise RuntimeError("HTTP client has been closed")

        if self._session is None:
            async with self._session_lock:
                # Double-check pattern for thread safety
                if self._session is None:
                    self._session = self._build_session()

        return self._session

    def _build_session(self) -> aiohttp.ClientSession:
        """
        Build the aiohttp ClientSession with standardized configuration.

        This method consolidates all the session building logic
        into a single, centralized location.
        """
        # Create the connector with appropriate settings
        connector_kwargs: dict[str, Any] = {}

        # High global limit and no per-host limit unless configured: every SDK
        # request targets one host, so a per-host limit would be the real cap.
        if self._connector_limit is not None:
            connector_kwargs["limit"] = self._connector_limit
        else:
            connector_kwargs["limit"] = 1000  # High global limit

        if self._connector_limit_per_host is not None:
            connector_kwargs["limit_per_host"] = self._connector_limit_per_host
        else:
            connector_kwargs["limit_per_host"] = 0  # No per-host limit - let scheduler control

        # Add any transport options
        connector_kwargs.update(self._http_transport_options)

        connector = aiohttp.TCPConnector(**connector_kwargs)

        # Prepare session kwargs
        # Note: aiohttp.ClientSession requires base_url to have a trailing slash
        base_url_with_slash = (
            self._base_url if self._base_url.endswith("/") else f"{self._base_url}/"
        )

        session_kwargs: dict[str, Any] = {
            "base_url": base_url_with_slash,
            "connector": connector,
            "timeout": self._default_timeout,
        }

        if self._proxy is not None:
            session_kwargs["proxy"] = self._proxy

        # trust_env belongs to ClientSession, not TCPConnector
        if self._trust_env is not None:
            session_kwargs["trust_env"] = self._trust_env

        # Use pre-computed static headers (performance optimization)
        session_kwargs["headers"] = self._static_headers.copy()

        # Add optional parameters
        if self._auto_decompress is not None:
            session_kwargs["auto_decompress"] = self._auto_decompress

        if self._cookie_jar is not None:
            session_kwargs["cookie_jar"] = self._cookie_jar

        if self._skip_auto_headers is not None:
            session_kwargs["skip_auto_headers"] = self._skip_auto_headers

        # Add retry middleware (enabled by default with sensible defaults)
        # When retry_options is None, create_retry_middleware uses RetryOptions()
        # which provides: 3 retries, backoff on 429/500/502/503/504, POST retries enabled
        retry_middleware = create_retry_middleware(self._retry_options)
        session_kwargs["middlewares"] = [retry_middleware]

        logger.debug(
            "Creating aiohttp ClientSession with base_url=%s, timeout=%s",
            self._base_url,
            self._default_timeout.total,
        )

        return aiohttp.ClientSession(**session_kwargs)

    def _build_static_headers(self) -> dict[str, str]:
        """
        Build static HTTP headers that don't change per-request.

        This is a performance optimization that pre-computes headers once
        instead of rebuilding them on every request. Static headers include:
        - Accept header
        - User-Agent
        - Authorization (if API key provided)
        - Custom headers from initialization

        Returns:
            Dictionary of static headers
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": self._config.http_client.user_agent,
        }

        # Add authentication headers if API key is provided
        if self._api_key:
            headers.update(create_auth_headers(self._api_key))

        # Merge custom headers (custom headers can override defaults)
        headers.update(self._custom_headers)

        return headers

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: Any | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """
        Make an HTTP request using the central session.

        Performance Optimization:
            Static headers are pre-computed at initialization. Only dynamic
            per-request headers need to be merged, improving request throughput.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL path (relative to base_url) or absolute URL
            params: Query parameters
            data: Request body data
            json: JSON data to send
            headers: Additional headers for this request (merged with static headers)
            timeout: Request timeout (overrides default)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            aiohttp ClientResponse

        Raises:
            aiohttp.ClientError: For HTTP-related errors
            RuntimeError: If client has been closed
        """
        session = await self.get_session()

        # Merge per-request headers with static headers if needed
        if headers:
            # Create a new dict to avoid modifying the original
            merged_headers = self._static_headers.copy()
            merged_headers.update(headers)
            kwargs["headers"] = merged_headers

        # Handle timeout
        if timeout is not None:
            if isinstance(timeout, (int, float)):
                timeout = aiohttp.ClientTimeout(total=timeout)
            kwargs["timeout"] = timeout

        logger.debug(
            "Making %s request to %s with timeout=%s",
            method.upper(),
            url,
            timeout.total if isinstance(timeout, aiohttp.ClientTimeout) else timeout,
        )

        return await session.request(
            method=method, url=url, params=params, data=data, json=json, **kwargs
        )

    @asynccontextmanager
    async def stream_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ):
        """
        Context manager for streaming requests.

        Usage:
            async with http_client.stream_request("GET", "/stream") as response:
                async for chunk in response.content.iter_chunked(1024):
                    # Process chunk
        """
        response = await self.request(method, url, **kwargs)
        try:
            yield response
        finally:
            response.close()

    async def close(self) -> None:
        """
        Close the HTTP client and clean up resources.

        This should be called when the client is no longer needed to prevent
        resource leaks.
        """
        if self._is_closed:
            return

        self._is_closed = True

        if self._session is not None:
            if not self._session.closed:
                await self._session.close()

                # Wait a bit for the underlying connection to close
                # This is recommended by aiohttp documentation
                await asyncio.sleep(0.1)

            self._session = None

        logger.debug("HTTP client closed successfully")

    @property
    def is_closed(self) -> bool:
        """Check if the HTTP client has been closed."""
        return self._is_closed

    @property
    def base_url(self) -> str | None:
        """Get the base URL for this client."""
        return self._base_url

    @property
    def default_timeout(self) -> aiohttp.ClientTimeout:
        """Get the default timeout for requests."""
        return self._default_timeout
