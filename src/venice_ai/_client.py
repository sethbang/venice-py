from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import aiohttp
from multidict import CIMultiDict
from pydantic import BaseModel, ValidationError
from yarl import URL

from . import _constants
from ._base_url import DEFAULT_API_VERSION, normalize_base_url
from .core.auth import create_auth_headers, resolve_api_key
from .core.http_client import ConnectionLimits, _extract_rate_limit_headers
from .exceptions import (
    APIError,
    APIResponseProcessingError,
    APIResponseValidationError,
    _make_status_error,
)
from .middleware import RetryOptions
from .middleware.retry import _SIWE_HEADER, _active_siwe_resigner
from .rate_limiting import RateLimiterProtocol
from .resources.api_keys import ApiKeys
from .resources.audio import Audio
from .resources.augment import Augment
from .resources.billing import Billing
from .resources.characters import Characters
from .resources.chat import ChatResource
from .resources.crypto import Crypto
from .resources.decisions import Decisions
from .resources.embeddings import Embeddings
from .resources.image import Image
from .resources.models import Models
from .resources.music import Music
from .resources.responses import Responses
from .resources.tee import Tee
from .resources.video import Video
from .resources.voice_changer import VoiceChanger
from .resources.x402 import X402
from .streaming import Stream
from .utils import NOT_GIVEN, NotGiven, serialize_form_value
from .utils.errors import read_body, resolve_timeout, wrap_aiohttp_errors

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .auth.x402 import X402Auth
    from .auth.x402_solana import SolanaX402Auth
    from .core.config import VeniceAIConfig
    from .core.http_client import VeniceHTTPClient
    from .costs import CostTracker


class VeniceClient:
    """
    Asynchronous client for the Venice AI API.

    Provides an interface for interacting with all features of the Venice AI API.
    This client is designed for asynchronous applications and uses `aiohttp` for
    HTTP requests. It supports rate limiting, automatic retries, and other
    advanced features.

    For synchronous applications, use the `VeniceClient` from the root of the
    library.
    """

    _session: aiohttp.ClientSession | None
    _session_lock: asyncio.Lock
    _is_closed: bool = False
    _should_close_session: bool

    chat: ChatResource
    responses: Responses
    models: Models
    image: Image
    audio: Audio
    music: Music
    billing: Billing
    decisions: Decisions
    embeddings: Embeddings
    api_keys: ApiKeys
    characters: Characters
    video: Video
    voice_changer: VoiceChanger
    crypto: Crypto
    tee: Tee

    rate_limiter: RateLimiterProtocol | None

    # -------------------------------------------------------------------
    # ClientProtocol interface
    # -------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Base URL of the API (ClientProtocol implementation)."""
        return str(self._base_url)

    @property
    def timeout(self) -> float:
        """Request timeout in seconds (ClientProtocol implementation)."""
        if self._timeout.total is not None:
            return self._timeout.total
        return 60.0  # Default fallback

    @property
    def retry_options(self) -> RetryOptions | None:
        """The retry policy requests from this task are sent with.

        Inside a :meth:`with_retries` block this is the block's override, since
        that is what those requests use; otherwise it is the construction-time
        policy resolved from ``retry_options``, ``max_retries`` and
        ``config.http_client``, with :class:`RetryOptions` defaults for anything
        unset. ``None`` when the client was given its own ``http_client``
        session: the SDK installs no retry middleware on a caller's session.
        :class:`RetryOptions` is frozen, so the returned policy cannot be
        changed; derive a variant with ``dataclasses.replace()``.
        """
        if self._venice_http_client is None:
            return None
        from .middleware.retry import _active_retry_options

        active = _active_retry_options.get()
        if active is not None:
            return active
        if isinstance(self._retry_options, RetryOptions):
            return self._retry_options
        return RetryOptions()

    @property
    def connection_limits(self) -> ConnectionLimits:
        """The connection-pool limits of this client's HTTP session.

        Read from the caller's session connector when an ``http_client`` was
        given, otherwise resolved from ``http_transport_options`` (``limit``,
        ``limit_per_host``), then ``connector_limit``,
        ``connector_limit_per_host`` and ``config.http_client.max_connections``,
        with the SDK defaults (1000 in total, no per-host cap).
        """
        if self._venice_http_client is not None:
            return self._venice_http_client.connection_limits
        connector = self._session.connector if self._session is not None else None
        return ConnectionLimits(
            limit=getattr(connector, "limit", 0) or 0,
            limit_per_host=getattr(connector, "limit_per_host", 0) or 0,
        )

    def get_headers(self) -> dict[str, str]:
        """Get default headers (ClientProtocol implementation)."""
        if self._headers is not NOT_GIVEN and isinstance(self._headers, dict):
            return self._headers  # Already validated as dict by isinstance check
        return {}

    @staticmethod
    def _resolve_not_given[U](value: U | NotGiven, default: U | None = None) -> U | None:
        """
        Convert NOT_GIVEN sentinel to a default value, otherwise return the actual value.

        This helper simplifies the common pattern of checking for NOT_GIVEN and casting.

        Args:
            value: The value to resolve, which may be NOT_GIVEN or an actual value
            default: The default value to return if value is NOT_GIVEN (defaults to None)

        Returns:
            The default if value is NOT_GIVEN, otherwise the actual value cast to type U
        """
        return default if value is NOT_GIVEN else cast(U, value)

    # -------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------

    def __init__(
        self,
        *,
        api_key: str | None = None,
        auth: X402Auth | SolanaX402Auth | None = None,
        base_url: str | URL | None = None,
        http_client: aiohttp.ClientSession | None = None,
        timeout: int | float | aiohttp.ClientTimeout | None = _constants.DEFAULT_TIMEOUT,
        default_timeout: int | float | aiohttp.ClientTimeout | None = None,
        max_retries: int | None = None,
        rate_limiter: RateLimiterProtocol | None = None,
        config: VeniceAIConfig | None = None,
        http_transport_options: dict[str, Any] | None = None,
        rate_limiter_config: dict[str, Any] | None = None,
        rate_limiter_config_path: str | Path | None = None,
        proxy: str | NotGiven = NOT_GIVEN,
        connector_limit: int | NotGiven = NOT_GIVEN,
        connector_limit_per_host: int | NotGiven = NOT_GIVEN,
        trust_env: bool | NotGiven = NOT_GIVEN,
        auto_decompress: bool | NotGiven = NOT_GIVEN,
        cookie_jar: aiohttp.CookieJar | NotGiven = NOT_GIVEN,
        headers: dict[str, str] | NotGiven = NOT_GIVEN,
        skip_auto_headers: list[str] | NotGiven = NOT_GIVEN,
        retry_options: RetryOptions | NotGiven = NOT_GIVEN,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """
        Initializes the asynchronous VeniceClient.

        This constructor is designed for dependency injection. For typical use
        cases, it is recommended to use the `VeniceClientFactory` to create
        a client instance, as it will handle the creation and configuration of
        all dependencies.

        Args:
            api_key: The API key for authenticating with Venice AI. The key
                is resolved in one order, the first value that is not ``None``
                wins: this argument, then ``config.api_key`` when a ``config``
                is passed, then the ``VENICE_API_KEY`` environment variable.
                An explicit empty string means "no API key" and stops the
                lookup, which lets a wallet ``auth`` take over even when
                ``VENICE_API_KEY`` is set. When no key resolves and ``auth`` is
                unset, a :class:`ValueError` is raised. The key is sent as ``Authorization: Bearer`` on every
                API request, including requests made through a caller-supplied
                ``http_client``. :meth:`fetch_external` sends it only to URLs on
                the API's own origin, never to CDN or pre-signed media hosts.
            auth: Optional wallet auth for SIWE/SIWX-based authentication
                (Mode 2 — no API key, prepaid balance) —
                :class:`~venice_ai.auth.x402.X402Auth` for EVM wallets or
                :class:`~venice_ai.auth.x402_solana.SolanaX402Auth` for
                Solana. When set with ``api_key=None``,
                the SDK skips ``Authorization: Bearer`` and signs a fresh
                ``X-Sign-In-With-X`` header per request. When
                both ``api_key`` and ``auth`` are set, the API key wins
                for default request auth; the auth instance is stored
                so callers can pass it explicitly to per-call ``auth=``
                kwargs (e.g., ``client.x402.balance(auth=auth)``).
            base_url: The API root, version path included
                (``https://api.venice.ai/api/v1``). A bare host such as
                ``https://api.venice.ai`` gets ``/api/v1`` appended; any other
                path is used as given. A gateway must therefore expose the API
                under a path (``https://gateway.example.com/venice``): a URL
                without one, trailing slash or not, always gets ``/api/v1``
                appended, so an API served at a host's root cannot be
                addressed. Defaults to ``config.api_base_url`` when
                a ``config`` is passed, else the ``VENICE_API_BASE_URL``
                environment variable, else the official API.
            http_client: An optional pre-configured ``aiohttp.ClientSession``.
                If provided, the client will not manage the session lifecycle.
                The session is never modified: authentication headers are
                attached to each API request rather than to the session, and
                request URLs are absolute, so the session needs no
                ``base_url`` of its own.
            timeout: The default request timeout in seconds.
            default_timeout: A pre-configured ``aiohttp.ClientTimeout`` object
                that overrides the ``timeout`` setting.
            max_retries: The number of retries after the first attempt for a
                retryable failure (``0`` disables retries). Overrides
                ``config.http_client.max_retries``; ignored when
                ``retry_options`` is given.
            rate_limiter: The ``RateLimiterProtocol`` instance for rate limiting
                (injected by factory).
            config: A ``VeniceAIConfig`` object for configuring the central HTTP
                client: timeout, connection pool size, User-Agent and the
                retry policy (``max_retries`` / ``retry_backoff_factor``), plus
                the API key (``config.api_key``, used when ``api_key`` is not
                passed) and the base URL (``config.api_base_url``).
            http_transport_options: Additional options for the
                ``aiohttp.TCPConnector``, allowing fine-tuning of the HTTP
                transport layer. A ``limit`` or ``limit_per_host`` here takes
                precedence over ``connector_limit`` / ``connector_limit_per_host``.
            rate_limiter_config: A dictionary with rate limiter settings.
            rate_limiter_config_path: The file path to a rate limiter
                configuration file.
            proxy: The URL of an HTTP proxy that every request is sent through.
            connector_limit: The maximum number of simultaneous connections
                for the aiohttp connector. Overrides
                ``config.http_client.max_connections``.
            connector_limit_per_host: The maximum number of simultaneous
                connections to a single host. Unlimited unless given.
            trust_env: If ``True``, the ``aiohttp`` connector will trust
                environment variables for proxy settings.
            auto_decompress: If ``True``, ``aiohttp`` will automatically
                decompress response content.
            cookie_jar: A custom ``aiohttp.CookieJar`` for managing cookies.
            headers: Default headers to include in every API request, including
                requests sent through a caller-supplied ``http_client``.
            skip_auto_headers: A list of headers that ``aiohttp`` should not
                automatically add.
            retry_options: Configuration for the request retry strategy. Takes
                precedence over ``max_retries`` and ``config.http_client``.
            cost_tracker: Optional :class:`CostTracker` that the SDK will
                feed every chat-completion and embeddings response into,
                automatically. When ``None`` (default) no tracking is wired.
        """
        # --- API key / auth resolution ---
        # Either an api_key (Bearer) or a wallet auth (X402Auth / SolanaX402Auth,
        # SIWE/SIWX) is required. When both are set, the api_key wins for default
        # request auth; the auth instance is stored for explicit per-call use
        # (e.g., on x402 reads).
        self._api_key = resolve_api_key(api_key, config)

        # Store SIWE/SIWX auth (Mode 2) if provided. The auth classes are
        # lazily imported so neither the [x402] nor [x402-solana] extra is
        # forced on users who only need Bearer auth.
        self._auth = auth

        if not self._api_key and self._auth is None:
            raise ValueError(
                "No authentication provided. Pass api_key= to VeniceClient(), set "
                "api_key on the VeniceAIConfig passed as config=, set the "
                "VENICE_API_KEY environment variable, or pass a wallet "
                "auth for SIWE/SIWX authentication — auth=X402Auth(...) (EVM, "
                "requires the [x402] extra) or auth=SolanaX402Auth(...) (Solana, "
                "requires the [x402-solana] extra)."
            )

        # --- Base URL resolution ---
        # One form everywhere: the API root (see venice_ai._base_url). An
        # explicit base_url wins, then the config's api_base_url, then the
        # VENICE_API_BASE_URL environment variable the config also reads.
        if base_url is None or base_url == "":
            if config is not None:
                base_url = config.api_base_url
            else:
                base_url = os.environ.get("VENICE_API_BASE_URL") or _constants.DEFAULT_BASE_URL
        api_version = config.api_version if config is not None else DEFAULT_API_VERSION
        base_url = normalize_base_url(str(base_url), api_version)
        self._base_url = URL(base_url + "/")

        # --- Timeout resolution ---
        effective_timeout = default_timeout if default_timeout is not None else timeout

        if isinstance(effective_timeout, bool):
            self._timeout = _constants.DEFAULT_TIMEOUT
        elif isinstance(effective_timeout, int | float):
            self._timeout = aiohttp.ClientTimeout(total=float(effective_timeout))
        elif isinstance(effective_timeout, aiohttp.ClientTimeout):
            self._timeout = effective_timeout
        else:
            self._timeout = _constants.DEFAULT_TIMEOUT

        self._http_transport_options = http_transport_options

        # Store aiohttp-specific options
        self._proxy = proxy
        self._connector_limit = connector_limit
        self._connector_limit_per_host = connector_limit_per_host
        self._trust_env = trust_env
        self._auto_decompress = auto_decompress
        self._cookie_jar = cookie_jar
        self._headers = headers
        self._skip_auto_headers = skip_auto_headers
        self._retry_options = self._resolve_retry_options(retry_options, max_retries, config)
        self._cost_tracker = cost_tracker

        # --- Rate limiter configuration ---
        if http_client is None:
            if rate_limiter_config and rate_limiter_config_path:
                raise ValueError(
                    "Cannot provide both rate_limiter_config and rate_limiter_config_path"
                )

            self._rate_limiter_config = rate_limiter_config
            self._rate_limiter_config_path = rate_limiter_config_path
        else:
            # When http_client is provided, only disable rate limiter if explicitly requested
            if rate_limiter_config and rate_limiter_config.get("enabled") is False:
                self._rate_limiter_config = rate_limiter_config
                self._rate_limiter_config_path = rate_limiter_config_path
            else:
                self._rate_limiter_config = None
                self._rate_limiter_config_path = None

        self._config = config
        self._venice_http_client: VeniceHTTPClient | None = None

        if http_client:
            # Backward compatibility: use provided aiohttp.ClientSession
            if not isinstance(http_client, aiohttp.ClientSession):
                raise TypeError(
                    f"http_client must be an aiohttp.ClientSession, got {type(http_client)}"
                )
            self._session = http_client
            self._should_close_session = False
        else:
            if config is not None:
                # Explicit connector keywords win over the config. There is no
                # per-host default: every request goes to one host, so a
                # per-host cap below ``max_connections`` would silently become
                # the effective concurrency limit.
                connector_limit_value = self._resolve_not_given(self._connector_limit)
                if connector_limit_value is None:
                    connector_limit_value = config.http_client.max_connections
                self._venice_http_client = self._create_venice_http_client(
                    config=config,
                    base_url=base_url,
                    connector_limit=connector_limit_value,
                    connector_limit_per_host=self._resolve_not_given(
                        self._connector_limit_per_host
                    ),
                )
                self._session = None  # Will be lazy-loaded from VeniceHTTPClient
                self._should_close_session = True
            else:
                from .core.config import VeniceAIConfig

                config = VeniceAIConfig.create_minimal_config()

                # Resolve connector limits using helper method
                connector_limit_value = self._resolve_not_given(self._connector_limit)
                connector_limit_per_host_value = self._resolve_not_given(
                    self._connector_limit_per_host
                )

                self._venice_http_client = self._create_venice_http_client(
                    config=config,
                    base_url=base_url,
                    connector_limit=connector_limit_value,
                    connector_limit_per_host=connector_limit_per_host_value,
                )
                self._session = None  # Will be lazy-loaded from VeniceHTTPClient
                self._should_close_session = True

        # Initialize session lock for thread-safe lazy loading
        self._session_lock = asyncio.Lock()

        # Initialize rate limiter attributes (rate_limiter injected by factory or later)
        self.rate_limiter: RateLimiterProtocol | None = rate_limiter
        self._scheduler_manager: RateLimiterProtocol | None = rate_limiter

        # Initialize API resources
        self.chat = ChatResource(self)
        self.responses = Responses(self)
        self.models = Models(self)
        self.image = Image(self)
        self.audio = Audio(self)
        self.music = Music(self)
        self.billing = Billing(self)
        self.decisions = Decisions(self)
        self.embeddings = Embeddings(self)
        self.api_keys = ApiKeys(self)
        self.characters = Characters(self)
        self.video = Video(self)
        self.voice_changer = VoiceChanger(self)
        self.augment = Augment(self)
        self.x402 = X402(self)
        self.crypto = Crypto(self)
        self.tee = Tee(self)

    # -------------------------------------------------------------------
    # Cost tracker wiring
    # -------------------------------------------------------------------

    async def attach_cost_tracker(
        self,
        tracker: CostTracker,
        *,
        populate_pricing: bool = True,
    ) -> None:
        """Wire a :class:`CostTracker` into this client and (optionally) hydrate it.

        Solves the bootstrap order problem with :meth:`CostTracker.from_client`:
        that classmethod needs an *open* client to query the catalog, but
        :class:`VeniceClient`'s ``cost_tracker=`` constructor kwarg expects a
        tracker at construction time. Use this method instead — construct a
        bare :class:`CostTracker`, open the client, then attach::

            tracker = CostTracker()
            async with VeniceClient(api_key=...) as client:
                await client.attach_cost_tracker(tracker)
                # subsequent chat / embedding calls auto-feed the tracker

        :param tracker: The :class:`CostTracker` to receive every chat /
            embedding response from now on.
        :param populate_pricing: When ``True`` (default), fetch the live
            chat-pricing map from ``GET /models?type=chat`` and merge it
            into ``tracker.pricing_map`` (existing entries kept, missing
            entries filled in). Skip this if you've already hydrated the
            tracker yourself or want to defer the network round-trip.
        """
        if populate_pricing:
            from .types.api.models import LLMModelPricing

            catalog = await self.models.list(type="chat")
            for entry in catalog.data:
                spec = entry.model_spec
                if (
                    spec
                    and spec.pricing
                    and isinstance(spec.pricing, LLMModelPricing)
                    and entry.id not in tracker.pricing_map
                ):
                    tracker.pricing_map[entry.id] = spec.pricing
        self._cost_tracker = tracker

    # -------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------

    def _create_venice_http_client(
        self,
        config: VeniceAIConfig,
        base_url: str | URL | None,
        connector_limit: int | None,
        connector_limit_per_host: int | None,
    ) -> VeniceHTTPClient:
        """
        Create and configure a VeniceHTTPClient instance.

        This helper method extracts the common VeniceHTTPClient initialization
        logic to avoid code duplication.

        Args:
            config: The VeniceAIConfig instance
            base_url: The base URL for the API
            connector_limit: Maximum number of connections
            connector_limit_per_host: Maximum keepalive connections per host

        Returns:
            Configured VeniceHTTPClient instance
        """
        from .core.http_client import VeniceHTTPClient

        # The session carries no credentials: _request_headers() attaches
        # them to each API request, so media URLs on other hosts fetched
        # through the same session (CDN, pre-signed storage) never receive
        # the API key.
        return VeniceHTTPClient(
            config=config,
            api_key=None,
            base_url=str(base_url) if base_url is not None else None,
            headers=cast(
                dict[str, str] | None,
                self._headers if self._headers is not NOT_GIVEN else None,
            ),
            trust_env=cast(
                bool | None,
                self._trust_env if self._trust_env is not NOT_GIVEN else None,
            ),
            connector_limit=connector_limit,
            connector_limit_per_host=connector_limit_per_host,
            auto_decompress=cast(
                bool | None,
                self._auto_decompress if self._auto_decompress is not NOT_GIVEN else None,
            ),
            cookie_jar=cast(
                aiohttp.CookieJar | None,
                self._cookie_jar if self._cookie_jar is not NOT_GIVEN else None,
            ),
            skip_auto_headers=cast(
                list | None,
                self._skip_auto_headers if self._skip_auto_headers is not NOT_GIVEN else None,
            ),
            http_transport_options=self._http_transport_options,
            retry_options=cast(
                Any | None,
                self._retry_options if self._retry_options is not NOT_GIVEN else None,
            ),
            proxy=cast(str | None, self._resolve_not_given(self._proxy)),
        )

    @staticmethod
    def _resolve_retry_options(
        retry_options: RetryOptions | NotGiven,
        max_retries: int | None,
        config: VeniceAIConfig | None,
    ) -> RetryOptions | NotGiven:
        """Resolve the construction-time retry policy.

        Precedence, most specific first: an explicit ``retry_options``, the
        ``max_retries`` keyword, then ``config.http_client`` (``max_retries``
        and ``retry_backoff_factor``). With none of them the retry middleware
        uses :class:`RetryOptions` defaults.
        """
        if retry_options is not NOT_GIVEN and retry_options is not None:
            return retry_options
        if max_retries is None and config is None:
            return NOT_GIVEN
        options = RetryOptions()
        if config is not None:
            options = dataclasses.replace(
                options,
                max_attempts=config.http_client.max_retries,
                exponential_base=config.http_client.retry_backoff_factor,
            )
        if max_retries is not None:
            options = dataclasses.replace(options, max_attempts=max_retries)
        return options

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create the aiohttp session with event loop safety validation.

        Lazy initialization avoids event loop issues at client creation time.
        Uses double-check locking pattern to prevent race conditions when multiple
        coroutines call this method simultaneously before session is created.

        Raises:
            RuntimeError: If session accessed from different event loop than creation
        """
        # Validate event loop safety if session already exists
        if self._session is not None:
            current_loop = asyncio.get_running_loop()
            # Check if session was created in a different event loop
            # aiohttp.ClientSession stores the loop it was created in
            if hasattr(self._session, "_loop") and self._session._loop is not current_loop:
                raise RuntimeError(
                    "Session was created in a different event loop. "
                    "aiohttp sessions cannot be shared across event loops. "
                    "Create a new VeniceClient instance for each event loop."
                )

        if self._session is None:
            async with self._session_lock:
                # Double-check pattern: verify session is still None after acquiring lock
                if self._session is None:
                    # Always use VeniceHTTPClient (created in __init__)
                    if self._venice_http_client is None:
                        raise RuntimeError(
                            "VeniceHTTPClient not initialized. This should not happen - "
                            "please report this as a bug."
                        )
                    self._session = await self._venice_http_client.get_session()
        return self._session

    # -------------------------------------------------------------------
    # Convenience HTTP methods
    # -------------------------------------------------------------------

    async def fetch_external(self, url: str) -> bytes:
        """Fetch raw bytes from an absolute URL using the client's managed session.

        Honors the client's connector, proxy, SSL, timeout, and retry configuration.
        Intended for asset downloads (e.g. video / image CDN URLs returned by API
        responses). The client's credentials are attached only when ``url`` has
        the same origin (scheme, host and port) as the API base URL; a URL on
        any other host -- a CDN or a pre-signed storage URL -- is fetched
        without them, so the API key never reaches a third party.

        :param url: Absolute URL to fetch.
        :return: Response body as ``bytes``.
        :raises aiohttp.ClientResponseError: If the response status is >= 400.
        :raises APITimeoutError: If the request or its body does not arrive
            within the session timeout.
        :raises APIConnectionError: If the connection fails.
        """
        session = await self._get_session()
        headers = (
            self._request_headers(session) if URL(url).origin() == self._base_url.origin() else None
        )
        async with wrap_aiohttp_errors():
            resp = await session.get(url, headers=headers, timeout=self._timeout)
        try:
            resp.raise_for_status()
            return await read_body(resp)
        finally:
            resp.release()

    async def get[T: BaseModel](
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        cast_to: type[T] | None = None,
        force_direct: bool = False,
        **kwargs: Any,
    ) -> T | Any:
        """Convenience method for GET requests."""
        return await self._request(
            "GET",
            path,
            params=params,
            cast_to=cast_to,
            force_direct=force_direct,
            **kwargs,
        )

    async def post[T: BaseModel](
        self,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        cast_to: type[T] | None = None,
        force_direct: bool = False,
        **kwargs: Any,
    ) -> T | Any:
        """Convenience method for POST requests."""
        return await self._request(
            "POST",
            path,
            json_data=json_data,
            data=data,
            files=files,
            cast_to=cast_to,
            force_direct=force_direct,
            **kwargs,
        )

    async def put[T: BaseModel](
        self,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        cast_to: type[T] | None = None,
        force_direct: bool = False,
        **kwargs: Any,
    ) -> T | Any:
        """Convenience method for PUT requests."""
        return await self._request(
            "PUT",
            path,
            json_data=json_data,
            data=data,
            files=files,
            cast_to=cast_to,
            force_direct=force_direct,
            **kwargs,
        )

    async def delete[T: BaseModel](
        self,
        path: str,
        *,
        cast_to: type[T] | None = None,
        force_direct: bool = False,
        **kwargs: Any,
    ) -> T | Any:
        """Convenience method for DELETE requests."""
        return await self._request(
            "DELETE", path, cast_to=cast_to, force_direct=force_direct, **kwargs
        )

    async def patch[T: BaseModel](
        self,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        cast_to: type[T] | None = None,
        force_direct: bool = False,
        **kwargs: Any,
    ) -> T | Any:
        """Convenience method for PATCH requests."""
        return await self._request(
            "PATCH",
            path,
            json_data=json_data,
            data=data,
            files=files,
            cast_to=cast_to,
            force_direct=force_direct,
            **kwargs,
        )

    # -------------------------------------------------------------------
    # Scoped retry-policy override
    # -------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def with_retries(self, options: RetryOptions) -> AsyncIterator[None]:
        """Temporarily override the retry policy for the duration of a block.

        Sets a per-task :class:`contextvars.ContextVar` that the retry
        middleware consults at the start of every request. Calls inside the
        block — including any ``asyncio.create_task`` children — see *options*;
        calls outside (including any raced from outside) see the client's
        construction-time default.

        ``with_retries`` blocks may be nested; each restores the previous
        scope on exit. Exiting via exception still resets cleanly.

        Example::

            async with client.with_retries(RetryOptions(max_attempts=5, base_delay=2.0)):
                response = await client.chat.completions.create(...)
            # Outside the block, the default policy is back in effect.

        :param options: The :class:`RetryOptions` to use inside the block.
        """
        from .middleware.retry import _active_retry_options

        token = _active_retry_options.set(options)
        try:
            yield
        finally:
            _active_retry_options.reset(token)

    async def gather[T](
        self,
        awaitables: Iterable[Awaitable[T]],
        *,
        max_concurrency: int = 10,
        return_exceptions: bool = True,
    ) -> list[T | BaseException]:
        """Await many coroutines in parallel with a concurrency cap.

        Modality-agnostic alternative to :meth:`asyncio.gather`: accepts any
        awaitables — chat completions, image generations, embeddings,
        custom HTTP coroutines — and bounds in-flight count with a
        :class:`asyncio.Semaphore`. Result order matches input order.

        With ``return_exceptions=True`` (default) per-task failures land in
        their result slot instead of aborting the batch (mirrors
        :func:`asyncio.gather`'s ``return_exceptions=True``). Set
        ``False`` for all-or-nothing semantics.

        :param awaitables: Coroutines or other awaitables to run.
        :param max_concurrency: Maximum concurrent tasks in flight
            (default ``10``). Must be ``>= 1``.
        :param return_exceptions: If ``True`` (default), exceptions appear
            in their slot in the result list instead of raising.

        :raises ValueError: If ``max_concurrency < 1``.

        Example::

            results = await client.gather(
                [
                    client.image.create(model=img_model, prompt=p)
                    for p in prompts
                ],
                max_concurrency=3,
            )
        """
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        items = list(awaitables)
        if not items:
            return []

        sem = asyncio.Semaphore(max_concurrency)

        async def _wrapped(awaitable: Awaitable[T]) -> T:
            async with sem:
                return await awaitable

        results = await asyncio.gather(
            *(_wrapped(a) for a in items),
            return_exceptions=return_exceptions,
        )
        return list(results)

    # -------------------------------------------------------------------
    # Rate-limiter management
    # -------------------------------------------------------------------

    def _inject_rate_limiter(self, rate_limiter: RateLimiterProtocol) -> None:
        """
        Inject rate limiter after construction (called by factory).

        This method is used by VeniceClientFactory to inject the rate limiter
        after the client is created, breaking the circular dependency.

        Args:
            rate_limiter: The rate limiter instance to inject

        Raises:
            RuntimeError: If rate limiter is already injected
        """
        if self.rate_limiter is not None:
            raise RuntimeError("Rate limiter already injected")

        self.rate_limiter = rate_limiter

    def _should_use_rate_limiter(self) -> bool:
        """Determine if the rate limiter should be used based on configuration."""
        # If rate_limiter_config is provided, respect its 'enabled' setting
        if self._rate_limiter_config is not None:
            return bool(self._rate_limiter_config.get("enabled", True))

        if self._rate_limiter_config_path:
            return True
        return os.environ.get(_constants.ENV_RATE_LIMITER_ENABLED, "").lower() == "true"

    async def _ensure_rate_limiter_and_start(self) -> None:
        # If we have a rate limiter, ensure it's started
        if self.rate_limiter is not None:
            if not self.rate_limiter.is_running():
                await self.rate_limiter.start()
            return

        if not self._should_use_rate_limiter():
            return

        # Fallback: use scheduler_manager if available
        if self._scheduler_manager and not self._scheduler_manager.is_running():
            await self._scheduler_manager.start()

        self.rate_limiter = self._scheduler_manager

    # -------------------------------------------------------------------
    # Core request lifecycle
    # -------------------------------------------------------------------

    def _build_request_kwargs(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None,
        data: dict[str, Any] | aiohttp.FormData | None,
        headers: dict[str, str] | None,
        params: dict[str, Any] | None,
        timeout: float | aiohttp.ClientTimeout | None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``aiohttp.ClientSession.request()``.

        Centralises URL construction and timeout resolution shared by the
        rate-limited and direct request code paths.

        Args:
            method: HTTP method (``'GET'``, ``'POST'``, etc.).
            path: Endpoint path relative to ``base_url``.
            json_data: JSON body (passed as the ``json`` kwarg).
            data: Form / multipart body.
            headers: Already-merged request headers.
            params: URL query parameters.
            timeout: Per-request timeout; ``None`` falls back to the client
                default.

        Returns:
            A kwargs dict ready to be unpacked into
            ``session.request(**kwargs)``.
        """
        # --- URL construction ---
        base_path = self._base_url.path.rstrip("/")
        endpoint_path = path.lstrip("/")
        full_path = f"{base_path}/{endpoint_path}"
        url = self._base_url.with_path(full_path)

        return {
            "method": method,
            "url": url,
            "json": json_data,
            "data": data,
            "params": params,
            "headers": headers,
            "timeout": resolve_timeout(timeout, self._timeout),
        }

    def _default_siwe_header(self) -> str | None:
        """Return a freshly signed SIWE token for SIWE-only (Mode 2) auth.

        When the client was constructed with a wallet ``auth`` (``X402Auth``
        or ``SolanaX402Auth``) and no ``api_key``, this returns the
        base64-encoded ``X-Sign-In-With-X`` header value to attach to
        outgoing requests by default.

        A new envelope is signed for every request. Venice treats a SIWE
        nonce as single-use and answers a repeat with ``401 This nonce has
        already been used``, so reusing one — even well inside its TTL —
        makes every request after the first fail. Signing is one local
        elliptic-curve operation, negligible beside the request it
        authenticates.

        Returns ``None`` when no SIWE auth is configured, or when an API
        key is also set (in which case Bearer auth wins for default
        request authentication; the auth instance is still available for
        per-call ``auth=`` kwargs).
        """
        if self._auth is None or self._api_key:
            return None

        return self._auth.build_header()

    def _request_headers(
        self,
        session: aiohttp.ClientSession,
        headers: dict[str, str] | None = None,
        siwe_auth: X402Auth | SolanaX402Auth | None = None,
    ) -> dict[str, str]:
        """Return the complete header set for one API request.

        Every request to the Venice API builds its headers here, so
        authentication is attached per request and reaches the server whether
        the session was created by the SDK or supplied by the caller as
        ``http_client``. The session itself is never given credentials: a
        caller's session is left untouched, and :meth:`fetch_external` adds
        these headers only for URLs on the API's own origin.

        Later layers override earlier ones, with header names compared
        case-insensitively:

        1. The session's default headers.
        2. The client's authentication: ``Authorization: Bearer <api_key>``
           when an API key is set, otherwise a freshly signed
           ``X-Sign-In-With-X`` envelope from the client's wallet ``auth``
           (Venice accepts each nonce once, so an envelope is never reused).
        3. The client's ``headers=`` constructor argument.
        4. The per-call ``headers``.
        5. A per-call wallet (``siwe_auth``), which signs this request's
           ``X-Sign-In-With-X`` envelope in place of the client's own.

        Args:
            session: The session the request is sent through.
            headers: Per-call headers.
            siwe_auth: A per-call wallet that signs this request.

        Returns:
            A plain ``dict`` with one entry per header name.
        """
        merged: CIMultiDict[str] = CIMultiDict(session.headers)
        if self._api_key:
            merged.update(create_auth_headers(self._api_key))
        elif siwe_auth is None:
            siwe = self._default_siwe_header()
            if siwe is not None:
                merged[_SIWE_HEADER] = siwe
        if isinstance(self._headers, dict):
            merged.update(self._headers)
        if headers:
            merged.update(headers)
        if siwe_auth is not None:
            merged[_SIWE_HEADER] = siwe_auth.build_header()
        return dict(merged)

    @contextlib.contextmanager
    def _siwe_resigning(
        self,
        headers: dict[str, str] | None = None,
        siwe_auth: X402Auth | SolanaX402Auth | None = None,
    ) -> Iterator[None]:
        """Let the retry middleware re-sign the SIWE envelope of the request sent inside.

        Publishes :meth:`_resolve_siwe_resigner` for the duration of the
        block and resets it on the way out, so the signer never outlives the
        request it belongs to. Wrap only the call that sends the request.
        """
        token = _active_siwe_resigner.set(self._resolve_siwe_resigner(headers, siwe_auth))
        try:
            yield
        finally:
            _active_siwe_resigner.reset(token)

    def _resolve_siwe_resigner(
        self,
        headers: dict[str, str] | None,
        siwe_auth: X402Auth | SolanaX402Auth | None,
    ) -> Callable[[], str] | None:
        """Return the callable that re-signs this request's SIWE envelope.

        The retry middleware replays the same request object, so a
        wallet-authenticated retry needs a fresh envelope or the server rejects
        the repeated nonce. This resolves *which* wallet should sign it, and
        mirrors the header precedence in :meth:`_prepare_and_send_request`
        exactly — getting the two out of step would sign a retry with a
        different wallet than the one that signed the first attempt.

        Args:
            headers: Per-call headers, which override the client default.
            siwe_auth: A per-call wallet that signs this request in place of
                the client's own.

        Returns:
            A zero-argument callable producing a freshly signed header value,
            or ``None`` when nothing should be re-signed.
        """
        if siwe_auth is not None:
            return siwe_auth.build_header

        # A caller that hand-rolled the header owns it: re-signing with the
        # client's own wallet would swap in a signature from a different
        # address. Leave it alone and let the caller manage its lifetime.
        if headers and any(key.lower() == _SIWE_HEADER.lower() for key in headers):
            return None

        if self._auth is None or self._api_key:
            return None

        return self._auth.build_header

    async def _prepare_and_send_request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        data: dict[str, Any] | aiohttp.FormData | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
        force_direct: bool = False,
        siwe_auth: X402Auth | SolanaX402Auth | None = None,
    ) -> aiohttp.ClientResponse:
        """Shared request lifecycle for ``_request()`` and ``_stream_request()``.

        Handles rate-limiter routing, session acquisition, header merging, URL
        construction, timeout configuration, request transmission, and initial
        status validation.  Returns the raw response; the caller is responsible
        for body processing and (for streaming) response closure.

        Args:
            method: HTTP method (``'GET'``, ``'POST'``, etc.).
            path: Endpoint path relative to ``base_url``; leading slashes are
                normalised automatically.
            json_data: JSON request body (mutually exclusive with ``data``).
            data: Form data — plain ``dict`` or ``aiohttp.FormData`` for multipart.
            headers: Additional headers merged on top of session defaults.
            params: URL query parameters.
            timeout: Per-request timeout (float seconds or ``aiohttp.ClientTimeout``);
                falls back to the client default when ``None``.
            force_direct: Bypass rate limiting for internal/administrative requests
                that should never be queued. Overuse can cause 429 errors.
            siwe_auth: A per-call wallet that signs this request in place of
                the client's own. Every attempt, retries included, signs its
                own envelope, since Venice accepts each nonce once.

        Returns:
            Raw ``aiohttp.ClientResponse`` with a 2xx status.

        Raises:
            APITimeoutError: Request exceeded the timeout.
            APIConnectionError: DNS / TCP / TLS / proxy failure.
            APIError: Non-2xx response (mapped via :func:`~venice_ai.exceptions._make_status_error`).

        Note:
            URL construction and timeout resolution are centralised in
            :meth:`_build_request_kwargs` and shared by both the rate-limited
            and direct request paths.
        """
        # Route through scheduler if available and not bypassed
        if self.rate_limiter and not force_direct:
            await self._ensure_rate_limiter_and_start()

            # Extract model from json_data or params
            model_id = "unknown"
            if json_data and "model" in json_data:
                model_id = json_data["model"]
            elif params and "model" in params:
                model_id = params["model"]

            # Extract numeric timeout for classification
            timeout_value = timeout if timeout is not None else self._timeout
            if isinstance(timeout_value, aiohttp.ClientTimeout):
                numeric_timeout = timeout_value.total if timeout_value.total else 60.0
            else:
                numeric_timeout = float(timeout_value)

            # Create request dict for classification
            request_dict = {
                "model": model_id,
                "endpoint": path,
                "timeout": numeric_timeout,
            }

            # Classify request to create metadata
            if hasattr(self.rate_limiter, "classifier") and self.rate_limiter.classifier:
                metadata = await self.rate_limiter.classifier.classify(request_dict)
            else:
                # Fallback: create minimal metadata without classifier
                from ._queue_types import RequestMetadata, ResourceType

                metadata = RequestMetadata(
                    request_id=f"{method}:{path}",
                    model_id=model_id,
                    resource_type=ResourceType.LLM,
                    endpoint=path,
                )

            # Wrap HTTP call in callable for scheduler
            async def execute_http_request() -> aiohttp.ClientResponse:
                session = await self._get_session()
                # Built inside the callable, not at the call site: the rate
                # limiter re-invokes it for each 429 retry, so a SIWE envelope
                # minted once and captured would be resent after the server
                # had already spent its nonce.
                request_headers = self._request_headers(session, headers, siwe_auth)

                kwargs = self._build_request_kwargs(
                    method,
                    path,
                    json_data,
                    data,
                    request_headers,
                    params,
                    timeout,
                )
                # Published inside the closure because a scheduler may run it
                # in a worker task created before this call, which would never
                # see a value set outside; reset immediately so a long-lived
                # worker cannot carry this wallet into the next request.
                with self._siwe_resigning(headers, siwe_auth):
                    return await self._send(session, kwargs)

            # Submit through scheduler for queueing and rate limit management
            logger.debug(f"Routing request through scheduler. Path: {path}, Model: {model_id}")
            result = await self.rate_limiter.submit_request(
                metadata,
                execute_http_request,
                error_factory=_make_status_error,
            )

            # For INTELLIGENT mode, await the future to get actual response
            if hasattr(result, "request") and result.request and hasattr(result.request, "future"):
                logger.debug("Awaiting queued request future for completion")
                response = await result.request.future
            else:
                # BASIC mode returns result directly
                response = result
        else:
            # Direct execution (for force_direct or no scheduler)
            if force_direct:
                logger.debug(f"Request bypassing scheduler with force_direct=True. Path: {path}")
            else:
                logger.debug(f"No scheduler configured for request. Path: {path}")

            # Get the session and prepare headers
            session = await self._get_session()
            request_headers = self._request_headers(session, headers, siwe_auth)

            # Redact sensitive headers before logging
            safe_headers = request_headers.copy()
            for _sensitive in (
                "Authorization",
                "X-API-Key",
                _SIWE_HEADER,
                "X-402-Payment",
            ):
                if _sensitive in safe_headers:
                    safe_headers[_sensitive] = "[REDACTED]"

            logger.debug(f"Request headers: {safe_headers}")

            kwargs = self._build_request_kwargs(
                method,
                path,
                json_data,
                data,
                request_headers,
                params,
                timeout,
            )

            # Let the retry middleware re-sign this request's envelope; see
            # _resolve_siwe_resigner(). Reset on the way out so the signer
            # never outlives the request it belongs to.
            with self._siwe_resigning(headers, siwe_auth):
                response = await self._send(session, kwargs)

        # Validate response status (common for both paths)
        if not response.ok:
            # Extract rate limit headers before consuming the response body.
            # This ensures headers are available for RateLimitError's
            # cached_rate_limit_headers even if body parsing fails.
            rate_limit_headers = _extract_rate_limit_headers(response)

            body = None
            try:
                # Check for empty response before parsing JSON
                empty_result = self._handle_empty_response(
                    response, cast_to=None, is_error_path=True
                )
                if empty_result is not None:
                    body = empty_result
                else:
                    body = await response.json()
            except (aiohttp.ContentTypeError, ValueError) as e:
                # JSON parsing failed, try to get text instead
                try:
                    body = await response.text()
                except (TimeoutError, aiohttp.ClientError) as text_error:
                    body = f"Failed to parse response: {e}, text parsing failed: {text_error}"
            except (TimeoutError, aiohttp.ClientError) as read_error:
                # The status line arrived but the body did not. The status
                # error is still the one to raise: it is what the server said.
                body = f"Failed to read response body: {type(read_error).__name__}: {read_error}"

            raise _make_status_error(
                message=f"API request failed with status {response.status}",
                request=None,
                body=body,
                response=response,
                rate_limit_headers=rate_limit_headers,
            )

        return response  # type: ignore[no-any-return]

    @staticmethod
    async def _send(
        session: aiohttp.ClientSession, kwargs: dict[str, Any]
    ) -> aiohttp.ClientResponse:
        """Send one HTTP request, mapping transport failures to SDK errors.

        Both the direct and the rate-limited path send through here, so a
        timeout or connection failure surfaces as
        :class:`~venice_ai.exceptions.APITimeoutError` or
        :class:`~venice_ai.exceptions.APIConnectionError`, whichever path ran.
        On the rate-limited path the mapping happens inside the callable the
        limiter runs: a limiter never sees a bare ``TimeoutError`` from the
        request (which a scheduler could mistake for its own deadline), and a
        timeout the limiter itself raises is never relabelled as one from the
        HTTP request.
        """
        async with wrap_aiohttp_errors():
            return await session.request(**kwargs)

    # -------------------------------------------------------------------
    # Response handling helpers
    # -------------------------------------------------------------------

    def _handle_empty_response(
        self,
        response: Any,
        cast_to: type | None = None,
        is_error_path: bool = False,
    ) -> str | None:
        """
        Consistently handle empty HTTP responses.

        Args:
            response: The HTTP response object
            cast_to: Expected response type (if any)
            is_error_path: Whether this is in error handling flow

        Returns:
            Empty string for error paths, None for success paths with no cast_to

        Raises:
            APIResponseValidationError: If cast_to is specified but response is empty
        """
        if response.content_length == 0:
            if is_error_path:
                return ""
            if cast_to:
                raise APIResponseValidationError(
                    f"Expected {cast_to.__name__} but received empty response",
                    validation_error=ValueError("Empty response body"),
                    response_data=None,
                    model_name=cast_to.__name__,
                    response=response,
                )
            return None
        return None  # Not an empty response

    async def _request[T: BaseModel](
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        cast_to: type[T] | None = None,
        raw_response: bool = False,
        timeout: float | aiohttp.ClientTimeout | None = None,
        force_direct: bool = False,
        siwe_auth: X402Auth | SolanaX402Auth | None = None,
    ) -> T | Any | aiohttp.ClientResponse | bytes:
        """
        Makes an HTTP request to the Venice AI API.

        This is a general-purpose method for making requests. It handles JSON
        and form data, file uploads, and response parsing.

        Args:
            method: The HTTP method to use (e.g., "GET", "POST").
            path: The API endpoint path.
            json_data: A dictionary to be sent as the JSON request body.
            data: A dictionary to be sent as form data.
            files: A dictionary of files to upload.
            headers: A dictionary of additional headers for the request.
            params: A dictionary of query parameters.
            cast_to: A Pydantic model to which the response should be cast.
            raw_response: If `True`, returns the raw `aiohttp.ClientResponse`.
            timeout: The timeout for this specific request.
            force_direct: If `True`, bypasses the rate limiter.
            siwe_auth: A per-call wallet that signs this request in place of
                the client's own. Every attempt, retries included, signs its
                own envelope, since Venice accepts each nonce once.

        Returns:
            The parsed response, which can be a Pydantic model, a dictionary,
            or a raw `aiohttp.ClientResponse`.
        """
        # Handle file uploads with aiohttp.FormData
        form_data_to_send = None
        if files:
            form_data_to_send = aiohttp.FormData()
            for key, file_info in files.items():
                # file_info is expected to be a tuple: (filename, file_object, content_type)
                form_data_to_send.add_field(
                    key, file_info[1], filename=file_info[0], content_type=file_info[2]
                )
            if data:
                for key, value in data.items():
                    form_data_to_send.add_field(key, serialize_form_value(value))

        # Use the consolidated helper to prepare and send the request
        response = await self._prepare_and_send_request(
            method,
            path,
            json_data=json_data,
            data=form_data_to_send if form_data_to_send else data,
            headers=headers,
            params=params,
            timeout=timeout,
            force_direct=force_direct,
            siwe_auth=siwe_auth,
        )

        # Handle raw response requests
        if raw_response:
            return response

        # Check if this is a streaming response
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type and cast_to:
            # Return a Stream for streaming responses
            return Stream(response.content.iter_any(), client=self)

        # Parse JSON response - check for empty response first
        empty_result = self._handle_empty_response(response, cast_to=cast_to, is_error_path=False)
        if empty_result is not None or response.content_length == 0:
            return empty_result

        # The body is still being transferred under the request's timeout, so
        # a stall here surfaces as APITimeoutError like one before the
        # headers. It is read under the mapping and parsed outside it, so a
        # body that arrived but is not JSON is a parsing error, never mistaken
        # for a connection failure.
        await read_body(response)
        try:
            response_data = await response.json()
        except (aiohttp.ContentTypeError, ValueError) as e:
            if cast_to:
                raise APIResponseValidationError(
                    f"Expected {cast_to.__name__} but failed to parse JSON response",
                    validation_error=e,
                    response_data=None,
                    model_name=cast_to.__name__,
                    response=response,
                ) from e
            raise APIResponseProcessingError(
                "Failed to parse JSON response", original_error=e, response=response
            ) from e

        # Handle model validation
        if cast_to:
            try:
                # Use Pydantic's model_validate for proper validation
                validated_model = cast_to.model_validate(response_data)

                # Attach raw response metadata to model after validation.
                #
                # Design Pattern: Post-Validation Metadata Attachment
                # ===================================================
                # This pattern attaches the raw HTTP response to validated models
                # to enable access to response headers (e.g., rate limits, request IDs)
                # without including them in the validation schema.
                #
                # Rationale:
                # - Pydantic models represent API response data structure
                # - HTTP metadata (headers, status) is transport-layer concern
                # - Separating these concerns keeps models clean and focused
                #
                # Implementation:
                # - All VeniceBaseModel instances support _response attribute
                # - setattr() bypasses Pydantic's immutability for private attrs
                # - Mutation occurs after validation, preserving model integrity
                #
                # Trade-offs:
                # - Pro: Clean separation of data model vs transport metadata
                # - Pro: No need to define _response in every model schema
                # - Con: Bypasses Pydantic's immutability guarantees
                # - Con: _response not visible to type checkers without model definition
                #
                # Alternative approaches considered:
                # - ResponseWrapper[T] dataclass: Adds nesting, complicates API
                # - Model field with Optional[Response]: Pollutes data models
                # - Context variables: Loses request/response association
                validated_model._response = response  # type: ignore[attr-defined]

                # Auto-track cost when a tracker is wired on the client.
                # We feed chat / embedding responses into it; tracker errors
                # are swallowed so a tracking glitch never masks a successful
                # request.
                if self._cost_tracker is not None:
                    from .costs import _maybe_track_response

                    await _maybe_track_response(self._cost_tracker, validated_model)

                return validated_model
            except ValidationError as e:
                logger.error(
                    f"Pydantic validation failed for {cast_to.__name__}: {e}",
                    extra={
                        "model_name": cast_to.__name__,
                        "response_data": response_data,
                        "validation_errors": e.errors(),
                        "endpoint": path,
                        "method": method,
                    },
                )
                raise APIResponseValidationError(
                    f"API response validation failed for {cast_to.__name__}",
                    validation_error=e,
                    response_data=response_data,
                    model_name=cast_to.__name__,
                    response=response,
                ) from e

        return response_data

    # -------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------

    async def _stream_request[T: BaseModel](
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        cast_to: type[T],
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> AsyncIterator[T]:
        """
        Makes a streaming HTTP request to the Venice AI API.

        This method is used for endpoints that return a stream of server-sent
        events (SSE). It yields Pydantic models as they are received.

        Args:
            method: The HTTP method to use.
            path: The API endpoint path.
            json_data: The JSON request body.
            headers: Additional headers for the request.
            params: Query parameters for the request.
            cast_to: The Pydantic model to cast each event to.
            timeout: The timeout for the request.

        Yields:
            An asynchronous iterator of Pydantic models.
        """
        # Use the consolidated helper to prepare and send the request
        response = await self._prepare_and_send_request(
            method,
            path,
            json_data=json_data,
            headers=headers,
            params=params,
            timeout=timeout,
        )

        # Process the response as a streaming iterator
        # For SSE (Server-Sent Events), we need to read line by line
        try:
            # Check if we're in VCR mode (response has _body attribute and it is populated)
            # This handles VCR cassettes which might not stream properly in all cases
            # Note: aiohttp.ClientResponse always has _body (initially None), so we must check for value
            vcr_body = getattr(response, "_body", None)
            is_vcr = vcr_body is not None

            if is_vcr:
                # VCR compatibility path
                content = await response.content.read()

                # If read() returned empty but we have vcr_body, use that
                if not content and vcr_body:
                    logger.debug("Using content from VCR response body")
                    if isinstance(vcr_body, bytes):
                        content = vcr_body
                    elif hasattr(vcr_body, "string"):
                        content = vcr_body.string

                if content:
                    # Process VCR content as a single block
                    if asyncio.iscoroutine(content):
                        content = await content

                    if isinstance(content, bytes):
                        full_content = content.decode("utf-8")
                    else:
                        full_content = str(content)  # type: ignore[unreachable]

                    lines = full_content.split("\n")
                    for line in lines:
                        async for item in self._process_stream_line(line, cast_to, response):
                            yield item
                return

            # Real streaming path - iterate line by line
            async for raw_line in response.content:
                line_str = raw_line.decode("utf-8").strip()
                async for item in self._process_stream_line(line_str, cast_to, response):
                    yield item

        finally:
            # Ensure the response is properly closed
            response.close()

    async def _process_stream_line[T: BaseModel](
        self, line_str: str, cast_to: type[T], response: Any | None = None
    ) -> AsyncIterator[T]:
        """Helper to process a single line from the stream.

        Args:
            line_str: A single decoded line from the SSE body.
            cast_to: The Pydantic model each ``data:`` chunk is validated against.
            response: The originating HTTP response, used to attach context to a
                raised :class:`~venice_ai.exceptions.APIError` if the server emits
                an in-band error frame.

        Raises:
            APIError: If the line is a well-formed ``data:`` JSON object carrying a
                top-level ``error`` payload (an in-band error frame). Such frames
                must surface to the caller rather than be silently dropped, so a
                truncated stream is distinguishable from a complete one.
        """
        line_str = line_str.strip()

        # Skip empty lines and non-data lines
        if not line_str or not line_str.startswith("data: "):
            return

        # Handle the termination signal
        if line_str == "data: [DONE]":
            logger.debug("Found [DONE] signal")
            return

        try:
            # Extract JSON from the data: prefix
            json_str = line_str[6:]  # Remove "data: " prefix
            data = json.loads(json_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Genuinely-malformed / non-JSON keepalive noise: log and skip.
            logger.debug(f"Failed to parse streaming line: {line_str}, error: {e}")
            return

        # In-band error frame (e.g. ``data: {"error": "..."}`` or a nested
        # ``{"error": {"message": ..., "code": ...}}``). The HTTP response itself
        # succeeded, so this never reaches the status-code error path; surface it
        # here as an APIError instead of dropping it and ending the stream early.
        if isinstance(data, dict) and data.get("error"):
            raise self._make_stream_error(data, response)

        try:
            # Use model_validate for proper Pydantic instantiation
            if hasattr(cast_to, "model_validate"):
                yield cast_to.model_validate(data)
            else:
                yield cast_to(**data)
        except ValidationError as e:
            # A data: frame that isn't an error and doesn't match the chunk
            # schema: log and skip, preserving prior lenient behaviour for
            # forward-compatible / unknown chunk shapes.
            logger.debug(f"Failed to parse streaming line: {line_str}, error: {e}")

    @staticmethod
    def _make_stream_error(data: dict[str, Any], response: Any | None) -> APIError:
        """Build an :class:`APIError` from an in-band SSE error envelope.

        Mirrors the body parsing used by
        :func:`~venice_ai.exceptions._make_status_error` (which is keyed on an
        HTTP status code and therefore does not apply to a mid-stream error on
        an otherwise-200 response).
        """
        error_data = data.get("error")
        message = "Stream error"
        code: str | None = None
        if isinstance(error_data, dict):
            detail = error_data.get("message") or error_data.get("detail")
            message = f"Stream error: {detail}" if detail else message
            raw_code = error_data.get("code")
            code = str(raw_code) if raw_code is not None else None
        elif isinstance(error_data, str) and error_data:
            message = f"Stream error: {error_data}"

        exc = APIError(message, response=response, body=data, code=code)
        return exc

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def close(self) -> None:
        if self._is_closed:
            return

        if self._venice_http_client is not None:
            try:
                await self._venice_http_client.close()
            except (AttributeError, RuntimeError, OSError) as e:
                logger.warning(f"Error closing Venice HTTP client: {e}")

        if self._should_close_session and self._session:
            await self._session.close()

        if self._scheduler_manager:
            try:
                await self._scheduler_manager.stop()
            except (AttributeError, RuntimeError, OSError) as e:
                logger.warning(f"Error stopping scheduler: {e}")

        self._is_closed = True

    async def __aenter__(self) -> VeniceClient:
        return self

    async def __aexit__(self, exc_type: type | None, _exc_val: Any, _exc_tb: Any) -> None:
        await self.close()
