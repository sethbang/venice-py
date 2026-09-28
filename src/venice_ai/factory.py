"""
Venice AI SDK Factory - Composition Root for Dependency Injection
================================================================

This module implements the Composition Root pattern, providing a centralized factory
for creating fully configured Venice AI clients with proper dependency injection.
It serves as the single point where all SDK components are wired together.

The factory pattern enables:
    * **Centralized Configuration**: Single place to configure all components
    * **Dependency Injection**: Proper separation of concerns and testability
    * **Environment Management**: Easy switching between production and test configurations
    * **Resource Management**: Coordinated lifecycle management of clients

Architecture:
    The factory follows a specific dependency order to avoid circular references:

    1. **VeniceClient** - Main client orchestrator
    2. **RateLimiter** - Created after client, then injected

Key Components:
    * **VeniceClientFactory**: Main factory class for production clients
    * **create_venice_client()**: Convenience function for standard clients
    * **create_test_venice_client()**: Optimized factory for testing scenarios

Example:
    >>> from venice_ai.factory import VeniceClientFactory
    >>> from venice_ai.core.config import VeniceAIConfig
    >>>
    >>> # Create production client with custom config
    >>> config = VeniceAIConfig.create_production_config()
    >>> client = VeniceClientFactory.create_client(
    ...     config=config,
    ...     api_key="your-api-key"
    ... )
    >>>
    >>> # Or use convenience function for simple cases
    >>> from venice_ai import create_venice_client
    >>> client = create_venice_client(api_key="your-api-key")
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .core.config import BackendType, RedisBackendConfig, SchedulerConfig, VeniceAIConfig
from .core.config._deprecation import warn_user
from .rate_limiting import NoOpRateLimiter, RateLimiterProtocol, SimpleRateLimiter
from .rate_limiting.config import RateLimiterConfig, RateLimiterMode

if TYPE_CHECKING:
    from ._client import VeniceClient
    from ._queue_types import RequestMetadata

import aiohttp

logger = logging.getLogger(__name__)


class AdaptiveSchedulerAdapter:
    """
    Adapter that wraps AdaptiveScheduler to implement RateLimiterProtocol.

    The AdaptiveScheduler from adaptive-rate-limiter package has a different
    submit_request signature (no error_factory parameter). This adapter bridges
    the gap by accepting error_factory but not passing it to the underlying
    scheduler - the AdaptiveScheduler handles error creation internally.

    This follows the Adapter pattern to allow AdaptiveScheduler to be used
    where RateLimiterProtocol is expected.
    """

    def __init__(self, scheduler: Any) -> None:
        """
        Initialize the adapter.

        Args:
            scheduler: The AdaptiveScheduler instance to wrap
        """
        self._scheduler = scheduler

    async def submit_request(
        self,
        metadata: RequestMetadata,
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Callable[..., Exception] | None = None,
    ) -> Any:
        """
        Submit request for rate-limited execution.

        Delegates to the wrapped scheduler's submit_request, ignoring error_factory
        since AdaptiveScheduler handles error creation internally through its
        mode strategies.

        Args:
            metadata: Request metadata containing model_id, endpoint, etc.
            request_func: Async callable that executes the actual HTTP request.
            error_factory: Ignored - AdaptiveScheduler handles errors internally.

        Returns:
            The result from the scheduler's submit_request
        """
        # AdaptiveScheduler.submit_request doesn't take error_factory
        # It handles errors internally through its mode strategies
        return await self._scheduler.submit_request(metadata, request_func)

    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        result: bool = self._scheduler.is_running()
        return result

    async def start(self) -> None:
        """Start the scheduler."""
        await self._scheduler.start()

    async def stop(self) -> None:
        """Stop the scheduler."""
        await self._scheduler.stop()

    @property
    def classifier(self) -> Any | None:
        """Optional request classifier for VeniceClient compatibility."""
        return getattr(self._scheduler, "classifier", None)

    @property
    def circuit_breaker(self) -> Any | None:
        """Expose circuit_breaker from underlying scheduler for health checks."""
        return getattr(self._scheduler, "circuit_breaker", None)


# Cleared by ``create_test_client``, whose config is inert by design.
_check_inert_config: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "venice_check_inert_config", default=True
)


def _tuned_scheduler_fields(config: VeniceAIConfig) -> list[str]:
    """SchedulerConfig fields changed from their defaults.

    ``mode`` is left out (no rate limiter reads it) and so are deprecated
    fields, which already warn when they are set.
    """
    scheduler = config.scheduler
    defaults = SchedulerConfig()
    tuned = []
    for name, info in SchedulerConfig.model_fields.items():
        if name == "mode" or info.deprecated:
            continue
        if scheduler.__dict__.get(name) != defaults.__dict__.get(name):
            tuned.append(name)
    return tuned


def _warn_on_inert_config(config: VeniceAIConfig) -> None:
    """Warn about configuration sections that the selected rate limiter never reads."""
    mode = (config.rate_limiter or RateLimiterConfig()).mode
    if mode == RateLimiterMode.ADAPTIVE:
        return
    if config.backend.backend_type == BackendType.REDIS:
        warn_user(
            f"backend_type=BackendType.REDIS has no effect with rate_limiter.mode="
            f"{mode.value!r}: only the ADAPTIVE rate limiter contacts Redis, so the "
            "configured Redis backend is never used. Set rate_limiter.mode="
            "RateLimiterMode.ADAPTIVE, or use BackendType.MEMORY."
        )
    tuned = _tuned_scheduler_fields(config)
    if tuned:
        warn_user(
            f"SchedulerConfig settings {tuned} have no effect with rate_limiter.mode="
            f"{mode.value!r}: the scheduler only runs under RateLimiterMode.ADAPTIVE."
        )


def _redis_backend_kwargs(config: VeniceAIConfig, redis_url: str) -> dict[str, Any]:
    """Keyword arguments for the adaptive Redis backend from ``backend.redis``.

    In cluster mode ``redis_url`` is passed as the cluster seed as well, so it
    is not overridden by a ``REDIS_CLUSTER_URL`` environment variable.
    """
    redis_cfg = config.backend.redis
    if redis_cfg is None:
        return {}
    kwargs: dict[str, Any] = {
        "max_connections": redis_cfg.max_connections,
        "cluster_mode": redis_cfg.cluster_mode,
    }
    if redis_cfg.cluster_mode:
        kwargs["cluster_url"] = redis_url
    return kwargs


def _warn_on_ignored_key_prefix(config: VeniceAIConfig) -> None:
    """Warn when ``backend.redis.key_prefix`` was changed: the adaptive backend ignores it."""
    redis_cfg = config.backend.redis
    if redis_cfg is None:
        return
    default_prefix = RedisBackendConfig.model_fields["key_prefix"].default
    key_prefix = redis_cfg.__dict__.get("key_prefix", default_prefix)
    if key_prefix != default_prefix:
        warn_user(
            f"RedisBackendConfig.key_prefix={key_prefix!r} is deprecated and ignored: "
            "the adaptive rate limiter's Redis backend does not prefix its keys. "
            "Isolate deployments with separate Redis databases or account IDs; "
            "key_prefix will be removed in the next major release.",
            FutureWarning,
        )


def _adaptive_scheduler_kwargs(config: VeniceAIConfig) -> dict[str, Any]:
    """SchedulerConfig values for the upstream adaptive scheduler config.

    Every current field except ``mode`` shares its name and meaning with the
    upstream ``RateLimiterConfig``; deprecated fields have no upstream
    counterpart.
    """
    return {
        name: config.scheduler.__dict__[name]
        for name, info in SchedulerConfig.model_fields.items()
        if name != "mode" and not info.deprecated
    }


class VeniceClientFactory:
    """
    Factory for creating fully configured Venice AI clients with dependency injection.

    This class serves as the Composition Root in the dependency injection pattern,
    providing the single place where all Venice AI SDK dependencies are instantiated
    and wired together. It ensures proper configuration, initialization order, and
    resource management across all components.

    The factory is designed to handle various deployment scenarios:
    * Production environments with Redis backends
    * Testing environments with optimized configurations
    * Development setups with minimal dependencies
    * Custom configurations for specific use cases

    Design Principles:
        * **Single Responsibility**: Only handles dependency wiring
        * **Configuration-Driven**: All behavior controlled via VeniceAIConfig
        * **Environment Agnostic**: Works across different deployment contexts
        * **Resource Safe**: Proper lifecycle management and cleanup

    Class Methods:
        * create_client(): Main factory method for production clients
        * create_test_client(): Optimized factory for testing scenarios
        * create_minimal_client(): Simplified factory for basic use cases
    """

    @classmethod
    def create_client(
        cls,
        config: VeniceAIConfig,
        api_key: str | None = None,
        account_id: str = "default",
        account_key: str | None = None,
        http_client: aiohttp.ClientSession | None = None,
    ) -> VeniceClient:
        """
        Create a fully configured VeniceClient with proper dependency injection.

        This is the main factory method that creates a production-ready Venice AI client
        with all dependencies properly wired together. The method follows a specific
        initialization order to avoid circular dependencies and ensure all components
        are properly configured.

        Dependency Initialization Order:
            1. **VeniceClient** - Main client orchestrator
            2. **RateLimiter** - Created with client reference, then injected

        Args:
            config: Complete Venice AI configuration containing settings for all
                   components including backend, scheduler, rate limiting, and more.
            api_key: API key for Venice AI services. If not provided, must be set
                    via environment variables or account_key.
            account_id: Unique identifier for this account instance. Used for
                       multi-tenant scenarios and resource isolation.
            account_key: Account-specific API key that overrides the global api_key
                        for this account. Defaults to api_key if not provided.
            http_client: Pre-configured aiohttp.ClientSession for HTTP requests.
                        If not provided, the client will create its own session.

        Returns:
            Fully initialized VeniceClient with all dependencies injected and
            configured according to the provided configuration.

        Raises:
            ConfigurationError: If required configuration is missing or invalid

        Warns:
            UserWarning: When part of ``config`` can have no effect: a Redis
                backend (``backend.backend_type=REDIS``) or a ``SchedulerConfig``
                changed from its defaults while ``rate_limiter.mode`` is not
                ``ADAPTIVE``. Only the ADAPTIVE rate limiter contacts Redis or
                runs the scheduler.

        Example:
            >>> from venice_ai.factory import VeniceClientFactory
            >>> from venice_ai.core.config import VeniceAIConfig
            >>>
            >>> config = VeniceAIConfig.create_production_config()
            >>> client = VeniceClientFactory.create_client(
            ...     config=config,
            ...     api_key="your-api-key",
            ...     account_id="prod-account"
            ... )
            >>>
            >>> # Client is ready for use
            >>> response = await client.chat.completions.create(...)

        Note:
            The RateLimiter component is created with a client reference after
            the client is instantiated, then injected back into the client.
            This breaks the circular dependency between client and rate limiter.
        """
        if _check_inert_config.get():
            _warn_on_inert_config(config)

        logger.info(f"Creating Venice client with config environment: {config.environment}")

        # 1. Create VeniceClient
        from ._client import VeniceClient

        # Construct full base URL with API version
        full_base_url = f"{config.api_base_url}/api/{config.api_version}"

        client = VeniceClient(
            api_key=api_key,
            base_url=full_base_url,
            http_client=http_client,
            timeout=config.http_client.timeout,
            max_retries=config.http_client.max_retries,
            config=config,
        )

        # 2. Create Rate Limiter with client reference
        rate_limiter = cls._create_rate_limiter(config, client, account_id)

        # 3. Inject rate limiter into client
        client._inject_rate_limiter(rate_limiter)
        logger.debug("Rate limiter injected into client")

        logger.info(f"Venice client created successfully for account: {account_id}")
        return client

    @classmethod
    def create_test_client(
        cls,
        enable_redis: bool = True,
        test_rate_multiplier: float = 10.0,
        scheduler_mode: Any | None = None,
        **kwargs: Any,
    ) -> VeniceClient:
        """
        Create a Venice client optimized for testing.

        A convenience factory method for constructing a client preconfigured for testing.

        Args:
            enable_redis: Whether to use Redis or memory backend
            test_rate_multiplier: Rate limit multiplier for faster tests
            scheduler_mode: Optional scheduler mode (SchedulerMode enum) to use for testing.
                          If provided, this will be passed to create_test_config().
            **kwargs: Additional arguments passed to create_client

        Returns:
            VeniceClient configured for testing
        """
        # Build config kwargs
        config_kwargs: dict[str, Any] = {
            "enable_redis": enable_redis,
            "test_rate_multiplier": test_rate_multiplier,
        }
        if scheduler_mode is not None:
            config_kwargs["scheduler_mode"] = scheduler_mode

        config = VeniceAIConfig.create_test_config(**config_kwargs)

        # Separate http_client and other parameters to avoid type conflicts
        http_client = kwargs.pop("http_client", None)

        defaults = {
            "api_key": "test-api-key",
            "account_id": "test-account",
            "http_client": http_client,
        }
        defaults.update(kwargs)

        # The test config pairs a Redis backend and a tuned scheduler with the
        # SIMPLE rate limiter on purpose, so the inert-config checks are skipped.
        token = _check_inert_config.set(False)
        try:
            return cls.create_client(config, **defaults)
        finally:
            _check_inert_config.reset(token)

    @classmethod
    def create_developer_client(
        cls,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> VeniceClient:
        """Create a Venice client tuned for interactive local development.

        Pairs ``create_development_config()`` (memory backend, BASIC scheduler,
        debug logging) with fail-loud timeout/retry overrides so failures
        surface fast rather than waiting through production-grade backoff.

        Distinct from ``create_test_venice_client`` (unit-test isolation,
        fake API key, faster rate multipliers) — this one expects a real
        API key and hits the real API.

        Args:
            api_key: API key. If ``None``, read from ``VENICE_API_KEY`` env var.
            timeout: HTTP request timeout in seconds. Tighter than
                ``create_development_config`` (60 s) so dev iteration fails
                loud rather than hanging.
            max_retries: HTTP retry count. ``1`` (one retry) makes flakes
                visible without removing all resilience.
            **kwargs: Forwarded to :meth:`create_client`.
        """
        from .presets.development import create_development_config

        config = create_development_config(timeout=timeout)
        # create_development_config sets max_retries=2 by default; tighten.
        config.http_client.max_retries = max_retries

        defaults: dict[str, Any] = {"api_key": api_key}
        defaults.update(kwargs)
        return cls.create_client(config, **defaults)

    @classmethod
    def create_minimal_client(cls, api_key: str, **kwargs: Any) -> VeniceClient:
        """
        Create a minimal Venice client for simple use cases.

        Args:
            api_key: API key for Venice AI services
            **kwargs: Additional arguments passed to create_client

        Returns:
            VeniceClient with minimal configuration
        """
        config = VeniceAIConfig.create_minimal_config(api_key=api_key)
        return cls.create_client(config, api_key=api_key, **kwargs)

    @classmethod
    def _create_rate_limiter(
        cls,
        config: VeniceAIConfig,
        client: Any,  # VeniceClient instance (implements ClientProtocol)
        account_id: str = "",
    ) -> RateLimiterProtocol:
        """
        Create rate limiter based on configuration.

        Args:
            config: Venice AI configuration
            client: VeniceClient instance for the adaptive scheduler to use
            account_id: Account ID for Redis key scoping (required for adaptive mode)

        Returns:
            SimpleRateLimiter by default
            AdaptiveScheduler if mode=ADAPTIVE and package installed

        Raises:
            ImportError: If ADAPTIVE mode selected but adaptive-rate-limiter not installed
            RuntimeError: If AdaptiveScheduler creation fails for other reasons
        """
        rate_config = config.rate_limiter or RateLimiterConfig()

        if rate_config.mode == RateLimiterMode.DISABLED:
            import logging

            logging.getLogger("venice_ai.rate_limiting").critical(
                "RATE LIMITING DISABLED: This is not recommended for production. "
                "You may experience 429 errors without backoff protection. "
                "Set rate_limiter.mode=RateLimiterMode.SIMPLE to enable rate limiting."
            )
            import warnings

            warnings.warn(
                "Rate limiting is DISABLED. This will cause uncontrolled 429 errors. "
                "Set rate_limiter.mode=RateLimiterMode.SIMPLE to enable.",
                UserWarning,
                stacklevel=2,
            )
            return NoOpRateLimiter()

        if rate_config.mode == RateLimiterMode.ADAPTIVE:
            _warn_on_ignored_key_prefix(config)
            try:
                from adaptive_rate_limiter.backends import (
                    RedisBackend as AdaptiveRedisBackend,
                )
                from adaptive_rate_limiter.scheduler import (
                    RateLimiterConfig as AdaptiveConfig,
                )
                from adaptive_rate_limiter.scheduler import (
                    Scheduler as AdaptiveScheduler,
                )
                from adaptive_rate_limiter.scheduler import (
                    SchedulerMode as AdaptiveSchedulerMode,
                )
                from adaptive_rate_limiter.scheduler import (
                    StateManager as AdaptiveStateManager,
                )

                from ._request_classifier import RequestClassifier
                from .core.rate_limit_discovery import RateLimitDiscovery
                from .provider.classifier_adapter import VeniceClassifierAdapter

                # Import Venice-specific adapters for INTELLIGENT mode
                # These bridge the SDK client to the extracted library's protocols
                from .provider.venice_provider import VeniceProvider

                # Validate required configuration
                redis_url = rate_config.redis_url
                if (
                    not redis_url
                    and hasattr(config, "backend")
                    and config.backend.redis is not None
                ):
                    redis_url = config.backend.redis.redis_url

                if not redis_url:
                    raise ValueError(
                        "redis_url is required for ADAPTIVE mode. "
                        "Set rate_limiter.redis_url or backend.redis.redis_url in config."
                    )

                effective_account_id = rate_config.account_id or account_id
                if not effective_account_id:
                    import warnings

                    warnings.warn(
                        "account_id not provided for ADAPTIVE mode. "
                        "Using 'default' which may cause key collisions in multi-tenant setups.",
                        UserWarning,
                        stacklevel=2,
                    )
                    effective_account_id = "default"

                # Create the rate-limit backend scoped to this account_id.
                backend = AdaptiveRedisBackend(
                    redis_url=redis_url,
                    account_id=effective_account_id,
                    **_redis_backend_kwargs(config, redis_url),
                )

                # Create Venice-specific provider and classifier for INTELLIGENT mode
                # VeniceProvider implements ProviderInterface - discovers rate limit buckets
                # VeniceClassifierAdapter implements ClassifierProtocol - classifies requests
                #
                # Note: provider/classifier are constructed BEFORE state_manager because
                # StateManager requires the provider for header-based rate-limit state
                # updates. Without it, update_state_from_headers
                # short-circuits with "Provider required for header-based state updates",
                # forcing every request through the cold-start probe path indefinitely.

                # 1. Create RateLimitDiscovery (shared by provider and classifier)
                discovery = RateLimitDiscovery(client=client, account_id=effective_account_id)

                # 2. Create Provider with discovery
                provider = VeniceProvider(client=client, rate_limit_discovery=discovery)

                # 3. Create RequestClassifier with discovery
                request_classifier = RequestClassifier(rate_limit_discovery=discovery)

                # 4. Create Classifier Adapter
                classifier = VeniceClassifierAdapter(classifier=request_classifier)

                # 5. Create state manager with backend AND provider.
                # Provider is required so update_state_from_headers can resolve buckets
                # for response-header-driven state updates; without it the scheduler
                # falls back to cold-start probes for every request.
                state_manager = AdaptiveStateManager(backend=backend, provider=provider)

                # The adaptive scheduler always runs in INTELLIGENT mode; every
                # other shared SchedulerConfig field is forwarded.
                adaptive_config = AdaptiveConfig(
                    mode=AdaptiveSchedulerMode.INTELLIGENT,
                    **_adaptive_scheduler_kwargs(config),
                )

                # Create and return adaptive scheduler with ALL required dependencies
                # CRITICAL: INTELLIGENT mode requires provider and classifier
                scheduler = AdaptiveScheduler(
                    client=client,
                    config=adaptive_config,
                    provider=provider,
                    classifier=classifier,
                    state_manager=state_manager,
                )

                # Wrap scheduler with adapter to match RateLimiterProtocol
                # The adapter handles the error_factory parameter difference
                adapted_scheduler = AdaptiveSchedulerAdapter(scheduler)

                logger.info(
                    f"Created AdaptiveScheduler with Redis backend "
                    f"(account_id={effective_account_id[:8]}..., mode=intelligent)"
                )

                return adapted_scheduler

            except ImportError:
                raise ImportError(
                    "adaptive-rate-limiter package is required for ADAPTIVE rate limiting mode. "
                    "Install it with: pip install 'venice-py[adaptive]'"
                ) from None
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create AdaptiveScheduler: {exc}. "
                    "Check your scheduler/backend configuration."
                ) from exc

        # Default: SimpleRateLimiter
        return SimpleRateLimiter(
            min_backoff=rate_config.min_backoff,
            max_backoff=rate_config.max_backoff,
            failure_threshold=rate_config.failure_threshold,
            failure_window=rate_config.failure_window,
            block_duration=rate_config.block_duration,
            max_models=rate_config.max_models,
            stale_threshold=rate_config.stale_threshold,
            max_retries=rate_config.max_retries,
        )


# Convenience functions for common use cases
def create_venice_client(
    api_key: str, config: VeniceAIConfig | None = None, **kwargs: Any
) -> VeniceClient:
    """
    Convenience function to create a Venice client with default configuration.

    Args:
        api_key: API key for Venice AI services
        config: Optional configuration (uses minimal config if not provided)
        **kwargs: Additional arguments passed to factory

    Returns:
        Configured VeniceClient instance
    """
    if config is None:
        config = VeniceAIConfig.create_minimal_config(api_key=api_key)

    return VeniceClientFactory.create_client(config, api_key=api_key, **kwargs)


def create_test_venice_client(**kwargs: Any) -> VeniceClient:
    """
    Convenience function to create a Venice client for testing.

    Args:
        **kwargs: Arguments passed to create_test_client

    Returns:
        VeniceClient configured for testing
    """
    return VeniceClientFactory.create_test_client(**kwargs)


def create_developer_client(**kwargs: Any) -> VeniceClient:
    """Create a Venice client tuned for interactive local development.

    Convenience for :meth:`VeniceClientFactory.create_developer_client` —
    pairs ``create_development_config()`` (memory backend, BASIC scheduler,
    debug logging) with fail-loud timeout/retry defaults (30 s timeout,
    1 retry).

    Distinct from :func:`create_test_venice_client` (unit-test isolation):
    this one hits the real API with a real key. Reads ``VENICE_API_KEY``
    from env when ``api_key`` is not supplied.

    >>> from venice_ai import create_developer_client
    >>> client = create_developer_client()  # reads VENICE_API_KEY

    Args:
        **kwargs: Arguments forwarded to ``create_developer_client``
            (``api_key``, ``timeout``, ``max_retries``, etc.).

    Returns:
        Configured VeniceClient.
    """
    return VeniceClientFactory.create_developer_client(**kwargs)
