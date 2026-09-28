"""
Production configuration preset for Venice AI SDK.

This preset provides battle-tested configuration optimized for production deployments
with enterprise-grade features including:
- Redis backend for distributed state management
- Intelligent scheduler with rate limiting
- Circuit breaker protection
- Optimized connection pooling
- Conservative timeouts for reliability
"""

import os

from ..core.config import (
    BackendConfig,
    BackendType,
    CachePolicy,
    CircuitBreakerConfig,
    HttpClientConfig,
    RedisBackendConfig,
    SchedulerConfig,
    SchedulerMode,
    StateConfig,
    VeniceAIConfig,
)
from ..rate_limiting.config import RateLimiterConfig, RateLimiterMode


def _key_prefix(redis_key_prefix: str | None) -> str:
    if redis_key_prefix is None:
        default: str = RedisBackendConfig.model_fields["key_prefix"].default
        return default
    return redis_key_prefix


def create_production_config(
    redis_url: str | None = None,
    redis_key_prefix: str | None = None,
    max_concurrent_executions: int = 100,
    max_queue_size: int = 5000,
    enable_metrics: bool = True,
    _allow_localhost_for_testing: bool = False,
) -> VeniceAIConfig:
    """
    Create a production-optimized configuration.

    This configuration is designed for production deployments and includes:
    - Redis backend for distributed state across multiple instances
    - Intelligent scheduler with automatic rate limit discovery
    - Circuit breaker with conservative thresholds
    - Connection pooling optimized for high throughput
    - Retry logic with exponential backoff

    Args:
        redis_url: Redis connection URL. **REQUIRED** for production use.
            Set via VENICE_REDIS_URL environment variable or pass explicitly.
            Example: redis://user:pass@prod-redis:6379/0
            **CRITICAL:** Using localhost in production will cause failures in
            distributed/multi-instance environments.
        redis_key_prefix: Deprecated and ignored: the adaptive rate limiter does not
            prefix its Redis keys. Passing a value emits a ``FutureWarning`` when the
            client is built. Isolate environments with separate Redis databases.
        max_concurrent_executions: Maximum concurrent requests (default: 100)
        max_queue_size: Maximum queue size before rejecting requests (default: 5000)
        enable_metrics: Enable metrics collection (default: True)

    Returns:
        VeniceAIConfig configured for production use

    Raises:
        ValueError: If redis_url is not provided via parameter or VENICE_REDIS_URL environment variable

    Example:
        >>> import os
        >>> os.environ['VENICE_REDIS_URL'] = 'redis://prod-redis:6379'
        >>> from venice_ai.presets import create_production_config
        >>> from venice_ai import VeniceClient
        >>>
        >>> config = create_production_config(
        ...     max_concurrent_executions=200
        ... )
        >>> client = VeniceClient(config=config, api_key="your-key")

    Best Practices:
        - **ALWAYS** set VENICE_REDIS_URL environment variable in production
        - Use a dedicated Redis instance for production
        - Monitor Redis connection pool metrics
        - Adjust max_concurrent_executions based on your rate limits
        - Enable metrics for production observability
        - Use separate Redis databases (or instances) for different environments
    """
    # Validate Redis URL is provided
    if redis_url is None:
        redis_url = os.getenv("VENICE_REDIS_URL")
        if redis_url is None:
            raise ValueError(
                "Production Redis URL required. Set VENICE_REDIS_URL "
                "environment variable or pass redis_url parameter."
            )

    # STRICT validation: reject localhost in production
    # Allow override for testing purposes only
    if not _allow_localhost_for_testing and ("localhost" in redis_url or "127.0.0.1" in redis_url):
        raise ValueError(
            f"Invalid Redis URL for production: {redis_url}\n"
            "localhost/127.0.0.1 URLs are not allowed in production mode.\n"
            "This will fail in distributed/multi-instance environments.\n"
            "Use a network-accessible Redis instance instead."
        )

    return VeniceAIConfig(
        environment="production",
        debug=False,
        # Redis backend for distributed coordination
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(
                redis_url=redis_url,
                max_connections=50,  # Connection pool size
                default_ttl=3600,  # 1 hour default cache TTL
                key_prefix=_key_prefix(redis_key_prefix),
                connection_timeout=5.0,
                max_retries=3,
                retry_delay=1.0,
            ),
        ),
        # Production state management with WRITE_THROUGH for data integrity
        state=StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,  # Safe for production
            is_production=True,  # Enable production safety checks
            batch_size=100,  # Still useful for WRITE_THROUGH batching
            batch_timeout=5.0,
        ),
        # Production HTTP settings
        http_client=HttpClientConfig(
            timeout=30.0,  # Conservative timeout for reliability
            max_connections=200,  # Total pool size
            max_retries=3,  # Retry failed requests
        ),
        # Intelligent scheduler with rate limiting
        scheduler=SchedulerConfig(
            mode=SchedulerMode.INTELLIGENT,
            max_concurrent_executions=max_concurrent_executions,
            max_queue_size=max_queue_size,
            enable_rate_limiting=True,
            rate_limit_buffer_ratio=0.9,  # Use 90% of available limit
            overflow_policy="reject",  # Reject requests when queue full
            metrics_enabled=enable_metrics,
            enable_performance_tracking=enable_metrics,
        ),
        # Adaptive rate limiter — wires the configured Redis backend through
        # to the rate-limit state store. Without ADAPTIVE mode, BackendType.REDIS
        # is silently ignored and Redis is never contacted.
        rate_limiter=RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url=redis_url,
        ),
        # Circuit breaker for fault tolerance
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=10,  # Open after 10 consecutive failures
            reset_timeout=60.0,  # Try to close after 60 seconds
            success_threshold=2,  # Close after 2 consecutive successes
        ),
    )


def create_production_config_high_throughput(
    redis_url: str | None = None,
    redis_key_prefix: str | None = None,
    _allow_localhost_for_testing: bool = False,
) -> VeniceAIConfig:
    """
    Create a high-throughput production configuration.

    Optimized for maximum throughput when you have high rate limits.
    Uses more aggressive settings than standard production config.

    Args:
        redis_url: Redis connection URL. **REQUIRED** for production use.
            Set via VENICE_REDIS_URL environment variable or pass explicitly.
            **CRITICAL:** Using localhost in production will cause failures in
            distributed/multi-instance environments.
        redis_key_prefix: Deprecated and ignored; see :func:`create_production_config`.

    Returns:
        VeniceAIConfig optimized for high throughput

    Raises:
        ValueError: If redis_url is not provided via parameter or VENICE_REDIS_URL environment variable

    Warning:
        Only use this preset if you have sufficient rate limits and
        infrastructure to handle high concurrency.
    """
    # Validate Redis URL is provided
    if redis_url is None:
        redis_url = os.getenv("VENICE_REDIS_URL")
        if redis_url is None:
            raise ValueError(
                "Production Redis URL required. Set VENICE_REDIS_URL "
                "environment variable or pass redis_url parameter."
            )

    # STRICT validation: reject localhost in production
    # Allow override for testing purposes only
    if not _allow_localhost_for_testing and ("localhost" in redis_url or "127.0.0.1" in redis_url):
        raise ValueError(
            f"Invalid Redis URL for production: {redis_url}\n"
            "localhost/127.0.0.1 URLs are not allowed in production mode.\n"
            "This will fail in distributed/multi-instance environments.\n"
            "Use a network-accessible Redis instance instead."
        )

    return VeniceAIConfig(
        environment="production",
        debug=False,
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(
                redis_url=redis_url,
                max_connections=100,  # Larger pool for high throughput
                default_ttl=1800,  # 30 min TTL
                key_prefix=_key_prefix(redis_key_prefix),
                connection_timeout=3.0,
                max_retries=2,
            ),
        ),
        # Production state management with WRITE_THROUGH for data integrity
        state=StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,  # Safe for production
            is_production=True,  # Enable production safety checks
            batch_size=100,  # Still useful for WRITE_THROUGH batching
            batch_timeout=5.0,
        ),
        http_client=HttpClientConfig(
            timeout=45.0,  # Longer timeout for batch operations
            max_connections=500,  # Much larger pool
            max_retries=2,
        ),
        scheduler=SchedulerConfig(
            mode=SchedulerMode.INTELLIGENT,
            max_concurrent_executions=300,  # High concurrency
            max_queue_size=10000,  # Larger queue
            enable_rate_limiting=True,
            rate_limit_buffer_ratio=0.95,  # Use 95% of limit
            overflow_policy="reject",
            metrics_enabled=True,
            enable_performance_tracking=True,
        ),
        rate_limiter=RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url=redis_url,
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=15,  # More lenient threshold
            reset_timeout=45.0,
            success_threshold=3,
        ),
    )


def create_production_config_conservative(
    redis_url: str | None = None,
    redis_key_prefix: str | None = None,
    _allow_localhost_for_testing: bool = False,
) -> VeniceAIConfig:
    """
    Create a conservative production configuration.

    Optimized for maximum reliability over throughput.
    Use this when stability is more important than speed.

    Args:
        redis_url: Redis connection URL. **REQUIRED** for production use.
            Set via VENICE_REDIS_URL environment variable or pass explicitly.
            **CRITICAL:** Using localhost in production will cause failures in
            distributed/multi-instance environments.
        redis_key_prefix: Deprecated and ignored; see :func:`create_production_config`.

    Returns:
        VeniceAIConfig optimized for reliability

    Raises:
        ValueError: If redis_url is not provided via parameter or VENICE_REDIS_URL environment variable
    """
    # Validate Redis URL is provided
    if redis_url is None:
        redis_url = os.getenv("VENICE_REDIS_URL")
        if redis_url is None:
            raise ValueError(
                "Production Redis URL required. Set VENICE_REDIS_URL "
                "environment variable or pass redis_url parameter."
            )

    # STRICT validation: reject localhost in production
    # Allow override for testing purposes only
    if not _allow_localhost_for_testing and ("localhost" in redis_url or "127.0.0.1" in redis_url):
        raise ValueError(
            f"Invalid Redis URL for production: {redis_url}\n"
            "localhost/127.0.0.1 URLs are not allowed in production mode.\n"
            "This will fail in distributed/multi-instance environments.\n"
            "Use a network-accessible Redis instance instead."
        )

    return VeniceAIConfig(
        environment="production",
        debug=False,
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(
                redis_url=redis_url,
                max_connections=30,  # Smaller pool
                default_ttl=7200,  # 2 hour TTL for stability
                key_prefix=_key_prefix(redis_key_prefix),
                connection_timeout=10.0,  # More lenient timeout
                max_retries=5,  # More retries
                retry_delay=2.0,
            ),
        ),
        # Production state management with WRITE_THROUGH for data integrity
        state=StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,  # Safe for production
            is_production=True,  # Enable production safety checks
            batch_size=100,  # Still useful for WRITE_THROUGH batching
            batch_timeout=5.0,
        ),
        http_client=HttpClientConfig(
            timeout=60.0,  # Generous timeout
            max_connections=100,
            max_retries=5,  # More retries
        ),
        scheduler=SchedulerConfig(
            mode=SchedulerMode.INTELLIGENT,
            max_concurrent_executions=50,  # Lower concurrency
            max_queue_size=2000,
            enable_rate_limiting=True,
            rate_limit_buffer_ratio=0.8,  # Conservative 80% limit usage
            overflow_policy="reject",
            metrics_enabled=True,
            enable_performance_tracking=True,
        ),
        rate_limiter=RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url=redis_url,
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,  # Strict threshold
            reset_timeout=120.0,  # Longer recovery time
            success_threshold=3,
        ),
    )
