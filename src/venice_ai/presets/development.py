"""
Development configuration preset for Venice AI SDK.

This preset is optimized for local development with:
- Memory backend for simplicity (no Redis required)
- SIMPLE reactive rate limiter for predictable behavior
- Relaxed timeouts for debugging
- Comprehensive logging enabled
"""

from ..core.config import (
    BackendConfig,
    BackendType,
    CachePolicy,
    CircuitBreakerConfig,
    HttpClientConfig,
    SchedulerConfig,
    SchedulerMode,
    StateConfig,
    VeniceAIConfig,
)


def create_development_config(
    enable_debug: bool = True,
    timeout: float = 60.0,
) -> VeniceAIConfig:
    """
    Create a development-optimized configuration.

    This configuration is designed for local development and provides:
    - Memory backend (no Redis installation required)
    - The SIMPLE reactive rate limiter (no scheduler, no Redis)
    - Generous timeouts for debugging
    - Debug mode enabled by default
    - Lower concurrency for easier debugging (a 20-connection pool)
    - State management matching production behavior for consistency

    Args:
        enable_debug: Enable debug logging (default: True)
        timeout: HTTP request timeout in seconds (default: 60.0)

    Returns:
        VeniceAIConfig configured for development use

    Example:
        >>> from venice_ai.presets import create_development_config
        >>> from venice_ai import VeniceClient
        >>>
        >>> config = create_development_config()
        >>> client = VeniceClient(config=config, api_key="your-key")

    Best Practices:
        - Use separate API keys for development vs production
        - Enable debug mode to see detailed logging
        - Don't use this preset in production
        - Consider using environment-specific .env files
    """
    return VeniceAIConfig(
        environment="development",
        debug=enable_debug,
        # Simple memory backend (no Redis needed)
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        # Development state management - matches production for consistency
        state=StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,  # Match production for consistency
            is_production=False,
            batch_size=50,  # Smaller than production's 100
            batch_timeout=10.0,  # More lenient for debugging
        ),
        # Development HTTP settings
        http_client=HttpClientConfig(
            timeout=timeout,  # Generous timeout for debugging
            max_connections=20,  # Lower concurrency
            max_retries=2,
        ),
        # The scheduler only runs under RateLimiterMode.ADAPTIVE; the SIMPLE
        # rate limiter used here does not read it.
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
        # Lenient circuit breaker for development
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=20,  # More lenient
            reset_timeout=30.0,  # Faster recovery
            success_threshold=1,
        ),
    )


def create_development_config_with_rate_limiting(
    enable_debug: bool = True,
) -> VeniceAIConfig:
    """
    Create a development config with a larger connection pool and a stricter circuit breaker.

    Rate limiting here is the SIMPLE reactive rate limiter, the same one
    :func:`create_development_config` uses: it backs off when Venice returns
    429 and reads its limits from the response headers. The proactive
    scheduler only runs under ``RateLimiterMode.ADAPTIVE`` (the optional
    ``adaptive-rate-limiter`` package); set ``rate_limiter`` on the returned
    config to test it locally.

    Compared with :func:`create_development_config`, this preset uses a 60 s
    timeout, a 30-connection pool and a circuit breaker that opens after 15
    failures and needs 2 successes to close.

    Args:
        enable_debug: Enable debug logging (default: True)

    Returns:
        VeniceAIConfig using the SIMPLE rate limiter
    """
    return VeniceAIConfig(
        environment="development",
        debug=enable_debug,
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        # Development state management - matches production for consistency
        state=StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,  # Match production for consistency
            is_production=False,
            batch_size=50,  # Smaller than production's 100
            batch_timeout=10.0,  # More lenient for debugging
        ),
        http_client=HttpClientConfig(
            timeout=60.0,
            max_connections=30,
            max_retries=2,
        ),
        # The scheduler only runs under RateLimiterMode.ADAPTIVE; the SIMPLE
        # rate limiter used here does not read it.
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
        # Stricter circuit breaker than basic dev config since rate limiting
        # adds complexity - fail faster to surface integration issues early
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=15,  # Lower than basic (20) to catch rate limit issues faster
            reset_timeout=30.0,
            success_threshold=2,  # Require more successes to confirm recovery
        ),
    )
