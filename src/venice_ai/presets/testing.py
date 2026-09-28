"""
Testing configuration preset for Venice AI SDK.

This preset is optimized for automated testing with:
- Memory backend for test isolation
- The SIMPLE reactive rate limiter (no scheduler, no Redis)
- Short timeouts and a single retry for fast test runs
- Circuit breaker disabled for consistent testing
"""

from ..core.config import (
    BackendConfig,
    BackendType,
    CircuitBreakerConfig,
    HttpClientConfig,
    SchedulerConfig,
    SchedulerMode,
    VeniceAIConfig,
)


def _scheduler(test_rate_multiplier: float | None) -> SchedulerConfig:
    """Scheduler section for the SIMPLE rate limiter, which does not read it.

    ``test_rate_multiplier`` is only kept when the caller passed one, so that
    ``VeniceClientFactory.create_client`` can report it as having no effect.
    """
    if test_rate_multiplier is None:
        return SchedulerConfig(mode=SchedulerMode.BASIC)
    return SchedulerConfig(mode=SchedulerMode.BASIC, test_rate_multiplier=test_rate_multiplier)


def create_testing_config(
    test_rate_multiplier: float | None = None,
    enable_circuit_breaker: bool = False,
) -> VeniceAIConfig:
    """
    Create a testing-optimized configuration.

    This configuration is designed for automated testing and provides:
    - Memory backend for test isolation (no shared state)
    - The SIMPLE reactive rate limiter; the scheduler section keeps its defaults
    - Circuit breaker disabled by default (can be enabled for testing CB logic)
    - Minimal timeouts and a single retry for fast test execution

    Args:
        test_rate_multiplier: Has no effect with this preset: only the
            scheduler of ``RateLimiterMode.ADAPTIVE`` reads it. A value passed
            here is stored on ``scheduler.test_rate_multiplier``, and
            ``VeniceClientFactory.create_client`` warns that it is unused.
        enable_circuit_breaker: Enable circuit breaker for testing CB logic (default: False)

    Note:
        "Circuit breaker" here refers to the failure-recovery feature of the
        extracted ``adaptive-rate-limiter`` package's scheduler. It only takes
        effect when ``RateLimiterMode.ADAPTIVE`` is in use; under SIMPLE or
        DISABLED modes the configuration is inert.

    Returns:
        VeniceAIConfig configured for testing

    Example:
        >>> from venice_ai.presets import create_testing_config
        >>> from venice_ai import VeniceClient
        >>>
        >>> # In your test fixtures
        >>> @pytest.fixture
        >>> async def test_client():
        ...     config = create_testing_config()
        ...     async with VeniceClient(config=config, api_key="test-key") as client:
        ...         yield client

    Best Practices:
        - Use this preset in your pytest fixtures
        - Don't use real API keys in tests
        - Use VCR.py or mocks for external API calls
        - Tests should be isolated and deterministic
    """
    return VeniceAIConfig(
        environment="test",
        debug=False,  # Less noise in test output
        # Memory backend for test isolation
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        # Fast HTTP settings for tests
        http_client=HttpClientConfig(
            timeout=10.0,  # Short timeout for fast failure
            max_connections=10,  # Low concurrency for tests
            max_retries=1,  # Minimal retries for faster tests
        ),
        # The scheduler only runs under RateLimiterMode.ADAPTIVE; the SIMPLE
        # rate limiter used here does not read it.
        scheduler=_scheduler(test_rate_multiplier),
        # Circuit breaker configuration
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=100 if not enable_circuit_breaker else 5,
            reset_timeout=1.0,  # Fast recovery in tests
            success_threshold=1,
        )
        if enable_circuit_breaker
        else CircuitBreakerConfig(
            failure_threshold=999,  # Effectively disabled
            reset_timeout=1.0,
            success_threshold=1,
        ),
    )


def create_testing_config_with_intelligent_scheduler(
    test_rate_multiplier: float | None = None,
) -> VeniceAIConfig:
    """
    Create a testing config; despite its name, no scheduler runs.

    The returned config uses the SIMPLE reactive rate limiter, which never
    runs the intelligent scheduler, so this is equivalent to
    :func:`create_testing_config` apart from a 15 s timeout, a 20-connection
    pool and two retries. It will be removed in the next major release.

    To test the intelligent scheduler, set
    ``rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE)`` on the
    config. That requires the optional ``adaptive-rate-limiter`` package
    (``pip install 'venice-py[adaptive]'``) and an ``account_id``.

    Args:
        test_rate_multiplier: Has no effect with this preset; see
            :func:`create_testing_config`.

    Returns:
        VeniceAIConfig using the SIMPLE rate limiter
    """
    return VeniceAIConfig(
        environment="test",
        debug=False,
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        http_client=HttpClientConfig(
            timeout=15.0,
            max_connections=20,
            max_retries=2,
        ),
        # The scheduler only runs under RateLimiterMode.ADAPTIVE; the SIMPLE
        # rate limiter used here does not read it.
        scheduler=_scheduler(test_rate_multiplier),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=999,  # Disabled
            reset_timeout=1.0,
            success_threshold=1,
        ),
    )


def create_testing_config_for_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 5.0,
) -> VeniceAIConfig:
    """
    Create testing config specifically for testing circuit breaker logic.

    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        reset_timeout: Seconds before testing recovery (default: 5.0)

    Note:
        "Circuit breaker" here refers to the failure-recovery feature of the
        extracted ``adaptive-rate-limiter`` package's scheduler. It only takes
        effect when ``RateLimiterMode.ADAPTIVE`` is in use; under SIMPLE or
        DISABLED modes the configuration is inert.

    Returns:
        VeniceAIConfig configured for circuit breaker testing
    """
    return VeniceAIConfig(
        environment="test",
        debug=True,  # Enable debug for CB testing
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        http_client=HttpClientConfig(
            timeout=5.0,
            max_connections=5,
            max_retries=0,  # No retries for testing CB
        ),
        # The scheduler only runs under RateLimiterMode.ADAPTIVE; the SIMPLE
        # rate limiter used here does not read it.
        scheduler=_scheduler(None),
        # Circuit breaker with test-friendly settings
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            success_threshold=1,
        ),
    )
