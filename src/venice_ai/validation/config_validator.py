"""
Configuration validation and recommendations for Venice AI SDK.

This module analyzes VeniceAIConfig objects and provides:
- Error detection (invalid/incompatible settings)
- Warnings (suboptimal configurations)
- Recommendations (best practice suggestions)
"""

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel

from ..core.config import (
    BackendType,
    SchedulerMode,
    VeniceAIConfig,
)
from ..rate_limiting.config import RateLimiterMode


class Severity(StrEnum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: Severity
    message: str
    category: str
    fix_suggestion: str | None = None


@dataclass
class ConfigValidation:
    """
    Result of configuration validation.

    Attributes:
        is_valid: True if no errors found
        errors: List of error messages
        warnings: List of warning messages
        recommendations: List of optimization suggestions
        issues: Detailed list of all issues found
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, message: str, category: str = "general", fix: str | None = None):
        """Add an error to the validation result."""
        self.errors.append(message)
        self.issues.append(ValidationIssue(Severity.ERROR, message, category, fix))
        self.is_valid = False

    def add_warning(self, message: str, category: str = "general", fix: str | None = None):
        """Add a warning to the validation result."""
        self.warnings.append(message)
        self.issues.append(ValidationIssue(Severity.WARNING, message, category, fix))

    def add_recommendation(self, message: str, category: str = "general"):
        """Add a recommendation to the validation result."""
        self.recommendations.append(message)
        self.issues.append(ValidationIssue(Severity.INFO, message, category))


def validate_config(config: VeniceAIConfig) -> ConfigValidation:
    """
    Validate a Venice AI configuration and provide recommendations.

    This function performs comprehensive validation including:
    - Compatibility checks (scheduler mode vs backend type)
    - Performance analysis (timeouts, pool sizes, queue depths)
    - Best practice verification
    - Security recommendations

    Args:
        config: The VeniceAIConfig to validate

    Returns:
        ConfigValidation with errors, warnings, and recommendations

    Example:
        >>> from venice_ai.validation import validate_config
        >>> from venice_ai.core.config import VeniceAIConfig
        >>>
        >>> config = VeniceAIConfig()
        >>> result = validate_config(config)
        >>>
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(f"ERROR: {error}")
    """
    result = ConfigValidation(is_valid=True)

    # Validate scheduler + backend compatibility
    _validate_scheduler_backend_compatibility(config, result)

    # Validate timeout settings
    _validate_timeouts(config, result)

    # Validate connection pool settings
    _validate_connection_pools(config, result)

    # Report deprecated settings that have no effect
    _validate_deprecated_fields(config, result)

    # Validate rate limiting configuration
    _validate_rate_limiting(config, result)

    # Validate circuit breaker settings
    _validate_circuit_breaker(config, result)

    # Validate Redis configuration if applicable
    if config.backend.backend_type == BackendType.REDIS:
        _validate_redis_config(config, result)

    # Performance recommendations
    _add_performance_recommendations(config, result)

    # Security recommendations
    _add_security_recommendations(config, result)

    return result


def _validate_scheduler_backend_compatibility(
    config: VeniceAIConfig, result: ConfigValidation
) -> None:
    """Validate scheduler mode is compatible with backend type."""
    if (
        config.scheduler.mode == SchedulerMode.INTELLIGENT
        and config.backend.backend_type != BackendType.REDIS
    ):
        result.add_warning(
            "INTELLIGENT scheduler mode works best with Redis backend. "
            "Consider switching to Redis for distributed coordination or using BASIC mode.",
            category="compatibility",
            fix="Use create_production_config() preset or set scheduler.mode=SchedulerMode.BASIC",
        )


def _validate_timeouts(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Validate timeout settings are reasonable."""
    timeout = config.http_client.timeout

    if timeout < 5.0:
        result.add_warning(
            f"HTTP timeout is very short ({timeout}s). "
            "This may cause failures for image generation or long completions.",
            category="timeout",
            fix="Consider increasing to at least 30s for production",
        )
    elif timeout < 10.0:
        result.add_warning(
            f"HTTP timeout is {timeout}s. "
            "Image generation and long completions may occasionally fail.",
            category="timeout",
        )

    if timeout > 300.0:
        result.add_warning(
            f"HTTP timeout is very long ({timeout}s). "
            "This may cause requests to hang indefinitely on network issues.",
            category="timeout",
            fix="Consider reducing to 60-120s range",
        )


def _validate_connection_pools(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Validate connection pool configuration."""
    max_conn = config.http_client.max_connections

    # Check pool size
    if max_conn < 10:
        result.add_warning(
            f"Connection pool size is small ({max_conn}). "
            "This may limit throughput for concurrent requests.",
            category="performance",
        )
    elif max_conn < 50 and config.environment == "production":
        result.add_recommendation(
            f"Connection pool size is {max_conn}. "
            "For high-throughput production use, consider 200+.",
            category="performance",
        )


def _validate_deprecated_fields(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Report deprecated settings that were changed from their defaults.

    Values are read from each section's ``__dict__`` so pydantic's access-time
    deprecation warning is not raised by the validator itself.
    """
    sections: list[tuple[str, BaseModel | None]] = [
        ("http_client", config.http_client),
        ("scheduler", config.scheduler),
        ("backend.redis", config.backend.redis),
    ]
    for prefix, section in sections:
        if section is None:
            continue
        for name, info in type(section).model_fields.items():
            if not info.deprecated:
                continue
            if section.__dict__.get(name) == info.get_default(call_default_factory=True):
                continue
            result.add_warning(
                f"{prefix}.{name} is deprecated and has no effect.",
                category="deprecated",
                fix=f"Remove {name} from the configuration",
            )


def _validate_rate_limiting(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Validate rate limiting configuration."""
    if not config.scheduler.enable_rate_limiting:
        if config.environment == "production":
            result.add_warning(
                "Rate limiting is disabled in production environment. "
                "This may lead to 429 errors and failed requests.",
                category="rate_limiting",
                fix="Set scheduler.enable_rate_limiting=True",
            )
        return

    # Check buffer ratio
    buffer = config.scheduler.rate_limit_buffer_ratio
    if buffer > 0.95:
        result.add_warning(
            f"Rate limit buffer is aggressive ({buffer:.0%}). "
            "You may experience occasional 429 errors during bursts.",
            category="rate_limiting",
            fix="Consider reducing to 0.85-0.90 for more headroom",
        )
    elif buffer < 0.5:
        result.add_warning(
            f"Rate limit buffer is very conservative ({buffer:.0%}). "
            "You're only using {buffer:.0%} of your available rate limit.",
            category="rate_limiting",
        )

    # Check queue size
    queue_size = config.scheduler.max_queue_size
    if queue_size < 100 and config.environment == "production":
        result.add_warning(
            f"Queue size is small ({queue_size}) for production. "
            "This may cause request rejections during bursts.",
            category="rate_limiting",
            fix="Consider increasing to 1000+ for production",
        )

    # Check overflow policy
    if config.scheduler.overflow_policy == "drop_oldest":
        result.add_warning(
            "Overflow policy is 'drop_oldest' which silently drops requests. "
            "Consider 'reject' for explicit error handling.",
            category="rate_limiting",
        )


def _validate_circuit_breaker(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Validate circuit breaker configuration.

    Note:
        These limits are validated up-front by the SDK but enforced by the
        extracted ``adaptive-rate-limiter`` package's scheduler at runtime.
        Values flow through ``factory._create_rate_limiter`` to
        ``AdaptiveScheduler``.
    """
    if not config.circuit_breaker:
        return

    threshold = config.circuit_breaker.failure_threshold
    reset_timeout = config.circuit_breaker.reset_timeout

    # Check threshold
    if threshold < 3:
        result.add_warning(
            f"Circuit breaker failure threshold is very strict ({threshold}). "
            "May open too frequently on transient errors.",
            category="circuit_breaker",
            fix="Consider increasing to 5-10 for production",
        )
    elif threshold > 50:
        result.add_warning(
            f"Circuit breaker failure threshold is very lenient ({threshold}). "
            "May not protect against cascading failures effectively.",
            category="circuit_breaker",
        )

    # Check reset timeout
    if reset_timeout < 10.0:
        result.add_warning(
            f"Circuit breaker reset timeout is short ({reset_timeout}s). "
            "Services may not have enough time to recover.",
            category="circuit_breaker",
        )
    elif reset_timeout > 300.0:
        result.add_warning(
            f"Circuit breaker reset timeout is very long ({reset_timeout}s). "
            "Recovery from failures will be slow.",
            category="circuit_breaker",
        )


def _validate_redis_config(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Validate Redis-specific configuration."""
    # Inside the SDK today, Redis is only actually contacted when the
    # rate limiter runs in ADAPTIVE mode (it owns the Redis client). Setting
    # ``backend_type=REDIS`` with any other rate-limiter mode is a no-op —
    # the user thinks state is going to Redis but it never is. Surface this
    # as an error so the misleading config doesn't ship silently.
    rate_mode = config.rate_limiter.mode if config.rate_limiter else RateLimiterMode.SIMPLE
    if rate_mode != RateLimiterMode.ADAPTIVE:
        result.add_error(
            "backend_type=BackendType.REDIS only takes effect when "
            f"rate_limiter.mode=RateLimiterMode.ADAPTIVE. With mode={rate_mode.value!r} "
            "the SDK never connects to Redis and the configured Redis URL is ignored.",
            category="redis",
            fix=(
                "Either set rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, "
                "redis_url=...) so Redis is actually used, or set "
                "backend_type=BackendType.MEMORY to make the in-memory backend explicit."
            ),
        )

    if not config.backend.redis:
        result.add_error(
            "Backend type is REDIS but redis configuration is missing",
            category="redis",
            fix="Provide RedisBackendConfig in backend.redis",
        )
        return

    redis_conf = config.backend.redis

    # Check connection pool
    if redis_conf.max_connections < 10:
        result.add_warning(
            f"Redis connection pool is small ({redis_conf.max_connections}). "
            "May cause connection exhaustion under load.",
            category="redis",
            fix="Consider increasing to 30+ for production",
        )

    # Check connection timeout
    if redis_conf.connection_timeout < 1.0:
        result.add_warning(
            f"Redis connection timeout is very short ({redis_conf.connection_timeout}s). "
            "May fail on slow network conditions.",
            category="redis",
        )
    elif redis_conf.connection_timeout > 30.0:
        result.add_warning(
            f"Redis connection timeout is very long ({redis_conf.connection_timeout}s). "
            "Failed connections will block for too long.",
            category="redis",
        )

    # Check TTL
    if redis_conf.default_ttl < 300:
        result.add_recommendation(
            f"Redis default TTL is short ({redis_conf.default_ttl}s). "
            "This may increase cache misses. Consider 1800-3600s for production.",
            category="redis",
        )

    # Validate URL format
    if not redis_conf.redis_url.startswith(("redis://", "rediss://")):
        result.add_error(
            f"Invalid Redis URL format: {redis_conf.redis_url}",
            category="redis",
            fix="Use redis://host:port or rediss://host:port format",
        )


def _add_performance_recommendations(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Add performance optimization recommendations."""
    # Check if metrics are enabled
    if not config.scheduler.metrics_enabled and config.environment == "production":
        result.add_recommendation(
            "Metrics collection is disabled. "
            "Enable metrics for production observability and debugging.",
            category="observability",
        )

    # Check concurrency vs pool size
    max_concurrent = config.scheduler.max_concurrent_executions
    max_conn = config.http_client.max_connections

    if max_concurrent > max_conn:
        result.add_warning(
            f"Max concurrent executions ({max_concurrent}) exceeds HTTP connection pool ({max_conn}). "
            "Requests may be queued waiting for connections.",
            category="performance",
            fix=f"Increase max_connections to at least {max_concurrent}",
        )

    # Recommend connection pool ratio
    if max_concurrent > 0 and max_conn / max_concurrent < 1.5:
        result.add_recommendation(
            f"Connection pool to concurrency ratio is tight ({max_conn}/{max_concurrent} = {max_conn / max_concurrent:.1f}x). "
            "Consider 2x ratio for better performance.",
            category="performance",
        )


def _add_security_recommendations(config: VeniceAIConfig, result: ConfigValidation) -> None:
    """Add security-related recommendations."""
    # Check debug mode in production
    if config.debug and config.environment == "production":
        result.add_warning(
            "Debug mode is enabled in production environment. "
            "This may expose sensitive information in logs.",
            category="security",
            fix="Set debug=False for production",
        )


def print_validation_report(result: ConfigValidation, verbose: bool = False) -> None:
    """
    Print a formatted validation report to console.

    Args:
        result: ConfigValidation result to print
        verbose: If True, include all details and recommendations

    Example:
        >>> from venice_ai.validation import validate_config, print_validation_report
        >>> result = validate_config(config)
        >>> print_validation_report(result, verbose=True)
    """
    print("=" * 60)
    print("Venice AI Configuration Validation Report")
    print("=" * 60)
    print()

    if result.is_valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors that must be fixed")

    print()

    # Print errors
    if result.errors:
        print(f"🔴 Errors ({len(result.errors)}):")
        print("-" * 60)
        for i, error in enumerate(result.errors, 1):
            print(f"{i}. {error}")
            if verbose:
                # Find matching issue for fix suggestion
                for issue in result.issues:
                    if issue.message == error and issue.fix_suggestion:
                        print(f"   Fix: {issue.fix_suggestion}")
        print()

    # Print warnings
    if result.warnings:
        print(f"⚠️  Warnings ({len(result.warnings)}):")
        print("-" * 60)
        for i, warning in enumerate(result.warnings, 1):
            print(f"{i}. {warning}")
            if verbose:
                for issue in result.issues:
                    if issue.message == warning and issue.fix_suggestion:
                        print(f"   Fix: {issue.fix_suggestion}")
        print()

    # Print recommendations
    if result.recommendations and verbose:
        print(f"💡 Recommendations ({len(result.recommendations)}):")
        print("-" * 60)
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
        print()
    elif result.recommendations and not verbose:
        print(f"💡 {len(result.recommendations)} recommendations available (use --verbose to see)")
        print()

    # Summary
    print("=" * 60)
    if result.is_valid:
        if result.warnings or result.recommendations:
            print("Status: ✅ VALID (with suggestions)")
        else:
            print("Status: ✅ PERFECT")
    else:
        print("Status: ❌ INVALID (requires fixes)")
    print("=" * 60)


def validate_config_for_environment(
    config: VeniceAIConfig,
    target_environment: str,
) -> ConfigValidation:
    """
    Validate configuration for a specific target environment.

    Applies stricter checks for production environments.

    Args:
        config: Configuration to validate
        target_environment: Target environment ("production", "staging", "development", "test")

    Returns:
        ConfigValidation with environment-specific checks
    """
    result = validate_config(config)

    # Production-specific validations
    if target_environment == "production":
        if config.backend.backend_type != BackendType.REDIS:
            result.add_warning(
                "Production deployment without Redis backend. "
                "Multi-instance deployments require Redis for state coordination.",
                category="production",
                fix="Use create_production_config() preset",
            )

        if not config.scheduler.enable_rate_limiting:
            result.add_error(
                "Rate limiting must be enabled in production",
                category="production",
                fix="Set scheduler.enable_rate_limiting=True",
            )

        if config.scheduler.mode == SchedulerMode.BASIC:
            result.add_recommendation(
                "Using BASIC scheduler in production. "
                "INTELLIGENT mode provides better rate limit management.",
                category="production",
            )

        if not config.scheduler.metrics_enabled:
            result.add_warning(
                "Metrics are disabled in production. Enable for observability and debugging.",
                category="production",
            )

    # Test environment validations
    elif target_environment == "test":
        if config.backend.backend_type == BackendType.REDIS:
            result.add_recommendation(
                "Using Redis backend in test environment. "
                "Consider memory backend for test isolation.",
                category="testing",
            )

        if config.scheduler.enable_rate_limiting and not hasattr(
            config.scheduler, "test_rate_multiplier"
        ):
            result.add_recommendation(
                "Rate limiting enabled in tests without test_rate_multiplier. Tests may be slow.",
                category="testing",
            )

    return result


def get_configuration_score(config: VeniceAIConfig) -> int:
    """
    Calculate a configuration quality score (0-100).

    Args:
        config: Configuration to score

    Returns:
        Score from 0 (poor) to 100 (perfect)
    """
    result = validate_config(config)

    # Start with perfect score
    score = 100

    # Deduct points for issues
    score -= len(result.errors) * 20  # Errors are serious
    score -= len(result.warnings) * 5  # Warnings are moderate
    score -= min(len(result.recommendations) * 2, 20)  # Recommendations are minor

    return max(0, score)
