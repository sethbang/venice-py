"""
Unit tests for configuration validation module.

Tests cover:
- Configuration validation with various scenarios
- Error detection and reporting
- Warning generation
- Recommendations
- Environment-specific validation
- Configuration scoring
"""

from unittest.mock import patch

import pytest

from venice_ai.core.config import (
    BackendConfig,
    BackendType,
    CircuitBreakerConfig,
    HttpClientConfig,
    RedisBackendConfig,
    SchedulerConfig,
    SchedulerMode,
    VeniceAIConfig,
)
from venice_ai.validation import (
    ConfigValidation,
    get_configuration_score,
    print_validation_report,
    validate_config,
    validate_config_for_environment,
)


class TestConfigValidation:
    """Test ConfigValidation class."""

    def test_init_valid(self):
        """Test initialization of valid config."""
        result = ConfigValidation(is_valid=True)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_add_error(self):
        """Test adding errors."""
        result = ConfigValidation(is_valid=True)
        result.add_error("Test error", category="test", fix="Fix it")

        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Test error" in result.errors
        assert len(result.issues) == 1
        assert result.issues[0].fix_suggestion == "Fix it"

    def test_add_warning(self):
        """Test adding warnings."""
        result = ConfigValidation(is_valid=True)
        result.add_warning("Test warning", category="test")

        assert result.is_valid  # Warnings don't invalidate
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings

    def test_add_recommendation(self):
        """Test adding recommendations."""
        result = ConfigValidation(is_valid=True)
        result.add_recommendation("Test recommendation", category="test")

        assert result.is_valid
        assert len(result.recommendations) == 1
        assert "Test recommendation" in result.recommendations


class TestValidateSchedulerBackendCompatibility:
    """Test scheduler and backend compatibility validation."""

    def test_intelligent_with_memory_warning(self):
        """Test warning for INTELLIGENT mode with memory backend."""
        config = VeniceAIConfig(
            backend=BackendConfig(backend_type=BackendType.MEMORY),
            scheduler=SchedulerConfig(mode=SchedulerMode.INTELLIGENT),
        )
        result = validate_config(config)

        assert result.is_valid
        assert any("INTELLIGENT scheduler" in w for w in result.warnings)

    def test_intelligent_with_redis_no_warning(self):
        """Test no warning for INTELLIGENT mode with Redis."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            scheduler=SchedulerConfig(mode=SchedulerMode.INTELLIGENT),
        )
        result = validate_config(config)

        # May have other warnings but not about scheduler/backend compatibility
        scheduler_warnings = [w for w in result.warnings if "INTELLIGENT scheduler" in w]
        assert len(scheduler_warnings) == 0


class TestValidateTimeouts:
    """Test timeout validation."""

    def test_very_short_timeout(self):
        """Test warning for very short timeout."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(timeout=3.0),
        )
        result = validate_config(config)

        assert any("very short" in w.lower() for w in result.warnings)

    def test_short_timeout(self):
        """Test warning for short timeout."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(timeout=8.0),
        )
        result = validate_config(config)

        # Should have timeout warning
        assert any("timeout" in w.lower() for w in result.warnings)

    def test_very_long_timeout(self):
        """Test warning for very long timeout."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(timeout=400.0),
        )
        result = validate_config(config)

        assert any("very long" in w.lower() for w in result.warnings)

    def test_reasonable_timeout(self):
        """Test no warning for reasonable timeout."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(timeout=30.0),
        )
        result = validate_config(config)

        # No timeout warnings
        timeout_warnings = [w for w in result.warnings if "timeout" in w.lower()]
        assert len(timeout_warnings) == 0


class TestValidateConnectionPools:
    """Test connection pool validation."""

    def test_small_pool(self):
        """Test warning for small connection pool."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(max_connections=5),
        )
        result = validate_config(config)

        assert any("pool size is small" in w.lower() for w in result.warnings)

    def test_small_pool_production(self):
        """Test recommendation for small pool in production."""
        config = VeniceAIConfig(
            environment="production",
            http_client=HttpClientConfig(max_connections=30),
        )
        result = validate_config(config)

        assert any("connection pool" in r.lower() for r in result.recommendations)

    def test_deprecated_keepalive_is_reported(self):
        """A non-default max_keepalive_connections has no effect and is reported."""
        # VeniceAIConfig re-validates the nested model, which warns again.
        with pytest.warns(FutureWarning, match="max_keepalive_connections"):
            http_client = HttpClientConfig(max_connections=10, max_keepalive_connections=5)
            config = VeniceAIConfig(http_client=http_client)
        result = validate_config(config)

        assert result.is_valid
        assert any("max_keepalive_connections is deprecated" in w for w in result.warnings)

    def test_default_keepalive_is_not_reported(self):
        """Leaving the deprecated field alone produces no deprecation finding."""
        result = validate_config(VeniceAIConfig(http_client=HttpClientConfig(max_connections=100)))

        assert not any("keepalive" in issue.message.lower() for issue in result.issues)


class TestValidateRateLimiting:
    """Test rate limiting validation."""

    def test_disabled_in_production_warning(self):
        """Test warning when rate limiting disabled in production."""
        config = VeniceAIConfig(
            environment="production",
            scheduler=SchedulerConfig(enable_rate_limiting=False),
        )
        result = validate_config(config)

        assert any("rate limiting is disabled" in w.lower() for w in result.warnings)

    def test_aggressive_buffer(self):
        """Test warning for aggressive rate limit buffer."""
        config = VeniceAIConfig(
            scheduler=SchedulerConfig(
                enable_rate_limiting=True,
                rate_limit_buffer_ratio=0.98,
            ),
        )
        result = validate_config(config)

        assert any("aggressive" in w.lower() for w in result.warnings)

    def test_conservative_buffer(self):
        """Test warning for very conservative buffer."""
        config = VeniceAIConfig(
            scheduler=SchedulerConfig(
                enable_rate_limiting=True,
                rate_limit_buffer_ratio=0.4,
            ),
        )
        result = validate_config(config)

        assert any("conservative" in w.lower() for w in result.warnings)

    def test_small_queue_production(self):
        """Test warning for small queue in production."""
        config = VeniceAIConfig(
            environment="production",
            scheduler=SchedulerConfig(
                enable_rate_limiting=True,
                max_queue_size=50,
            ),
        )
        result = validate_config(config)

        assert any("queue size is small" in w.lower() for w in result.warnings)

    def test_drop_oldest_policy(self):
        """Test warning for drop_oldest overflow policy."""
        config = VeniceAIConfig(
            scheduler=SchedulerConfig(
                enable_rate_limiting=True,
                overflow_policy="drop_oldest",
            ),
        )
        result = validate_config(config)

        assert any("drop_oldest" in w for w in result.warnings)


class TestValidateCircuitBreaker:
    """Test circuit breaker validation."""

    def test_strict_threshold(self):
        """Test warning for very strict threshold."""
        config = VeniceAIConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=2),
        )
        result = validate_config(config)

        assert any("very strict" in w.lower() for w in result.warnings)

    def test_lenient_threshold(self):
        """Test warning for very lenient threshold."""
        config = VeniceAIConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=60),
        )
        result = validate_config(config)

        assert any("very lenient" in w.lower() for w in result.warnings)

    def test_short_reset_timeout(self):
        """Test warning for short reset timeout."""
        config = VeniceAIConfig(
            circuit_breaker=CircuitBreakerConfig(reset_timeout=5.0),
        )
        result = validate_config(config)

        assert any("short" in w.lower() and "reset timeout" in w.lower() for w in result.warnings)

    def test_long_reset_timeout(self):
        """Test warning for very long reset timeout."""
        config = VeniceAIConfig(
            circuit_breaker=CircuitBreakerConfig(reset_timeout=400.0),
        )
        result = validate_config(config)

        assert any("very long" in w.lower() for w in result.warnings)

    def test_no_circuit_breaker(self):
        """Test validation works without circuit breaker warnings."""
        from venice_ai.core.config import BackendConfig, BackendType

        config = VeniceAIConfig(
            backend=BackendConfig(backend_type=BackendType.MEMORY),
            http_client=HttpClientConfig(timeout=30.0),
            scheduler=SchedulerConfig(enable_rate_limiting=False),
        )
        result = validate_config(config)

        # Should still be valid even with default circuit breaker
        assert result.is_valid or len(result.errors) == 0


class TestValidateRedisConfig:
    """Test Redis configuration validation."""

    def test_missing_redis_config_error(self):
        """Test error when Redis type but no config."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
        )

        # Mock the redis attribute to be None to test the error path
        with patch.object(config.backend, "redis", None):
            result = validate_config(config)

            assert not result.is_valid
            assert any("redis configuration is missing" in e.lower() for e in result.errors)

    def test_small_connection_pool(self):
        """Test warning for small Redis connection pool."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url="redis://localhost:6379",
                    max_connections=5,
                ),
            ),
        )
        result = validate_config(config)

        assert any("pool is small" in w.lower() for w in result.warnings)

    def test_short_connection_timeout(self):
        """Test warning for short connection timeout."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url="redis://localhost:6379",
                    connection_timeout=0.5,
                ),
            ),
        )
        result = validate_config(config)

        assert any("very short" in w.lower() for w in result.warnings)

    def test_long_connection_timeout(self):
        """Test warning for very long connection timeout."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url="redis://localhost:6379",
                    connection_timeout=40.0,
                ),
            ),
        )
        result = validate_config(config)

        assert any("very long" in w.lower() for w in result.warnings)

    def test_short_ttl(self):
        """Test recommendation for short TTL."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url="redis://localhost:6379",
                    default_ttl=200,
                ),
            ),
        )
        result = validate_config(config)

        assert any("ttl is short" in r.lower() for r in result.recommendations)

    def test_invalid_redis_url(self):
        """Test error for invalid Redis URL format."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url="http://localhost:6379",  # Wrong protocol
                ),
            ),
        )
        result = validate_config(config)

        assert not result.is_valid
        assert any("invalid redis url" in e.lower() for e in result.errors)


class TestPerformanceRecommendations:
    """Test performance recommendations."""

    def test_metrics_disabled_production(self):
        """Test recommendation when metrics disabled in production."""
        config = VeniceAIConfig(
            environment="production",
            scheduler=SchedulerConfig(metrics_enabled=False),
        )
        result = validate_config(config)

        assert any("metrics" in r.lower() for r in result.recommendations)

    def test_concurrent_exceeds_connections(self):
        """Test warning when max concurrent exceeds connection pool."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(max_connections=50),
            scheduler=SchedulerConfig(max_concurrent_executions=100),
        )
        result = validate_config(config)

        assert any("exceeds http connection pool" in w.lower() for w in result.warnings)

    def test_tight_connection_ratio(self):
        """Test recommendation for tight connection/concurrency ratio."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(max_connections=100),
            scheduler=SchedulerConfig(max_concurrent_executions=80),
        )
        result = validate_config(config)

        assert any("ratio is tight" in r.lower() for r in result.recommendations)


class TestSecurityRecommendations:
    """Test security recommendations."""

    def test_debug_in_production(self):
        """Test warning for debug mode in production."""
        config = VeniceAIConfig(
            environment="production",
            debug=True,
        )
        result = validate_config(config)

        assert any("debug mode" in w.lower() and "production" in w.lower() for w in result.warnings)

    def test_deprecated_redis_key_prefix_is_reported(self):
        """A custom Redis key_prefix is ignored at runtime and reported as deprecated."""
        config = VeniceAIConfig(
            environment="production",
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url="redis://localhost:6379",
                    key_prefix="myapp:",
                ),
            ),
        )
        result = validate_config(config)

        assert any("backend.redis.key_prefix is deprecated" in w for w in result.warnings)


class TestEnvironmentSpecificValidation:
    """Test environment-specific validation."""

    def test_production_without_redis(self):
        """Test warning for production without Redis."""
        config = VeniceAIConfig(
            environment="development",
            backend=BackendConfig(backend_type=BackendType.MEMORY),
        )
        result = validate_config_for_environment(config, "production")

        assert any("without redis" in w.lower() for w in result.warnings)

    def test_production_without_rate_limiting(self):
        """Test error for production without rate limiting."""
        config = VeniceAIConfig(
            environment="development",
            scheduler=SchedulerConfig(enable_rate_limiting=False),
        )
        result = validate_config_for_environment(config, "production")

        assert not result.is_valid
        assert any("rate limiting must be enabled" in e.lower() for e in result.errors)

    def test_production_basic_scheduler(self):
        """Test recommendation for BASIC scheduler in production."""
        config = VeniceAIConfig(
            environment="development",
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            scheduler=SchedulerConfig(
                mode=SchedulerMode.BASIC,
                enable_rate_limiting=True,
            ),
        )
        result = validate_config_for_environment(config, "production")

        assert any("basic scheduler" in r.lower() for r in result.recommendations)

    def test_production_metrics_disabled(self):
        """Test warning for disabled metrics in production."""
        config = VeniceAIConfig(
            environment="development",
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            scheduler=SchedulerConfig(
                enable_rate_limiting=True,
                metrics_enabled=False,
            ),
        )
        result = validate_config_for_environment(config, "production")

        assert any("metrics are disabled" in w.lower() for w in result.warnings)

    def test_test_with_redis(self):
        """Test recommendation for Redis in test environment."""
        config = VeniceAIConfig(
            environment="test",
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
        )
        result = validate_config_for_environment(config, "test")

        assert any("redis backend in test" in r.lower() for r in result.recommendations)


class TestConfigurationScore:
    """Test configuration quality scoring."""

    def test_perfect_score(self):
        """Test perfect configuration gets high score."""
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig(
            environment="production",
            debug=False,
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            # BackendType.REDIS only takes effect with adaptive rate limiting —
            # without this, the validator (rightly) errors and the score drops.
            rate_limiter=RateLimiterConfig(
                mode=RateLimiterMode.ADAPTIVE,
                redis_url="redis://localhost:6379",
            ),
            http_client=HttpClientConfig(
                timeout=30.0,
                max_connections=200,
            ),
            scheduler=SchedulerConfig(
                mode=SchedulerMode.INTELLIGENT,
                enable_rate_limiting=True,
                metrics_enabled=True,
            ),
        )
        score = get_configuration_score(config)

        assert score >= 80  # Should be quite high

    def test_low_score_with_errors(self):
        """Test configuration with errors gets low score."""
        config = VeniceAIConfig(
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),  # Error: Redis is never used without the ADAPTIVE rate limiter
        )
        score = get_configuration_score(config)

        assert score < 100  # Should be penalized

    def test_score_bounds(self):
        """Test score stays within 0-100 bounds."""
        # Config with many issues
        config = VeniceAIConfig(
            environment="production",
            debug=True,  # Warning
            backend=BackendConfig(backend_type=BackendType.MEMORY),  # Warning for prod
            http_client=HttpClientConfig(
                timeout=2.0,  # Warning
                max_connections=5,  # Warning
            ),
            scheduler=SchedulerConfig(
                enable_rate_limiting=False,  # Warning
            ),
        )
        score = get_configuration_score(config)

        assert 0 <= score <= 100


class TestPrintValidationReport:
    """Test validation report printing."""

    def test_print_valid_config(self, capsys):
        """Test printing report for valid config."""
        config = VeniceAIConfig()
        result = validate_config(config)

        print_validation_report(result, verbose=False)
        captured = capsys.readouterr()

        assert "Configuration Validation Report" in captured.out

    def test_print_with_errors(self, capsys):
        """Test printing report with errors."""
        result = ConfigValidation(is_valid=False)
        result.add_error("Test error", fix="Fix it")

        print_validation_report(result, verbose=True)
        captured = capsys.readouterr()

        assert "Errors" in captured.out
        assert "Test error" in captured.out
        assert "Fix: Fix it" in captured.out

    def test_print_with_warnings(self, capsys):
        """Test printing report with warnings."""
        result = ConfigValidation(is_valid=True)
        result.add_warning("Test warning", fix="Consider this")

        print_validation_report(result, verbose=True)
        captured = capsys.readouterr()

        assert "Warnings" in captured.out
        assert "Test warning" in captured.out

    def test_print_with_recommendations_verbose(self, capsys):
        """Test printing recommendations in verbose mode."""
        result = ConfigValidation(is_valid=True)
        result.add_recommendation("Test recommendation")

        print_validation_report(result, verbose=True)
        captured = capsys.readouterr()

        assert "Recommendations" in captured.out
        assert "Test recommendation" in captured.out

    def test_print_with_recommendations_non_verbose(self, capsys):
        """Test recommendations summary in non-verbose mode."""
        result = ConfigValidation(is_valid=True)
        result.add_recommendation("Test recommendation")

        print_validation_report(result, verbose=False)
        captured = capsys.readouterr()

        assert "recommendations available" in captured.out
        assert "Test recommendation" not in captured.out
