"""Tests for venice_ai.validation.config_validator module."""

import pytest

from venice_ai.core.config import (
    BackendConfig,
    BackendType,
    CircuitBreakerConfig,
    HttpClientConfig,
    SchedulerConfig,
    SchedulerMode,
    VeniceAIConfig,
)
from venice_ai.validation.config_validator import (
    ConfigValidation,
    Severity,
    ValidationIssue,
    get_configuration_score,
    print_validation_report,
    validate_config,
    validate_config_for_environment,
)


def _make_config(**kwargs) -> VeniceAIConfig:
    """Helper to create configs bypassing env var loading."""
    return VeniceAIConfig.model_construct(**kwargs)


class TestConfigValidation:
    """Test ConfigValidation dataclass methods."""

    def test_add_error(self):
        result = ConfigValidation(is_valid=True)
        result.add_error("something broke", category="test", fix="fix it")

        assert not result.is_valid
        assert "something broke" in result.errors
        assert result.issues[0].severity == Severity.ERROR
        assert result.issues[0].fix_suggestion == "fix it"

    def test_add_warning(self):
        result = ConfigValidation(is_valid=True)
        result.add_warning("be careful", category="test")

        assert result.is_valid  # warnings don't invalidate
        assert "be careful" in result.warnings

    def test_add_recommendation(self):
        result = ConfigValidation(is_valid=True)
        result.add_recommendation("try this", category="perf")

        assert "try this" in result.recommendations
        assert result.issues[0].severity == Severity.INFO


class TestValidateConfig:
    """Test validate_config with various configurations."""

    def test_default_config_is_valid(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert result.is_valid

    def test_intelligent_scheduler_without_redis_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(mode=SchedulerMode.INTELLIGENT),
            backend=BackendConfig(backend_type=BackendType.MEMORY),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("INTELLIGENT" in w for w in result.warnings)

    def test_very_short_timeout_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(timeout=3.0),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("very short" in w.lower() for w in result.warnings)

    def test_short_timeout_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(timeout=8.0),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("timeout" in w.lower() for w in result.warnings)

    def test_very_long_timeout_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(timeout=400.0),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("very long" in w.lower() for w in result.warnings)

    def test_deprecated_keepalive_is_a_warning_not_an_error(self):
        with pytest.warns(FutureWarning, match="max_keepalive_connections"):
            http_client = HttpClientConfig(max_connections=10, max_keepalive_connections=5)
        config = _make_config(
            environment="production",
            debug=False,
            http_client=http_client,
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert not any("keepalive" in e.lower() for e in result.errors)
        assert any("max_keepalive_connections is deprecated" in w for w in result.warnings)

    def test_small_connection_pool_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(max_connections=5),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("pool size is small" in w.lower() for w in result.warnings)

    def test_rate_limiting_disabled_in_production_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(enable_rate_limiting=False),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("rate limiting is disabled" in w.lower() for w in result.warnings)

    def test_aggressive_rate_limit_buffer_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(enable_rate_limiting=True, rate_limit_buffer_ratio=0.98),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("aggressive" in w.lower() for w in result.warnings)

    def test_conservative_rate_limit_buffer_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(enable_rate_limiting=True, rate_limit_buffer_ratio=0.3),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("conservative" in w.lower() for w in result.warnings)

    def test_drop_oldest_overflow_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(enable_rate_limiting=True, overflow_policy="drop_oldest"),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("drop_oldest" in w for w in result.warnings)

    def test_strict_circuit_breaker_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=2),
        )
        result = validate_config(config)
        assert any("very strict" in w.lower() for w in result.warnings)

    def test_lenient_circuit_breaker_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=60),
        )
        result = validate_config(config)
        assert any("very lenient" in w.lower() for w in result.warnings)

    def test_short_circuit_breaker_reset_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(reset_timeout=5.0),
        )
        result = validate_config(config)
        assert any("reset timeout is short" in w.lower() for w in result.warnings)

    def test_long_circuit_breaker_reset_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(reset_timeout=400.0),
        )
        result = validate_config(config)
        assert any("reset timeout is very long" in w.lower() for w in result.warnings)

    def test_concurrent_exceeds_connections_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(max_connections=10),
            scheduler=SchedulerConfig(max_concurrent_executions=20),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("concurrent" in w.lower() for w in result.warnings)

    def test_debug_in_production_warns(self):
        config = _make_config(
            environment="production",
            debug=True,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("debug mode" in w.lower() for w in result.warnings)


class TestRedisValidation:
    """Test Redis-specific validation."""

    def test_redis_backend_without_config_errors(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(backend_type=BackendType.REDIS, redis=None),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert not result.is_valid
        assert any("redis configuration is missing" in e.lower() for e in result.errors)

    def test_invalid_redis_url_errors(self):
        from venice_ai.core.config import RedisBackendConfig

        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="http://invalid"),
            ),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert any("invalid redis url" in e.lower() for e in result.errors)

    def test_redis_backend_without_adaptive_mode_errors(self):
        """``BackendType.REDIS`` is a no-op without ``RateLimiterMode.ADAPTIVE`` —
        the validator must surface this so users don't think state is going to
        Redis when it isn't."""
        from venice_ai.core.config import RedisBackendConfig
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            rate_limiter=RateLimiterConfig(mode=RateLimiterMode.SIMPLE),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        assert not result.is_valid
        assert any(
            "only takes effect when" in e.lower() and "adaptive" in e.lower() for e in result.errors
        )

    def test_redis_backend_with_adaptive_mode_does_not_error(self):
        from venice_ai.core.config import RedisBackendConfig
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config(config)
        # No "only takes effect" error in the ADAPTIVE happy path.
        assert not any("only takes effect when" in e.lower() for e in result.errors)


class TestValidateForEnvironment:
    """Test environment-specific validation."""

    def test_production_without_redis_warns(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(backend_type=BackendType.MEMORY),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config_for_environment(config, "production")
        assert any("without redis" in w.lower() for w in result.warnings)

    def test_production_without_rate_limiting_errors(self):
        config = _make_config(
            environment="production",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(enable_rate_limiting=False),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config_for_environment(config, "production")
        assert any("rate limiting must be enabled" in e.lower() for e in result.errors)

    def test_test_environment_with_redis_recommends(self):
        from venice_ai.core.config import RedisBackendConfig

        config = _make_config(
            environment="test",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(redis_url="redis://localhost:6379"),
            ),
            circuit_breaker=CircuitBreakerConfig(),
        )
        result = validate_config_for_environment(config, "test")
        assert any("test" in r.lower() and "memory" in r.lower() for r in result.recommendations)


class TestConfigurationScore:
    """Test get_configuration_score."""

    def test_perfect_config_scores_high(self):
        config = _make_config(
            environment="development",
            debug=False,
            http_client=HttpClientConfig(),
            scheduler=SchedulerConfig(),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        score = get_configuration_score(config)
        assert score >= 80

    def test_bad_config_scores_low(self):
        config = _make_config(
            environment="production",
            debug=True,
            http_client=HttpClientConfig(timeout=2.0, max_connections=5),
            scheduler=SchedulerConfig(enable_rate_limiting=False),
            backend=BackendConfig(),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
        )
        score = get_configuration_score(config)
        assert score < 80

    def test_score_never_negative(self):
        # Extreme bad config
        config = _make_config(
            environment="production",
            debug=True,
            http_client=HttpClientConfig(timeout=1.0, max_connections=2),
            scheduler=SchedulerConfig(
                enable_rate_limiting=False,
                max_concurrent_executions=1000,
            ),
            backend=BackendConfig(backend_type=BackendType.REDIS, redis=None),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1, reset_timeout=1.0),
        )
        score = get_configuration_score(config)
        assert score >= 0

    def test_score_is_clamped_at_zero(self, monkeypatch):
        from venice_ai.validation import config_validator

        result = ConfigValidation(is_valid=False)
        result.errors.extend(f"error {i}" for i in range(10))
        monkeypatch.setattr(config_validator, "validate_config", lambda _config: result)
        assert get_configuration_score(_make_config()) == 0


class TestPrintValidationReport:
    """Test print_validation_report output."""

    def test_print_valid_report(self, capsys):
        result = ConfigValidation(is_valid=True)
        print_validation_report(result)
        output = capsys.readouterr().out
        assert "valid" in output.lower()

    def test_print_invalid_report(self, capsys):
        result = ConfigValidation(is_valid=False)
        result.errors.append("Bad config")
        result.issues.append(
            ValidationIssue(Severity.ERROR, "Bad config", "test", fix_suggestion="Fix it")
        )
        print_validation_report(result, verbose=True)
        output = capsys.readouterr().out
        assert "Bad config" in output
        assert "Fix it" in output

    def test_print_warnings_and_recommendations(self, capsys):
        result = ConfigValidation(is_valid=True)
        result.warnings.append("Be careful")
        result.issues.append(ValidationIssue(Severity.WARNING, "Be careful", "test"))
        result.recommendations.append("Try this")
        result.issues.append(ValidationIssue(Severity.INFO, "Try this", "test"))

        print_validation_report(result, verbose=True)
        output = capsys.readouterr().out
        assert "Be careful" in output
        assert "Try this" in output

    def test_print_non_verbose_hides_recommendations(self, capsys):
        result = ConfigValidation(is_valid=True)
        result.recommendations.append("Hidden rec")
        result.issues.append(ValidationIssue(Severity.INFO, "Hidden rec", "test"))

        print_validation_report(result, verbose=False)
        output = capsys.readouterr().out
        assert "Hidden rec" not in output
        assert "recommendations available" in output
