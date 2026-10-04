"""
Comprehensive tests for production.py preset coverage.

This module focuses on covering:
- Environment variable fallback paths (lines 85-86, 183-184, 272-273)
- Localhost/127.0.0.1 validation rejection (lines 96, 194, 283)
- All branches in localhost validation conditions
"""

import os
from unittest.mock import patch

import pytest

from venice_ai.core.config import BackendType, SchedulerMode
from venice_ai.presets.production import (
    create_production_config,
    create_production_config_conservative,
    create_production_config_high_throughput,
)


class TestCreateProductionConfigEnvFallback:
    """Test environment variable fallback for create_production_config."""

    def test_redis_url_from_env_variable(self):
        """Test that redis_url is read from VENICE_REDIS_URL when not provided."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://env-redis:6379"),
            clear=True,
        ):
            config = create_production_config()

            assert config.backend.redis is not None
            assert config.backend.redis.redis_url == "redis://env-redis:6379"
            assert config.backend.backend_type == BackendType.REDIS

    def test_redis_url_missing_raises_value_error(self):
        """Test that missing redis_url raises ValueError (lines 85-86)."""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Also need to clear the specific var if it exists
            env = os.environ.copy()
            env.pop("VENICE_REDIS_URL", None)
            env.pop("VENICE_BACKEND__REDIS__REDIS_URL", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    create_production_config(redis_url=None)

                assert "Production Redis URL required" in str(exc_info.value)
                assert "VENICE_REDIS_URL" in str(exc_info.value)

    def test_explicit_redis_url_overrides_env(self):
        """Test that explicit redis_url parameter takes precedence over env."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://env-redis:6379"),
            clear=True,
        ):
            config = create_production_config(redis_url="redis://explicit-redis:6379")

            assert config.backend.redis is not None
            assert config.backend.redis.redis_url == "redis://explicit-redis:6379"


class TestCreateProductionConfigLocalhostValidation:
    """Test localhost validation for create_production_config (line 96)."""

    def test_localhost_rejected_by_default(self):
        """Test that localhost URLs are rejected without _allow_localhost_for_testing."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config(redis_url="redis://localhost:6379")

        error_msg = str(exc_info.value)
        assert "Invalid Redis URL for production" in error_msg
        assert "localhost/127.0.0.1 URLs are not allowed" in error_msg

    def test_127_0_0_1_rejected_by_default(self):
        """Test that 127.0.0.1 URLs are rejected (covers second branch condition)."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config(redis_url="redis://127.0.0.1:6379")

        error_msg = str(exc_info.value)
        assert "Invalid Redis URL for production" in error_msg
        assert "localhost/127.0.0.1 URLs are not allowed" in error_msg

    def test_localhost_allowed_with_testing_flag(self):
        """Test that localhost is allowed with _allow_localhost_for_testing=True."""
        config = create_production_config(
            redis_url="redis://localhost:6379",
            _allow_localhost_for_testing=True,
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://localhost:6379"

    def test_127_0_0_1_allowed_with_testing_flag(self):
        """Test that 127.0.0.1 is allowed with _allow_localhost_for_testing=True."""
        config = create_production_config(
            redis_url="redis://127.0.0.1:6379",
            _allow_localhost_for_testing=True,
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://127.0.0.1:6379"

    def test_localhost_in_path_rejected(self):
        """Test localhost appearing in path is also rejected."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config(redis_url="redis://myhost:6379/localhost")

        assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_env_var_with_localhost_rejected(self):
        """Test that localhost from env var is also rejected (line 96 from env path)."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://localhost:6379"),
            clear=True,
        ):
            with pytest.raises(ValueError) as exc_info:
                create_production_config()

            assert "Invalid Redis URL for production" in str(exc_info.value)


class TestCreateProductionConfigHighThroughputEnvFallback:
    """Test environment variable fallback for create_production_config_high_throughput."""

    def test_redis_url_from_env_variable(self):
        """Test that redis_url is read from VENICE_REDIS_URL when not provided."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://env-fast-redis:6379"),
            clear=True,
        ):
            config = create_production_config_high_throughput()

            assert config.backend.redis is not None
            assert config.backend.redis.redis_url == "redis://env-fast-redis:6379"

    def test_redis_url_missing_raises_value_error(self):
        """Test that missing redis_url raises ValueError (lines 183-184)."""
        env = os.environ.copy()
        env.pop("VENICE_REDIS_URL", None)
        env.pop("VENICE_BACKEND__REDIS__REDIS_URL", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_production_config_high_throughput(redis_url=None)

            assert "Production Redis URL required" in str(exc_info.value)

    def test_explicit_redis_url_overrides_env(self):
        """Test that explicit redis_url takes precedence over env."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://env-redis:6379"),
            clear=True,
        ):
            config = create_production_config_high_throughput(
                redis_url="redis://explicit-fast-redis:6379"
            )

            assert config.backend.redis is not None
            assert config.backend.redis.redis_url == "redis://explicit-fast-redis:6379"


class TestCreateProductionConfigHighThroughputLocalhostValidation:
    """Test localhost validation for create_production_config_high_throughput (line 194)."""

    def test_localhost_rejected_by_default(self):
        """Test that localhost URLs are rejected."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config_high_throughput(redis_url="redis://localhost:6379")

        assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_127_0_0_1_rejected_by_default(self):
        """Test that 127.0.0.1 URLs are rejected (covers or branch)."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config_high_throughput(redis_url="redis://127.0.0.1:6379")

        assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_localhost_allowed_with_testing_flag(self):
        """Test that localhost is allowed with _allow_localhost_for_testing=True."""
        config = create_production_config_high_throughput(
            redis_url="redis://localhost:6379",
            _allow_localhost_for_testing=True,
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://localhost:6379"

    def test_env_var_with_localhost_rejected(self):
        """Test that localhost from env var is also rejected."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://localhost:6379"),
            clear=True,
        ):
            with pytest.raises(ValueError) as exc_info:
                create_production_config_high_throughput()

            assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_env_var_with_127_0_0_1_rejected(self):
        """Test that 127.0.0.1 from env var is also rejected."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://127.0.0.1:6379"),
            clear=True,
        ):
            with pytest.raises(ValueError) as exc_info:
                create_production_config_high_throughput()

            assert "Invalid Redis URL for production" in str(exc_info.value)


class TestCreateProductionConfigConservativeEnvFallback:
    """Test environment variable fallback for create_production_config_conservative."""

    def test_redis_url_from_env_variable(self):
        """Test that redis_url is read from VENICE_REDIS_URL when not provided."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://env-safe-redis:6379"),
            clear=True,
        ):
            config = create_production_config_conservative()

            assert config.backend.redis is not None
            assert config.backend.redis.redis_url == "redis://env-safe-redis:6379"

    def test_redis_url_missing_raises_value_error(self):
        """Test that missing redis_url raises ValueError (lines 272-273)."""
        env = os.environ.copy()
        env.pop("VENICE_REDIS_URL", None)
        env.pop("VENICE_BACKEND__REDIS__REDIS_URL", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_production_config_conservative(redis_url=None)

            assert "Production Redis URL required" in str(exc_info.value)

    def test_explicit_redis_url_overrides_env(self):
        """Test that explicit redis_url takes precedence over env."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://env-redis:6379"),
            clear=True,
        ):
            config = create_production_config_conservative(
                redis_url="redis://explicit-safe-redis:6379"
            )

            assert config.backend.redis is not None
            assert config.backend.redis.redis_url == "redis://explicit-safe-redis:6379"


class TestCreateProductionConfigConservativeLocalhostValidation:
    """Test localhost validation for create_production_config_conservative (line 283)."""

    def test_localhost_rejected_by_default(self):
        """Test that localhost URLs are rejected."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config_conservative(redis_url="redis://localhost:6379")

        assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_127_0_0_1_rejected_by_default(self):
        """Test that 127.0.0.1 URLs are rejected (covers or branch)."""
        with pytest.raises(ValueError) as exc_info:
            create_production_config_conservative(redis_url="redis://127.0.0.1:6379")

        assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_localhost_allowed_with_testing_flag(self):
        """Test that localhost is allowed with _allow_localhost_for_testing=True."""
        config = create_production_config_conservative(
            redis_url="redis://localhost:6379",
            _allow_localhost_for_testing=True,
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://localhost:6379"

    def test_env_var_with_localhost_rejected(self):
        """Test that localhost from env var is also rejected."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://localhost:6379"),
            clear=True,
        ):
            with pytest.raises(ValueError) as exc_info:
                create_production_config_conservative()

            assert "Invalid Redis URL for production" in str(exc_info.value)

    def test_env_var_with_127_0_0_1_rejected(self):
        """Test that 127.0.0.1 from env var is also rejected."""
        with patch.dict(
            os.environ,
            _env_without_redis_urls(VENICE_REDIS_URL="redis://127.0.0.1:6379"),
            clear=True,
        ):
            with pytest.raises(ValueError) as exc_info:
                create_production_config_conservative()

            assert "Invalid Redis URL for production" in str(exc_info.value)


class TestProductionConfigValidUrls:
    """Test that valid production Redis URLs work correctly."""

    def test_production_redis_url(self):
        """Test a typical production Redis URL."""
        config = create_production_config(redis_url="redis://prod-redis.example.com:6379")

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://prod-redis.example.com:6379"
        assert config.environment == "production"
        assert config.debug is False

    def test_redis_url_with_auth(self):
        """Test Redis URL with authentication."""
        config = create_production_config(redis_url="redis://user:password@prod-redis:6379")

        assert config.backend.redis is not None
        assert "user:password" in config.backend.redis.redis_url

    def test_redis_url_with_database(self):
        """Test Redis URL with database number."""
        config = create_production_config(redis_url="redis://prod-redis:6379/5")

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://prod-redis:6379/5"

    def test_redis_cluster_url(self):
        """Test Redis cluster-like URL."""
        config = create_production_config(redis_url="redis://redis-cluster.internal:6379")

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://redis-cluster.internal:6379"


class TestProductionConfigSettings:
    """Test configuration settings are correctly applied."""

    def test_standard_config_settings(self):
        """Test standard production config has correct settings."""
        config = create_production_config(
            redis_url="redis://prod:6379",
            max_concurrent_executions=150,
            max_queue_size=8000,
            enable_metrics=True,
            redis_key_prefix="custom:prefix:",
        )

        assert config.scheduler.max_concurrent_executions == 150
        assert config.scheduler.max_queue_size == 8000
        assert config.scheduler.metrics_enabled is True
        assert config.backend.redis is not None
        # key_prefix is deprecated: reading it warns, and the passed value round-trips.
        with pytest.deprecated_call(match="does not apply a prefix"):
            key_prefix = config.backend.redis.key_prefix
        assert key_prefix == "custom:prefix:"
        assert config.scheduler.mode == SchedulerMode.INTELLIGENT

    def test_high_throughput_config_settings(self):
        """Test high throughput config has aggressive settings."""
        config = create_production_config_high_throughput(
            redis_url="redis://prod:6379",
            redis_key_prefix="fast:",
        )

        assert config.scheduler.max_concurrent_executions == 300
        assert config.scheduler.max_queue_size == 10000
        assert config.http_client.max_connections == 500
        assert config.scheduler.rate_limit_buffer_ratio == 0.95
        assert config.backend.redis is not None
        assert config.backend.redis.max_connections == 100

    def test_conservative_config_settings(self):
        """Test conservative config has safe settings."""
        config = create_production_config_conservative(
            redis_url="redis://prod:6379",
            redis_key_prefix="safe:",
        )

        assert config.scheduler.max_concurrent_executions == 50
        assert config.scheduler.max_queue_size == 2000
        assert config.scheduler.rate_limit_buffer_ratio == 0.8
        assert config.http_client.timeout == 60.0
        assert config.http_client.max_retries == 5
        assert config.backend.redis is not None
        assert config.backend.redis.max_retries == 5
        assert config.circuit_breaker.failure_threshold == 5


PRESETS = [
    create_production_config,
    create_production_config_high_throughput,
    create_production_config_conservative,
]


def _env_without_redis_urls(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VENICE_REDIS_URL", "VENICE_BACKEND__REDIS__REDIS_URL")
    }
    env.update(extra)
    return env


@pytest.mark.parametrize("preset", PRESETS, ids=lambda f: f.__name__)
class TestRedisUrlEnvPrecedence:
    """VENICE_BACKEND__REDIS__REDIS_URL wins; VENICE_REDIS_URL is the fallback."""

    def test_nested_setting_alone_is_used(self, preset):
        env = _env_without_redis_urls(VENICE_BACKEND__REDIS__REDIS_URL="redis://nested:6379")
        with patch.dict(os.environ, env, clear=True):
            config = preset()
        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://nested:6379"
        assert config.rate_limiter.redis_url == "redis://nested:6379"

    def test_nested_setting_wins_over_short_form(self, preset):
        env = _env_without_redis_urls(
            VENICE_BACKEND__REDIS__REDIS_URL="redis://nested:6379",
            VENICE_REDIS_URL="redis://short:6379",
        )
        with patch.dict(os.environ, env, clear=True):
            config = preset()
        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://nested:6379"

    def test_empty_nested_setting_falls_back_to_short_form(self, preset):
        env = _env_without_redis_urls(
            VENICE_BACKEND__REDIS__REDIS_URL="", VENICE_REDIS_URL="redis://short:6379"
        )
        with patch.dict(os.environ, env, clear=True):
            config = preset()
        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://short:6379"

    def test_explicit_argument_wins_over_both(self, preset):
        env = _env_without_redis_urls(
            VENICE_BACKEND__REDIS__REDIS_URL="redis://nested:6379",
            VENICE_REDIS_URL="redis://short:6379",
        )
        with patch.dict(os.environ, env, clear=True):
            config = preset(redis_url="redis://explicit:6379")
        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://explicit:6379"

    def test_nested_setting_localhost_rejected(self, preset):
        env = _env_without_redis_urls(VENICE_BACKEND__REDIS__REDIS_URL="redis://localhost:6379")
        with patch.dict(os.environ, env, clear=True), pytest.raises(ValueError, match="localhost"):
            preset()

    def test_missing_names_both_variables(self, preset):
        with (
            patch.dict(os.environ, _env_without_redis_urls(), clear=True),
            pytest.raises(ValueError) as exc_info,
        ):
            preset()
        assert "VENICE_BACKEND__REDIS__REDIS_URL" in str(exc_info.value)
        assert "VENICE_REDIS_URL" in str(exc_info.value)
