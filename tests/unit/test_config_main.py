"""Tests for VeniceAIConfig validators and utility methods in core/config/main.py."""

import os

import pytest
from pydantic import ValidationError

from venice_ai.core.config.main import VeniceAIConfig, create_minimal_config


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Remove VENICE_* env vars so pydantic-settings doesn't pollute defaults."""
    for key in list(os.environ):
        if key.startswith("VENICE_"):
            monkeypatch.delenv(key)


class TestVeniceAIConfigValidators:
    """Test VeniceAIConfig field validators."""

    def test_invalid_environment(self):
        with pytest.raises(ValidationError, match="Environment must be one of"):
            VeniceAIConfig(environment="invalid_env")

    def test_valid_environments(self):
        for env in ("production", "staging", "development", "test"):
            config = VeniceAIConfig(environment=env)
            assert config.environment == env

    def test_environment_case_insensitive(self):
        config = VeniceAIConfig(environment="Production")
        assert config.environment == "production"

    def test_api_base_url_missing_protocol(self):
        with pytest.raises(ValidationError, match="absolute http:// or https:// URL"):
            VeniceAIConfig(api_base_url="api.venice.ai")

    def test_api_base_url_trailing_slash_stripped(self):
        config = VeniceAIConfig(api_base_url="https://api.venice.ai/")
        assert not config.api_base_url.endswith("/")

    def test_api_base_url_bare_host_becomes_the_api_root(self):
        config = VeniceAIConfig(api_base_url="https://api.venice.ai")
        assert config.api_base_url == "https://api.venice.ai/api/v1"

    def test_api_base_url_api_root_is_kept(self):
        config = VeniceAIConfig(api_base_url="https://api.venice.ai/api/v1/")
        assert config.api_base_url == "https://api.venice.ai/api/v1"


class TestVeniceAIConfigMethods:
    """Test VeniceAIConfig utility methods."""

    def test_get_redis_url_no_redis_config(self):
        config = VeniceAIConfig()
        with pytest.raises(ValueError, match="Redis backend configuration is required"):
            config.get_redis_url()

    def test_is_test_environment_by_env(self):
        config = VeniceAIConfig(environment="test")
        assert config.is_test_environment() is True

    def test_is_test_environment_by_scheduler_test_mode(self):
        from venice_ai.core.config.enterprise import SchedulerConfig

        config = VeniceAIConfig(
            environment="production",
            scheduler=SchedulerConfig(test_mode=True),
        )
        assert config.is_test_environment() is True

    def test_is_test_environment_false(self):
        config = VeniceAIConfig(environment="production")
        assert config.is_test_environment() is False

    def test_is_debug_enabled_true(self):
        config = VeniceAIConfig(debug=True)
        assert config.is_debug_enabled() is True

    def test_is_debug_enabled_false(self):
        config = VeniceAIConfig(debug=False)
        assert config.is_debug_enabled() is False


class TestCreateMinimalConfig:
    """Test module-level create_minimal_config convenience function."""

    def test_returns_config(self):
        config = create_minimal_config(api_key="test-key")
        assert isinstance(config, VeniceAIConfig)
        assert config.api_key == "test-key"

    def test_without_api_key(self):
        config = create_minimal_config()
        assert isinstance(config, VeniceAIConfig)
        assert config.api_key is None
