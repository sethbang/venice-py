"""
Test module for VeniceClientFactory functionality.

This module provides comprehensive test coverage for the factory pattern implementation
that serves as the composition root for dependency injection in Venice AI v2.0.0.
"""

import warnings
from unittest.mock import Mock, patch

import pytest

from venice_ai.core.config import (
    BackendType,
    RateLimiterMode,
    SchedulerConfig,
    SchedulerMode,
    VeniceAIConfig,
)
from venice_ai.factory import (
    VeniceClientFactory,
    create_developer_client,
    create_test_venice_client,
    create_venice_client,
)


class TestVeniceClientFactoryBasicCreation:
    """Test basic client creation through the factory."""

    @patch("venice_ai._client.VeniceClient")
    def test_create_client_basic(self, mock_venice_client_class):
        """Test basic client creation with minimal parameters."""
        mock_client = Mock()
        mock_venice_client_class.return_value = mock_client

        config = VeniceAIConfig.create_minimal_config()

        result = VeniceClientFactory.create_client(config)

        # Verify VeniceClient creation
        mock_venice_client_class.assert_called_once()
        call_kwargs = mock_venice_client_class.call_args[1]
        assert call_kwargs["config"] == config

        assert result == mock_client

    @patch("venice_ai._client.VeniceClient")
    def test_create_client_with_all_parameters(self, mock_venice_client_class):
        """Test client creation with all parameters specified."""
        mock_client = Mock()
        mock_venice_client_class.return_value = mock_client

        config = VeniceAIConfig.create_test_config()
        mock_http_client = Mock()

        result = VeniceClientFactory.create_client(
            config=config,
            api_key="custom-api-key",
            account_id="custom-account",
            http_client=mock_http_client,
        )

        # Verify VeniceClient creation with custom parameters
        call_kwargs = mock_venice_client_class.call_args[1]
        assert call_kwargs["api_key"] == "custom-api-key"
        assert call_kwargs["http_client"] == mock_http_client
        # The client reads the API root from config.api_base_url itself.
        assert "base_url" not in call_kwargs
        assert call_kwargs["config"] is config

        assert result == mock_client

    @patch("venice_ai._client.VeniceClient")
    def test_create_test_client(self, mock_venice_client_class):
        """Test create_test_client method."""
        mock_client = Mock()
        mock_venice_client_class.return_value = mock_client

        with patch.object(VeniceClientFactory, "create_client") as mock_create_client:
            mock_create_client.return_value = mock_client

            result = VeniceClientFactory.create_test_client(
                enable_redis=False,
                test_rate_multiplier=5.0,
                custom_param="test",
            )

            # Verify test config creation and client creation
            mock_create_client.assert_called_once()
            call_args = mock_create_client.call_args
            config_arg = call_args[0][0]  # First positional argument

            # Verify test configuration
            assert config_arg.environment == "test"
            assert config_arg.debug is True
            # create_test_config defaults to BASIC so callers without
            # tier-discovery setup get a working test client out of the box.
            assert config_arg.scheduler.mode == SchedulerMode.BASIC
            assert config_arg.is_test_environment() is True
            assert config_arg.scheduler.test_rate_multiplier == 5.0
            assert config_arg.backend.backend_type == BackendType.MEMORY

            # Verify additional kwargs
            call_kwargs = call_args[1]
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["account_id"] == "test-account"
            assert call_kwargs["custom_param"] == "test"

            assert result == mock_client

    @patch("venice_ai._client.VeniceClient")
    def test_create_minimal_client(self, mock_venice_client_class):
        """Test create_minimal_client method."""
        mock_client = Mock()
        mock_venice_client_class.return_value = mock_client

        with patch.object(VeniceClientFactory, "create_client") as mock_create_client:
            mock_create_client.return_value = mock_client

            result = VeniceClientFactory.create_minimal_client(
                api_key="minimal-key", custom_option="value"
            )

            # Verify minimal config creation and client creation
            mock_create_client.assert_called_once()
            call_args = mock_create_client.call_args
            config_arg = call_args[0][0]  # First positional argument

            # Verify minimal configuration
            # The scheduler is inert outside ADAPTIVE mode, so the minimal config
            # keeps its defaults (anything else would warn when the client is built).
            assert config_arg.scheduler.mode == SchedulerMode.BASIC
            assert config_arg.scheduler.model_dump(exclude={"mode"}) == (
                SchedulerConfig().model_dump(exclude={"mode"})
            )
            assert config_arg.rate_limiter.mode == RateLimiterMode.SIMPLE
            assert config_arg.backend.backend_type == BackendType.MEMORY

            # Verify parameters
            call_kwargs = call_args[1]
            assert call_kwargs["api_key"] == "minimal-key"
            assert call_kwargs["custom_option"] == "value"

            assert result == mock_client


class TestVeniceClientFactoryConvenienceFunctions:
    """Test convenience functions for client creation."""

    @patch("venice_ai.factory.VeniceClientFactory.create_client")
    def test_create_venice_client_with_config(self, mock_create_client):
        """Test create_venice_client function with provided config."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        config = VeniceAIConfig.create_test_config()

        result = create_venice_client(
            api_key="convenience-key", config=config, account_id="convenience-account"
        )

        mock_create_client.assert_called_once_with(
            config, api_key="convenience-key", account_id="convenience-account"
        )
        assert result == mock_client

    @patch("venice_ai.factory.VeniceClientFactory.create_client")
    def test_create_venice_client_without_config(self, mock_create_client):
        """Test create_venice_client function without config (uses minimal)."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        result = create_venice_client(api_key="no-config-key")

        mock_create_client.assert_called_once()
        call_args = mock_create_client.call_args
        config_arg = call_args[0][0]

        # Should use minimal config
        assert config_arg.scheduler.mode == SchedulerMode.BASIC
        assert config_arg.backend.backend_type == BackendType.MEMORY

        call_kwargs = call_args[1]
        assert call_kwargs["api_key"] == "no-config-key"

        assert result == mock_client

    @patch("venice_ai.factory.VeniceClientFactory.create_test_client")
    def test_create_test_venice_client(self, mock_create_test_client):
        """Test create_test_venice_client convenience function."""
        mock_client = Mock()
        mock_create_test_client.return_value = mock_client

        result = create_test_venice_client(
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=True,
            test_rate_multiplier=8.0,
        )

        mock_create_test_client.assert_called_once_with(
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=True,
            test_rate_multiplier=8.0,
        )
        assert result == mock_client


class TestVeniceClientFactoryDependencyInjection:
    """Test the complete dependency injection flow."""

    @patch("venice_ai._client.VeniceClient")
    def test_full_dependency_injection_flow(
        self,
        mock_client_class,
    ):
        """Test the complete dependency injection flow."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VeniceAIConfig.create_test_config()

        result = VeniceClientFactory.create_client(
            config=config, api_key="full-test-key", account_id="full-test-account"
        )

        # Verify VeniceClient created with config
        mock_client_class.assert_called_once()
        client_call_kwargs = mock_client_class.call_args[1]
        assert client_call_kwargs["config"] == config
        assert client_call_kwargs["api_key"] == "full-test-key"

        assert result == mock_client


class TestVeniceClientFactoryConfigurationVariants:
    """Test different configuration scenarios."""

    def test_create_test_client_with_custom_kwargs(self):
        """Test create_test_client with additional kwargs."""
        with patch.object(VeniceClientFactory, "create_client") as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # Test with http_client in kwargs (should be handled specially)
            mock_http_client = Mock()
            VeniceClientFactory.create_test_client(
                scheduler_mode=SchedulerMode.ACCOUNT,
                enable_redis=False,
                test_rate_multiplier=3.0,
                http_client=mock_http_client,
                custom_param="test_value",
            )

            # Verify http_client was separated from other kwargs
            call_kwargs = mock_create_client.call_args[1]
            assert call_kwargs["http_client"] == mock_http_client
            assert call_kwargs["custom_param"] == "test_value"
            assert call_kwargs["api_key"] == "test-api-key"  # Default
            assert call_kwargs["account_id"] == "test-account"  # Default


class TestVeniceClientFactoryLogging:
    """Test logging behavior in factory methods."""

    def test_create_client_logging(self, caplog):
        """Test that client creation is properly logged."""
        config = VeniceAIConfig.create_test_config()

        with patch("venice_ai._client.VeniceClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with caplog.at_level("INFO"):
                VeniceClientFactory.create_client(config, account_id="logged-account")

            # Check for creation and success logs
            log_messages = [record.message for record in caplog.records]
            assert any(
                "Creating Venice client with config environment: test" in msg
                for msg in log_messages
            )
            assert any(
                "Venice client created successfully for account: logged-account" in msg
                for msg in log_messages
            )


class TestVeniceClientFactoryParameterHandling:
    """Test parameter handling and edge cases."""

    @patch("venice_ai._client.VeniceClient")
    def test_create_test_client_http_client_separation(self, mock_client_class):
        """Test that http_client is properly separated from other kwargs."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with patch.object(VeniceClientFactory, "create_client") as mock_create_client:
            mock_create_client.return_value = mock_client

            mock_http_client = Mock()

            VeniceClientFactory.create_test_client(
                http_client=mock_http_client, additional_param="test"
            )

            # Verify http_client was popped from kwargs before processing
            call_kwargs = mock_create_client.call_args[1]
            assert call_kwargs["http_client"] == mock_http_client
            assert call_kwargs["additional_param"] == "test"
            # Verify defaults were applied
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["account_id"] == "test-account"


class TestCreateDeveloperClient:
    """Tests for create_developer_client (H1)."""

    def test_returns_venice_client(self):
        client = create_developer_client(api_key="test-key")
        # Sanity: returns the real client class
        from venice_ai._client import VeniceClient

        assert isinstance(client, VeniceClient)

    def test_scheduler_is_basic(self):
        # BASIC scheduler avoids tier-discovery latency at construction time
        client = create_developer_client(api_key="test-key")
        assert client._config is not None
        assert client._config.scheduler.mode == SchedulerMode.BASIC

    def test_default_timeout_is_30_seconds(self):
        client = create_developer_client(api_key="test-key")
        assert client._config is not None
        assert client._config.http_client.timeout == 30.0

    def test_default_max_retries_is_one(self):
        # 1 = one retry; lower than create_development_config's 2 (fail loud)
        client = create_developer_client(api_key="test-key")
        assert client._config is not None
        assert client._config.http_client.max_retries == 1

    def test_timeout_override_honored(self):
        client = create_developer_client(api_key="test-key", timeout=15.0)
        assert client._config is not None
        assert client._config.http_client.timeout == 15.0

    def test_max_retries_override_honored(self):
        client = create_developer_client(api_key="test-key", max_retries=3)
        assert client._config is not None
        assert client._config.http_client.max_retries == 3

    def test_uses_memory_backend(self):
        # Memory backend → no Redis required for local dev
        client = create_developer_client(api_key="test-key")
        assert client._config is not None
        assert client._config.backend.backend_type == BackendType.MEMORY

    def test_debug_flag_enabled(self):
        # create_development_config sets debug=True
        client = create_developer_client(api_key="test-key")
        assert client._config is not None
        assert client._config.debug is True

    def test_factory_method_and_module_function_equivalent(self):
        # Module-level convenience function is a thin pass-through to the
        # factory method; both should produce equivalent clients.
        c1 = VeniceClientFactory.create_developer_client(api_key="test-key")
        c2 = create_developer_client(api_key="test-key")
        assert c1._config is not None
        assert c2._config is not None
        assert c1._config.scheduler.mode == c2._config.scheduler.mode
        assert c1._config.http_client.timeout == c2._config.http_client.timeout
        assert c1._config.http_client.max_retries == c2._config.http_client.max_retries

    def test_api_key_from_env_when_not_passed(self, monkeypatch):
        monkeypatch.setenv("VENICE_API_KEY", "env-resolved-key")
        client = create_developer_client()
        # VeniceClient strips and stashes the key on construction
        assert client._api_key == "env-resolved-key"


class TestConfigBuildersThroughFactory:
    """First-party config builders must not trip the factory's own warnings."""

    @pytest.mark.parametrize(
        "build",
        [
            VeniceAIConfig.create_test_config,
            VeniceAIConfig.create_minimal_config,
        ],
        ids=["create_test_config", "create_minimal_config"],
    )
    async def test_builder_default_through_create_client_is_silent(self, build):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = build()
            client = VeniceClientFactory.create_client(
                config, api_key="test-key", account_id="acct"
            )
            await client.close()

        assert [f"{w.category.__name__}: {w.message}" for w in caught] == []

    def test_create_test_config_redis_opt_in_is_honoured_and_reported(self):
        """Asking for Redis under the SIMPLE rate limiter still builds it, and says it is unused."""
        config = VeniceAIConfig.create_test_config(enable_redis=True, test_rate_multiplier=10.0)

        assert config.backend.backend_type == BackendType.REDIS
        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://localhost:6379/15"
        assert config.scheduler.test_rate_multiplier == 10.0

        with (
            patch("venice_ai._client.VeniceClient"),
            pytest.warns(UserWarning) as record,
        ):
            VeniceClientFactory.create_client(config, api_key="test-key")

        messages = [str(w.message) for w in record]
        assert any("backend_type=BackendType.REDIS has no effect" in m for m in messages)
        assert any("['test_rate_multiplier']" in m for m in messages)
