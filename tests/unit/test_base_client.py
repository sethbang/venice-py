"""
Test coverage for VeniceClient initialization and configuration.
Tests the client initialization logic.
"""

import os
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from venice_ai._client import VeniceClient
from venice_ai.middleware import RetryOptions


class TestVeniceClientInitialization:
    """Test VeniceClient initialization with various configurations."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = VeniceClient(api_key="test-api-key")
        assert client._api_key == "test-api-key"

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"VENICE_API_KEY": "env-api-key"}):
            client = VeniceClient()
            assert client._api_key == "env-api-key"

    def test_init_with_whitespace_api_key(self):
        """Test that API key whitespace is stripped."""
        client = VeniceClient(api_key="  test-api-key  ")
        assert client._api_key == "test-api-key"

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="No authentication provided"),
        ):
            VeniceClient()

    def test_init_empty_api_key(self):
        """Test initialization fails with empty API key."""
        with pytest.raises(ValueError, match="No authentication provided"):
            VeniceClient(api_key="")

    def test_init_whitespace_only_api_key(self):
        """Test initialization fails with whitespace-only API key."""
        with pytest.raises(ValueError, match="No authentication provided"):
            VeniceClient(api_key="   ")

    def test_init_none_api_key(self):
        """Test initialization with None API key falls back to env."""
        with patch.dict(os.environ, {"VENICE_API_KEY": "env-key"}):
            client = VeniceClient(api_key=None)
            assert client._api_key == "env-key"

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        client = VeniceClient(api_key="test", base_url="https://custom.api.com/v2/")
        assert str(client._base_url) == "https://custom.api.com/v2/"

    def test_init_with_bare_host_base_url_gets_the_api_path(self):
        client = VeniceClient(api_key="test", base_url="https://custom.api.com")
        assert str(client._base_url) == "https://custom.api.com/api/v1/"

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        client = VeniceClient(api_key="test", timeout=120.0)
        # VeniceClient converts float timeout to ClientTimeout object
        assert client._timeout.total == 120.0
        assert isinstance(client._timeout, aiohttp.ClientTimeout)

    def test_init_with_retry_options(self):
        """Test initialization with retry options."""
        retry_options = RetryOptions(max_attempts=5)
        client = VeniceClient(api_key="test", retry_options=retry_options)
        assert client._retry_options == retry_options

    def test_init_with_rate_limiter_config_path(self):
        """Test initialization with rate limiter config path."""
        client = VeniceClient(api_key="test", rate_limiter_config_path="config.yaml")
        assert client._rate_limiter_config_path == "config.yaml"

    def test_init_with_rate_limiter_config_dict(self):
        """Test initialization with rate limiter config dict."""
        config = {"requests_per_second": 10}
        client = VeniceClient(api_key="test", rate_limiter_config=config)
        assert client._rate_limiter_config == config

    def test_init_with_both_rate_limiter_configs_raises_error(self):
        """Test initialization with both rate limiter configs raises error."""
        config = {"requests_per_second": 10}
        with pytest.raises(ValueError, match="Cannot provide both"):
            VeniceClient(
                api_key="test",
                rate_limiter_config=config,
                rate_limiter_config_path="config.yaml",
            )

    def test_init_with_http_client(self):
        """Test initialization with external HTTP client."""
        mock_client = MagicMock(spec=aiohttp.ClientSession)
        client = VeniceClient(api_key="test", http_client=mock_client)
        assert hasattr(client, "_base_url")

    def test_init_with_http_client_and_rate_limiter_config(self):
        """Test initialization with HTTP client and rate limiter config."""
        mock_client = MagicMock(spec=aiohttp.ClientSession)
        config = {"requests_per_second": 10}
        client = VeniceClient(api_key="test", http_client=mock_client, rate_limiter_config=config)
        # With http_client, rate_limiter_config is set to None unless explicitly disabled
        assert client._rate_limiter_config is None
        assert client._rate_limiter_config_path is None
        assert hasattr(client, "_base_url")

    def test_init_with_http_client_and_disabled_rate_limiter(self):
        """Test initialization with HTTP client and disabled rate limiter."""
        mock_client = MagicMock(spec=aiohttp.ClientSession)
        config = {"enabled": False}
        client = VeniceClient(api_key="test", http_client=mock_client, rate_limiter_config=config)
        assert hasattr(client, "_base_url")
        assert client._rate_limiter_config == config

    @pytest.mark.asyncio
    async def test_init_with_cookie_jar(self):
        """Test initialization with cookie jar."""
        # Create cookie jar in async context
        jar = aiohttp.CookieJar()
        client = VeniceClient(api_key="test", cookie_jar=jar)
        assert client._cookie_jar == jar

    def test_init_with_headers(self):
        """Test initialization with custom headers."""
        headers = {"X-Custom": "value"}
        client = VeniceClient(api_key="test", headers=headers)
        assert client._headers == headers

    def test_init_with_skip_auto_headers(self):
        """Test initialization with skip_auto_headers."""
        client = VeniceClient(api_key="test", skip_auto_headers=["User-Agent"])
        assert client._skip_auto_headers == ["User-Agent"]

    def test_init_with_auto_decompress(self):
        """Test initialization with auto_decompress."""
        client = VeniceClient(api_key="test", auto_decompress=False)
        assert client._auto_decompress is False


class TestShouldUseRateLimiter:
    """Test _should_use_rate_limiter method."""

    def test_rate_limiter_config_enabled(self):
        """Test rate limiter when config has enabled=True."""
        config = {"enabled": True}
        client = VeniceClient(api_key="test", rate_limiter_config=config)
        assert client._should_use_rate_limiter() is True

    def test_rate_limiter_config_disabled(self):
        """Test rate limiter when config has enabled=False."""
        config = {"enabled": False}
        client = VeniceClient(api_key="test", rate_limiter_config=config)
        assert client._should_use_rate_limiter() is False

    def test_rate_limiter_disabled_with_external_http_client(self):
        """Test rate limiter disabled with external HTTP client."""
        mock_client = MagicMock(spec=aiohttp.ClientSession)
        client = VeniceClient(api_key="test", http_client=mock_client)
        assert client._should_use_rate_limiter() is False

    def test_rate_limiter_enabled_by_env_var(self):
        """Test rate limiter enabled by environment variable."""
        with patch.dict(os.environ, {"VENICE_RATE_LIMITER_FEATURES_ENABLED": "true"}):
            client = VeniceClient(api_key="test")
            assert client._should_use_rate_limiter() is True

    def test_rate_limiter_disabled_by_env_var(self):
        """Test rate limiter disabled by environment variable."""
        with patch.dict(os.environ, {"VENICE_RATE_LIMITER_FEATURES_ENABLED": "false"}):
            client = VeniceClient(api_key="test")
            assert client._should_use_rate_limiter() is False

    def test_rate_limiter_default_disabled(self):
        """Test rate limiter default state (disabled)."""
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(api_key="test")
            # Default should be False (v2.0.0: opt-in)
            assert client._should_use_rate_limiter() is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_base_url_trailing_slash_handling(self):
        """Test that base URL trailing slash is handled correctly."""
        client1 = VeniceClient(api_key="test", base_url="https://api.example.com")
        client2 = VeniceClient(api_key="test", base_url="https://api.example.com/")

        # Both should result in the same URL with trailing slash
        assert str(client1._base_url) == str(client2._base_url)
        assert str(client1._base_url).endswith("/")

    def test_invalid_timeout_type(self):
        """Test that invalid timeout type uses default."""
        client = VeniceClient(api_key="test", timeout=None)
        # Should use default timeout
        assert client._timeout is not None

    def test_rate_limiter_config_path_not_exists(self):
        """Test rate limiter config path that doesn't exist."""
        client = VeniceClient(api_key="test", rate_limiter_config_path="nonexistent.yaml")
        # Should handle gracefully
        assert client._rate_limiter_config_path == "nonexistent.yaml"
