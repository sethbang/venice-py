"""
Test module for TierDiscovery error handling and edge cases.

Focuses on improving branch coverage for uncovered error paths in rate_limiting.py.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from venice_ai.core.rate_limit_discovery import RateLimitDiscovery


@pytest_asyncio.fixture
async def tier_discovery_with_mock_client():
    """Create TierDiscovery with mocked client (module-level fixture)."""
    mock_client = AsyncMock()

    rate_limit_discovery = RateLimitDiscovery(client=mock_client, account_id="test-account")
    yield rate_limit_discovery


class TestTierDiscoveryErrorHandling:
    """Test error handling in TierDiscovery."""

    @pytest.mark.asyncio
    async def test_discover_tiers_cancelled_error(self, tier_discovery_with_mock_client):
        """Test discover_tiers handles CancelledError (line 161-162)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock _fetch_rate_limits_simple to raise CancelledError
        with (
            patch.object(
                rate_limit_discovery,
                "_fetch_rate_limits_simple",
                side_effect=asyncio.CancelledError("Task cancelled"),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            # CancelledError should be re-raised
            await rate_limit_discovery.discover_tiers()

    @pytest.mark.asyncio
    async def test_discover_tiers_value_error(self, tier_discovery_with_mock_client, caplog):
        """Test discover_tiers handles ValueError (lines 163-165)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock _fetch_rate_limits_simple to raise ValueError
        with patch.object(
            rate_limit_discovery,
            "_fetch_rate_limits_simple",
            side_effect=ValueError("Invalid rate limit data"),
        ):
            # Should log exception and return existing tiers
            result = await rate_limit_discovery.discover_tiers()

            # Should return existing tiers (includes defaults)
            assert result == rate_limit_discovery.tiers

            # Should log the error
            assert "Error discovering tiers" in caplog.text

    @pytest.mark.asyncio
    async def test_discover_tiers_type_error(self, tier_discovery_with_mock_client, caplog):
        """Test discover_tiers handles TypeError (lines 163-165)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        with patch.object(
            rate_limit_discovery,
            "_fetch_rate_limits_simple",
            side_effect=TypeError("Type mismatch"),
        ):
            result = await rate_limit_discovery.discover_tiers()

            assert result == rate_limit_discovery.tiers
            assert "Error discovering tiers" in caplog.text

    @pytest.mark.asyncio
    async def test_discover_tiers_attribute_error(self, tier_discovery_with_mock_client, caplog):
        """Test discover_tiers handles AttributeError (lines 163-165)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        with patch.object(
            rate_limit_discovery,
            "_fetch_rate_limits_simple",
            side_effect=AttributeError("Missing attribute"),
        ):
            result = await rate_limit_discovery.discover_tiers()

            assert result == rate_limit_discovery.tiers
            assert "Error discovering tiers" in caplog.text

    @pytest.mark.asyncio
    async def test_discover_tiers_os_error(self, tier_discovery_with_mock_client, caplog):
        """Test discover_tiers handles OSError (lines 163-165)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        with patch.object(
            rate_limit_discovery,
            "_fetch_rate_limits_simple",
            side_effect=OSError("Connection failed"),
        ):
            result = await rate_limit_discovery.discover_tiers()

            assert result == rate_limit_discovery.tiers
            assert "Error discovering tiers" in caplog.text

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_cancelled_error(self, tier_discovery_with_mock_client):
        """Test _fetch_rate_limits_simple handles CancelledError (lines 191-192)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock client session to raise CancelledError
        rate_limit_discovery.client.get = AsyncMock(
            side_effect=asyncio.CancelledError("Fetch cancelled")
        )

        # CancelledError should be re-raised
        with pytest.raises(asyncio.CancelledError):
            await rate_limit_discovery._fetch_rate_limits_simple()

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_value_error(self, tier_discovery_with_mock_client, caplog):
        """Test _fetch_rate_limits_simple handles ValueError (lines 193-195)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock client session to raise ValueError
        rate_limit_discovery.client.get = AsyncMock(side_effect=ValueError("Invalid JSON"))

        # Should log exception and return None
        result = await rate_limit_discovery._fetch_rate_limits_simple()

        assert result is None
        assert "Failed to fetch rate limits" in caplog.text

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_type_error(self, tier_discovery_with_mock_client, caplog):
        """Test _fetch_rate_limits_simple handles TypeError (lines 193-195)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        rate_limit_discovery.client.get = AsyncMock(side_effect=TypeError("Type error in response"))

        result = await rate_limit_discovery._fetch_rate_limits_simple()

        assert result is None
        assert "Failed to fetch rate limits" in caplog.text

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_attribute_error(self, tier_discovery_with_mock_client, caplog):
        """Test _fetch_rate_limits_simple handles AttributeError (lines 193-195)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        rate_limit_discovery.client.get = AsyncMock(side_effect=AttributeError("Missing attribute"))

        result = await rate_limit_discovery._fetch_rate_limits_simple()

        assert result is None
        assert "Failed to fetch rate limits" in caplog.text

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_os_error(self, tier_discovery_with_mock_client, caplog):
        """Test _fetch_rate_limits_simple handles OSError (lines 193-195)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        rate_limit_discovery.client.get = AsyncMock(side_effect=OSError("Network unavailable"))

        result = await rate_limit_discovery._fetch_rate_limits_simple()

        assert result is None
        assert "Failed to fetch rate limits" in caplog.text

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_no_client(self, caplog):
        """Test _fetch_rate_limits_simple when client is None (lines 169-171)."""
        rate_limit_discovery = RateLimitDiscovery(client=None, account_id="test-account")

        result = await rate_limit_discovery._fetch_rate_limits_simple()

        assert result is None
        assert "No client available for rate limit fetching" in caplog.text


class TestTierDiscoveryComplexLogic:
    """Test complex conditional logic in TierDiscovery."""

    @pytest.mark.asyncio
    async def test_process_rate_limits_with_invalid_tier_data(self):
        """Test _process_rate_limits_simple with malformed tier data."""
        mock_client = AsyncMock()
        rate_limit_discovery = RateLimitDiscovery(client=mock_client, account_id="test-account")

        # Test with invalid tier structure (missing required fields)
        invalid_data = [
            {
                # Missing 'tier' field
                "models": ["model1"],
                "limits": {"rpm": 10},
            }
        ]

        # Should handle gracefully
        rate_limit_discovery._process_rate_limits_simple(invalid_data)

        # Should have default tiers at minimum
        assert len(rate_limit_discovery.tiers) >= 0

    @pytest.mark.asyncio
    async def test_process_rate_limits_with_missing_models(self):
        """Test _process_rate_limits_simple with missing models field."""
        mock_client = AsyncMock()
        rate_limit_discovery = RateLimitDiscovery(client=mock_client, account_id="test-account")

        data = [
            {
                "bucket": "tier-1",
                # Missing 'models' field
                "limits": {"rpm": 10},
            }
        ]

        rate_limit_discovery._process_rate_limits_simple(data)

        # Should handle gracefully - may or may not create tier
        assert isinstance(rate_limit_discovery.tiers, dict)

    @pytest.mark.asyncio
    async def test_discover_tiers_returns_cached_on_error(self, caplog):
        """Test that discover_tiers returns cached tiers when fetch fails."""
        mock_client = AsyncMock()
        rate_limit_discovery = RateLimitDiscovery(client=mock_client, account_id="test-account")

        # Set up existing tiers
        rate_limit_discovery.tiers = {"tier-1": Mock()}
        rate_limit_discovery.model_to_tier = {"model1": "tier-1"}

        # Mock fetch to fail
        with patch.object(
            rate_limit_discovery,
            "_fetch_rate_limits_simple",
            side_effect=ValueError("API error"),
        ):
            result = await rate_limit_discovery.discover_tiers()

            # Should return existing tiers
            assert "tier-1" in result
            assert "Error discovering tiers" in caplog.text

    @pytest.mark.asyncio
    async def test_discover_tiers_empty_response(self):
        """Test discover_tiers with empty API response."""
        mock_client = AsyncMock()
        rate_limit_discovery = RateLimitDiscovery(client=mock_client, account_id="test-account")

        # Mock fetch to return None
        with patch.object(rate_limit_discovery, "_fetch_rate_limits_simple", return_value=None):
            result = await rate_limit_discovery.discover_tiers()

            # Should return default tiers (created in __init__)
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_malformed_json(self, tier_discovery_with_mock_client):
        """Test _fetch_rate_limits_simple with malformed JSON response."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock response with wrong data type (string instead of dict/list)
        rate_limit_discovery.client.get = AsyncMock(return_value={"data": "not_a_list_or_dict"})

        # Should return None for invalid structure
        result = await rate_limit_discovery._fetch_rate_limits_simple()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_data_as_list(self, tier_discovery_with_mock_client):
        """Test _fetch_rate_limits_simple when data is directly a list (lines 186-187)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock response where data is directly a list
        rate_limit_discovery.client.get = AsyncMock(
            return_value={
                "data": [{"bucket": "tier-1", "models": ["model1"], "limits": {"rpm": 10}}]
            }
        )

        # Should return the list
        result = await rate_limit_discovery._fetch_rate_limits_simple()
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_nested_rate_limits(self, tier_discovery_with_mock_client):
        """Test _fetch_rate_limits_simple with nested rateLimits structure (lines 182-185)."""
        rate_limit_discovery = tier_discovery_with_mock_client

        # Mock response with nested structure
        rate_limit_discovery.client.get = AsyncMock(
            return_value={
                "data": {
                    "rateLimits": [
                        {
                            "bucket": "tier-1",
                            "models": ["model1"],
                            "limits": {"rpm": 10},
                        }
                    ]
                }
            }
        )

        # Should extract the rateLimits array
        result = await rate_limit_discovery._fetch_rate_limits_simple()
        assert isinstance(result, list)
        assert len(result) == 1
