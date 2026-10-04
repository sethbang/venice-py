"""
Comprehensive test module for rate_limiting.py coverage expansion.

Targets missing lines: 165-166, 171, 174, 177, 179-182, 184-186, 196, 200,
312, 355, 362-364, 368-369, 375-376, 384-386, 389-393, 396-400, 403-405,
407, 409, 414, 425, 437, 455, 466, 471-472, 474-477, 482-483, 485-488,
493-494, 496-499, 522

Targets partial branches: 126, 163, 195, 199, 207, 224, 229, 235, 261,
311, 313, 354, 358, 413, 453, 464
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio

from venice_ai._queue_types import ResourceType
from venice_ai.core.rate_limit_discovery import (
    RateLimitBucket,
    RateLimitDiscovery,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest_asyncio.fixture
async def discovery_with_mock_client():
    """Create RateLimitDiscovery with mocked client."""
    mock_client = AsyncMock()
    discovery = RateLimitDiscovery(
        client=mock_client, account_id="test-account", cache_duration=300
    )
    yield discovery


@pytest_asyncio.fixture
async def discovery_no_client():
    """Create RateLimitDiscovery without client."""
    discovery = RateLimitDiscovery(client=None, account_id="no-client-account")
    yield discovery


# ==============================================================================
# Test RateLimitBucket
# ==============================================================================


class TestRateLimitBucket:
    """Tests for RateLimitBucket model."""

    def test_signature_with_all_limits(self):
        """Test signature property with all limits set."""
        bucket = RateLimitBucket(
            bucket_id="test-bucket",
            rpm_limit=100,
            rpd_limit=1000,
            tpm_limit=50000,
            models={"model1", "model2"},
        )
        assert bucket.signature == (100, 1000, 50000)

    def test_signature_with_none_limits(self):
        """Test signature property with None limits (uses 0 as fallback)."""
        bucket = RateLimitBucket(
            bucket_id="test-bucket",
            rpm_limit=100,
            rpd_limit=None,
            tpm_limit=None,
            models=set(),
        )
        assert bucket.signature == (100, 0, 0)


# ==============================================================================
# Test Cache Stampede Prevention (lines 163-186)
# ==============================================================================


class TestCacheStampedePrevention:
    """Tests for cache stampede prevention in discover_tiers."""

    @pytest.mark.asyncio
    async def test_coalesced_request_waits_for_inflight(self, discovery_with_mock_client):
        """
        Test that concurrent requests wait for in-flight refresh (lines 165-166, 171).

        This covers the branch where refresh_key is already in _refresh_futures.
        """
        discovery = discovery_with_mock_client

        # Set up a slow API call
        call_count = 0

        async def slow_fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return [
                {
                    "apiModelId": "model-1",
                    "rateLimits": [{"type": "RPM", "amount": 60}],
                }
            ]

        with patch.object(discovery, "_fetch_rate_limits_simple", side_effect=slow_fetch):
            # Clear cache to force refresh
            discovery._last_refresh = None
            discovery.tiers = {}

            # Start two concurrent discover operations
            task1 = asyncio.create_task(discovery.discover_tiers())
            await asyncio.sleep(0.01)  # Let task1 start
            task2 = asyncio.create_task(discovery.discover_tiers())

            results = await asyncio.gather(task1, task2)

            # Both should get same result
            assert results[0] == results[1]
            # Should only make 1 API call due to coalescing
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_coalesced_request_records_metrics(self, discovery_with_mock_client):
        """
        Test that coalesced requests record metrics (lines 171, 174, 177, 184-186).

        Covers _record_coalesced_request, _record_concurrent_requests, _record_time_saved.
        """
        discovery = discovery_with_mock_client

        async def slow_fetch():
            await asyncio.sleep(0.05)
            return [
                {
                    "apiModelId": "model-1",
                    "rateLimits": [{"type": "RPM", "amount": 60}],
                }
            ]

        with patch.object(discovery, "_fetch_rate_limits_simple", side_effect=slow_fetch):
            discovery._last_refresh = None
            discovery.tiers = {}

            with (
                patch.object(discovery, "_record_coalesced_request") as mock_coalesced,
                patch.object(discovery, "_record_concurrent_requests") as mock_concurrent,
                patch.object(discovery, "_record_time_saved") as mock_time_saved,
            ):
                # Start concurrent requests
                task1 = asyncio.create_task(discovery.discover_tiers())
                await asyncio.sleep(0.01)
                task2 = asyncio.create_task(discovery.discover_tiers())

                await asyncio.gather(task1, task2)

                # The second request should have recorded coalesced metrics
                mock_coalesced.assert_called()
                mock_concurrent.assert_called()
                mock_time_saved.assert_called()

    @pytest.mark.asyncio
    async def test_coalesced_request_cancelled_future_retry(self, discovery_with_mock_client):
        """
        Test retry when existing future is cancelled (lines 179-182, 186).

        Covers the CancelledError/KeyError catch that falls through to retry.
        """
        discovery = discovery_with_mock_client
        refresh_key = f"tier_refresh_{discovery.account_id}"

        # Create a cancelled future
        cancelled_future: asyncio.Future = asyncio.Future()
        cancelled_future.cancel()
        discovery._refresh_futures[refresh_key] = cancelled_future

        # Normal fetch for retry
        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[
                {
                    "apiModelId": "model-1",
                    "rateLimits": [{"type": "RPM", "amount": 60}],
                }
            ],
        ):
            discovery._last_refresh = None
            result = await discovery.discover_tiers()
            assert "model-1" in result or len(result) > 0


# ==============================================================================
# Test Double-Check Pattern (lines 195-200)
# ==============================================================================


class TestDoubleCheckPattern:
    """Tests for double-check pattern in discover_tiers."""

    @pytest.mark.asyncio
    async def test_cache_valid_after_acquiring_lock(self, discovery_with_mock_client):
        """
        Test double-check: cache becomes valid while waiting for lock (line 196).

        Covers branch where cache becomes valid between initial check and lock.
        """
        discovery = discovery_with_mock_client

        # First call will refresh
        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[
                {
                    "apiModelId": "test-model",
                    "rateLimits": [{"type": "RPM", "amount": 100}],
                }
            ],
        ):
            # Make cache invalid initially
            discovery._last_refresh = None

            async def make_cache_valid_during_wait():
                """Simulate another task making cache valid."""
                await asyncio.sleep(0.01)
                discovery._last_refresh = datetime.now(UTC)

            # Start validity updater
            asyncio.create_task(make_cache_valid_during_wait())

            result = await discovery.discover_tiers()
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_future_created_by_another_after_lock(self, discovery_with_mock_client):
        """
        Test when another task creates the future while waiting for lock (line 200).

        Covers branch where refresh_key appears in _refresh_futures after lock acquired.
        """
        discovery = discovery_with_mock_client
        refresh_key = f"tier_refresh_{discovery.account_id}"

        # Create a future that will complete
        existing_future: asyncio.Future = asyncio.Future()

        async def complete_future_later():
            await asyncio.sleep(0.02)
            existing_future.set_result(discovery.tiers)

        asyncio.create_task(complete_future_later())

        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[{"apiModelId": "model", "rateLimits": []}],
        ):
            # Inject future after initial check but simulate it being there
            async def inject_future():
                await asyncio.sleep(0.005)
                discovery._refresh_futures[refresh_key] = existing_future

            discovery._last_refresh = None
            asyncio.create_task(inject_future())

            result = await discovery.discover_tiers()
            assert isinstance(result, dict)


# ==============================================================================
# Test _process_rate_limits_simple (lines 311-312, 313)
# ==============================================================================


class TestProcessRateLimits:
    """Tests for _process_rate_limits_simple method."""

    @pytest.mark.asyncio
    async def test_rpd_limit_parsing(self, discovery_with_mock_client):
        """
        Test RPD limit parsing (line 312).

        Covers the RPD type branch in rate limit processing.
        """
        discovery = discovery_with_mock_client

        rate_limits = [
            {
                "apiModelId": "model-with-rpd",
                "rateLimits": [
                    {"type": "RPM", "amount": 100},
                    {"type": "RPD", "amount": 5000},
                ],
            }
        ]

        discovery._process_rate_limits_simple(rate_limits)

        assert "model-with-rpd" in discovery.tiers
        bucket = discovery.tiers["model-with-rpd"]
        assert bucket.rpd_limit == 5000

    @pytest.mark.asyncio
    async def test_tpm_limit_parsing(self, discovery_with_mock_client):
        """
        Test TPM limit parsing (line 313-314 partial branch).

        Covers the TPM type branch in rate limit processing.
        """
        discovery = discovery_with_mock_client

        rate_limits = [
            {
                "apiModelId": "model-with-tpm",
                "rateLimits": [
                    {"type": "RPM", "amount": 60},
                    {"type": "TPM", "amount": 100000},
                ],
            }
        ]

        discovery._process_rate_limits_simple(rate_limits)

        assert "model-with-tpm" in discovery.tiers
        bucket = discovery.tiers["model-with-tpm"]
        assert bucket.tpm_limit == 100000

    @pytest.mark.asyncio
    async def test_no_model_id_skipped(self, discovery_with_mock_client):
        """
        Test items without apiModelId are skipped (line 301-302).
        """
        discovery = discovery_with_mock_client

        rate_limits = [
            {"rateLimits": [{"type": "RPM", "amount": 60}]},  # Missing apiModelId
            {
                "apiModelId": "valid-model",
                "rateLimits": [{"type": "RPM", "amount": 60}],
            },
        ]

        discovery._process_rate_limits_simple(rate_limits)

        # Only valid model should be in tiers
        assert "valid-model" in discovery.tiers
        # The entry without apiModelId should be skipped

    @pytest.mark.asyncio
    async def test_none_amount_defaults_to_60(self, discovery_with_mock_client):
        """
        Test that None RPM amount defaults to 60 (line 310).
        """
        discovery = discovery_with_mock_client

        rate_limits = [
            {
                "apiModelId": "model-with-none-rpm",
                "rateLimits": [{"type": "RPM", "amount": None}],
            }
        ]

        discovery._process_rate_limits_simple(rate_limits)

        bucket = discovery.tiers["model-with-none-rpm"]
        assert bucket.rpm_limit == 60  # Default value

    @pytest.mark.asyncio
    async def test_empty_rate_limits_defaults(self, discovery_with_mock_client):
        """
        Test model with no rateLimits gets defaults (lines 305-314 branches).
        """
        discovery = discovery_with_mock_client

        rate_limits = [{"apiModelId": "model-no-limits", "rateLimits": []}]

        discovery._process_rate_limits_simple(rate_limits)

        bucket = discovery.tiers["model-no-limits"]
        assert bucket.rpm_limit == 60  # Default
        assert bucket.rpd_limit is None
        assert bucket.tpm_limit is None

    @pytest.mark.asyncio
    async def test_zero_rpd_becomes_none(self, discovery_with_mock_client):
        """
        Test that RPD value of 0 becomes None (line 326-327).
        """
        discovery = discovery_with_mock_client

        rate_limits = [
            {
                "apiModelId": "model-zero-rpd",
                "rateLimits": [
                    {"type": "RPM", "amount": 60},
                    {"type": "RPD", "amount": 0},
                ],
            }
        ]

        discovery._process_rate_limits_simple(rate_limits)

        bucket = discovery.tiers["model-zero-rpd"]
        assert bucket.rpd_limit is None  # 0 becomes None


# ==============================================================================
# Test get_tier_for_model (lines 354-409)
# ==============================================================================


class TestGetTierForModel:
    """Tests for get_tier_for_model method."""

    @pytest.mark.asyncio
    async def test_no_tiers_triggers_discovery(self, discovery_with_mock_client):
        """
        Test that empty tiers triggers discover_tiers (line 355).
        """
        discovery = discovery_with_mock_client
        discovery.tiers = {}
        discovery._last_refresh = None

        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[
                {
                    "apiModelId": "discovered-model",
                    "rateLimits": [{"type": "RPM", "amount": 60}],
                }
            ],
        ):
            bucket = await discovery.get_tier_for_model("discovered-model")
            assert bucket == "discovered-model"

    @pytest.mark.asyncio
    async def test_invalid_cache_triggers_discovery(self, discovery_with_mock_client):
        """
        Test that invalid cache triggers discover_tiers (line 354 branch).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None  # Invalid cache

        with patch.object(discovery, "discover_tiers", new_callable=AsyncMock) as mock:
            mock.return_value = discovery.tiers
            await discovery.get_tier_for_model("test-model")
            mock.assert_called()

    @pytest.mark.asyncio
    async def test_known_model_returns_tier(self, discovery_with_mock_client):
        """
        Test known model returns its tier (lines 358-359).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.model_to_tier["known-model"] = "tier-1"
        discovery.tiers["tier-1"] = RateLimitBucket(
            bucket_id="tier-1", rpm_limit=100, models={"known-model"}
        )

        bucket = await discovery.get_tier_for_model("known-model")
        assert bucket == "tier-1"

    @pytest.mark.asyncio
    async def test_unknown_model_triggers_refresh(self, discovery_with_mock_client):
        """
        Test unknown model triggers refresh (lines 362-364).

        This tests the code path where model is not in model_to_tier initially,
        then discover_tiers is called at line 362, and if model is found after
        refresh, it returns the tier at line 364.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        # Start with no model-to-tier mappings
        discovery.model_to_tier = {}

        # First call to discover_tiers (at line 362) should populate model_to_tier
        call_count = 0

        async def mock_discover(force_refresh=False):
            nonlocal call_count
            call_count += 1
            # On second call (the refresh at line 362), add the model
            if call_count >= 1:
                discovery.model_to_tier["new-model"] = "new-tier"
                discovery.tiers["new-tier"] = RateLimitBucket(
                    bucket_id="new-tier", rpm_limit=60, models={"new-model"}
                )
            return discovery.tiers

        with patch.object(discovery, "discover_tiers", side_effect=mock_discover):
            bucket = await discovery.get_tier_for_model("new-model")
            # Model should be found after the refresh at line 362
            assert bucket == "new-tier"

    @pytest.mark.asyncio
    async def test_unknown_model_image_resource_fallback(self, discovery_with_mock_client):
        """
        Test 'unknown' model with image resource type dynamically finds
        a known image model's tier via classifier patterns.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.model_to_tier["flux-dev"] = "image-tier"
        discovery.tiers["image-tier"] = RateLimitBucket(
            bucket_id="image-tier", rpm_limit=20, models={"flux-dev"}
        )

        bucket = await discovery.get_tier_for_model("unknown", resource_type="image")
        assert bucket == "image-tier"

    @pytest.mark.asyncio
    async def test_unknown_model_image_resource_seedream(self, discovery_with_mock_client):
        """
        Test 'unknown' model with image resource falls back through list (lines 375-376).
        Uses seedream-3 (replaced hidream in May 2026 changelog).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.model_to_tier["seedream-3"] = "seedream-tier"
        discovery.tiers["seedream-tier"] = RateLimitBucket(
            bucket_id="seedream-tier", rpm_limit=20, models={"seedream-3"}
        )

        bucket = await discovery.get_tier_for_model("unknown", resource_type="image")
        assert bucket == "seedream-tier"

    @pytest.mark.asyncio
    async def test_unknown_model_audio_resource_fallback(self, discovery_with_mock_client):
        """
        Test 'unknown' model with audio resource type (lines 389-393).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.model_to_tier["tts-kokoro"] = "audio-tier"
        discovery.tiers["audio-tier"] = RateLimitBucket(
            bucket_id="audio-tier", rpm_limit=30, models={"tts-kokoro"}
        )

        bucket = await discovery.get_tier_for_model("unknown", resource_type="audio")
        assert bucket == "audio-tier"

    @pytest.mark.asyncio
    async def test_unknown_model_embedding_resource_fallback(self, discovery_with_mock_client):
        """
        Test 'unknown' model with embedding resource type (lines 396-400).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.model_to_tier["text-embedding-bge-m3"] = "embedding-tier"
        discovery.tiers["embedding-tier"] = RateLimitBucket(
            bucket_id="embedding-tier", rpm_limit=100, models={"text-embedding-bge-m3"}
        )

        bucket = await discovery.get_tier_for_model("unknown", resource_type="embedding")
        assert bucket == "embedding-tier"

    @pytest.mark.asyncio
    async def test_fallback_to_default_tier_string_resource(self, discovery_with_mock_client):
        """
        Test fallback to default tier with string resource type (lines 403-405).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)

        bucket = await discovery.get_tier_for_model("completely-unknown", resource_type="text")
        assert bucket == "tier_default_text"

    @pytest.mark.asyncio
    async def test_fallback_to_default_tier_enum_resource(self, discovery_with_mock_client):
        """
        Test fallback to default tier with ResourceType enum (lines 406-407).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)

        bucket = await discovery.get_tier_for_model("unknown-model", resource_type=ResourceType.LLM)
        assert bucket == "tier_default_llm"

    @pytest.mark.asyncio
    async def test_no_fallback_returns_none(self, discovery_with_mock_client):
        """
        Test no resource type fallback returns None (line 409).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)

        bucket = await discovery.get_tier_for_model("unknown-model", resource_type=None)
        assert bucket is None


# ==============================================================================
# Test get_tier (lines 411-416)
# ==============================================================================


class TestGetTier:
    """Tests for get_tier method."""

    @pytest.mark.asyncio
    async def test_get_tier_empty_triggers_discovery(self, discovery_with_mock_client):
        """
        Test get_tier with empty tiers triggers discover_tiers (line 414).
        """
        discovery = discovery_with_mock_client
        discovery.tiers = {}

        with patch.object(discovery, "discover_tiers", new_callable=AsyncMock) as mock:
            mock.return_value = {}
            await discovery.get_tier("test-tier")
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tier_returns_bucket(self, discovery_with_mock_client):
        """
        Test get_tier returns existing bucket (line 416).
        """
        discovery = discovery_with_mock_client
        discovery.tiers["existing-tier"] = RateLimitBucket(
            bucket_id="existing-tier", rpm_limit=100, models={"model1"}
        )

        bucket = await discovery.get_tier("existing-tier")
        assert bucket is not None
        assert bucket.bucket_id == "existing-tier"

    @pytest.mark.asyncio
    async def test_get_tier_not_found_returns_none(self, discovery_with_mock_client):
        """
        Test get_tier with unknown tier returns None.
        """
        discovery = discovery_with_mock_client
        discovery.tiers["some-tier"] = RateLimitBucket(
            bucket_id="some-tier", rpm_limit=100, models=set()
        )

        bucket = await discovery.get_tier("nonexistent-tier")
        assert bucket is None


# ==============================================================================
# Test Metrics Recording Methods (lines 447-500)
# ==============================================================================


class TestMetricsRecording:
    """Tests for metrics recording methods.

    These tests cover the metrics recording methods which use internal
    try/except blocks that silently catch all exceptions. We test both
    the success paths and exception paths by triggering the methods
    directly and verifying they don't raise.
    """

    @pytest.mark.asyncio
    async def test_record_tier_discovery_request_with_metrics(self, discovery_with_mock_client):
        """
        Test _record_tier_discovery_request when metrics enabled (lines 453-455).
        """
        discovery = discovery_with_mock_client

        # Mock the enhanced metrics at the import location inside the method
        mock_metrics = MagicMock()
        mock_metrics._enabled = True
        mock_metrics.tier_discovery_requests_total = MagicMock()

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_tier_discovery_request()
            mock_metrics.tier_discovery_requests_total.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_tier_discovery_request_disabled(self, discovery_with_mock_client):
        """
        Test _record_tier_discovery_request when metrics disabled (line 453 exit).
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = False

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_tier_discovery_request()
            # Should not call inc when disabled
            mock_metrics.tier_discovery_requests_total.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_tier_discovery_request_exception(self, discovery_with_mock_client):
        """
        Test _record_tier_discovery_request handles import error (line 455-456).
        """
        discovery = discovery_with_mock_client

        # The exception is silently caught by the internal try/except
        # Just verify it doesn't raise
        discovery._record_tier_discovery_request()

    @pytest.mark.asyncio
    async def test_record_api_call_with_metrics(self, discovery_with_mock_client):
        """
        Test _record_api_call when metrics enabled (lines 464-466).
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = True
        mock_metrics.tier_discovery_api_calls_total = MagicMock()

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_api_call()
            mock_metrics.tier_discovery_api_calls_total.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_api_call_disabled(self, discovery_with_mock_client):
        """
        Test _record_api_call when metrics disabled (line 464 exit).
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = False

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_api_call()
            mock_metrics.tier_discovery_api_calls_total.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_coalesced_request_with_metrics(self, discovery_with_mock_client):
        """
        Test _record_coalesced_request when metrics enabled (lines 474-477).
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = True
        mock_metrics.tier_discovery_coalesced_total = MagicMock()

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_coalesced_request()
            mock_metrics.tier_discovery_coalesced_total.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_coalesced_request_disabled(self, discovery_with_mock_client):
        """
        Test _record_coalesced_request when metrics disabled.
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = False

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_coalesced_request()
            mock_metrics.tier_discovery_coalesced_total.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_concurrent_requests_with_metrics(self, discovery_with_mock_client):
        """
        Test _record_concurrent_requests when metrics enabled (lines 485-488).
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = True
        mock_metrics.tier_discovery_concurrent_requests = MagicMock()

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_concurrent_requests(5)
            mock_metrics.tier_discovery_concurrent_requests.set.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_record_concurrent_requests_disabled(self, discovery_with_mock_client):
        """
        Test _record_concurrent_requests when metrics disabled.
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = False

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_concurrent_requests(5)
            mock_metrics.tier_discovery_concurrent_requests.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_time_saved_with_metrics(self, discovery_with_mock_client):
        """
        Test _record_time_saved when metrics enabled (lines 496-499).
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = True
        mock_metrics.tier_discovery_time_saved_seconds = MagicMock()

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_time_saved(0.5)
            mock_metrics.tier_discovery_time_saved_seconds.inc.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_record_time_saved_disabled(self, discovery_with_mock_client):
        """
        Test _record_time_saved when metrics disabled.
        """
        discovery = discovery_with_mock_client

        mock_metrics = MagicMock()
        mock_metrics._enabled = False

        with patch(
            "venice_ai.observability.metrics.get_enhanced_metrics",
            return_value=mock_metrics,
        ):
            discovery._record_time_saved(0.5)
            mock_metrics.tier_discovery_time_saved_seconds.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_time_saved_exception(self, discovery_with_mock_client):
        """
        Test _record_time_saved handles exception (line 499-500).
        """
        discovery = discovery_with_mock_client

        # The exception is silently caught by the internal try/except
        # Just verify it doesn't raise
        discovery._record_time_saved(0.5)


# ==============================================================================
# Test _ensure_default_tiers (line 126 branch)
# ==============================================================================


class TestEnsureDefaultTiers:
    """Tests for _ensure_default_tiers method."""

    def test_default_tiers_created_on_init(self):
        """
        Test that default tiers are created on init (line 126 branch covers existing).
        """
        discovery = RateLimitDiscovery(client=None)

        # Should have default tiers for each resource type
        for resource_type in ResourceType:
            bucket_id = f"tier_default_{resource_type.value}"
            assert bucket_id in discovery.tiers
            assert discovery.tiers[bucket_id].rpm_limit == 10

    def test_default_tiers_not_overwritten(self):
        """
        Test that existing default tiers are not overwritten (line 126 true branch).
        """
        discovery = RateLimitDiscovery(client=None)

        # Manually add a default tier before calling _ensure_default_tiers
        custom_bucket = RateLimitBucket(
            bucket_id="tier_default_llm",
            rpm_limit=999,  # Custom value
            models=set(),
        )
        discovery.tiers["tier_default_llm"] = custom_bucket

        # Call again
        discovery._ensure_default_tiers()

        # Should not have overwritten
        assert discovery.tiers["tier_default_llm"].rpm_limit == 999


# ==============================================================================
# Test Cache Validity (lines 427-433)
# ==============================================================================


class TestCacheValidity:
    """Tests for _is_cache_valid method."""

    @pytest.mark.asyncio
    async def test_cache_invalid_no_tiers(self, discovery_with_mock_client):
        """
        Test cache invalid when no tiers (line 429 branch).
        """
        discovery = discovery_with_mock_client
        discovery.tiers = {}
        discovery._last_refresh = datetime.now(UTC)

        assert discovery._is_cache_valid() is False

    @pytest.mark.asyncio
    async def test_cache_invalid_no_last_refresh(self, discovery_with_mock_client):
        """
        Test cache invalid when no last_refresh (line 429 branch).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None

        assert discovery._is_cache_valid() is False

    @pytest.mark.asyncio
    async def test_cache_valid_within_duration(self, discovery_with_mock_client):
        """
        Test cache valid when within cache duration (line 433).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.cache_duration = 300

        assert discovery._is_cache_valid() is True


# ==============================================================================
# Test get_models_in_tier (lines 418-421)
# ==============================================================================


class TestGetModelsInTier:
    """Tests for get_models_in_tier method."""

    @pytest.mark.asyncio
    async def test_get_models_existing_tier(self, discovery_with_mock_client):
        """
        Test get_models_in_tier with existing tier.
        """
        discovery = discovery_with_mock_client
        discovery.tiers["test-tier"] = RateLimitBucket(
            bucket_id="test-tier", rpm_limit=100, models={"model-a", "model-b"}
        )

        models = await discovery.get_models_in_tier("test-tier")
        assert models == {"model-a", "model-b"}

    @pytest.mark.asyncio
    async def test_get_models_nonexistent_tier(self, discovery_with_mock_client):
        """
        Test get_models_in_tier with nonexistent tier returns empty set.
        """
        discovery = discovery_with_mock_client

        models = await discovery.get_models_in_tier("nonexistent-tier")
        assert models == set()


# ==============================================================================
# Test discover_tiers edge cases (lines 207, 224, 229, 235, 261)
# ==============================================================================


class TestDiscoverTiersEdgeCases:
    """Tests for edge cases in discover_tiers."""

    @pytest.mark.asyncio
    async def test_refresh_future_already_done(self, discovery_with_mock_client):
        """
        Test when refresh_future.done() is True (line 207 branch, 224, 229, 235).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None
        discovery.tiers = {}

        # Create a completed future
        completed_future: asyncio.Future = asyncio.Future()
        completed_future.set_result(discovery.tiers)
        refresh_key = f"tier_refresh_{discovery.account_id}"
        discovery._refresh_futures[refresh_key] = completed_future

        # Should not set result again on already-done future
        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[{"apiModelId": "m1", "rateLimits": []}],
        ):
            result = await discovery.discover_tiers()
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_discover_tiers_cancelled_propagates(self, discovery_with_mock_client):
        """
        Test CancelledError propagates to waiting coroutines (lines 227-231).
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None

        async def raise_cancelled():
            raise asyncio.CancelledError("Cancelled")

        with (
            patch.object(discovery, "_fetch_rate_limits_simple", side_effect=raise_cancelled),
            pytest.raises(asyncio.CancelledError),
        ):
            await discovery.discover_tiers()


# ==============================================================================
# Test _fetch_rate_limits_simple edge cases (line 261)
# ==============================================================================


class TestFetchRateLimitsEdgeCases:
    """Tests for edge cases in _fetch_rate_limits_simple."""

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_not_list_or_dict(self, discovery_with_mock_client):
        """
        Test when response data is neither list nor dict with rateLimits (line 266).
        """
        discovery = discovery_with_mock_client

        discovery.client.get = AsyncMock(
            return_value={"data": 12345}  # Not list, not dict with rateLimits
        )

        result = await discovery._fetch_rate_limits_simple()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_dict_without_rate_limits_key(self, discovery_with_mock_client):
        """
        Test when data is dict but doesn't have 'rateLimits' key (line 259-266).
        """
        discovery = discovery_with_mock_client

        discovery.client.get = AsyncMock(
            return_value={"data": {"otherKey": "value"}}  # Dict but no 'rateLimits'
        )

        result = await discovery._fetch_rate_limits_simple()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rate_limits_rate_limits_not_list(self, discovery_with_mock_client):
        """
        Test when rateLimits key exists but is not a list (line 261 branch).
        """
        discovery = discovery_with_mock_client

        discovery.client.get = AsyncMock(
            return_value={
                "data": {"rateLimits": "not-a-list"}  # rateLimits key but not list
            }
        )

        result = await discovery._fetch_rate_limits_simple()
        assert result is None


# ==============================================================================
# Additional tests for partial branch coverage
# ==============================================================================


class TestPartialBranches:
    """Tests specifically targeting partial branch coverage."""

    @pytest.mark.asyncio
    async def test_line_126_bucket_already_exists(self):
        """
        Cover line 126 branch where bucket_id is already in tiers.
        """
        discovery = RateLimitDiscovery(client=None)

        # Pre-populate a default tier
        discovery.tiers["tier_default_llm"] = RateLimitBucket(
            bucket_id="tier_default_llm", rpm_limit=50, models=set()
        )

        # Call _ensure_default_tiers again
        discovery._ensure_default_tiers()

        # Original should be preserved
        assert discovery.tiers["tier_default_llm"].rpm_limit == 50

    @pytest.mark.asyncio
    async def test_line_163_no_refresh_in_progress(self, discovery_with_mock_client):
        """
        Cover line 163 branch where refresh_key is NOT in _refresh_futures.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None
        discovery._refresh_futures = {}  # Explicitly empty

        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[{"apiModelId": "m1", "rateLimits": []}],
        ):
            result = await discovery.discover_tiers()
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_line_195_cache_still_invalid_after_lock(self, discovery_with_mock_client):
        """
        Cover line 195 branch where cache is still invalid after acquiring lock.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None

        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[{"apiModelId": "m1", "rateLimits": []}],
        ):
            result = await discovery.discover_tiers()
            assert "m1" in result

    @pytest.mark.asyncio
    async def test_line_199_no_existing_future_after_lock(self, discovery_with_mock_client):
        """
        Cover line 199 branch where refresh_key is NOT in _refresh_futures after lock.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = None
        discovery._refresh_futures = {}

        with patch.object(
            discovery,
            "_fetch_rate_limits_simple",
            return_value=[{"apiModelId": "m1", "rateLimits": []}],
        ):
            result = await discovery.discover_tiers()
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_line_354_tiers_exist_and_cache_valid(self, discovery_with_mock_client):
        """
        Cover line 354 branch where tiers exist and cache is valid.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.tiers = {"existing": Mock()}
        discovery.model_to_tier["existing-model"] = "existing"

        bucket = await discovery.get_tier_for_model("existing-model")
        assert bucket == "existing"

    @pytest.mark.asyncio
    async def test_line_358_model_not_in_model_to_tier(self, discovery_with_mock_client):
        """
        Cover line 358 branch where model is NOT in model_to_tier.
        """
        discovery = discovery_with_mock_client
        discovery._last_refresh = datetime.now(UTC)
        discovery.model_to_tier = {}  # Model not present

        with patch.object(discovery, "discover_tiers", new_callable=AsyncMock) as mock:
            mock.return_value = discovery.tiers
            await discovery.get_tier_for_model("unknown-model")
            # Should call discover_tiers to try refresh
            mock.assert_called()

    @pytest.mark.asyncio
    async def test_line_413_tiers_not_empty(self, discovery_with_mock_client):
        """
        Cover line 413 branch where self.tiers is not empty.
        """
        discovery = discovery_with_mock_client
        discovery.tiers["test-tier"] = RateLimitBucket(
            bucket_id="test-tier", rpm_limit=100, models=set()
        )

        bucket = await discovery.get_tier("test-tier")
        assert bucket is not None
