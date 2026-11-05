"""
Comprehensive tests for observability enhanced metrics system.

Tests cover all enhanced metrics functionality including Prometheus integration
and the singleton pattern for the 9 active metrics.
"""

from unittest.mock import Mock, patch

import pytest

import venice_ai.observability.metrics as metrics_module
from venice_ai.observability.metrics import (
    EnhancedMetrics,
    EnhancedMetricsConfig,
    get_enhanced_metrics,
)


class TestEnhancedMetricsConfig:
    """Test EnhancedMetricsConfig model."""

    def test_config_creation(self):
        """Test creating enhanced metrics config."""
        config = EnhancedMetricsConfig(
            enabled=True,
            include_detailed_metrics=True,
            prometheus_port=9090,
            prometheus_host="0.0.0.0",
        )

        assert config.enabled is True
        assert config.include_detailed_metrics is True
        assert config.prometheus_port == 9090
        assert config.prometheus_host == "0.0.0.0"

    def test_config_defaults(self):
        """Test default configuration values."""
        config = EnhancedMetricsConfig()

        assert config.enabled is True
        assert config.include_detailed_metrics is False
        assert config.prometheus_port == 8000
        assert config.prometheus_host == "127.0.0.1"


class TestEnhancedMetricsInitialization:
    """Test EnhancedMetrics initialization."""

    def test_init_with_prometheus_available(self):
        """Test initialization when Prometheus is available."""
        config = EnhancedMetricsConfig(enabled=True)

        with (
            patch("venice_ai.observability.metrics.PROMETHEUS_AVAILABLE", True),
            patch.object(EnhancedMetrics, "_init_enhanced_metrics"),
        ):
            metrics = EnhancedMetrics(config=config)

            assert metrics.config == config
            assert metrics._enabled

    def test_init_without_prometheus(self):
        """Test initialization when Prometheus is not available."""
        config = EnhancedMetricsConfig(enabled=True)

        with patch("venice_ai.observability.metrics.PROMETHEUS_AVAILABLE", False):
            metrics = EnhancedMetrics(config=config)

            assert metrics.config == config
            assert metrics._enabled is False

    def test_init_with_config_disabled(self):
        """Test initialization when metrics are disabled in config."""
        config = EnhancedMetricsConfig(enabled=False)
        metrics = EnhancedMetrics(config=config)

        assert metrics._enabled is False

    def test_init_without_config(self):
        """Test initialization without providing config."""
        metrics = EnhancedMetrics()

        assert isinstance(metrics.config, EnhancedMetricsConfig)
        assert metrics.config.enabled is True

    def test_init_with_base_collector(self):
        """Test initialization with base metrics collector."""
        mock_collector = Mock()
        mock_collector.registry = Mock()

        metrics = EnhancedMetrics(base_collector=mock_collector)

        assert metrics._base_collector == mock_collector


class TestEnhancedMetricsWithDummyMetrics:
    """Test EnhancedMetrics with dummy metrics (Prometheus not available)."""

    @pytest.fixture
    def metrics(self):
        """Create EnhancedMetrics with dummy metrics."""
        with patch("venice_ai.observability.metrics.PROMETHEUS_AVAILABLE", False):
            return EnhancedMetrics(config=EnhancedMetricsConfig(enabled=True))

    def test_dummy_metrics_have_active_attributes(self, metrics):
        """Test that dummy metrics have all 9 active metric attributes."""
        assert hasattr(metrics, "streaming_fallback_total")
        assert hasattr(metrics, "custom_stream_created_total")
        assert hasattr(metrics, "custom_stream_bytes_total")
        assert hasattr(metrics, "custom_stream_duration_seconds")
        assert hasattr(metrics, "tier_discovery_requests_total")
        assert hasattr(metrics, "tier_discovery_api_calls_total")
        assert hasattr(metrics, "tier_discovery_coalesced_total")
        assert hasattr(metrics, "tier_discovery_concurrent_requests")
        assert hasattr(metrics, "tier_discovery_time_saved_seconds")


class TestEnhancedMetricsInitErrors:
    """Test error handling during initialization."""

    def test_init_enhanced_metrics_exception(self):
        """Test handling of exception during Prometheus metric initialization."""
        config = EnhancedMetricsConfig(enabled=True)

        with (
            patch("venice_ai.observability.metrics.PROMETHEUS_AVAILABLE", True),
            patch(
                "venice_ai.observability.metrics.Counter",
                side_effect=Exception("Init error"),
            ),
            patch("venice_ai.observability.metrics.Histogram", None),
            patch("venice_ai.observability.metrics.Gauge", None),
        ):
            # Mock Counter/Histogram/Gauge but not None
            metrics = EnhancedMetrics(config=config)

            # Should fall back to dummy metrics
            assert hasattr(metrics, "streaming_fallback_total")

    def test_prometheus_imports_none(self):
        """Test when Prometheus imports result in None."""
        config = EnhancedMetricsConfig(enabled=True)

        with (
            patch("venice_ai.observability.metrics.PROMETHEUS_AVAILABLE", True),
            patch("venice_ai.observability.metrics.Counter", None),
            patch("venice_ai.observability.metrics.Histogram", None),
            patch("venice_ai.observability.metrics.Gauge", None),
        ):
            metrics = EnhancedMetrics(config=config)

            # Should use dummy metrics
            assert hasattr(metrics, "streaming_fallback_total")


class TestGetEnhancedMetrics:
    """Test global enhanced metrics singleton."""

    def test_get_enhanced_metrics_creates_instance(self):
        """Test that get_enhanced_metrics creates an instance."""
        # Reset global instance

        metrics_module._enhanced_metrics = None

        config = EnhancedMetricsConfig(enabled=True)
        metrics = get_enhanced_metrics(config=config)

        assert isinstance(metrics, EnhancedMetrics)
        assert metrics.config.enabled is True

    def test_get_enhanced_metrics_singleton(self):
        """Test that get_enhanced_metrics returns the same instance."""

        metrics_module._enhanced_metrics = None

        metrics1 = get_enhanced_metrics()
        metrics2 = get_enhanced_metrics()

        assert metrics1 is metrics2

    def test_get_enhanced_metrics_with_base_collector(self):
        """Test get_enhanced_metrics with base collector."""

        metrics_module._enhanced_metrics = None

        mock_get_collector = Mock(return_value=Mock())

        with patch("venice_ai.observability.metrics.get_metrics_collector", mock_get_collector):
            metrics = get_enhanced_metrics()

            assert isinstance(metrics, EnhancedMetrics)
            mock_get_collector.assert_called_once()

    def test_get_enhanced_metrics_no_base_collector(self):
        """Test get_enhanced_metrics when get_metrics_collector is None."""

        metrics_module._enhanced_metrics = None

        with patch("venice_ai.observability.metrics.get_metrics_collector", None):
            metrics = get_enhanced_metrics()

            assert isinstance(metrics, EnhancedMetrics)
            assert metrics._base_collector is None
