"""
Enhanced metrics collection for Venice AI observability.

This module extends the core metrics system with additional observability-focused
metrics for production monitoring and alerting.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import core metrics
try:
    from venice_ai.core.metrics import (
        PROMETHEUS_AVAILABLE,
        MetricsCollector,
        get_metrics_collector,
    )
except ImportError:
    MetricsCollector = None  # type: ignore[assignment,misc]  # optional dep fallback
    get_metrics_collector = None  # type: ignore[assignment]  # optional dep fallback
    PROMETHEUS_AVAILABLE = False

# Type stubs for Prometheus
if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

# Try to import Prometheus at runtime
if PROMETHEUS_AVAILABLE:
    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:
        Counter = None  # type: ignore[assignment,misc]  # optional dep fallback
        Histogram = None  # type: ignore[assignment,misc]  # optional dep fallback
        Gauge = None  # type: ignore[assignment,misc]  # optional dep fallback
        PROMETHEUS_AVAILABLE = False


class EnhancedMetricsConfig(BaseModel):
    """Configuration for enhanced metrics collection."""

    enabled: bool = Field(default=True, description="Enable enhanced metrics")
    include_detailed_metrics: bool = Field(
        default=False, description="Include detailed per-model metrics"
    )
    prometheus_port: int = Field(
        default=8000,
        description=(
            "Not read by the SDK, which starts no metrics server; "
            "pass your port to start_http_server."
        ),
    )
    prometheus_host: str = Field(
        default="127.0.0.1",
        description=(
            "Not read by the SDK, which starts no metrics server; "
            "pass your address to start_http_server."
        ),
    )


class EnhancedMetrics:
    """
    Enhanced metrics collector for Venice AI observability.

    Extends the core MetricsCollector with additional production-focused metrics:
    - Custom stream usage (created, bytes, duration)
    - Streaming fallback tracking
    - Tier discovery coalescing metrics

    **Usage:**
    ```python
    from prometheus_client import start_http_server

    from venice_ai.core.metrics import get_metrics_collector
    from venice_ai.observability import EnhancedMetricsConfig, get_enhanced_metrics

    metrics = get_enhanced_metrics(EnhancedMetricsConfig(enabled=True))
    start_http_server(8000, addr="127.0.0.1", registry=get_metrics_collector().registry)
    ```
    """

    def __init__(
        self,
        config: EnhancedMetricsConfig | None = None,
        base_collector: Any | None = None,
    ):
        """
        Initialize enhanced metrics.

        Args:
            config: Enhanced metrics configuration
            base_collector: Collector whose registry holds these metrics. Without
                one they are in no registry and cannot be served;
                :func:`get_enhanced_metrics` supplies it.
        """
        self.config = config or EnhancedMetricsConfig()
        self._base_collector = base_collector
        self._enabled = self.config.enabled and PROMETHEUS_AVAILABLE

        # Initialize enhanced Prometheus metrics if available
        if self._enabled and PROMETHEUS_AVAILABLE:
            self._init_enhanced_metrics()
        else:
            self._create_dummy_metrics()

        logger.info(f"EnhancedMetrics initialized (enabled: {self._enabled})")

    def _is_prometheus_available(self) -> bool:
        """Check if Prometheus metrics are fully available.

        Returns:
            True if prometheus_client is installed and all metric types are available
        """
        return (
            PROMETHEUS_AVAILABLE
            and Counter is not None
            and Histogram is not None
            and Gauge is not None
        )

    def _init_enhanced_metrics(self) -> None:
        """Initialize enhanced Prometheus metrics."""
        if not self._is_prometheus_available():
            self._create_dummy_metrics()
            return

        # Type narrowing - after _is_prometheus_available check, these cannot be None
        assert Counter is not None and Histogram is not None and Gauge is not None  # nosec B101

        try:
            registry = self._base_collector.registry if self._base_collector else None

            # Streaming fallback metrics
            self.streaming_fallback_total = Counter(
                "venice_streaming_fallback_total",
                "Total streaming fallback occurrences",
                ["endpoint", "reason"],
                registry=registry,
            )

            # Custom stream usage metrics
            self.custom_stream_created_total = Counter(
                "venice_custom_stream_created_total",
                "Total custom stream instances created",
                ["stream_type"],
                registry=registry,
            )

            self.custom_stream_bytes_total = Counter(
                "venice_custom_stream_bytes_total",
                "Total bytes streamed through custom streams",
                ["stream_type"],
                registry=registry,
            )

            self.custom_stream_duration_seconds = Histogram(
                "venice_custom_stream_duration_seconds",
                "Duration of custom stream operations",
                ["stream_type"],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
                registry=registry,
            )

            # Tier discovery coalescing metrics
            self.tier_discovery_requests_total = Counter(
                "venice_tier_discovery_requests_total",
                "Total tier discovery requests (including coalesced)",
                registry=registry,
            )

            self.tier_discovery_api_calls_total = Counter(
                "venice_tier_discovery_api_calls_total",
                "Actual API calls for tier discovery (unique requests)",
                registry=registry,
            )

            self.tier_discovery_coalesced_total = Counter(
                "venice_tier_discovery_coalesced_total",
                "Total requests that were coalesced (cache hits)",
                registry=registry,
            )

            self.tier_discovery_concurrent_requests = Gauge(
                "venice_tier_discovery_concurrent_requests",
                "Number of concurrent requests during coalescing",
                registry=registry,
            )

            self.tier_discovery_time_saved_seconds = Counter(
                "venice_tier_discovery_time_saved_seconds",
                "Total time saved by request coalescing",
                registry=registry,
            )

            logger.info("Enhanced Prometheus metrics initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize enhanced Prometheus metrics: {e}")
            self._create_dummy_metrics()

    def _create_dummy_metrics(self) -> None:
        """Create dummy metrics when Prometheus is not available."""
        from venice_ai.core.metrics import DummyMetric

        dummy = DummyMetric()
        self.streaming_fallback_total = dummy  # type: ignore[assignment]  # dummy fallback when prom unavailable
        self.custom_stream_created_total = dummy  # type: ignore[assignment]
        self.custom_stream_bytes_total = dummy  # type: ignore[assignment]
        self.custom_stream_duration_seconds = dummy  # type: ignore[assignment]
        self.tier_discovery_requests_total = dummy  # type: ignore[assignment]
        self.tier_discovery_api_calls_total = dummy  # type: ignore[assignment]
        self.tier_discovery_coalesced_total = dummy  # type: ignore[assignment]
        self.tier_discovery_concurrent_requests = dummy  # type: ignore[assignment]
        self.tier_discovery_time_saved_seconds = dummy  # type: ignore[assignment]

        logger.info("Dummy enhanced metrics initialized")


# Singleton instance
_enhanced_metrics: EnhancedMetrics | None = None
_metrics_lock = threading.Lock()


def get_enhanced_metrics(
    config: EnhancedMetricsConfig | None = None,
) -> EnhancedMetrics:
    """
    Get or create the global enhanced metrics instance.

    Args:
        config: Enhanced metrics configuration

    Returns:
        EnhancedMetrics instance
    """
    global _enhanced_metrics

    if _enhanced_metrics is None:
        with _metrics_lock:
            if _enhanced_metrics is None:
                # Get base collector
                base_collector = None
                if get_metrics_collector is not None:
                    base_collector = get_metrics_collector()

                _enhanced_metrics = EnhancedMetrics(
                    config=config or EnhancedMetricsConfig(),
                    base_collector=base_collector,
                )

    return _enhanced_metrics
