"""
Metrics collection for Venice AI.

Provides lightweight Prometheus integration for validation metrics.
When ``prometheus_client`` is not installed, all metric operations
silently no-op via :class:`DummyMetric`.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, cast

logger = logging.getLogger(__name__)

# Type stubs for when prometheus is not available
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry, Counter

# Try to import prometheus_client at runtime
try:
    from prometheus_client import CollectorRegistry, Counter

    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None  # type: ignore[assignment,misc]  # optional dep fallback
    CollectorRegistry = None  # type: ignore[assignment,misc]  # optional dep fallback
    PROMETHEUS_AVAILABLE = False


class DummyMetric:
    """Dummy metric class for when Prometheus is not available."""

    def labels(self, **kwargs: Any) -> "DummyMetric":
        return self

    def inc(self, *_: Any, **kwargs: Any) -> None:
        pass

    def dec(self, *_: Any, **kwargs: Any) -> None:
        pass

    def set(self, *_: Any, **kwargs: Any) -> None:
        pass

    def observe(self, *_: Any, **kwargs: Any) -> None:
        pass

    def time(self) -> "DummyMetric":
        return self

    def __enter__(self) -> "DummyMetric":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class MetricsCollector:
    """
    Lightweight metrics collector for Venice AI.

    Maintains only the two Prometheus metrics that are actively used in
    production: ``venice_validation_successful_total`` and
    ``venice_validation_failed_total``.  When Prometheus is unavailable,
    :class:`DummyMetric` instances are used instead.
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        **_kwargs: Any,
    ) -> None:
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        self._registry: Any | None = None

        self._init_prometheus_metrics()

        logger.info("MetricsCollector initialized (Prometheus: %s)", self.enable_prometheus)

    @property
    def registry(self) -> Any | None:
        """The ``CollectorRegistry`` holding the SDK's metrics, or ``None`` without Prometheus.

        The SDK's metrics are not in ``prometheus_client``'s default registry, and
        the SDK starts no server. To serve them, pass this registry explicitly::

            from prometheus_client import start_http_server

            from venice_ai.core.metrics import get_metrics_collector

            registry = get_metrics_collector().registry
            start_http_server(8000, addr="127.0.0.1", registry=registry)

        :meth:`reset` replaces the registry, so read it again after a reset.
        """
        return self._registry

    # ------------------------------------------------------------------
    # Prometheus initialisation
    # ------------------------------------------------------------------

    def _init_prometheus_metrics(self) -> None:
        """Initialise Prometheus metrics and registry."""
        if not self.enable_prometheus or not PROMETHEUS_AVAILABLE:
            self._create_dummy_metrics()
            return

        if CollectorRegistry is None or Counter is None:
            logger.error("Prometheus dependencies not available")
            self.enable_prometheus = False
            self._create_dummy_metrics()
            return

        assert CollectorRegistry is not None  # nosec B101
        assert Counter is not None  # nosec B101

        try:
            self._registry = CollectorRegistry()

            self.validation_successful_total = Counter(
                "venice_validation_successful_total",
                "Total number of successful response validations",
                ["model_name"],
                registry=self._registry,
            )

            self.validation_failed_total = Counter(
                "venice_validation_failed_total",
                "Total number of failed response validations",
                ["model_name"],
                registry=self._registry,
            )

            logger.info("Prometheus metrics initialized")
        except (ImportError, AttributeError, ValueError) as exc:
            logger.warning("Failed to initialize Prometheus metrics: %s", exc)
            self._create_dummy_metrics()

    def _create_dummy_metrics(self) -> None:
        """Create dummy metrics when Prometheus is not available."""
        dummy = DummyMetric()
        self.validation_successful_total = cast(Any, dummy)
        self.validation_failed_total = cast(Any, dummy)
        logger.info("Dummy metrics initialized (Prometheus not available)")

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_validation_success(self, model_name: str, response_type: str = "") -> None:
        """Record a successful validation for *model_name*."""
        del response_type  # reserved for future labelling; not plumbed through yet
        self.validation_successful_total.labels(model_name=model_name).inc()

    def record_validation_failure(
        self, model_name: str, response_type: str = "", error: str = ""
    ) -> None:
        """Record a failed validation for *model_name*."""
        del response_type  # reserved for future labelling; not plumbed through yet
        self.validation_failed_total.labels(model_name=model_name).inc()

    # ------------------------------------------------------------------
    # Lifecycle helpers (kept for potential future use / testing)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-initialise metrics (mainly for testing)."""
        self._init_prometheus_metrics()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_metrics_collector: MetricsCollector | None = None
_lock = threading.Lock()


def get_metrics_collector(
    enable_prometheus: bool = True,
    **kwargs: Any,
) -> MetricsCollector:
    """Return (or create) the global :class:`MetricsCollector` singleton."""
    global _metrics_collector

    with _lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector(
                enable_prometheus=enable_prometheus,
                **kwargs,
            )

    return _metrics_collector


def reset_global_metrics_collector() -> None:
    """Reset the global metrics collector (mainly for testing)."""
    global _metrics_collector

    with _lock:
        _metrics_collector = None
