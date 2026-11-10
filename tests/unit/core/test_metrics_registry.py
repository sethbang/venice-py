"""The metrics registry is reachable so an application can serve it."""

import urllib.request
from unittest.mock import patch

import pytest

from venice_ai.core import metrics as core_metrics
from venice_ai.core.metrics import MetricsCollector, get_metrics_collector

prometheus_client = pytest.importorskip("prometheus_client")


def test_registry_is_the_collectors_registry() -> None:
    collector = MetricsCollector(enable_prometheus=True)
    collector.record_validation_success("stand-in-model")

    output = prometheus_client.generate_latest(collector.registry).decode()

    assert 'venice_validation_successful_total{model_name="stand-in-model"} 1.0' in output


def test_registry_is_none_when_prometheus_is_disabled() -> None:
    assert MetricsCollector(enable_prometheus=False).registry is None


def test_registry_is_none_without_prometheus_client() -> None:
    with patch.object(core_metrics, "PROMETHEUS_AVAILABLE", False):
        assert MetricsCollector().registry is None


def test_singleton_registry_can_be_served_over_http() -> None:
    core_metrics.reset_global_metrics_collector()
    try:
        collector = get_metrics_collector()
        collector.record_validation_failure("stand-in-model")
        server, thread = prometheus_client.start_http_server(
            0, addr="127.0.0.1", registry=collector.registry
        )
        try:
            url = f"http://127.0.0.1:{server.server_port}/metrics"
            with urllib.request.urlopen(url, timeout=5) as response:
                body = response.read().decode()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        core_metrics.reset_global_metrics_collector()

    assert 'venice_validation_failed_total{model_name="stand-in-model"} 1.0' in body


def test_enhanced_metrics_register_on_the_public_registry() -> None:
    from venice_ai.observability.metrics import EnhancedMetrics

    class Collector:
        registry = prometheus_client.CollectorRegistry()

    metrics = EnhancedMetrics(base_collector=Collector())
    metrics.tier_discovery_requests_total.inc()

    output = prometheus_client.generate_latest(Collector.registry).decode()
    assert "venice_tier_discovery_requests_total 1.0" in output
