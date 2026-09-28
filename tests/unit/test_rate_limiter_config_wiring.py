"""Rate-limiter, scheduler and Redis-backend configuration must take effect.

The factory composes the rate limiter from ``VeniceAIConfig``. These tests
observe what the composed objects actually received — the real upstream
scheduler config, the real Redis backend's key names, the keyword arguments
reaching the backend constructor — rather than asserting that the SDK config
object holds the value the caller put there. A setting that does not reach a
runtime consumer must at least be reported, never silently dropped.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("adaptive_rate_limiter", reason="adaptive extra not installed")

from adaptive_rate_limiter.backends import RedisBackend as UpstreamRedisBackend  # noqa: E402
from adaptive_rate_limiter.scheduler import (  # noqa: E402
    RateLimiterConfig as UpstreamSchedulerConfig,
)

from venice_ai import VeniceClient  # noqa: E402
from venice_ai.core.config import (  # noqa: E402
    BackendConfig,
    BackendType,
    RedisBackendConfig,
    SchedulerConfig,
    VeniceAIConfig,
)
from venice_ai.factory import VeniceClientFactory, create_developer_client  # noqa: E402
from venice_ai.presets import create_production_config  # noqa: E402
from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode  # noqa: E402

# Never resolved or connected to: every backend here is constructed only.
_DEAD_REDIS_URL = "redis://redis.invalid:6379/0"


def _scheduler_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and "scheduler" in str(w.message).lower()
    ]


def _redis_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and "redis" in str(w.message).lower()
    ]


@contextmanager
def _record_warnings() -> Iterator[list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


# ---------------------------------------------------------------------------
# Redis backend selected but never used
# ---------------------------------------------------------------------------


def test_redis_backend_without_adaptive_limiter_is_reported() -> None:
    """``backend_type=REDIS`` only takes effect with the ADAPTIVE rate limiter.
    With any other mode the SDK never contacts Redis, so building such a client
    must not be silent — a dead Redis URL would otherwise go unnoticed."""
    config = VeniceAIConfig.create_minimal_config(api_key="test-key")
    config.backend = BackendConfig(
        backend_type=BackendType.REDIS,
        redis=RedisBackendConfig(redis_url=_DEAD_REDIS_URL),
    )
    assert config.rate_limiter.mode == RateLimiterMode.SIMPLE

    with _record_warnings() as caught, patch.dict(os.environ, {}, clear=True):
        try:
            VeniceClientFactory.create_client(config, api_key="test-key", account_id="acct")
        except (ValueError, RuntimeError) as exc:
            assert "redis" in str(exc).lower(), f"raised for an unrelated reason: {exc!r}"
            return

    assert _redis_warnings(caught), (
        "create_client() accepted backend_type=REDIS with rate_limiter.mode=SIMPLE "
        "and emitted no warning; Redis is never contacted in that combination"
    )


@pytest.mark.parametrize(
    "tuning",
    [
        pytest.param({"max_concurrent_executions": 5}, id="max_concurrent_executions"),
        pytest.param({"overflow_policy": "drop_oldest"}, id="overflow_policy"),
        pytest.param({"request_timeout": 12.0}, id="request_timeout"),
    ],
)
def test_scheduler_config_without_adaptive_limiter_is_reported(tuning: dict[str, Any]) -> None:
    """``SchedulerConfig`` only has a consumer in ADAPTIVE mode. Under the
    default SIMPLE limiter a caller tuning it gets no effect and no signal."""
    config = VeniceAIConfig.create_minimal_config(api_key="test-key")
    config.scheduler = SchedulerConfig(**tuning)
    assert config.rate_limiter.mode == RateLimiterMode.SIMPLE

    with _record_warnings() as caught, patch.dict(os.environ, {}, clear=True):
        try:
            VeniceClientFactory.create_client(config, api_key="test-key", account_id="acct")
        except (ValueError, RuntimeError) as exc:
            assert "scheduler" in str(exc).lower(), f"raised for an unrelated reason: {exc!r}"
            return

    assert _scheduler_warnings(caught), (
        f"create_client() accepted SchedulerConfig({tuning}) with "
        "rate_limiter.mode=SIMPLE and emitted no warning; nothing reads it in that mode"
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: VeniceClientFactory.create_client(
                VeniceAIConfig.create_minimal_config(api_key="k"), api_key="k"
            ),
            id="minimal-config",
        ),
        pytest.param(lambda: create_developer_client(api_key="k"), id="developer-client"),
        pytest.param(lambda: VeniceClient(api_key="k"), id="bare-client"),
    ],
)
def test_default_clients_emit_no_inert_config_warning(build: Any) -> None:
    with _record_warnings() as caught, patch.dict(os.environ, {}, clear=True):
        build()
    assert _redis_warnings(caught) == []
    assert _scheduler_warnings(caught) == []


# ---------------------------------------------------------------------------
# Adaptive mode: capture what the upstream objects received
# ---------------------------------------------------------------------------


@contextmanager
def _adaptive_capture(*, real_backend: bool) -> Iterator[dict[str, Any]]:
    """Run the factory's ADAPTIVE branch with no network.

    Only the upstream ``Scheduler`` is replaced (it would start tasks). The
    Redis backend is either the real class (construction does not connect) or
    a recorder, and the upstream scheduler *config* class is always real.
    """
    captured: dict[str, Any] = {}

    def build_backend(**kwargs: Any) -> Any:
        captured["backend_kwargs"] = kwargs
        backend = UpstreamRedisBackend(**kwargs) if real_backend else MagicMock()
        captured["backend"] = backend
        return backend

    with (
        patch("adaptive_rate_limiter.backends.RedisBackend", side_effect=build_backend),
        patch("adaptive_rate_limiter.scheduler.Scheduler") as scheduler_cls,
    ):
        yield captured
        if scheduler_cls.call_args is not None:
            captured["scheduler_config"] = scheduler_cls.call_args.kwargs["config"]


def _adaptive_config(**overrides: Any) -> VeniceAIConfig:
    config = VeniceAIConfig.create_minimal_config(api_key="test-key")
    for section, value in overrides.items():
        setattr(config, section, value)
    config.rate_limiter = RateLimiterConfig(
        mode=RateLimiterMode.ADAPTIVE, redis_url=_DEAD_REDIS_URL, account_id="acct"
    )
    return config


def test_adaptive_redis_keys_honour_the_configured_key_prefix() -> None:
    config = create_production_config(redis_url=_DEAD_REDIS_URL, redis_key_prefix="venice:example:")
    assert config.rate_limiter.mode == RateLimiterMode.ADAPTIVE

    with _record_warnings() as caught, _adaptive_capture(real_backend=True) as captured:
        VeniceClientFactory._create_rate_limiter(config, MagicMock(), account_id="acct")

    backend = captured["backend"]
    sample_keys = [backend._get_state_key("some-model"), backend.model_limits_key]
    if all(key.startswith("venice:example:") for key in sample_keys):
        return
    prefix_warnings = [str(w.message) for w in caught if "key_prefix" in str(w.message)]
    assert prefix_warnings, (
        "RedisBackendConfig(key_prefix='venice:example:') was silently ignored: the "
        f"adaptive backend writes keys such as {sample_keys} and no warning named key_prefix"
    )


# RedisBackendConfig fields whose names are parameters of the upstream backend.
_SHARED_REDIS_FIELDS = sorted(
    set(RedisBackendConfig.model_fields)
    & set(inspect.signature(UpstreamRedisBackend.__init__).parameters)
)
_REDIS_PERTURBATIONS: dict[str, Any] = {
    "redis_url": "redis://other-host.invalid:6380/2",
    "max_connections": 37,
    "cluster_mode": True,
}


def test_shared_redis_fields_are_known() -> None:
    assert len(_SHARED_REDIS_FIELDS) > 0
    assert set(_SHARED_REDIS_FIELDS) == set(_REDIS_PERTURBATIONS)


@pytest.mark.parametrize("name", _SHARED_REDIS_FIELDS)
def test_redis_backend_config_reaches_the_adaptive_backend(name: str) -> None:
    value = _REDIS_PERTURBATIONS[name]
    redis_cfg = RedisBackendConfig(**{"redis_url": _DEAD_REDIS_URL, name: value})
    config = _adaptive_config(
        backend=BackendConfig(backend_type=BackendType.REDIS, redis=redis_cfg)
    )
    # The backend's own settings are the subject; don't let the rate limiter's
    # URL shadow them.
    config.rate_limiter.redis_url = None

    with _adaptive_capture(real_backend=False) as captured:
        VeniceClientFactory._create_rate_limiter(config, MagicMock(), account_id="acct")

    received = captured["backend_kwargs"].get(name, "<not passed>")
    assert received == value, (
        f"RedisBackendConfig({name}={value!r}) did not reach the adaptive Redis backend; "
        f"it received {name}={received!r}"
    )


# SchedulerConfig mirrors the upstream scheduler config. ``mode`` is excluded:
# the factory deliberately runs the upstream scheduler in INTELLIGENT mode.
_SHARED_SCHEDULER_FIELDS = sorted(
    (
        set(SchedulerConfig.model_fields)
        & {f.name for f in dataclasses.fields(UpstreamSchedulerConfig)}
    )
    - {"mode"}
)
_SCHEDULER_PERTURBATIONS: dict[str, Any] = {
    "max_concurrent_executions": 7,
    "max_queue_size": 77,
    "overflow_policy": "drop_oldest",
    "scheduler_interval": 0.2,
    "request_timeout": 12.0,
    "enable_priority_scheduling": False,
    "enable_rate_limiting": False,
    "rate_limit_buffer_ratio": 0.5,
    "enable_state_persistence": False,
    "enable_graceful_degradation": False,
    "health_check_interval": 11.0,
    "max_consecutive_failures": 9,
    "conservative_multiplier": 0.3,
    "metrics_enabled": False,
    "enable_performance_tracking": False,
    "metrics_export_interval": 13.0,
    "test_mode": True,
    "test_rate_multiplier": 4.0,
}


def test_shared_scheduler_fields_are_known() -> None:
    assert len(_SHARED_SCHEDULER_FIELDS) > 0
    assert set(_SHARED_SCHEDULER_FIELDS) == set(_SCHEDULER_PERTURBATIONS)
    upstream_defaults = UpstreamSchedulerConfig()
    for name, value in _SCHEDULER_PERTURBATIONS.items():
        assert getattr(upstream_defaults, name) != value, f"{name} perturbation equals default"


@pytest.mark.parametrize("name", _SHARED_SCHEDULER_FIELDS)
def test_scheduler_config_reaches_the_adaptive_scheduler(name: str) -> None:
    value = _SCHEDULER_PERTURBATIONS[name]
    config = _adaptive_config(scheduler=SchedulerConfig(**{name: value}))

    with _adaptive_capture(real_backend=False) as captured:
        VeniceClientFactory._create_rate_limiter(config, MagicMock(), account_id="acct")

    upstream = captured["scheduler_config"]
    assert isinstance(upstream, UpstreamSchedulerConfig)
    received = getattr(upstream, name)
    assert received == value, (
        f"SchedulerConfig({name}={value!r}) did not reach the adaptive scheduler; "
        f"it ran with {name}={received!r}"
    )


def test_every_scheduler_config_field_has_a_runtime_consumer() -> None:
    """A SchedulerConfig field is only meaningful if something reads it: the
    adaptive scheduler (a same-named upstream field) or an explicit deprecation."""
    upstream_names = {f.name for f in dataclasses.fields(UpstreamSchedulerConfig)}
    orphans = sorted(
        name
        for name, info in SchedulerConfig.model_fields.items()
        if name not in upstream_names and not info.deprecated
    )
    assert len(SchedulerConfig.model_fields) > 0
    assert orphans == [], (
        f"SchedulerConfig fields with no consumer and no deprecation marker: {orphans}"
    )
