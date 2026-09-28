"""
Enterprise configuration models for Venice AI.

These models are optional and only needed for advanced deployments:
circuit breaker, state management, scheduler, and metrics.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._deprecation import warn_on_deprecated_fields
from .enums import CachePolicy, SchedulerMode

# =============================================================================
# Circuit Breaker Configuration
# =============================================================================


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern implementation.

    Enterprise feature: Wraps backend/account calls with a circuit breaker that
    opens after ``failure_threshold`` failures, preventing cascading failures.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_threshold: int = Field(
        default=5, ge=1, description="Number of failures before opening circuit"
    )
    reset_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Time in seconds before attempting circuit reset",
    )
    success_threshold: int = Field(
        default=1,
        ge=1,
        description="Number of successes required to close circuit in half-open state",
    )

    @field_validator("reset_timeout")
    @classmethod
    def validate_reset_timeout(cls, v: float) -> float:
        """Ensure reset timeout is reasonable."""
        if v > 3600:  # 1 hour
            raise ValueError("Reset timeout cannot exceed 1 hour")
        return v


# =============================================================================
# State Management Configuration
# =============================================================================


class StateConfig(BaseModel):
    """Configuration for the unified state management system.

    Enterprise feature: Persists rate-limit state, reservations, and account health
    for the adaptive scheduler.  Not required for basic SDK usage.

    Key areas: cache policy (write-through/back/around), TTL settings, batch sizes,
    concurrency control, and cleanup intervals.
    """

    model_config = ConfigDict(extra="forbid")

    # Cache configuration
    cache_ttl: float = Field(
        default=1.0,
        gt=0,
        description="Cache TTL in seconds. How long entries stay in memory before expiration. "
        "Lower values = more backend reads, higher values = stale data risk.",
    )
    max_cache_size: int | None = Field(
        default=1000,
        ge=1,
        description="Maximum number of cache entries before LRU eviction. "
        "Set based on memory constraints and working set size.",
    )
    cache_policy: CachePolicy = Field(
        default=CachePolicy.WRITE_BACK,
        description="Cache write policy. See CachePolicy for options.",
    )

    # Production safety configuration
    warn_write_back_production: bool = Field(
        default=True,
        description="Emit warning when WRITE_BACK cache policy is used in production",
    )
    is_production: bool = Field(
        default=False,
        description="Set to True in production environments to enable production safety checks",
    )

    # Batch processing
    batch_size: int = Field(default=50, ge=1, le=1000, description="Batch size for backend writes")
    batch_timeout: float = Field(default=0.1, gt=0, description="Batch timeout in seconds")

    # Cleanup and maintenance
    cleanup_interval: float = Field(default=30.0, gt=0, description="Cleanup interval in seconds")
    enable_background_cleanup: bool = Field(
        default=True, description="Enable automatic background cleanup"
    )

    # Cleanup operation timeouts
    cleanup_task_cancel_timeout: float = Field(
        default=2.0,
        gt=0,
        description="Timeout for cancelling cross-loop cleanup tasks in seconds",
    )
    cleanup_task_wait_timeout: float = Field(
        default=1.0,
        gt=0,
        description="Timeout for waiting on same-loop cleanup task cancellation in seconds",
    )

    # State persistence TTL
    state_ttl: int = Field(
        default=3600,
        ge=60,
        description="TTL for state keys in backend storage (seconds, default 1 hour)",
    )

    # Reservation cleanup configuration
    reservation_cleanup_interval: float = Field(
        default=3600.0,
        gt=0,
        description="Interval for cleaning up expired reservations in seconds",
    )
    reservation_ttl: float = Field(
        default=300.0, gt=0, description="TTL for reservation entries in seconds"
    )

    # Account state cleanup configuration
    account_state_ttl: float = Field(
        default=86400.0,
        gt=0,
        description="TTL for account state entries in seconds (24 hours default)",
    )
    account_state_max_size: int | None = Field(
        default=10000,
        ge=1,
        description="Maximum number of account state entries before LRU eviction",
    )

    # Versioning and recovery
    enable_versioning: bool = Field(
        default=True, description="Enable state versioning for recovery"
    )
    max_versions: int = Field(
        default=10, ge=1, le=100, description="Maximum versions to keep per state entry"
    )

    # Concurrency control
    lock_free_reads: bool = Field(
        default=True,
        description=(
            "Enable lock-free reads for better performance. "
            "When True, reads skip locks (approximate metrics, non-atomic updates). "
            "When False, all ops use asyncio.Lock for strict consistency. "
            "Use False for critical rate-limiting or financial data. "
            "For atomic ops in lock-free mode, use Cache.atomic_update()."
        ),
    )
    max_concurrent_operations: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent state operations to prevent resource exhaustion. "
        "Higher values = more parallelism but higher memory/CPU usage.",
    )

    # Namespace isolation
    namespace: str = Field(
        default="default", description="State namespace for multi-tenant isolation"
    )


# =============================================================================
# Scheduler Configuration (Enhanced from existing)
# =============================================================================


class SchedulerConfig(BaseModel):
    """Unified configuration for the Venice AI request scheduler.

    Enterprise feature: Only active when ``RateLimiterMode.ADAPTIVE`` is selected
    (``VeniceAIConfig.rate_limiter.mode = RateLimiterMode.ADAPTIVE``).
    Requires: ``pip install 'venice-py[adaptive]'`` and a Redis backend.

    For basic SDK usage the default ``RateLimiterMode.SIMPLE`` mode does not use
    this scheduler at all — only ``HttpClientConfig`` is needed. Building a
    client through :class:`~venice_ai.factory.VeniceClientFactory` with a
    non-default ``SchedulerConfig`` outside ADAPTIVE mode emits a ``UserWarning``.

    In ADAPTIVE mode every field that shares a name with the upstream
    ``adaptive_rate_limiter`` scheduler configuration is forwarded to it, with
    one exception: ``mode`` is ignored, because the adaptive scheduler always
    runs in its INTELLIGENT mode.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # === Core Scheduling Configuration ===

    mode: SchedulerMode = Field(
        default=SchedulerMode.INTELLIGENT,
        description="Scheduler operation mode. Ignored in ADAPTIVE mode, where the "
        "adaptive scheduler always runs in INTELLIGENT mode.",
    )

    strategy: str = Field(
        default="weighted_round_robin",
        deprecated="no scheduler reads it",
        description="Deprecated and ignored.",
    )

    max_concurrent_executions: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of concurrent executions",
    )

    max_queue_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum size of each model's request queue",
    )

    overflow_policy: str = Field(
        default="reject", description="Policy when queue is full: reject or drop_oldest"
    )

    scheduler_interval: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="Interval between scheduler loops in seconds",
    )

    # === Request Processing ===

    request_timeout: float = Field(
        default=30.0, gt=0, le=3600, description="Request timeout duration in seconds"
    )

    enable_priority_scheduling: bool = Field(
        default=True, description="Enable priority-based request scheduling"
    )

    enable_request_batching: bool = Field(
        default=False,
        deprecated="no scheduler reads it",
        description="Deprecated and ignored.",
    )

    # === Rate Limiting Integration ===

    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting enforcement")

    rate_limit_buffer_ratio: float = Field(
        default=0.9, gt=0, le=1.0, description="Ratio of rate limit to use as buffer"
    )

    # === State Management Integration ===

    enable_state_persistence: bool = Field(
        default=True, description="Enable state persistence to backend"
    )

    # === Graceful Degradation ===

    enable_graceful_degradation: bool = Field(
        default=True, description="Enable graceful degradation on failures"
    )

    health_check_interval: float = Field(
        default=30.0, gt=0, description="Interval between health checks in seconds"
    )

    max_consecutive_failures: int = Field(
        default=3, ge=1, description="Maximum consecutive failures before degradation"
    )

    conservative_multiplier: float = Field(
        default=0.6,
        gt=0,
        le=1.0,
        description="Multiplier for conservative capacity during degradation",
    )

    # === Model Management ===

    model_fallbacks: dict[str, str] = Field(
        default_factory=dict,
        deprecated="no scheduler reads it",
        description="Deprecated and ignored.",
    )

    enable_model_discovery: bool = Field(
        default=True,
        deprecated="no scheduler reads it",
        description="Deprecated and ignored.",
    )

    # === Metrics and Monitoring ===

    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")

    enable_performance_tracking: bool = Field(
        default=True, description="Enable detailed performance tracking"
    )

    metrics_export_interval: float = Field(
        default=60.0, gt=0, description="Interval for metrics export in seconds"
    )

    # === Testing Support ===

    test_mode: bool = Field(default=False, description="Enable test mode with relaxed constraints")

    test_rate_multiplier: float = Field(
        default=1.0, gt=0, description="Rate limit multiplier for testing"
    )

    # Validation

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate scheduling strategy."""
        valid_strategies = {
            "weighted_round_robin",
            "deficit_round_robin",
            "priority",
            "fair_queue",
            "adaptive",
        }
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}. Must be one of {valid_strategies}")
        return v

    @field_validator("overflow_policy")
    @classmethod
    def validate_overflow_policy(cls, v: str) -> str:
        """Validate overflow policy."""
        valid_policies = {"reject", "drop_oldest"}
        if v not in valid_policies:
            raise ValueError(f"Invalid overflow_policy: {v}. Must be one of {valid_policies}")
        return v

    @model_validator(mode="after")
    def _warn_deprecated(self) -> Self:
        warn_on_deprecated_fields(
            self,
            ("strategy", "enable_request_batching", "model_fallbacks", "enable_model_discovery"),
        )
        return self


# =============================================================================
# Metrics Configuration
# =============================================================================


class MetricsConfig(BaseModel):
    """Configuration for metrics collection and cleanup.

    Enterprise feature: Controls the internal Prometheus-compatible metrics store
    used by the observability subsystem.  Only needed when metrics export is enabled.
    """

    model_config = ConfigDict(extra="forbid")

    # Metrics cleanup configuration
    max_metric_metadata_size: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of metric metadata entries before LRU eviction",
    )
    metric_metadata_cleanup_interval: float = Field(
        default=3600.0,
        gt=0,
        description="Interval for cleaning up old metric metadata in seconds",
    )


__all__ = [
    "CircuitBreakerConfig",
    "StateConfig",
    "SchedulerConfig",
    "MetricsConfig",
]
