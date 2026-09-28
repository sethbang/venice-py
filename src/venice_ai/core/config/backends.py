"""
Backend configuration models for Venice AI.

Enterprise feature: Configures the storage backends used by state management,
account tracking, and the adaptive scheduler.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import BackendType


class RedisBackendConfig(BaseModel):
    """Configuration for Redis backend operations.

    Enterprise feature: Required when ``BackendConfig.backend_type`` is ``REDIS``.
    ``redis_url`` has no default to prevent accidental localhost connections in production.

    Redis is only contacted by the ADAPTIVE rate limiter, which receives
    ``redis_url``, ``max_connections`` and ``cluster_mode``. A non-default
    ``key_prefix`` is reported with a ``FutureWarning`` when that rate limiter
    is built.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Connection settings
    redis_url: str = Field(
        description="Redis connection URL (REQUIRED - no default for production safety)"
    )
    max_connections: int = Field(default=20, ge=1, description="Maximum Redis connections in pool")
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")
    connection_timeout: float = Field(
        default=5.0, gt=0, description="Connection timeout in seconds"
    )

    # Key management
    key_prefix: str = Field(
        default="venice:v2:",
        deprecated=(
            "the adaptive rate limiter's Redis backend names its keys itself and "
            "does not apply a prefix; isolate deployments with separate Redis "
            "databases or account IDs instead"
        ),
        description="Deprecated and ignored: Redis key names are not prefixed.",
    )
    default_ttl: int = Field(default=3600, ge=0, description="Default TTL for keys in seconds")

    # Cluster mode
    cluster_mode: bool = Field(
        default=False,
        description="Whether to use the Redis Cluster client. ``redis_url`` is then the "
        "seed node used to discover the rest of the cluster.",
    )

    # Performance tuning
    max_retries: int = Field(
        default=3, ge=0, description="Maximum retry attempts for Redis operations"
    )
    retry_delay: float = Field(default=0.1, ge=0, description="Delay between retries in seconds")

    @field_validator("default_ttl")
    @classmethod
    def validate_default_ttl(cls, v: int) -> int:
        """Validate default TTL with recommended maximum of 7 days."""
        from venice_ai.validation.validators import validate_ttl

        return validate_ttl(v, min_val=0, max_val=604800, param_name="default_ttl")

    @field_validator("connection_timeout")
    @classmethod
    def validate_connection_timeout(cls, v: float) -> float:
        """Validate connection timeout with reasonable bounds."""
        from venice_ai.validation.validators import validate_timeout

        return validate_timeout(v, min_val=0.1, max_val=300.0, param_name="connection_timeout")

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int) -> int:
        """Validate max connections with reasonable upper bound."""
        from venice_ai.validation.validators import validate_collection_size

        return validate_collection_size(v, min_val=1, max_val=10000, param_name="max_connections")


class MemoryBackendConfig(BaseModel):
    """Configuration for memory backend cleanup operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Released reservations cleanup configuration
    released_reservations_ttl: float = Field(
        default=3600.0,
        gt=0,
        description="TTL for released reservation tracking in seconds (1 hour default)",
    )
    released_reservations_cleanup_interval: float = Field(
        default=1800.0,
        gt=0,
        description="Interval for cleaning up old released reservations in seconds",
    )


class BackendConfig(BaseModel):
    """Configuration for data persistence backends.

    Enterprise feature: Configures the storage backend used by state management,
    account tracking, and the adaptive scheduler.

    By default uses ``BackendType.MEMORY`` — no external dependencies needed.
    Set ``backend_type=BackendType.REDIS`` for distributed/multi-process deployments
    and provide a ``RedisBackendConfig`` with a valid ``redis_url``.
    """

    model_config = ConfigDict(extra="forbid")

    # Backend selection — defaults to MEMORY so the SDK works out of the box
    backend_type: BackendType = Field(
        default=BackendType.MEMORY, description="Type of backend to use"
    )

    # Backend-specific configurations
    redis: RedisBackendConfig | None = Field(
        default=None,
        description="Redis backend configuration (required if backend_type is REDIS)",
    )
    memory: MemoryBackendConfig = Field(
        default_factory=MemoryBackendConfig, description="Memory backend configuration"
    )

    # General backend settings
    namespace: str = Field(default="default", description="Backend namespace for isolation")
    enable_compression: bool = Field(
        default=False, description="Enable data compression for storage"
    )


__all__ = [
    "RedisBackendConfig",
    "MemoryBackendConfig",
    "BackendConfig",
]
