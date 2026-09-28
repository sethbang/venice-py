"""
Root configuration model for the Venice AI SDK.

Core configs (``HttpClientConfig``, ``VeniceAIConfig``) are required for basic SDK
operation.  Enterprise configs (``BackendConfig``, ``StateConfig``, ``SchedulerConfig``,
etc.) are optional and only needed for advanced deployments.

Quick start::

    from venice_ai.core.config import create_minimal_config
    cfg = create_minimal_config(api_key="your-key")

Environment variables use the ``VENICE_`` prefix with ``__`` as the nested delimiter.
"""

import warnings
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    # For type-checking purposes, always import the real BaseSettings/SettingsConfigDict.
    # Pylance sees this branch and resolves VeniceAIConfig's base class correctly.
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _pydantic_settings_available: bool
else:
    # At runtime: try the real package, fall back to pydantic's BaseModel/ConfigDict
    # when pydantic-settings is not installed.
    try:
        from pydantic_settings import BaseSettings, SettingsConfigDict

        _pydantic_settings_available = True
    except ImportError:  # pragma: no cover
        # Fall back to plain BaseModel when pydantic-settings is not installed.
        # VeniceAIConfig will not support env-var population in this mode, but
        # the module will import without raising ImportError.
        BaseSettings = BaseModel  # type: ignore[assignment,misc]
        SettingsConfigDict = ConfigDict  # type: ignore[assignment,misc]
        _pydantic_settings_available = False

from ...rate_limiting.config import RateLimiterConfig
from .backends import BackendConfig, RedisBackendConfig
from .enterprise import CircuitBreakerConfig, MetricsConfig, SchedulerConfig, StateConfig
from .enums import BackendType, CachePolicy, SchedulerMode
from .http import HttpClientConfig

# =============================================================================
# Root Configuration Model
# =============================================================================


class VeniceAIConfig(BaseSettings):
    """Root configuration model for the Venice AI SDK.

    Quick start::

        cfg = VeniceAIConfig(api_key="your-key")

    Environment variables use the ``VENICE_`` prefix (e.g. ``VENICE_API_KEY``).

    Core fields: ``api_key``, ``api_base_url``, ``debug``, ``http_client``.
    Enterprise fields: ``backend``, ``scheduler``, ``state``, ``circuit_breaker``,
    ``rate_limiter``, ``metrics``.
    """

    model_config = SettingsConfigDict(
        env_prefix="VENICE_",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="forbid",
        validate_assignment=True,
        case_sensitive=False,
    )

    def model_post_init(self, __context: Any) -> None:
        """Warn when pydantic-settings is not installed, and detect Redis/backend mismatches.

        Without pydantic-settings, VeniceAIConfig inherits from BaseModel
        instead of BaseSettings, which means VENICE_* environment variables
        (including VENICE_API_KEY) are silently ignored.
        """
        super().model_post_init(__context)
        # stacklevel=3 targets the user's VeniceAIConfig(...) call assuming
        # pydantic v2's __init__ → model_post_init is exactly 2 frames deep.
        if not _pydantic_settings_available:
            warnings.warn(
                "pydantic-settings is not installed. VENICE_* environment variables "
                "(e.g. VENICE_API_KEY) will NOT be read automatically. "
                "Install it with: pip install pydantic-settings  "
                "or: pip install 'venice-py[enterprise]'",
                UserWarning,
                stacklevel=3,
            )
        if (
            self.backend.backend_type == BackendType.MEMORY
            and self.backend.redis is not None
            and self.backend.redis.redis_url
        ):
            warnings.warn(
                "Redis URL is configured but backend_type is MEMORY. "
                "Set backend_type=BackendType.REDIS to use Redis. "
                "The default backend_type changed from REDIS to MEMORY in this version.",
                UserWarning,
                stacklevel=3,
            )

    # === Core / Global Settings ===

    api_key: str | None = Field(
        default=None,
        description="Venice AI API key. Can also be set via the VENICE_API_KEY env-var.",
    )

    api_base_url: str = Field(
        default="https://api.venice.ai", description="Base URL for Venice AI API"
    )

    api_version: str = Field(default="v1", description="API version to use")

    debug: bool = Field(default=False, description="Enable debug mode")

    environment: str = Field(
        default="production",
        description="Environment: production, staging, development, test",
    )

    # === Core Component Configurations ===

    http_client: HttpClientConfig = Field(
        default_factory=HttpClientConfig, description="HTTP client configuration"
    )

    rate_limiter: RateLimiterConfig = Field(
        default_factory=RateLimiterConfig,
        description="Rate limiter configuration (simple reactive or adaptive proactive)",
    )

    # === Enterprise Component Configurations ===

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig, description="Request scheduler configuration"
    )

    state: StateConfig = Field(
        default_factory=StateConfig, description="State management configuration"
    )

    backend: BackendConfig = Field(
        default_factory=BackendConfig, description="Backend storage configuration"
    )

    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )

    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig, description="Metrics collection configuration"
    )

    # === Validation ===

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = {"production", "staging", "development", "test"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, v: str) -> str:
        """Validate API base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("API base URL must start with http:// or https://")
        if v.endswith("/"):
            v = v.rstrip("/")
        return v

    # === Convenience Methods ===

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format."""
        result: dict[str, Any] = self.model_dump(exclude_none=True)
        return result

    def get_redis_url(self) -> str:
        """Get the effective Redis URL for the configuration."""
        # Validate Redis backend is configured
        if self.backend.redis is None:
            raise ValueError(
                "Redis backend configuration is required when backend_type is REDIS. "
                "Provide RedisBackendConfig with redis_url parameter."
            )

        return self.backend.redis.redis_url

    def is_test_environment(self) -> bool:
        """Check if running in test environment."""
        return self.environment == "test" or self.scheduler.test_mode

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug

    @classmethod
    def create_test_config(
        cls,
        scheduler_mode: SchedulerMode = SchedulerMode.BASIC,
        enable_redis: bool = True,
        test_rate_multiplier: float = 10.0,
    ) -> "VeniceAIConfig":
        """
        Create a configuration optimized for testing.

        Args:
            scheduler_mode: Scheduler mode to use. Defaults to ``BASIC`` so that
                callers without a tier-discovery setup get a working test client
                out of the box. Pass ``SchedulerMode.INTELLIGENT`` explicitly
                when exercising tier-aware scheduling paths in the fixture.
            enable_redis: Whether to enable Redis backend
            test_rate_multiplier: Rate limit multiplier for faster testing

        Returns:
            VeniceAIConfig instance optimized for testing
        """
        return cls(
            environment="test",
            debug=True,
            scheduler=SchedulerConfig(
                mode=scheduler_mode,
                test_mode=True,
                test_rate_multiplier=test_rate_multiplier,
                max_concurrent_executions=10,
                max_queue_size=100,
                scheduler_interval=0.01,
            ),
            backend=BackendConfig(
                backend_type=BackendType.REDIS if enable_redis else BackendType.MEMORY,
                redis=RedisBackendConfig(
                    redis_url="redis://localhost:6379/15",  # db 15 is the throwaway test database
                    default_ttl=300,  # Shorter TTL for tests
                )
                if enable_redis
                else None,
            ),
            state=StateConfig(
                cache_ttl=0.1,
                batch_timeout=0.05,
                cleanup_interval=10.0,
                namespace="test",
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=10,  # Higher threshold for tests
                reset_timeout=5.0,  # Shorter reset for tests
            ),
        )

    @classmethod
    def create_minimal_config(cls, api_key: str | None = None, **kwargs: Any) -> "VeniceAIConfig":
        """Create a minimal configuration for basic SDK usage.

        Only an API key is required.  All enterprise features (scheduler, state
        management, account tracking, Redis backend) are disabled by default so
        users are not surprised by unexpected dependencies or background tasks.
        The scheduler section keeps its defaults: it only has an effect once
        ``rate_limiter.mode`` is ``RateLimiterMode.ADAPTIVE``.

        Args:
            api_key: Venice AI API key (can also be provided via VENICE_API_KEY env-var).
            **kwargs: Any additional ``VeniceAIConfig`` field overrides.

        Returns:
            ``VeniceAIConfig`` instance with sensible minimal defaults.

        Example::

            from venice_ai.core.config import create_minimal_config
            cfg = create_minimal_config(api_key="your-key")
        """
        return cls(
            api_key=api_key,
            scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
            backend=BackendConfig(backend_type=BackendType.MEMORY),
            state=StateConfig(
                cache_policy=CachePolicy.WRITE_THROUGH,
                enable_versioning=False,
                enable_background_cleanup=False,
            ),
            **kwargs,
        )


# =============================================================================
# Module-level convenience function
# =============================================================================


def create_minimal_config(api_key: str | None = None, **kwargs: Any) -> VeniceAIConfig:
    """Create a minimal :class:`VeniceAIConfig` for basic SDK usage.

    Delegates to :meth:`VeniceAIConfig.create_minimal_config`.

    Args:
        api_key: Venice AI API key (can also be set via ``VENICE_API_KEY`` env-var).
        **kwargs: Additional ``VeniceAIConfig`` field overrides.

    Returns:
        :class:`VeniceAIConfig` with sensible minimal defaults.

    Example::

        from venice_ai.core.config import create_minimal_config
        cfg = create_minimal_config(api_key="your-key")
    """
    return VeniceAIConfig.create_minimal_config(api_key=api_key, **kwargs)


__all__ = [
    "VeniceAIConfig",
    "create_minimal_config",
]
