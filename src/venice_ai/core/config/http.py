"""
HTTP client configuration for Venice AI.

Core feature: Used by all SDK clients for every API request.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._deprecation import warn_on_deprecated_fields


def _default_user_agent() -> str:
    # Lazy import to avoid a circular import (venice_ai.__init__ imports the
    # client, which imports this config module).
    from venice_ai import __version__

    return f"venice-py/{__version__}"


class HttpClientConfig(BaseModel):
    """Configuration for HTTP client operations.

    Core feature: Used by all SDK clients for every API request.
    """

    model_config = ConfigDict(extra="forbid")

    # Connection settings
    timeout: float = Field(default=30.0, gt=0, description="Default request timeout in seconds")

    max_connections: int = Field(
        default=100,
        ge=1,
        description="Maximum simultaneous HTTP connections in the pool. There is no "
        "separate per-host limit, so this is also the concurrency cap for API requests.",
    )

    max_keepalive_connections: int = Field(
        default=20,
        ge=1,
        deprecated=(
            "the aiohttp transport has no keepalive-pool-size setting, so this value "
            "is ignored; use max_connections to bound the connection pool"
        ),
        description="Deprecated and ignored. Use ``max_connections``.",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retries after the first attempt for a retryable failure "
        "(``0`` disables retries).",
    )

    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential base of the retry backoff: the delay before the k-th "
        "retry (counting from 0) is ``1s * retry_backoff_factor ** k``, capped and jittered.",
    )

    # Headers
    user_agent: str = Field(default_factory=_default_user_agent, description="User agent string")

    @model_validator(mode="after")
    def _warn_deprecated(self) -> Self:
        warn_on_deprecated_fields(self, ("max_keepalive_connections",))
        return self


__all__ = [
    "HttpClientConfig",
]
