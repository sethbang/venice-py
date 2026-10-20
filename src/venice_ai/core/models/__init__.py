"""
Venice AI Core Models Package.

This package contains comprehensive Pydantic models for all Venice AI API interactions.
All models provide full validation, type safety, and auto-completion for Python clients.

The package is organized into modules:

* **base**    — ``VeniceBaseModel``, ``TimestampMixin``
* **enums**   — ``ModelType``, ``APIKeyType``, ``Currency``, ``FinishReason``, ``MessageRole``
* **headers** — ``RateLimitInfo``, ``DeprecationInfo``, ``BalanceInfo``, ``ContentSafetyInfo``, ``ModelInfo``, ``PaginationInfo``
* **metrics** — ``UsageInfo``, ``TimingInfo``, ``SchedulerMetrics``, ``CacheStats``
* **common**  — Content types, tool types, Venice parameters, and remaining models (re-exports everything above)

All models are designed to work seamlessly with the Venice AI API and provide
comprehensive type safety and validation through Pydantic v2.
"""

# ---------------------------------------------------------------------------
# Import from focused submodules (the canonical sources)
# Order matters: submodules must load before .common which calls model_rebuild()
# ---------------------------------------------------------------------------
from .base import TimestampMixin, VeniceBaseModel  # noqa: I001
from .enums import APIKeyType, Currency, FinishReason, MessageRole, ModelType
from .headers import (
    BalanceInfo,
    ContentSafetyInfo,
    DeprecationInfo,
    ModelInfo,
    PaginationInfo,
    RateLimitInfo,
)
from .metrics import CacheStats, SchedulerMetrics, TimingInfo, UsageInfo

# ---------------------------------------------------------------------------
# Import remaining types from common (which also re-exports all of the above)
# ---------------------------------------------------------------------------
from .common import (
    KNOWN_WEB_SEARCH_MODES,
    AudioContent,
    Balances,
    ConsumptionLimit,
    DateRangeParams,
    HealthCheckResult,
    ImageContent,
    ImageUrl,
    JSONObjectFormat,
    JSONSchemaFormat,
    ListResponse,
    PaginationParams,
    RequestEcho,
    SpecificToolChoice,
    StreamOptions,
    SuccessResponse,
    TextContent,
    Tool,
    ToolChoiceFunction,
    ToolFunction,
    ValidationResult,
    VeniceParameters,
    VeniceParametersResponse,
    VideoContent,
)

# Note: Response models from types.api are not imported here to avoid circular imports.
# Import them directly from venice_ai.types instead.

# Create consolidated __all__ list as a static list to avoid Pylance warnings
__all__ = [
    # From base module
    "VeniceBaseModel",
    "TimestampMixin",
    # From enums module
    "ModelType",
    "APIKeyType",
    "Currency",
    "FinishReason",
    "MessageRole",
    # From headers module
    "PaginationInfo",
    "RateLimitInfo",
    "DeprecationInfo",
    "BalanceInfo",
    "ContentSafetyInfo",
    "ModelInfo",
    # From metrics module
    "UsageInfo",
    "TimingInfo",
    "SchedulerMetrics",
    "CacheStats",
    # From common module — Pagination params
    "PaginationParams",
    "DateRangeParams",
    # From common module — Consumption and pricing
    "ConsumptionLimit",
    "Balances",
    # From common module — Structured response models
    "HealthCheckResult",
    "RequestEcho",
    "ValidationResult",
    # From common module — Content components
    "TextContent",
    "ImageUrl",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    # From common module — Tools and functions
    "ToolFunction",
    "Tool",
    "ToolChoiceFunction",
    "SpecificToolChoice",
    # From common module — Response formats
    "JSONSchemaFormat",
    "JSONObjectFormat",
    # From common module — Venice-specific
    "VeniceParameters",
    "VeniceParametersResponse",
    "KNOWN_WEB_SEARCH_MODES",
    # From common module — Stream options
    "StreamOptions",
    # From common module — Generic responses
    "SuccessResponse",
    "ListResponse",
]
