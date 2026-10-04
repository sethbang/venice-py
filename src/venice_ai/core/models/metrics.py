"""
Usage and metrics models for the Venice AI SDK.

Dependencies: ``base.py`` (for ``VeniceBaseModel``).

Note:
    ``UsageInfo`` references ``PromptTokensDetails`` from
    ``venice_ai.types.api.common`` via a ``TYPE_CHECKING`` guard.
    The forward reference is resolved at runtime by a ``model_rebuild()``
    call in ``common.py`` (the module that orchestrates the deferred
    import).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from .base import VeniceBaseModel

if TYPE_CHECKING:
    from venice_ai.types.api.common import PromptTokensDetails


# ============================================================================
# Usage and Metrics Models
# ============================================================================


class UsageInfo(VeniceBaseModel):
    """Base usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")
    prompt_tokens_details: PromptTokensDetails | None = Field(
        default=None, description="Breakdown of tokens used in the prompt"
    )


class TimingInfo(VeniceBaseModel):
    """Performance timing information."""

    inference_duration: float = Field(..., description="Duration of inference in milliseconds")
    inference_preprocessing_time: float = Field(
        ..., description="Duration of preprocessing in milliseconds"
    )
    inference_queue_time: float = Field(..., description="Duration of queueing in milliseconds")
    total: float = Field(..., description="Total duration of the request in milliseconds")


class SchedulerMetrics(VeniceBaseModel):
    """Scheduler performance metrics."""

    active_queues: int = Field(..., description="Number of active queues")
    total_queue_depth: int = Field(..., description="Total depth across all queues")
    avg_queue_depth: float = Field(..., description="Average queue depth")
    failed_request_count: int = Field(..., description="Number of failed requests")


class CacheStats(VeniceBaseModel):
    """Cache statistics and performance metrics."""

    size: int = Field(..., description="Current cache size")
    max_size: int | None = Field(..., description="Maximum cache size")
    hits: int = Field(..., description="Number of cache hits")
    misses: int = Field(..., description="Number of cache misses")
    hit_ratio: float = Field(..., description="Cache hit ratio (0.0 to 1.0)")
    evictions: int = Field(..., description="Number of cache evictions")
    cleanups: int = Field(..., description="Number of cleanup operations")
    ttl_seconds: float = Field(..., description="Time-to-live in seconds")
    cleanup_interval: float = Field(..., description="Cleanup interval in seconds")
    running: bool = Field(..., description="Whether cache cleanup is running")


__all__ = [
    "CacheStats",
    "SchedulerMetrics",
    "TimingInfo",
    "UsageInfo",
]
