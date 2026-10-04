"""
Base models and shared types for Venice AI API.

This module contains foundational Pydantic models that are used across
multiple API endpoints, including error responses and common data structures.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import ErrorDetails, PromptTokensDetails

# ============================================================================
# Base Error Models
# ============================================================================


class StandardError(BaseModel):
    """Standard error response schema"""

    error: str = Field(..., description="A description of the error")


class DetailedError(BaseModel):
    """Detailed error response with field-specific information"""

    error: str = Field(..., description="A description of the error")
    details: ErrorDetails | None = Field(
        default=None, description="Details about the incorrect input"
    )


# ============================================================================
# Common Shared Models
# ============================================================================


class TimingInfo(BaseModel):
    """Timing information for API operations"""

    inferenceDuration: float = Field(..., description="Duration of inference in milliseconds")
    inferencePreprocessingTime: float = Field(..., description="Preprocessing time in milliseconds")
    inferenceQueueTime: float = Field(..., description="Queue waiting time in milliseconds")
    total: float = Field(..., description="Total request time in milliseconds")


class UsageData(BaseModel):
    """Token usage statistics.

    ``extra='allow'`` so live usage keys this shared model doesn't yet type
    (e.g. completion_tokens_details / cache_* on non-streaming endpoints) land
    on ``model_extra`` instead of being dropped — consistent with ChatUsage.
    """

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")
    prompt_tokens_details: PromptTokensDetails | None = Field(
        default=None, description="Breakdown of tokens used in the prompt"
    )


# ============================================================================
# Base Response Models
# ============================================================================


class BaseListResponse(BaseModel):
    """Base model for list responses"""

    object: Literal["list"] = Field(..., description="Response type identifier")


class BaseSuccessResponse(BaseModel):
    """Base model for success responses"""

    success: bool = Field(..., description="Whether the operation was successful")


__all__ = [
    "StandardError",
    "DetailedError",
    "TimingInfo",
    "UsageData",
    "BaseListResponse",
    "BaseSuccessResponse",
]
