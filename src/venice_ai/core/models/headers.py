"""
Header-derived models for the Venice AI SDK.

Models in this module represent structured information extracted from HTTP
response headers (rate limits, deprecation notices, balances, content
safety, model metadata, and pagination).

Dependencies: ``base.py`` (for ``VeniceBaseModel``).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from .base import VeniceBaseModel

# ============================================================================
# Pagination
# ============================================================================


class PaginationInfo(VeniceBaseModel):
    """Pagination information for list responses."""

    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, le=500, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")


# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimitInfo(VeniceBaseModel):
    """Rate limit information from headers."""

    limit_requests: int | None = Field(
        default=None, description="Total requests allowed in current window"
    )
    remaining_requests: int | None = Field(
        default=None, description="Requests remaining in current window"
    )
    reset_requests: datetime | None = Field(default=None, description="When request limits reset")
    limit_tokens: int | None = Field(
        default=None, description="Total tokens allowed in current window"
    )
    remaining_tokens: int | None = Field(
        default=None, description="Tokens remaining in current window"
    )
    reset_tokens: float | None = Field(
        default=None,
        description=(
            "Absolute Unix timestamp (seconds) when the token rate limit resets "
            "(normalized from the ms-epoch x-ratelimit-reset-tokens header)."
        ),
    )
    type: str | None = Field(
        default=None, description="Rate limit type: 'user', 'api_key', or 'global'"
    )


# ============================================================================
# Deprecation
# ============================================================================


class DeprecationInfo(VeniceBaseModel):
    """Model deprecation information from response headers."""

    warning: str | None = Field(default=None, description="Deprecation warning message")
    date: datetime | None = Field(default=None, description="Date when model will be deprecated")

    @property
    def is_deprecated(self) -> bool:
        """Check if the model has deprecation information."""
        return self.warning is not None or self.date is not None


# ============================================================================
# Balance
# ============================================================================


class BalanceInfo(VeniceBaseModel):
    """Spendable balance of the calling API key, from response headers.

    Read from ``x-venice-balance-usd`` and ``x-venice-balance-diem``, which
    report the value before the request was processed. Like ``balances`` on
    ``GET /api_keys/rate_limits``, this is what the key can still spend: the
    lesser of the account's balance and what remains under the key's
    consumption limit. It is not the account balance; for that use
    ``client.billing.get_balance()``.
    """

    diem: float | None = Field(
        default=None,
        description=(
            "DIEM the calling key could spend in the current epoch before this "
            "request (1 DIEM is $1 of the staking allowance). ``None`` when the "
            "header is absent. Whether it is capped by the key's DIEM consumption "
            "limit has not been verified"
        ),
    )
    usd: float | None = Field(
        default=None,
        description=(
            "USD the calling key could spend before this request: the lesser of "
            "the account's USD balance and the key's remaining consumption limit"
        ),
    )


# ============================================================================
# Content Safety
# ============================================================================


class ContentSafetyInfo(VeniceBaseModel):
    """Content safety information from response headers."""

    model_config = ConfigDict(extra="allow")

    is_blurred: bool | None = Field(default=None, description="Whether generated image was blurred")
    is_content_violation: bool | None = Field(
        default=None, description="Whether content violates policies"
    )
    is_adult_model_content_violation: bool | None = Field(
        default=None,
        description="Whether content violates adult model policies",
    )
    contains_minor: bool | None = Field(default=None, description="Whether image contains minors")


# ============================================================================
# Model Info
# ============================================================================


class ModelInfo(VeniceBaseModel):
    """Model information from response headers."""

    model_config = ConfigDict(extra="allow")

    model_id: str | None = Field(default=None, description="Model ID used for the request")
    model_name: str | None = Field(default=None, description="Friendly model name")
    model_router: str | None = Field(
        default=None, description="Router/backend that handled inference"
    )
    deprecation_warning: str | None = Field(default=None, description="Deprecation warning message")
    deprecation_date: str | None = Field(default=None, description="Deprecation date (ISO 8601)")


__all__ = [
    "BalanceInfo",
    "ContentSafetyInfo",
    "DeprecationInfo",
    "ModelInfo",
    "PaginationInfo",
    "RateLimitInfo",
]
