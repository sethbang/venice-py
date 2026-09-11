"""
API key management models for Venice AI API.

This module contains Pydantic models for API key creation, listing, rate limits,
and Web3 authentication functionality.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ConsumptionLimits(BaseModel):
    """API key consumption limits"""

    usd: float | None = Field(None, description="USD limit")
    diem: float | None = Field(None, description="Diem limit")
    vcu: float | None = Field(
        None,
        description=(
            "VCU (legacy Diem) limit — deprecated; use ``diem`` instead. "
            "Still accepted by the API for backwards compatibility."
        ),
    )


class TrailingSevenDaysUsage(BaseModel):
    """Usage statistics for trailing 7 days"""

    usd: str = Field(..., description="USD usage in the trailing 7 days")
    diem: str = Field(..., description="Diem usage in the trailing 7 days")
    vcu: str | None = Field(
        None,
        description=(
            "VCU (legacy Diem) usage in the trailing 7 days. Present on the wire "
            "for backwards compatibility; prefer ``diem``."
        ),
    )

    @field_validator("usd", "diem", "vcu", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str | None:
        """Convert numeric values to strings for consistency."""
        if v is None:
            return None
        return str(v)


class CurrentPeriodUsage(BaseModel):
    """Usage accrued in the current limit period (per ``limitPeriod``)."""

    usd: str | None = Field(None, description="USD usage in the current period")
    diem: str | None = Field(None, description="Diem usage in the current period")
    vcu: str | None = Field(None, description="VCU (legacy Diem) usage in the current period")

    @field_validator("usd", "diem", "vcu", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str | None:
        """Convert numeric values to strings for consistency."""
        if v is None:
            return None
        return str(v)


class ApiKeyUsage(BaseModel):
    """API key usage statistics"""

    trailingSevenDays: TrailingSevenDaysUsage = Field(
        ..., description="Usage statistics aggregated over the trailing 7-day period"
    )


class ApiKey(BaseModel):
    """API key information"""

    id: str = Field(..., description="API Key ID")
    apiKeyType: Literal["INFERENCE", "ADMIN"] = Field(..., description="API Key type")
    description: str = Field(..., description="API Key description")
    last6Chars: str = Field(..., description="Last 6 characters of the API Key")
    createdAt: str | None = Field(None, description="API Key creation date")
    expiresAt: str | None = Field(None, description="API Key expiration date")
    lastUsedAt: str | None = Field(None, description="API Key last used date")
    consumptionLimits: ConsumptionLimits | None = Field(
        None,
        description=(
            "API Key consumption limits. Optional — the docs list this as a "
            "non-required response field."
        ),
    )
    modelPrivacy: Literal["ALL", "PRIVATE_TEXT", "PRIVATE_ONLY"] | None = Field(
        default=None,
        description=(
            "Which models this key may call, by privacy tier. ``ALL`` allows every "
            "model. ``PRIVATE_TEXT`` requires text and embedding models to be "
            "Private, TEE, or E2EE while other modalities may be Anonymous or "
            "Private. ``PRIVATE_ONLY`` requires every model to be Private, TEE, or "
            "E2EE; Anonymous models are rejected."
        ),
    )
    limitPeriod: Literal["EPOCH", "MONTH", "LIFETIME"] | None = Field(
        None,
        description=(
            "Period over which the consumption limit resets. One of "
            "``EPOCH``, ``MONTH``, or ``LIFETIME``."
        ),
    )
    usage: ApiKeyUsage | None = Field(None, description="Usage statistics")
    currentPeriodUsage: CurrentPeriodUsage | None = Field(
        None, description="Usage accrued in the current limit period"
    )


class ApiKeysListResponse(BaseModel):
    """API keys list response"""

    object: Literal["list"] = Field(..., description="Object type")
    data: list[ApiKey] = Field(..., description="List of active API keys")


class CreatedApiKey(BaseModel):
    """Created API key response data"""

    id: str = Field(..., description="The API Key ID")
    apiKey: str = Field(..., description="The full API key - save immediately!")
    apiKeyType: Literal["INFERENCE", "ADMIN"] = Field(..., description="The API Key type")
    description: str = Field(..., description="The API Key description")
    expiresAt: str | None = Field(None, description="The API Key expiration date")
    consumptionLimit: ConsumptionLimits = Field(..., description="The API Key consumption limits")
    limitPeriod: Literal["EPOCH", "MONTH", "LIFETIME"] | None = Field(
        None, description="The consumption-limit period (EPOCH/MONTH/LIFETIME)."
    )


class CreateApiKeyResponse(BaseModel):
    """Create API key response"""

    success: bool = Field(..., description="Success status")
    data: CreatedApiKey = Field(..., description="Created API key information")


class DeleteApiKeyResponse(BaseModel):
    """Delete API key response"""

    success: bool = Field(..., description="Success status")


class ApiKeyDetailsResponse(BaseModel):
    """API key details response.

    The PATCH ``/api_keys`` response carries a ``success`` flag alongside
    ``data``; it is typed here (optional, not required) so it is no longer
    silently dropped. The GET ``/api_keys/{id}`` response omits ``success``, so
    the field defaults to ``None`` there.
    """

    data: ApiKey = Field(..., description="API key details")
    success: bool | None = Field(
        default=None, description="Whether the update was successful (present on PATCH responses)"
    )


class RateLimit(BaseModel):
    """Individual rate limit rule"""

    type: str = Field(..., description="Rate limit type (RPM, RPD, TPM)")
    amount: float = Field(..., description="Rate limit amount")


class ModelRateLimit(BaseModel):
    """Rate limits for a specific model"""

    apiModelId: str | None = Field(None, description="The ID of the API model")
    rateLimits: list[RateLimit] = Field(..., description="Rate limit rules for this model")


class ApiTier(BaseModel):
    """API tier information"""

    id: str = Field(..., description="The ID of the API tier")
    isCharged: bool = Field(..., description="Whether the API key is pay per use")


# Re-export from canonical location
from venice_ai.core.models.common import Balances  # noqa: E402


class RateLimitsData(BaseModel):
    """Rate limits response data"""

    accessPermitted: bool = Field(..., description="Whether API key has access to inference APIs")
    apiTier: ApiTier = Field(..., description="API tier information")
    balances: Balances = Field(..., description="Account balances")
    keyExpiration: str | None = Field(None, description="API key expiration timestamp")
    nextEpochBegins: str = Field(..., description="When the next epoch begins")
    rateLimits: list[ModelRateLimit] = Field(..., description="Rate limits for each model")


class RateLimitsResponse(BaseModel):
    """Rate limits endpoint response"""

    data: RateLimitsData = Field(..., description="Rate limits and balance information")


class RateLimitType(StrEnum):
    """The kinds of rate limit that can be exceeded.

    These are the values ``RateLimitLogEntry.rateLimitType`` carries. Compare
    against these rather than hand-writing the strings::

        if entry.rateLimitType == RateLimitType.UNSUPPORTED_FEATURE_REQUESTS:
            ...

    ``rateLimitType`` stays typed as ``str`` so a value Venice adds later still
    parses instead of failing the whole log listing — the same arrangement as
    :class:`~venice_ai.exceptions.VeniceAPIErrorCode` and ``APIError.code``.
    """

    #: Requests per day.
    RPD = "RPD"
    #: Requests per minute.
    RPM = "RPM"
    #: Tokens per minute.
    TPM = "TPM"
    #: Too many requests returned a non-success status code within the window.
    FAILED_REQUESTS = "FAILED_REQUESTS"
    #: Too many requests asked a model for a feature it does not support
    #: (e.g. vision or tool calling from a model without that capability).
    UNSUPPORTED_FEATURE_REQUESTS = "UNSUPPORTED_FEATURE_REQUESTS"


class RateLimitLogEntry(BaseModel):
    """Rate limit violation log entry"""

    apiKeyId: str = Field(..., description="API key that exceeded the limit")
    modelId: str = Field(..., description="Model being used when limit was exceeded")
    rateLimitTier: str = Field(..., description="API tier of the rate limit")
    rateLimitType: str = Field(
        ...,
        description=(
            "Type of rate limit exceeded. Compare against :class:`RateLimitType`; "
            "left as ``str`` so an unrecognised future value still parses."
        ),
    )
    timestamp: str = Field(..., description="When the rate limit was exceeded")


class RateLimitLogsResponse(BaseModel):
    """Rate limit logs response"""

    object: Literal["list"] = Field(..., description="Object type")
    data: list[RateLimitLogEntry] = Field(..., description="The last 50 rate limit logs")


__all__ = [
    "RateLimitType",
    "ConsumptionLimits",
    "TrailingSevenDaysUsage",
    "CurrentPeriodUsage",
    "ApiKeyUsage",
    "ApiKey",
    "ApiKeysListResponse",
    "CreatedApiKey",
    "CreateApiKeyResponse",
    "DeleteApiKeyResponse",
    "ApiKeyDetailsResponse",
    "RateLimit",
    "ModelRateLimit",
    "ApiTier",
    "Balances",
    "RateLimitsData",
    "RateLimitsResponse",
    "RateLimitLogEntry",
    "RateLimitLogsResponse",
]
