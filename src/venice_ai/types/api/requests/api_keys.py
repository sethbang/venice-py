"""
API key management request models for Venice.ai API.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...._date_validation import validate_expires_at as validate_expires_at_format
from .common import ConsumptionLimit

# ============================================================================
# API Keys Request Models
# ============================================================================


class CreateApiKeyRequest(BaseModel):
    """Create API key request"""

    apiKeyType: Literal["INFERENCE", "ADMIN"] = Field(..., description="API key type")
    description: str = Field(..., description="API key description")
    consumptionLimit: ConsumptionLimit | None = Field(None, description="Spending limits")
    modelPrivacy: Literal["ALL", "PRIVATE_TEXT", "PRIVATE_ONLY"] | None = Field(
        default=None,
        description=(
            "Which models this key may call, by privacy tier. ``ALL`` allows every model. ``PRIVATE_TEXT`` requires text and embedding models to be Private, TEE, or E2EE while other modalities may be Anonymous or Private. ``PRIVATE_ONLY`` requires every model to be Private, TEE, or E2EE; Anonymous models are rejected. Omit to let the account default apply."
        ),
    )
    limitPeriod: Literal["EPOCH", "MONTH", "LIFETIME"] | None = Field(
        default=None, description="Period over which the consumption limit resets"
    )
    expiresAt: str | None = Field(
        None, description="Expiration date (ISO format, date, or empty string)"
    )

    @field_validator("expiresAt")
    @classmethod
    def validate_expires_at(cls, v: Any) -> Any:
        if v is not None:
            # Use shared validation utility
            return validate_expires_at_format(v, "expiresAt")
        return v


class UpdateApiKeyRequest(BaseModel):
    """Request model for updating an existing API key."""

    id: str = Field(..., description="ID of the API key to update")
    description: str | None = Field(default=None, description="New description for the key")
    expiresAt: str | None = Field(default=None, description="New expiration date (ISO 8601)")
    consumptionLimit: ConsumptionLimit | None = Field(
        default=None, description="Epoch consumption limits"
    )
    modelPrivacy: Literal["ALL", "PRIVATE_TEXT", "PRIVATE_ONLY"] | None = Field(
        default=None,
        description=(
            "Which models this key may call, by privacy tier. ``ALL`` allows every "
            "model. ``PRIVATE_TEXT`` requires text and embedding models to be "
            "Private, TEE, or E2EE while other modalities may be Anonymous or "
            "Private. ``PRIVATE_ONLY`` requires every model to be Private, TEE, or "
            "E2EE; Anonymous models are rejected. Omit to let the account default apply."
        ),
    )
    limitPeriod: Literal["EPOCH", "MONTH", "LIFETIME"] | None = Field(
        default=None, description="Period over which the consumption limit resets"
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("expiresAt")
    @classmethod
    def validate_expires_at(cls, v: Any) -> Any:
        if v is not None:
            return validate_expires_at_format(v, "expiresAt")
        return v


class Web3CreateApiKeyRequest(BaseModel):
    """Web3 API key creation request"""

    apiKeyType: Literal["INFERENCE", "ADMIN"] = Field(..., description="API key type")
    address: str = Field(..., description="Wallet address")
    signature: str = Field(..., description="Signed token")
    token: str = Field(..., description="Token from generate_web3_key endpoint")

    # Optional parameters
    description: str | None = Field("Web3 API Key", description="API key description")
    consumptionLimit: ConsumptionLimit | None = Field(None, description="Spending limits")
    modelPrivacy: Literal["ALL", "PRIVATE_TEXT", "PRIVATE_ONLY"] | None = Field(
        default=None,
        description=(
            "Which models this key may call, by privacy tier. ``ALL`` allows every "
            "model. ``PRIVATE_TEXT`` requires text and embedding models to be "
            "Private, TEE, or E2EE while other modalities may be Anonymous or "
            "Private. ``PRIVATE_ONLY`` requires every model to be Private, TEE, or "
            "E2EE; Anonymous models are rejected. Omit to let the account default apply."
        ),
    )
    limitPeriod: Literal["EPOCH", "MONTH", "LIFETIME"] | None = Field(
        default=None, description="Period over which the consumption limit resets"
    )
    expiresAt: str | None = Field(None, description="Expiration date")


# ============================================================================
# Query Parameter Models
# ============================================================================


_API_TYPE_ALIASES = {"chat": "text"}


def _normalize_model_type(v: str | None) -> str | None:
    if isinstance(v, str):
        return _API_TYPE_ALIASES.get(v, v)
    return v


class ModelsQueryParams(BaseModel):
    """Query parameters for models endpoint"""

    type: str | None = Field(
        None,
        description=(
            "Filter models by type. Official API enum: asr, embedding, image, music, text, "
            "tts, upscale, inpaint, video. Also accepts 'code' and 'all'. "
            "SDK alias: 'chat' is normalized to 'text' to match the user-facing language "
            "used elsewhere in the SDK (e.g. ``client.models.resolve(type='chat')``)."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def _alias_chat_to_text(cls, v: Any) -> Any:
        return _normalize_model_type(v)


class ModelTraitsQueryParams(BaseModel):
    """Query parameters for model traits endpoint"""

    type: str | None = Field(
        "text",
        description=(
            "Filter traits by model type. Official API enum: asr, embedding, image, music, "
            "text, tts, upscale, inpaint, video. Also accepts 'code' and 'all'. "
            "SDK alias: 'chat' is normalized to 'text'."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def _alias_chat_to_text(cls, v: Any) -> Any:
        return _normalize_model_type(v)


class BillingUsageHistoryQueryParams(BaseModel):
    """Query parameters for GET /billing/usage-history.

    The first request of a walk takes the filters; a continuation request sends
    only ``cursor`` (the filters travel inside the cursor, and the server rejects
    filter parameters sent alongside one).
    """

    currency: str | None = Field(
        None, description="Filter by consumable currency (USD, DIEM, BUNDLED_CREDITS)"
    )
    cursor: str | None = Field(
        None,
        description=(
            "Opaque continuation token from a prior response's nextCursor. Sent "
            "alone; never combined with filter parameters."
        ),
    )
    startTimestamp: str | None = Field(
        None,
        description="Inclusive lower bound on entry timestamps (ISO 8601 UTC). First page only.",
    )
    endTimestamp: str | None = Field(
        None, description="Exclusive upper bound on entry timestamps (ISO 8601 UTC)."
    )
    pageSize: int | None = Field(
        None, ge=10, le=1000, description="Entries per page (10-1000, server default 1000)."
    )


class DeleteApiKeyQueryParams(BaseModel):
    """Query parameters for deleting API key"""

    id: str | None = Field(None, description="API key ID to delete")


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    "CreateApiKeyRequest",
    "UpdateApiKeyRequest",
    "Web3CreateApiKeyRequest",
    "ModelsQueryParams",
    "ModelTraitsQueryParams",
    "BillingUsageHistoryQueryParams",
    "DeleteApiKeyQueryParams",
]
