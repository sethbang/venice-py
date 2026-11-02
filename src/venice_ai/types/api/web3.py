"""
Web3 authentication models for Venice AI API.

This module contains Pydantic models for Web3-based API key generation
and blockchain wallet authentication.
"""

from pydantic import BaseModel, ConfigDict, Field

from .api_keys import CreatedApiKey


class Web3TokenData(BaseModel):
    """Web3 token response data"""

    model_config = ConfigDict(extra="allow")

    token: str = Field(..., description="The token to sign with the wallet")


class Web3TokenResponse(BaseModel):
    """Web3 token generation response"""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(..., description="Success status")
    data: Web3TokenData = Field(..., description="Token data")


class Web3ApiKeyResponse(BaseModel):
    """Web3 API key creation response"""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(..., description="Success status")
    data: CreatedApiKey = Field(..., description="Created API key information")


__all__ = [
    "Web3TokenData",
    "Web3TokenResponse",
    "Web3ApiKeyResponse",
]
