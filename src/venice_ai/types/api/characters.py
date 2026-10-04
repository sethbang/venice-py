"""
Character models for Venice AI API.

This module contains Pydantic models for character management and information,
including character listings and statistics.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Cross-package import: VeniceBaseModel provides shared configuration and
# custom serialization behavior used across all response models in the SDK
from ...core.models.common import VeniceBaseModel
from ..identifiers import ModelId


class CharacterStats(BaseModel):
    """Character usage statistics"""

    model_config = ConfigDict(extra="allow")

    imports: int = Field(..., description="Number of times character has been imported")
    averageRating: float | None = Field(
        default=None, description="Average user rating across all reviews (0-5)"
    )
    ratingCount: int | None = Field(default=None, description="Number of user ratings submitted")
    ratingSum: int | None = Field(default=None, description="Sum of all submitted ratings")
    userRating: int | None = Field(
        default=None, description="The authenticated caller's rating, if any"
    )


class Character(BaseModel):
    """Character information"""

    model_config = ConfigDict(extra="allow")

    slug: str = Field(..., description="Unique character identifier for API usage")
    name: str = Field(..., description="Human-readable character name")
    description: str | None = Field(
        default=None, description="Character description and background"
    )
    shareUrl: str | None = Field(default=None, description="Public sharing URL for the character")
    photoUrl: str | None = Field(default=None, description="URL of the character's photo")
    adult: bool = Field(..., description="Whether character is classified as adult content")
    webEnabled: bool = Field(..., description="Whether character is enabled for web use")
    createdAt: str = Field(..., description="ISO timestamp when character was created")
    updatedAt: str = Field(..., description="ISO timestamp when character was last updated")
    tags: list[str] = Field(..., description="Array of descriptive tags")
    stats: CharacterStats = Field(..., description="Character usage statistics")
    modelId: ModelId = Field(..., description="Model ID used for the character")
    # Fields added by the March–April 2026 Characters public API expansion.
    id: str | None = Field(default=None, description="Stable character UUID")
    author: str | None = Field(
        default=None, description="Handle of the user who authored the character"
    )
    featured: bool | None = Field(default=None, description="Whether the character is featured")
    isOwner: bool | None = Field(
        default=None,
        description=(
            "True when the authenticated caller owns the character. "
            "Present only on authenticated requests."
        ),
    )


class CharactersListResponse(VeniceBaseModel):
    """Characters list response"""

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = Field(..., description="Object type")
    data: list[Character] = Field(..., description="Array of character objects")


class CharacterResponse(VeniceBaseModel):
    """Single character response"""

    model_config = ConfigDict(extra="allow")

    object: Literal["character"] = Field(..., description="Object type")
    data: Character = Field(..., description="Character object")


class CharacterReview(BaseModel):
    """A single public review for a character."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Review UUID")
    characterId: str = Field(..., description="UUID of the reviewed character")
    createdAt: str = Field(..., description="ISO timestamp when the review was created")
    rating: int = Field(..., description="User rating on a 1–5 scale")
    message: str | None = Field(default=None, description="Review body text")
    locale: str | None = Field(
        default=None, description="BCP-47 locale code for the review message"
    )
    username: str | None = Field(default=None, description="Public username of the reviewer")
    userAvatarUrl: str | None = Field(
        default=None, description="URL of the reviewer's avatar image"
    )
    isOwner: bool | None = Field(
        default=None, description="True when the review was written by the authenticated caller"
    )


class CharacterReviewsPagination(BaseModel):
    """Pagination metadata for a character-reviews page."""

    model_config = ConfigDict(extra="allow")

    page: int = Field(..., description="1-indexed page number returned")
    pageSize: int = Field(..., description="Number of reviews per page")
    total: int = Field(..., description="Total number of reviews across all pages")
    totalPages: int = Field(..., description="Total number of pages available")


class CharacterReviewsSummary(BaseModel):
    """Aggregate review statistics for the character."""

    model_config = ConfigDict(extra="allow")

    averageRating: float = Field(..., description="Average rating across all reviews (0–5)")
    totalReviews: int = Field(..., description="Total number of reviews submitted")


class CharacterReviewsResponse(VeniceBaseModel):
    """Response from ``GET /characters/{slug}/reviews``."""

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = Field(..., description="Object type")
    data: list[CharacterReview] = Field(..., description="Array of review objects for this page")
    pagination: CharacterReviewsPagination = Field(..., description="Pagination metadata")
    summary: CharacterReviewsSummary = Field(..., description="Aggregate rating summary")


__all__ = [
    "CharacterStats",
    "Character",
    "CharactersListResponse",
    "CharacterResponse",
    "CharacterReview",
    "CharacterReviewsPagination",
    "CharacterReviewsSummary",
    "CharacterReviewsResponse",
]
