"""
Augment endpoint models for Venice AI API.

These cover the ``/augment/scrape``, ``/augment/search``, and
``/augment/text-parser`` endpoints documented at
``api-reference/endpoint/augment/``. The API is marked experimental in the
Venice docs; request and response shapes may change without notice.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.models.common import VeniceBaseModel


class AugmentScrapeRequest(BaseModel):
    """Request body for POST /augment/scrape."""

    url: str = Field(..., description="The URL to scrape (must be publicly accessible).")


class AugmentScrapeResponse(VeniceBaseModel):
    """Response body for POST /augment/scrape.

    Extends VeniceBaseModel so the documented ``X-Balance-Remaining`` (and other)
    response headers are reachable via ``.headers`` instead of being discarded.
    """

    model_config = ConfigDict(extra="allow")

    url: str = Field(..., description="The URL that was scraped.")
    content: str = Field(..., description="The scraped content in markdown format.")
    format: Literal["markdown"] = Field(
        ..., description="The format of the scraped content (always ``markdown``)."
    )


class AugmentSearchRequest(BaseModel):
    """Request body for POST /augment/search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="The search query.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of results to return (default 10, max 20).",
    )
    search_provider: Literal["brave", "google"] | None = Field(
        default=None,
        description=(
            "Search provider. ``brave`` uses Brave Search with Zero Data Retention "
            "(default). ``google`` proxies through Venice for anonymised queries."
        ),
    )


class AugmentSearchResult(BaseModel):
    """A single search result returned by /augment/search."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(..., description="Result title.")
    url: str = Field(..., description="Result URL.")
    content: str = Field(..., description="Snippet or extracted content.")
    date: str | None = Field(default=None, description="Publication date if available.")


class AugmentSearchResponse(VeniceBaseModel):
    """Response body for POST /augment/search.

    Extends VeniceBaseModel so the documented ``X-Balance-Remaining`` (and other)
    response headers are reachable via ``.headers`` instead of being discarded.
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(..., description="The query that was executed.")
    results: list[AugmentSearchResult] = Field(..., description="The search results.")


class AugmentTextParserResponse(BaseModel):
    """Response body for POST /augment/text-parser when ``response_format='json'``.

    When the caller passes ``response_format='text'`` the API returns a plain
    string instead of this object; ``client.augment.parse_text`` handles both
    shapes transparently.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="The extracted text content from the document.")
    tokens: int = Field(..., description="The token count of the extracted text.")


__all__ = [
    "AugmentScrapeRequest",
    "AugmentScrapeResponse",
    "AugmentSearchRequest",
    "AugmentSearchResult",
    "AugmentSearchResponse",
    "AugmentTextParserResponse",
]
