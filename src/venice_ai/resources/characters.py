"""
Venice AI Characters Resource Module.

This module provides access to Venice AI's character system, which enables the use of
pre-configured AI personalities and specialized assistants in chat interactions. Characters
are carefully crafted AI personas with distinct personalities, knowledge bases, and
communication styles that can enhance user interactions by providing more engaging
and contextually appropriate responses.

The character system allows developers to leverage expertly designed AI personalities
without needing to craft complex system prompts or persona definitions. Each character
maintains consistent behavior patterns and specialized knowledge domains.

Key Features:
    - Access to curated AI character personalities
    - Specialized assistant characters for various domains
    - Consistent character behavior and knowledge
    - Easy integration with chat completion requests

Classes:
    Characters: Asynchronous resource for character discovery and management

Note:
    The Characters API is currently in Preview and may evolve in future releases.
    Breaking changes will be communicated through version updates and migration guides.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from .._pagination import DEFAULT_PAGE_SIZE, Paginator, _PageResult
from .._resource import APIResource
from ..types.api import (
    Character,
    CharacterResponse,
    CharacterReview,
    CharacterReviewsResponse,
    CharactersListResponse,
)

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401


CharacterSortBy = Literal[
    "featured",
    "highestRating",
    "highlyRated",
    "highlyRatedAndRecent",
    "imports",
    "mostRecent",
    "ratingCount",
]
CharacterSortOrder = Literal["asc", "desc"]


class Characters(APIResource["VeniceClient"]):
    """
    Asynchronous resource for discovering and managing AI character definitions.

    This class provides access to Venice AI's character ecosystem, allowing developers
    to discover available AI personalities and specialized assistants that can be
    integrated into chat applications. Characters offer pre-configured personas with
    distinct personalities, expertise areas, and communication styles.

    Characters can be referenced in chat completion requests to provide more engaging
    and contextually appropriate AI interactions without requiring custom system
    prompt engineering.

    Key Capabilities:
        - Discover available character personalities
        - Access character metadata and descriptions
        - Integrate characters into chat workflows
        - Preview character capabilities and specializations

    Args:
        client: The Venice AI client instance for making authenticated API requests.

    Warning:
        The Characters API is currently in Preview. Features and interfaces may
        change in future releases. Always check the latest documentation for updates.

    Example:
        Basic character discovery:

        .. code-block:: python

            async with VeniceClient() as client:
                # Discover available characters
                characters = await client.characters.list()

                # Browse character options
                for character in characters.data:
                    print(f"{character.name}: {character.description}")

                # Use character in chat (example)
                model = await client.models.resolve_chat()
                chat_response = await client.chat.completions.create(
                    model=model,
                    venice_parameters={"character_slug": character.slug},
                    messages=[{"role": "user", "content": "Hello!"}]
                )
    """

    async def list(
        self,
        *,
        categories: list[str] | None = None,
        is_adult: bool | None = None,
        is_pro: bool | None = None,
        is_web_enabled: bool | None = None,
        limit: int | None = None,
        model_id: list[str] | None = None,
        offset: int | None = None,
        search: str | None = None,
        sort_by: CharacterSortBy | None = None,
        sort_order: CharacterSortOrder | None = None,
        tags: list[str] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> CharactersListResponse:
        """Retrieve a paginated list of available AI characters.

        All filter parameters are optional; pass only what you want to constrain.
        List-valued filters (``categories``, ``model_id``, ``tags``) are sent as
        comma-separated query values, which the API accepts interchangeably with
        repeated query parameters.

        Args:
            categories: Filter by category names.
            is_adult: Filter by adult content flag.
            is_pro: Filter to only show characters using pro models.
            is_web_enabled: Filter to only show web-enabled characters.
            limit: Number of characters to return (max 100).
            model_id: Filter by model ID(s).
            offset: Number of characters to skip for pagination.
            search: Free-text search across name, description, and tags.
                Hashtag search is supported.
            sort_by: Sort results using a supported character-discovery mode.
            sort_order: Sort order applied to the selected ``sort_by`` mode.
            tags: Filter by tag names.
            extra_headers: Additional HTTP headers for the request.
            extra_query: Additional query parameters, merged after the typed filters.
            timeout: Request timeout in seconds. Uses client default if not specified.

        Returns:
            Character catalog with detailed metadata including names, descriptions,
            slugs for integration, and capability information.

        Raises:
            APIError: If the request fails due to server error or invalid parameters.
            AuthenticationError: If the API key is invalid or expired.
            APIConnectionError: If unable to connect to the Venice AI service.

        Example:
            Filtered discovery::

                result = await client.characters.list(
                    is_web_enabled=True,
                    sort_by="highlyRated",
                    limit=20,
                )
        """
        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        params: dict[str, Any] = {}
        if categories is not None:
            params["categories"] = ",".join(categories)
        if is_adult is not None:
            params["isAdult"] = "true" if is_adult else "false"
        if is_pro is not None:
            params["isPro"] = "true" if is_pro else "false"
        if is_web_enabled is not None:
            params["isWebEnabled"] = "true" if is_web_enabled else "false"
        if limit is not None:
            params["limit"] = limit
        if model_id is not None:
            params["modelId"] = ",".join(model_id)
        if offset is not None:
            params["offset"] = offset
        if search is not None:
            params["search"] = search
        if sort_by is not None:
            params["sortBy"] = sort_by
        if sort_order is not None:
            params["sortOrder"] = sort_order
        if tags is not None:
            params["tags"] = ",".join(tags)
        if extra_query:
            params.update(extra_query)

        result = await self._client.get(
            "characters",
            headers=headers if headers else None,
            params=params if params else None,
            timeout=timeout,
            cast_to=CharactersListResponse,
        )
        return result

    def iter_all(
        self,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_items: int | None = None,
        categories: Sequence[str] | None = None,
        is_adult: bool | None = None,
        is_pro: bool | None = None,
        is_web_enabled: bool | None = None,
        model_id: Sequence[str] | None = None,
        search: str | None = None,
        sort_by: CharacterSortBy | None = None,
        sort_order: CharacterSortOrder | None = None,
        tags: Sequence[str] | None = None,
    ) -> Paginator[Character]:
        """Lazily iterate every character matching the filters.

        Wraps :meth:`list` for unbounded enumeration. Termination: this
        endpoint has no pagination envelope, so the iterator stops on the
        first short page (``len(items) < page_size``), or on a page that adds
        no character the walk has not yet yielded. Each slug is yielded once.

        Filter kwargs mirror :meth:`list`. Each per-page call passes the
        same filters with growing ``offset``.

        :param page_size: Server page size (default 100, max 100).
        :param max_items: Optional cap on total items yielded.

        Example::

            async for ch in client.characters.iter_all(is_web_enabled=True):
                print(ch.slug, ch.name)
        """
        filters: dict[str, Any] = {
            "categories": categories,
            "is_adult": is_adult,
            "is_pro": is_pro,
            "is_web_enabled": is_web_enabled,
            "model_id": model_id,
            "search": search,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "tags": tags,
        }

        async def _fetch_page(page_index: int) -> _PageResult[Character]:
            response = await self.list(
                limit=page_size,
                offset=page_index * page_size,
                **{k: v for k, v in filters.items() if v is not None},
            )
            items = list(response.data)
            return _PageResult(items=items, has_more=len(items) == page_size)

        return Paginator(
            _fetch_page, page_size=page_size, max_items=max_items, key=lambda ch: ch.slug
        )

    async def get(
        self,
        slug: str,
        *,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> CharacterResponse:
        """
        Retrieve detailed information for a specific AI character by slug.

        Fetches complete character definition including metadata, descriptions,
        URLs, model information, and all other character details from the Venice AI
        platform. This experimental endpoint provides access to individual character
        data for integration and display purposes.

        Args:
            slug: The unique slug identifier for the character to retrieve.
            extra_headers: Additional HTTP headers for the request (e.g., custom auth).
            extra_query: Additional query parameters for customization.
            timeout: Request timeout in seconds. Uses client default if not specified.

        Returns:
            Complete character information including name, description, URLs,
            model ID, statistics, and all metadata for the specified character.

        Raises:
            APIError: If the request fails due to server error or invalid parameters.
            AuthenticationError: If the API key is invalid or expired.
            APIConnectionError: If unable to connect to the Venice AI service.
            NotFoundError: If the character with the specified slug is not found.

        Example:
            Get specific character details:

            .. code-block:: python

                async with VeniceClient() as client:
                    # Get character by slug
                    character = await client.characters.get("alan-watts")

                    print(f"Character: {character.data.name}")
                    print(f"Description: {character.data.description}")
                    print(f"Model: {character.data.modelId}")
                    print(f"Photo URL: {character.data.photoUrl}")
                    print(f"Share URL: {character.data.shareUrl}")

                    # Use in chat
                    response = await client.chat.completions.create(
                        model=character.data.modelId,
                        venice_parameters={"character_slug": character.data.slug},
                        messages=[{"role": "user", "content": "Hello!"}]
                    )

        Warning:
            This is an experimental endpoint and may be subject to change.
        """
        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        params = {}
        if extra_query:
            params.update(extra_query)

        result = await self._client.get(
            f"characters/{slug}",
            headers=headers if headers else None,
            params=params if params else None,
            timeout=timeout,
            cast_to=CharacterResponse,
        )
        return result

    async def reviews(
        self,
        slug: str,
        *,
        page: int | None = None,
        page_size: int | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> CharacterReviewsResponse:
        """Retrieve paginated public reviews for a single character.

        Wraps ``GET /api/v1/characters/{slug}/reviews``.

        Args:
            slug: Slug of the character whose reviews should be retrieved.
            page: 1-indexed page number (server default: 1).
            page_size: Number of reviews per page (server default 20, max 100).
            extra_headers: Additional HTTP headers for the request.
            extra_query: Additional query parameters.
            timeout: Request timeout in seconds. Uses client default if not specified.

        Returns:
            Paginated review list with pagination metadata and aggregate summary.

        Raises:
            APIError: If the request fails due to server error or invalid parameters.
            AuthenticationError: If the API key is invalid or expired.
            NotFoundError: If the character with the specified slug is not found.

        Example::

            reviews = await client.characters.reviews("alan-watts", page=1, page_size=50)
            print(f"Average rating: {reviews.summary.averageRating}")
            for review in reviews.data:
                print(f"{review.rating}/5 — {review.message}")

        Warning:
            This is an experimental endpoint and may be subject to change.
        """
        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        params: dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["pageSize"] = page_size
        if extra_query:
            params.update(extra_query)

        return await self._client.get(
            f"characters/{slug}/reviews",
            headers=headers if headers else None,
            params=params if params else None,
            timeout=timeout,
            cast_to=CharacterReviewsResponse,
        )

    def iter_reviews(
        self,
        slug: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_items: int | None = None,
    ) -> Paginator[CharacterReview]:
        """Lazily iterate every review for *slug*.

        Wraps :meth:`reviews` for unbounded enumeration. Termination uses
        the response's ``pagination.totalPages`` field against the page number
        requested, so iteration stops when the server says there's nothing
        left, or on a page that adds no review the walk has not yet yielded.

        :param slug: Character slug whose reviews to iterate.
        :param page_size: Server page size (default 100, max 100).
        :param max_items: Optional cap on total items yielded.

        Example::

            async for review in client.characters.iter_reviews("alan-watts"):
                print(review.rating, review.message)
        """

        async def _fetch_page(page_index: int) -> _PageResult[CharacterReview]:
            response = await self.reviews(slug, page=page_index + 1, page_size=page_size)
            items = list(response.data)
            has_more = page_index + 1 < response.pagination.totalPages
            return _PageResult(items=items, has_more=has_more)

        return Paginator(
            _fetch_page, page_size=page_size, max_items=max_items, key=lambda review: review.id
        )
