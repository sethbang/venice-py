"""
Pagination helpers for SDK list endpoints.

The Venice API uses three different pagination shapes across its list
endpoints (page+limit, limit+offset, page+page_size) and three different
response envelopes (none, sibling ``pagination``, nested ``data.pagination``).

:class:`Paginator` hides that variation behind an async iterator interface.
Each list endpoint exposes a sibling ``iter_<noun>()`` method that returns
a ``Paginator``, so callers never need to think about page math::

    async for api_key in client.api_keys.iter_all():
        ...
    async for character in client.characters.iter_all(categories=["..."]):
        ...
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

# Default page size when the caller doesn't specify one. 100 matches the
# server-side max for most Venice list endpoints; smaller wastes round trips,
# larger gets rejected. Per-endpoint iter_* methods may override.
DEFAULT_PAGE_SIZE = 100


@dataclass(frozen=True)
class _PageResult[T]:
    """Result of a single page fetch.

    Per-endpoint adapters return one of these from their ``fetch_page``
    callback. ``has_more`` decides whether the iterator advances.
    """

    items: list[T]
    has_more: bool


class Paginator[T]:
    """Async iterator that lazily exhausts a paginated endpoint.

    Wraps a per-endpoint ``fetch_page(page_index_zero_based)`` callback that
    knows how to call the underlying ``list()``-style method and read the
    response envelope. Yields items one at a time, advancing pages until
    the callback signals ``has_more=False``.

    :param fetch_page: Callable taking a zero-based page index and
        returning :class:`_PageResult`.
    :param page_size: Items per page (forwarded to the callback via closure
        in per-endpoint adapters; surfaced here for documentation and the
        ``max_items`` cap).
    :param max_items: Optional cap on total items yielded. ``None`` (default)
        means iterate until the endpoint says there are no more pages.

    Iteration is single-shot: calling ``async for`` again on the same
    ``Paginator`` re-runs from page 0. That matches Python iterator
    semantics for things built like generators.
    """

    def __init__(
        self,
        fetch_page: Callable[[int], Awaitable[_PageResult[T]]],
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_items: int | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._page_size = page_size
        self._max_items = max_items

    @property
    def page_size(self) -> int:
        """Page size in use (mainly for tests / introspection)."""
        return self._page_size

    async def __aiter__(self) -> AsyncIterator[T]:
        page_index = 0
        yielded = 0
        while True:
            page = await self._fetch_page(page_index)
            for item in page.items:
                if self._max_items is not None and yielded >= self._max_items:
                    return
                yield item
                yielded += 1
            # A cap reached on a page boundary ends the walk without fetching
            # a page whose items would all be discarded.
            if not page.has_more or (self._max_items is not None and yielded >= self._max_items):
                return
            page_index += 1
