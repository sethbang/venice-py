"""Fixtures shared by the unit tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest

from venice_ai._pagination import Paginator, _PageResult

#: A walk past this many requests has stopped terminating.
_RUNAWAY_REQUESTS = 200
_SECONDS_PER_REQUEST = 5


@pytest.fixture(autouse=True)
def _every_walk_is_bounded(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make a :class:`Paginator` that never ends fail its test instead of hanging CI.

    A walk over ``AsyncMock`` pages never suspends, so a timeout around the test
    alone could not fire: the request count is the bound that stops it. The timeout
    covers a page that does suspend and never answers.
    """
    real_init = Paginator.__init__

    def bounded_init(
        self: Paginator[Any],
        fetch_page: Callable[[int], Awaitable[_PageResult[Any]]],
        **kwargs: Any,
    ) -> None:
        requests = [0]

        async def bounded_fetch(index: int) -> _PageResult[Any]:
            requests[0] += 1
            if requests[0] > _RUNAWAY_REQUESTS:
                pytest.fail(f"the walk did not terminate: {requests[0]} requests")
            async with asyncio.timeout(_SECONDS_PER_REQUEST):
                return await fetch_page(index)

        real_init(self, bounded_fetch, **kwargs)

    monkeypatch.setattr(Paginator, "__init__", bounded_init)
    yield
