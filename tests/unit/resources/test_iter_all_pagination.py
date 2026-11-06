"""Unit tests for iter_* pagination helpers across resources.

Tests the per-endpoint Paginator wiring in:

- ``client.api_keys.iter_all``
- ``client.characters.iter_all``
- ``client.characters.iter_reviews``
- ``client.x402.iter_transactions``
- ``client.billing.iter_usage_history``

Each test mocks the underlying ``list``/``reviews``/``transactions``/
``get_usage_history`` method to return a sequence of pages, then verifies the
iterator yields the right items and terminates correctly.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai._pagination import MAX_IDLE_PAGES, Paginator, _PageResult
from venice_ai.resources.api_keys import ApiKeys
from venice_ai.resources.billing import Billing
from venice_ai.resources.characters import Characters
from venice_ai.resources.x402 import X402
from venice_ai.types.api import ApiKey
from venice_ai.types.api.billing import BillingUsageHistoryCsvPage
from venice_ai.types.api.characters import (
    CharacterReview,
    CharacterReviewsResponse,
    CharactersListResponse,
)
from venice_ai.types.api.x402 import X402TransactionsResponse

# ---------------------------------------------------------------------------
# Helpers — fabricate response shapes for each endpoint
# ---------------------------------------------------------------------------


T = TypeVar("T")

# A walk past this many items has stopped terminating.
RUNAWAY = 50


async def _capped(walk: AsyncIterator[T] | Paginator[T]) -> AsyncIterator[T]:
    """Fail a walk that does not end, rather than hang the test run."""
    count = 0
    async for item in walk:
        count += 1
        if count > RUNAWAY:
            pytest.fail("the walk did not terminate")
        yield item


def _review(review_id: str, rating: int) -> CharacterReview:
    return CharacterReview.model_validate(
        {
            "id": review_id,
            "characterId": "ch-1",
            "rating": rating,
            "message": f"r{rating}",
            "createdAt": "2025-01-01T00:00:00Z",
        }
    )


def _make_api_key(idx: int) -> ApiKey:
    """Build a minimal ApiKey for paginator tests."""
    return ApiKey.model_validate(
        {
            "id": f"key_{idx}",
            "apiKey": f"sk-{idx}",
            "apiKeyType": "INFERENCE",
            "description": f"Key {idx}",
            "createdAt": "2025-01-01T00:00:00Z",
            "expiresAt": None,
            "lastUsedAt": None,
            "last6Chars": f"{idx:06d}",
            "consumptionLimits": {"usd": None, "diem": None},
            "usage": {"trailingSevenDays": {"usd": "0", "diem": "0"}},
        }
    )


# ---------------------------------------------------------------------------
# api_keys.iter_all: the endpoint does not page (loopback coverage in
# test_api_keys_iter_all_loopback.py)
# ---------------------------------------------------------------------------


class TestApiKeysIterAll:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        c = MagicMock()
        c.get = AsyncMock()
        return c

    @pytest.fixture
    def resource(self, mock_client: MagicMock) -> ApiKeys:
        return ApiKeys(mock_client)

    @pytest.mark.asyncio
    async def test_returns_paginator_instance(self, resource: ApiKeys) -> None:
        assert isinstance(resource.iter_all(), Paginator)

    @pytest.mark.asyncio
    async def test_bare_list_response_is_one_walk(
        self, resource: ApiKeys, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = [
            _make_api_key(1).model_dump(by_alias=True),
            _make_api_key(2).model_dump(by_alias=True),
        ]
        ids = [k.id async for k in resource.iter_all(page_size=2)]
        assert ids == ["key_1", "key_2"]
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_sends_no_paging_parameters(
        self, resource: ApiKeys, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = []
        async for _ in resource.iter_all(page_size=10):
            pass
        assert mock_client.get.call_args.kwargs.get("params") is None

    @pytest.mark.asyncio
    async def test_max_items_caps_iteration(
        self, resource: ApiKeys, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = [
            _make_api_key(i).model_dump(by_alias=True) for i in range(1, 11)
        ]
        ids = [k.id async for k in resource.iter_all(page_size=10, max_items=3)]
        assert ids == ["key_1", "key_2", "key_3"]

    @pytest.mark.asyncio
    async def test_immediate_empty_page_terminates(
        self, resource: ApiKeys, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = []
        ids = [k.id async for k in resource.iter_all(page_size=5)]
        assert ids == []

    @pytest.mark.asyncio
    async def test_paginator_page_size_property(self, resource: ApiKeys) -> None:
        # Sanity: configured page_size is reachable for tests / introspection
        assert resource.iter_all(page_size=42).page_size == 42


# ---------------------------------------------------------------------------
# characters.iter_all — limit/offset, wrapper, terminate on short page
# ---------------------------------------------------------------------------


class TestCharactersIterAll:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        c = MagicMock()
        c.get = AsyncMock()
        return c

    @pytest.fixture
    def resource(self, mock_client: MagicMock) -> Characters:
        return Characters(mock_client)

    def _page(self, ids: list[str]) -> CharactersListResponse:
        return CharactersListResponse.model_validate(
            {
                "object": "list",
                "data": [
                    {
                        "slug": s,
                        "name": s.title(),
                        "adult": False,
                        "webEnabled": True,
                        "createdAt": "2025-01-01T00:00:00Z",
                        "updatedAt": "2025-01-02T00:00:00Z",
                        "tags": [],
                        "stats": {"imports": 0},
                        "modelId": "test-model",
                    }
                    for s in ids
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_exhausts_pages(self, resource: Characters, mock_client: MagicMock) -> None:
        mock_client.get.side_effect = [
            self._page(["alice", "bob"]),
            self._page(["carol", "dan"]),
            self._page(["eve"]),  # short → done
        ]
        slugs = [c.slug async for c in resource.iter_all(page_size=2)]
        assert slugs == ["alice", "bob", "carol", "dan", "eve"]

    @pytest.mark.asyncio
    async def test_offset_advances_per_page(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        mock_client.get.side_effect = [
            self._page(["a", "b"]),
            self._page(["c"]),  # short → done
        ]
        async for _ in resource.iter_all(page_size=2):
            pass
        # First page: offset=0, limit=2
        # Second page: offset=2, limit=2
        first_params = mock_client.get.call_args_list[0].kwargs["params"]
        second_params = mock_client.get.call_args_list[1].kwargs["params"]
        assert first_params["offset"] == 0
        assert first_params["limit"] == 2
        assert second_params["offset"] == 2

    @pytest.mark.asyncio
    async def test_filters_forwarded_to_each_page(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = self._page([])  # immediate empty
        async for _ in resource.iter_all(page_size=10, is_web_enabled=True, search="ai"):
            pass
        params = mock_client.get.call_args_list[0].kwargs["params"]
        assert params["isWebEnabled"] == "true"
        assert params["search"] == "ai"

    @pytest.mark.asyncio
    async def test_server_ignoring_offset_ends_the_walk(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = self._page(["a", "b"])
        slugs = [c.slug async for c in _capped(resource.iter_all(page_size=2))]
        assert slugs == ["a", "b"]
        # The first page, then the bound on pages that add nothing.
        assert mock_client.get.call_count == 1 + MAX_IDLE_PAGES

    @pytest.mark.asyncio
    async def test_repeat_across_a_page_boundary_is_yielded_once(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        mock_client.get.side_effect = [
            self._page(["a", "b"]),
            self._page(["b", "c"]),
            self._page([]),
        ]
        slugs = [c.slug async for c in resource.iter_all(page_size=2)]
        assert slugs == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# characters.iter_reviews — page-based, totalPages envelope termination
# ---------------------------------------------------------------------------


class TestCharactersIterReviews:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        c = MagicMock()
        c.get = AsyncMock()
        return c

    @pytest.fixture
    def resource(self, mock_client: MagicMock) -> Characters:
        return Characters(mock_client)

    def _page(self, ratings: list[int], page: int, total_pages: int) -> CharacterReviewsResponse:
        return CharacterReviewsResponse.model_validate(
            {
                "object": "list",
                "data": [
                    {
                        "id": f"rev-{page}-{i}",
                        "characterId": "ch-1",
                        "rating": r,
                        "message": f"r{r}",
                        "createdAt": "2025-01-01T00:00:00Z",
                    }
                    for i, r in enumerate(ratings)
                ],
                "pagination": {
                    "page": page,
                    "pageSize": len(ratings) if ratings else 10,
                    "total": total_pages * (len(ratings) or 1),
                    "totalPages": total_pages,
                },
                "summary": {
                    "averageRating": 4.2,
                    "totalReviews": total_pages * (len(ratings) or 1),
                },
            }
        )

    @pytest.mark.asyncio
    async def test_terminates_at_total_pages(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        mock_client.get.side_effect = [
            self._page([5, 4], page=1, total_pages=3),
            self._page([3, 2], page=2, total_pages=3),
            self._page([1, 1], page=3, total_pages=3),  # last page
        ]
        ratings = [r.rating async for r in resource.iter_reviews("alan-watts", page_size=2)]
        assert ratings == [5, 4, 3, 2, 1, 1]
        # Should not have made a 4th call
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_uses_page_size_camelcase_wire(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        mock_client.get.return_value = self._page([5], page=1, total_pages=1)
        async for _ in resource.iter_reviews("alan-watts", page_size=7):
            pass
        params = mock_client.get.call_args_list[0].kwargs["params"]
        assert params["pageSize"] == 7  # camelCase on the wire
        assert params["page"] == 1

    @pytest.mark.asyncio
    async def test_server_ignoring_page_ends_the_walk(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        # Echoing page 1 of 2 every time must not keep the walk alive.
        mock_client.get.return_value = self._page([5, 4], page=1, total_pages=2)
        ratings = [r.rating async for r in _capped(resource.iter_reviews("slug", page_size=2))]
        assert ratings == [5, 4]
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_echoed_page_does_not_extend_the_walk(
        self, resource: Characters, mock_client: MagicMock
    ) -> None:
        # The server's echo says page 1 of 2 on the second request: two were promised.
        mock_client.get.side_effect = [
            self._page([5], page=1, total_pages=2),
            self._page([3], page=1, total_pages=2).model_copy(
                update={"data": [_review("rev-other", 3)]}
            ),
        ]
        ratings = [r.rating async for r in _capped(resource.iter_reviews("slug", page_size=1))]
        assert ratings == [5, 3]
        assert mock_client.get.call_count == 2


# ---------------------------------------------------------------------------
# x402.iter_transactions — limit/offset, hasMore envelope termination
# ---------------------------------------------------------------------------


class TestX402IterTransactions:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        c = MagicMock()
        c.get = AsyncMock()
        return c

    @pytest.fixture
    def resource(self, mock_client: MagicMock) -> X402:
        return X402(mock_client)

    @pytest.fixture
    def fake_auth(self) -> MagicMock:
        # X402Auth has external deps (eth-account / siwe). The resource only
        # reads wallet_address and hands the auth to the client, which does the
        # signing, so a bare mock is enough here.
        auth = MagicMock()
        auth.wallet_address = "0xabc"
        return auth

    def _page(self, ids: list[str], has_more: bool) -> X402TransactionsResponse:
        return X402TransactionsResponse.model_validate(
            {
                "success": True,
                "data": {
                    "walletAddress": "0xabc",
                    "currentBalance": 100.0,
                    "transactions": [
                        {
                            "id": tid,
                            "amount": -1.0,
                            "balanceAfter": 99.0,
                            "type": "USAGE",
                            "createdAt": "2025-01-01T00:00:00Z",
                        }
                        for tid in ids
                    ],
                    "pagination": {
                        "limit": 100,
                        "offset": 0,
                        "hasMore": has_more,
                    },
                },
            }
        )

    @pytest.mark.asyncio
    async def test_terminates_when_hasmore_false(
        self,
        resource: X402,
        mock_client: MagicMock,
        fake_auth: MagicMock,
    ) -> None:
        mock_client.get.side_effect = [
            self._page(["t1", "t2"], has_more=True),
            self._page(["t3", "t4"], has_more=True),
            self._page(["t5"], has_more=False),
        ]
        ids = [t.id async for t in resource.iter_transactions(auth=fake_auth, page_size=2)]
        assert ids == ["t1", "t2", "t3", "t4", "t5"]
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_server_ignoring_offset_ends_the_walk(
        self,
        resource: X402,
        mock_client: MagicMock,
        fake_auth: MagicMock,
    ) -> None:
        mock_client.get.return_value = self._page(["t1", "t2"], has_more=True)
        ids = [t.id async for t in _capped(resource.iter_transactions(auth=fake_auth, page_size=2))]
        assert ids == ["t1", "t2"]
        assert mock_client.get.call_count == 1 + MAX_IDLE_PAGES

    @pytest.mark.asyncio
    async def test_empty_page_claiming_more_ends_the_walk(
        self,
        resource: X402,
        mock_client: MagicMock,
        fake_auth: MagicMock,
    ) -> None:
        mock_client.get.return_value = self._page([], has_more=True)
        ids = [t.id async for t in _capped(resource.iter_transactions(auth=fake_auth, page_size=2))]
        assert ids == []
        assert mock_client.get.call_count == MAX_IDLE_PAGES


# ---------------------------------------------------------------------------
# billing.iter_usage_history — cursor walk, nextCursor termination
# ---------------------------------------------------------------------------


class TestBillingIterUsageHistory:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        c = MagicMock()
        c.get = AsyncMock()
        c._request = AsyncMock()
        return c

    @pytest.fixture
    def resource(self, mock_client: MagicMock) -> Billing:
        return Billing(mock_client)

    def _page(self, amounts: list[float], next_cursor: str | None) -> dict:
        return {
            "data": [
                {
                    "amount": a,
                    "currency": "USD",
                    "sku": "test",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "units": 1,
                    "pricePerUnitUsd": a,
                    "notes": "",
                }
                for a in amounts
            ],
            "nextCursor": next_cursor,
        }

    @pytest.mark.asyncio
    async def test_terminates_when_cursor_null(
        self, resource: Billing, mock_client: MagicMock
    ) -> None:
        # billing uses _request directly with raw_response=False for JSON
        mock_client._request.side_effect = [
            self._page([0.10, 0.20], next_cursor="CURSOR_2"),
            self._page([0.30, 0.40], next_cursor=None),
        ]
        amounts = [e.amount async for e in resource.iter_usage_history(page_size=10)]
        assert amounts == [0.10, 0.20, 0.30, 0.40]
        assert mock_client._request.call_count == 2

    @pytest.mark.asyncio
    async def test_first_page_filters_then_cursor_only(
        self, resource: Billing, mock_client: MagicMock
    ) -> None:
        mock_client._request.side_effect = [
            self._page([0.10], next_cursor="CURSOR_2"),
            self._page([0.20], next_cursor=None),
        ]
        amounts = [
            e.amount
            async for e in resource.iter_usage_history(
                page_size=10,
                currency="USD",
                startTimestamp="2025-01-01T00:00:00Z",
                endTimestamp="2025-01-31T23:59:59Z",
            )
        ]
        assert amounts == [0.10, 0.20]

        # First page carries the filters (BillingUsageHistoryQueryParams.model_dump).
        first_params = mock_client._request.call_args_list[0].kwargs["params"]
        assert first_params == {
            "currency": "USD",
            "startTimestamp": "2025-01-01T00:00:00Z",
            "endTimestamp": "2025-01-31T23:59:59Z",
            "pageSize": 10,
        }
        # Continuation carries ONLY the cursor.
        second_params = mock_client._request.call_args_list[1].kwargs["params"]
        assert second_params == {"cursor": "CURSOR_2"}

    @pytest.mark.asyncio
    async def test_a_repeated_cursor_ends_the_walk(
        self, resource: Billing, mock_client: MagicMock
    ) -> None:
        mock_client._request.side_effect = [
            self._page([0.10], next_cursor="CURSOR_2"),
            self._page([0.20], next_cursor="CURSOR_2"),
            self._page([0.30], next_cursor="CURSOR_2"),
        ]
        amounts = [e.amount async for e in _capped(resource.iter_usage_history(page_size=10))]
        assert amounts == [0.10, 0.20]
        assert mock_client._request.call_count == 2

    @pytest.mark.asyncio
    async def test_a_cursor_cycle_ends_the_walk(
        self, resource: Billing, mock_client: MagicMock
    ) -> None:
        mock_client._request.side_effect = [
            self._page([0.10], next_cursor="A"),
            self._page([0.20], next_cursor="B"),
            self._page([0.30], next_cursor="A"),
            self._page([0.40], next_cursor="B"),
        ]
        amounts = [e.amount async for e in _capped(resource.iter_usage_history(page_size=10))]
        assert amounts == [0.10, 0.20, 0.30]

    @pytest.mark.asyncio
    async def test_csv_walk_stops_on_a_repeated_cursor(
        self, resource: Billing, mock_client: MagicMock
    ) -> None:
        page = BillingUsageHistoryCsvPage(content=b"a,b\n", nextCursor="CURSOR_2")
        resource.get_usage_history = AsyncMock(return_value=page)  # type: ignore[method-assign]
        pages = [p async for p in _capped(resource.iter_usage_history_csv())]
        assert len(pages) == 2
        assert resource.get_usage_history.await_count == 2


# ---------------------------------------------------------------------------
# Paginator
# ---------------------------------------------------------------------------


class TestPaginatorKey:
    @staticmethod
    def _fetcher(pages: list[list[int]], has_more: bool = True) -> AsyncMock:
        async def fetch(index: int) -> _PageResult[int]:
            return _PageResult(items=pages[min(index, len(pages) - 1)], has_more=has_more)

        return AsyncMock(side_effect=fetch)

    @pytest.mark.asyncio
    async def test_keyed_walk_does_not_count_repeats_against_max_items(self) -> None:
        fetch = self._fetcher([[1, 2], [2, 3], [3, 4]])
        paginator = Paginator(fetch, key=lambda n: n, max_items=3)
        assert [n async for n in paginator] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_keyed_walk_stops_on_a_page_with_nothing_new(self) -> None:
        fetch = self._fetcher([[1, 2]])
        paginator = Paginator(fetch, key=lambda n: n)
        assert [n async for n in _capped(paginator)] == [1, 2]
        assert fetch.await_count == 1 + MAX_IDLE_PAGES

    @pytest.mark.asyncio
    async def test_second_walk_starts_with_nothing_seen(self) -> None:
        fetch = self._fetcher([[1, 2]], has_more=False)
        paginator = Paginator(fetch, key=lambda n: n)
        assert [n async for n in paginator] == [1, 2]
        assert [n async for n in paginator] == [1, 2]

    @pytest.mark.asyncio
    async def test_unkeyed_walk_yields_repeats_and_keeps_going(self) -> None:
        pages = [[1], [1], [2]]

        async def fetch(index: int) -> _PageResult[int]:
            return _PageResult(items=pages[index], has_more=index < 2)

        assert [n async for n in Paginator(fetch)] == [1, 1, 2]


# ---------------------------------------------------------------------------
# Paginator: pages that add nothing
# ---------------------------------------------------------------------------


def _walk(pages: list[list[int]], *, keyed: bool = True, last_says_more: bool = False) -> Any:
    """A paginator over *pages*; a page past the end repeats the last one."""
    fetched: list[int] = []

    async def fetch(index: int) -> _PageResult[int]:
        fetched.append(index)
        items = pages[min(index, len(pages) - 1)]
        return _PageResult(items=items, has_more=last_says_more or index < len(pages) - 1)

    walk = Paginator(fetch, key=(lambda item: item) if keyed else None)
    return walk, fetched


class TestPagesThatAddNothing:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
    async def test_a_leading_empty_page_does_not_end_the_walk(self, keyed: bool) -> None:
        walk, _ = _walk([[], [1]], keyed=keyed)
        assert [item async for item in walk] == [1]

    @pytest.mark.asyncio
    async def test_a_repeated_page_does_not_lose_what_follows_it(self) -> None:
        walk, _ = _walk([[1, 2], [1, 2], [3]])
        assert [item async for item in walk] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_pages_mixed_with_repeats_lose_nothing_within_the_bound(self) -> None:
        walk, _ = _walk([[1], [], [1], [2]])
        assert [item async for item in walk] == [1, 2]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
    async def test_endless_empty_pages_end_the_walk(self, keyed: bool) -> None:
        walk, fetched = _walk([[]], keyed=keyed, last_says_more=True)
        assert [item async for item in walk] == []
        assert len(fetched) == MAX_IDLE_PAGES

    @pytest.mark.asyncio
    async def test_an_endlessly_repeated_page_ends_the_walk(self) -> None:
        walk, fetched = _walk([[1, 2]], last_says_more=True)
        assert [item async for item in walk] == [1, 2]
        assert len(fetched) == 1 + MAX_IDLE_PAGES

    @pytest.mark.asyncio
    async def test_empty_and_repeated_pages_in_turn_end_the_walk(self) -> None:
        pages = [[1], []]
        fetched: list[int] = []

        async def fetch(index: int) -> _PageResult[int]:
            fetched.append(index)
            return _PageResult(items=pages[index % 2], has_more=True)

        assert [item async for item in Paginator(fetch, key=lambda item: item)] == [1]
        assert len(fetched) == 1 + MAX_IDLE_PAGES

    @pytest.mark.asyncio
    async def test_a_page_that_adds_something_resets_the_bound(self) -> None:
        idle = [[]] * (MAX_IDLE_PAGES - 1)
        walk, _ = _walk([[1], *idle, [2], *idle, [3]])
        assert [item async for item in walk] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_stopping_early_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        walk, _ = _walk([[]], last_says_more=True)
        with caplog.at_level(logging.WARNING, logger="venice_ai._pagination"):
            assert [item async for item in walk] == []
        assert "more" in caplog.text


def test_every_paginator_in_these_tests_is_bounded() -> None:
    """A walk that never ends must fail on the guard in conftest.py, not hang."""

    async def endless(_index: int) -> _PageResult[int]:
        return _PageResult(items=[1], has_more=True)

    async def walk_it() -> None:
        async for _item in Paginator(endless):
            pass

    with pytest.raises(pytest.fail.Exception, match="did not terminate"):
        asyncio.run(walk_it())
