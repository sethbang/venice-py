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

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai._pagination import Paginator
from venice_ai.resources.api_keys import ApiKeys
from venice_ai.resources.billing import Billing
from venice_ai.resources.characters import Characters
from venice_ai.resources.x402 import X402
from venice_ai.types.api import ApiKey
from venice_ai.types.api.characters import (
    CharacterReviewsResponse,
    CharactersListResponse,
)
from venice_ai.types.api.x402 import X402TransactionsResponse

# ---------------------------------------------------------------------------
# Helpers — fabricate response shapes for each endpoint
# ---------------------------------------------------------------------------


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
# api_keys.iter_all — page-based, no envelope, terminate on short page
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
    async def test_exhausts_three_pages(self, resource: ApiKeys, mock_client: MagicMock) -> None:
        # Three full pages of 2, then a short page → done
        mock_client.get.side_effect = [
            [
                _make_api_key(1).model_dump(by_alias=True),
                _make_api_key(2).model_dump(by_alias=True),
            ],
            [
                _make_api_key(3).model_dump(by_alias=True),
                _make_api_key(4).model_dump(by_alias=True),
            ],
            [_make_api_key(5).model_dump(by_alias=True)],  # short page → terminate
        ]
        ids = [k.id async for k in resource.iter_all(page_size=2)]
        assert ids == ["key_1", "key_2", "key_3", "key_4", "key_5"]

    @pytest.mark.asyncio
    async def test_advances_one_based_page(self, resource: ApiKeys, mock_client: MagicMock) -> None:
        mock_client.get.side_effect = [[], []]  # immediate empty page
        async for _ in resource.iter_all(page_size=10):
            pass
        # First call: page=1 (1-based), limit=10
        first_call = mock_client.get.call_args_list[0]
        assert first_call.kwargs["params"] == {"page": 1, "limit": 10}

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
                        "id": f"rev-{i}",
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
