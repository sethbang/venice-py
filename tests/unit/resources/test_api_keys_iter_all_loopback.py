"""``api_keys.iter_all`` against loopback servers.

``GET /api_keys`` declares no query parameters, and the live endpoint ignores
``page`` and ``limit``: every request answers with every key. The paginator has
to yield each key once and stop, whatever ``page_size`` the caller passed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web

from venice_ai import SyncVeniceClient, VeniceClient
from venice_ai._pagination import MAX_IDLE_PAGES

TOTAL_KEYS = 12
# A walk past this many items has stopped terminating.
RUNAWAY = TOTAL_KEYS * 5


def _key(index: int) -> dict[str, Any]:
    return {
        "id": f"key_{index}",
        "apiKey": "",
        "apiKeyType": "INFERENCE",
        "description": f"Key {index}",
        "createdAt": "2026-01-01T00:00:00Z",
        "expiresAt": None,
        "lastUsedAt": None,
        "last6Chars": f"{index:06d}",
        "consumptionLimits": {"usd": None, "diem": None},
        "usage": {"trailingSevenDays": {"usd": "0", "diem": "0"}},
    }


ALL_KEYS = [_key(i) for i in range(1, TOTAL_KEYS + 1)]
ALL_IDS = [key["id"] for key in ALL_KEYS]


def _ignores_paging(_query: dict[str, str]) -> dict[str, Any]:
    """The live endpoint: every request answers with every key."""
    return {"data": ALL_KEYS}


def _ignores_paging_and_claims_more(_query: dict[str, str]) -> dict[str, Any]:
    return {"data": ALL_KEYS, "pagination": {"hasMore": True}}


def _pages_with_total(query: dict[str, str]) -> dict[str, Any]:
    page = int(query.get("page", "1"))
    size = 5
    start = (page - 1) * size
    total_pages = (TOTAL_KEYS + size - 1) // size
    return {
        "data": ALL_KEYS[start : start + size],
        "pagination": {
            "page": page,
            "pageSize": size,
            "total": TOTAL_KEYS,
            "totalPages": total_pages,
        },
    }


def _pages_with_has_more(query: dict[str, str]) -> dict[str, Any]:
    page = int(query.get("page", "1"))
    size = 5
    start = (page - 1) * size
    return {
        "data": ALL_KEYS[start : start + size],
        "pagination": {"hasMore": start + size < TOTAL_KEYS},
    }


@asynccontextmanager
async def _serving(
    answer: Callable[[dict[str, str]], dict[str, Any]],
) -> AsyncIterator[tuple[str, list[dict[str, str]]]]:
    queries: list[dict[str, str]] = []

    async def handler(request: web.Request) -> web.Response:
        query = dict(request.query)
        queries.append(query)
        return web.json_response(answer(query))

    app = web.Application()
    app.router.add_get("/api/v1/api_keys", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = site._server.sockets[0].getsockname()[:2]  # type: ignore[union-attr]
    try:
        yield f"http://{host}:{port}/api/v1", queries
    finally:
        await runner.cleanup()


async def _walk(answer: Callable[[dict[str, str]], dict[str, Any]], **kwargs: Any) -> Any:
    """Return the ids a walk yields and the query of every request it sent."""
    ids: list[str] = []
    async with (
        _serving(answer) as (base, queries),
        VeniceClient(api_key="fake-key", base_url=base) as client,
    ):

        async def run() -> None:
            async for key in client.api_keys.iter_all(**kwargs):
                ids.append(key.id)
                if len(ids) > RUNAWAY:
                    pytest.fail(f"iter_all did not terminate: {len(queries)} requests")

        await asyncio.wait_for(run(), timeout=10)
    return ids, queries


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default-page-size"),
        pytest.param({"page_size": 5}, id="below-total"),
        pytest.param({"page_size": TOTAL_KEYS}, id="equal-to-total"),
        pytest.param({"page_size": 1}, id="one"),
        pytest.param({"page_size": 100}, id="above-total"),
    ],
)
@pytest.mark.asyncio
async def test_server_that_ignores_paging_yields_each_key_once(kwargs: dict[str, int]) -> None:
    ids, queries = await _walk(_ignores_paging, **kwargs)

    assert ids == ALL_IDS
    assert queries == [{}], "one request, carrying no parameter the server ignores"


@pytest.mark.parametrize("cap", [1, 3, TOTAL_KEYS, TOTAL_KEYS + 5])
@pytest.mark.asyncio
async def test_max_items_is_honoured(cap: int) -> None:
    ids, queries = await _walk(_ignores_paging, page_size=2, max_items=cap)

    assert ids == ALL_IDS[:cap]
    assert len(queries) == 1


@pytest.mark.parametrize("answer", [_pages_with_total, _pages_with_has_more])
@pytest.mark.asyncio
async def test_server_that_pages_is_followed_from_its_envelope(answer: Any) -> None:
    ids, queries = await _walk(answer)

    assert ids == ALL_IDS
    assert queries == [{}, {"page": "2"}, {"page": "3"}]


@pytest.mark.asyncio
async def test_max_items_stops_a_paged_walk_at_the_cap() -> None:
    ids, queries = await _walk(_pages_with_total, max_items=5)

    assert ids == ALL_IDS[:5]
    assert len(queries) == 1, "a cap met on a page boundary fetches nothing further"


@pytest.mark.asyncio
async def test_envelope_that_never_ends_stops_when_a_page_adds_no_key() -> None:
    ids, queries = await _walk(_ignores_paging_and_claims_more)

    assert ids == ALL_IDS
    assert len(queries) == 1 + MAX_IDLE_PAGES


@pytest.mark.asyncio
async def test_a_second_walk_of_one_paginator_starts_over() -> None:
    async with (
        _serving(_ignores_paging) as (base, _queries),
        VeniceClient(api_key="fake-key", base_url=base) as client,
    ):
        paginator = client.api_keys.iter_all()
        first = [key.id async for key in paginator]
        second = [key.id async for key in paginator]

    assert first == ALL_IDS
    assert second == ALL_IDS


def _ids(keys: Iterable[Any]) -> list[str]:
    return [key.id for key in keys]


@pytest.mark.asyncio
async def test_sync_client_walk_terminates_and_yields_each_key_once() -> None:
    async with _serving(_ignores_paging) as (base, queries):

        def walk() -> list[str]:
            with SyncVeniceClient(api_key="fake-key", base_url=base) as client:
                return _ids(client.api_keys.iter_all(page_size=TOTAL_KEYS))

        ids = await asyncio.wait_for(asyncio.to_thread(walk), timeout=10)

    assert ids == ALL_IDS
    assert queries == [{}]
