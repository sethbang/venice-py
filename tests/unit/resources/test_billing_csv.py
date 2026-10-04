"""CSV pages of GET /billing/usage-history, against a local server.

A CSV page carries its continuation token in the ``x-next-cursor`` header and
its file name in ``Content-Disposition``; both reach the caller on the typed
page, and the CSV iterator walks the cursor exactly like the JSON one. The
billing deadline covers the CSV body as well as the response.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.exceptions import BillingTimeoutError
from venice_ai.resources import billing as billing_module
from venice_ai.types.api.billing import BillingUsageHistoryCsvPage
from venice_ai.types.enums import BillingFormatEnum

HEADER = b"timestamp,sku,currency,amount\n"
PAGES = {
    None: (HEADER + b"2026-06-01T00:00:00Z,a,USD,0.1\n", "c2"),
    "c2": (HEADER + b"2026-06-02T00:00:00Z,b,USD,0.2\n", None),
}


@asynccontextmanager
async def _serve(handler: web.StreamResponse | object) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get("/api/v1/billing/usage-history", handler)  # type: ignore[arg-type]
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}/api/v1"
    finally:
        await runner.cleanup()


def _paged_handler(seen: list[dict[str, str]]):
    async def handler(request: web.Request) -> web.Response:
        seen.append(dict(request.query))
        assert request.headers["Accept"] == "text/csv"
        cursor = request.query.get("cursor")
        body, next_cursor = PAGES[cursor]
        headers = {
            "Content-Disposition": (
                f"attachment; filename=billing-usage-history-2026070{len(seen)}T091530123Z.csv"
            )
        }
        if next_cursor is not None:
            headers["x-next-cursor"] = next_cursor
        return web.Response(body=body, content_type="text/csv", headers=headers)

    return handler


async def test_csv_page_carries_body_cursor_and_filename() -> None:
    seen: list[dict[str, str]] = []
    async with (
        _serve(_paged_handler(seen)) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        page = await client.billing.get_usage_history(
            format=BillingFormatEnum.CSV, startTimestamp="2026-06-01T00:00:00Z", pageSize=10
        )

    assert isinstance(page, BillingUsageHistoryCsvPage)
    assert page.content == PAGES[None][0]
    assert page.text.startswith("timestamp,")
    assert page.nextCursor == "c2"
    assert page.filename == "billing-usage-history-20260701T091530123Z.csv"
    assert seen == [{"startTimestamp": "2026-06-01T00:00:00Z", "pageSize": "10"}]


async def test_last_csv_page_has_no_cursor() -> None:
    seen: list[dict[str, str]] = []
    async with (
        _serve(_paged_handler(seen)) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        page = await client.billing.get_usage_history(format=BillingFormatEnum.CSV, cursor="c2")
    assert page.nextCursor is None
    assert seen == [{"cursor": "c2"}]


async def test_iter_usage_history_csv_walks_every_page_with_cursor_only() -> None:
    seen: list[dict[str, str]] = []
    async with (
        _serve(_paged_handler(seen)) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        pages = [
            page
            async for page in client.billing.iter_usage_history_csv(
                currency="USD", startTimestamp="2026-06-01T00:00:00Z"
            )
        ]

    assert [p.content for p in pages] == [PAGES[None][0], PAGES["c2"][0]]
    assert seen == [
        {"currency": "USD", "startTimestamp": "2026-06-01T00:00:00Z"},
        {"cursor": "c2"},
    ]


async def test_iter_usage_history_csv_honors_max_pages() -> None:
    seen: list[dict[str, str]] = []
    async with (
        _serve(_paged_handler(seen)) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        pages = [p async for p in client.billing.iter_usage_history_csv(max_pages=1)]
    assert len(pages) == 1
    assert len(seen) == 1


async def test_stalled_csv_body_fails_at_the_billing_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The billing deadline, not the much longer client timeout, bounds the CSV body."""
    billing_deadline = 0.3
    client_timeout = 30.0
    monkeypatch.setattr(billing_module, "BILLING_REQUEST_TIMEOUT_SECONDS", billing_deadline)

    async def stall(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/csv"})
        response.content_length = 10_000
        await response.prepare(request)
        await response.write(HEADER)
        await asyncio.sleep(client_timeout)
        return response

    async with (
        _serve(stall) as base,
        VeniceClient(api_key="sk-test", base_url=base, timeout=client_timeout) as client,
    ):
        started = time.monotonic()
        with pytest.raises(BillingTimeoutError):
            await client.billing.get_usage_history(format=BillingFormatEnum.CSV)
        elapsed = time.monotonic() - started

    assert elapsed < client_timeout / 4
