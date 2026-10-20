"""Enum-like response fields accept values the SDK has not seen.

Venice extends server-side enums between SDK releases: the balance endpoint
began reporting ``consumptionCurrency: "EARNED_CREDITS"`` while the SDK still
declared a closed ``Literal`` without it, so ``client.billing.get_balance()``
raised a validation error for every account on that currency.

Each field below is typed as ``str`` on its response model. A value outside the
known set must round-trip as a plain string, and every known value must still
parse.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web
from pydantic import BaseModel

from venice_ai import VeniceClient
from venice_ai.core.models.common import VeniceParametersResponse
from venice_ai.types import (
    KNOWN_API_KEY_LIMIT_PERIODS,
    KNOWN_API_KEY_MODEL_PRIVACY_TIERS,
    KNOWN_API_KEY_TYPES,
    KNOWN_CONSUMPTION_CURRENCIES,
    KNOWN_USAGE_CURRENCIES,
    KNOWN_WEB_SEARCH_MODES,
)
from venice_ai.types.api import (
    ApiKey,
    BillingBalanceResponse,
    BillingUsageEntry,
    CreatedApiKey,
)
from venice_ai.types.api.streaming import ChatCompletionChunkChoiceDelta

UNKNOWN_VALUE = "FUTURE_VALUE_NOT_IN_THIS_RELEASE"

_USAGE_ENTRY = {
    "sku": "llama-3.3-70b-llm-input-mtoken",
    "amount": -0.01,
    "currency": "USD",
    "units": 1000,
    "pricePerUnitUsd": 0.00001,
    "notes": "API Inference",
    "timestamp": "2026-10-01T00:00:00.000Z",
}
_API_KEY = {
    "id": "key-id",
    "apiKeyType": "INFERENCE",
    "description": "Example",
    "last6Chars": "abcdef",
}
_CREATED_API_KEY = {
    "id": "key-id",
    "apiKey": "vk_example",
    "apiKeyType": "INFERENCE",
    "description": "Example",
    "consumptionLimit": {},
}
_VENICE_PARAMETERS = {
    "enable_web_search": "off",
    "enable_web_scraping": False,
    "enable_web_citations": False,
    "include_venice_system_prompt": True,
    "include_search_results_in_stream": False,
    "return_search_results_as_documents": False,
    "strip_thinking_response": False,
    "disable_thinking": False,
}

# (model, base payload, wire key, attribute name, known values)
CASES: list[tuple[type[BaseModel], dict[str, Any], str, str, tuple[str, ...]]] = [
    (
        BillingBalanceResponse,
        {},
        "consumptionCurrency",
        "consumption_currency",
        KNOWN_CONSUMPTION_CURRENCIES,
    ),
    (BillingUsageEntry, _USAGE_ENTRY, "currency", "currency", KNOWN_USAGE_CURRENCIES),
    (ApiKey, _API_KEY, "apiKeyType", "apiKeyType", KNOWN_API_KEY_TYPES),
    (ApiKey, _API_KEY, "modelPrivacy", "modelPrivacy", KNOWN_API_KEY_MODEL_PRIVACY_TIERS),
    (ApiKey, _API_KEY, "limitPeriod", "limitPeriod", KNOWN_API_KEY_LIMIT_PERIODS),
    (CreatedApiKey, _CREATED_API_KEY, "apiKeyType", "apiKeyType", KNOWN_API_KEY_TYPES),
    (CreatedApiKey, _CREATED_API_KEY, "limitPeriod", "limitPeriod", KNOWN_API_KEY_LIMIT_PERIODS),
    (
        VeniceParametersResponse,
        _VENICE_PARAMETERS,
        "enable_web_search",
        "enable_web_search",
        KNOWN_WEB_SEARCH_MODES,
    ),
    (
        ChatCompletionChunkChoiceDelta,
        {},
        "role",
        "role",
        ("system", "user", "assistant", "tool"),
    ),
]
IDS = [f"{model.__name__}.{attr}" for model, _, _, attr, _ in CASES]


@pytest.mark.parametrize(("model", "base", "key", "attr", "known"), CASES, ids=IDS)
def test_unknown_value_parses_as_a_string(
    model: type[BaseModel], base: dict[str, Any], key: str, attr: str, known: tuple[str, ...]
) -> None:
    parsed = model.model_validate({**base, key: UNKNOWN_VALUE})
    assert getattr(parsed, attr) == UNKNOWN_VALUE
    assert UNKNOWN_VALUE not in known


@pytest.mark.parametrize(("model", "base", "key", "attr", "known"), CASES, ids=IDS)
def test_every_known_value_still_parses(
    model: type[BaseModel], base: dict[str, Any], key: str, attr: str, known: tuple[str, ...]
) -> None:
    assert known
    for value in known:
        assert getattr(model.model_validate({**base, key: value}), attr) == value


_EARNED_CREDITS_BALANCE = {
    "canConsume": True,
    "consumptionCurrency": "EARNED_CREDITS",
    "balances": {"usd": 3.5, "diem": None},
    "diemEpochAllocation": 0,
}


@asynccontextmanager
async def _serve_balance(body: dict[str, Any]) -> AsyncIterator[str]:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(body)

    app = web.Application()
    app.router.add_get("/api/v1/billing/balance", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}/api/v1"
    finally:
        await runner.cleanup()


def test_earned_credits_is_a_known_currency() -> None:
    assert "EARNED_CREDITS" in KNOWN_CONSUMPTION_CURRENCIES
    assert "EARNED_CREDITS" in KNOWN_USAGE_CURRENCIES


async def test_get_balance_returns_earned_credits_balance() -> None:
    async with (
        _serve_balance(_EARNED_CREDITS_BALANCE) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        balance = await client.billing.get_balance()

    assert isinstance(balance, BillingBalanceResponse)
    assert balance.consumption_currency == "EARNED_CREDITS"
    assert balance.balances is not None
    assert balance.balances.usd == 3.5
