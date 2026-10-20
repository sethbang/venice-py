"""API key IP allowlists, against a local server.

``ipAllowlist`` must reach the wire when given and be absent otherwise (so
existing request bodies are unchanged); an empty list on update is the clear
signal and must be sent as ``[]``. A 403 from an IP-restricted key surfaces as
:class:`ApiKeyIpRestrictedError`, while other 403s stay plain
:class:`PermissionDeniedError`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web
from pydantic import ValidationError

from venice_ai import ApiKeyIpRestrictedError as TopLevelApiKeyIpRestrictedError
from venice_ai import VeniceClient
from venice_ai.exceptions import (
    ApiKeyIpRestrictedError,
    PermissionDeniedError,
    _make_status_error,
)
from venice_ai.types.api import CreateApiKeyRequest, Web3CreateApiKeyRequest
from venice_ai.types.api.api_keys import ApiKey, CreatedApiKey
from venice_ai.types.api.billing import BillingUsageEntry
from venice_ai.types.api.requests.api_keys import UpdateApiKeyRequest

IP_RESTRICTED_BODY = {
    "error": (
        "This API key is restricted to configured IP addresses, and this "
        "request's source IP is not allowed"
    )
}

KEY_ID = "e28e82dc-9df2-4b47-b726-d0a222ef2ab5"


def _api_key_wire(**extra: Any) -> dict[str, Any]:
    return {
        "id": KEY_ID,
        "apiKeyType": "INFERENCE",
        "description": "k",
        "last6Chars": "2V2jNW",
        **extra,
    }


def _created_wire(**extra: Any) -> dict[str, Any]:
    return {
        "id": KEY_ID,
        "apiKey": "vn_secret",
        "apiKeyType": "INFERENCE",
        "description": "k",
        "expiresAt": None,
        "consumptionLimit": {},
        **extra,
    }


@asynccontextmanager
async def _serve(
    seen: list[tuple[str, bytes]],
    *,
    status: int = 200,
    reply: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    async def handler(request: web.Request) -> web.Response:
        seen.append((request.method, await request.read()))
        if status != 200:
            return web.json_response(reply, status=status)
        if request.path.endswith("generate_web3_key"):
            return web.json_response({"success": True, "data": _created_wire()})
        if request.method == "POST":
            return web.json_response({"success": True, "data": _created_wire()})
        return web.json_response({"success": True, "data": _api_key_wire()})

    app = web.Application()
    app.router.add_route("*", "/api/v1/api_keys", handler)
    app.router.add_route("*", "/api/v1/api_keys/generate_web3_key", handler)
    app.router.add_route("*", "/api/v1/api_keys/rate_limits", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}/api/v1"
    finally:
        await runner.cleanup()


def _body(raw: bytes) -> dict[str, Any]:
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


# ---------------------------------------------------------------------------
# Outgoing bodies
# ---------------------------------------------------------------------------


async def test_create_without_allowlist_sends_no_ip_allowlist_key() -> None:
    seen: list[tuple[str, bytes]] = []
    async with (
        _serve(seen) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        await client.api_keys.create(
            api_key_request=CreateApiKeyRequest(apiKeyType="INFERENCE", description="k")
        )

    (method, raw), *_ = seen
    assert method == "POST"
    assert b"ipAllowlist" not in raw
    assert _body(raw) == {"apiKeyType": "INFERENCE", "description": "k"}


async def test_create_sends_ip_allowlist() -> None:
    seen: list[tuple[str, bytes]] = []
    allow = ["11.22.33.44", "11.22.33.0/24", "2a00:1::/48"]
    async with (
        _serve(seen) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        await client.api_keys.create(
            api_key_request=CreateApiKeyRequest(
                apiKeyType="INFERENCE", description="k", ipAllowlist=allow
            )
        )

    assert _body(seen[0][1])["ipAllowlist"] == allow


async def test_update_without_allowlist_leaves_it_out() -> None:
    seen: list[tuple[str, bytes]] = []
    async with (
        _serve(seen) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        await client.api_keys.update(id=KEY_ID, description="new")

    (method, raw), *_ = seen
    assert method == "PATCH"
    assert b"ipAllowlist" not in raw
    assert _body(raw) == {"id": KEY_ID, "description": "new"}


async def test_update_sends_replacement_allowlist() -> None:
    seen: list[tuple[str, bytes]] = []
    async with (
        _serve(seen) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        await client.api_keys.update(id=KEY_ID, ip_allowlist=["11.22.33.0/24"])

    assert _body(seen[0][1]) == {"id": KEY_ID, "ipAllowlist": ["11.22.33.0/24"]}


async def test_update_with_empty_list_sends_empty_list_to_clear() -> None:
    seen: list[tuple[str, bytes]] = []
    async with (
        _serve(seen) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        await client.api_keys.update(id=KEY_ID, ip_allowlist=[])

    assert _body(seen[0][1]) == {"id": KEY_ID, "ipAllowlist": []}


async def test_web3_create_sends_ip_allowlist_and_omits_it_when_unset() -> None:
    seen: list[tuple[str, bytes]] = []
    common = {"apiKeyType": "INFERENCE", "address": "0xabc", "signature": "0xsig", "token": "t"}
    async with (
        _serve(seen) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        await client.api_keys.create_web3_key(
            web3_key_request=Web3CreateApiKeyRequest(**common)  # type: ignore[arg-type]
        )
        await client.api_keys.create_web3_key(
            web3_key_request=Web3CreateApiKeyRequest(
                **common,  # type: ignore[arg-type]
                ipAllowlist=["2a00:1::/48"],
            )
        )

    assert b"ipAllowlist" not in seen[0][1]
    assert _body(seen[1][1])["ipAllowlist"] == ["2a00:1::/48"]


# ---------------------------------------------------------------------------
# Request model validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: CreateApiKeyRequest(apiKeyType="INFERENCE", description="k", **kw),
        lambda **kw: UpdateApiKeyRequest(id="k", **kw),
        lambda **kw: Web3CreateApiKeyRequest(
            apiKeyType="INFERENCE", address="0xabc", signature="0xsig", token="t", **kw
        ),
    ],
    ids=["create", "update", "web3"],
)
class TestRequestModels:
    def test_unknown_field_is_rejected_not_dropped(self, build: Any) -> None:
        # A misspelt restriction must fail loudly rather than create an
        # unrestricted key.
        with pytest.raises(ValidationError):
            build(ip_allowlist=["11.22.33.44"])

    def test_more_than_100_entries_rejected(self, build: Any) -> None:
        with pytest.raises(ValidationError):
            build(ipAllowlist=[f"11.22.{i // 256}.{i % 256}" for i in range(101)])

    def test_entry_longer_than_64_characters_rejected(self, build: Any) -> None:
        with pytest.raises(ValidationError):
            build(ipAllowlist=["1" * 65])

    def test_omitted_allowlist_is_not_dumped(self, build: Any) -> None:
        assert "ipAllowlist" not in build().model_dump(exclude_none=True)

    def test_empty_allowlist_is_dumped(self, build: Any) -> None:
        assert build(ipAllowlist=[]).model_dump(exclude_none=True)["ipAllowlist"] == []


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


def test_api_key_parses_with_and_without_ip_allowlist() -> None:
    restricted = ApiKey.model_validate(_api_key_wire(ipAllowlist=["11.22.33.44/32"]))
    assert restricted.ipAllowlist == ["11.22.33.44/32"]
    assert ApiKey.model_validate(_api_key_wire(ipAllowlist=[])).ipAllowlist == []
    assert ApiKey.model_validate(_api_key_wire()).ipAllowlist is None


def test_created_api_key_parses_with_and_without_ip_allowlist() -> None:
    created = CreatedApiKey.model_validate(_created_wire(ipAllowlist=["2a00:1::/48"]))
    assert created.ipAllowlist == ["2a00:1::/48"]
    assert CreatedApiKey.model_validate(_created_wire()).ipAllowlist is None


def test_billing_usage_entry_carries_api_key_id() -> None:
    row = {
        "amount": -0.1,
        "currency": "DIEM",
        "inferenceDetails": None,
        "notes": "API Inference",
        "pricePerUnitUsd": 0.1,
        "sku": "some-sku",
        "timestamp": "2026-06-14T19:52:45.087Z",
        "units": 1,
    }
    assert BillingUsageEntry.model_validate({**row, "apiKeyId": KEY_ID}).apiKeyId == KEY_ID
    assert BillingUsageEntry.model_validate({**row, "apiKeyId": None}).apiKeyId is None
    assert BillingUsageEntry.model_validate(row).apiKeyId is None


# ---------------------------------------------------------------------------
# ApiKeyIpRestrictedError
# ---------------------------------------------------------------------------


async def test_ip_restricted_403_raises_typed_error_end_to_end() -> None:
    seen: list[tuple[str, bytes]] = []
    async with (
        _serve(seen, status=403, reply=IP_RESTRICTED_BODY) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        with pytest.raises(ApiKeyIpRestrictedError) as exc_info:
            await client.api_keys.get_rate_limits()

    exc = exc_info.value
    assert isinstance(exc, PermissionDeniedError)
    assert exc.status_code == 403
    assert exc.body == IP_RESTRICTED_BODY
    assert "restricted to configured IP addresses" in str(exc)
    assert len(seen) == 1


async def test_other_403_stays_plain_permission_denied_end_to_end() -> None:
    seen: list[tuple[str, bytes]] = []
    async with (
        _serve(seen, status=403, reply={"error": "Unauthorized access"}) as base,
        VeniceClient(api_key="sk-test", base_url=base) as client,
    ):
        with pytest.raises(PermissionDeniedError) as exc_info:
            await client.api_keys.get_rate_limits()

    assert not isinstance(exc_info.value, ApiKeyIpRestrictedError)


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers: dict[str, str] = {}


@pytest.mark.parametrize(
    "body",
    [
        IP_RESTRICTED_BODY,
        json.dumps(IP_RESTRICTED_BODY),
        {"error": {"message": IP_RESTRICTED_BODY["error"]}},
        {"error": IP_RESTRICTED_BODY["error"].upper()},
    ],
    ids=["parsed", "raw-text", "nested-message", "case-insensitive"],
)
def test_make_status_error_recognises_ip_restriction(body: Any) -> None:
    exc = _make_status_error(None, body=body, response=_Resp(403))
    assert type(exc) is ApiKeyIpRestrictedError


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (403, {"error": "Unauthorized access", "code": "UNAUTHORIZED"}),
        (403, {"error": "API access has been disabled for this account."}),
        (403, None),
        (403, "not json"),
        (403, {"details": {}, "issues": []}),
        (401, IP_RESTRICTED_BODY),
    ],
)
def test_make_status_error_leaves_other_errors_alone(status: int, body: Any) -> None:
    exc = _make_status_error(None, body=body, response=_Resp(status))
    assert not isinstance(exc, ApiKeyIpRestrictedError)


def test_exported_from_package_root() -> None:
    assert TopLevelApiKeyIpRestrictedError is ApiKeyIpRestrictedError
