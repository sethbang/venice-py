"""CLI IP allowlist options for ``api-keys create`` and ``account keys update``.

Leaving the IP options out must omit ``ipAllowlist`` from the request body (an
empty list would clear the key's allowlist); only ``--clear-ip-allowlist``
sends ``[]``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from click.testing import CliRunner

from venice_ai.cli import config as cli_config
from venice_ai.cli.cli import cli

KEY_ID = "e28e82dc-9df2-4b47-b726-d0a222ef2ab5"


@asynccontextmanager
async def _serve(seen: list[tuple[str, bytes]]) -> AsyncIterator[str]:
    async def handler(request: web.Request) -> web.Response:
        seen.append((request.method, await request.read()))
        data: dict[str, Any] = {
            "id": KEY_ID,
            "apiKeyType": "INFERENCE",
            "description": "k",
            "last6Chars": "2V2jNW",
        }
        if request.method == "POST":
            data = {**data, "apiKey": "vn_secret", "consumptionLimit": {}}
        return web.json_response({"success": True, "data": data})

    app = web.Application()
    app.router.add_route("*", "/api/v1/api_keys", handler)
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
# CLI: no IP option must never turn into an empty (clearing) list
# ---------------------------------------------------------------------------


def _invoke_cli(module: str, target: str, args: list[str]) -> tuple[Any, dict[str, Any]]:
    fake = MagicMock(return_value=None)
    with (
        patch(f"venice_ai.cli.commands.{module}.{target}", fake),
        patch(f"venice_ai.cli.commands.{module}.asyncio.run"),
    ):
        result = CliRunner().invoke(cli, args)
    kwargs = fake.call_args.kwargs if fake.call_args else {}
    return result, kwargs


def test_cli_update_without_ip_option_omits_allowlist() -> None:
    result, kwargs = _invoke_cli(
        "account", "_keys_update_async", ["account", "keys", "update", KEY_ID, "--description", "x"]
    )
    assert result.exit_code == 0, result.output
    assert kwargs["ip_allowlist"] is None


def test_cli_update_ip_allow_is_repeatable() -> None:
    result, kwargs = _invoke_cli(
        "account",
        "_keys_update_async",
        [
            "account",
            "keys",
            "update",
            KEY_ID,
            "--ip-allow",
            "203.0.113.1",
            "--ip-allow",
            "2a00:1::/48",
        ],
    )
    assert result.exit_code == 0, result.output
    assert kwargs["ip_allowlist"] == ["203.0.113.1", "2a00:1::/48"]


def test_cli_update_clear_sends_empty_list_and_counts_as_an_option() -> None:
    result, kwargs = _invoke_cli(
        "account",
        "_keys_update_async",
        ["account", "keys", "update", KEY_ID, "--clear-ip-allowlist"],
    )
    assert result.exit_code == 0, result.output
    assert kwargs["ip_allowlist"] == []


def test_cli_update_ip_options_are_mutually_exclusive() -> None:
    result, kwargs = _invoke_cli(
        "account",
        "_keys_update_async",
        ["account", "keys", "update", KEY_ID, "--ip-allow", "203.0.113.1", "--clear-ip-allowlist"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert kwargs == {}


def test_cli_create_ip_allow() -> None:
    result, kwargs = _invoke_cli(
        "api_keys", "_create_key_async", ["api-keys", "create", "--name", "k"]
    )
    assert result.exit_code == 0, result.output
    assert kwargs["ip_allowlist"] is None

    result, kwargs = _invoke_cli(
        "api_keys",
        "_create_key_async",
        ["api-keys", "create", "--name", "k", "--ip-allow", "11.22.33.0/24"],
    )
    assert result.exit_code == 0, result.output
    assert kwargs["ip_allowlist"] == ["11.22.33.0/24"]


async def test_cli_create_body_against_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from venice_ai.cli.commands.api_keys import _create_key_async

    seen: list[tuple[str, bytes]] = []
    async with _serve(seen) as base:
        monkeypatch.setattr(
            cli_config, "get_client_kwargs", lambda: {"api_key": "sk-test", "base_url": base}
        )
        ctx = SimpleNamespace(obj={"plain": True})
        await _create_key_async(ctx, name="k", output_json=True)
        await _create_key_async(ctx, name="k", ip_allowlist=["203.0.113.1"], output_json=True)

    assert b"ipAllowlist" not in seen[0][1]
    assert _body(seen[1][1])["ipAllowlist"] == ["203.0.113.1"]


async def test_cli_update_body_against_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from venice_ai.cli.commands.account import _keys_update_async

    seen: list[tuple[str, bytes]] = []
    common: dict[str, Any] = {
        "api_key_id": KEY_ID,
        "description": "d",
        "expires_at": None,
        "consumption_limit": None,
        "output_json": True,
    }
    async with _serve(seen) as base:
        monkeypatch.setattr(
            cli_config, "get_client_kwargs", lambda: {"api_key": "sk-test", "base_url": base}
        )
        ctx = SimpleNamespace(obj={"plain": True})
        await _keys_update_async(ctx, **common)
        await _keys_update_async(ctx, **common, ip_allowlist=[])

    assert _body(seen[0][1]) == {"id": KEY_ID, "description": "d"}
    assert _body(seen[1][1]) == {"id": KEY_ID, "description": "d", "ipAllowlist": []}
