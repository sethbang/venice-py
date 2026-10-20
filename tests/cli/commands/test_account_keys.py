"""Tests for ``venice-py account keys get|update`` (cli/commands/account.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.account import (
    _keys_get_async,
    _keys_rate_limit_logs_async,
    _keys_rate_limits_async,
    _keys_update_async,
)


def _make_ctx(plain: bool = False):
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client(MockVeniceClient, mock_client):
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


def _make_api_key(
    id="key_abc123",
    description="my key",
    apiKeyType="INFERENCE",
    createdAt="2025-01-01T00:00:00Z",
    expiresAt=None,
    lastUsedAt=None,
    last6Chars="abcdef",
):
    obj = SimpleNamespace(
        id=id,
        apiKeyType=apiKeyType,
        description=description,
        last6Chars=last6Chars,
        createdAt=createdAt,
        expiresAt=expiresAt,
        lastUsedAt=lastUsedAt,
        consumptionLimits=None,
        usage=None,
    )
    obj.model_dump = lambda: {
        "id": obj.id,
        "apiKeyType": obj.apiKeyType,
        "description": obj.description,
        "last6Chars": obj.last6Chars,
        "createdAt": obj.createdAt,
        "expiresAt": obj.expiresAt,
        "lastUsedAt": obj.lastUsedAt,
        "consumptionLimits": obj.consumptionLimits,
        "usage": obj.usage,
    }
    return obj


# ---------------------------------------------------------------------------
# `venice-py account keys` group + `keys get` subcommand
# ---------------------------------------------------------------------------


class TestAccountKeysGroup:
    def test_keys_group_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "--help"])
        assert result.exit_code == 0
        assert "get" in result.output and "update" in result.output


class TestKeysGetCLI:
    def test_keys_get_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "get", "--help"])
        assert result.exit_code == 0
        assert "API_KEY_ID" in result.output

    def test_keys_get_invokes_asyncio_run(self):
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["account", "keys", "get", "key_abc123"])
            assert mock_run.called


class TestKeysGetAsync:
    @pytest.mark.asyncio
    async def test_keys_get_invokes_retrieve(self):
        mock_client = AsyncMock()
        mock_client.api_keys.retrieve = AsyncMock(return_value=_make_api_key())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _keys_get_async(_make_ctx(), "key_abc123", output_json=False)

            mock_client.api_keys.retrieve.assert_awaited_once_with(api_key_id="key_abc123")

    @pytest.mark.asyncio
    async def test_keys_get_json_output_uses_model_dump(self):
        mock_client = AsyncMock()
        mock_client.api_keys.retrieve = AsyncMock(return_value=_make_api_key())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _keys_get_async(_make_ctx(), "key_abc123", output_json=True)

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload["id"] == "key_abc123"


# ---------------------------------------------------------------------------
# `keys update --limit-period`
# ---------------------------------------------------------------------------


class TestKeysUpdateLimitPeriod:
    def test_keys_update_help_lists_limit_period(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "update", "--help"])
        assert result.exit_code == 0
        assert "--limit-period" in result.output

    @pytest.mark.asyncio
    async def test_keys_update_forwards_limit_period(self):
        mock_client = AsyncMock()
        mock_client.api_keys.update = AsyncMock(return_value=_make_api_key())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _keys_update_async(
                _make_ctx(),
                api_key_id="key_abc123",
                description=None,
                expires_at=None,
                consumption_limit={"diem": 5.0},
                limit_period="MONTH",
                output_json=False,
            )

            mock_client.api_keys.update.assert_awaited_once_with(
                id="key_abc123",
                description=None,
                expires_at=None,
                consumption_limit={"diem": 5.0},
                limit_period="MONTH",
                ip_allowlist=None,
            )

    def test_keys_update_limit_period_alone_is_valid(self):
        """Providing only --limit-period should not raise a UsageError."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            result = runner.invoke(
                cli,
                ["account", "keys", "update", "key_abc123", "--limit-period", "LIFETIME"],
            )
            assert result.exit_code == 0
            assert mock_run.called


# ---------------------------------------------------------------------------
# `keys rate-limits` and `keys rate-limit-logs`
# ---------------------------------------------------------------------------


def _make_rate_limits_response():
    rate_limit = SimpleNamespace(type="RPM", amount=500)
    rate_limit_tpm = SimpleNamespace(type="TPM", amount=200000)
    model_rl = SimpleNamespace(apiModelId="some-model", rateLimits=[rate_limit, rate_limit_tpm])
    data = SimpleNamespace(
        accessPermitted=True,
        apiTier=SimpleNamespace(id="tier1", isCharged=True),
        balances=SimpleNamespace(usd=10.0, diem=5.0),
        keyExpiration=None,
        nextEpochBegins="2026-01-01T00:00:00Z",
        rateLimits=[model_rl],
    )
    return SimpleNamespace(data=data)


def _make_rate_limit_logs_response():
    entry = SimpleNamespace(
        apiKeyId="key_abc123",
        modelId="some-model",
        rateLimitTier="tier1",
        rateLimitType="RPM",
        timestamp="2026-01-01T00:00:00Z",
    )
    return SimpleNamespace(object="list", data=[entry])


class TestKeysRateLimitsCLI:
    def test_rate_limits_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "rate-limits", "--help"])
        assert result.exit_code == 0

    def test_rate_limit_logs_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "rate-limit-logs", "--help"])
        assert result.exit_code == 0

    def test_rate_limits_group_lists_subcommands(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "--help"])
        assert result.exit_code == 0
        assert "rate-limits" in result.output
        assert "rate-limit-logs" in result.output


class TestKeysRateLimitsAsync:
    @pytest.mark.asyncio
    async def test_rate_limits_calls_get_rate_limits(self):
        mock_client = AsyncMock()
        mock_client.api_keys.get_rate_limits = AsyncMock(return_value=_make_rate_limits_response())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client(MockClient, mock_client)
            await _keys_rate_limits_async(_make_ctx(), output_json=False)

            mock_client.api_keys.get_rate_limits.assert_awaited_once()
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_rate_limits_json_output(self):
        mock_client = AsyncMock()
        resp = _make_rate_limits_response()
        resp.model_dump = lambda: {"data": {"accessPermitted": True}}
        mock_client.api_keys.get_rate_limits = AsyncMock(return_value=resp)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _keys_rate_limits_async(_make_ctx(), output_json=True)

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload["data"]["accessPermitted"] is True

    @pytest.mark.asyncio
    async def test_rate_limit_logs_calls_get_rate_limit_logs(self):
        mock_client = AsyncMock()
        mock_client.api_keys.get_rate_limit_logs = AsyncMock(
            return_value=_make_rate_limit_logs_response()
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client(MockClient, mock_client)
            await _keys_rate_limit_logs_async(_make_ctx(), output_json=False)

            mock_client.api_keys.get_rate_limit_logs.assert_awaited_once()
            assert mock_console.print.called


# ---------------------------------------------------------------------------
# `keys update` subcommand
# ---------------------------------------------------------------------------


class TestKeysUpdateCLI:
    def test_keys_update_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "update", "--help"])
        assert result.exit_code == 0
        for fragment in ("--description", "--expiry", "--limit-usd"):
            assert fragment in result.output

    def test_keys_update_without_fields_is_usage_error(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "keys", "update", "key_abc123"])
        # UsageError exits with code 2 — Click default.
        assert result.exit_code == 2
        assert "at least one" in result.output.lower()

    def test_keys_update_invokes_asyncio_run_when_field_provided(self):
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(
                cli,
                [
                    "account",
                    "keys",
                    "update",
                    "key_abc123",
                    "--description",
                    "Production",
                ],
            )
            assert mock_run.called


class TestKeysUpdateAsync:
    @pytest.mark.asyncio
    async def test_keys_update_passes_id_kwarg_to_sdk(self):
        # The SDK's update() takes id=, not api_key_id=. The CLI must remap.
        mock_client = AsyncMock()
        mock_client.api_keys.update = AsyncMock(return_value=_make_api_key())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _keys_update_async(
                _make_ctx(),
                api_key_id="key_abc123",
                description="New",
                expires_at=None,
                consumption_limit=None,
                output_json=False,
            )

            mock_client.api_keys.update.assert_awaited_once_with(
                id="key_abc123",
                description="New",
                expires_at=None,
                consumption_limit=None,
                limit_period=None,
                ip_allowlist=None,
            )

    @pytest.mark.asyncio
    async def test_keys_update_with_consumption_limit(self):
        mock_client = AsyncMock()
        mock_client.api_keys.update = AsyncMock(return_value=_make_api_key())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _keys_update_async(
                _make_ctx(),
                api_key_id="key_abc123",
                description=None,
                expires_at="2026-12-31",
                consumption_limit={"usd": 50.0, "diem": 5.0},
                output_json=False,
            )

            mock_client.api_keys.update.assert_awaited_once_with(
                id="key_abc123",
                description=None,
                expires_at="2026-12-31",
                consumption_limit={"usd": 50.0, "diem": 5.0},
                limit_period=None,
                ip_allowlist=None,
            )

    @pytest.mark.asyncio
    async def test_keys_update_json_output(self):
        mock_client = AsyncMock()
        mock_client.api_keys.update = AsyncMock(return_value=_make_api_key())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _keys_update_async(
                _make_ctx(),
                api_key_id="key_abc123",
                description="x",
                expires_at=None,
                consumption_limit=None,
                output_json=True,
            )

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload["id"] == "key_abc123"
