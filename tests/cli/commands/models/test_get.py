"""Tests for ``venice-py models get`` (cli/commands/models/get.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.models.get import _get_async


def _make_ctx():
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": False}
    ctx.exit = MagicMock(side_effect=SystemExit(1))
    return ctx


def _setup_client(MockVeniceClient, mock_client):
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


def _make_model(model_id="llama-3.3-70b"):
    return SimpleNamespace(
        id=model_id,
        type="text",
        created=1700000000,
        model_spec=SimpleNamespace(
            name="Llama 3.3 70B",
            traits=["default"],
            offline=False,
            beta=False,
            availableContextTokens=131072,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.5, diem=0.5),
                output=SimpleNamespace(usd=2.0, diem=2.0),
            ),
            capabilities=SimpleNamespace(
                supportsFunctionCalling=True,
                supportsVision=False,
                supportsReasoning=True,
                supportsWebSearch=True,
                optimizedForCode=False,
                supportsResponseSchema=True,
                supportsLogProbs=True,
                quantization="fp8",
            ),
        ),
    )


class TestGetCLI:
    def test_get_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "get", "--help"])
        assert result.exit_code == 0
        assert "MODEL_ID" in result.output

    def test_get_invokes_asyncio_run(self):
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.models.get.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["models", "get", "llama-3.3-70b"])
            assert mock_run.called


class TestGetAsync:
    @pytest.mark.asyncio
    async def test_get_invokes_models_get_with_model_id(self):
        mock_client = AsyncMock()
        mock_client.models.get = AsyncMock(return_value=_make_model())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.get.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _get_async(_make_ctx(), model_id="llama-3.3-70b", output_json=False)

            mock_client.models.get.assert_awaited_once_with(model_id="llama-3.3-70b")

    @pytest.mark.asyncio
    async def test_get_json_output_uses_model_dump(self):
        mock_model = MagicMock()
        mock_model.model_dump = MagicMock(return_value={"id": "llama-3.3-70b", "type": "text"})
        mock_client = AsyncMock()
        mock_client.models.get = AsyncMock(return_value=mock_model)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.get.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _get_async(_make_ctx(), model_id="llama-3.3-70b", output_json=True)

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload["id"] == "llama-3.3-70b"

    @pytest.mark.asyncio
    async def test_get_json_output_keeps_capabilities(self):
        from venice_ai.types.api.models import ModelResponse

        model = ModelResponse.model_validate(
            {
                "id": "llama-3.3-70b",
                "object": "model",
                "created": 0,
                "owned_by": "venice.ai",
                "type": "text",
                "model_spec": {
                    "name": "Llama",
                    "availableContextTokens": 131072,
                    "capabilities": {
                        "optimizedForCode": False,
                        "quantization": "fp8",
                        "supportsFunctionCalling": True,
                        "supportsReasoning": False,
                        "supportsResponseSchema": True,
                        "supportsVision": True,
                        "supportsWebSearch": False,
                        "supportsLogProbs": False,
                    },
                },
            }
        )
        mock_client = AsyncMock()
        mock_client.models.get = AsyncMock(return_value=model)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.get.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _get_async(_make_ctx(), model_id="llama-3.3-70b", output_json=True)

        spec = json.loads(mock_echo.call_args[0][0])["model_spec"]
        assert spec["availableContextTokens"] == 131072
        assert spec["capabilities"]["supportsVision"] is True

    @pytest.mark.asyncio
    async def test_get_value_error_exits_nonzero(self):
        mock_client = AsyncMock()
        mock_client.models.get = AsyncMock(side_effect=ValueError("not found"))

        ctx = _make_ctx()
        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.get.console"),
            pytest.raises(SystemExit),
        ):
            _setup_client(MockClient, mock_client)
            await _get_async(ctx, model_id="missing", output_json=False)

        ctx.exit.assert_called_with(1)
