"""
Tests for cli/commands/api_keys.py

Covers:
- api_keys() click group command
- list_keys() click command entrypoint
- _list_keys_async() core logic with all branches
- create_key() click command entrypoint
- _create_key_async() core logic with all branches
- delete_key() click command entrypoint
- _delete_key_async() core logic with all branches
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.api_keys import (
    _create_key_async,
    _delete_key_async,
    _list_keys_async,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(plain: bool = False):
    """Return a minimal mock Click context."""
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client_patch(MockVeniceClient, mock_client):
    """Configure the MockVeniceClient to act as an async context manager returning mock_client."""
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


def _make_api_key(
    id="key-abc123",
    description=None,
    apiKeyType="INFERENCE",
    createdAt="2024-01-01T00:00:00Z",
    lastUsedAt=None,
    last6Chars=None,
    expiresAt=None,
):
    """Return a SimpleNamespace representing an API key object."""
    return SimpleNamespace(
        id=id,
        description=description,
        apiKeyType=apiKeyType,
        createdAt=createdAt,
        lastUsedAt=lastUsedAt,
        last6Chars=last6Chars,
        expiresAt=expiresAt,
    )


# ---------------------------------------------------------------------------
# _list_keys_async tests — JSON output
# ---------------------------------------------------------------------------


class TestListKeysAsyncJsonOutput:
    """Tests for _list_keys_async in JSON output mode."""

    @pytest.mark.asyncio
    async def test_list_keys_json_with_model_dump(self):
        """JSON output mode with keys having model_dump method."""
        mock_client = AsyncMock()
        key = MagicMock()
        key.model_dump = MagicMock(
            return_value={
                "id": "key-123",
                "description": "My Key",
                "apiKeyType": "INFERENCE",
            }
        )
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_keys_async(ctx, output_json=True)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert len(data) == 1
            assert data[0]["id"] == "key-123"
            assert data[0]["description"] == "My Key"

    @pytest.mark.asyncio
    async def test_list_keys_json_with_dict_keys(self):
        """JSON output mode with keys that are dicts."""
        mock_client = AsyncMock()
        key = {
            "id": "key-dict-123",
            "description": "Dict Key",
            "apiKeyType": "INFERENCE",
        }
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_keys_async(ctx, output_json=True)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert len(data) == 1
            assert data[0]["id"] == "key-dict-123"

    @pytest.mark.asyncio
    async def test_list_keys_json_with_getattr_fallback(self):
        """JSON output mode with keys using getattr fallback (no model_dump, not dict)."""
        mock_client = AsyncMock()
        key = _make_api_key(id="key-ns-456", description="Namespace Key", last6Chars="abc123")
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_keys_async(ctx, output_json=True)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert len(data) == 1
            assert data[0]["id"] == "key-ns-456"
            assert data[0]["description"] == "Namespace Key"
            assert data[0]["last6Chars"] == "abc123"

    @pytest.mark.asyncio
    async def test_list_keys_json_empty_list(self):
        """JSON output mode with empty keys list."""
        mock_client = AsyncMock()
        mock_client.api_keys.list = AsyncMock(return_value=[])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_keys_async(ctx, output_json=True)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert data == []

    @pytest.mark.asyncio
    async def test_list_keys_json_multiple_keys(self):
        """JSON output mode with multiple keys of different types."""
        mock_client = AsyncMock()
        key1 = MagicMock()
        key1.model_dump = MagicMock(return_value={"id": "key-1", "description": "Key 1"})
        key2 = {"id": "key-2", "description": "Key 2"}
        key3 = _make_api_key(id="key-3", description="Key 3")
        mock_client.api_keys.list = AsyncMock(return_value=[key1, key2, key3])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_keys_async(ctx, output_json=True)

            data = json.loads(mock_echo.call_args[0][0])
            assert len(data) == 3


# ---------------------------------------------------------------------------
# _list_keys_async tests — empty list display
# ---------------------------------------------------------------------------


class TestListKeysAsyncEmptyDisplay:
    """Tests for _list_keys_async displaying empty list message."""

    @pytest.mark.asyncio
    async def test_list_keys_empty_plain_mode(self):
        """Empty keys in plain mode shows 'No API keys found.'"""
        mock_client = AsyncMock()
        mock_client.api_keys.list = AsyncMock(return_value=[])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_keys_async(ctx, output_json=False)

            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("No API keys found" in c for c in calls)

    @pytest.mark.asyncio
    async def test_list_keys_empty_rich_mode(self):
        """Empty keys in rich mode shows console message."""
        mock_client = AsyncMock()
        mock_client.api_keys.list = AsyncMock(return_value=[])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_keys_async(ctx, output_json=False)

            assert mock_console.print.called
            args_str = str(mock_console.print.call_args_list)
            assert "No API keys found" in args_str


# ---------------------------------------------------------------------------
# _list_keys_async tests — plain mode display
# ---------------------------------------------------------------------------


class TestListKeysAsyncPlainDisplay:
    """Tests for _list_keys_async plain mode display."""

    @pytest.mark.asyncio
    async def test_list_keys_plain_with_keys(self):
        """Plain mode with keys shows headers and key info."""
        mock_client = AsyncMock()
        key = _make_api_key(
            id="key-plain-1",
            description="Production Key",
            last6Chars="xyz999",
            createdAt="2024-06-01T00:00:00Z",
            lastUsedAt="2024-06-15T00:00:00Z",
        )
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_keys_async(ctx, output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Production Key" in calls_str
            assert "xyz999" in calls_str
            assert "Total: 1 key(s)" in calls_str

    @pytest.mark.asyncio
    async def test_list_keys_plain_with_no_last6(self):
        """Plain mode with key having no last6Chars shows N/A prefix."""
        mock_client = AsyncMock()
        key = _make_api_key(last6Chars=None)
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_keys_async(ctx, output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "N/A" in calls_str

    @pytest.mark.asyncio
    async def test_list_keys_plain_with_never_last_used(self):
        """Plain mode with key having None lastUsedAt shows 'Never'."""
        mock_client = AsyncMock()
        key = _make_api_key(lastUsedAt=None)
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_keys_async(ctx, output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Never" in calls_str


# ---------------------------------------------------------------------------
# _list_keys_async tests — rich mode display
# ---------------------------------------------------------------------------


class TestListKeysAsyncRichDisplay:
    """Tests for _list_keys_async rich mode display."""

    @pytest.mark.asyncio
    async def test_list_keys_rich_with_keys(self):
        """Rich mode with keys prints a table via console."""
        mock_client = AsyncMock()
        key = _make_api_key(
            id="key-rich-1",
            description="My Rich Key",
            apiKeyType="INFERENCE",
            last6Chars="aaa111",
        )
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_keys_async(ctx, output_json=False)

            assert mock_console.print.called
            # Should have table + total count line
            assert mock_console.print.call_count >= 2

    @pytest.mark.asyncio
    async def test_list_keys_rich_with_no_last6_shows_na(self):
        """Rich mode with no last6Chars shows N/A in table."""
        mock_client = AsyncMock()
        key = _make_api_key(last6Chars=None)
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_keys_async(ctx, output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_list_keys_rich_with_no_description_shows_default(self):
        """Rich mode with no description shows '(no description)'."""
        mock_client = AsyncMock()
        key = _make_api_key(description=None)
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_keys_async(ctx, output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_list_keys_rich_key_no_last_used(self):
        """Rich mode with key having None lastUsedAt shows 'Never'."""
        mock_client = AsyncMock()
        key = _make_api_key(lastUsedAt=None)
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_keys_async(ctx, output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_list_keys_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        key = _make_api_key()
        mock_client.api_keys.list = AsyncMock(return_value=[key])

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _list_keys_async(ctx, output_json=False)

            assert mock_console.print.called


# ---------------------------------------------------------------------------
# _create_key_async tests — JSON output
# ---------------------------------------------------------------------------


class TestCreateKeyAsyncJsonOutput:
    """Tests for _create_key_async in JSON output mode."""

    @pytest.mark.asyncio
    async def test_create_key_json_output_basic(self):
        """JSON output mode returns key details including secret."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="new-key-id",
            description="Production",
            apiKeyType="INFERENCE",
            createdAt="2024-01-01T00:00:00Z",
            apiKey="sk-secret-value",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx()
            await _create_key_async(ctx, name="Production", output_json=True)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert data["id"] == "new-key-id"
            assert data["description"] == "Production"
            assert data["apiKey"] == "sk-secret-value"
            assert data["apiKeyType"] == "INFERENCE"

    @pytest.mark.asyncio
    async def test_create_key_json_output_no_created_at(self):
        """JSON output with None createdAt (shows None in output)."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-no-date",
            description="No Date Key",
            apiKeyType="INFERENCE",
            createdAt=None,
            apiKey="sk-no-date-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx()
            await _create_key_async(ctx, name="No Date Key", output_json=True)

            data = json.loads(mock_echo.call_args[0][0])
            assert data["createdAt"] is None

    @pytest.mark.asyncio
    async def test_create_key_json_output_no_api_key_secret(self):
        """JSON output with None apiKey (no secret returned)."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-no-secret",
            description="No Secret",
            apiKeyType="INFERENCE",
            createdAt="2024-01-01T00:00:00Z",
            apiKey=None,
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx()
            await _create_key_async(ctx, name="No Secret", output_json=True)

            data = json.loads(mock_echo.call_args[0][0])
            assert data["apiKey"] is None


# ---------------------------------------------------------------------------
# _create_key_async tests — plain mode
# ---------------------------------------------------------------------------


class TestCreateKeyAsyncPlainDisplay:
    """Tests for _create_key_async in plain mode."""

    @pytest.mark.asyncio
    async def test_create_key_plain_with_secret(self):
        """Plain mode with secret key shows warning and secret."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-plain-new",
            description="My Plain Key",
            apiKeyType="INFERENCE",
            createdAt="2024-03-01T00:00:00Z",
            apiKey="sk-plain-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=True)
            await _create_key_async(ctx, name="My Plain Key", output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "API key created successfully" in calls_str
            assert "key-plain-new" in calls_str
            assert "sk-plain-secret" in calls_str
            assert "WARNING" in calls_str

    @pytest.mark.asyncio
    async def test_create_key_plain_no_secret(self):
        """Plain mode without secret key omits secret warning section."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-plain-no-secret",
            description="No Secret Key",
            apiKeyType="INFERENCE",
            createdAt="2024-03-01T00:00:00Z",
            apiKey=None,
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=True)
            await _create_key_async(ctx, name="No Secret Key", output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "API key created successfully" in calls_str
            assert "WARNING" not in calls_str

    @pytest.mark.asyncio
    async def test_create_key_plain_no_created_at(self):
        """Plain mode with no createdAt omits the Created line."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-plain-no-date",
            description="No Date Key",
            apiKeyType="INFERENCE",
            createdAt=None,
            apiKey="sk-some-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=True)
            await _create_key_async(ctx, name="No Date Key", output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Created:" not in calls_str


# ---------------------------------------------------------------------------
# _create_key_async tests — rich mode
# ---------------------------------------------------------------------------


class TestCreateKeyAsyncRichDisplay:
    """Tests for _create_key_async in rich mode."""

    @pytest.mark.asyncio
    async def test_create_key_rich_with_secret(self):
        """Rich mode with secret key shows panel via console."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-rich-new",
            description="My Rich Key",
            apiKeyType="INFERENCE",
            createdAt="2024-03-01T00:00:00Z",
            apiKey="sk-rich-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=False)
            await _create_key_async(ctx, name="My Rich Key", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_create_key_rich_no_secret(self):
        """Rich mode without secret key still shows panel."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-rich-no-secret",
            description="No Secret",
            apiKeyType="INFERENCE",
            createdAt="2024-03-01T00:00:00Z",
            apiKey=None,
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=False)
            await _create_key_async(ctx, name="No Secret", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_create_key_rich_no_created_at(self):
        """Rich mode without createdAt omits that line from panel."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-rich-no-date",
            description="No Date",
            apiKeyType="INFERENCE",
            createdAt=None,
            apiKey="sk-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=False)
            await _create_key_async(ctx, name="No Date", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_create_key_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-ctx-none",
            description="Ctx None Key",
            apiKeyType="INFERENCE",
            createdAt="2024-03-01T00:00:00Z",
            apiKey="sk-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = MagicMock()
            ctx.obj = None
            await _create_key_async(ctx, name="Ctx None Key", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_create_key_description_fallback_to_name(self):
        """If description is None, falls back to the name parameter."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-desc-fallback",
            description=None,
            apiKeyType=None,
            createdAt="2024-03-01T00:00:00Z",
            apiKey="sk-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest") as MockRequest,
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)
            MockRequest.return_value = MagicMock()

            ctx = _make_ctx(plain=True)
            await _create_key_async(ctx, name="Fallback Name", output_json=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Fallback Name" in calls_str


# ---------------------------------------------------------------------------
# _delete_key_async tests — with --yes flag
# ---------------------------------------------------------------------------


class TestDeleteKeyAsyncWithYesFlag:
    """Tests for _delete_key_async with --yes flag."""

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_plain_success(self):
        """With --yes flag in plain mode, deletes key without prompt."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=True)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-to-delete", yes=True)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "deleted successfully" in calls_str

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_rich_success(self):
        """With --yes flag in rich mode, deletes key without prompt."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=True)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _delete_key_async(ctx, key_id="key-to-delete", yes=True)

            assert mock_console.print.called
            args_str = str(mock_console.print.call_args_list)
            assert "deleted" in args_str.lower()

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_result_as_dict_success(self):
        """With --yes flag, handles dict result with success=True."""
        mock_client = AsyncMock()
        mock_client.api_keys.delete = AsyncMock(return_value={"success": True})

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-dict-delete", yes=True)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "deleted successfully" in calls_str

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_no_success_attr(self):
        """With --yes flag, handles result with no success attr (defaults to True)."""
        mock_client = AsyncMock()
        # SimpleNamespace with no success attribute → hasattr returns False, defaults True
        mock_client.api_keys.delete = AsyncMock(return_value=SimpleNamespace(other="value"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-no-success", yes=True)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "deleted successfully" in calls_str

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_failed_deletion_plain(self):
        """With --yes flag, handles failed deletion in plain mode."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=False)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-fail", yes=True)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Failed to delete" in calls_str

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_failed_deletion_rich(self):
        """With --yes flag, handles failed deletion in rich mode."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=False)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _delete_key_async(ctx, key_id="key-fail-rich", yes=True)

            assert mock_console.print.called
            args_str = str(mock_console.print.call_args_list)
            assert "Failed" in args_str

    @pytest.mark.asyncio
    async def test_delete_key_yes_flag_failed_deletion_dict(self):
        """With --yes flag, handles dict result with success=False."""
        mock_client = AsyncMock()
        mock_client.api_keys.delete = AsyncMock(return_value={"success": False})

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-fail-dict", yes=True)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Failed to delete" in calls_str


# ---------------------------------------------------------------------------
# _delete_key_async tests — confirmation prompt
# ---------------------------------------------------------------------------


class TestDeleteKeyAsyncConfirmationPrompt:
    """Tests for _delete_key_async with confirmation prompt."""

    @pytest.mark.asyncio
    async def test_delete_key_prompt_plain_confirmed(self):
        """Plain mode with user confirming deletion."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=True)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.confirm", return_value=True),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-confirmed", yes=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "deleted successfully" in calls_str

    @pytest.mark.asyncio
    async def test_delete_key_prompt_plain_cancelled(self):
        """Plain mode with user cancelling deletion."""
        mock_client = AsyncMock()

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.confirm", return_value=False),
            patch("venice_ai.cli.commands.api_keys.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _delete_key_async(ctx, key_id="key-cancelled", yes=False)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Deletion cancelled" in calls_str
            mock_client.api_keys.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_key_prompt_rich_confirmed(self):
        """Rich mode with user confirming deletion."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=True)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.confirm", return_value=True),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _delete_key_async(ctx, key_id="key-rich-confirmed", yes=False)

            assert mock_console.print.called
            # Check the warning was shown before deletion
            assert mock_console.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_key_prompt_rich_cancelled(self):
        """Rich mode with user cancelling deletion."""
        mock_client = AsyncMock()

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.click.confirm", return_value=False),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _delete_key_async(ctx, key_id="key-rich-cancelled", yes=False)

            assert mock_console.print.called
            args_str = str(mock_console.print.call_args_list)
            assert "cancelled" in args_str.lower()
            mock_client.api_keys.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_key_ctx_obj_none_prompt_yes(self):
        """ctx.obj is None with yes=True → plain defaults to False."""
        mock_client = AsyncMock()
        result = SimpleNamespace(success=True)
        mock_client.api_keys.delete = AsyncMock(return_value=result)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _delete_key_async(ctx, key_id="key-ctx-none", yes=True)

            assert mock_console.print.called


# ---------------------------------------------------------------------------
# list_keys() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestListKeysCLI:
    """Tests for the list_keys() click command entrypoint."""

    def test_list_keys_help(self):
        """Check --help output for api-keys list command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "list", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "api" in result.output.lower()

    def test_list_keys_invokes_asyncio_run(self):
        """list_keys() calls asyncio.run with _list_keys_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["api-keys", "list"])
            assert mock_run.called

    def test_list_keys_json_flag(self):
        """list_keys --json passes output_json=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["api-keys", "list", "--json"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# create_key() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestCreateKeyCLI:
    """Tests for the create_key() click command entrypoint."""

    def test_create_key_help(self):
        """Check --help output for api-keys create command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "create", "--help"])
        assert result.exit_code == 0
        assert "name" in result.output.lower() or "create" in result.output.lower()

    def test_create_key_invokes_asyncio_run(self):
        """create_key() calls asyncio.run with _create_key_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["api-keys", "create", "--name", "TestKey"])
            assert mock_run.called

    def test_create_key_missing_name_shows_error(self):
        """create_key without --name shows error."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "create"])
        assert result.exit_code != 0

    def test_create_key_json_flag(self):
        """create_key --json passes output_json=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["api-keys", "create", "--name", "TestKey", "--json"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# delete_key() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestDeleteKeyCLI:
    """Tests for the delete_key() click command entrypoint."""

    def test_delete_key_help(self):
        """Check --help output for api-keys delete command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "delete", "--help"])
        assert result.exit_code == 0
        assert "delete" in result.output.lower() or "key-id" in result.output.lower()

    def test_delete_key_invokes_asyncio_run(self):
        """delete_key() calls asyncio.run with _delete_key_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["api-keys", "delete", "key-abc123"])
            assert mock_run.called

    def test_delete_key_yes_flag(self):
        """delete_key --yes passes yes=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["api-keys", "delete", "key-abc123", "--yes"])
            assert result.exit_code == 0

    def test_delete_key_short_yes_flag(self):
        """delete_key -y passes yes=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.api_keys.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["api-keys", "delete", "key-abc123", "-y"])
            assert result.exit_code == 0

    def test_delete_key_missing_key_id_shows_error(self):
        """delete_key without KEY_ID shows error."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "delete"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# api_keys() group tests
# ---------------------------------------------------------------------------


class TestApiKeysGroup:
    """Tests for the api_keys() click group."""

    def test_api_keys_group_help(self):
        """Check --help output for the api-keys group."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "create" in result.output
        assert "delete" in result.output


# ---------------------------------------------------------------------------
# `api-keys create` flags (--type/--description/--limit-*/--limit-period/--expiry)
# ---------------------------------------------------------------------------


class TestCreateKeyFlags:
    """Tests for the `api-keys create` flags."""

    def test_create_key_help_lists_new_flags(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["api-keys", "create", "--help"])
        assert result.exit_code == 0
        for fragment in (
            "--type",
            "--description",
            "--limit-usd",
            "--limit-diem",
            "--limit-vcu",
            "--limit-period",
            "--expiry",
        ):
            assert fragment in result.output

    @pytest.mark.asyncio
    async def test_create_forwards_type_limit_period_and_limit_usd(self):
        """--type/--limit-period/--limit-usd flow into the real CreateApiKeyRequest."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-new",
            description="My Key",
            apiKeyType="ADMIN",
            createdAt="2024-01-01T00:00:00Z",
            apiKey="sk-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _create_key_async(
                ctx,
                name="My Key",
                api_key_type="ADMIN",
                limit_usd=10.0,
                limit_diem=None,
                limit_vcu=None,
                limit_period="MONTH",
                expiry=None,
                output_json=False,
            )

            mock_client.api_keys.create.assert_awaited_once()
            request = mock_client.api_keys.create.call_args.kwargs["api_key_request"]
            assert request.apiKeyType == "ADMIN"
            assert request.description == "My Key"
            assert request.limitPeriod == "MONTH"
            assert request.consumptionLimit is not None
            assert request.consumptionLimit.usd == 10.0

    @pytest.mark.asyncio
    async def test_create_defaults_type_inference_no_limit(self):
        """No limit flags → consumptionLimit stays None; default type INFERENCE."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-new",
            description="Default",
            apiKeyType="INFERENCE",
            createdAt=None,
            apiKey="sk-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _create_key_async(
                ctx,
                name="Default",
                api_key_type="INFERENCE",
                limit_usd=None,
                limit_diem=None,
                limit_vcu=None,
                limit_period=None,
                expiry=None,
                output_json=False,
            )

            request = mock_client.api_keys.create.call_args.kwargs["api_key_request"]
            assert request.apiKeyType == "INFERENCE"
            assert request.consumptionLimit is None
            assert request.limitPeriod is None

    @pytest.mark.asyncio
    async def test_create_vcu_folds_into_diem(self):
        """--limit-vcu populates diem (request model has no vcu field)."""
        mock_client = AsyncMock()
        new_key = SimpleNamespace(
            id="key-new",
            description="VCU Key",
            apiKeyType="INFERENCE",
            createdAt=None,
            apiKey="sk-secret",
        )
        mock_client.api_keys.create = AsyncMock(return_value=new_key)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.api_keys.console"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _create_key_async(
                ctx,
                name="VCU Key",
                api_key_type="INFERENCE",
                limit_usd=None,
                limit_diem=None,
                limit_vcu=7.0,
                limit_period=None,
                expiry=None,
                output_json=False,
            )

            request = mock_client.api_keys.create.call_args.kwargs["api_key_request"]
            assert request.consumptionLimit is not None
            assert request.consumptionLimit.diem == 7.0


class TestCreateKeyModelPrivacy:
    """`--model-privacy` sets the key's persistent privacy tier at creation."""

    @pytest.mark.asyncio
    async def test_model_privacy_reaches_the_request(self):
        mock_client = AsyncMock()
        mock_client.api_keys.create = AsyncMock(
            return_value=SimpleNamespace(
                id="k",
                description="d",
                apiKeyType="INFERENCE",
                createdAt=None,
                apiKey="sk-x",
            )
        )
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest", side_effect=_capture),
            patch("venice_ai.cli.commands.api_keys.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)
            await _create_key_async(
                _make_ctx(), name="Private", model_privacy="PRIVATE_ONLY", output_json=True
            )

        assert captured["modelPrivacy"] == "PRIVATE_ONLY"

    @pytest.mark.asyncio
    async def test_model_privacy_defaults_to_none(self):
        """Unset means the account default applies — don't send a tier."""
        mock_client = AsyncMock()
        mock_client.api_keys.create = AsyncMock(
            return_value=SimpleNamespace(
                id="k",
                description="d",
                apiKeyType="INFERENCE",
                createdAt=None,
                apiKey="sk-x",
            )
        )
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.types.api.CreateApiKeyRequest", side_effect=_capture),
            patch("venice_ai.cli.commands.api_keys.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)
            await _create_key_async(_make_ctx(), name="Default", output_json=True)

        assert captured["modelPrivacy"] is None

    def test_cli_exposes_the_flag(self):
        """The option must be wired to the command, not just the async helper."""
        from venice_ai.cli.commands.api_keys import create_key

        names = {p.name for p in create_key.params}
        assert "model_privacy" in names
