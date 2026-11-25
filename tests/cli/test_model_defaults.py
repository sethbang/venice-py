"""Tests for the CLI runtime model-default resolver."""

from unittest.mock import AsyncMock, MagicMock

import click
import pytest

from venice_ai.cli._model_defaults import resolve_default_model


@pytest.mark.asyncio
async def test_explicit_flag_wins():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="should-not-be-called")
    result = await resolve_default_model(client, {"defaults": {}}, "chat", explicit="my-model")
    assert result == "my-model"
    client.models.resolve.assert_not_called()


@pytest.mark.asyncio
async def test_saved_config_default_used_when_no_flag():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="api-model")
    config = {"defaults": {"chat_model": "saved-model"}}
    result = await resolve_default_model(client, config, "chat", explicit=None)
    assert result == "saved-model"
    client.models.resolve.assert_not_called()


@pytest.mark.asyncio
async def test_api_resolution_when_no_flag_or_config():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="api-model")
    result = await resolve_default_model(client, {"defaults": {}}, "chat", explicit=None)
    assert result == "api-model"
    client.models.resolve.assert_awaited_once_with(type="chat")


@pytest.mark.asyncio
async def test_video_t2v_passes_video_type():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="t2v-model")
    result = await resolve_default_model(client, {"defaults": {}}, "video_t2v", explicit=None)
    assert result == "t2v-model"
    client.models.resolve.assert_awaited_once_with(type="video", video_type="text-to-video")


@pytest.mark.asyncio
async def test_stt_maps_to_asr():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="asr-model")
    await resolve_default_model(client, {"defaults": {}}, "stt", explicit=None)
    client.models.resolve.assert_awaited_once_with(type="asr")


@pytest.mark.asyncio
async def test_decision_resolves_decision_type():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="decision-model")
    result = await resolve_default_model(client, {"defaults": {}}, "decision", explicit=None)
    assert result == "decision-model"
    client.models.resolve.assert_awaited_once_with(type="decision")


@pytest.mark.asyncio
async def test_offline_raises_clickexception():
    client = MagicMock()
    client.models.resolve = AsyncMock(side_effect=RuntimeError("network down"))
    with pytest.raises(click.ClickException) as exc:
        await resolve_default_model(client, {"defaults": {}}, "chat", explicit=None)
    assert "--model" in str(exc.value)
    assert "venice-py configure" in str(exc.value)


@pytest.mark.asyncio
async def test_requirements_narrow_the_api_resolution():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="hdr-model")
    result = await resolve_default_model(
        client, {"defaults": {}}, "video_t2v", requirements={"require_hdr": True}
    )
    assert result == "hdr-model"
    client.models.resolve.assert_awaited_once_with(
        type="video", video_type="text-to-video", require_hdr=True
    )


@pytest.mark.asyncio
async def test_requirements_leave_a_saved_default_alone():
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value="api-model")
    config = {"defaults": {"video_t2v_model": "saved-model"}}
    result = await resolve_default_model(
        client, config, "video_t2v", requirements={"require_hdr": True}
    )
    assert result == "saved-model"
    client.models.resolve.assert_not_called()
