"""Pre-flight duration validation in music.run/submit and video.run/submit.

These tests cover :func:`_preflight_validate_music_duration` and
:func:`_preflight_validate_video_duration` — best-effort client-side checks
that catch obvious enum violations before sending the request.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.music import (
    Music,
    _preflight_validate_music_duration,
)
from venice_ai.resources.video import (
    Video,
    _preflight_validate_video_duration,
)
from venice_ai.types.api.models import (
    ModelResponse,
    MusicModelSpec,
    VideoModelConstraints,
    VideoModelSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _music_entry(
    model_id: str,
    *,
    duration_options: list[int] | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    default_duration: int | None = None,
) -> ModelResponse:
    spec = MusicModelSpec(
        name=model_id,
        duration_options=duration_options,
        min_duration=min_duration,
        max_duration=max_duration,
        default_duration=default_duration,
    )
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "type": "music",
            "object": "model",
            "owned_by": "venice.ai",
            "model_spec": spec.model_dump(),
        }
    )


def _video_entry(model_id: str, *, durations: list[str]) -> ModelResponse:
    spec = VideoModelSpec(
        name=model_id,
        constraints=VideoModelConstraints(model_type="text-to-video", durations=durations),
    )
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "type": "video",
            "object": "model",
            "owned_by": "venice.ai",
            "model_spec": spec.model_dump(),
        }
    )


def _client_with_model(entry: ModelResponse) -> Mock:
    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(return_value=entry)
    return client


# ---------------------------------------------------------------------------
# Music — duration_options enum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_music_duration_in_enum_passes() -> None:
    client = _client_with_model(
        _music_entry("ace-step-15", duration_options=[60, 90, 120, 150, 180, 210])
    )
    await _preflight_validate_music_duration(client, "ace-step-15", 90)
    # No raise = pass


@pytest.mark.asyncio
async def test_music_duration_not_in_enum_raises() -> None:
    """The agent's exact failure mode: 30 is not in [60, 90, ...]."""
    client = _client_with_model(
        _music_entry("ace-step-15", duration_options=[60, 90, 120, 150, 180, 210])
    )
    with pytest.raises(ValueError, match=r"\[60, 90, 120, 150, 180, 210\]"):
        await _preflight_validate_music_duration(client, "ace-step-15", 30)


@pytest.mark.asyncio
async def test_music_duration_string_form_validated() -> None:
    """Stringified ints get coerced before enum check."""
    client = _client_with_model(_music_entry("ace-step-15", duration_options=[60, 90, 120]))
    with pytest.raises(ValueError, match="not a supported value"):
        await _preflight_validate_music_duration(client, "ace-step-15", "30")


# ---------------------------------------------------------------------------
# Music — min/max range fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_music_duration_below_min_raises() -> None:
    client = _client_with_model(_music_entry("elevenlabs-music", min_duration=10, max_duration=300))
    with pytest.raises(ValueError, match="below the minimum 10"):
        await _preflight_validate_music_duration(client, "elevenlabs-music", 5)


@pytest.mark.asyncio
async def test_music_duration_above_max_raises() -> None:
    client = _client_with_model(_music_entry("elevenlabs-music", min_duration=10, max_duration=300))
    with pytest.raises(ValueError, match="above the maximum 300"):
        await _preflight_validate_music_duration(client, "elevenlabs-music", 400)


@pytest.mark.asyncio
async def test_music_duration_within_range_passes() -> None:
    client = _client_with_model(_music_entry("elevenlabs-music", min_duration=10, max_duration=300))
    await _preflight_validate_music_duration(client, "elevenlabs-music", 60)


# ---------------------------------------------------------------------------
# Music — best-effort behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_music_duration_none_skips_lookup() -> None:
    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(side_effect=AssertionError("should not be called"))
    await _preflight_validate_music_duration(client, "elevenlabs-music", None)


@pytest.mark.asyncio
async def test_music_catalog_miss_falls_through() -> None:
    """If models.get raises (network/auth), validation silently defers to server."""
    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(side_effect=RuntimeError("catalog unreachable"))
    await _preflight_validate_music_duration(client, "elevenlabs-music", 60)


# ---------------------------------------------------------------------------
# Video — string-tier enum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_duration_in_enum_passes() -> None:
    client = _client_with_model(_video_entry("wan-2", durations=["5s", "10s"]))
    await _preflight_validate_video_duration(client, "wan-2", "5s")


@pytest.mark.asyncio
async def test_video_duration_not_in_enum_raises() -> None:
    client = _client_with_model(_video_entry("wan-2", durations=["5s", "10s"]))
    with pytest.raises(ValueError, match=r"\['5s', '10s'\]"):
        await _preflight_validate_video_duration(client, "wan-2", "30s")


@pytest.mark.asyncio
async def test_video_duration_none_skips_lookup() -> None:
    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(side_effect=AssertionError("should not be called"))
    await _preflight_validate_video_duration(client, "wan-2", None)


@pytest.mark.asyncio
async def test_video_catalog_miss_falls_through() -> None:
    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(side_effect=RuntimeError("catalog unreachable"))
    await _preflight_validate_video_duration(client, "wan-2", "5s")


@pytest.mark.asyncio
async def test_video_empty_durations_falls_through() -> None:
    """Some i2v models have empty durations because they inherit from input."""
    client = _client_with_model(_video_entry("wan-i2v", durations=[]))
    await _preflight_validate_video_duration(client, "wan-i2v", "5s")


# ---------------------------------------------------------------------------
# Sanity: helper modules export the validators
# ---------------------------------------------------------------------------


def test_music_helper_is_module_level() -> None:
    assert callable(_preflight_validate_music_duration)


# ---------------------------------------------------------------------------
# Music — force_instrumental capability preflight
# ---------------------------------------------------------------------------


def _music_entry_with_force_instrumental(
    model_id: str, *, supports_force_instrumental: bool | None
) -> ModelResponse:
    spec = MusicModelSpec(
        name=model_id,
        supports_force_instrumental=supports_force_instrumental,
    )
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "type": "music",
            "object": "model",
            "owned_by": "venice.ai",
            "model_spec": spec.model_dump(),
        }
    )


@pytest.mark.asyncio
async def test_force_instrumental_unsupported_raises() -> None:
    """The agent's exact failure: ``ace-step-15`` returns
    ``This model does not support force_instrumental``. Catch it client-side."""
    from venice_ai.resources.music import _preflight_validate_force_instrumental

    client = _client_with_model(
        _music_entry_with_force_instrumental("ace-step-15", supports_force_instrumental=False)
    )
    with pytest.raises(ValueError, match="force_instrumental"):
        await _preflight_validate_force_instrumental(client, "ace-step-15", True)


@pytest.mark.asyncio
async def test_force_instrumental_unsupported_blocks_false_too() -> None:
    """Even ``force_instrumental=False`` should be rejected when the model doesn't
    accept the field at all — the API rejects the *presence* of the param,
    not the value."""
    from venice_ai.resources.music import _preflight_validate_force_instrumental

    client = _client_with_model(
        _music_entry_with_force_instrumental("ace-step-15", supports_force_instrumental=False)
    )
    with pytest.raises(ValueError, match="force_instrumental"):
        await _preflight_validate_force_instrumental(client, "ace-step-15", False)


@pytest.mark.asyncio
async def test_force_instrumental_supported_passes() -> None:
    from venice_ai.resources.music import _preflight_validate_force_instrumental

    client = _client_with_model(
        _music_entry_with_force_instrumental("elevenlabs-music", supports_force_instrumental=True)
    )
    await _preflight_validate_force_instrumental(client, "elevenlabs-music", True)


@pytest.mark.asyncio
async def test_force_instrumental_unknown_capability_falls_through() -> None:
    """If the spec doesn't declare ``supports_force_instrumental`` (None), defer
    to the server — a missing capability flag is not the same as ``False``."""
    from venice_ai.resources.music import _preflight_validate_force_instrumental

    client = _client_with_model(
        _music_entry_with_force_instrumental("mystery-model", supports_force_instrumental=None)
    )
    await _preflight_validate_force_instrumental(client, "mystery-model", True)


@pytest.mark.asyncio
async def test_force_instrumental_none_skips_lookup() -> None:
    """Caller didn't pass force_instrumental — never hit the catalog."""
    from venice_ai.resources.music import _preflight_validate_force_instrumental

    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(side_effect=AssertionError("should not be called"))
    await _preflight_validate_force_instrumental(client, "any-model", None)


@pytest.mark.asyncio
async def test_force_instrumental_catalog_miss_falls_through() -> None:
    """Network/auth/etc. when fetching the model spec defers to server validation."""
    from venice_ai.resources.music import _preflight_validate_force_instrumental

    client = Mock()
    client.models = Mock()
    client.models.get = AsyncMock(side_effect=RuntimeError("catalog unreachable"))
    await _preflight_validate_force_instrumental(client, "ace-step-15", True)


@pytest.mark.asyncio
async def test_music_submit_invokes_force_instrumental_preflight() -> None:
    """End-to-end: ``Music.submit(force_instrumental=True, model=...)`` must
    raise client-side when the model spec disallows the flag — without making
    the queue POST.

    This is the integration assertion the agent's report effectively asks for:
    the value should be checked against the spec at submit time, not after a
    round-trip to the server."""
    client = _client_with_model(
        _music_entry_with_force_instrumental("ace-step-15", supports_force_instrumental=False)
    )
    client.post = AsyncMock(side_effect=AssertionError("should not POST when preflight fails"))
    music = Music(client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="force_instrumental"):
        await music.submit(
            model="ace-step-15",
            prompt="a simple test",
            force_instrumental=True,
        )
    assert callable(_preflight_validate_video_duration)


def test_resources_still_importable() -> None:
    """No regressions to top-level imports."""
    assert Music is not None
    assert Video is not None


# ---------------------------------------------------------------------------
# Music — loop capability preflight
# ---------------------------------------------------------------------------


def _music_entry_with_loop(model_id: str, *, supports_loop: bool | None) -> ModelResponse:
    spec = MusicModelSpec(name=model_id, supports_loop=supports_loop)
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "type": "music",
            "object": "model",
            "owned_by": "venice.ai",
            "model_spec": spec.model_dump(),
        }
    )


@pytest.mark.asyncio
async def test_loop_unsupported_raises() -> None:
    from venice_ai.resources.music import _preflight_validate_loop

    client = _client_with_model(_music_entry_with_loop("ace-step-15", supports_loop=False))
    with pytest.raises(ValueError, match="loop"):
        await _preflight_validate_loop(client, "ace-step-15", True)


@pytest.mark.asyncio
async def test_loop_unsupported_blocks_false_too() -> None:
    """The API rejects the presence of the field, not just a True value."""
    from venice_ai.resources.music import _preflight_validate_loop

    client = _client_with_model(_music_entry_with_loop("ace-step-15", supports_loop=False))
    with pytest.raises(ValueError, match="loop"):
        await _preflight_validate_loop(client, "ace-step-15", False)


@pytest.mark.asyncio
async def test_loop_supported_passes() -> None:
    from venice_ai.resources.music import _preflight_validate_loop

    client = _client_with_model(_music_entry_with_loop("elevenlabs-music", supports_loop=True))
    await _preflight_validate_loop(client, "elevenlabs-music", True)


@pytest.mark.asyncio
async def test_loop_unknown_capability_falls_through() -> None:
    """An undeclared capability defers to the server; it is not a declared False."""
    from venice_ai.resources.music import _preflight_validate_loop

    client = _client_with_model(_music_entry_with_loop("mystery-model", supports_loop=None))
    await _preflight_validate_loop(client, "mystery-model", True)
