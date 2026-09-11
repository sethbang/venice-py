"""
Unit tests for music generation request models.

Mirrors ``test_video_requests.py``. Targets the
``_validate_duration_seconds`` field validator on ``MusicQueueRequest`` and
``MusicQuoteRequest`` — the str/int parse, exception path, and non-positive
rejection are otherwise unexercised.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.music import (
    MusicCompleteRequest,
    MusicQueueRequest,
    MusicQuoteRequest,
    MusicRetrieveRequest,
)


class TestMusicQueueRequest:
    """Validation for ``POST /audio/queue`` body."""

    def test_minimal_request(self):
        request = MusicQueueRequest(
            model="elevenlabs-music",
            prompt="Uplifting cinematic orchestral opener.",
        )  # type: ignore
        assert request.model == "elevenlabs-music"
        assert request.duration_seconds is None
        assert request.force_instrumental is None
        assert request.lyrics_optimizer is None
        assert request.speed is None

    def test_all_fields(self):
        request = MusicQueueRequest(
            model="elevenlabs-music",
            prompt="Verse describing autumn.",
            lyrics_prompt="Verse 1: leaves are falling...",
            duration_seconds=30,
            force_instrumental=False,
            lyrics_optimizer=False,
            voice="alto",
            language_code="en",
            speed=1.25,
        )  # type: ignore
        assert request.duration_seconds == 30
        assert request.lyrics_prompt is not None
        assert request.lyrics_prompt.startswith("Verse 1")
        assert request.speed == 1.25
        assert request.language_code == "en"

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            MusicQueueRequest(model="elevenlabs-music", prompt="")  # type: ignore

    def test_speed_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            MusicQueueRequest(model="elevenlabs-music", prompt="x", speed=0.1)  # type: ignore
        with pytest.raises(ValidationError):
            MusicQueueRequest(model="elevenlabs-music", prompt="x", speed=5.0)  # type: ignore

    @pytest.mark.parametrize("value", [30, "30", "30.5", 30.0])
    def test_duration_seconds_accepts_numeric_str_or_int(self, value):
        request = MusicQueueRequest(
            model="elevenlabs-music",
            prompt="x",
            duration_seconds=value,
        )  # type: ignore
        # Validator returns the original value unchanged on success.
        assert request.duration_seconds == value

    def test_duration_seconds_none_passes(self):
        request = MusicQueueRequest(
            model="elevenlabs-music",
            prompt="x",
            duration_seconds=None,
        )  # type: ignore
        assert request.duration_seconds is None

    def test_duration_seconds_non_numeric_string_rejected(self):
        """Non-numeric string raises via the TypeError/ValueError except branch."""
        with pytest.raises(ValidationError) as exc_info:
            MusicQueueRequest(
                model="elevenlabs-music",
                prompt="x",
                duration_seconds="not-a-number",
            )  # type: ignore
        assert "duration_seconds must be numeric" in str(exc_info.value)

    @pytest.mark.parametrize("bad", [0, -1, "0", "-30", "-0.001"])
    def test_duration_seconds_non_positive_rejected(self, bad):
        """Zero and negative values are rejected by the > 0 check."""
        with pytest.raises(ValidationError) as exc_info:
            MusicQueueRequest(
                model="elevenlabs-music",
                prompt="x",
                duration_seconds=bad,
            )  # type: ignore
        assert "duration_seconds must be > 0" in str(exc_info.value)


class TestMusicQuoteRequest:
    """Validation for ``POST /audio/quote`` body. The ``_validate_duration_seconds``
    validator is duplicated on this model — covering it explicitly prevents drift
    if one validator is updated and the other isn't."""

    def test_minimal_request(self):
        request = MusicQuoteRequest(model="elevenlabs-music")  # type: ignore
        assert request.duration_seconds is None
        assert request.character_count is None

    def test_character_count_must_be_positive(self):
        with pytest.raises(ValidationError):
            MusicQuoteRequest(model="elevenlabs-music", character_count=0)  # type: ignore
        with pytest.raises(ValidationError):
            MusicQuoteRequest(model="elevenlabs-music", character_count=-5)  # type: ignore

    @pytest.mark.parametrize("value", [10, "10", "10.5", 10.0])
    def test_duration_seconds_accepts_numeric_str_or_int(self, value):
        request = MusicQuoteRequest(model="elevenlabs-music", duration_seconds=value)  # type: ignore
        assert request.duration_seconds == value

    def test_duration_seconds_non_numeric_string_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            MusicQuoteRequest(model="elevenlabs-music", duration_seconds="abc")  # type: ignore
        assert "duration_seconds must be numeric" in str(exc_info.value)

    @pytest.mark.parametrize("bad", [0, -1, "0", "-30"])
    def test_duration_seconds_non_positive_rejected(self, bad):
        with pytest.raises(ValidationError) as exc_info:
            MusicQuoteRequest(model="elevenlabs-music", duration_seconds=bad)  # type: ignore
        assert "duration_seconds must be > 0" in str(exc_info.value)


class TestMusicRetrieveAndCompleteRequests:
    """Smoke coverage for the simple polling/cleanup request bodies."""

    def test_retrieve_defaults(self):
        request = MusicRetrieveRequest(model="elevenlabs-music", queue_id="q-123")  # type: ignore
        assert request.delete_media_on_completion is False

    def test_retrieve_with_delete_flag(self):
        request = MusicRetrieveRequest(
            model="elevenlabs-music",
            queue_id="q-123",
            delete_media_on_completion=True,
        )
        assert request.delete_media_on_completion is True

    def test_complete_request(self):
        request = MusicCompleteRequest(model="elevenlabs-music", queue_id="q-123")
        assert request.model == "elevenlabs-music"
        assert request.queue_id == "q-123"


class TestMusicLoopParameter:
    """``loop`` renders a seamlessly-splicing clip on models that support it."""

    def test_loop_defaults_to_none(self):
        """Omitted means "model default" — the SDK must not invent a value."""
        request = MusicQueueRequest(
            model="elevenlabs-music",
            prompt="Ambient pad",
            duration_seconds=30,
            lyrics_prompt=None,
            force_instrumental=None,
            lyrics_optimizer=None,
            voice=None,
            language_code=None,
            speed=None,
            loop=None,
        )
        assert request.loop is None

    def test_loop_accepts_bool(self):
        request = MusicQueueRequest(
            model="elevenlabs-music",
            prompt="Ambient pad",
            duration_seconds=30,
            lyrics_prompt=None,
            force_instrumental=None,
            lyrics_optimizer=None,
            voice=None,
            language_code=None,
            speed=None,
            loop=True,
        )
        assert request.loop is True
