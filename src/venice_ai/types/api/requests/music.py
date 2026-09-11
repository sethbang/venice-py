"""Music generation request models for Venice AI API.

Music generation is served through the ``/audio/*`` queue family
(``queue`` / ``quote`` / ``retrieve`` / ``complete``). Each model advertises
its own capabilities (``/models?type=music`` exposes ``supports_lyrics``,
``supports_force_instrumental``, ``supports_language_code``, ``supports_loop``,
``supports_speed``,
``supports_lyrics_optimizer``, and tiered ``min_duration`` / ``max_duration``),
so validation here is intentionally permissive — enforcement happens
server-side against the selected model's capability matrix.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class MusicQueueRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/queue``.

    Example:
        >>> req = MusicQueueRequest(
        ...     model="elevenlabs-music",
        ...     prompt="A warm spoken narration introducing a product launch.",
        ...     lyrics_prompt="Verse 1: Walking through the city lights...",
        ...     duration_seconds=60,
        ...     force_instrumental=False,
        ... )
    """

    model: str = Field(..., description="Music model ID (e.g. 'elevenlabs-music')")
    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Prompt describing the audio. Min / max prompt length varies by model; "
            "see /models for ``min_prompt_length`` and ``prompt_character_limit``."
        ),
    )
    lyrics_prompt: str | None = Field(
        None,
        description=(
            "Optional lyrics for lyric-capable models. Required when ``/models`` "
            "reports ``lyrics_required=true``; rejected when ``supports_lyrics=false``."
        ),
    )
    duration_seconds: int | str | None = Field(
        None,
        description=(
            "Optional output duration hint in seconds. Accepts integer or numeric "
            "string. Applies only to models that expose duration metadata. Must be > 0."
        ),
    )

    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration_seconds(cls, v: Any) -> Any:
        if v is None:
            return v
        try:
            numeric = float(v) if isinstance(v, str) else float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_seconds must be numeric") from exc
        if numeric <= 0:
            raise ValueError("duration_seconds must be > 0")
        return v

    force_instrumental: bool | None = Field(
        None,
        description="Generate without vocals when supported (``supports_force_instrumental``).",
    )
    lyrics_optimizer: bool | None = Field(
        None,
        description=(
            "When true, auto-generates lyrics from the prompt. Requires "
            "``supports_lyrics_optimizer=true``; incompatible with ``lyrics_prompt``."
        ),
    )
    voice: str | None = Field(
        None,
        description="Optional voice selection for voice-enabled models.",
    )
    language_code: str | None = Field(
        None,
        description="Optional ISO 639-1 language code for ``supports_language_code`` models.",
    )
    speed: float | None = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Optional audio speed multiplier; range 0.25–4.",
    )
    loop: bool | None = Field(
        None,
        description=(
            "Render the clip so its end splices back into its start without an "
            "audible seam. Only supported when ``/models`` reports ``supports_loop=true``."
        ),
    )


class MusicQuoteRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/quote``."""

    model: str = Field(..., description="Music model ID")
    duration_seconds: int | str | None = Field(
        None,
        description="Optional output duration hint for the quote (must be > 0).",
    )
    character_count: int | None = Field(
        None,
        gt=0,
        description=(
            "Character count of the prompt + lyrics, when the model charges by chars. Must be > 0."
        ),
    )

    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration_seconds(cls, v: Any) -> Any:
        if v is None:
            return v
        try:
            numeric = float(v) if isinstance(v, str) else float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_seconds must be numeric") from exc
        if numeric <= 0:
            raise ValueError("duration_seconds must be > 0")
        return v


class MusicRetrieveRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/retrieve``.

    Poll until status is ``COMPLETED`` / ``FAILED``, or the endpoint streams
    the audio bytes inline.
    """

    model: str = Field(..., description="Model ID used for generation")
    queue_id: str = Field(..., description="Queue ID from the queue response")
    delete_media_on_completion: bool = Field(
        False, description="Auto-delete media after successful retrieval"
    )


class MusicCompleteRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/complete``.

    Call this after successfully downloading the audio so Venice can clean up
    server-side storage. Skip if ``delete_media_on_completion=True`` was set
    on the retrieve request.
    """

    model: str = Field(..., description="Model ID used for generation")
    queue_id: str = Field(..., description="Queue ID to complete")


__all__ = [
    "MusicQueueRequest",
    "MusicQuoteRequest",
    "MusicRetrieveRequest",
    "MusicCompleteRequest",
]
