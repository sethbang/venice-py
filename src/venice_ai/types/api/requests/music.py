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

from pydantic import BaseModel, Field, field_validator, model_validator

#: The ``voice`` value that leaves the voice to the prompt; the only preset
#: allowed alongside ``reference_audio_urls``.
_DESCRIBE_IN_PROMPT = "Describe in prompt"


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
        default=None,
        description=(
            "Optional lyrics for lyric-capable models. Required when ``/models`` "
            "reports ``lyrics_required=true``; rejected when ``supports_lyrics=false``."
        ),
    )
    duration_seconds: int | str | None = Field(
        default=None,
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
        default=None,
        description="Generate without vocals when supported (``supports_force_instrumental``).",
    )
    lyrics_optimizer: bool | None = Field(
        default=None,
        description=(
            "When true, auto-generates lyrics from the prompt. Requires "
            "``supports_lyrics_optimizer=true``; incompatible with ``lyrics_prompt``."
        ),
    )
    voice: str | None = Field(
        default=None,
        description="Optional voice selection for voice-enabled models.",
    )
    language_code: str | None = Field(
        default=None,
        description="Optional ISO 639-1 language code for ``supports_language_code`` models.",
    )
    speed: float | None = Field(
        default=None,
        ge=0.25,
        le=4.0,
        description="Optional audio speed multiplier; range 0.25–4.",
    )
    loop: bool | None = Field(
        default=None,
        description=(
            "Render the clip so its end splices back into its start without an "
            "audible seam. Only supported when ``/models`` reports ``supports_loop=true``."
        ),
    )
    reference_audio_urls: list[str] | None = Field(
        default=None,
        description=(
            "Reference clips the prompt names in order as ``@Audio1``, ``@Audio2``... "
            "Each is a base64 ``data:`` URL of audio without video (at most 30 s and "
            "10 MB). Only for models whose ``/models`` entry reports "
            "``supports_reference_audios=true``, up to ``max_reference_audios``. "
            "Cannot be combined with a preset ``voice``."
        ),
    )

    @field_validator("reference_audio_urls")
    @classmethod
    def _require_audio_data_urls(cls, v: list[str] | None) -> list[str] | None:
        """The API takes inline audio only; it does not fetch http(s) URLs.

        An empty list is dropped, so a model without the capability never sees the field.
        """
        if not v:
            return None
        for url in v:
            if not url.startswith("data:"):
                raise ValueError(
                    "each reference_audio_urls entry must be a base64 data: URL "
                    "(data:audio/mpeg;base64,...); http(s) URLs are not supported."
                )
        return v

    @model_validator(mode="after")
    def _reject_reference_audio_with_preset_voice(self) -> "MusicQueueRequest":
        """Reference clips replace the preset voice, so the two cannot be combined."""
        if self.reference_audio_urls and self.voice not in (None, _DESCRIBE_IN_PROMPT):
            raise ValueError(
                "reference_audio_urls cannot be combined with a preset voice; omit "
                f"voice or set it to {_DESCRIBE_IN_PROMPT!r}."
            )
        return self


class MusicQuoteRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/quote``."""

    model: str = Field(..., description="Music model ID")
    duration_seconds: int | str | None = Field(
        default=None,
        description="Optional output duration hint for the quote (must be > 0).",
    )
    character_count: int | None = Field(
        default=None,
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
        default=False, description="Auto-delete media after successful retrieval"
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
