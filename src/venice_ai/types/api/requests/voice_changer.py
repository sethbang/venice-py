"""Voice-changer request models for Venice AI API.

Voice changing is served through its own queue family under
``/audio/voice-changer/*`` (``queue`` / ``quote`` / ``retrieve`` /
``complete``), parallel to the music family on ``/audio/*``. It converts an
existing recording into a target voice, preserving the original timing —
which is why the *source* length is the billable quantity.

Voice-changer models are not a distinct model type: they report
``type="music"`` with ``voice_changer: true`` on ``model_spec``. Capability
details (``voices``, ``accepted_audio_formats``,
``max_source_audio_duration_seconds``, ``supports_seed``,
``supports_background_noise_removal``) come from ``/models``, so validation
here stays permissive and the server enforces against the selected model.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QueueVoiceChangerRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/voice-changer/queue``.

    The source recording is supplied either as ``file`` (multipart) or as
    ``audio_url``, never both. The spec states the exclusivity in prose only,
    so it is enforced here.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        description=(
            "Voice-changer model ID. Models that are not voice changers are "
            "rejected here — use the /audio endpoints for those."
        ),
    )
    file: Any | None = Field(
        default=None,
        description=(
            "Source recording, sent as multipart form data. Accepted containers "
            "are listed per model as ``accepted_audio_formats`` in /models. "
            "Mutually exclusive with ``audio_url``."
        ),
    )
    audio_url: str | None = Field(
        default=None,
        description=(
            "Publicly reachable http(s) URL of the source recording. Venice "
            "fetches and validates the bytes itself and forwards only those "
            "bytes to the provider, so the URL is never handed onward. "
            "Mutually exclusive with ``file``."
        ),
    )
    voice: str | None = Field(
        default=None,
        description=(
            "Target voice: one of the model's ``voices`` from /models, or a "
            "provider Voice ID when the model reports "
            "``supports_custom_voice_id``. Defaults to the model's "
            "``default_voice``."
        ),
    )
    remove_background_noise: bool | None = Field(
        default=None,
        description="Strip background noise from the source before conversion.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Seed for reproducible output. Omit for a non-deterministic result.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "QueueVoiceChangerRequest":
        if self.file is not None and self.audio_url is not None:
            raise ValueError("Provide either file or audio_url, not both")
        if self.file is None and self.audio_url is None:
            raise ValueError("A source recording is required: pass file or audio_url")
        return self


class QuoteVoiceChangerRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/voice-changer/quote``.

    ``duration_seconds`` is a **bare count of seconds** — an integer or a
    digits-only string. It is not the ``"60s"`` form the video endpoints take,
    and must not be run through the video duration formatter.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Voice-changer model ID to price.")
    duration_seconds: int | str = Field(
        ...,
        description=(
            "Length of the source recording in seconds. Voice changing preserves "
            "timing, so the source length is the billable quantity, priced in "
            "whole-minute tiers. This is an estimate — the charge is computed "
            "from the length Venice measures when the recording is queued. The "
            "per-model ceiling is ``max_source_audio_duration_seconds`` in /models."
        ),
    )

    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration_seconds(cls, v: int | str) -> int | str:
        """Accept ``int`` or a digits-only string, both strictly positive."""
        if isinstance(v, str):
            if not v.isdigit():
                raise ValueError(
                    "duration_seconds must be a whole number of seconds "
                    f"(digits only, no unit suffix); got {v!r}"
                )
            parsed = int(v)
        else:
            parsed = v
        if parsed <= 0:
            raise ValueError(f"duration_seconds must be greater than 0; got {parsed}")
        return v


class RetrieveVoiceChangerRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/voice-changer/retrieve``."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="The model running the conversion.")
    queue_id: str = Field(..., description="Queue ID returned by the queue call.")
    delete_media_on_completion: bool = Field(
        default=False,
        description=(
            "Release the provider-held media as soon as this call returns the "
            "audio, making a separate ``complete`` call unnecessary. The audio "
            "cannot be retrieved again afterwards."
        ),
    )


class CompleteVoiceChangerRequest(BaseModel):
    """Request body for ``POST /api/v1/audio/voice-changer/complete``."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="The model running the conversion.")
    queue_id: str = Field(..., description="Queue ID returned by the queue call.")


__all__ = [
    "QueueVoiceChangerRequest",
    "QuoteVoiceChangerRequest",
    "RetrieveVoiceChangerRequest",
    "CompleteVoiceChangerRequest",
]
