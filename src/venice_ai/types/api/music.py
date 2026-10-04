"""Music generation response models for Venice AI API.

Endpoints covered (all under the ``/audio/*`` namespace — music generation
shares the queue family with video-style async generation):

- ``POST /api/v1/audio/queue``     -> :class:`MusicQueueResponse`
- ``POST /api/v1/audio/quote``     -> :class:`MusicQuoteResponse`
- ``POST /api/v1/audio/retrieve``  -> :data:`MusicRetrieveResponse` (union)
- ``POST /api/v1/audio/complete``  -> :class:`MusicCompleteResponse`
"""

from typing import Literal

from pydantic import ConfigDict, Field, PrivateAttr

from ...core.models.common import VeniceBaseModel
from .audio_format import audio_format_for


class MusicQueueResponse(VeniceBaseModel):
    """Response from the music queue endpoint.

    ``POST /api/v1/audio/queue`` -> ``200 OK``. Contains the ``queue_id``
    needed to poll for results.
    """

    model: str = Field(..., description="Model ID used for generation")
    queue_id: str = Field(..., description="Unique queue ID for tracking")
    status: Literal["QUEUED"] = Field(default="QUEUED", description="Initial status")


class MusicQuoteResponse(VeniceBaseModel):
    """Response from the music quote endpoint.

    ``POST /api/v1/audio/quote`` -> ``200 OK``. Provides estimated cost
    before queuing. Quote may be returned as an integer or float.
    """

    model_config = ConfigDict(extra="allow")

    quote: int | float = Field(..., description="Estimated cost in USD")


class MusicProcessingStatus(VeniceBaseModel):
    """Status response when music is still being generated.

    ``POST /api/v1/audio/retrieve`` -> ``200 OK`` (while processing).
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["PROCESSING"] = Field(..., description="Current status")
    average_execution_time: float = Field(..., description="Average execution time in ms")
    execution_duration: float = Field(..., description="Current execution duration in ms")

    @property
    def progress_percent(self) -> float:
        """Estimate progress as a 0–100 percentage.

        Elapsed time divided by the model's typical run time, capped at 100.
        It is a time estimate, not a report from the job: a job that runs
        longer than usual sits at 100 while still ``PROCESSING``.
        """
        if self.average_execution_time <= 0:
            return 0.0
        return min(100.0, (self.execution_duration / self.average_execution_time) * 100)

    @property
    def estimated_remaining_ms(self) -> float:
        return max(0.0, self.average_execution_time - self.execution_duration)


class MusicFailedStatus(VeniceBaseModel):
    """Status response when music generation failed."""

    model_config = ConfigDict(extra="allow")

    status: Literal["FAILED"] = Field(..., description="Failed status")
    error: str | None = Field(default=None, description="Error message if available")
    error_code: str | None = Field(default=None, description="Error code if available")


class MusicCompletedStatus(VeniceBaseModel):
    """Status response when music generation is complete.

    The completed audio may be delivered in one of two ways:

    1. **JSON with download URL** — ``url`` is set; download separately.
    2. **Inline binary** — the API streams the audio bytes directly.
       In this case ``url`` is ``None`` and the raw bytes are accessible
       via :attr:`data`.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["COMPLETED"] = Field(..., description="Completed status")
    url: str | None = Field(default=None, description="Download URL if provided")
    expires_at: str | None = Field(default=None, description="URL expiration timestamp")
    content_type: str | None = Field(
        default=None,
        description=(
            "Media type of inline audio, from the retrieve response's Content-Type "
            "(e.g. ``audio/flac``). ``None`` for a JSON status."
        ),
    )

    _data: bytes | None = PrivateAttr(default=None)

    @property
    def audio_format(self) -> str | None:
        """File extension for the inline audio (``"mp3"``, ``"flac"``, ``"wav"``, ``"m4a"``).

        Derived from :attr:`content_type`, so a caller can name the file
        without inspecting the bytes. ``None`` when the type is unknown or the
        status carried no inline audio.
        """
        return audio_format_for(self.content_type)

    def _set_data(self, data: bytes) -> None:
        """Set binary audio data (Pydantic PrivateAttr-compatible setter)."""
        object.__setattr__(self, "_data", data)

    @property
    def data(self) -> bytes | None:
        """Raw audio bytes when delivered inline, else ``None``."""
        return self._data


class MusicCompleteResponse(VeniceBaseModel):
    """Response from the music complete/cleanup endpoint.

    ``POST /api/v1/audio/complete`` -> ``200 OK``.
    """

    model_config = ConfigDict(extra="allow")

    success: bool = Field(..., description="Whether cleanup succeeded")


MusicRetrieveResponse = MusicProcessingStatus | MusicFailedStatus | MusicCompletedStatus
"""Retrieve can return processing / failed / completed JSON, or inline bytes
(bytes are handled at the HTTP layer before this union applies)."""


__all__ = [
    "MusicQueueResponse",
    "MusicQuoteResponse",
    "MusicProcessingStatus",
    "MusicFailedStatus",
    "MusicCompletedStatus",
    "MusicCompleteResponse",
    "MusicRetrieveResponse",
]
