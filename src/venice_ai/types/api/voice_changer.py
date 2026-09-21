"""Voice-changer response models for Venice AI API.

Endpoints covered:
- POST /api/v1/audio/voice-changer/queue    -> VoiceChangerQueueResponse
- POST /api/v1/audio/voice-changer/quote    -> VoiceChangerQuoteResponse
- POST /api/v1/audio/voice-changer/retrieve -> VoiceChangerRetrieveResponse (union)
- POST /api/v1/audio/voice-changer/complete -> VoiceChangerCompleteResponse
"""

from typing import Literal

from pydantic import ConfigDict, Field, PrivateAttr

from ...core.models.common import VeniceBaseModel


class VoiceChangerQueueResponse(VeniceBaseModel):
    """Response from ``POST /api/v1/audio/voice-changer/queue``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model running the conversion.")
    queue_id: str = Field(
        ...,
        description="Pass to ``retrieve`` to poll for the audio, and to ``complete`` when done.",
    )
    status: Literal["QUEUED"] = Field(
        ..., description="Always ``QUEUED`` — the provider has accepted the job."
    )
    duration_seconds: float = Field(
        ...,
        description=(
            "Length of the source recording in seconds, measured server-side, and "
            "therefore the exact quantity billed for this request. Reconcile it "
            "against the estimate from ``quote`` — that one is only an estimate."
        ),
    )


class VoiceChangerQuoteResponse(VeniceBaseModel):
    """Response from ``POST /api/v1/audio/voice-changer/quote``."""

    model_config = ConfigDict(extra="allow")

    quote: float = Field(
        ..., description="Estimated price in USD for converting a recording of this length."
    )
    duration_seconds: float = Field(
        ..., description="The source length this quote was computed for."
    )


class VoiceChangerProcessingStatus(VeniceBaseModel):
    """The conversion is still running — poll again."""

    model_config = ConfigDict(extra="allow")

    status: Literal["PROCESSING"] = Field(..., description="Conversion still running.")
    average_execution_time: float = Field(
        ...,
        description=(
            "Recent average end-to-end time for this model, in milliseconds. "
            "Use it to pace polling."
        ),
    )
    execution_duration: float = Field(
        ..., description="Milliseconds elapsed since the conversion was queued."
    )

    @property
    def progress_percent(self) -> float:
        """Rough completion estimate, clamped to 0-100.

        Derived from elapsed time against the model's recent average, so it is
        a pacing hint rather than real progress — a slower-than-average run
        sits at 100 while still processing.
        """
        if self.average_execution_time <= 0:
            return 0.0
        return min(100.0, 100.0 * self.execution_duration / self.average_execution_time)


class VoiceChangerCompletedStatus(VeniceBaseModel):
    """The converted audio, delivered inline as bytes.

    Unlike video, this endpoint has **no JSON completed arm**: there is no
    ``url`` to download from and no ``expires_at``. A finished conversion comes
    back as an ``audio/mpeg`` body, which the resource attaches here.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["COMPLETED"] = Field(
        default="COMPLETED",
        description=(
            "Synthesised by the SDK. The server signals completion with an "
            "audio body and no status field of its own."
        ),
    )

    # Binary audio. A PrivateAttr because it is set programmatically from the
    # HTTP body rather than parsed from JSON.
    _data: bytes | None = PrivateAttr(default=None)

    def _set_data(self, data: bytes) -> None:
        """Attach the raw audio bytes (``object.__setattr__`` for PrivateAttr)."""
        object.__setattr__(self, "_data", data)

    @property
    def data(self) -> bytes | None:
        """Raw converted audio bytes."""
        return self._data


class VoiceChangerCompleteResponse(VeniceBaseModel):
    """Response from ``POST /api/v1/audio/voice-changer/complete``."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(
        ..., description="Whether the provider-held media for this conversion was released."
    )


# The retrieve endpoint has exactly TWO arms, not the three the video family
# has. The spec declares a PROCESSING JSON response and an audio/mpeg body —
# there is no FAILED status and no COMPLETED JSON arm. Failures surface as
# HTTP errors (404 media gone, 500 inference failed, 504 never reached the
# provider), carrying ``credits_refunded``.
VoiceChangerRetrieveResponse = VoiceChangerProcessingStatus | VoiceChangerCompletedStatus


__all__ = [
    "VoiceChangerQueueResponse",
    "VoiceChangerQuoteResponse",
    "VoiceChangerProcessingStatus",
    "VoiceChangerCompletedStatus",
    "VoiceChangerCompleteResponse",
    "VoiceChangerRetrieveResponse",
]
