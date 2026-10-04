"""
Video generation response models for Venice AI API.

Endpoints covered:
- POST /api/v1/video/queue -> VideoQueueResponse
- POST /api/v1/video/quote -> VideoQuoteResponse
- POST /api/v1/video/retrieve -> VideoRetrieveResponse (union type)
- POST /api/v1/video/complete -> VideoCompleteResponse
"""

from typing import Literal

from pydantic import ConfigDict, Field, PrivateAttr

from ...core.models.common import VeniceBaseModel


class VideoQueueResponse(VeniceBaseModel):
    """
    Response from queue video generation.

    POST /api/v1/video/queue -> 200 OK

    Contains the queue_id needed to poll for results.
    """

    model: str = Field(..., description="Model ID used for generation")
    queue_id: str = Field(..., description="Unique queue ID for tracking")
    download_url: str | None = Field(
        default=None,
        description=(
            "Pre-signed URL to download the completed video. Only present for "
            "VPS-backed models; valid for 24 hours. When provided, /video/retrieve "
            "returns JSON status only — fetch this URL after COMPLETED to get the "
            "video/mp4 file."
        ),
    )


class VideoQuoteResponse(VeniceBaseModel):
    """
    Response from video quote endpoint.

    POST /api/v1/video/quote -> 200 OK

    Provides estimated cost before queuing actual generation.
    Note: Quote may be returned as integer or float depending on value.
    """

    quote: int | float = Field(..., description="Estimated cost in USD")


class VideoProcessingStatus(VeniceBaseModel):
    """
    Status response when video is still being generated.

    POST /api/v1/video/retrieve -> 200 OK (when processing)

    Returned when the video is not yet ready. Poll again after delay.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["PROCESSING"] = Field(..., description="Current status")
    average_execution_time: float = Field(..., description="Average execution time in milliseconds")
    execution_duration: float = Field(..., description="Current execution duration in milliseconds")

    @property
    def progress_percent(self) -> float:
        """Estimate progress as percentage (0-100)."""
        if self.average_execution_time <= 0:
            return 0.0
        return min(100.0, (self.execution_duration / self.average_execution_time) * 100)

    @property
    def estimated_remaining_ms(self) -> float:
        """Estimate remaining time in milliseconds."""
        return max(0.0, self.average_execution_time - self.execution_duration)


class VideoFailedStatus(VeniceBaseModel):
    """
    Status response when video generation failed.

    POST /api/v1/video/retrieve -> 200 OK (when failed)

    Unknown error fields returned by the API are preserved via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["FAILED"] = Field(..., description="Failed status")
    error: str | None = Field(default=None, description="Error message if available")
    error_code: str | None = Field(default=None, description="Error code if available")


class VideoCompletedStatus(VeniceBaseModel):
    """
    Status response when video generation is complete.

    The completed video may be delivered in one of two ways:

    1. **JSON with download URL** — ``url`` is set; download separately.
    2. **Inline binary** — the API streams the video bytes directly.
       In this case ``url`` is ``None`` and the raw bytes are accessible
       via the :attr:`data` property.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["COMPLETED"] = Field(..., description="Completed status")
    url: str | None = Field(default=None, description="Download URL if provided")
    expires_at: str | None = Field(default=None, description="URL expiration timestamp")

    # Binary video data when the API returns the video inline (non-JSON).
    # Stored as a PrivateAttr because it is set programmatically, not
    # parsed from JSON, and we don't want Pydantic validation to interfere.
    _data: bytes | None = PrivateAttr(default=None)

    def _set_data(self, data: bytes) -> None:
        """Set binary video data. Uses object.__setattr__ for Pydantic PrivateAttr compatibility."""
        object.__setattr__(self, "_data", data)

    @property
    def data(self) -> bytes | None:
        """Raw video bytes when delivered inline, else ``None``."""
        return self._data


class VideoCompleteResponse(VeniceBaseModel):
    """
    Response from complete/cleanup video endpoint.

    POST /api/v1/video/complete -> 200 OK
    """

    success: bool = Field(..., description="Whether cleanup succeeded")


# Type alias for retrieve response
# The retrieve endpoint can return:
# 1. VideoProcessingStatus - when still processing
# 2. VideoFailedStatus - when generation failed
# 3. VideoCompletedStatus - when complete (JSON response)
# 4. bytes - when complete (binary video data, handled at HTTP level)
VideoRetrieveResponse = VideoProcessingStatus | VideoFailedStatus | VideoCompletedStatus


class VideoTranscriptionResponse(VeniceBaseModel):
    """Response from ``POST /api/v1/video/transcriptions`` (JSON format)."""

    transcript: str = Field(..., description="The transcribed text from the video")
    lang: str | None = Field(
        default=None, description="Detected language code for the transcript (e.g. 'en')"
    )


__all__ = [
    "VideoQueueResponse",
    "VideoQuoteResponse",
    "VideoProcessingStatus",
    "VideoFailedStatus",
    "VideoCompletedStatus",
    "VideoCompleteResponse",
    "VideoRetrieveResponse",
    "VideoTranscriptionResponse",
]
