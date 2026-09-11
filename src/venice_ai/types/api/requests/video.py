"""
Video generation request models for Venice AI API.

This module provides separate request models for:
- Text-to-Video (T2V): VideoTextToVideoRequest
- Image-to-Video (I2V): VideoImageToVideoRequest

This separation ensures correct field requiredness since image_url is
required for I2V models but not applicable for T2V models.

All parameter constraints (durations, resolutions, aspect_ratios) should be
validated against the model's constraints from the models API response -
NOT hardcoded validators. Each model supports different subsets.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _validate_media_url(v: str) -> str:
    """Ensure a media URL is an http(s) URL or a data: URI."""
    if not (v.startswith("http://") or v.startswith("https://") or v.startswith("data:")):
        raise ValueError("URL must start with 'http://', 'https://', or 'data:'")
    return v


class VideoElement(BaseModel):
    """Structured element for advanced element-aware video models
    (e.g. Kling O3 R2V). Each element defines a character or object that can
    be referenced in the prompt as ``@Element1``, ``@Element2``, etc.
    """

    frontal_image_url: str = Field(
        ..., description="Frontal reference image for this element (HTTP URL or data: URI)."
    )
    reference_image_urls: list[str] | None = Field(
        None,
        max_length=3,
        description="Optional additional reference images for this element (up to 3).",
    )
    video_url: str | None = Field(
        None,
        description=(
            "Optional reference video for this element (HTTP URL or data: URI). "
            "Used by element-aware models that accept per-element motion donors."
        ),
    )

    @field_validator("frontal_image_url")
    @classmethod
    def _validate_frontal(cls, v: str) -> str:
        return _validate_media_url(v)

    @field_validator("reference_image_urls")
    @classmethod
    def _validate_refs(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for url in v:
                _validate_media_url(url)
        return v


class SeedanceConsents(BaseModel):
    """Seedance face-media consent attestations.

    Required by ``POST /video/queue`` only when the submitted media contains
    faces; the API returns a 409 ``needs_consent`` (with the policy text)
    otherwise. Each flag must be ``True`` (swagger declares ``enum: [true]``).
    """

    confirmed_terms_and_privacy: Literal[True] = Field(
        ...,
        description="Confirms acceptance of the current Seedance face-media policy text.",
    )
    confirmed_legal_right: Literal[True] = Field(
        ...,
        description="Confirms the API user has the legal right to use the face-bearing media.",
    )
    confirmed_screening_acknowledged: Literal[True] = Field(
        ...,
        description="Acknowledges submitted media may be screened before private asset submission.",
    )


class VideoConsents(BaseModel):
    """Provider-specific consent attestations for video generation."""

    seedance: SeedanceConsents | None = Field(
        None, description="Seedance face-media consent attestations."
    )


class VideoRequestBase(BaseModel):
    """
    Base fields shared by all video generation requests.

    Note: This class does NOT validate duration/resolution/aspect_ratio
    against hardcoded lists. Each model supports different values.
    Validate against model.model_spec.constraints at runtime.
    """

    model: str = Field(
        ...,
        description="Video model ID (e.g., 'wan-2.6-text-to-video', 'wan-3-0-prime-image-to-video')",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Text prompt for video generation (max 20000 chars on newer models)",
    )
    duration: str = Field(
        ...,
        description="Duration of generated video (e.g., '5s', '10s'). Valid values vary by model.",
    )
    negative_prompt: str | None = Field(
        None,
        max_length=20000,
        description=(
            "Optional negative prompt. Per-model max length varies (default 2500, "
            "up to 20000). The API has no default — omit to skip."
        ),
    )
    resolution: str | None = Field(
        None,
        description=(
            "Output resolution (e.g., '720p', '1080p'). Valid values vary by "
            "model; some models don't accept resolution. Omit to use the model's "
            "default (use ``upscale_factor`` for upscale models)."
        ),
    )
    audio: bool | None = Field(
        None,
        description="Generate audio if model supports it. Check model.model_spec.constraints.audio_configurable.",
    )
    upscale_factor: Literal[1, 2, 4] | None = Field(
        None,
        description=(
            "For upscale models only: 1 = quality enhancement, 2 = double resolution "
            "(default for topaz-video-upscale), 4 = quadruple."
        ),
    )
    end_image_url: str | None = Field(
        None,
        description=(
            "For models supporting end images / transitions, the end-frame image "
            "(HTTP URL or data: URI)."
        ),
    )
    audio_url: str | None = Field(
        None,
        description=(
            "For models supporting background audio input (WAV/MP3, max 30s, 15MB). "
            "HTTP URL or data: URI."
        ),
    )
    video_url: str | None = Field(
        None,
        description=(
            "For video-to-video / upscale models, the source video "
            "(MP4/MOV/WebM). HTTP URL or data: URI."
        ),
    )
    reference_image_urls: list[str] | None = Field(
        None,
        max_length=9,
        description=(
            "Up to 9 reference images for character/style consistency. Each must be "
            "a URL or data URL."
        ),
    )
    reference_audio_urls: list[str] | None = Field(
        None,
        max_length=3,
        description=(
            "Up to 3 reference audio URLs (role 'reference_audio') for R2V models "
            "(e.g. Seedance 2.0 R2V) used as donors for vocal timbre/narration/SFX. "
            "Per-clip 2-15s, .wav/.mp3; aggregate <=15s. Must be paired with at least "
            "one reference image or reference video. Each must be a URL or data URL."
        ),
    )
    reference_video_urls: list[str] | None = Field(
        None,
        max_length=3,
        description=(
            "Up to 3 reference video URLs (role 'reference_video') for R2V models "
            "(e.g. Seedance 2.0 R2V) used to inherit subject motion, camera movement, "
            "and overall style. Per-clip 2-15s, .mp4/.mov, <=50MB; aggregate <=15s. "
            "Each must be a URL or data URL."
        ),
    )
    elements: list[VideoElement] | None = Field(
        None,
        max_length=4,
        description=(
            "Up to 4 structured elements for advanced element-aware models (Kling O3 R2V). "
            "Reference in prompt as @Element1, @Element2, etc."
        ),
    )
    scene_image_urls: list[str] | None = Field(
        None,
        max_length=4,
        description=(
            "Up to 4 scene reference images for advanced element-aware models. Reference "
            "in prompt as @Image1, @Image2, etc."
        ),
    )
    bitrate_mode: Literal["standard", "high"] | None = Field(
        None,
        description=(
            "Output encode bitrate. ``high`` is a higher-quality, larger-file "
            "encode; omitting the field is equivalent to ``standard``. Supported "
            "on public Seedance 2.0 (including Fast) and 2.5 models only — other "
            "video families reject it. Queue-only: it does not change price, and "
            "``/video/quote`` does not accept it."
        ),
    )
    consents: VideoConsents | None = Field(
        None,
        description=(
            "Provider-specific consent attestations. Required only when submitted "
            "media contains faces (e.g. Seedance) — the API returns a 409 "
            "``needs_consent`` otherwise."
        ),
    )

    @field_validator("end_image_url", "audio_url", "video_url")
    @classmethod
    def _validate_optional_media_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_media_url(v)

    @field_validator(
        "reference_image_urls", "reference_audio_urls", "reference_video_urls", "scene_image_urls"
    )
    @classmethod
    def _validate_optional_url_list(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for url in v:
                _validate_media_url(url)
        return v


class VideoTextToVideoRequest(VideoRequestBase):
    """
    Request for text-to-video generation.

    POST /api/v1/video/queue (with T2V model)

    Use this for models with model_spec.constraints.model_type == "text-to-video".
    For T2V models, aspect_ratio is typically required.

    Example:
        >>> request = VideoTextToVideoRequest(
        ...     model="wan-2.6-text-to-video",
        ...     prompt="A sunset over the ocean with gentle waves",
        ...     duration="5s",
        ...     aspect_ratio="16:9",
        ...     resolution="1080p"
        ... )
    """

    aspect_ratio: str | None = Field(
        None,
        description="Aspect ratio (e.g., '16:9', '9:16'). Check model constraints for valid values.",
    )


class VideoImageToVideoRequest(VideoRequestBase):
    """
    Request for image-to-video generation.

    POST /api/v1/video/queue (with I2V model)

    Use this for models with model_spec.constraints.model_type == "image-to-video".
    For I2V models, image_url is REQUIRED and aspect_ratio may be ignored
    (the input image's ratio is used).

    Example:
        >>> request = VideoImageToVideoRequest(
        ...     model="wan-2.6-image-to-video",
        ...     prompt="Make this photo come to life with subtle movement",
        ...     duration="5s",
        ...     image_url="https://example.com/photo.jpg",
        ...     resolution="1080p"
        ... )
    """

    image_url: str = Field(
        ...,
        description="Reference image (HTTP URL or data: URI). REQUIRED for I2V models.",
    )
    aspect_ratio: str | None = Field(
        None,
        description="Aspect ratio (usually ignored for I2V, uses input image ratio)",
    )

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """Validate image URL format."""
        if not (v.startswith("http://") or v.startswith("https://") or v.startswith("data:")):
            raise ValueError("image_url must start with 'http://', 'https://', or 'data:'")
        return v


# Union type for the queue endpoint (accepts either T2V or I2V request)
VideoQueueRequest = VideoTextToVideoRequest | VideoImageToVideoRequest


class VideoQuoteRequest(BaseModel):
    """
    Request for video generation price quote.

    POST /api/v1/video/quote

    Per the Venice API spec, ``/video/quote`` accepts a strict subset of the
    ``/video/queue`` body: only the fields that affect pricing. Prompt text,
    reference images, and element/scene URLs are not part of the quote.
    """

    model_config = {"extra": "forbid"}

    model: str = Field(..., description="Video model ID (e.g., 'wan-2-7-text-to-video').")
    duration: str = Field(
        ...,
        description="Duration of generated video (e.g., '5s', '10s'). Valid values vary by model.",
    )
    aspect_ratio: str | None = Field(
        None,
        description="Aspect ratio (e.g., '16:9', '9:16'). Optional; valid values vary by model.",
    )
    resolution: str | None = Field(
        None,
        description="Output resolution (e.g., '720p', '1080p'). Optional; valid values vary by model.",
    )
    upscale_factor: Literal[1, 2, 4] | None = Field(
        None,
        description="For upscale models: 1 = quality enhancement, 2 = double, 4 = quadruple.",
    )
    audio: bool | None = Field(
        None,
        description="Generate audio if the model supports it.",
    )
    video_url: str | None = Field(
        None,
        description=(
            "For video-to-video / upscale models, the source video "
            "(MP4/MOV/WebM). HTTP URL or data: URI."
        ),
    )
    reference_video_total_duration: float | None = Field(
        None,
        ge=0,
        description=(
            "For R2V models (e.g. Seedance 2.0 R2V), the aggregate duration in "
            "seconds of all reference videos to include in the quote (max 45s; "
            "per-clip 2-15s, total <=15s for Seedance). When provided, the quote "
            "reflects the 'input with video' rate tier; when omitted, the no-reference "
            "baseline is returned."
        ),
    )

    @field_validator("video_url")
    @classmethod
    def _validate_video_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_media_url(v)


class VideoRetrieveRequest(BaseModel):
    """
    Request to retrieve video generation result.

    POST /api/v1/video/retrieve

    Poll this endpoint with the queue_id from the queue response
    until the video is ready (returns binary) or an error occurs.
    """

    model: str = Field(..., description="Model ID used for generation")
    queue_id: str = Field(..., description="Queue ID from queue response")
    delete_media_on_completion: bool = Field(
        False, description="Auto-delete media after successful retrieval"
    )


class VideoCompleteRequest(BaseModel):
    """
    Request to complete/cleanup video after download.

    POST /api/v1/video/complete

    Call this after successfully downloading the video to clean up
    server-side storage. Not needed if delete_media_on_completion
    was set to True in the retrieve request.
    """

    model: str = Field(..., description="Model ID used for generation")
    queue_id: str = Field(..., description="Queue ID to complete")


class VideoTranscriptionRequest(BaseModel):
    """Request for video transcription (``POST /api/v1/video/transcriptions``).

    Takes a publicly accessible YouTube URL and returns the detected language
    plus transcribed text in either JSON or plain-text form.
    """

    url: str = Field(..., description="YouTube video URL to transcribe")
    response_format: Literal["json", "text"] = Field(
        "json",
        description=(
            "Transcript output format: ``json`` (default) returns a structured "
            "object with ``transcript`` and ``lang``; ``text`` returns plain text."
        ),
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Require an http(s) URL."""
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("url must start with 'http://' or 'https://'")
        return v


__all__ = [
    "VideoElement",
    "VideoRequestBase",
    "VideoTextToVideoRequest",
    "VideoImageToVideoRequest",
    "VideoQueueRequest",
    "VideoQuoteRequest",
    "VideoRetrieveRequest",
    "VideoCompleteRequest",
    "VideoTranscriptionRequest",
]
