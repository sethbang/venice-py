"""
Unit tests for video generation request models.

Tests validation for VideoTextToVideoRequest, VideoImageToVideoRequest,
VideoQuoteRequest, VideoRetrieveRequest, and VideoCompleteRequest.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.video import (
    VideoCompleteRequest,
    VideoElement,
    VideoImageToVideoRequest,
    VideoQuoteRequest,
    VideoRetrieveRequest,
    VideoTextToVideoRequest,
)


class TestVideoTextToVideoRequest:
    """Test T2V request validation."""

    def test_valid_request(self):
        request = VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset over the ocean",
            duration="5s",
            aspect_ratio="16:9",
            resolution="1080p",
        )  # type: ignore
        assert request.model == "wan-2.6-text-to-video"
        assert request.duration == "5s"
        assert request.aspect_ratio == "16:9"
        assert request.resolution == "1080p"

    def test_minimal_request(self):
        """Test request with only required fields."""
        request = VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset over the ocean",
            duration="5s",
        )  # type: ignore
        assert request.model == "wan-2.6-text-to-video"
        # Optional fields default to None so the server can apply per-model defaults.
        assert request.resolution is None
        assert request.audio is None

    def test_prompt_length_validation(self):
        with pytest.raises(ValidationError):
            VideoTextToVideoRequest(
                model="wan-2.6-text-to-video",
                prompt="",  # Empty not allowed
                duration="5s",
            )  # type: ignore

    def test_prompt_max_length(self):
        """Prompts over 20000 chars are rejected (the max is model-dependent;
        the SDK enforces the highest known ceiling — 20000 per swagger)."""
        # 20000 is accepted.
        VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A" * 20000,
            duration="5s",
        )  # type: ignore
        # 20001 is rejected.
        with pytest.raises(ValidationError):
            VideoTextToVideoRequest(
                model="wan-2.6-text-to-video",
                prompt="A" * 20001,  # Over limit
                duration="5s",
            )  # type: ignore

    def test_negative_prompt_max_length(self):
        """negative_prompt accepts up to 20000 chars, rejects 20001."""
        VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset",
            duration="5s",
            negative_prompt="B" * 20000,
        )  # type: ignore
        with pytest.raises(ValidationError):
            VideoTextToVideoRequest(
                model="wan-2.6-text-to-video",
                prompt="A sunset",
                duration="5s",
                negative_prompt="B" * 20001,
            )  # type: ignore

    def test_negative_prompt_defaults_to_none(self):
        """negative_prompt has no default per the API spec; callers must opt in."""
        request = VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset",
            duration="5s",
        )  # type: ignore
        assert request.negative_prompt is None

    def test_audio_option(self):
        """Test audio parameter."""
        request = VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset",
            duration="5s",
            audio=True,
        )  # type: ignore
        assert request.audio is True


class TestVideoImageToVideoRequest:
    """Test I2V request validation."""

    def test_requires_image_url(self):
        with pytest.raises(ValidationError):
            VideoImageToVideoRequest(
                model="wan-2.6-image-to-video",
                prompt="Animate this image",
                duration="5s",
                # Missing image_url - should fail
            )  # type: ignore

    def test_valid_http_url(self):
        request = VideoImageToVideoRequest(
            model="wan-2.6-image-to-video",
            prompt="Animate this image",
            duration="5s",
            image_url="https://example.com/image.jpg",
        )  # type: ignore
        assert request.image_url == "https://example.com/image.jpg"

    def test_valid_https_url(self):
        request = VideoImageToVideoRequest(
            model="wan-2.6-image-to-video",
            prompt="Animate this image",
            duration="5s",
            image_url="https://secure.example.com/image.png",
        )  # type: ignore
        assert request.image_url.startswith("https://")

    def test_valid_data_url(self):
        request = VideoImageToVideoRequest(
            model="wan-2.6-image-to-video",
            prompt="Animate this image",
            duration="5s",
            image_url="data:image/png;base64,iVBORw0K...",
        )  # type: ignore
        assert request.image_url.startswith("data:")

    def test_invalid_url_scheme(self):
        with pytest.raises(ValidationError, match="must start with"):
            VideoImageToVideoRequest(
                model="wan-2.6-image-to-video",
                prompt="Animate this image",
                duration="5s",
                image_url="ftp://invalid.com/image.jpg",
            )  # type: ignore

    def test_invalid_url_no_scheme(self):
        with pytest.raises(ValidationError, match="must start with"):
            VideoImageToVideoRequest(
                model="wan-2.6-image-to-video",
                prompt="Animate this image",
                duration="5s",
                image_url="/local/path/image.jpg",
            )  # type: ignore


class TestVideoQuoteRequest:
    """Test quote request validation — ``/video/quote`` accepts only the
    pricing-relevant subset (model, duration, aspect_ratio, resolution,
    upscale_factor, audio, video_url)."""

    def test_minimal_request(self):
        request = VideoQuoteRequest(
            model="wan-2-7-text-to-video",
            duration="5s",
        )  # type: ignore
        assert request.model == "wan-2-7-text-to-video"
        assert request.duration == "5s"

    def test_aspect_ratio_optional(self):
        request = VideoQuoteRequest(
            model="wan-2-7-text-to-video",
            duration="5s",
            aspect_ratio="16:9",
        )  # type: ignore
        assert request.aspect_ratio == "16:9"

    def test_with_video_url(self):
        request = VideoQuoteRequest(
            model="topaz-video-upscale",
            duration="Auto",
            video_url="https://example.com/source.mp4",
            upscale_factor=2,
        )  # type: ignore
        assert request.video_url == "https://example.com/source.mp4"
        assert request.upscale_factor == 2

    def test_validates_video_url_if_provided(self):
        with pytest.raises(ValidationError, match="must start with"):
            VideoQuoteRequest(
                model="topaz-video-upscale",
                duration="Auto",
                video_url="ftp://invalid.com/v.mp4",
            )  # type: ignore

    def test_rejects_prompt_and_image_fields(self):
        """Prompt and reference-image fields are not part of the quote spec."""
        for extra in (
            {"prompt": "anything"},
            {"image_url": "https://example.com/x.png"},
            {"negative_prompt": "blurry"},
            {"reference_image_urls": ["https://example.com/a.png"]},
            {"elements": [{"frontal_image_url": "https://example.com/a.png"}]},
            {"scene_image_urls": ["https://example.com/a.png"]},
        ):
            with pytest.raises(ValidationError):
                VideoQuoteRequest(
                    model="wan-2-7-text-to-video",
                    duration="5s",
                    **extra,  # type: ignore[arg-type]
                )


class TestVideoRetrieveRequest:
    """Test retrieve request validation."""

    def test_valid_request(self):
        request = VideoRetrieveRequest(
            model="wan-2.6-text-to-video",
            queue_id="550e8400-e29b-41d4-a716-446655440000",
        )  # type: ignore
        assert request.model == "wan-2.6-text-to-video"
        assert request.queue_id == "550e8400-e29b-41d4-a716-446655440000"
        assert request.delete_media_on_completion is False  # Default

    def test_delete_media_option(self):
        request = VideoRetrieveRequest(
            model="wan-2.6-text-to-video",
            queue_id="abc-123",
            delete_media_on_completion=True,
        )
        assert request.delete_media_on_completion is True

    def test_requires_model(self):
        with pytest.raises(ValidationError):
            VideoRetrieveRequest(
                queue_id="abc-123",
            )  # type: ignore

    def test_requires_queue_id(self):
        with pytest.raises(ValidationError):
            VideoRetrieveRequest(
                model="wan-2.6-text-to-video",
            )  # type: ignore


class TestVideoCompleteRequest:
    """Test complete request validation."""

    def test_valid_request(self):
        request = VideoCompleteRequest(
            model="wan-2.6-text-to-video",
            queue_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert request.model == "wan-2.6-text-to-video"
        assert request.queue_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_requires_both_fields(self):
        with pytest.raises(ValidationError):
            VideoCompleteRequest(model="wan-2.6-text-to-video")  # type: ignore

        with pytest.raises(ValidationError):
            VideoCompleteRequest(queue_id="abc-123")  # type: ignore


class TestVideoRequestSerialization:
    """Test serialization of video requests."""

    def test_t2v_to_dict(self):
        request = VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset",
            duration="5s",
            aspect_ratio="16:9",
        )  # type: ignore
        data = request.model_dump()
        assert data["model"] == "wan-2.6-text-to-video"
        assert data["prompt"] == "A sunset"
        assert data["duration"] == "5s"
        assert data["aspect_ratio"] == "16:9"

    def test_i2v_to_dict(self):
        request = VideoImageToVideoRequest(
            model="wan-2.6-image-to-video",
            prompt="Animate",
            duration="5s",
            image_url="https://example.com/image.jpg",
        )  # type: ignore
        data = request.model_dump()
        assert data["image_url"] == "https://example.com/image.jpg"

    def test_excludes_none_values(self):
        request = VideoTextToVideoRequest(
            model="wan-2.6-text-to-video",
            prompt="A sunset",
            duration="5s",
        )  # type: ignore
        data = request.model_dump(exclude_none=True)
        # aspect_ratio is None so should be excluded
        assert "aspect_ratio" not in data or data.get("aspect_ratio") is None


class TestVideoElement:
    """Test VideoElement validation."""

    def test_video_url_retained_in_dump(self):
        """elements[].video_url is supported and round-trips via model_dump."""
        element = VideoElement(
            frontal_image_url="https://example.com/a.png",
            video_url="https://x",
        )  # type: ignore
        assert element.video_url == "https://x"
        assert element.model_dump()["video_url"] == "https://x"

    def test_reference_image_urls_max_items(self):
        """The inner reference_image_urls is capped at 3 (swagger maxItems: 3)."""
        # 3 is accepted.
        VideoElement(
            frontal_image_url="https://example.com/a.png",
            reference_image_urls=[
                "https://example.com/1.png",
                "https://example.com/2.png",
                "https://example.com/3.png",
            ],
        )
        # 4 is rejected.
        with pytest.raises(ValidationError):
            VideoElement(
                frontal_image_url="https://example.com/a.png",
                reference_image_urls=[
                    "https://example.com/1.png",
                    "https://example.com/2.png",
                    "https://example.com/3.png",
                    "https://example.com/4.png",
                ],
            )
