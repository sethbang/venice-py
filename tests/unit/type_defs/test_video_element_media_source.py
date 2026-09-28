"""Contract tests for ``VideoElement`` media sources.

An element is described either by images (a frontal image, optionally with up
to three extra reference images) or by a single reference video. The API
rejects an element that carries both kinds ("Cannot provide both image URLs
and video URL for the same element"), and the spec marks none of the element
fields as required, so the model has to accept each kind on its own and
reject the combination and the empty element.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.video import (
    VideoElement,
    VideoImageToVideoRequest,
    VideoTextToVideoRequest,
)

IMG = "https://example.com/front.png"
REF = "https://example.com/side.png"
VID = "https://example.com/donor.mp4"


def _assert_rejected_for_source_rule(exc: ValidationError) -> None:
    """The element must fail the one-source rule, not a single missing field."""
    missing = [e["loc"] for e in exc.errors() if e["type"] == "missing"]
    assert not missing, f"rejected only because {missing} is required: {exc}"


class TestVideoOnlyElement:
    def test_video_only_element_is_constructible(self):
        element = VideoElement(video_url=VID)  # type: ignore[call-arg]
        assert element.video_url == VID
        assert element.frontal_image_url is None

    def test_video_only_element_serializes_without_image_keys(self):
        element = VideoElement.model_validate({"video_url": VID})
        assert element.model_dump(exclude_none=True) == {"video_url": VID}

    def test_video_only_element_accepted_on_queue_request(self):
        request = VideoTextToVideoRequest.model_validate(
            {
                "model": "kling-o3-r2v",
                "prompt": "@Element1 walks through the market",
                "duration": "5s",
                "elements": [{"video_url": VID}],
            }
        )
        body = request.model_dump(exclude_none=True)
        assert body["elements"] == [{"video_url": VID}]

    def test_video_url_scheme_is_validated(self):
        with pytest.raises(ValidationError, match="must start with"):
            VideoElement(video_url="ftp://example.com/donor.mp4")  # type: ignore[call-arg]


class TestImageAndVideoAreMutuallyExclusive:
    @pytest.mark.parametrize(
        "payload",
        [
            {"frontal_image_url": IMG, "video_url": VID},
            {"reference_image_urls": [REF], "video_url": VID},
            {"frontal_image_url": IMG, "reference_image_urls": [REF], "video_url": VID},
        ],
        ids=["frontal+video", "refs+video", "frontal+refs+video"],
    )
    def test_images_combined_with_video_rejected(self, payload):
        with pytest.raises(ValidationError) as exc_info:
            VideoElement.model_validate(payload)
        _assert_rejected_for_source_rule(exc_info.value)

    def test_combined_element_rejected_inside_i2v_request(self):
        with pytest.raises(ValidationError):
            VideoImageToVideoRequest.model_validate(
                {
                    "model": "kling-o3-r2v",
                    "prompt": "@Element1",
                    "duration": "5s",
                    "image_url": IMG,
                    "elements": [{"frontal_image_url": IMG, "video_url": VID}],
                }
            )

    def test_empty_element_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            VideoElement.model_validate({})
        _assert_rejected_for_source_rule(exc_info.value)


class TestImageElementStillWorks:
    def test_frontal_only(self):
        element = VideoElement(frontal_image_url=IMG)
        assert element.model_dump(exclude_none=True) == {"frontal_image_url": IMG}

    def test_frontal_with_references(self):
        element = VideoElement(frontal_image_url=IMG, reference_image_urls=[REF])
        assert element.reference_image_urls == [REF]
