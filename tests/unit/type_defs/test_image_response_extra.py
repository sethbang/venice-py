"""TDD: ImageGenerationResponse must tolerate unknown server fields.

The /image/generate 200 schema has no ``additionalProperties: false`` (unlike
/images/generations, which does), so the native response model must not reject
forward-compatible extra fields. SimpleImageGenerationResponse stays strict.
"""

from venice_ai.types.api.images import (
    ImageGenerationResponse,
    SimpleImageGenerationResponse,
)

_TIMING = {
    "inferenceDuration": 1.0,
    "inferencePreprocessingTime": 0.1,
    "inferenceQueueTime": 0.2,
    "total": 1.3,
}


def test_image_generation_response_preserves_unknown_fields():
    resp = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": ["YmFzZTY0"],
            "timing": _TIMING,
            "a_future_server_field": "keep-me",
        }
    )
    assert resp.id == "img-1"
    assert (resp.model_extra or {}).get("a_future_server_field") == "keep-me"


def test_simple_image_generation_response_keeps_unknown_fields():
    # The spec declares additionalProperties:false, but a field the server adds
    # later must not fail the whole response before the SDK models it.
    resp = SimpleImageGenerationResponse.model_validate(
        {"created": 1, "data": [], "an_unexpected_field": True}
    )
    assert (resp.model_extra or {}).get("an_unexpected_field") is True


def test_content_policy_violation_count_is_typed():
    resp = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": ["YmFzZTY0"],
            "timing": _TIMING,
            "content_policy_violation_count": 2,
        }
    )
    assert resp.content_policy_violation_count == 2
    assert "content_policy_violation_count" not in (resp.model_extra or {})


def test_content_policy_violation_count_defaults_to_none():
    resp = ImageGenerationResponse.model_validate(
        {"id": "img-1", "images": ["YmFzZTY0"], "timing": _TIMING}
    )
    assert resp.content_policy_violation_count is None
