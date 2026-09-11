"""
Unit tests for the newly-documented fields on POST /video/queue and
POST /video/quote.

The Venice API body schema for ``/video/queue`` accepts nine fields the SDK
forwards:

- ``upscale_factor`` (``Literal[1, 2, 4]``) — required for the upscale model
- ``end_image_url`` (str) — end-frame reference
- ``audio_url`` (str) — background audio input
- ``video_url`` (str) — video-to-video / upscale input
- ``reference_image_urls`` (list[str], max 9) — identity/style references
- ``reference_audio_urls`` (list[str], max 3) — R2V audio donors (Seedance 2.0)
- ``reference_video_urls`` (list[str], max 3) — R2V video donors (Seedance 2.0)
- ``elements`` (list[VideoElement], max 4) — structured character/scene elements
- ``scene_image_urls`` (list[str], max 4) — scene references

``/video/quote`` accepts only the pricing-relevant subset (no prompt/image
references); see ``api-reference/endpoint/video/quote.md``.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.video import Video
from venice_ai.types.api import VideoQueueResponse, VideoQuoteResponse


@pytest.fixture
def video_resource() -> Video:
    client = Mock()
    client.post = AsyncMock()
    return Video(client)


def _post_mock(video_resource: Video) -> Any:
    return cast(Any, video_resource._client.post)


_ELEMENTS = [
    {
        "frontal_image_url": "https://example.com/char1.png",
        "reference_image_urls": ["https://example.com/char1-ref.png"],
    }
]
_NEW_FIELDS = {
    "upscale_factor": 2,
    "end_image_url": "https://example.com/end.png",
    "audio_url": "https://example.com/bgm.mp3",
    "video_url": "https://example.com/source.mp4",
    "reference_image_urls": ["https://example.com/ref1.png"],
    "reference_audio_urls": ["https://example.com/voice1.mp3"],
    "reference_video_urls": ["https://example.com/clip1.mp4"],
    "elements": _ELEMENTS,
    "scene_image_urls": ["https://example.com/scene1.png"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", list(_NEW_FIELDS.items()))
async def test_queue_forwards_new_field(video_resource: Video, field, value) -> None:
    post = _post_mock(video_resource)
    post.return_value = VideoQueueResponse(model="some-video-model", queue_id="q-new")

    await video_resource.submit(
        model="some-video-model",
        prompt="Test prompt",
        duration_seconds="5s",
        **{field: value},
    )

    body = post.call_args.kwargs["json_data"]
    assert body[field] == value


_QUOTE_FIELDS = {
    "upscale_factor": 2,
    "video_url": "https://example.com/source.mp4",
    "audio": True,
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "reference_video_total_duration": 5,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", list(_QUOTE_FIELDS.items()))
async def test_quote_forwards_field(video_resource: Video, field, value) -> None:
    post = _post_mock(video_resource)
    post.return_value = VideoQuoteResponse(quote=1.23)

    await video_resource.quote(
        model="some-video-model",
        duration_seconds="5s",
        **{field: value},
    )

    body = post.call_args.kwargs["json_data"]
    assert body[field] == value


@pytest.mark.asyncio
async def test_quote_omits_prompt_and_reference_fields(video_resource: Video) -> None:
    """``/video/quote`` does not accept prompt/reference fields per the spec;
    ``Video.quote`` must not expose them and the serialized body must exclude
    them entirely."""
    post = _post_mock(video_resource)
    post.return_value = VideoQuoteResponse(quote=1.23)

    await video_resource.quote(
        model="some-video-model",
        duration_seconds="5s",
        aspect_ratio="16:9",
        resolution="720p",
    )

    body = post.call_args.kwargs["json_data"]
    for k in (
        "prompt",
        "negative_prompt",
        "image_url",
        "end_image_url",
        "audio_url",
        "reference_image_urls",
        "elements",
        "scene_image_urls",
    ):
        assert k not in body


@pytest.mark.asyncio
async def test_queue_upscale_path_uses_video_url(video_resource: Video) -> None:
    """A queue call with ``video_url`` (upscale) must still reach the endpoint."""
    post = _post_mock(video_resource)
    post.return_value = VideoQueueResponse(model="topaz-video-upscale", queue_id="q-up")

    await video_resource.submit(
        model="topaz-video-upscale",
        prompt="upscale",
        duration_seconds="Auto",
        video_url="https://example.com/source.mp4",
        upscale_factor=4,
    )

    body = post.call_args.kwargs["json_data"]
    assert body["video_url"] == "https://example.com/source.mp4"
    assert body["upscale_factor"] == 4


@pytest.mark.asyncio
async def test_run_forwards_reference_video_urls(video_resource: Video) -> None:
    """``Video.run`` delegates to ``submit``, so reference_video_urls must reach
    the queue payload through the high-level lifecycle entry point too."""
    post = _post_mock(video_resource)
    post.return_value = VideoQueueResponse(model="seedance-2-0-r2v", queue_id="q-r2v")

    await video_resource.run(
        model="seedance-2-0-r2v",
        prompt="inherit this motion",
        duration_seconds="5s",
        reference_video_urls=["https://example.com/clip1.mp4"],
    )

    body = post.call_args.kwargs["json_data"]
    assert body["reference_video_urls"] == ["https://example.com/clip1.mp4"]


@pytest.mark.asyncio
async def test_queue_omits_new_fields_when_unset(video_resource: Video) -> None:
    post = _post_mock(video_resource)
    post.return_value = VideoQueueResponse(model="some-t2v-model", queue_id="q-clean")

    await video_resource.submit(
        model="some-t2v-model",
        prompt="clean",
        duration_seconds="5s",
    )

    body = post.call_args.kwargs["json_data"]
    for k in _NEW_FIELDS:
        assert k not in body


# ---------------------------------------------------------------------------
# Seedance bitrate_mode — a queue-only encoding option.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["standard", "high"])
async def test_queue_forwards_bitrate_mode(video_resource: Video, mode: str) -> None:
    post = _post_mock(video_resource)
    await video_resource.submit(
        model="seedance-2-5-text-to-video-basic",
        prompt="A neon city at night",
        duration_seconds=5,
        bitrate_mode=mode,
    )
    body = post.call_args[1]["json_data"]
    assert body["bitrate_mode"] == mode


@pytest.mark.asyncio
async def test_queue_omits_bitrate_mode_when_unset(video_resource: Video) -> None:
    """Omitting it is documented as equivalent to "standard"; don't send one."""
    post = _post_mock(video_resource)
    await video_resource.submit(
        model="seedance-2-5-text-to-video-basic",
        prompt="A neon city at night",
        duration_seconds=5,
    )
    assert "bitrate_mode" not in post.call_args[1]["json_data"]


@pytest.mark.asyncio
async def test_queue_rejects_unknown_bitrate_mode(video_resource: Video) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await video_resource.submit(
            model="seedance-2-5-text-to-video-basic",
            prompt="A neon city at night",
            duration_seconds=5,
            bitrate_mode="ultra",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_run_forwards_bitrate_mode(video_resource: Video) -> None:
    post = _post_mock(video_resource)
    await video_resource.run(
        model="seedance-2-5-text-to-video-basic",
        prompt="A neon city at night",
        duration_seconds=5,
        bitrate_mode="high",
    )
    assert post.call_args[1]["json_data"]["bitrate_mode"] == "high"


# ---------------------------------------------------------------------------
# Remaining /video/queue fields: the Topaz enhancement cluster plus the
# general-purpose keyframes / omni_reference_task_type / reference_document_urls
# / output_format.
# ---------------------------------------------------------------------------

_ENHANCEMENT_FIELDS = {
    "enhancement_model": "prob-4",
    "compression": 0.3,
    "creativity": 0.4,
    "grain": 0.05,
    "h264_output": True,
    "halo": 0.2,
    "noise": 0.5,
    "realism": 0.6,
    "recover_detail": 0.7,
    "sharp": 0.5,
    "softness": 3,
    "output_format": "prores",
    "target_fps": 60,
    "slowdown_factor": 2,
}

_GENERAL_QUEUE_FIELDS = {
    "omni_reference_task_type": "edit",
    "reference_document_urls": ["https://example.com/brief.pdf"],
    "keyframes": [
        {"image_url": "https://example.com/a.png", "frame_index": 0},
        {"image_url": "https://example.com/b.png", "frame_index": 120},
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", list(_ENHANCEMENT_FIELDS.items()) + list(_GENERAL_QUEUE_FIELDS.items())
)
async def test_queue_forwards_remaining_field(video_resource: Video, field, value) -> None:
    post = _post_mock(video_resource)
    await video_resource.submit(
        model="topaz-video-upscale",
        prompt="Enhance this",
        duration_seconds=5,
        **{field: value},
    )
    assert post.call_args[1]["json_data"][field] == value


@pytest.mark.asyncio
async def test_queue_omits_remaining_fields_when_unset(video_resource: Video) -> None:
    post = _post_mock(video_resource)
    await video_resource.submit(
        model="topaz-video-upscale", prompt="Enhance this", duration_seconds=5
    )
    body = post.call_args[1]["json_data"]
    for field in list(_ENHANCEMENT_FIELDS) + list(_GENERAL_QUEUE_FIELDS):
        assert field not in body, f"{field} should be omitted when unset"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [("enhancement_model", "prob-4"), ("target_fps", 60), ("slowdown_factor", 2)],
)
async def test_quote_forwards_enhancement_field(video_resource: Video, field, value) -> None:
    """These three affect price, so /video/quote accepts them too."""
    post = _post_mock(video_resource)
    await video_resource.quote(model="topaz-video-upscale", duration_seconds=5, **{field: value})
    assert post.call_args[1]["json_data"][field] == value


@pytest.mark.asyncio
async def test_queue_rejects_out_of_range_enhancement_value(video_resource: Video) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await video_resource.submit(
            model="topaz-video-upscale",
            prompt="Enhance this",
            duration_seconds=5,
            grain=0.5,  # spec caps grain at 0.1
        )
