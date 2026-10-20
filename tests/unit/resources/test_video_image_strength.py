"""``image_strength`` on POST /video/queue, and the ``2:1`` aspect ratio.

``image_strength`` (number, 0-1) sets how strongly the input image guides an
image-to-video generation. It is queue-only: ``QuoteVideoRequest`` does not
list it and it does not change the price. It is sent only when set, so bodies
built without it are unchanged.
"""

import inspect
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from venice_ai.models.selection import cheapest_video_params
from venice_ai.resources.video import Video
from venice_ai.types.api import VideoQueueResponse, VideoQuoteResponse
from venice_ai.types.api.requests.video import (
    VideoImageToVideoRequest,
    VideoQuoteRequest,
    VideoTextToVideoRequest,
)

_IMAGE = "https://example.com/photo.jpg"


@pytest.fixture
def video_resource() -> Video:
    client = Mock()
    client.post = AsyncMock(
        return_value=VideoQueueResponse(model="some-i2v-model", queue_id="q-strength")
    )
    return Video(client)


def _body(video_resource: Video) -> dict[str, Any]:
    return cast(
        dict[str, Any], cast(Any, video_resource._client.post).call_args.kwargs["json_data"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strength", [0, 0.35, 1])
async def test_submit_forwards_image_strength(video_resource: Video, strength: float) -> None:
    await video_resource.submit(
        model="some-i2v-model",
        prompt="Gentle motion",
        duration_seconds=5,
        image_url=_IMAGE,
        image_strength=strength,
    )
    assert _body(video_resource)["image_strength"] == strength


@pytest.mark.asyncio
async def test_submit_omits_image_strength_when_unset(video_resource: Video) -> None:
    await video_resource.submit(
        model="some-i2v-model",
        prompt="Gentle motion",
        duration_seconds=5,
        image_url=_IMAGE,
    )
    assert _body(video_resource) == {
        "model": "some-i2v-model",
        "prompt": "Gentle motion",
        "duration": "5s",
        "image_url": _IMAGE,
    }


@pytest.mark.asyncio
async def test_run_forwards_image_strength(video_resource: Video) -> None:
    await video_resource.run(
        model="some-i2v-model",
        prompt="Gentle motion",
        duration_seconds=5,
        image_url=_IMAGE,
        image_strength=0.5,
    )
    assert _body(video_resource)["image_strength"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("strength", [-0.1, 1.01])
async def test_submit_rejects_out_of_range_image_strength(
    video_resource: Video, strength: float
) -> None:
    with pytest.raises(ValidationError):
        await video_resource.submit(
            model="some-i2v-model",
            prompt="Gentle motion",
            duration_seconds=5,
            image_url=_IMAGE,
            image_strength=strength,
        )
    cast(Any, video_resource._client.post).assert_not_called()


def test_request_models_accept_image_strength() -> None:
    i2v = VideoImageToVideoRequest(
        model="some-i2v-model", prompt="p", duration="5s", image_url=_IMAGE, image_strength=0.4
    )
    assert i2v.model_dump(exclude_none=True)["image_strength"] == 0.4
    t2v = VideoTextToVideoRequest(model="some-t2v-model", prompt="p", duration="5s")
    assert "image_strength" not in t2v.model_dump(exclude_none=True)


def test_quote_does_not_take_image_strength() -> None:
    assert "image_strength" not in inspect.signature(Video.quote).parameters
    assert "image_strength" not in VideoQuoteRequest.model_fields


# ---------------------------------------------------------------------------
# 2:1 aspect ratio — aspect ratios are open strings checked against the
# catalog, so a newly listed ratio needs no SDK change.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_passes_two_to_one_aspect_ratio(video_resource: Video) -> None:
    await video_resource.submit(
        model="some-t2v-model", prompt="Wide vista", duration_seconds=5, aspect_ratio="2:1"
    )
    assert _body(video_resource)["aspect_ratio"] == "2:1"


@pytest.mark.asyncio
async def test_quote_passes_two_to_one_aspect_ratio(video_resource: Video) -> None:
    post = cast(Any, video_resource._client.post)
    post.return_value = VideoQuoteResponse(quote=0.5)
    await video_resource.quote(model="some-t2v-model", duration_seconds=5, aspect_ratio="2:1")
    assert post.call_args.kwargs["json_data"]["aspect_ratio"] == "2:1"


def test_cheapest_video_params_accepts_two_to_one() -> None:
    constraints = {"durations": ["5s"], "aspect_ratios": ["2:1", "1:1"]}
    assert cheapest_video_params(constraints)["aspect_ratio"] == "2:1"
    assert cheapest_video_params(constraints, aspect_ratio="2:1")["aspect_ratio"] == "2:1"
