"""``transparent_background`` on POST /image/generate, /image/edit and /image/multi-edit.

The flag must reach the JSON body verbatim when passed (``True`` and
``False``) and be absent when omitted. The omitted case is pinned against the
exact serialized body each method sent before the field existed, so recorded
cassettes keep matching.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api import ImageGenerationResponse
from venice_ai.types.api.base import TimingInfo
from venice_ai.types.api.requests.images import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageMultiEditRequest,
)

_PNG = b"\x89PNG\r\n\x1a\n"
_URL = "https://example.com/a.png"


def _fake_response() -> ImageGenerationResponse:
    return ImageGenerationResponse(
        id="gen-tb",
        images=["ZmFrZQ=="],
        request=None,
        timing=TimingInfo(
            inferenceDuration=0, inferencePreprocessingTime=0, inferenceQueueTime=0, total=0
        ),
    )


@pytest.fixture
def image() -> Image:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response())
    client._request = AsyncMock(return_value=_PNG)
    return Image(client)  # type: ignore[arg-type]


def _generate_body(image: Image) -> dict[str, Any]:
    return image._client.post.await_args.kwargs["json_data"]  # type: ignore[attr-defined,no-any-return]


def _binary_body(image: Image) -> dict[str, Any]:
    return image._client._request.await_args.kwargs["json_data"]  # type: ignore[attr-defined,no-any-return]


# ---------------------------------------------------------------------------
# Forwarded when passed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_generate_forwards_transparent_background(image: Image, value: bool) -> None:
    await image.create(
        model="some-image-model", prompt="a cat", format="png", transparent_background=value
    )
    assert _generate_body(image)["transparent_background"] is value


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_edit_forwards_transparent_background(image: Image, value: bool) -> None:
    await image.edit(prompt="cut it out", image=_URL, transparent_background=value)
    assert _binary_body(image)["transparent_background"] is value


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_multi_edit_forwards_transparent_background(image: Image, value: bool) -> None:
    await image.multi_edit(prompt="blend", image=_URL, transparent_background=value)
    assert _binary_body(image)["transparent_background"] is value


@pytest.mark.asyncio
async def test_submit_forwards_transparent_background(image: Image) -> None:
    job = await image.submit(model="some-image-model", prompt="a cat", transparent_background=True)
    assert job._kwargs["transparent_background"] is True


@pytest.mark.asyncio
async def test_submit_omits_transparent_background_when_unset(image: Image) -> None:
    job = await image.submit(model="some-image-model", prompt="a cat")
    assert "transparent_background" not in job._kwargs


# ---------------------------------------------------------------------------
# Omitted when not passed: bodies are byte-identical to the pre-field shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_default_body_unchanged(image: Image) -> None:
    await image.create(model="some-image-model", prompt="a cat")
    body = _generate_body(image)
    assert "transparent_background" not in body
    assert json.dumps(body) == '{"model": "some-image-model", "prompt": "a cat"}'


@pytest.mark.asyncio
async def test_edit_default_body_unchanged(image: Image) -> None:
    await image.edit(prompt="make it blue", image=_URL)
    body = _binary_body(image)
    assert "transparent_background" not in body
    assert json.dumps(body) == '{"prompt": "make it blue", "image": "https://example.com/a.png"}'


@pytest.mark.asyncio
async def test_multi_edit_default_body_unchanged(image: Image) -> None:
    await image.multi_edit(prompt="blend", image=_URL)
    body = _binary_body(image)
    assert "transparent_background" not in body
    assert json.dumps(body) == '{"prompt": "blend", "images": ["https://example.com/a.png"]}'


@pytest.mark.asyncio
async def test_edit_binary_input_default_body_has_no_new_key(image: Image) -> None:
    await image.edit(prompt="make it blue", image=_PNG + b"fake")
    assert set(_binary_body(image)) == {"prompt", "image"}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (ImageGenerationRequest, {"model": "m", "prompt": "p"}),
        (ImageEditRequest, {"prompt": "p", "image": _URL}),
        (ImageMultiEditRequest, {"prompt": "p", "images": [_URL]}),
    ],
)
def test_request_models_default_to_none_and_dump_without_it(
    model_cls: type, kwargs: dict[str, Any]
) -> None:
    request = model_cls(**kwargs)
    assert request.transparent_background is None
    assert "transparent_background" not in request.model_dump(exclude_none=True)
    assert (
        model_cls(**kwargs, transparent_background=True).model_dump(exclude_none=True)[
            "transparent_background"
        ]
        is True
    )
