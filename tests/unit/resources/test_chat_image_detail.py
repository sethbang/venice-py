"""``image_url.detail`` on chat image content parts.

POST ``/chat/completions`` accepts an optional ``detail`` on the
``image_url`` object of an ``image_url`` content part (``auto``, ``low``,
``high``, ``original``). The SDK sends it only when set, so bodies built
without it are unchanged.
"""

from typing import Any, get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from venice_ai.core.models.common import ImageContent, ImageDetail, ImageUrl
from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.types.api import ChatCompletionResponse
from venice_ai.types.api.requests.chat import UserMessage

_IMAGE = "https://example.com/cat.jpg"


def _fake_response() -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "a cat"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


@pytest.fixture
def chat_resource() -> ChatCompletions:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response())
    return ChatCompletions(client)  # type: ignore[arg-type]


def _body_sent(chat_resource: ChatCompletions) -> dict[str, Any]:
    return chat_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined,no-any-return]


def test_image_detail_values_match_spec() -> None:
    assert set(get_args(ImageDetail)) == {"auto", "low", "high", "original"}


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", get_args(ImageDetail))
async def test_builder_detail_is_sent(chat_resource: ChatCompletions, detail: str) -> None:
    msg = UserMessage.builder().text("What is this?").image(_IMAGE, detail=detail).build()  # type: ignore[arg-type]
    await chat_resource.create(model="some-vision-model", messages=[msg])

    part = _body_sent(chat_resource)["messages"][0]["content"][1]
    assert part == {"type": "image_url", "image_url": {"url": _IMAGE, "detail": detail}}


@pytest.mark.asyncio
async def test_builder_without_detail_omits_key(chat_resource: ChatCompletions) -> None:
    msg = UserMessage.builder().text("What is this?").image(_IMAGE).build()
    await chat_resource.create(model="some-vision-model", messages=[msg])

    part = _body_sent(chat_resource)["messages"][0]["content"][1]
    assert part == {"type": "image_url", "image_url": {"url": _IMAGE}}


@pytest.mark.asyncio
async def test_dict_form_detail_is_sent(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-vision-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": _IMAGE, "detail": "low"}},
                ],
            }
        ],
    )

    part = _body_sent(chat_resource)["messages"][0]["content"][1]
    assert part["image_url"] == {"url": _IMAGE, "detail": "low"}


@pytest.mark.asyncio
async def test_model_form_without_detail_omits_key(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-vision-model",
        messages=[
            UserMessage(content=[ImageContent(type="image_url", image_url=ImageUrl(url=_IMAGE))])
        ],
    )

    part = _body_sent(chat_resource)["messages"][0]["content"][0]
    assert part == {"type": "image_url", "image_url": {"url": _IMAGE}}


def test_unknown_detail_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ImageUrl(url=_IMAGE, detail="ultra")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        UserMessage.builder().image(_IMAGE, detail="ultra")  # type: ignore[arg-type]
