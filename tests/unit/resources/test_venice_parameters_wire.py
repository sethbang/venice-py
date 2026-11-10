"""What ``venice_parameters`` puts on the wire.

Only fields the caller set are sent, on both ``/chat/completions`` and
``/responses``, so a default on the model cannot contradict a top-level option
such as ``web_search=True`` or override a server-side default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.resources.responses import Responses
from venice_ai.types.api.requests.common import VeniceParameters

_MESSAGES = [{"role": "user", "content": "hi"}]


def _client() -> Mock:
    client = Mock()
    client.post = AsyncMock(return_value=Mock())
    return client


async def _chat_body(venice_parameters: Any) -> dict[str, Any]:
    client = _client()
    await ChatCompletions(client).create(
        model="m", messages=list(_MESSAGES), venice_parameters=venice_parameters
    )
    body: dict[str, Any] = client.post.call_args.kwargs["json_data"]
    return body


async def _responses_body(venice_parameters: Any, **kwargs: Any) -> dict[str, Any]:
    client = _client()
    await Responses(client).create(
        model="m", input="hi", venice_parameters=venice_parameters, **kwargs
    )
    body: dict[str, Any] = client.post.call_args.kwargs["json_data"]
    return body


_BODIES = pytest.mark.parametrize("build_body", [_chat_body, _responses_body])


@pytest.mark.asyncio
async def test_responses_web_search_is_not_contradicted_by_unset_defaults() -> None:
    body = await _responses_body(VeniceParameters(character_slug="c"), web_search=True)
    assert body["web_search"] is True
    assert body["venice_parameters"] == {"character_slug": "c"}


@pytest.mark.asyncio
@_BODIES
async def test_only_fields_the_caller_set_are_sent(build_body: Any) -> None:
    body = await build_body(VeniceParameters(enable_web_citations=True))
    assert body["venice_parameters"] == {"enable_web_citations": True}


@pytest.mark.asyncio
@_BODIES
async def test_a_value_equal_to_its_default_is_still_sent_when_set(build_body: Any) -> None:
    body = await build_body(
        VeniceParameters(enable_web_search="off", include_venice_system_prompt=True)
    )
    assert body["venice_parameters"] == {
        "enable_web_search": "off",
        "include_venice_system_prompt": True,
    }


@pytest.mark.asyncio
@_BODIES
async def test_mapping_sends_only_its_own_keys(build_body: Any) -> None:
    body = await build_body({"enable_web_search": "on"})
    assert body["venice_parameters"] == {"enable_web_search": "on"}


@pytest.mark.asyncio
@_BODIES
async def test_mapping_and_model_send_the_same_object(build_body: Any) -> None:
    from_mapping = await build_body({"include_venice_system_prompt": False})
    from_model = await build_body(VeniceParameters(include_venice_system_prompt=False))
    assert from_mapping == from_model


@pytest.mark.asyncio
@_BODIES
async def test_unrecognized_key_survives(build_body: Any) -> None:
    body = await build_body(VeniceParameters.model_validate({"some_future_flag": True}))
    assert body["venice_parameters"] == {"some_future_flag": True}


@pytest.mark.asyncio
@_BODIES
async def test_field_assigned_after_construction_is_sent(build_body: Any) -> None:
    params = VeniceParameters()
    params.enable_web_search = "auto"
    body = await build_body(params)
    assert body["venice_parameters"] == {"enable_web_search": "auto"}


@pytest.mark.asyncio
@_BODIES
async def test_explicit_none_is_not_sent(build_body: Any) -> None:
    body = await build_body(VeniceParameters(character_slug=None, enable_web_citations=True))
    assert body["venice_parameters"] == {"enable_web_citations": True}


@pytest.mark.asyncio
@_BODIES
@pytest.mark.parametrize("empty", [VeniceParameters(), {}])
async def test_nothing_set_omits_the_object(build_body: Any, empty: Any) -> None:
    body = await build_body(empty)
    assert "venice_parameters" not in body


@pytest.mark.asyncio
async def test_responses_still_strips_chat_only_keys_on_a_model() -> None:
    body = await _responses_body(
        VeniceParameters(disable_thinking=True, strip_thinking_response=True, character_slug="c")
    )
    assert body["venice_parameters"] == {"character_slug": "c"}


@pytest.mark.asyncio
async def test_responses_omits_the_object_when_only_chat_only_keys_were_set() -> None:
    body = await _responses_body(VeniceParameters(disable_thinking=True))
    assert "venice_parameters" not in body


def test_model_attributes_still_read_their_defaults() -> None:
    params = VeniceParameters()
    assert params.enable_web_search == "off"
    assert params.include_venice_system_prompt is True
