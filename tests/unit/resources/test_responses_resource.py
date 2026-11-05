"""Unit tests for the Responses API (Alpha) resource.

Covers:
- ``client.responses.create()`` builds the correct request body and forwards
  it to ``POST /responses``.
- ``ResponsesResponse`` correctly deserialises each output block type
  (``reasoning``, ``message``, ``function_call``, ``web_search_call``).
- ``ResponsesRequest`` validates input ranges for ``temperature`` / ``top_p``.
- The request classifier routes ``responses`` to ``ResourceType.LLM``.
- ``stream=True`` routes through ``_stream_request`` and yields parsed
  :class:`ResponsesStreamEvent` chunks.
- ``text`` reaches the body with the API's own keys, and invalid formats fail
  before sending.
- ``web_search_call`` items replay in ``input``, with statuses of their own.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from venice_ai._queue_types import ResourceType
from venice_ai._request_classifier import RequestClassifier
from venice_ai.resources.responses import Responses
from venice_ai.types.api import (
    ResponsesFunctionCallOutput,
    ResponsesInputCustomToolCall,
    ResponsesInputWebSearchCall,
    ResponsesMessageOutput,
    ResponsesReasoningOutput,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponsesTextConfig,
    ResponsesTextFormatJSONSchema,
    ResponsesWebSearchCallOutput,
)


def _fake_response() -> ResponsesResponse:
    return ResponsesResponse.model_validate(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "zai-org-glm-5-1",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": ["thought"],
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "total_tokens": 11,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


@pytest.fixture
def resource() -> Responses:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response())
    return Responses(client)  # type: ignore[arg-type]


def _body_sent(resource: Responses) -> dict:
    return resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]


class TestResponsesCreate:
    @pytest.mark.asyncio
    async def test_minimal_call_posts_to_responses(self, resource: Responses) -> None:
        result = await resource.create(model="zai-org-glm-5-1", input="hi")
        assert isinstance(result, ResponsesResponse)
        resource._client.post.assert_called_once()  # type: ignore[attr-defined]
        path = resource._client.post.call_args.args[0]  # type: ignore[attr-defined]
        assert path == "responses"
        body = _body_sent(resource)
        assert body == {"model": "zai-org-glm-5-1", "input": "hi"}

    @pytest.mark.asyncio
    async def test_forwards_all_optional_params(self, resource: Responses) -> None:
        await resource.create(
            model="zai-org-glm-5-1",
            input="hi",
            include=["foo"],
            max_output_tokens=100,
            temperature=0.5,
            top_p=0.9,
            reasoning={"effort": "high", "summary": "concise"},
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            web_search=True,
            venice_parameters={"enable_web_search": "on"},
        )
        body = _body_sent(resource)
        assert body["include"] == ["foo"]
        assert body["max_output_tokens"] == 100
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.9
        assert body["reasoning"] == {"effort": "high", "summary": "concise"}
        assert body["tools"] == [{"type": "web_search"}]
        assert body["tool_choice"] == "auto"
        assert body["web_search"] is True
        assert body["venice_parameters"]["enable_web_search"] == "on"

    @pytest.mark.asyncio
    async def test_omits_none_params(self, resource: Responses) -> None:
        await resource.create(model="zai-org-glm-5-1", input="hi")
        body = _body_sent(resource)
        # No keys other than the two required ones.
        assert set(body.keys()) == {"model", "input"}

    @pytest.mark.asyncio
    async def test_list_input_forwarded_verbatim(self, resource: Responses) -> None:
        items = [{"type": "message", "role": "user", "content": "hi"}]
        await resource.create(model="zai-org-glm-5-1", input=items)
        body = _body_sent(resource)
        assert body["input"] == items


class TestResponsesResponseDeserialization:
    def test_deserialises_reasoning_and_message_blocks(self) -> None:
        resp = _fake_response()
        assert len(resp.output) == 2
        assert isinstance(resp.output[0], ResponsesReasoningOutput)
        assert resp.output[0].summary == ["thought"]
        assert isinstance(resp.output[1], ResponsesMessageOutput)
        assert resp.output[1].content[0].text == "Hi"

    def test_deserialises_function_call_block(self) -> None:
        payload = {
            "id": "resp_f",
            "object": "response",
            "created_at": 0,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "do_thing",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
        }
        resp = ResponsesResponse.model_validate(payload)
        assert isinstance(resp.output[0], ResponsesFunctionCallOutput)
        assert resp.output[0].name == "do_thing"
        assert resp.output[0].call_id == "call_1"

    def test_deserialises_web_search_call_block(self) -> None:
        payload = {
            "id": "resp_w",
            "object": "response",
            "created_at": 0,
            "model": "m",
            "status": "completed",
            "output": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}],
        }
        resp = ResponsesResponse.model_validate(payload)
        assert isinstance(resp.output[0], ResponsesWebSearchCallOutput)

    def test_error_block_populated_on_failure(self) -> None:
        payload = {
            "id": "resp_e",
            "object": "response",
            "created_at": 0,
            "model": "m",
            "status": "failed",
            "output": [],
            "error": {"code": "bad_thing", "message": "oops"},
        }
        resp = ResponsesResponse.model_validate(payload)
        assert resp.status == "failed"
        assert resp.error is not None
        assert resp.error.code == "bad_thing"


class TestResponsesRequestValidation:
    def test_rejects_temperature_above_2(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", temperature=2.5)

    def test_rejects_top_p_above_1(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", top_p=1.5)

    def test_rejects_zero_max_output_tokens(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", max_output_tokens=0)


class TestResponsesClassifier:
    @pytest.fixture
    def classifier(self) -> RequestClassifier:
        return RequestClassifier(MagicMock())

    @pytest.mark.asyncio
    async def test_responses_routes_to_llm(self, classifier: RequestClassifier) -> None:
        request = {"endpoint": "responses", "model": "zai-org-glm-5-1"}
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.LLM


class TestResponsesStreaming:
    @pytest.fixture
    def stream_resource(self) -> Responses:
        async def _events() -> AsyncIterator[ResponsesStreamEvent]:
            yield ResponsesStreamEvent.model_validate(
                {"type": "response.created", "sequence_number": 0}
            )
            yield ResponsesStreamEvent.model_validate(
                {"type": "response.output_text.delta", "sequence_number": 1, "delta": "hi"}
            )
            yield ResponsesStreamEvent.model_validate(
                {"type": "response.completed", "sequence_number": 2}
            )

        client = MagicMock()
        client._stream_request = MagicMock(return_value=_events())
        return Responses(client)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_stream_true_routes_to_stream_request(self, stream_resource: Responses) -> None:
        result: Any = await stream_resource.create(model="zai-org-glm-5-1", input="hi", stream=True)
        # Iteration consumes the underlying async generator.
        events = [e async for e in result]
        assert len(events) == 3
        assert events[0].type == "response.created"
        assert events[-1].type == "response.completed"
        # Verify the body sent to _stream_request had stream=True.
        call_kwargs = stream_resource._client._stream_request.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["path"] == "responses"
        assert call_kwargs["json_data"]["stream"] is True
        assert call_kwargs["cast_to"] is ResponsesStreamEvent


GREET_SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
}


class TestStructuredOutput:
    """``text.format`` asks for structured output from ``/responses``."""

    @pytest.mark.asyncio
    async def test_json_schema_dict_reaches_the_body_with_the_wire_key(
        self, resource: Responses
    ) -> None:
        text = {
            "format": {
                "type": "json_schema",
                "name": "greet",
                "schema": GREET_SCHEMA,
                "strict": True,
            }
        }
        await resource.create(model="m", input="hi", text=text)
        assert _body_sent(resource)["text"] == text

    @pytest.mark.asyncio
    async def test_typed_json_schema_format_sends_schema_not_schema_underscore(
        self, resource: Responses
    ) -> None:
        await resource.create(
            model="m",
            input="hi",
            text=ResponsesTextConfig(
                format=ResponsesTextFormatJSONSchema(
                    type="json_schema", name="greet", schema=GREET_SCHEMA
                )
            ),
        )
        assert _body_sent(resource)["text"] == {
            "format": {"type": "json_schema", "name": "greet", "schema": GREET_SCHEMA}
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["text", "json_object"])
    async def test_formats_without_a_schema(self, resource: Responses, kind: str) -> None:
        await resource.create(model="m", input="hi", text={"format": {"type": kind}})
        assert _body_sent(resource)["text"] == {"format": {"type": kind}}

    @pytest.mark.asyncio
    async def test_keys_the_sdk_does_not_model_are_still_sent(self, resource: Responses) -> None:
        text = {"format": {"type": "text", "future": 1}, "verbosity": "low"}
        await resource.create(model="m", input="hi", text=text)
        assert _body_sent(resource)["text"] == text

    @pytest.mark.parametrize(
        "bad",
        [
            {"format": {"type": "yaml"}},
            {"format": {"type": "json_schema", "schema": GREET_SCHEMA}},
            {"format": {"type": "json_schema", "name": "", "schema": GREET_SCHEMA}},
            {"format": {"type": "json_schema", "name": "greet"}},
        ],
    )
    def test_formats_the_api_rejects_fail_before_sending(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", text=bad)

    @pytest.mark.asyncio
    async def test_by_alias_dump_leaves_the_other_fields_unchanged(
        self, resource: Responses
    ) -> None:
        kwargs: dict[str, Any] = {
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "reasoning": {"effort": "high", "summary": "concise"},
            "tools": [{"type": "web_search", "external_web_access": False}],
            "venice_parameters": {"enable_web_search": "on", "character_slug": "c"},
            "anon_user_id": "user-1",
        }
        await resource.create(**kwargs)
        body = _body_sent(resource)
        assert body["tools"] == kwargs["tools"]
        assert body["input"] == kwargs["input"]
        assert body["reasoning"] == kwargs["reasoning"]
        assert body["anon_user_id"] == "user-1"
        assert body["venice_parameters"]["character_slug"] == "c"
        assert body["venice_parameters"]["enable_web_search"] == "on"


class TestChatOnlyVeniceParameters:
    @pytest.mark.asyncio
    async def test_are_left_out_of_the_body(self, resource: Responses) -> None:
        chat_only = {
            "strip_thinking_response": True,
            "disable_thinking": True,
            "return_search_results_as_documents": True,
            "enable_x_search": True,
        }
        await resource.create(
            model="m",
            input="hi",
            venice_parameters={**chat_only, "enable_web_search": "on"},
        )
        sent = _body_sent(resource)["venice_parameters"]
        assert not set(chat_only) & set(sent)
        assert sent["enable_web_search"] == "on"


class TestWebSearchCallReplay:
    """A ``web_search_call`` can go back in ``input`` as history."""

    @pytest.mark.parametrize(
        "status", ["in_progress", "searching", "completed", "failed", "incomplete"]
    )
    def test_typed_item_takes_every_status(self, status: str) -> None:
        item = ResponsesInputWebSearchCall(type="web_search_call", id="ws_1", status=status)
        assert item.status == status

    def test_typed_item_rejects_an_unknown_status(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesInputWebSearchCall(type="web_search_call", id="ws_1", status="bogus")

    @pytest.mark.parametrize("status", ["searching", "failed"])
    def test_web_search_statuses_stay_on_web_search_items(self, status: str) -> None:
        with pytest.raises(ValidationError):
            ResponsesInputCustomToolCall(
                type="custom_tool_call", call_id="c", name="n", input="i", status=status
            )

    @pytest.mark.parametrize(
        "action",
        [
            {
                "type": "search",
                "query": "q",
                "queries": ["q"],
                "sources": [{"type": "url", "url": "https://example.com"}],
            },
            {"type": "open_page", "url": "https://example.com"},
            {"type": "find_in_page", "url": "https://example.com", "pattern": "x"},
        ],
    )
    def test_typed_item_serialises_each_action(self, action: dict[str, Any]) -> None:
        item = ResponsesInputWebSearchCall(
            type="web_search_call", id="ws_1", status="completed", action=action
        )
        request = ResponsesRequest(model="m", input=[item])
        assert request.model_dump(exclude_none=True)["input"][0]["action"] == action

    def test_find_in_page_needs_a_pattern(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesInputWebSearchCall(
                type="web_search_call",
                id="ws_1",
                status="completed",
                action={"type": "find_in_page", "url": "https://example.com"},
            )

    def test_parsed_output_replays_through_model_dump(self) -> None:
        raw = {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "open_page", "url": "https://example.com"},
        }
        response = ResponsesResponse(
            id="r", object="response", created_at=1, model="m", status="completed", output=[raw]
        )
        block = response.output[0]
        assert isinstance(block, ResponsesWebSearchCallOutput)
        assert block.action == raw["action"]
        request = ResponsesRequest(model="m", input=[block.model_dump(exclude_none=True)])
        assert request.model_dump(exclude_none=True)["input"] == [raw]
