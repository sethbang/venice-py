"""Responses API request fields, tool types, input items, and output items.

Covers ``instructions``, ``parallel_tool_calls``, ``prompt_cache_retention``,
the ``tool_search`` and deferred-loading function tools, named
``tool_choice`` objects, the ``custom_tool_call`` / ``tool_search`` input
items, and the ``custom_tool_call`` / ``tool_search_call`` output items.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from venice_ai._sync_client import SyncVeniceClient
from venice_ai.resources.responses import Responses
from venice_ai.types.api import (
    KNOWN_RESPONSE_CUSTOM_TOOL_CALL_STATUSES,
    KNOWN_RESPONSE_TOOL_SEARCH_CALL_STATUSES,
    KNOWN_RESPONSE_TOOL_SEARCH_EXECUTIONS,
    ResponsesCustomToolCallOutput,
    ResponsesFunctionCallOutput,
    ResponsesFunctionTool,
    ResponsesFunctionToolChoice,
    ResponsesFunctionToolDefinition,
    ResponsesInputAdditionalTools,
    ResponsesInputCustomToolCall,
    ResponsesInputCustomToolCallOutput,
    ResponsesInputToolSearchCall,
    ResponsesInputToolSearchOutput,
    ResponsesNamedToolChoice,
    ResponsesReasoningOutput,
    ResponsesReasoningSummaryText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponsesToolChoiceFunctionName,
    ResponsesToolSearchCallOutput,
    ResponsesToolSearchTool,
    ResponsesUnknownOutput,
    Tool,
    ToolFunction,
)

BASE = {"id": "r", "object": "response", "created_at": 1, "model": "m", "status": "completed"}


def _ok() -> ResponsesResponse:
    return ResponsesResponse.model_validate({**BASE, "output": []})


@pytest.fixture
def resource() -> Responses:
    client = MagicMock()
    client.post = AsyncMock(return_value=_ok())
    return Responses(client)


@pytest.fixture
def stream_resource() -> Responses:
    async def _events() -> AsyncIterator[ResponsesStreamEvent]:
        yield ResponsesStreamEvent.model_validate({"type": "response.completed"})

    client = MagicMock()
    client._stream_request = MagicMock(return_value=_events())
    return Responses(client)


def _body(resource: Responses) -> dict[str, Any]:
    body: dict[str, Any] = resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    return body


def _stream_body(resource: Responses) -> dict[str, Any]:
    body: dict[str, Any] = resource._client._stream_request.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    return body


NEW_PARAMS: list[tuple[str, Any]] = [
    ("instructions", "Answer in French."),
    ("parallel_tool_calls", False),
    ("prompt_cache_retention", "24h"),
]


class TestNewTopLevelParams:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("name", "value"), NEW_PARAMS)
    async def test_param_sent_when_passed(self, resource: Responses, name: str, value: Any) -> None:
        await resource.create(model="m", input="hi", **{name: value})
        assert _body(resource) == {"model": "m", "input": "hi", name: value}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("name", "value"), NEW_PARAMS)
    async def test_param_sent_when_streaming(
        self, stream_resource: Responses, name: str, value: Any
    ) -> None:
        result: Any = await stream_resource.create(
            model="m", input="hi", stream=True, **{name: value}
        )
        _ = [e async for e in result]
        assert _stream_body(stream_resource) == {
            "model": "m",
            "input": "hi",
            name: value,
            "stream": True,
        }

    @pytest.mark.asyncio
    async def test_params_omitted_when_not_passed(self, resource: Responses) -> None:
        await resource.create(model="m", input="hi")
        body = _body(resource)
        for name, _ in NEW_PARAMS:
            assert name not in body

    @pytest.mark.asyncio
    async def test_params_omitted_when_streaming(self, stream_resource: Responses) -> None:
        result: Any = await stream_resource.create(model="m", input="hi", stream=True)
        _ = [e async for e in result]
        assert _stream_body(stream_resource) == {"model": "m", "input": "hi", "stream": True}

    @pytest.mark.parametrize("value", ["in_memory", "24h", "default", "extended"])
    def test_prompt_cache_retention_accepts_spec_values(self, value: str) -> None:
        req = ResponsesRequest(model="m", input="hi", prompt_cache_retention=value)  # type: ignore[arg-type]
        assert req.model_dump(exclude_none=True)["prompt_cache_retention"] == value

    def test_prompt_cache_retention_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", prompt_cache_retention="1h")  # type: ignore[arg-type]


class TestToolChoice:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "choice",
        [
            {"type": "function", "name": "get_weather"},
            {"type": "custom", "name": "apply_patch", "namespace": "editor"},
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "allowed_tools", "mode": "auto", "tools": []},
        ],
    )
    async def test_dict_choice_sent_verbatim(
        self, resource: Responses, choice: dict[str, Any]
    ) -> None:
        await resource.create(model="m", input="hi", tool_choice=choice)
        assert _body(resource)["tool_choice"] == choice

    @pytest.mark.asyncio
    async def test_named_custom_choice_model(self, resource: Responses) -> None:
        await resource.create(
            model="m",
            input="hi",
            tool_choice=ResponsesNamedToolChoice(type="custom", name="apply_patch", namespace="ed"),
        )
        assert _body(resource)["tool_choice"] == {
            "type": "custom",
            "name": "apply_patch",
            "namespace": "ed",
        }

    @pytest.mark.asyncio
    async def test_named_choice_omits_namespace_when_not_set(self, resource: Responses) -> None:
        await resource.create(
            model="m", input="hi", tool_choice=ResponsesNamedToolChoice(type="function", name="f")
        )
        assert _body(resource)["tool_choice"] == {"type": "function", "name": "f"}

    @pytest.mark.asyncio
    async def test_chat_shape_choice_model(self, resource: Responses) -> None:
        await resource.create(
            model="m",
            input="hi",
            tool_choice=ResponsesFunctionToolChoice(
                type="function", function=ResponsesToolChoiceFunctionName(name="f")
            ),
        )
        assert _body(resource)["tool_choice"] == {"type": "function", "function": {"name": "f"}}

    @pytest.mark.parametrize("mode", ["auto", "none", "required"])
    def test_mode_strings_accepted(self, mode: str) -> None:
        req = ResponsesRequest(model="m", input="hi", tool_choice=mode)  # type: ignore[arg-type]
        assert req.tool_choice == mode

    def test_unknown_mode_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", tool_choice="any")  # type: ignore[arg-type]

    def test_named_choice_rejects_unknown_type(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesNamedToolChoice(type="shell", name="x")  # type: ignore[arg-type]


class TestTools:
    @pytest.mark.asyncio
    async def test_deferred_function_tool(self, resource: Responses) -> None:
        tool = ResponsesFunctionTool(
            type="function",
            defer_loading=True,
            function=ResponsesFunctionToolDefinition(
                name="get_weather", parameters={"type": "object"}, defer_loading=True
            ),
        )
        await resource.create(model="m", input="hi", tools=[tool])
        assert _body(resource)["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                    "defer_loading": True,
                },
                "defer_loading": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_tool_search_tool(self, resource: Responses) -> None:
        tool = ResponsesToolSearchTool(
            type="tool_search",
            execution="client",
            parameters={"type": "object"},
            description="Find tools by keyword.",
        )
        await resource.create(model="m", input="hi", tools=[tool])
        assert _body(resource)["tools"] == [
            {
                "type": "tool_search",
                "execution": "client",
                "parameters": {"type": "object"},
                "description": "Find tools by keyword.",
            }
        ]

    def test_tool_search_requires_execution_and_parameters(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesToolSearchTool(type="tool_search")  # type: ignore[call-arg]

    def test_tool_search_rejects_server_execution(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesToolSearchTool(
                type="tool_search",
                execution="server",  # type: ignore[arg-type]
                parameters={"type": "object"},
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "tool_search", "execution": "client", "parameters": {"type": "object"}},
            {"type": "function", "defer_loading": True, "function": {"name": "f"}},
            {"type": "function", "name": "flat", "parameters": {"type": "object"}},
            {"type": "custom", "name": "apply_patch"},
            {"type": "x_search", "allowed_x_handles": ["venice"]},
        ],
    )
    async def test_dict_tools_sent_verbatim(
        self, resource: Responses, tool: dict[str, Any]
    ) -> None:
        await resource.create(model="m", input="hi", tools=[tool])
        assert _body(resource)["tools"] == [tool]

    def test_existing_tool_bodies_unchanged(self) -> None:
        """Function tools that validate as the shared ``Tool`` keep their prior JSON."""
        req = ResponsesRequest(
            model="m",
            input="hi",
            tools=[
                {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}},
                {"type": "web_search"},
                Tool(type="function", function=ToolFunction(name="g")),
            ],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
        assert json.dumps(req.model_dump(exclude_none=True)) == json.dumps(
            {
                "model": "m",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "object"},
                            "strict": False,
                        },
                    },
                    {"type": "web_search"},
                    {
                        "type": "function",
                        "function": {"name": "g", "parameters": {}, "strict": False},
                    },
                ],
                "tool_choice": {"type": "function", "function": {"name": "f"}},
            }
        )


class TestInputItems:
    @pytest.mark.asyncio
    async def test_typed_custom_tool_round_trip(self, resource: Responses) -> None:
        await resource.create(
            model="m",
            input=[
                {"type": "message", "role": "user", "content": "patch it"},
                ResponsesInputCustomToolCall(
                    type="custom_tool_call",
                    call_id="call_1",
                    name="apply_patch",
                    namespace="editor",
                    input="*** Begin Patch",
                ),
                ResponsesInputCustomToolCallOutput(
                    type="custom_tool_call_output", call_id="call_1", output="applied"
                ),
            ],
        )
        assert _body(resource)["input"] == [
            {"type": "message", "role": "user", "content": "patch it"},
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch",
                "namespace": "editor",
            },
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "applied"},
        ]

    @pytest.mark.asyncio
    async def test_typed_tool_search_round_trip(self, resource: Responses) -> None:
        await resource.create(
            model="m",
            input=[
                ResponsesInputToolSearchCall(
                    type="tool_search_call",
                    call_id="ts_1",
                    execution="client",
                    arguments={"query": "weather"},
                ),
                ResponsesInputToolSearchOutput(
                    type="tool_search_output",
                    call_id="ts_1",
                    execution="client",
                    tools=[
                        ResponsesFunctionTool(
                            type="function",
                            function=ResponsesFunctionToolDefinition(name="get_weather"),
                        )
                    ],
                ),
                ResponsesInputAdditionalTools(
                    type="additional_tools", role="developer", tools=[{"type": "web_search"}]
                ),
            ],
        )
        assert _body(resource)["input"] == [
            {
                "type": "tool_search_call",
                "call_id": "ts_1",
                "execution": "client",
                "arguments": {"query": "weather"},
            },
            {
                "type": "tool_search_output",
                "call_id": "ts_1",
                "execution": "client",
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
            {"type": "additional_tools", "tools": [{"type": "web_search"}], "role": "developer"},
        ]

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "custom_tool_call", "call_id": "c", "name": "n", "input": "raw"},
            {"type": "custom_tool_call_output", "call_id": "c", "output": {"ok": True}},
            {"type": "custom_tool_call_output", "call_id": "c", "output": 3},
            {"type": "tool_search_call", "call_id": "c", "execution": "client", "arguments": {}},
            {
                "type": "tool_search_output",
                "call_id": "c",
                "execution": "client",
                "status": "completed",
                "tools": [{"type": "function", "function": {"name": "g"}}],
            },
            {
                "type": "function_call",
                "call_id": "c",
                "name": "f",
                "namespace": "ns",
                "arguments": "{}",
            },
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "t"}]},
        ],
    )
    def test_dict_items_sent_verbatim(self, item: dict[str, Any]) -> None:
        req = ResponsesRequest(model="m", input=[item])
        assert req.model_dump(exclude_none=True)["input"] == [item]

    def test_custom_tool_call_rejects_unknown_status(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesInputCustomToolCall(
                type="custom_tool_call",
                call_id="c",
                name="n",
                input="x",
                status="queued",  # type: ignore[arg-type]
            )

    def test_tool_search_call_requires_non_empty_call_id(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesInputToolSearchCall(
                type="tool_search_call", call_id="", execution="client", arguments={}
            )


class TestOutputItems:
    def test_custom_tool_call(self) -> None:
        r = ResponsesResponse.model_validate(
            {
                **BASE,
                "output": [
                    {
                        "type": "custom_tool_call",
                        "id": "ctc_1",
                        "call_id": "call_1",
                        "name": "apply_patch",
                        "namespace": "editor",
                        "input": "*** Begin Patch",
                        "status": "completed",
                    }
                ],
            }
        )
        item = r.output[0]
        assert isinstance(item, ResponsesCustomToolCallOutput)
        assert item.input == "*** Begin Patch"
        assert item.namespace == "editor"
        assert item.status in KNOWN_RESPONSE_CUSTOM_TOOL_CALL_STATUSES

    def test_tool_search_call(self) -> None:
        r = ResponsesResponse.model_validate(
            {
                **BASE,
                "output": [
                    {
                        "type": "tool_search_call",
                        "id": "tsc_1",
                        "call_id": "ts_1",
                        "execution": "client",
                        "arguments": {"query": "weather"},
                        "status": "completed",
                    }
                ],
            }
        )
        item = r.output[0]
        assert isinstance(item, ResponsesToolSearchCallOutput)
        assert item.arguments == {"query": "weather"}
        assert item.execution in KNOWN_RESPONSE_TOOL_SEARCH_EXECUTIONS
        assert item.status in KNOWN_RESPONSE_TOOL_SEARCH_CALL_STATUSES

    def test_unknown_status_and_execution_values_are_kept(self) -> None:
        r = ResponsesResponse.model_validate(
            {
                **BASE,
                "output": [
                    {
                        "type": "tool_search_call",
                        "id": "tsc_1",
                        "call_id": "ts_1",
                        "execution": "server",
                        "arguments": {},
                        "status": "searching",
                    },
                    {
                        "type": "custom_tool_call",
                        "id": "ctc_1",
                        "call_id": "c",
                        "name": "n",
                        "input": "",
                        "status": "queued",
                    },
                ],
            }
        )
        assert isinstance(r.output[0], ResponsesToolSearchCallOutput)
        assert r.output[0].execution == "server"
        assert isinstance(r.output[1], ResponsesCustomToolCallOutput)
        assert r.output[1].status == "queued"

    def test_function_call_namespace(self) -> None:
        r = ResponsesResponse.model_validate(
            {
                **BASE,
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "c",
                        "name": "f",
                        "namespace": "ns",
                        "arguments": "{}",
                        "status": "completed",
                    }
                ],
            }
        )
        item = r.output[0]
        assert isinstance(item, ResponsesFunctionCallOutput)
        assert item.namespace == "ns"

    def test_function_call_without_namespace(self) -> None:
        r = ResponsesResponse.model_validate(
            {
                **BASE,
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "c",
                        "name": "f",
                        "arguments": "{}",
                        "status": "completed",
                    }
                ],
            }
        )
        item = r.output[0]
        assert isinstance(item, ResponsesFunctionCallOutput)
        assert item.namespace is None

    def test_reasoning_summary_text_parts(self) -> None:
        r = ResponsesResponse.model_validate(
            {
                **BASE,
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "Checked the docs."}],
                        "encrypted_content": "enc",
                    }
                ],
            }
        )
        item = r.output[0]
        assert isinstance(item, ResponsesReasoningOutput)
        assert item.summary is not None
        part = item.summary[0]
        assert isinstance(part, ResponsesReasoningSummaryText)
        assert part.text == "Checked the docs."

    def test_incomplete_known_type_falls_back_to_unknown(self) -> None:
        r = ResponsesResponse.model_validate(
            {**BASE, "output": [{"type": "custom_tool_call", "id": "ctc_1"}]}
        )
        assert isinstance(r.output[0], ResponsesUnknownOutput)
        assert r.output[0].type == "custom_tool_call"


class TestSyncClientForwarding:
    def test_sync_client_forwards_new_params(self) -> None:
        inner = MagicMock()
        inner.post = AsyncMock(return_value=_ok())
        async_client = Mock()
        async_client.responses = Responses(inner)
        async_client.base_url = "https://api.venice.ai/api/v1/"
        async_client.close = AsyncMock()
        with patch("venice_ai._sync_client.VeniceClient", return_value=async_client):
            client = SyncVeniceClient(api_key="test-key")
        try:
            result = client.responses.create(
                model="m",
                input="hi",
                instructions="Be brief.",
                parallel_tool_calls=True,
                prompt_cache_retention="extended",
                tool_choice={"type": "custom", "name": "apply_patch"},
            )
        finally:
            client.close()
        assert isinstance(result, ResponsesResponse)
        assert inner.post.call_args.kwargs["json_data"] == {
            "model": "m",
            "input": "hi",
            "instructions": "Be brief.",
            "parallel_tool_calls": True,
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "prompt_cache_retention": "extended",
        }


class TestSequenceArguments:
    @pytest.mark.asyncio
    async def test_tuple_input_and_tools_sent_as_lists(self, resource: Responses) -> None:
        items: tuple[dict[str, Any], ...] = ({"type": "message", "role": "user", "content": "hi"},)
        tools: tuple[dict[str, Any], ...] = ({"type": "web_search"},)
        await resource.create(model="m", input=items, tools=tools)
        body = _body(resource)
        assert body["input"] == [{"type": "message", "role": "user", "content": "hi"}]
        assert body["tools"] == [{"type": "web_search"}]


class TestDictsSentAsWritten:
    """A dict that happens to match a typed model is still sent exactly as
    written: same key order, explicit ``None`` values kept."""

    @pytest.mark.asyncio
    async def test_input_dict_matching_typed_item(self, resource: Responses) -> None:
        item = {
            "input": "i",
            "type": "custom_tool_call",
            "name": "n",
            "call_id": "c",
            "namespace": None,
        }
        await resource.create(model="m", input=[item])
        sent = _body(resource)["input"][0]
        assert json.dumps(sent) == json.dumps(item)

    @pytest.mark.asyncio
    async def test_tool_choice_dict_matching_named_choice(self, resource: Responses) -> None:
        choice = {"name": "f", "type": "function"}
        await resource.create(model="m", input="x", tool_choice=choice)
        assert json.dumps(_body(resource)["tool_choice"]) == json.dumps(choice)

    @pytest.mark.asyncio
    async def test_tool_dicts_matching_typed_tools(self, resource: Responses) -> None:
        tools = [
            {"type": "tool_search", "execution": "client", "parameters": {}, "description": None},
            {"type": "function", "defer_loading": True, "function": {"name": "f"}},
        ]
        await resource.create(model="m", input="x", tools=tools)
        assert json.dumps(_body(resource)["tools"]) == json.dumps(tools)

    @pytest.mark.asyncio
    async def test_typed_models_still_serialize_typed(self, resource: Responses) -> None:
        await resource.create(
            model="m",
            input="x",
            tool_choice=ResponsesNamedToolChoice(type="custom", name="apply_patch"),
        )
        assert _body(resource)["tool_choice"] == {"type": "custom", "name": "apply_patch"}
