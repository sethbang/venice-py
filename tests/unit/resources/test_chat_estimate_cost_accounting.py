"""``estimate_cost`` counts every billed prompt component, not only message words.

Measured live on 2026-10-02 with the Venice system prompt off: the same 9-word
user prompt was billed 14 (mistral-small), 18 (gpt-4o-mini), 21 (qwen3-5-9b),
23 (glm-4.7-flash) and 27 (nemotron-3-nano) prompt tokens. The word heuristic
alone gives 11, below every one of them.
"""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from venice_ai.costs import (
    CHAT_MESSAGE_TOKEN_ALLOWANCE,
    CHAT_TEMPLATE_TOKEN_ALLOWANCE,
    SCHEMA_JSON_CHARS_PER_TOKEN,
    TOOLS_TEMPLATE_TOKEN_ALLOWANCE,
)
from venice_ai.resources.chat.completions import ChatCompletions, _schema_response_format
from venice_ai.types.api import SystemMessage, Tool, UserMessage
from venice_ai.types.api.models import ModelResponse, ModelsListResponse

_MODEL = "fake-chat-test-model"
_OFF = {"include_venice_system_prompt": False}
_MEASURED_PROMPT = "Give me three benefits of unit tests, one line each."
_MEASURED_BILLED = (14, 18, 21, 23, 27)


def _chat() -> ChatCompletions:
    entry = ModelResponse.model_validate(
        {
            "id": _MODEL,
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": _MODEL,
                "availableContextTokens": 8192.0,
                "pricing": {
                    "input": {"usd": 1.0, "diem": 1.0},
                    "output": {"usd": 2.0, "diem": 2.0},
                },
            },
        }
    )
    client = MagicMock()
    client.models.list = AsyncMock(
        return_value=ModelsListResponse(object="list", type="text", data=[entry])
    )
    return ChatCompletions(client)


async def test_estimate_is_an_upper_bound_on_measured_prompts() -> None:
    estimate = await _chat().estimate_cost(
        model=_MODEL, messages=[UserMessage(content=_MEASURED_PROMPT)], venice_parameters=_OFF
    )
    assert estimate.prompt_tokens >= max(_MEASURED_BILLED)
    assert estimate.template_overhead_tokens == (
        CHAT_TEMPLATE_TOKEN_ALLOWANCE + CHAT_MESSAGE_TOKEN_ALLOWANCE
    )


async def test_each_message_adds_its_allowance() -> None:
    chat = _chat()
    one = await chat.estimate_cost(
        model=_MODEL, messages=[UserMessage(content="hi")], venice_parameters=_OFF
    )
    two = await chat.estimate_cost(
        model=_MODEL,
        messages=[SystemMessage(content="hi"), UserMessage(content="hi")],
        venice_parameters=_OFF,
    )
    assert two.template_overhead_tokens - one.template_overhead_tokens == (
        CHAT_MESSAGE_TOKEN_ALLOWANCE
    )


class Answer(BaseModel):
    title: str
    score: int


# Tool and schema definitions measured live on 2026-10-02 (Venice system
# prompt off, max_completion_tokens=1) on five chat models from five providers.
_SMALL_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up a customer order by its id and return its status.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order id"}},
            "required": ["order_id"],
        },
    },
}
_LARGE_PARAMS = {
    "type": "object",
    "properties": {
        "origin": {
            "type": "string",
            "description": "IATA code of the departure airport, for example SFO or LHR.",
        },
        "destination": {"type": "string", "description": "IATA code of the arrival airport."},
        "departure_date": {
            "type": "string",
            "description": "Departure date in ISO 8601 format (YYYY-MM-DD).",
        },
        "return_date": {
            "type": "string",
            "description": "Optional return date in ISO 8601 format; omit for one-way trips.",
        },
        "cabin": {
            "type": "string",
            "enum": ["economy", "premium_economy", "business", "first"],
            "description": "Cabin class to search.",
        },
        "passengers": {
            "type": "object",
            "description": "Passenger counts by type.",
            "properties": {
                "adults": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Travellers aged 12 or over.",
                },
                "children": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Travellers aged 2 to 11.",
                },
                "infants": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Travellers under 2 sitting on a lap.",
                },
            },
            "required": ["adults"],
        },
        "max_stops": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
            "description": "Maximum number of connections allowed.",
        },
        "flexible_dates": {
            "type": "boolean",
            "description": "Search three days either side of the given dates.",
        },
    },
    "required": ["origin", "destination", "departure_date", "cabin", "passengers"],
}
_LARGE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": (
            "Search available flights between two airports and return fares, "
            "durations and connection details for each itinerary."
        ),
        "parameters": _LARGE_PARAMS,
    },
}
_SMALL_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        },
    },
}
_LARGE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "flight_query", "strict": True, "schema": _LARGE_PARAMS},
}
_TOOL_PROMPT = "List three benefits of unit tests, one line each."
_REQUESTS: dict[str, dict] = {
    "tool_small": {"tools": [_SMALL_TOOL]},
    "tool_large": {"tools": [_LARGE_TOOL]},
    "tools_2": {"tools": [_SMALL_TOOL, _LARGE_TOOL]},
    "rf_small": {"response_format": _SMALL_FORMAT},
    "rf_large": {"response_format": _LARGE_FORMAT},
}
# Billed prompt tokens per model (one row per chat template measured). Three
# models enforce a json_schema while decoding and bill nothing for it; the
# fourth model's json_schema requests failed server-side, so it has no rf rows.
_BILLED: list[dict[str, int]] = [
    {"tool_small": 294, "tool_large": 737, "tools_2": 814},
    {"tool_small": 90, "tool_large": 444, "tools_2": 517, "rf_small": 14, "rf_large": 14},
    {"tool_small": 290, "tool_large": 761, "tools_2": 884, "rf_small": 20, "rf_large": 20},
    {"tool_small": 174, "tool_large": 515, "tools_2": 588, "rf_small": 19, "rf_large": 19},
    {"tool_small": 61, "tool_large": 268, "tools_2": 301, "rf_small": 47, "rf_large": 314},
]


def _wire_chars(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":")))


async def test_schema_estimate_bounds_every_measured_template() -> None:
    chat = _chat()
    for name, extra in _REQUESTS.items():
        estimate = await chat.estimate_cost(
            model=_MODEL,
            messages=[UserMessage(content=_TOOL_PROMPT)],
            venice_parameters=_OFF,
            **extra,
        )
        for row in _BILLED:
            if name in row:
                assert estimate.prompt_tokens >= row[name], (name, row[name])


async def test_tools_add_the_preamble_and_their_json() -> None:
    chat = _chat()
    base = await chat.estimate_cost(
        model=_MODEL, messages=[UserMessage(content="hi")], venice_parameters=_OFF
    )
    with_tools = await chat.estimate_cost(
        model=_MODEL,
        messages=[UserMessage(content="hi")],
        venice_parameters=_OFF,
        tools=[_SMALL_TOOL, _LARGE_TOOL],
    )
    expected = TOOLS_TEMPLATE_TOKEN_ALLOWANCE + math.ceil(
        _wire_chars([_SMALL_TOOL, _LARGE_TOOL]) / SCHEMA_JSON_CHARS_PER_TOKEN
    )
    assert with_tools.schema_tokens == expected
    assert with_tools.prompt_tokens == base.prompt_tokens + expected
    assert with_tools.prompt_cost_usd > base.prompt_cost_usd

    no_tools = await chat.estimate_cost(
        model=_MODEL, messages=[UserMessage(content="hi")], venice_parameters=_OFF, tools=[]
    )
    assert no_tools.schema_tokens == 0


async def test_tool_models_are_counted_like_dicts() -> None:
    chat = _chat()
    as_dict = await chat.estimate_cost(
        model=_MODEL, messages=[UserMessage(content="hi")], tools=[_SMALL_TOOL]
    )
    as_model = await chat.estimate_cost(
        model=_MODEL, messages=[UserMessage(content="hi")], tools=[Tool.model_validate(_SMALL_TOOL)]
    )
    assert as_model.schema_tokens >= as_dict.schema_tokens > TOOLS_TEMPLATE_TOKEN_ALLOWANCE


async def test_response_format_class_is_counted_as_the_format_sent() -> None:
    chat = _chat()
    with_schema = await chat.estimate_cost(
        model=_MODEL,
        messages=[UserMessage(content="hi")],
        venice_parameters=_OFF,
        response_format=Answer,
    )
    sent = _schema_response_format(Answer).model_dump(exclude_none=True, by_alias=True)
    assert sent["json_schema"]["schema"] == Answer.model_json_schema()
    # No tool preamble: a response format alone adds only its JSON.
    assert with_schema.schema_tokens == math.ceil(_wire_chars(sent) / SCHEMA_JSON_CHARS_PER_TOKEN)
