#!/usr/bin/env python3
"""
Venice AI SDK - Responses API Example (beta)
=============================================

Venice exposes an OpenAI-compatible **Responses API** at ``POST /responses``,
wrapped by ``client.responses.create(...)``. Unlike ``/chat/completions``, the
endpoint is *stateless* (no conversation state is persisted between calls) and
returns a typed ``output`` array of blocks — ``reasoning``, ``message``,
``function_call``, ``web_search_call`` — rather than ``choices``.

This example demonstrates:

- A basic ``client.responses.create`` call with a plain-string ``input``
- Reading the typed ``output`` blocks and extracting the assistant text
- A structured (multi-message) ``input``
- A reasoning model's ``reasoning`` block next to its ``message`` block
- Hitting ``max_output_tokens`` on purpose: ``status="incomplete"`` plus
  ``incomplete_details`` explain why the text stops early
- Inspecting the ``usage`` token block on the result
- ``text.format``: structured output that follows a JSON Schema
- ``instructions``: system instructions kept apart from ``input``
- ``stream=True``: text deltas, then the full response on the terminal event
- A function-tool round trip: ``function_call`` block in, ``function_call_output``
  back, with the conversation resent because the endpoint stores nothing
- ``web_search=True``: a search, read back from the output blocks

``status`` is the Responses API's equivalent of ``finish_reason``: only
``"completed"`` means the model finished its answer.

The endpoint is tagged **beta** by Venice; request/response shapes may change
without notice, and some accounts may not be entitled. A demo the account
cannot access (HTTP 403), or one whose kind of model the catalog lacks, prints
a ``Section skipped:`` line. The basic call is the core of the example: when it
cannot run, the script prints ``SKIPPED:`` and exits 77, otherwise it exits 0
only if every demo that ran passed.

Cost: about $0.015 per run. The ``web_search`` demo accounts for most of that:
Venice's pricing page lists web search at $10 per 1,000 requests (about $0.01 a
request), on top of its token charges. Every other demo is a fraction of a cent
in tokens. The basic, structured-input, reasoning and truncation demos use the
catalog's default ranking, so their price follows that pick; the others use the
cheapest suitable model that does not reason.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import json
import re
import sys
from typing import Any

from pydantic import ValidationError

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import PermissionDeniedError, VeniceError
from venice_ai.types.api import TextModelSpec
from venice_ai.types.api.requests import VeniceParameters
from venice_ai.types.api.requests.common import Tool, ToolFunction
from venice_ai.types.api.responses import (
    ResponsesFunctionCallOutput,
    ResponsesMessageOutput,
    ResponsesOutputText,
    ResponsesReasoningOutput,
    ResponsesReasoningSummaryText,
    ResponsesResponse,
    ResponsesWebSearchCallOutput,
)

# Leave out Venice's default system prompt. A model's own prompt template can
# still add input tokens of its own, which this switch does not remove.
NO_SYSTEM_PROMPT = VeniceParameters(include_venice_system_prompt=False)

#: Exit code for "skipped": no chat model in the catalog, or the account has
#: no access to the Responses API.
EXIT_SKIPPED = 77

#: Strict structured output: every property required, no others allowed.
ORDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "integer"},
        "quantity": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["order_id", "quantity", "city"],
    "additionalProperties": False,
}
EXPECTED_ORDER = {"order_id": 1842, "quantity": 3, "city": "Oslo"}

#: An invented value: an answer that states it can only have come from the tool.
LOCKER_CODE = "QX-7319"

LOCKER_TOOL = Tool(
    type="function",
    function=ToolFunction(
        name="lookup_locker_code",
        description="Look up the locker code of a gym member.",
        parameters={
            "type": "object",
            "properties": {"member": {"type": "string", "description": "The member's name."}},
            "required": ["member"],
        },
    ),
)


def extract_text(response: ResponsesResponse) -> str:
    """Pull the assistant text out of a Responses ``output`` array.

    The Responses API returns a list of typed blocks instead of a single
    ``message``. Assistant text lives inside ``message`` blocks, whose
    ``content`` is a list of ``output_text`` items. We concatenate every
    ``output_text`` we find across all ``message`` blocks.
    """
    parts: list[str] = []
    for block in response.output:
        if isinstance(block, ResponsesMessageOutput):
            for item in block.content:
                # ``output_text`` items carry the text; a content type the SDK
                # does not know (e.g. a refusal) arrives as a plain dict.
                if isinstance(item, ResponsesOutputText):
                    parts.append(item.text)
    return "\n".join(parts).strip()


def summarize_output_blocks(response: ResponsesResponse) -> None:
    """Print a short breakdown of the typed blocks in ``response.output``."""
    print(f"   📦 {len(response.output)} output block(s):")
    for i, block in enumerate(response.output):
        # ``.type`` is present on every block variant (including the
        # forward-compat fallback for unmodeled types).
        label = block.type
        if isinstance(block, ResponsesReasoningOutput):
            summary = block.summary or []
            label += f" (summary field: {len(summary)} item(s))"
        elif isinstance(block, ResponsesMessageOutput):
            label += f" (role={block.role}, status={block.status})"
        print(f"      {i + 1}. {label}")


def print_usage(response: ResponsesResponse) -> None:
    """Print the usage block, including reasoning tokens when reported."""
    if not response.usage:
        return
    u = response.usage
    print("\n📊 Token Usage:")
    print(f"   Input tokens:  {u.input_tokens}")
    print(f"   Output tokens: {u.output_tokens}")
    if u.output_tokens_details and u.output_tokens_details.reasoning_tokens:
        print(f"   (of which reasoning: {u.output_tokens_details.reasoning_tokens})")
    print(f"   Total tokens:  {u.total_tokens}")


def reached_cap(response: ResponsesResponse, max_output_tokens: int) -> bool:
    """``True`` when the response used its whole ``max_output_tokens`` budget."""
    return response.usage is not None and response.usage.output_tokens >= max_output_tokens


def finished(response: ResponsesResponse, text: str, max_output_tokens: int) -> bool:
    """``True`` when the response completed and produced assistant text.

    Not every backend sets ``status="incomplete"`` when the cap cuts the text
    off, so a "completed" response that used the whole budget is not trusted.
    """
    if response.status != "completed":
        reason = response.incomplete_details.reason if response.incomplete_details else "?"
        print(f"\n❌ Response did not complete: status={response.status} (reason: {reason})")
        return False
    if reached_cap(response, max_output_tokens):
        print(f"\n❌ Reported completed, but used all {max_output_tokens} output tokens")
        return False
    if not text:
        print("\n❌ Completed, but no assistant text was found in the output blocks")
        return False
    return True


async def basic_responses_create(client: VeniceClient, model: str) -> bool:
    """A basic ``client.responses.create`` call with a string input."""
    print("💬 Basic Responses API Call")
    print("-" * 40)

    # The simplest form: ``input`` is a plain string prompt.
    max_output_tokens = 1024
    response = await client.responses.create(
        model=model,
        input="Summarize the Treaty of Versailles in two sentences.",
        max_output_tokens=max_output_tokens,
        # Venice prepends its own system prompt unless told not to, and bills
        # it as input tokens (over a thousand for a one-line prompt). This
        # request needs nothing from it, so it is left out. Some models add a
        # built-in system prompt of their own, which is still billed, so the
        # input count can stay well above the length of the prompt.
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    print(f"🆔 Response id: {response.id}")
    print(f"   Object:  {response.object}")
    print(f"   Status:  {response.status}")
    print(f"   Model:   {response.model}")

    summarize_output_blocks(response)

    text = extract_text(response)
    if text:
        print(f"\n🗣️ Assistant:\n{text}")

    print_usage(response)
    print("   ℹ️  Venice's system prompt was left out (include_venice_system_prompt=False);")
    print("      a model's own built-in prompt can still add input tokens beyond the prompt")

    return finished(response, text, max_output_tokens)


async def structured_input_responses(client: VeniceClient, model: str) -> bool:
    """A structured (multi-message) ``input`` payload.

    ``input`` also accepts a list of structured input items (OpenAI Responses
    shape) instead of a plain string. Here we pass a system-style instruction
    followed by a user turn.
    """
    print("\n🧱 Structured Input (message list)")
    print("-" * 40)

    # Each item mirrors the OpenAI Responses input-message shape.
    structured_input = [
        {
            "role": "system",
            "content": "You are a terse assistant. Answer in one short sentence.",
        },
        {
            "role": "user",
            "content": "What is the capital of Japan?",
        },
    ]

    max_output_tokens = 256
    response = await client.responses.create(
        model=model,
        input=structured_input,
        max_output_tokens=max_output_tokens,
        temperature=0.2,
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    print(f"🆔 Response id: {response.id}  (status={response.status})")
    summarize_output_blocks(response)

    text = extract_text(response)
    if text:
        print(f"\n🗣️ Assistant: {text}")

    if not finished(response, text, max_output_tokens):
        return False
    if "tokyo" not in text.lower():
        print("\n❌ Expected the answer to name Tokyo")
        return False
    return True


async def reasoning_output_blocks(client: VeniceClient, e2ee_models: list[str]) -> bool | None:
    """Read the ``reasoning`` block a reasoning model returns before its answer.

    The model comes from the resolver's default ranking for reasoning models,
    which follows Venice's own recommendations. Not every backend returns a
    ``reasoning`` block on ``/responses`` even when it bills reasoning tokens;
    the check below tells that case apart from a model that did not reason.

    Returns ``None`` (section skipped) when the catalog has no reasoning model,
    or the chosen backend reasons without returning the block.
    """
    print("\n🧠 Reasoning Output Blocks")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(require_reasoning=True, exclude_models=e2ee_models)
    except NoMatchingModelError as e:
        print(f"Section skipped: no reasoning model in the catalog ({e})")
        return None
    print(f"🤖 Using reasoning model: {model}")

    max_output_tokens = 2048
    response = await client.responses.create(
        model=model,
        input=(
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
            "the ball. How much does the ball cost? Answer in one sentence."
        ),
        # Reasoning tokens count against this cap, so leave plenty of room.
        max_output_tokens=max_output_tokens,
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    summarize_output_blocks(response)

    reasoning = [b for b in response.output if isinstance(b, ResponsesReasoningOutput)]
    for block in reasoning:
        for item in block.summary or []:
            # The block's ``summary`` field can hold the model's raw reasoning
            # text rather than a condensed summary, so it is labelled as
            # reasoning text and only its start is shown. Items are plain
            # strings or ``summary_text`` parts; a part of a type the SDK does
            # not know arrives as a dict and is skipped.
            if isinstance(item, ResponsesReasoningSummaryText):
                raw = item.text
            elif isinstance(item, str):
                raw = item
            else:
                continue
            shown = " ".join(raw.split())
            if len(shown) > 300:
                shown = f"{shown[:300]}… ({len(raw)} chars in total)"
            print(f"\n💭 Reasoning text (from the block's summary field): {shown}")

    text = extract_text(response)
    if text:
        print(f"\n🗣️ Assistant: {text}")
    print_usage(response)

    if not finished(response, text, max_output_tokens):
        return False
    details = response.usage.output_tokens_details if response.usage else None
    reasoning_tokens = (details.reasoning_tokens or 0) if details else 0
    if not reasoning and reasoning_tokens:
        # The SDK parsed what came back; this backend does not send the block.
        print(
            f"Section skipped: {model} billed {reasoning_tokens} reasoning tokens but "
            "/responses returned no reasoning block"
        )
        return None
    if not reasoning:
        print("\n❌ The reasoning model returned no reasoning block and no reasoning tokens")
        return False
    return True


async def truncated_response(client: VeniceClient, model: str) -> bool:
    """Hit ``max_output_tokens`` on purpose and read why the text stops.

    A response cut short by the cap comes back with ``status="incomplete"``
    and ``incomplete_details.reason == "max_output_tokens"``. The partial text
    is still available, but it must not be treated as a finished answer.
    """
    print("\n✂️ Truncation (max_output_tokens reached on purpose)")
    print("-" * 40)

    max_output_tokens = 16
    response = await client.responses.create(
        model=model,
        input="Write a detailed paragraph about the history of the Venetian Republic.",
        max_output_tokens=max_output_tokens,
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    reason = response.incomplete_details.reason if response.incomplete_details else None
    print(f"🆔 Response id: {response.id}")
    print(f"   Status: {response.status}")
    print(f"   Incomplete reason: {reason}")
    summarize_output_blocks(response)
    print(f"\n🗣️ Partial text: {extract_text(response)!r}")

    if response.status == "completed" and reached_cap(response, max_output_tokens):
        print(
            f"\n❌ {response.model} reported status='completed' after using all "
            f"{max_output_tokens} output tokens: the server did not flag the truncation"
        )
        return False
    if response.status != "incomplete" or reason != "max_output_tokens":
        print("\n❌ Expected status='incomplete' with reason 'max_output_tokens'")
        return False
    print("\n✅ Truncation detected: check status before using the text")
    return True


async def text_format_structured_output(
    client: VeniceClient, e2ee_models: list[str]
) -> bool | None:
    """Ask for JSON that follows a schema through ``text.format``.

    The JSON comes back as a string in the message text, so it is parsed here.
    A model without structured output support answers with a 400, which is why
    the catalog is filtered on it. Returns ``None`` when no model qualifies.
    """
    print("\n📐 Structured Output (text.format)")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(
            require_response_schema=True,
            exclude_reasoning=True,
            exclude_models=e2ee_models,
            prefer="cheapest",
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: no non-reasoning model supports structured output ({e})")
        return None
    print(f"🤖 Using model: {model}")

    max_output_tokens = 256
    response = await client.responses.create(
        model=model,
        input="Extract the order details: order 1842 ships 3 lanterns to Oslo.",
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "order",
                "schema": ORDER_SCHEMA,
                "strict": True,
            }
        },
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    text = extract_text(response)
    print(f"🗣️ Raw message text: {text}")
    if not finished(response, text, max_output_tokens):
        return False
    try:
        order = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n❌ The message text is not valid JSON: {e}")
        return False
    if order != EXPECTED_ORDER:
        print(f"\n❌ Expected {EXPECTED_ORDER}, got {order}")
        return False
    print("✅ Parsed JSON matches the schema and the order in the prompt")
    return True


async def instructions_parameter(client: VeniceClient, e2ee_models: list[str]) -> bool | None:
    """Pass system instructions through ``instructions`` instead of ``input``.

    Returns ``None`` when the catalog has no suitable model.
    """
    print("\n📜 Instructions")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(
            exclude_reasoning=True, exclude_models=e2ee_models, prefer="cheapest"
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: no non-reasoning chat model in the catalog ({e})")
        return None
    print(f"🤖 Using model: {model}")

    max_output_tokens = 64
    response = await client.responses.create(
        model=model,
        input="What is the capital of Japan?",
        instructions="Reply with only the city name, in uppercase letters.",
        max_output_tokens=max_output_tokens,
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    text = extract_text(response)
    print(f"🗣️ Assistant: {text}")
    if not finished(response, text, max_output_tokens):
        return False
    # Left to itself a model writes the name in mixed case, so the capitals show
    # the instructions were applied. Some models still wrap the word in a sentence.
    if "TOKYO" not in text or "Tokyo" in text:
        print("\n❌ Expected TOKYO in capitals, as the instructions ask")
        return False
    return True


async def streaming_response(client: VeniceClient, e2ee_models: list[str]) -> bool | None:
    """Stream a response: text deltas, then the full response on the last event.

    Events share one model and keep their payload as extra fields, so they are
    read from ``event.model_extra``. A stream ends with ``response.completed``,
    ``response.incomplete`` or ``response.failed``, whose ``response`` field
    holds the same object a non-streaming call returns, usage included.

    Returns ``None`` when the catalog has no suitable model.
    """
    print("\n🌊 Streaming")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(
            exclude_reasoning=True, exclude_models=e2ee_models, prefer="cheapest"
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: no non-reasoning chat model in the catalog ({e})")
        return None
    print(f"🤖 Using model: {model}")

    max_output_tokens = 128
    stream = await client.responses.create(
        model=model,
        input="Name the three primary colors of light in one sentence.",
        max_output_tokens=max_output_tokens,
        venice_parameters=NO_SYSTEM_PROMPT,
        stream=True,
    )

    deltas: list[str] = []
    event_count = 0
    last_type: str | None = None
    payload: object = None
    print("🗣️ Assistant: ", end="", flush=True)
    async for event in stream:
        event_count += 1
        extra = event.model_extra or {}
        if event.type == "response.output_text.delta":
            delta = extra.get("delta")
            if isinstance(delta, str):
                print(delta, end="", flush=True)
                deltas.append(delta)
        elif event.type in ("response.completed", "response.incomplete", "response.failed"):
            last_type = event.type
            payload = extra.get("response")
    print(f"\n   📦 {event_count} events, {len(deltas)} text deltas, ended with {last_type}")

    if last_type != "response.completed":
        print("\n❌ The stream did not end with response.completed")
        return False
    try:
        response = ResponsesResponse.model_validate(payload)
    except ValidationError as e:
        print(f"\n❌ response.completed carried no readable response: {e}")
        return False

    text = extract_text(response)
    print_usage(response)
    if not finished(response, text, max_output_tokens):
        return False
    if "".join(deltas).strip() != text:
        print("\n❌ The text deltas do not add up to the text of the final response")
        return False
    if not all(color in text.lower() for color in ("red", "green", "blue")):
        print("\n❌ Expected the answer to name red, green and blue")
        return False
    return True


async def function_tool_round_trip(client: VeniceClient, e2ee_models: list[str]) -> bool | None:
    """Call a function tool and send its result back for a final answer.

    The endpoint stores nothing, so the second request resends the whole
    conversation: the user's message, the output blocks of the first response
    as dicts, and one ``function_call_output`` per call, matched on ``call_id``.

    Returns ``None`` when the catalog has no suitable model.
    """
    print("\n🔧 Function Tool Round Trip")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(
            require_function_calling=True,
            exclude_reasoning=True,
            exclude_models=e2ee_models,
            prefer="cheapest",
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: no non-reasoning model supports function calling ({e})")
        return None
    print(f"🤖 Using model: {model}")

    max_output_tokens = 256
    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": "What is the locker code of the member named Ada?"}
    ]
    first = await client.responses.create(
        model=model,
        input=conversation,
        tools=[LOCKER_TOOL],
        tool_choice="auto",
        max_output_tokens=max_output_tokens,
        venice_parameters=NO_SYSTEM_PROMPT,
    )
    summarize_output_blocks(first)

    # A call cut off at the cap would carry truncated JSON arguments.
    if first.status != "completed" or reached_cap(first, max_output_tokens):
        print(f"\n❌ The tool-calling turn did not finish: status={first.status}")
        return False
    calls = [block for block in first.output if isinstance(block, ResponsesFunctionCallOutput)]
    if not calls:
        print(f"\n❌ The model answered without calling the tool: {extract_text(first)!r}")
        return False

    conversation.extend(block.model_dump(exclude_none=True) for block in first.output)
    for call in calls:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as e:
            print(f"\n❌ {call.name} arguments are not valid JSON: {e}")
            return False
        print(f"   🔧 {call.name}({arguments})  call_id={call.call_id}")
        if call.name != "lookup_locker_code" or not isinstance(arguments, dict):
            print("\n❌ Unexpected tool call")
            return False
        if "ada" not in str(arguments.get("member", "")).lower():
            print("\n❌ Expected the call to ask about Ada")
            return False
        conversation.append(
            {"type": "function_call_output", "call_id": call.call_id, "output": LOCKER_CODE}
        )

    final = await client.responses.create(
        model=model,
        input=conversation,
        tools=[LOCKER_TOOL],
        max_output_tokens=max_output_tokens,
        venice_parameters=NO_SYSTEM_PROMPT,
    )
    text = extract_text(final)
    print(f"🗣️ Assistant: {text}")
    if not finished(final, text, max_output_tokens):
        return False
    # Models may write the hyphen as a typographic one, such as U+2011.
    if LOCKER_CODE not in re.sub(r"[\u2010-\u2015\u2212]", "-", text):
        print(f"\n❌ Expected the answer to state the tool's result, {LOCKER_CODE}")
        return False
    return True


async def web_search_response(client: VeniceClient, e2ee_models: list[str]) -> bool | None:
    """Let the model search the web with ``web_search=True``.

    The search is billed on top of the tokens, and the results it brings in may
    add input tokens. Returns ``None`` when no model supports web search.
    """
    print("\n🌐 Web Search")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(
            require_web_search=True,
            exclude_reasoning=True,
            exclude_models=e2ee_models,
            prefer="cheapest",
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: no non-reasoning model supports web search ({e})")
        return None
    print(f"🤖 Using model: {model}")
    print("💸 This section is billed extra: about $0.01 per search request, on top of tokens")

    max_output_tokens = 512
    response = await client.responses.create(
        model=model,
        input="In one or two sentences, what is the latest stable release of Python?",
        web_search=True,
        max_output_tokens=max_output_tokens,
        # VeniceParameters defaults enable_web_search to "off" and
        # enable_web_citations to False, and sends both, so an instance passed
        # alongside web_search=True has to set them as well.
        venice_parameters=VeniceParameters(
            enable_web_search="on",
            enable_web_citations=True,
            include_venice_system_prompt=False,
        ),
    )
    summarize_output_blocks(response)

    text = extract_text(response)
    print(f"🗣️ Assistant: {text}")
    print_usage(response)

    searches = [
        block for block in response.output if isinstance(block, ResponsesWebSearchCallOutput)
    ]
    for search in searches:
        print(f"   🔍 web_search_call: status={search.status}, action={search.action}")
    annotations = [
        annotation
        for block in response.output
        if isinstance(block, ResponsesMessageOutput)
        for item in block.content
        if isinstance(item, ResponsesOutputText)
        for annotation in item.annotations or []
    ]
    print(f"   📎 {len(annotations)} annotation(s) on the answer")

    if not finished(response, text, max_output_tokens):
        return False
    # An answer alone cannot show that a search ran; a model can answer this
    # from memory.
    if not searches and not annotations:
        print("\n❌ No web_search_call block or annotation shows that a search ran")
        return False
    return True


async def list_e2ee_models(client: VeniceClient) -> list[str]:
    """Return the IDs of the catalog's end-to-end encrypted chat models."""
    catalog = await client.models.list(type="text")
    specs = ((model.id, model.model_spec) for model in catalog.data)
    return [
        model_id
        for model_id, spec in specs
        if isinstance(spec, TextModelSpec) and spec.capabilities and spec.capabilities.supportsE2EE
    ]


async def main() -> int:
    """Run all Responses API demos.

    Returns ``1`` if any demo failed, ``77`` when the basic call could not run
    (no chat model in the catalog, or no access to the endpoint), and ``0``
    otherwise. Other demos may be skipped individually; the basic call
    decides whether the example as a whole was verified.
    """
    print("🚀 Venice AI Responses API Example (beta)")
    print("=" * 50)

    results: list[tuple[str, bool]] = []
    skipped: list[str] = []
    async with VeniceClient() as client:
        # The default ranking (not prefer="cheapest") is used here: the
        # truncation demo needs a backend that reports status="incomplete" at
        # the cap, and not every one does (the demo detects that case). The
        # default ranking can select E2EE models, which /responses rejects, so
        # they are excluded by ID.
        e2ee_models = await list_e2ee_models(client)
        try:
            model = await client.models.resolve_chat(exclude_models=e2ee_models)
        except NoMatchingModelError as e:
            print(f"SKIPPED: no chat model in the catalog ({e})")
            return EXIT_SKIPPED
        print(f"🤖 Using model: {model}\n")

        sections = [
            ("basic_responses_create", lambda: basic_responses_create(client, model)),
            ("structured_input_responses", lambda: structured_input_responses(client, model)),
            ("reasoning_output_blocks", lambda: reasoning_output_blocks(client, e2ee_models)),
            ("truncated_response", lambda: truncated_response(client, model)),
            (
                "text_format_structured_output",
                lambda: text_format_structured_output(client, e2ee_models),
            ),
            ("instructions_parameter", lambda: instructions_parameter(client, e2ee_models)),
            ("streaming_response", lambda: streaming_response(client, e2ee_models)),
            ("function_tool_round_trip", lambda: function_tool_round_trip(client, e2ee_models)),
            ("web_search_response", lambda: web_search_response(client, e2ee_models)),
        ]
        for name, section in sections:
            try:
                outcome = await section()
            except PermissionDeniedError as e:
                # The Responses API is beta; a model or account may still
                # reject it. That is an access condition, not a failure.
                if name == "basic_responses_create":
                    # The other demos would be billed attempts at the same endpoint.
                    print(f"\nSKIPPED: this account has no access to the Responses API (beta): {e}")
                    return EXIT_SKIPPED
                print(f"\nSection skipped: {name} is not available on this account: {e}")
                skipped.append(name)
                continue
            except VeniceError as e:
                print(f"\n❌ {name} failed: {type(e).__name__}: {e}")
                results.append((name, False))
                continue
            if outcome is None:
                skipped.append(name)
            else:
                results.append((name, outcome))

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    if skipped:
        print(f"\n✅ {len(results)} demos passed, {len(skipped)} skipped: {', '.join(skipped)}")
        return 0

    print("\n✨ Responses API example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.responses.create with a plain-string input")
    print("   - client.responses.create with a structured message list")
    print("   - Reading typed output blocks (reasoning and message)")
    print("   - status / incomplete_details when max_output_tokens is reached")
    print("   - Inspecting the usage token block")
    print("   - text.format structured output, parsed and checked")
    print("   - instructions kept apart from input")
    print("   - stream=True: text deltas and the final response")
    print("   - A function tool round trip with function_call_output")
    print("   - web_search=True, read back from the output blocks")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and your account has access to the "
            "Responses API (beta).",
            file=sys.stderr,
        )
        sys.exit(1)
