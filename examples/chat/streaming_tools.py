#!/usr/bin/env python3
"""
Venice AI SDK - Streaming With Tools
====================================

Passing ``tools=`` to ``client.chat.completions.stream()`` works as it does on
a non-streaming call, but a tool call does not arrive as text. Its name and
arguments come in fragments on ``delta.tool_calls``, which the stream's
collectors put back together.

This example demonstrates:

- A stream that passes ``tools=``, with any text shown live
- Reading the assembled calls from
  ``stream.final_response.choices[0].message.tool_calls``
- Completing the round trip: ``AssistantMessage.from_response()``, a
  ``ToolMessage`` for each call, then a second stream for the answer
- Checking that the answer states the value the tool returned

``text_deltas()`` yields text only, so it cannot be used here: a tool-calling
turn often yields nothing from it. Use ``collect_with_deltas()``, or ``collect()``
when nothing is shown live.

Cost: two small requests on the cheapest function-calling model that does not
reason, typically well under $0.001 per run.

Exits 0 if the round trip passed its checks, 1 on any failure, and 77 (with a
``SKIPPED:`` line) if the catalog lists no such model.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import json
import sys
from typing import Any

from _helpers import states_reading

from venice_ai import (
    AssistantMessage,
    NoMatchingModelError,
    ToolChoice,
    ToolMessage,
    UserMessage,
    VeniceClient,
    VeniceError,
    tool_from_function,
)
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import VeniceParameters

# Roomy for a one-sentence answer or a single tool call, from a model that
# spends nothing on reasoning.
MAX_COMPLETION_TOKENS = 256

# The tool definition and the user message are all the context these requests
# need, so Venice's own system prompt is left out.
PARAMS = VeniceParameters(include_venice_system_prompt=False)

EXIT_SKIPPED = 77


def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    # Stand-in for a real weather API call. The docstring above is what the
    # model sees as the tool description.
    return f"{location}: 17°C, light rain"


WEATHER_TOOL = tool_from_function(get_weather)


async def stream_turn(
    client: VeniceClient, model: str, messages: list[Any], tool_choice: Any = "auto"
) -> ChatCompletionResponse | None:
    """Stream one turn, showing its text live, and return the assembled response.

    Returns ``None`` if the turn did not come back whole. Prints why.
    """
    stream = await client.chat.completions.stream(
        model=model,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice=tool_choice,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        venice_parameters=PARAMS,
    )
    try:
        async with stream:
            # Only text is yielded. Tool-call fragments are collected on the
            # way through and appear on final_response.
            async for text in stream.collect_with_deltas():
                print(text, end="", flush=True)
    except ValueError as e:
        print(f"\n❌ The stream ended early: {e}")
        return None
    print()

    response = stream.final_response
    if response is None or not response.choices:
        print("❌ The stream ended without a final response")
        return None
    finish_reason = response.choices[0].finish_reason
    used = response.usage.completion_tokens if response.usage else None
    print(f"   ↳ finish_reason: {finish_reason} ({used} of {MAX_COMPLETION_TOKENS} tokens)")
    # Some models report a cut-off turn as "stop", so the count is checked too.
    # A cut-off tool call would carry truncated JSON arguments.
    if finish_reason == "length" or used is None or used >= MAX_COMPLETION_TOKENS:
        print("❌ The turn may be cut off at max_completion_tokens")
        return None
    return response


async def streaming_tool_round_trip(client: VeniceClient, model: str) -> bool:
    """Stream a tool call, run the tool, and stream the answer that uses its result."""
    print("🌊 Streaming Tool Round Trip")
    print("-" * 40)

    messages: list[Any] = [UserMessage(content="What's the weather like in Tokyo right now?")]

    print("1️⃣ Model turn (text, if any): ", end="", flush=True)
    # Pinned because a small model on "auto" may only say it will use the tool.
    first = await stream_turn(client, model, messages, ToolChoice.function("get_weather"))
    if first is None:
        return False

    # This is where streamed tool calls end up: nothing above was printed for
    # them, because they are not text.
    tool_calls = first.choices[0].message.tool_calls or []
    if not tool_calls:
        print(f"❌ The model answered without calling get_weather: {first.text!r}")
        return False

    # The assistant turn that carries the tool_calls goes into the history
    # before the tool results that answer it.
    messages.append(AssistantMessage.from_response(first))
    tool_outputs: list[str] = []
    for call in tool_calls:
        if call.function is None:
            print(f"❌ Unexpected {call.type!r} tool call: {call.model_extra}")
            return False
        try:
            args = call.function.arguments_dict
        except (json.JSONDecodeError, TypeError) as e:
            print(f"❌ The assembled arguments are not a JSON object: {e}")
            return False
        print(f"   🔧 {call.function.name}({args})  id={call.id}")
        location = args.get("location")
        if call.function.name != "get_weather" or not isinstance(location, str):
            print("❌ Expected a get_weather call with a location")
            return False
        if "tokyo" not in location.lower():
            print(f"❌ Expected the call to ask about Tokyo, got {location!r}")
            return False
        result = get_weather(location)
        print(f"   📦 Tool result: {result}")
        tool_outputs.append(result)
        messages.append(ToolMessage(tool_call_id=call.id, content=result))

    print("\n2️⃣ Final answer: ", end="", flush=True)
    final = await stream_turn(client, model, messages)
    if final is None:
        return False

    if final.choices[0].message.tool_calls:
        print("❌ The model called the tool again instead of answering")
        return False
    answer = final.text or ""
    if not answer.strip():
        print("❌ The second turn produced no answer")
        return False
    if not any(states_reading(answer, output) for output in tool_outputs):
        print(f"❌ Expected the answer to state the tool's reading: {tool_outputs}")
        return False
    print("✅ The answer states the temperature the tool returned")
    return True


async def main() -> int:
    """Run the example; return a process exit code."""
    print("🚀 Venice AI Streaming With Tools")
    print("=" * 50)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_chat(
                require_function_calling=True, exclude_reasoning=True, prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no non-reasoning function-calling model in the catalog ({e})")
            return EXIT_SKIPPED
        print(f"🤖 Using model: {model}\n")

        try:
            ok = await streaming_tool_round_trip(client, model)
        except VeniceError as e:
            print(f"\n❌ {type(e).__name__}: {e}")
            ok = False

    if not ok:
        print("\n❌ The streaming tool round trip failed")
        return 1

    print("\n✨ Streaming with tools completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - tools= on client.chat.completions.stream()")
    print("   - final_response.choices[0].message.tool_calls holds the assembled calls")
    print("   - The full round trip: tool_calls → ToolMessage → streamed answer")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Stream interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
