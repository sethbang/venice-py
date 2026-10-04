#!/usr/bin/env python3
"""
Venice AI SDK - Function Calling Examples
=========================================

This example demonstrates function calling capabilities with Venice AI models
by driving the tool-call round trip by hand. (``agent_loop.py`` shows the
batteries-included ``run_with_tools`` helper that automates the same loop.)
It showcases:

- Basic function/tool definition and calling
- Processing function call results: executing the tool, returning a
  ``ToolMessage`` and getting a final answer grounded in the tool output
- Multiple tools with different schemas, called in parallel
- Tool choice control (none, specific function; auto is the default)
- Error handling for function calls: a tool that raises, reported back to the
  model so it can respond gracefully
"""

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Literal

from venice_ai import (
    AssistantMessage,
    NoMatchingModelError,
    SystemMessage,
    ToolChoice,
    ToolMessage,
    UserMessage,
    VeniceClient,
    VeniceError,
    tool_from_function,
)
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import VeniceParameters
from venice_ai.types.api.requests.common import Tool, ToolFunction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import states_number, states_reading  # noqa: E402

# The cheapest function-calling model is often a reasoning model, which spends
# part of the completion budget thinking before it emits a tool call or an
# answer. Small reasoning models vary widely in how long they think, so the cap
# leaves several thousand tokens of headroom over any answer here.
MAX_COMPLETION_TOKENS = 16384

# The tool definitions and the user message are all the context these requests
# need, so Venice's own system prompt is left out.
PARAMS = VeniceParameters(include_venice_system_prompt=False)

# Canned data so the example is deterministic: (temperature °C, conditions).
_WEATHER_TABLE = {
    "new york city": (22, "sunny"),
    "new york": (22, "sunny"),
}


def get_weather(
    location: str,
    unit: Literal["celsius", "fahrenheit"] = "fahrenheit",
) -> str:
    """Get the current weather for a specific location."""
    # Stand-in for a real weather API call. The docstring above is what the
    # model sees as the tool description.
    temp_c, sky = _WEATHER_TABLE.get(location.lower(), (18, "partly cloudy"))
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    symbol = "°C" if unit == "celsius" else "°F"
    return json.dumps(
        {"location": location, "temperature": f"{temp}{symbol}", "sky": sky},
        ensure_ascii=False,
    )


def _first_choice(response: ChatCompletionResponse) -> Any:
    if not response.choices:
        raise ValueError("the model returned no choices")
    return response.choices[0]


def _print_answer(label: str, text: str) -> None:
    print(f"   {label}:")
    print(textwrap.indent(text.strip() or "(empty)", "      "))


def _cut_off(response: ChatCompletionResponse) -> str | None:
    """Say why an answer may be truncated, or return ``None`` if it finished.

    Some models report an answer cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked against the cap as well.
    """
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    if finish_reason == "length" or used is None or used >= MAX_COMPLETION_TOKENS:
        return (
            f"finish_reason={finish_reason!r}, completion tokens {used} of {MAX_COMPLETION_TOKENS}"
        )
    return None


async def basic_function_calling(client: VeniceClient, model_id: str) -> bool:
    """Complete one full tool round trip with a weather tool.

    1. The model asks for ``get_weather``.
    2. We run the function and send its result back as a ``ToolMessage``.
    3. The model answers using that result.

    Returns ``True`` only if every step happened and the final answer uses the
    tool's result.
    """
    print("🔧 Basic Function Calling (full round trip)")
    print("-" * 40)

    # ``tool_from_function`` introspects type hints + docstring to build the
    # JSON schema the model sees.
    weather_tool = tool_from_function(get_weather)
    assert weather_tool.function is not None  # tool_from_function always sets .function

    print("🛠️ Tool Definition (built by tool_from_function):")
    print(f"   Function: {weather_tool.function.name}")
    print(f"   Description: {weather_tool.function.description}")

    messages: list[Any] = [
        UserMessage(content="What's the weather like in New York City? I prefer Celsius.")
    ]

    try:
        # Step 1: the model decides to call the tool.
        first = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=[weather_tool],
            tool_choice="auto",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            venice_parameters=PARAMS,
        )
        choice = _first_choice(first)
        print(f"\n1️⃣ Model turn finish reason: {choice.finish_reason}")
        tool_calls = choice.message.tool_calls or []
        if not tool_calls:
            print("❌ The model answered without calling get_weather:")
            _print_answer("Response", first.text or "")
            return False

        # Step 2: execute each requested call and collect ToolMessages. The
        # assistant turn that carries the tool_calls must go into the history
        # before the tool results that answer it.
        messages.append(AssistantMessage.from_response(first))
        tool_outputs: list[str] = []
        for call in tool_calls:
            args = call.function.arguments_dict
            print(f"   🔧 {call.function.name}({args})  id={call.id}")
            if call.function.name != "get_weather":
                print(f"❌ Unexpected tool requested: {call.function.name}")
                return False
            result = get_weather(**args)
            print(f"   📦 Tool result: {result}")
            tool_outputs.append(result)
            messages.append(ToolMessage(tool_call_id=call.id, content=result))

        # Step 3: send the tool results back so the model can answer.
        final = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=[weather_tool],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            venice_parameters=PARAMS,
        )
    except VeniceError as e:
        print(f"❌ Request failed ({type(e).__name__}): {e}")
        return False

    final_choice = _first_choice(final)
    answer = final.text or ""
    print(f"\n2️⃣ Final turn finish reason: {final_choice.finish_reason}")
    _print_answer("Final answer", answer)

    truncated = _cut_off(final)
    if final_choice.finish_reason != "stop" or truncated:
        print(f"❌ Expected a finished answer ({truncated or final_choice.finish_reason!r})")
        return False
    if not any(states_reading(answer, output) for output in tool_outputs):
        print(f"❌ The final answer does not state the reading the tool returned: {tool_outputs}")
        return False
    print("\n✅ Tool executed and the final answer is grounded in its result.")
    return True


async def multiple_tools_example(client: VeniceClient, model_id: str) -> bool:
    """Request two independent tools in one turn (parallel function calling).

    Returns ``True`` only if the model called both tools with valid arguments.
    """
    print("\n🛠️ Multiple Tools Example (parallel calls)")
    print("-" * 40)

    tools = [
        Tool(
            type="function",
            function=ToolFunction(
                name="calculate_area",
                description="Calculate the area of a rectangle",
                parameters={
                    "type": "object",
                    "properties": {
                        "width": {
                            "type": "number",
                            "description": "Width in meters",
                            "minimum": 0,
                        },
                        "height": {
                            "type": "number",
                            "description": "Height in meters",
                            "minimum": 0,
                        },
                    },
                    "required": ["width", "height"],
                },
                strict=True,
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="get_random_fact",
                description="Get a random interesting fact about a topic",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic to get a fact about",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["science", "history", "nature", "technology"],
                            "description": "Category of fact",
                        },
                    },
                    "required": ["topic"],
                },
                strict=False,
            ),
        ),
    ]

    print("📋 Available Tools:")
    for tool in tools:
        func = tool.function
        assert func is not None
        strict_status = "Strict" if func.strict else "Relaxed"
        print(f"   • {func.name} ({strict_status} validation)")

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                UserMessage(
                    content=(
                        "I need to calculate the area of a 5x3 meter rectangle, and also "
                        "tell me an interesting science fact about mathematics."
                    )
                )
            ],
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            venice_parameters=PARAMS,
        )
    except VeniceError as e:
        print(f"❌ Request failed ({type(e).__name__}): {e}")
        return False

    choice = _first_choice(response)
    tool_calls = choice.message.tool_calls or []
    print(f"\n🏁 Finish reason: {choice.finish_reason}")
    print(f"✉️ {len(tool_calls)} tool call(s) made:")
    for i, tool_call in enumerate(tool_calls, 1):
        print(f"   {i}. {tool_call.function.name}({tool_call.function.arguments_dict})")

    names = {call.function.name for call in tool_calls}
    missing = {"calculate_area", "get_random_fact"} - names
    if missing:
        print(f"❌ Expected both tools in one turn; missing: {', '.join(sorted(missing))}")
        if not tool_calls:
            _print_answer("Response", response.text or "")
        return False

    area_args = next(c for c in tool_calls if c.function.name == "calculate_area")
    dims = area_args.function.arguments_dict
    if sorted([dims.get("width"), dims.get("height")]) != [3, 5]:
        print(f"❌ calculate_area received the wrong dimensions: {dims}")
        return False

    fact_call = next(c for c in tool_calls if c.function.name == "get_random_fact")
    topic = fact_call.function.arguments_dict.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        print(f"❌ get_random_fact received no topic: {fact_call.function.arguments_dict}")
        return False

    print("\n✅ Both tools were requested in a single turn with the right arguments.")
    return True


async def tool_choice_control(client: VeniceClient, model_id: str) -> bool:
    """Show how ``tool_choice`` changes whether the model calls a tool.

    - ``"auto"`` (the default, used in the sections above): the model decides.
    - ``"none"``: the model must answer directly, without a tool call.
    - a named function: the model must call exactly that tool.

    Returns ``True`` only if "none" and the forced choice behaved as specified.
    """
    print("\n🎯 Tool Choice Control")
    print("-" * 40)

    calc_tool = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Perform basic mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Mathematical operation",
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["operation", "a", "b"],
            },
            strict=False,
        ),
    )

    # (label, tool_choice, expectation) where expectation is "tool" or "text".
    test_cases: list[tuple[str, Any, str]] = [
        ("No Tools — must answer directly", "none", "text"),
        (
            "Forced Tool — must call 'calculate'",
            ToolChoice.function("calculate"),
            "tool",
        ),
    ]

    ok = True
    for label, tool_choice, expectation in test_cases:
        print(f"\n🧪 {label}")
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[UserMessage(content="What is 15 multiplied by 8?")],
                tools=[calc_tool],
                tool_choice=tool_choice,
                # A reasoning model barred from its tools can deliberate until
                # it runs out of tokens; this one-step question needs no reasoning.
                reasoning_effort="none",
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                venice_parameters=PARAMS,
            )
        except VeniceError as e:
            print(f"   ❌ Request failed ({type(e).__name__}): {e}")
            ok = False
            continue

        choice = _first_choice(response)
        tool_calls = choice.message.tool_calls or []
        print(f"   🏁 Finish reason: {choice.finish_reason}")
        if tool_calls:
            call = tool_calls[0]
            print(f"   🔧 Tool called: {call.function.name}({call.function.arguments_dict})")
            outcome = "tool"
        else:
            content = (response.text or "").strip()
            shown = content if len(content) <= 100 else content[:100] + "..."
            print(f"   💬 Direct response: {shown}")
            outcome = "text"
            truncated = _cut_off(response)
            if truncated:
                print(f"   ❌ The direct answer may be cut off ({truncated})")
                ok = False
                continue
            if not states_number(content, 15 * 8):
                print("   ❌ The direct answer does not state the correct product 120")
                ok = False
                continue

        if outcome != expectation:
            print(f"   ❌ Expected a {expectation} response, got a {outcome} response")
            ok = False
        elif expectation == "tool" and tool_calls[0].function.name != "calculate":
            print("   ❌ The forced tool was not the one called")
            ok = False
        else:
            print("   ✅ Behaved as specified")

    return ok


class ToolExecutionError(Exception):
    """Raised by a tool implementation when it cannot fulfil a request."""


def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> float:
    """Evaluate one arithmetic operation, raising on invalid input."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ToolExecutionError("division by zero is undefined")
        return a / b
    raise ToolExecutionError(f"unsupported operation {operation!r}")


async def tool_error_handling(client: VeniceClient, model_id: str) -> bool:
    """Report a failing tool back to the model instead of crashing.

    We force a call to ``calculate`` with a question whose arguments make the
    tool raise (division by zero). The error is caught, sent back as the
    ``ToolMessage`` content, and the model explains the problem to the user.

    Returns ``True`` only if the tool really raised and the model then produced
    a complete answer.
    """
    print("\n🧯 Error Handling for Function Calls")
    print("-" * 40)

    calc_tool = tool_from_function(calculate)
    messages: list[Any] = [
        SystemMessage(content="Reply in plain text. Do not use LaTeX or Markdown math."),
        UserMessage(content="Use the calculator: what is 42 divided by 0?"),
    ]

    try:
        first = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=[calc_tool],
            tool_choice=ToolChoice.function("calculate"),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            venice_parameters=PARAMS,
        )
        tool_calls = _first_choice(first).message.tool_calls or []
        if not tool_calls:
            print("❌ The forced calculate call was not made")
            return False

        messages.append(AssistantMessage.from_response(first))
        tool_errors = 0
        argument_errors = 0
        for call in tool_calls:
            args = call.function.arguments_dict
            print(f"   🔧 {call.function.name}({args})")
            try:
                content = str(calculate(**args))
            except ToolExecutionError as e:
                # Tell the model what went wrong so it can respond sensibly.
                tool_errors += 1
                content = f"Error: {e}"
            except TypeError as e:
                # Arguments that don't match the function signature are also
                # reported back, but they are not the failure this section shows.
                argument_errors += 1
                content = f"Error: invalid arguments ({e})"
            print(f"   📦 Tool result sent back: {content}")
            messages.append(ToolMessage(tool_call_id=call.id, content=content))

        if tool_errors == 0:
            if argument_errors:
                print("❌ The model sent arguments calculate() does not accept, so the")
                print("   tool never ran and the ToolExecutionError path was not exercised")
            else:
                print("❌ The tool did not raise, so the error path was not exercised")
            return False

        final = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=[calc_tool],
            tool_choice="none",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            venice_parameters=PARAMS,
        )
    except VeniceError as e:
        print(f"❌ Request failed ({type(e).__name__}): {e}")
        return False

    final_choice = _first_choice(final)
    answer = final.text or ""
    print(f"\n🏁 Finish reason: {final_choice.finish_reason}")
    _print_answer("Model's reply after the tool error", answer)
    truncated = _cut_off(final)
    if final_choice.finish_reason != "stop" or truncated or not answer.strip():
        print("❌ The model did not produce a complete reply after the tool error")
        if truncated:
            print(f"   ({truncated})")
        return False

    print("\n✅ The tool error was reported to the model and handled without crashing.")
    return True


def function_calling_best_practices() -> None:
    """Print best practices for function calling (informational only)."""
    print("\n📚 Function Calling Best Practices")
    print("-" * 40)

    print("💡 Design Guidelines:")
    print("   ✅ Use descriptive function names and descriptions")
    print("   ✅ Define clear parameter schemas with types and constraints")
    print("   ✅ Include helpful descriptions for all parameters")
    print("   ✅ Mark required parameters appropriately")
    print("   ✅ Use enums for limited value sets")
    print("   ✅ Set appropriate validation (strict vs relaxed)")

    print("\n⚡ Performance Tips:")
    print("   ✅ Use parallel_tool_calls=True for independent operations")
    print("   ✅ Keep function descriptions concise but informative")
    print("   ✅ Limit the number of tools to avoid confusion")
    print("   ✅ Use tool_choice strategically to guide model behavior")

    print("\n🔧 Error Handling:")
    print("   ✅ Always validate function call arguments")
    print("   ✅ Return tool failures to the model as the ToolMessage content")
    print("   ✅ Append the assistant tool_calls turn before its ToolMessages")
    print("   ✅ Trust an answer only if finish_reason is 'stop' AND its completion")
    print("      tokens stayed under the cap: some models report a cut-off answer as 'stop'")


async def main() -> int:
    """Run all function calling examples.

    Returns ``0`` only if every demo succeeded, ``77`` if the catalog has no
    suitable model, and ``1`` otherwise, so a real API failure or a
    behavioural mismatch surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI Function Calling Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        # tool_choice_control sends reasoning_effort="none". Accepted values
        # differ per model and some models reject the field with HTTP 400, so
        # require_reasoning_effort="none" picks a model whose catalog entry
        # lists "none" as an option.
        try:
            model_id = await client.models.resolve_chat(
                require_function_calling=True, require_reasoning_effort="none", prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no function-calling model accepts reasoning_effort='none' ({e})")
            return 77
        print(f"🤖 Using model: {model_id}\n")

        results: list[tuple[str, bool]] = [
            ("basic_function_calling", await basic_function_calling(client, model_id)),
            ("multiple_tools_example", await multiple_tools_example(client, model_id)),
            ("tool_choice_control", await tool_choice_control(client, model_id)),
            ("tool_error_handling", await tool_error_handling(client, model_id)),
        ]

    function_calling_best_practices()

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    print("\n✨ Function calling examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Tool definitions via tool_from_function and explicit Tool schemas")
    print("   - The full round trip: tool_calls → ToolMessage → final answer")
    print("   - Parallel function calling")
    print("   - Tool choice control strategies")
    print("   - Reporting tool errors back to the model")
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
            "Check that your API key is valid and you have a model that supports function calling.",
            file=sys.stderr,
        )
        sys.exit(1)
