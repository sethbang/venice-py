#!/usr/bin/env python3
"""
Venice AI SDK - Agent Loop with run_with_tools
==============================================

This example demonstrates the agent-loop convenience method
``client.chat.completions.run_with_tools(...)``, which drives the canonical
"call model -> execute requested tools -> feed results back -> repeat until a
final answer" loop for you. Compared with ``tool_calling.py`` (which builds
tool definitions and dispatches calls by hand), this is the batteries-included
path: hand it plain Python functions and it does the orchestration.

It showcases:

- Defining simple Python tools (a calculator and a mock weather lookup) and
  letting the SDK introspect them via ``tool_from_function``
- Running a multi-step agent loop where the model actually calls the tools and
  the SDK executes them, feeding results back automatically
- Observing each successful tool call as it happens with the ``on_tool_call``
  hook (a call that raises goes to ``on_tool_error`` instead)
- Inspecting the ``ToolLoopResult`` (final text, round-trip count, token usage,
  full message history)
- ``on_tool_error``, ``parallel=True``, extra ``create()`` arguments and
  ``Conversation.run_with_tools``

Cost: two tool loops of two or three short requests each on the cheapest model
with function calling, typically under $0.002 in total.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys
import textwrap
from typing import Any, Literal

from _helpers import completion_cap, states_number, states_reading

from venice_ai import (
    Conversation,
    MaxIterationsExceededError,
    NoMatchingModelError,
    ToolMessage,
    UserMessage,
    VeniceClient,
    VeniceError,
    tool_from_function,
)
from venice_ai.types.api.chat import ToolCall
from venice_ai.types.api.requests import VeniceParameters

# ---------------------------------------------------------------------------
# Tool implementations
#
# These are plain Python functions. ``run_with_tools`` introspects each one
# with ``tool_from_function`` (type hints + docstring -> JSON schema) and also
# registers the function body as the dispatch handler, so the SDK both
# advertises the tool to the model AND runs it when the model calls it. The
# return value is stringified and sent back to the model as a tool result.
# ---------------------------------------------------------------------------


def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> float:
    """Perform a basic arithmetic operation on two numbers."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("division by zero")
        return a / b
    raise ValueError(f"unknown operation: {operation!r}")


def get_weather(
    city: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    """Look up the current weather for a city."""
    # The docstring above becomes the tool description the model sees, so keep
    # it user-facing. This body is a stand-in for a real weather API call and
    # returns canned data so the example is deterministic.
    table = {
        "san francisco": (18, "foggy"),
        "tokyo": (24, "clear"),
        "london": (12, "rainy"),
    }
    temp_c, sky = table.get(city.lower(), (21, "partly cloudy"))
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    symbol = "°C" if unit == "celsius" else "°F"
    return f"{city.title()}: {temp}{symbol}, {sky}"


async def agent_loop_demo(client: VeniceClient, model_id: str, cap: int) -> bool:
    """Run a multi-step agent loop that exercises both tools.

    Returns ``True`` only when the model called both tools, the loop finished
    with a complete answer, and that answer uses the tool results. Returns
    ``False`` on an API error, a loop that never converged, a truncated answer,
    or an answer that ignored the tools. ``cap`` is the per-round completion
    budget: the cheapest function-calling model is often a reasoning model, which
    spends part of it thinking before it emits a tool call or the answer.
    """
    print("\n🤖 Agent Loop (run_with_tools)")
    print("-" * 40)
    print(f"🤖 Using model: {model_id}")

    # Observation hook: called after each tool runs successfully with the
    # ToolCall the model emitted and the handler's return value. It is
    # read-only and does not change what the model sees.
    tool_calls_seen: list[tuple[str, dict[str, Any], Any]] = []

    def on_tool_call(call: ToolCall, result: Any) -> None:
        # run_with_tools only dispatches function calls, so ``function`` is set.
        if call.function is None:
            return
        args = call.function.arguments_dict
        tool_calls_seen.append((call.function.name, args, result))
        print(f"   🔧 {call.function.name}({args}) -> {result!r}")

    print("\n💬 Question: a multi-step task that needs both tools.")
    print("   (compute 137 * 4, then get the weather in Tokyo)\n")

    try:
        # The SDK runs the full loop: it calls the model, executes any tools
        # the model requests, feeds the results back, and repeats until the
        # model produces a final (non-tool-call) answer. Extra keyword
        # arguments are passed to every create() call in the loop.
        result = await client.chat.completions.run_with_tools(
            model=model_id,
            messages=[
                UserMessage(
                    content=(
                        "First, what is 137 multiplied by 4? "
                        "Then tell me the current weather in Tokyo in celsius. "
                        "Use the tools provided, then summarize both answers "
                        "in plain text (no LaTeX or Markdown)."
                    )
                )
            ],
            # Bare callables — auto-converted with tool_from_function and
            # registered as their own dispatch handlers.
            tools=[calculator, get_weather],
            on_tool_call=on_tool_call,
            max_iterations=6,
            max_completion_tokens=cap,
            # The tools and the user message are all the context this task
            # needs, so Venice's own system prompt is left out.
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )
    except MaxIterationsExceededError as e:
        # The loop ran but never converged — surface it honestly.
        print(f"❌ Tool loop did not converge: {e}")
        return False
    except VeniceError as e:
        print(f"❌ Agent loop request failed ({type(e).__name__}): {e}")
        return False

    answer = result.text or ""
    print(f"\n🔁 Loop converged in {result.iterations} round-trip(s).")
    print(f"   Tool calls that succeeded: {len(tool_calls_seen)}")
    print(f"   Finish reason: {result.finish_reason}")
    print("\n📝 Final answer:")
    print(textwrap.indent(answer.strip(), "   "))

    if result.usage:
        print(
            f"\n📊 Token usage (final round): prompt={result.usage.prompt_tokens}, "
            f"completion={result.usage.completion_tokens}, "
            f"total={result.usage.total_tokens}"
        )

    # Verify the loop actually did what it claims: both tools ran with the
    # right inputs, the final answer is complete, and it states what the tools
    # returned.
    problems: list[str] = []
    product = next(
        (
            result
            for name, _args, result in tool_calls_seen
            if name == "calculator" and result == 137 * 4
        ),
        None,
    )
    tokyo = next(
        (
            str(result)
            for name, args, result in tool_calls_seen
            if name == "get_weather" and str(args.get("city", "")).lower() == "tokyo"
        ),
        None,
    )
    if product is None:
        problems.append("no calculator call returned 137 * 4")
    if tokyo is None:
        problems.append("the model never called get_weather for Tokyo")
    # Some models report an answer cut off at the cap as finish_reason="stop",
    # so the final round's completion count is checked against the cap too.
    used = result.usage.completion_tokens if result.usage else None
    if result.finish_reason == "length" or used is None or used >= cap:
        problems.append(
            "the final answer may be cut off: "
            f"finish_reason={result.finish_reason}, completion tokens {used} "
            f"of {cap}"
        )
    if not answer.strip():
        problems.append("the final answer is empty")
    if product is not None and not states_number(answer, product):
        problems.append(f"the final answer does not state the calculator result {product:g}")
    if tokyo is not None and not states_reading(answer, tokyo):
        problems.append(
            f"the final answer does not state the reading get_weather returned ({tokyo})"
        )

    if problems:
        for problem in problems:
            print(f"❌ {problem}")
        return False

    print("\n✅ Both tools ran and the final answer uses their results.")
    return True


# ---------------------------------------------------------------------------
# Loop options: a tool that fails, and tools that can run at the same time
# ---------------------------------------------------------------------------

# Plain digits: models often rewrite a hyphen as U+2011, which would miss the table.
ORDER_STATUS = {"1001": "shipped"}

# Tells apart what on_tool_error returned from the SDK's default error text.
NOT_FOUND = "No such order on file."


class OrderLookup:
    """An async tool that records how many of its calls overlapped."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def lookup_order(self, order_id: str) -> str:
        """Look up the status of one order by its ID."""
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(0.05)  # stands in for a real network call
            if order_id not in ORDER_STATUS:
                raise LookupError(f"unknown order {order_id}")
            return ORDER_STATUS[order_id]
        finally:
            self.running -= 1


async def loop_options_demo(client: VeniceClient, model_id: str, cap: int) -> bool:
    """Run ``Conversation.run_with_tools`` with a tool that fails.

    - ``on_tool_call`` sees only calls that succeeded; a call that raises goes to
      ``on_tool_error``, and the string it returns is what the model reads.
    - ``parallel=True`` runs the calls of one assistant turn concurrently. Only
      use it when the handlers are safe to run together. A model may still ask
      for one call per turn, so the overlap is checked only if it asked for two.
    - Every other keyword argument goes to ``create()`` on each round.
    - ``Conversation.run_with_tools`` appends what the loop produced to the
      conversation, which is then ready for the next user turn.
    """
    print("\n🧰 Loop Options (on_tool_error, parallel, Conversation)")
    print("-" * 40)

    succeeded: list[str] = []
    failed: list[str] = []
    orders = OrderLookup()

    def on_tool_call(call: ToolCall, result: Any) -> None:
        if call.function is not None:
            succeeded.append(str(call.function.arguments_dict.get("order_id", "")))
            print(f"   🔧 {call.function.name}({call.function.arguments}) -> {result!r}")

    def on_tool_error(call: ToolCall, error: Exception) -> str:
        if call.function is not None:
            failed.append(str(call.function.arguments_dict.get("order_id", "")))
        print(f"   ⚠️  Tool raised {type(error).__name__}: {error}")
        return NOT_FOUND

    conversation = Conversation()
    conversation.add_user(
        "Call lookup_order for order 1001 and for order 2002, in the same turn if you can. "
        "Then state each order's status in plain text. If a lookup fails, say so."
    )
    before = len(conversation.messages)

    try:
        result = await conversation.run_with_tools(
            client,
            model=model_id,
            tools=[orders.lookup_order],
            on_tool_call=on_tool_call,
            on_tool_error=on_tool_error,
            parallel=True,
            max_iterations=6,
            max_completion_tokens=cap,
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )
    except MaxIterationsExceededError as e:
        print(f"❌ Tool loop did not converge: {e}")
        return False
    except VeniceError as e:
        print(f"❌ Agent loop request failed ({type(e).__name__}): {e}")
        return False

    answer = result.text or ""
    print("\n📝 Final answer:")
    print(textwrap.indent(answer.strip(), "   "))

    tool_turns = [m.tool_calls for m in result.messages if getattr(m, "tool_calls", None)]
    most_in_one_turn = max((len(calls) for calls in tool_turns), default=0)
    tool_results = [m.content for m in result.messages if isinstance(m, ToolMessage)]
    grown = len(conversation.messages) - before
    print(
        f"\n📜 result.messages: {len(result.messages)} messages, {result.iterations} round-trip(s)"
    )
    print(
        f"   Most tool calls in one turn: {most_in_one_turn}; most running at once: {orders.peak}"
    )
    print(f"   The conversation grew by {grown} messages")

    problems: list[str] = []
    if "1001" not in succeeded or "2002" in succeeded:
        problems.append(f"on_tool_call should see 1001 only, saw {succeeded}")
    # A model may try the failed lookup again, so this allows repeats.
    if set(failed) != {"2002"}:
        problems.append(f"on_tool_error should see 2002 only, saw {failed}")
    if NOT_FOUND not in tool_results:
        problems.append("the string on_tool_error returned is not in result.messages")
    if most_in_one_turn >= 2 and orders.peak < 2:
        problems.append("two calls in one turn did not overlap despite parallel=True")
    if grown != len(result.messages) - before:
        problems.append("the conversation did not gain the messages the loop produced")
    used = result.usage.completion_tokens if result.usage else None
    if result.finish_reason == "length" or used is None or used >= cap:
        problems.append(f"the final answer may be cut off ({used} of {cap} tokens)")
    if "shipped" not in answer.lower():
        problems.append("the final answer does not state order 1001's status (shipped)")

    if problems:
        for problem in problems:
            print(f"❌ {problem}")
        return False
    print("\n✅ The failure reached on_tool_error and the model carried on from its message.")
    return True


async def inspect_tool_schema_demo() -> bool:
    """Show and check what ``tool_from_function`` builds from a plain function.

    Pure introspection, with no network call. It makes the work behind
    ``run_with_tools`` visible: the same conversion happens automatically when
    you pass bare callables to the loop. Returns ``False`` if the schema does
    not match the function's signature.
    """
    print("\n🔍 Inspecting tool_from_function output")
    print("-" * 40)

    weather_tool = tool_from_function(get_weather)
    assert weather_tool.function is not None  # tool_from_function always sets .function

    print(f"   Function name: {weather_tool.function.name}")
    print(f"   Description:   {weather_tool.function.description}")
    print(f"   Parameters:    {weather_tool.function.parameters}")

    parameters = weather_tool.function.parameters or {}
    unit = parameters.get("properties", {}).get("unit", {})
    if parameters.get("required") != ["city"] or unit.get("enum") != ["celsius", "fahrenheit"]:
        print("   ❌ The schema does not match get_weather's signature")
        return False
    print("   ✅ city is required; unit is optional and limited to its Literal values")
    return True


async def main() -> int:
    """Run all agent-loop demos.

    Returns ``0`` only if every demo succeeded, ``77`` if the catalog has no
    function-calling model, and ``1`` otherwise, so a real API failure
    surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Agent Loop Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("inspect_tool_schema_demo", await inspect_tool_schema_demo()),
    ]
    async with VeniceClient() as client:
        try:
            # The loop needs a model that supports function calling; the
            # cheapest one is enough.
            model_id = await client.models.resolve_chat(
                require_function_calling=True, prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"\nSKIPPED: no function-calling chat model in the catalog ({e})")
            return 77
        cap = await completion_cap(client, model_id)
        results.append(("agent_loop_demo", await agent_loop_demo(client, model_id, cap)))
        results.append(("loop_options_demo", await loop_options_demo(client, model_id, cap)))

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    print("\n✨ Agent loop examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - run_with_tools drives the full tool-call loop for you")
    print("   - Plain Python functions become tools via tool_from_function")
    print("   - on_tool_call observes each successful call; on_tool_error handles failures")
    print("   - parallel=True runs one turn's calls concurrently")
    print("   - Conversation.run_with_tools keeps the history for the next turn")
    print("   - ToolLoopResult carries the final text, iterations, and usage")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
