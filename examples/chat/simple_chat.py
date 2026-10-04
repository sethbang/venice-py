#!/usr/bin/env python3
"""
Venice AI SDK - Simple Chat Completions
=======================================

This example demonstrates basic chat completion functionality with the Venice AI SDK.
Learn how to create simple conversational AI interactions.

Every answer is checked for truncation: a response that stopped because it
hit ``max_completion_tokens`` is cut off and counts as a failed section, so the
script exits non-zero instead of printing a truncated fragment as if it were
the model's answer. ``finish_reason == "length"`` is the usual signal, but some
models report a cut-off answer as ``"stop"``, so the completion-token count is
compared with the cap as well.
"""

import asyncio
import sys
import textwrap

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# Room for a reasoning model to think and still answer. Small reasoning models
# vary widely in how long they think, even on easy questions, so the cap leaves
# several thousand tokens of headroom over a typical run.
REASONING_CAP = 16384

# After the first section, requests leave Venice's own system prompt out: these
# prompts need no instructions beyond our own.
NO_VENICE_PROMPT = VeniceParameters(include_venice_system_prompt=False)


def report_answer(response: ChatCompletionResponse, cap: int, indent: str = "") -> bool:
    """Print the answer and its ``finish_reason``; return ``False`` if unusable.

    An empty answer, or one cut off at the token cap (``cap`` is the request's
    ``max_completion_tokens``), is not a real answer.
    """
    text = (response.text or "").strip()
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    print(textwrap.indent(text or "(empty response)", indent))
    print(f"{indent}↳ finish_reason: {finish_reason} ({used} of {cap} completion tokens)")
    if finish_reason == "length" or used is None or used >= cap:
        print(f"{indent}❌ Answer may be cut off at max_completion_tokens")
        return False
    if not text:
        print(f"{indent}❌ Model returned no visible answer")
        return False
    return True


async def basic_chat_completion(client: VeniceClient, chat_model: str) -> bool:
    """Create a simple chat completion."""
    print("💬 Basic Chat Completion")
    print("-" * 30)
    print(f"📍 Using model: {chat_model}")

    response = await client.chat.completions.create(
        model=chat_model,
        messages=[
            UserMessage(content="Explain quantum computing in simple terms, in 3 or 4 sentences.")
        ],
        max_completion_tokens=400,
        temperature=0.7,
    )

    print("🤖 Assistant:")
    ok = report_answer(response, 400, indent="   ")

    # Show usage information
    if response.usage:
        usage = response.usage
        print("\n📊 Token Usage:")
        print(f"   Input tokens: {usage.prompt_tokens} (cached: {usage.cached_tokens})")
        print(f"   Output tokens: {usage.completion_tokens}")
        print(f"   Total tokens: {usage.total_tokens}")
        print(
            "   ℹ️ Input includes Venice's default system prompt, which the server adds to\n"
            "      every request unless told not to (some models serve it from the cache).\n"
            "      The next section sends VeniceParameters(include_venice_system_prompt=False)\n"
            "      to leave it out."
        )
    return ok


async def chat_with_system_message(client: VeniceClient, chat_model: str) -> bool:
    """Create a chat completion with a system message."""
    print("\n🎭 Chat with System Message")
    print("-" * 30)
    print(f"📍 Using model: {chat_model}")

    # Chat with system message to set personality
    response = await client.chat.completions.create(
        model=chat_model,
        messages=[
            SystemMessage(
                content=(
                    "You are a helpful assistant that explains things like you're talking "
                    "to a 10-year-old. Use simple words and one fun analogy. "
                    "Answer in at most 4 sentences."
                )
            ),
            UserMessage(content="What is gravity?"),
        ],
        # Our system message sets the persona, so Venice's own system prompt is
        # left out; otherwise both would apply.
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=300,
        temperature=0.8,
    )

    print("🤖 Kid-friendly Assistant:")
    ok = report_answer(response, 300, indent="   ")
    if response.usage:
        print(f"   📊 Input tokens: {response.usage.prompt_tokens}")
    return ok


async def different_models_comparison(client: VeniceClient, standard_model: str) -> bool | None:
    """Compare a standard chat model with a reasoning model on the same question.

    Reasoning models spend part of ``max_completion_tokens`` on hidden
    reasoning before they write the visible answer, so they need a larger cap
    than a standard model for the same length of answer. Returns ``None``
    (section skipped) when the catalog lists no reasoning model.
    """
    print("\n🔬 Different Models Comparison")
    print("-" * 30)

    question = "In two sentences: what's the meaning of life?"

    try:
        reasoning_model = await client.models.resolve_chat(
            require_reasoning=True, prefer="cheapest"
        )
    except NoMatchingModelError:
        print("Section skipped: the catalog lists no reasoning model")
        return None

    # (model, max_completion_tokens) pairs; the reasoning model gets room to think.
    candidates = [(standard_model, 300), (reasoning_model, REASONING_CAP)]
    print(f"📍 Comparing models: {[model for model, _ in candidates]}")

    ok = True
    for model, cap in candidates:
        print(f"\n🤖 {model} (max_completion_tokens={cap}) says:")
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content=question)],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=cap,
            temperature=0.7,
        )
        ok = report_answer(response, cap, indent="   ") and ok
        if response.usage:
            details = response.usage.completion_tokens_details
            reasoning_tokens = details.reasoning_tokens if details else None
            print(
                f"   📊 completion_tokens={response.usage.completion_tokens}"
                f" (reasoning_tokens={reasoning_tokens})"
            )
    return ok


async def parameter_variations(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate different temperature settings on the same prompt."""
    print("\n⚙️ Parameter Variations")
    print("-" * 30)
    print(f"📍 Using model: {chat_model}")

    prompt = "Write a haiku about programming. Reply with the haiku only."

    # Lower values make word choice more focused and repeatable, higher values
    # more varied. How high a model stays coherent differs between models; the
    # check below only confirms a complete reply, not its quality.
    temperatures = [0.2, 1.0]

    ok = True
    for temp in temperatures:
        print(f"\n🌡️ Temperature {temp}:")

        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content=prompt)],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=100,
            temperature=temp,
        )

        ok = report_answer(response, 100, indent="   ") and ok
    return ok


async def main() -> int:
    """Run all chat completion examples; return a process exit code."""
    print("🚀 Venice AI Simple Chat Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        # Pick the cheapest model that answers directly (no hidden reasoning
        # eating the small token caps below) and reuse it across sections.
        # prefer="cheapest" ranks strictly by price, and the cheapest model may
        # be a reasoning model; exclude_reasoning=True keeps to models that
        # answer directly.
        try:
            chat_model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no non-reasoning chat model ({e})")
            return 77

        results: list[tuple[str, bool | None]] = [
            ("basic_chat_completion", await basic_chat_completion(client, chat_model)),
            ("chat_with_system_message", await chat_with_system_message(client, chat_model)),
            (
                "different_models_comparison",
                await different_models_comparison(client, chat_model),
            ),
            ("parameter_variations", await parameter_variations(client, chat_model)),
        ]

    failed = [name for name, ok in results if ok is False]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1
    skipped = [name for name, ok in results if ok is None]
    if skipped:
        print(f"\nℹ️ {len(skipped)} section(s) skipped: {', '.join(skipped)}")

    print("\n✨ Simple chat examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Basic chat completions")
    print("   - System messages for personality")
    print("   - Standard vs reasoning model selection with resolve_chat(prefer='cheapest')")
    print("     (exclude_reasoning=True vs require_reasoning=True)")
    print("   - Parameter tuning (temperature)")
    print("   - Checking finish_reason and token usage")
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
