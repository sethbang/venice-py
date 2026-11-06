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

Besides the basics, this script sends the request parameters most chat APIs
share: ``stop``, ``n``, ``top_p``, the repetition penalties, ``seed`` and
``logprobs``. ``stop``, ``n`` and ``logprobs`` have an effect a run can check.
``top_p``, the penalties and ``seed`` are requests to the model's sampler, so
their effect is reported rather than asserted. ``seed`` asks for reproducible
sampling; identical output is not promised.

Cost: about a dozen short requests on the cheapest models, typically well under
$0.001 in total. ``n=2`` bills the tokens of both choices. A reasoning model
spends more of its cap, and the comparison section uses one.
"""

import asyncio
import math
import sys
import textwrap
from typing import Any

from _helpers import completion_cap, states_number
from pydantic import ValidationError

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError, model_price
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, TextModelSpec, UserMessage
from venice_ai.types.api.requests import VeniceParameters

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
    candidates = [
        (standard_model, 300),
        (reasoning_model, await completion_cap(client, reasoning_model)),
    ]
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


async def stop_sequences(client: VeniceClient, chat_model: str) -> bool:
    """End the answer at a stop string; the stop string itself is not returned."""
    print("\n🛑 Stop Sequences")
    print("-" * 30)

    response = await client.chat.completions.create(
        model=chat_model,
        messages=[
            UserMessage(content="Count from 1 to 9 in digits, separated by commas. List only.")
        ],
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=60,
        temperature=0.0,
        stop=["5"],
    )
    ok = report_answer(response, 60, indent="   ")
    text = response.text or ""
    # The model reached 4, then generation ended before it could write 5 or 6.
    if not states_number(text, 4) or "5" in text or "6" in text:
        print('   ❌ Expected the count to end after 4 with stop=["5"]')
        return False
    print('   ✅ stop=["5"] ended the answer before the 5')
    return ok


async def multiple_choices(client: VeniceClient, chat_model: str) -> bool:
    """Ask for two alternative answers with ``n=2``.

    ``n`` bills the generated tokens of every choice, so keep it at 1 unless you
    need the alternatives. ``response.text`` reads only the first choice.
    """
    print("\n🎲 Several Choices (n=2)")
    print("-" * 30)

    cap = 30
    response = await client.chat.completions.create(
        model=chat_model,
        messages=[UserMessage(content="Name one color. Reply with one word.")],
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=cap,
        temperature=1.0,
        n=2,
    )
    for choice in response.choices:
        print(f"   [{choice.index}] {choice.message.content!r} ({choice.finish_reason})")
    if response.usage:
        print(f"   📊 Completion tokens across all choices: {response.usage.completion_tokens}")
    if not response.choices or any(
        c.finish_reason == "length" or not c.message.content for c in response.choices
    ):
        print("   ❌ A choice is empty or was cut off")
        return False
    if len(response.choices) != 2:
        print(f"   ℹ️ asked for 2, got {len(response.choices)}")
        return True
    print("   ✅ Two complete choices returned")
    return True


async def sampling_controls(client: VeniceClient, chat_model: str) -> bool:
    """Send ``top_p``, the penalties and ``seed``.

    These shape how the model samples. One run cannot prove their effect, so
    the checks cover what a run can show: each request is accepted and answered
    correctly, and an out-of-range value is rejected before anything is sent.
    """
    print("\n🎛️ Sampling Controls")
    print("-" * 30)

    question = [UserMessage(content="What is the capital of France? Reply with one word.")]
    variants: list[tuple[str, dict[str, Any]]] = [
        ("top_p=0.5", {"top_p": 0.5}),
        (
            "frequency_penalty=0.5, presence_penalty=0.3",
            {"frequency_penalty": 0.5, "presence_penalty": 0.3},
        ),
        ("repetition_penalty=1.1", {"repetition_penalty": 1.1}),
    ]
    ok = True
    for label, sampling in variants:
        print(f"\n   {label}")
        response = await client.chat.completions.create(
            model=chat_model,
            messages=question,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=30,
            **sampling,
        )
        ok = report_answer(response, 30, indent="   ") and ok
        if "paris" not in (response.text or "").lower():
            print("   ❌ The answer does not name Paris")
            ok = False

    # Range checks run in the SDK, so this costs nothing.
    try:
        await client.chat.completions.create(
            model=chat_model,
            messages=question,
            max_completion_tokens=30,
            top_p=1.5,
        )
    except ValidationError as e:
        print(f"\n   ✅ top_p=1.5 rejected before sending: {e.errors()[0]['msg']}")
    else:
        print("\n   ❌ top_p=1.5 was sent and accepted")
        ok = False

    print("\n   seed=12345, twice (a request for reproducible sampling, not a promise)")
    answers: list[str] = []
    for _ in range(2):
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Invent a two-word band name. Reply with it only.")],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=30,
            temperature=1.0,
            seed=12345,
        )
        ok = report_answer(response, 30, indent="   ") and ok
        answers.append((response.text or "").strip())
    same = "identical" if answers[0] == answers[1] else "different"
    print(f"   📏 Measured: the two answers were {same} on this run")
    return ok


async def cheapest_logprobs_model(client: VeniceClient) -> str | None:
    """The cheapest direct-answer model whose catalog entry lists ``supportsLogProbs``."""
    candidates: list[tuple[float, str]] = []
    for model in (await client.models.list(type="text")).data:
        spec = model.model_spec
        if not isinstance(spec, TextModelSpec) or spec.beta or spec.offline:
            continue
        caps = spec.capabilities
        if caps is None or not caps.supportsLogProbs or caps.supportsReasoning or caps.supportsE2EE:
            continue
        price = model_price(model.model_dump())
        candidates.append((price if price is not None else math.inf, model.id))
    return min(candidates)[1] if candidates else None


async def log_probabilities(client: VeniceClient) -> bool | None:
    """Read a token's log probability with ``logprobs=True``.

    Returns ``None`` (section skipped) when no model lists ``supportsLogProbs``.
    """
    print("\n📈 Log Probabilities")
    print("-" * 30)

    model = await cheapest_logprobs_model(client)
    if model is None:
        print("Section skipped: no direct-answer model lists supportsLogProbs")
        return None
    print(f"📍 Using model: {model}")

    response = await client.chat.completions.create(
        model=model,
        messages=[UserMessage(content="Reply with the single word yes.")],
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=16,
        logprobs=True,
        top_logprobs=2,
    )
    ok = report_answer(response, 16, indent="   ")
    logprobs = response.choices[0].logprobs if response.choices else None
    # ``tokens`` reads both forms a server may send: one token, or one per token.
    tokens = logprobs.tokens if logprobs is not None else []
    if not tokens:
        print("   ❌ The model lists supportsLogProbs but returned no readable logprobs")
        return False
    first = tokens[0]
    print(f"   token={first.token!r} logprob={first.logprob}")
    print(f"   alternatives: {[(t.token, t.logprob) for t in first.top_logprobs or []]}")
    if any(t.logprob > 0 for t in tokens):
        print("   ❌ A log probability cannot be above 0")
        return False
    print("   ✅ Log probability returned")
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
            ("stop_sequences", await stop_sequences(client, chat_model)),
            ("multiple_choices", await multiple_choices(client, chat_model)),
            ("sampling_controls", await sampling_controls(client, chat_model)),
        ]
        # A model may list logprobs in a shape this SDK release does not parse, so
        # that request is kept from stopping the summary below.
        try:
            results.append(("log_probabilities", await log_probabilities(client)))
        except VeniceError as e:
            print(f"   ❌ The logprobs request failed ({type(e).__name__}): {e}")
            results.append(("log_probabilities", False))

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
    print("   - Parameter tuning (temperature, top_p, penalties, seed)")
    print("   - stop sequences, n choices and logprobs")
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
