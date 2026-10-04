#!/usr/bin/env python3
"""
Venice AI SDK - Reasoning and Thinking Examples
===============================================

This example demonstrates how to use models with reasoning capabilities and
how to control their thinking with ``VeniceParameters``.

Learn how to:
- Select reasoning models with ``client.models.resolve_chat(require_reasoning=True)``
  (``prefer="cheapest"`` picks the lowest-priced one)
- Read the reasoning trace (``response.thinking_blocks`` / ``reasoning_content``)
- Hide the trace with ``strip_thinking_response`` (the model still reasons)
- Switch reasoning off with ``disable_thinking``, ``reasoning_effort="none"`` or
  ``reasoning=ReasoningConfig(enabled=False)``
- Read typed reasoning-token usage (``completion_tokens_details``)

Reasoning models spend part of ``max_completion_tokens`` on hidden reasoning
before they write the answer, so the caps here are generous. The requests also
leave ``temperature`` at the server default: model publishers recommend their
own sampling settings for thinking mode, and they differ from model to model.
Every answer is checked: a reply cut off at the cap (``finish_reason ==
"length"``, or a completion count that reached the cap) or with a wrong final
answer fails its section, and the script exits non-zero. If the catalog has no
model a section needs, that section prints ``Section skipped:``; the script
exits 77 when no reasoning model exists at all.
"""

import asyncio
import io
import re
import sys
from typing import Literal

from venice_ai import NoMatchingModelError, ReasoningConfig, VeniceClient
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, TextModelSpec, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# Room for reasoning plus the visible answer. Small reasoning models vary
# widely in how long they think, even on easy questions, so the cap leaves
# several thousand tokens of headroom over a typical run.
REASONING_CAP = 16384

ANSWER_INSTRUCTION = "Finish with a final line of the form 'ANSWER: {format}'."


def answer_line(text: str) -> str:
    """Return the last ``ANSWER:`` line, lower-cased with whitespace collapsed ('' if none)."""
    matches = re.findall(r"ANSWER:\s*(.+)", text.replace("*", ""))
    return re.sub(r"\s+", " ", matches[-1]).strip().lower() if matches else ""


def contains_part(answer: str, part: str) -> bool:
    """True if ``part`` appears in ``answer`` as a whole value.

    Spaces are optional (``"5minutes"`` matches ``"5 minutes"``), but the part
    may not be glued to a neighbouring letter or digit, so ``"25 minutes"``
    does not count as ``"5minutes"``.
    """
    body = r"\s*".join(re.escape(char) for char in part.lower().replace(" ", ""))
    return re.search(rf"(?<![a-z0-9]){body}(?![a-z0-9])", answer) is not None


def check_response(response: ChatCompletionResponse, expected: list[str]) -> bool:
    """Print ``finish_reason`` and verify the ``ANSWER:`` line holds every expected part.

    Some models report an answer cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked against ``REASONING_CAP`` too.
    """
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    print(f"↳ finish_reason: {finish_reason}")
    if finish_reason == "length" or used is None or used >= REASONING_CAP:
        print(f"❌ The answer may be cut off: {used} of {REASONING_CAP} completion tokens used")
        return False
    answer = answer_line(response.text or "")
    if not answer:
        print("❌ No 'ANSWER:' line in the response")
        return False
    missing = [part for part in expected if not contains_part(answer, part)]
    if missing:
        print(f"❌ Final answer does not match the expected {expected} (got {answer!r})")
        return False
    print(f"✅ Final answer verified: {expected}")
    return True


def reasoning_tokens(response: ChatCompletionResponse) -> int | None:
    """Reasoning tokens reported in ``usage.completion_tokens_details``, if any."""
    usage = response.usage
    if usage is None or usage.completion_tokens_details is None:
        return None
    return usage.completion_tokens_details.reasoning_tokens


def print_usage(response: ChatCompletionResponse) -> None:
    """Print token usage including the reasoning share."""
    if response.usage:
        usage = response.usage
        print(
            f"📊 Token Usage: Input={usage.prompt_tokens}, Output={usage.completion_tokens} "
            f"(reasoning={reasoning_tokens(response)}), Total={usage.total_tokens}"
        )


def format_thinking_output(thinking_blocks: list[str], message: str, title: str = "Response"):
    """Show the start of the reasoning trace and the full answer."""
    print("\n" + "=" * 70)

    if thinking_blocks:
        print("💭 THINKING PROCESS (first lines)")
        print("-" * 70)
        for i, block in enumerate(thinking_blocks, 1):
            block_lines = block.strip().split("\n")
            print(f"  Block {i}:")
            for line in block_lines[:5]:
                print(f"    {line}")
            if len(block_lines) > 5:
                print(f"    ... ({len(block_lines) - 5} more lines)")
        print("-" * 70)
    else:
        print("ℹ️ No thinking blocks in the response")
        print("-" * 70)

    print(f"📝 {title.upper()}")
    print("-" * 70)
    print(message if message else "(No message content)")
    print("=" * 70)


async def basic_reasoning_with_venice_parameters(client: VeniceClient, model: str) -> bool:
    """Show the reasoning trace alongside the answer (``strip_thinking_response=False``)."""
    print("\n🧠 Basic Reasoning with VeniceParameters")
    print("=" * 70)
    print(f"📍 Using reasoning model: {model}")

    problem = (
        "I have a sequence: 2, 6, 12, 20, 30, ...\n"
        "What are the next three numbers in this sequence? Explain the pattern briefly. "
        + ANSWER_INSTRUCTION.format(format="<n1>, <n2>, <n3>")
    )

    # Only the fields you set are meaningful; the rest keep their defaults.
    # Our own system message is all the instruction this needs, so Venice's
    # system prompt is left out.
    venice_params = VeniceParameters(
        strip_thinking_response=False,
        disable_thinking=False,
        include_venice_system_prompt=False,
    )
    print(
        "\n🔬 VeniceParameters(strip_thinking_response=False, disable_thinking=False, "
        "include_venice_system_prompt=False)"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            SystemMessage(content="You are a mathematical problem solver."),
            UserMessage(content=problem),
        ],
        venice_parameters=venice_params,
        # No temperature override: the server default applies (see the module
        # docstring).
        max_completion_tokens=REASONING_CAP,
    )

    if response.venice_parameters:
        print("\n📡 Server echoed:")
        print(f"   strip_thinking_response: {response.venice_parameters.strip_thinking_response}")
        print(f"   disable_thinking: {response.venice_parameters.disable_thinking}")

    # thinking_blocks handles both server shapes: a separate reasoning_content
    # field, or <think> tags inline in the content.
    thinking_blocks = response.thinking_blocks
    format_thinking_output(thinking_blocks, (response.text or "").strip(), "FINAL ANSWER")
    print_usage(response)

    ok = check_response(response, ["42", "56", "72"])
    if not thinking_blocks:
        print("❌ Expected a visible reasoning trace with strip_thinking_response=False")
        ok = False
    return ok


async def test_strip_thinking_parameter(client: VeniceClient, model: str) -> bool:
    """Hide the reasoning trace with ``strip_thinking_response=True``.

    Stripping only removes the trace from the response. The model still
    reasons, and those reasoning tokens are still billed. The request also
    leaves out Venice's own system prompt, so only our system message applies.
    """
    print("\n🎭 Testing strip_thinking_response with VeniceParameters")
    print("=" * 70)
    print(f"📍 Using reasoning model: {model}")

    # The clues admit exactly one solution: Alice=blue, Bob=green, Charlie=red.
    puzzle = (
        "Alice, Bob, and Charlie each have a different favorite color: red, blue, or green.\n"
        "- Alice doesn't like red or green.\n"
        "- The person who likes blue is not Charlie.\n"
        "- Bob's favorite color comes before Charlie's alphabetically.\n"
        "What is each person's favorite color? "
        + ANSWER_INSTRUCTION.format(format="Alice=<color>, Bob=<color>, Charlie=<color>")
    )

    venice_params = VeniceParameters(
        strip_thinking_response=True,  # Clean output for the user
        disable_thinking=False,  # The model still thinks first
        include_venice_system_prompt=False,  # Only our own system prompt
    )
    print(
        "\n🔬 VeniceParameters(strip_thinking_response=True, disable_thinking=False, "
        "include_venice_system_prompt=False)"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            SystemMessage(content="You are a logic puzzle solver."),
            UserMessage(content=puzzle),
        ],
        venice_parameters=venice_params,
        # No temperature override: the server default applies (see the module
        # docstring).
        max_completion_tokens=REASONING_CAP,
    )

    if response.venice_parameters:
        print("\n📡 Server echoed:")
        print(f"   strip_thinking_response: {response.venice_parameters.strip_thinking_response}")
        print(
            "   include_venice_system_prompt: "
            f"{response.venice_parameters.include_venice_system_prompt}"
        )

    print("-" * 70)
    print("📝 SOLUTION")
    print("-" * 70)
    print((response.text or "").strip())
    print("-" * 70)
    print_usage(response)

    ok = check_response(response, ["alice=blue", "bob=green", "charlie=red"])
    if response.thinking_blocks:
        print("❌ A reasoning trace is still present despite strip_thinking_response=True")
        ok = False
    else:
        print("✅ No reasoning trace in the response")
    hidden = reasoning_tokens(response)
    if hidden:
        print(f"ℹ️ The model still reasoned: {hidden} reasoning tokens were used (and billed).")
    return ok


async def test_disable_thinking_parameter(client: VeniceClient, model: str) -> bool:
    """Compare reasoning on vs off, and verify that off really skips reasoning.

    A hidden trace alone proves nothing: the check is the reported
    ``reasoning_tokens`` (or, failing that, a large drop in completion tokens).
    There are three ways to switch reasoning off:

    - ``VeniceParameters(disable_thinking=True)``, Venice's parameter.
    - ``reasoning_effort="none"``, the OpenAI-style top-level field.
    - ``reasoning=ReasoningConfig(enabled=False)``, the nested toggle Venice
      recommends. Venice's API reference says ``enabled`` is ignored when an
      effort is also given, so this variant sends no effort.
    """
    print("\n🔄 Testing disable_thinking with VeniceParameters")
    print("=" * 70)
    print(f"📍 Using reasoning model: {model}")

    # A plain question keeps the "thinking on" baseline short: trick questions
    # can send a small reasoning model round in circles until it hits the cap.
    question = "How many minutes are in 2.5 hours? " + ANSWER_INSTRUCTION.format(
        format="<number> minutes"
    )
    messages = [UserMessage(content=question)]

    # Every variant leaves Venice's system prompt out, so the four requests
    # differ only in the reasoning switch.
    no_prompt = VeniceParameters(include_venice_system_prompt=False)
    variants: list[tuple[str, dict]] = [
        (
            "thinking on (default)",
            {
                "venice_parameters": VeniceParameters(
                    disable_thinking=False, include_venice_system_prompt=False
                )
            },
        ),
        (
            "disable_thinking=True",
            {
                "venice_parameters": VeniceParameters(
                    disable_thinking=True, include_venice_system_prompt=False
                )
            },
        ),
        ('reasoning_effort="none"', {"reasoning_effort": "none", "venice_parameters": no_prompt}),
        (
            "ReasoningConfig(enabled=False)",
            {"reasoning": ReasoningConfig(enabled=False), "venice_parameters": no_prompt},
        ),
    ]
    # Only the fields you set are sent, so the request carries {"enabled": false}.
    wire = ReasoningConfig(enabled=False).model_dump(exclude_none=True)
    print(f"🔬 reasoning=ReasoningConfig(enabled=False) is sent as: {wire}")

    ok = True
    responses: dict[str, ChatCompletionResponse] = {}
    for label, extra in variants:
        print(f"\n📍 {label}")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=REASONING_CAP,
            **extra,
        )
        responses[label] = response
        print(f"   Answer: {(response.text or '').strip()}")
        print(f"   Visible reasoning trace: {'yes' if response.thinking_blocks else 'no'}")
        print_usage(response)
        ok = check_response(response, ["150minutes"]) and ok

    baseline = responses["thinking on (default)"]
    base_reasoning = reasoning_tokens(baseline)
    base_completion = baseline.usage.completion_tokens if baseline.usage else None
    print("\n📊 Did the 'off' variants actually skip reasoning?")
    for label, _ in variants[1:]:
        response = responses[label]
        off_reasoning = reasoning_tokens(response)
        off_completion = response.usage.completion_tokens if response.usage else None
        if off_reasoning is not None:
            skipped = off_reasoning == 0 and (base_reasoning or 0) > 0
            evidence = f"reasoning_tokens {base_reasoning} -> {off_reasoning}"
        elif base_completion and off_completion is not None:
            skipped = off_completion < base_completion / 2
            evidence = f"completion_tokens {base_completion} -> {off_completion}"
        else:
            skipped, evidence = False, "no usage reported"
        print(f"   {'✅' if skipped else '❌'} {label}: {evidence}")
        ok = ok and skipped
    return ok


async def usage_breakdown_with_typed_access(client: VeniceClient) -> bool | None:
    """Read the typed ``completion_tokens_details`` and the cache accessors.

    Returns ``None`` when the catalog has no model for this section.

    Reasoning models populate the ``completion_tokens_details`` object so callers
    can separate visible output from internal reasoning tokens.
    ``usage.cached_tokens`` and ``usage.cache_write_tokens`` read the prompt
    cache counts whichever shape the server used (``prompt_tokens_details`` or
    the top-level ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
    mirrors), and read ``0`` when the server sent neither.

    ``reasoning_effort`` values differ per model, so the model is resolved with
    ``require_reasoning_effort`` set to the value this request sends.
    """
    print("\n🧮 Typed Usage Breakdown (reasoning + cache)")
    print("=" * 70)

    effort: Literal["low"] = "low"
    try:
        model = await client.models.resolve_chat(
            require_reasoning=True, require_reasoning_effort=effort, prefer="cheapest"
        )
    except NoMatchingModelError:
        print(f"Section skipped: no reasoning model accepts reasoning_effort={effort!r}")
        return None
    print(f"📍 Using reasoning model: {model} (reasoning_effort={effort!r})")

    # Venice's system prompt stays on here: it is a long prefix shared by many
    # requests, so it is often served from the prompt cache and the cache
    # counts below show a real reading instead of zeros.
    response = await client.chat.completions.create(
        model=model,
        messages=[
            UserMessage(content="What is 47 * 83? " + ANSWER_INSTRUCTION.format(format="<number>"))
        ],
        reasoning_effort=effort,
        max_completion_tokens=REASONING_CAP,
    )
    ok = check_response(response, ["3901"])

    usage = response.usage
    if not usage:
        print("❌ No usage info returned")
        return False

    print(
        "📊 Token Usage: "
        f"Input={usage.prompt_tokens}, Output={usage.completion_tokens}, "
        f"Total={usage.total_tokens}"
    )

    if usage.completion_tokens_details is not None:
        details = usage.completion_tokens_details
        print(
            "🧠 Completion details: "
            f"reasoning_tokens={details.reasoning_tokens}, "
            f"audio_tokens={details.audio_tokens}, "
            f"image_tokens={details.image_tokens}"
        )
    else:
        print("❌ This model reported no completion_tokens_details")
        ok = False

    print(
        f"💾 Prompt cache: cached_tokens={usage.cached_tokens}, "
        f"cache_write_tokens={usage.cache_write_tokens}"
    )
    return ok


async def model_capability_exploration(client: VeniceClient) -> bool:
    """List reasoning-capable models and their reasoning controls from the catalog."""
    print("\n🔍 Model Reasoning Capabilities")
    print("=" * 70)

    # ``capabilities`` only exists on text models; restrict the listing.
    all_models = await client.models.list(type="text")

    reasoning_models = []
    for model in all_models.data:
        spec = model.model_spec
        if (
            isinstance(spec, TextModelSpec)
            and spec.capabilities
            and spec.capabilities.supportsReasoning
        ):
            reasoning_models.append((model.id, spec.capabilities))

    if not reasoning_models:
        print("❌ The catalog lists no reasoning-capable text models")
        return False

    with_effort = [entry for entry in reasoning_models if entry[1].reasoningEffortOptions]
    switchable = [
        entry for entry in with_effort if "none" in (entry[1].reasoningEffortOptions or [])
    ]
    print(f"📊 {len(reasoning_models)} text models support reasoning")
    print(f"   {len(with_effort)} accept reasoning_effort")
    print(f"   {len(switchable)} can switch reasoning off with reasoning_effort='none'")
    print("   The rest reason at a fixed level the request can't change.\n")

    # Models with a reasoning_effort control first, since that is what you tune.
    fixed = [entry for entry in reasoning_models if not entry[1].reasoningEffortOptions]
    shown = (with_effort + fixed)[:5]
    print(f"Showing {len(shown)} of them:\n")
    for model_id, caps in shown:
        print(f"🤖 Model: {model_id}")
        if caps.reasoningEffortOptions:
            print(
                f"   ✓ Reasoning effort options: {caps.reasoningEffortOptions} "
                f"(default {caps.defaultReasoningEffort!r})"
            )
        else:
            print("   ✓ Reasoning effort: fixed (no reasoning_effort control)")
        print(f"   ✓ Vision: {caps.supportsVision}")
        print(f"   ✓ Functions: {caps.supportsFunctionCalling}")
        print(f"   ✓ Web Search: {caps.supportsWebSearch}")
        print()
    return True


async def main() -> int:
    """Run all reasoning and thinking examples; return a process exit code."""
    # Line-buffer stdout so the header printed *before* each (slow) completion
    # survives even if the process is killed mid-request — making it obvious
    # which call was in flight.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    print("🚀 Venice AI Reasoning & Thinking Examples with VeniceParameters")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            reasoning_model = await client.models.resolve_chat(
                require_reasoning=True, prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no reasoning model ({e})")
            return 77

        results: list[tuple[str, bool | None]] = [
            ("model_capability_exploration", await model_capability_exploration(client)),
            (
                "basic_reasoning_with_venice_parameters",
                await basic_reasoning_with_venice_parameters(client, reasoning_model),
            ),
            (
                "test_strip_thinking_parameter",
                await test_strip_thinking_parameter(client, reasoning_model),
            ),
        ]

        # The catalog has no flag for disable_thinking. A model that accepts
        # reasoning_effort="none" is one whose reasoning can be switched off.
        try:
            switchable_model = await client.models.resolve_chat(
                require_reasoning=True, require_reasoning_effort="none", prefer="cheapest"
            )
        except NoMatchingModelError:
            print("\nSection skipped: no reasoning model can switch reasoning off")
            results.append(("test_disable_thinking_parameter", None))
        else:
            results.append(
                (
                    "test_disable_thinking_parameter",
                    await test_disable_thinking_parameter(client, switchable_model),
                )
            )
        results.append(
            ("usage_breakdown_with_typed_access", await usage_breakdown_with_typed_access(client))
        )

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"\nℹ️ {len(skipped)} section(s) skipped: {', '.join(skipped)}")

    print("\n✨ Examples completed!")
    print("\n💡 Key Insights:")
    print("   - Pick reasoning models with resolve_chat(require_reasoning=True, prefer='cheapest')")
    print("   - strip_thinking_response hides the trace; the model still reasons and bills for it")
    print(
        "   - disable_thinking, reasoning_effort='none' and ReasoningConfig(enabled=False)"
        " skip reasoning on models that allow it"
    )
    print("   - Check usage.completion_tokens_details.reasoning_tokens to confirm")
    print("   - Check response.venice_parameters for the settings the server applied")
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
