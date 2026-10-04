#!/usr/bin/env python3
"""
Venice AI SDK - Model Feature Suffixes
======================================

This example demonstrates how to use model feature suffixes — parameters appended
directly to model IDs using a colon separator. The Venice API parses these suffixes
and applies them as if they were set in ``venice_parameters`` in the request body.

Suffix Format:
    model_id:param1=value1&param2=value2

This is useful for:
    • OpenAI-compatible clients that don't support extra request body parameters
    • Simple integrations where you can only configure the model name
    • Quick testing without modifying request structure

The SDK also provides a ``build_model_id()`` helper to construct suffixed model
strings programmatically.

Suffix parameters demonstrated here (each one's effect is checked, not assumed):
    • enable_web_search            — "on", "off", or "auto"
    • strip_thinking_response      — "true" or "false"
    • disable_thinking             — "true" or "false"
    • include_venice_system_prompt — "true" or "false"

The server echoes the settings it applied in ``response.venice_parameters``.

Every answer is also checked for truncation: ``finish_reason == "length"``,
or a completion count that reached the cap (some models report a cut-off
answer as ``"stop"``). The script exits 77 if the catalog has no model for the
web-search base, and a section whose reasoning model is missing prints
``Section skipped:``.

Requirements:
    - Venice AI API key (set as VENICE_API_KEY environment variable)
    - Python 3.13+
    - venice-py SDK
"""

import asyncio
import sys

from venice_ai import NoMatchingModelError, VeniceClient, build_model_id
from venice_ai.types.api import ChatCompletionResponse, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# Venice's own system prompt adds hundreds of tokens to every request; leaving
# it out must shrink the prompt by at least this much.
MIN_SYSTEM_PROMPT_TOKENS = 100

# Room for a reasoning model to think and still answer. Small reasoning models
# vary widely in how long they think, even on easy questions, so the cap leaves
# several thousand tokens of headroom over a typical run.
REASONING_CAP = 16384

# =============================================================================
# Helper Functions
# =============================================================================


def print_section_header(title: str, emoji: str = "📋") -> None:
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("=" * 70)


def print_subsection(title: str, emoji: str = "📍") -> None:
    """Print a formatted subsection header."""
    print(f"\n{emoji} {title}")
    print("-" * 50)


def print_response(response: ChatCompletionResponse, cap: int, label: str = "Response") -> bool:
    """Print the full answer, usage and applied settings; ``False`` if unusable.

    ``cap`` is the request's ``max_completion_tokens``; an answer that used
    all of it (or a response without usage) counts as cut off.
    """
    content = (response.text or "").strip()
    print(f"\n📝 {label}:")
    print(content or "(empty response)")

    usage = response.usage
    if usage is not None:
        details = usage.completion_tokens_details
        print(
            f"\n📊 Tokens: Input={usage.prompt_tokens}, Output={usage.completion_tokens} "
            f"(reasoning={details.reasoning_tokens if details else None}), "
            f"Total={usage.total_tokens}"
        )
    vp = response.venice_parameters
    if vp is not None:
        print(
            "📡 Server applied: "
            f"enable_web_search={vp.enable_web_search}, "
            f"strip_thinking_response={vp.strip_thinking_response}, "
            f"disable_thinking={vp.disable_thinking}, "
            f"include_venice_system_prompt={vp.include_venice_system_prompt}"
        )

    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = usage.completion_tokens if usage is not None else None
    print(f"↳ finish_reason: {finish_reason} ({used} of {cap} completion tokens)")
    if finish_reason == "length" or used is None or used >= cap:
        print("❌ The answer may be cut off at max_completion_tokens")
        return False
    if not content:
        print("❌ The model returned no visible answer")
        return False
    return True


def check(condition: bool, success: str, failure: str) -> bool:
    """Print a ✅/❌ line for one verified effect and return the condition."""
    print(f"{'✅' if condition else '❌'} {success if condition else failure}")
    return condition


async def default_prompt_tokens(client: VeniceClient, model: str, question: str) -> int:
    """Prompt tokens for ``question`` with default settings (Venice system prompt included).

    Used only as a reference point for measuring what a setting removed. Only
    ``usage.prompt_tokens`` is read, so the reply is capped very low (some
    models reject a cap below a small minimum).
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[UserMessage(content=question)],
        max_completion_tokens=16,
    )
    if response.usage is None:
        raise RuntimeError("The API returned no usage for the reference request")
    return response.usage.prompt_tokens


def reasoning_tokens(response: ChatCompletionResponse) -> int | None:
    """Reasoning tokens reported for a response, if any."""
    if response.usage is None or response.usage.completion_tokens_details is None:
        return None
    return response.usage.completion_tokens_details.reasoning_tokens


# =============================================================================
# Example Functions
# =============================================================================


async def basic_suffix_example(client: VeniceClient, search_model: str) -> bool:
    """Turn on web search from the model string alone."""
    print_section_header("Basic Model Feature Suffix", "🏷️")

    # "on" always searches. "auto" lets the model decide per question, so it
    # may answer from memory without searching.
    model_with_suffix = f"{search_model}:enable_web_search=on"

    print(f"📍 Model string: {model_with_suffix}")
    print("   The API parses the suffix and runs a web search before answering.")

    response = await client.chat.completions.create(
        model=model_with_suffix,
        messages=[
            UserMessage(content="In two or three sentences, what is the latest news about AI?")
        ],
        max_completion_tokens=600,
        temperature=0.5,
    )

    ok = print_response(response, 600, "Web Search Response (via suffix)")
    citations = response.web_search_citations
    for i, citation in enumerate(citations[:5], 1):
        print(f"   [{i}] {citation.title} - {citation.url}")
    ok = (
        check(
            len(citations) > 0,
            f"Web search ran: {len(citations)} citations returned",
            "No citations returned, so no web search happened",
        )
        and ok
    )
    return ok


async def multiple_params_example(client: VeniceClient, reasoning_model: str) -> bool:
    """Combine two parameters with the ``&`` separator."""
    print_section_header("Multiple Parameters via Suffix", "🔧")

    model_with_params = (
        f"{reasoning_model}:strip_thinking_response=true&include_venice_system_prompt=false"
    )

    print(f"📍 Model string: {model_with_params}")
    print("   Parameters parsed from suffix:")
    print("     • strip_thinking_response = true       (hide the reasoning trace)")
    print("     • include_venice_system_prompt = false (send only our own messages)")

    question = "Briefly explain quantum computing in two sentences."
    reference_tokens = await default_prompt_tokens(client, reasoning_model, question)
    response = await client.chat.completions.create(
        model=model_with_params,
        messages=[UserMessage(content=question)],
        max_completion_tokens=REASONING_CAP,
    )

    ok = print_response(response, REASONING_CAP, "Multi-param Suffix Response")
    # Stripping only means something if the model actually reasoned.
    used = reasoning_tokens(response) or 0
    ok = (
        check(
            used > 0 and not response.thinking_blocks,
            f"Reasoning trace stripped (the model still used {used} reasoning tokens)",
            "A reasoning trace is still present"
            if response.thinking_blocks
            else "The model reported no reasoning tokens, so there was no trace to strip",
        )
        and ok
    )
    prompt_tokens = response.usage.prompt_tokens if response.usage else None
    ok = (
        check(
            prompt_tokens is not None
            and reference_tokens - prompt_tokens >= MIN_SYSTEM_PROMPT_TOKENS,
            f"Venice system prompt left out: {reference_tokens} -> {prompt_tokens} prompt tokens",
            f"Prompt went {reference_tokens} -> {prompt_tokens} tokens, so the Venice system "
            "prompt was still added",
        )
        and ok
    )
    return ok


async def build_model_id_example(client: VeniceClient, reasoning_model: str) -> bool:
    """Build a suffixed model ID with the SDK's ``build_model_id()`` helper.

    The double-suffix ``ValueError`` demo at the end is an intentional input
    validation error; only an unexpected outcome there fails the section.
    """
    print_section_header("Using build_model_id() Helper", "⚡")

    model_id = build_model_id(
        reasoning_model, disable_thinking="true", include_venice_system_prompt="false"
    )

    print(f"📍 Built model ID: {model_id}")
    print("   build_model_id() constructs the suffix string for you.")

    response = await client.chat.completions.create(
        model=model_id,
        messages=[UserMessage(content="Name three benefits of renewable energy, one line each.")],
        max_completion_tokens=1024,
        temperature=0.5,
    )

    ok = print_response(response, 1024, "build_model_id() Response")
    ok = (
        check(
            reasoning_tokens(response) == 0 and not response.thinking_blocks,
            "disable_thinking applied: 0 reasoning tokens and no trace",
            f"The model still reasoned ({reasoning_tokens(response)} reasoning tokens)",
        )
        and ok
    )

    # build_model_id() refuses a model string that already carries a suffix.
    print_subsection("Error Handling", "🛡️")
    try:
        build_model_id(f"{reasoning_model}:enable_web_search=on", disable_thinking="true")
        print("❌ Expected a ValueError for a double suffix")
        ok = False
    except ValueError as e:
        print(f"✅ Expected error for double suffix: {e}")

    return ok


async def comparison_example(client: VeniceClient, base_model: str) -> bool:
    """Side-by-side comparison: model suffix vs ``venice_parameters``."""
    print_section_header("Suffix vs venice_parameters Comparison", "⚖️")

    question = "What is the capital of France? Answer in one sentence."
    params = VeniceParameters(include_venice_system_prompt=False)
    reference_tokens = await default_prompt_tokens(client, base_model, question)

    # --- Approach 1: Model suffix ---
    print_subsection("Approach 1: Model Feature Suffix", "🏷️")
    suffix_model = f"{base_model}:include_venice_system_prompt=false"
    print(f"📍 Model string: {suffix_model}")

    response_suffix = await client.chat.completions.create(
        model=suffix_model,
        messages=[UserMessage(content=question)],
        max_completion_tokens=200,
        temperature=0.3,
    )
    ok = print_response(response_suffix, 200, "Suffix Approach")

    # --- Approach 2: venice_parameters ---
    print_subsection("Approach 2: venice_parameters Object", "🔧")
    print(f"📍 Model: {base_model}")
    print("📍 venice_parameters: VeniceParameters(include_venice_system_prompt=False)")
    # A VeniceParameters object sends every field that has a default, not only
    # the one you set. Pass a plain dict instead to send a single field.
    print(f"   Sent on the wire: {params.model_dump(exclude_none=True)}")

    response_params = await client.chat.completions.create(
        model=base_model,
        messages=[UserMessage(content=question)],
        venice_parameters=params,
        max_completion_tokens=200,
        temperature=0.3,
    )
    ok = print_response(response_params, 200, "venice_parameters Approach") and ok

    # Summary
    print_subsection("Comparison Summary", "📊")
    suffix_tokens = response_suffix.usage.prompt_tokens if response_suffix.usage else None
    params_tokens = response_params.usage.prompt_tokens if response_params.usage else None
    print(
        f"   Prompt tokens — default: {reference_tokens}, suffix: {suffix_tokens}, "
        f"venice_parameters: {params_tokens}"
    )
    ok = (
        check(
            suffix_tokens is not None
            and suffix_tokens == params_tokens
            and reference_tokens - suffix_tokens >= MIN_SYSTEM_PROMPT_TOKENS,
            "Both approaches left out the Venice system prompt identically",
            "The two approaches did not remove the Venice system prompt the same way",
        )
        and ok
    )
    print("   • Suffix approach:   one setting, carried in the model string")
    print("   • venice_parameters: the object's defaults for the other fields are sent too;")
    print("     they match the server's defaults, so the effect is the same here")
    return ok


def use_cases_example() -> bool:
    """Document when to use suffixes vs venice_parameters (no API call needed)."""
    print_section_header("When to Use Model Feature Suffixes", "💡")

    print("""
   USE SUFFIXES WHEN:
     🏷️  Using an OpenAI-compatible client that only exposes the "model" field
     ⚡  Quick testing via curl or simple scripts
     🔌  A proxy can't modify the request body but can set the model name

   USE venice_parameters WHEN:
     🔧  You have full control over the request body
     📋  You want type-safe, IDE-autocompleted parameter names
     🔀  Settings change from request to request and you'd rather not rebuild model strings
""")
    return True


# =============================================================================
# Main Function
# =============================================================================


async def main() -> int:
    """Run all model feature suffix examples; return a process exit code."""
    print("🚀 Venice AI - Model Feature Suffixes")
    print("=" * 70)
    print("Append parameters directly to model IDs for lightweight configuration\n")

    async with VeniceClient() as client:
        # The web-search sections use small caps and compare prompt tokens, so
        # they need a model that answers directly rather than reasoning first.
        try:
            search_model = await client.models.resolve_chat(
                require_web_search=True, exclude_reasoning=True, prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no non-reasoning model supports web search ({e})")
            return 77
        print(f"🤖 Web search base model: {search_model}")
        results: list[tuple[str, bool | None]] = [
            ("basic_suffix_example", await basic_suffix_example(client, search_model)),
        ]

        # Thinking suffixes need a reasoning model whose reasoning can be switched off.
        try:
            reasoning_model = await client.models.resolve_chat(
                require_reasoning=True, require_reasoning_effort="none", prefer="cheapest"
            )
        except NoMatchingModelError:
            print("\nSection skipped: no reasoning model can switch reasoning off")
            results += [("multiple_params_example", None), ("build_model_id_example", None)]
        else:
            print(f"\n🤖 Reasoning base model: {reasoning_model}")
            results += [
                (
                    "multiple_params_example",
                    await multiple_params_example(client, reasoning_model),
                ),
                ("build_model_id_example", await build_model_id_example(client, reasoning_model)),
            ]
        results.append(("comparison_example", await comparison_example(client, search_model)))

    results.append(("use_cases_example", use_cases_example()))

    failed = [name for name, ok in results if ok is False]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1
    skipped = [name for name, ok in results if ok is None]
    if skipped:
        print(f"\nℹ️ {len(skipped)} section(s) skipped: {', '.join(skipped)}")

    print_section_header("Examples Completed! ✨", "🎉")
    print("\n💡 Key takeaways:")
    print("   • Append params to model IDs with : separator  (model:key=val)")
    print("   • Combine multiple params with & separator      (model:a=1&b=2)")
    print("   • Use build_model_id() for programmatic construction")
    print("   • Check response.venice_parameters and usage to confirm the effect")
    print("   • Prefer suffixes for OpenAI-compat clients; venice_parameters otherwise")
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
