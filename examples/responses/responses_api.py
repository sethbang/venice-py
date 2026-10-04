#!/usr/bin/env python3
"""
Venice AI SDK - Responses API Example (Alpha)
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

``status`` is the Responses API's equivalent of ``finish_reason``: only
``"completed"`` means the model finished its answer.

The endpoint is tagged **Alpha** by Venice; request/response shapes may change
without notice, and some accounts may not be entitled. A demo the account
cannot access (HTTP 403), or one whose kind of model the catalog lacks, prints
a ``Section skipped:`` line. The basic call is the core of the example: when it
cannot run, the script prints ``SKIPPED:`` and exits 77, otherwise it exits 0
only if every demo that ran passed.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import PermissionDeniedError, VeniceError
from venice_ai.types.api.requests import VeniceParameters
from venice_ai.types.api.responses import (
    ResponsesMessageOutput,
    ResponsesReasoningOutput,
    ResponsesResponse,
)

# Leave out Venice's default system prompt. A model's own prompt template can
# still add input tokens of its own, which this switch does not remove.
NO_SYSTEM_PROMPT = VeniceParameters(include_venice_system_prompt=False)

#: Exit code for "skipped": no chat model in the catalog, or the account has
#: no access to the Responses API.
EXIT_SKIPPED = 77


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
                # Each content item is an ``output_text`` block with a ``.text``.
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


async def reasoning_output_blocks(client: VeniceClient) -> bool | None:
    """Read the ``reasoning`` block a reasoning model returns before its answer.

    The model comes from the resolver's default ranking for reasoning models,
    which follows Venice's own recommendations. Not every backend returns a
    ``reasoning`` block on ``/responses`` even when it bills reasoning tokens;
    the check below tells that case apart from a model that did not reason.

    Returns ``None`` (section skipped) when the catalog has no reasoning model.
    """
    print("\n🧠 Reasoning Output Blocks")
    print("-" * 40)

    try:
        model = await client.models.resolve_chat(require_reasoning=True)
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
            # reasoning text and only its start is shown.
            shown = " ".join(item.split())
            if len(shown) > 300:
                shown = f"{shown[:300]}… ({len(item)} chars in total)"
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
        print(
            f"\n❌ {model} billed {reasoning_tokens} reasoning tokens but /responses "
            "returned no reasoning block (a server-side gap, not an SDK parse issue)"
        )
        return False
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


async def main() -> int:
    """Run all Responses API demos.

    Returns ``1`` if any demo failed, ``77`` when the basic call could not run
    (no chat model in the catalog, or no access to the endpoint), and ``0``
    otherwise. Other demos may be skipped individually; the basic call
    decides whether the example as a whole was verified.
    """
    print("🚀 Venice AI Responses API Example (Alpha)")
    print("=" * 50)

    results: list[tuple[str, bool]] = []
    skipped: list[str] = []
    async with VeniceClient() as client:
        # The default ranking (not prefer="cheapest") is used here: the
        # truncation demo needs a backend that reports status="incomplete" at
        # the cap, and not every one does (the demo detects that case). The
        # default ranking never selects E2EE models, which /responses rejects.
        try:
            model = await client.models.resolve_chat()
        except NoMatchingModelError as e:
            print(f"SKIPPED: no chat model in the catalog ({e})")
            return EXIT_SKIPPED
        print(f"🤖 Using model: {model}\n")

        sections = [
            ("basic_responses_create", lambda: basic_responses_create(client, model)),
            ("structured_input_responses", lambda: structured_input_responses(client, model)),
            ("reasoning_output_blocks", lambda: reasoning_output_blocks(client)),
            ("truncated_response", lambda: truncated_response(client, model)),
        ]
        for name, section in sections:
            try:
                outcome = await section()
            except PermissionDeniedError as e:
                # The Responses API is Alpha and not every account (or model)
                # is entitled. That is an access condition, not a failure.
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

    if "basic_responses_create" in skipped:
        print("\nSKIPPED: this account has no access to the Responses API (Alpha)")
        return EXIT_SKIPPED
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
            "Responses API (Alpha).",
            file=sys.stderr,
        )
        sys.exit(1)
