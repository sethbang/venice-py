#!/usr/bin/env python3
"""
Venice AI SDK - Quick Start Example
==================================

This example demonstrates the most basic usage of the Venice AI SDK.
Get up and running with just a few lines of code!

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"

Exit status: ``0`` for a complete answer, ``1`` for an empty or cut-off reply
or an API error, ``77`` (with a ``SKIPPED:`` line) when the catalog has no
chat model that answers directly.
"""

import asyncio
import sys

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.types.api import UserMessage
from venice_ai.types.api.requests import VeniceParameters

#: Plenty for a short greeting from a model that answers directly.
MAX_COMPLETION_TOKENS = 256

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77


async def main() -> int:
    """Quick start example showing basic Venice AI usage.

    Returns ``0`` when the model produced a complete answer, ``1`` when the
    reply was empty or cut off by the token cap, and ``77`` when no chat model
    in the catalog fits.
    """

    print("🚀 Venice AI Quick Start Example")
    print("=" * 40)

    # Create a Venice AI client (reads VENICE_API_KEY from environment)
    async with VeniceClient() as client:
        print("✅ Client created successfully")

        # Pick the cheapest chat model in the live catalog that answers
        # directly. Without exclude_reasoning=True the cheapest pick can be a
        # reasoning model, which thinks before it writes the visible reply.
        try:
            chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        except NoMatchingModelError as e:
            print(f"SKIPPED: no chat model in the catalog answers directly ({e})")
            return EXIT_SKIPPED
        print(f"📍 Using model: {chat_model}")

        # Simple chat completion
        print("\n💬 Creating a simple chat completion...")

        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Say hello and introduce yourself briefly.")],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            # Venice prepends its own system prompt unless told not to, and bills
            # it as prompt tokens (over a thousand for a one-line question).
            # This request needs nothing from it, so send only our message.
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )

        # Extract and display the response
        message = response.text or ""
        finish_reason = response.choices[0].finish_reason if response.choices else None
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        print(f"🤖 Assistant: {message}")
        print(f"🏁 Finish reason: {finish_reason}")

        # Show token usage
        if response.usage:
            usage = response.usage
            print("\n📊 Token Usage:")
            print(f"   Prompt tokens: {usage.prompt_tokens}")
            print(f"   Completion tokens: {usage.completion_tokens}")
            print(f"   Total tokens: {usage.total_tokens}")
            print("   ℹ️  Venice's system prompt was left out (include_venice_system_prompt=False)")

    # Some backends report finish_reason="stop" even when the reply ran into
    # the cap, so a reply that used the whole budget counts as cut off too.
    if finish_reason == "length" or completion_tokens >= MAX_COMPLETION_TOKENS:
        print("\n❌ The reply was cut off by max_completion_tokens.", file=sys.stderr)
        return 1
    if not message.strip():
        print("\n❌ The model returned no text.", file=sys.stderr)
        return 1

    print("\n✨ Quick start completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have internet connection.", file=sys.stderr)
        sys.exit(1)
