#!/usr/bin/env python3
"""
Venice AI SDK - API Compatibility and Migration
================================================

This example demonstrates how to migrate code written for other AI APIs
(OpenAI, Anthropic) to Venice AI:

- Converting the call pattern, message format and parameters
- Reading the live compatibility alias map, and why new code should pick its
  model with ``resolve_*()`` rather than rely on an alias
- Running the migrated function once, live, and inspecting the real
  OpenAI-shaped response it returns

The live sections make one short chat completion and one embedding call.

Exit status: ``0`` when every live section passed, ``1`` otherwise, and ``77``
(with a ``SKIPPED:`` line) when the catalog has no chat model that answers
directly. Without an embedding model only that section prints
``Section skipped:``.
"""

import asyncio
import inspect
import sys
from collections import Counter

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, UserMessage
from venice_ai.types.api.requests import VeniceParameters

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77

MAX_COMPLETION_TOKENS = 200

#: The resolver call this guide recommends for migrated code. exclude_reasoning
#: keeps the answer direct, like the non-reasoning models most OpenAI and
#: Anthropic code was written against; drop it to let reasoning models compete.
RESOLVE_CALL = "client.models.resolve_chat(prefer='cheapest', exclude_reasoning=True)"


async def get_response(client: VeniceClient, prompt: str) -> ChatCompletionResponse:
    """The migrated function: resolve a chat model, then call the chat API."""
    model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    return await client.chat.completions.create(
        model=model,
        messages=[
            SystemMessage(content="You are helpful. Answer in one sentence."),
            UserMessage(content=prompt),
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        # Send only these messages, without Venice's default system prompt.
        venice_parameters=VeniceParameters(include_venice_system_prompt=False),
    )


def _print_block(lines: list[str]) -> None:
    print("```python")
    for line in lines:
        print(line)
    print("```")


async def chat_migration(text_compat: dict[str, str], resolved_chat: str) -> bool:
    """Show OpenAI and Anthropic chat code next to its Venice equivalent."""
    print("🔄 OpenAI / Anthropic → Venice AI: Chat Completions")
    print("-" * 50)

    print("📝 Original OpenAI pattern:")
    _print_block(
        [
            "from openai import OpenAI",
            "client = OpenAI(api_key='...')",
            "response = client.chat.completions.create(",
            "    model=OPENAI_MODEL,",
            "    messages=[{'role': 'user', 'content': 'Hello!'}],",
            "    max_tokens=200,",
            ")",
        ]
    )

    print("\n📝 Original Anthropic pattern:")
    _print_block(
        [
            "from anthropic import Anthropic",
            "client = Anthropic(api_key='...')",
            "message = client.messages.create(",
            "    model=ANTHROPIC_MODEL,",
            "    system='You are helpful.',",
            "    messages=[{'role': 'user', 'content': 'Hello!'}],",
            "    max_tokens=200,",
            ")",
        ]
    )

    print("\n✨ Venice AI equivalent (async):")
    _print_block(
        [
            "from venice_ai import VeniceClient",
            "from venice_ai.types.api import SystemMessage, UserMessage",
            "",
            "async with VeniceClient() as client:",
            f"    model = await {RESOLVE_CALL}",
            "    response = await client.chat.completions.create(",
            "        model=model,",
            "        messages=[",
            "            SystemMessage(content='You are helpful.'),",
            "            UserMessage(content='Hello!'),",
            "        ],",
            "        max_completion_tokens=200,",
            "    )",
        ]
    )
    print("\n   Prefer blocking code? SyncVeniceClient offers the same methods without await.")

    print("\n🔍 Live text alias map:")
    if not text_compat:
        print(f"   ℹ️ No text aliases are published; {RESOLVE_CALL} is the only path.")
        return True

    targets = Counter(text_compat.values())
    print(f"   {len(text_compat)} external names are accepted as aliases, landing on:")
    for target, count in targets.most_common():
        note = (
            "same as the resolver's pick" if target == resolved_chat else "not the resolver's pick"
        )
        print(f"      {count:3d} → {target} ({note})")
    print(f"   {RESOLVE_CALL} currently picks: {resolved_chat}")
    print("\n   An alias keeps old code running, but its target is fixed server-side and")
    print(f"   can trail newer models. Migrated code should call {RESOLVE_CALL}.")
    return True


async def embedding_migration(client: VeniceClient) -> bool | None:
    """Migrate embeddings code, then make one live call to show the response shape.

    Returns ``None`` (section skipped) when the catalog has no embedding model.
    """
    print("\n🔄 Embeddings Migration")
    print("-" * 50)

    print("📝 Original OpenAI pattern:")
    _print_block(
        [
            "response = client.embeddings.create(",
            "    model=OPENAI_EMBEDDING_MODEL,",
            "    input=['text to embed'],",
            ")",
        ]
    )
    print("\n✨ Venice AI equivalent:")
    _print_block(
        [
            "model = await client.models.resolve_embedding(prefer='cheapest')",
            "response = await client.embeddings.create(model=model, input=['text to embed'])",
            "vector = response.data[0].embedding",
        ]
    )

    try:
        compat = (await client.models.list_compatibility(type="embedding")).data
        model = await client.models.resolve_embedding(prefer="cheapest")
        response = await client.embeddings.create(model=model, input=["text to embed"])
    except NoMatchingModelError as e:
        print(f"Section skipped: no embedding model in the catalog ({e})")
        return None
    except VeniceError as e:
        print(f"❌ Embeddings migration check failed: {e}")
        return False

    print("\n🔍 Live check:")
    for alias, target in compat.items():
        note = "same as the resolver's pick" if target == model else "differs from resolver"
        print(f"   alias {alias} → {target} ({note})")
    if not compat:
        print("   ℹ️ No embedding aliases are published.")

    if not response.data or not response.data[0].embedding:
        print("❌ The embeddings response carried no vector.")
        return False
    vector = response.data[0].embedding
    print(f"   resolve_embedding(prefer='cheapest') → {model}")
    print(f"   response.model:      {response.model}")
    print(f"   response.data[0].embedding: {len(vector)} floats")
    return True


def parameter_mapping_guide() -> None:
    """Show how common parameters map between APIs."""
    print("\n📊 Parameter Mapping Guide")
    print("-" * 50)

    print("\nOpenAI/Anthropic → Venice AI")
    print("─" * 40)

    param_mappings = [
        ("model", "model", "Pick it with resolve_chat(prefer='cheapest', ...)"),
        ("messages", "messages", "Typed message models (UserMessage, ...)"),
        ("system (Anthropic)", "SystemMessage", "First entry in messages"),
        ("temperature", "temperature", "Same meaning"),
        ("max_tokens", "max_completion_tokens", "Maximum tokens to generate"),
        ("top_p", "top_p", "Same meaning"),
        ("frequency_penalty", "frequency_penalty", "Same meaning"),
        ("presence_penalty", "presence_penalty", "Same meaning"),
        ("stream", "stream", "Same meaning"),
        ("seed", "seed", "Random seed for reproducibility; Venice requires > 0"),
    ]

    for old_param, new_param, description in param_mappings:
        indicator = "→" if old_param == new_param else "⚠️"
        print(f"   {indicator} {old_param:20s} → {new_param:25s} | {description}")

    print("\n💡 Notable Differences:")
    print("   • max_tokens → max_completion_tokens (max_tokens raises a TypeError)")
    print("   • Messages are typed Pydantic models")
    print("   • VeniceClient is async; SyncVeniceClient is the blocking equivalent")
    print("   • Venice adds its own system prompt by default, which shows up in")
    print("     usage.prompt_tokens; pass venice_parameters=")
    print("     {'include_venice_system_prompt': False} to send only your messages")


async def practical_migration_example(client: VeniceClient) -> bool:
    """Run the migrated function live and inspect the real response."""
    print("\n🚀 Complete Migration Example")
    print("-" * 50)

    print("\n📜 BEFORE (OpenAI):")
    _print_block(
        [
            "import os",
            "",
            "from openai import OpenAI",
            "",
            "def get_response(prompt: str) -> str:",
            "    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))",
            "    response = client.chat.completions.create(",
            "        model=OPENAI_MODEL,",
            "        messages=[",
            "            {'role': 'system', 'content': 'You are helpful.'},",
            "            {'role': 'user', 'content': prompt},",
            "        ],",
            "        max_tokens=200,",
            "    )",
            "    return response.choices[0].message.content",
        ]
    )

    print("\n✅ AFTER (Venice AI) — the function this script is about to run:")
    _print_block(
        [
            "from venice_ai import VeniceClient",
            "from venice_ai.types.api import ChatCompletionResponse, SystemMessage, UserMessage",
            "from venice_ai.types.api.requests import VeniceParameters",
            "",
            f"MAX_COMPLETION_TOKENS = {MAX_COMPLETION_TOKENS}",
            "",
            "",
            *inspect.getsource(get_response).rstrip().splitlines(),
        ]
    )
    print("   # call it with: asyncio.run(...) around an `async with VeniceClient()` block")

    try:
        response = await get_response(client, "What is the capital of France?")
    except (VeniceError, ValueError) as e:
        print(f"\n❌ Live migration call failed: {e}")
        return False

    if not response.choices:
        print("\n❌ The response carried no choices.")
        return False
    choice = response.choices[0]
    content = (response.text or "").strip()

    print("\n📦 The real response has the OpenAI shape:")
    print(f"   response.id:                          {response.id}")
    print(f"   response.model:                       {response.model}")
    print(f"   response.choices[0].message.role:     {choice.message.role}")
    print(f"   response.choices[0].message.content:  {content}")
    print(f"   response.choices[0].finish_reason:    {choice.finish_reason}")
    if response.usage is not None:
        usage = response.usage
        print(
            f"   response.usage:                       prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )

    if choice.finish_reason != "stop":
        print(f"\n❌ Expected finish_reason 'stop', got {choice.finish_reason!r}.")
        return False
    # Some backends report "stop" even when the reply ran into the cap, so the
    # completion count is checked too; with no usage to check, completeness
    # cannot be shown.
    if response.usage is None or response.usage.completion_tokens >= MAX_COMPLETION_TOKENS:
        print(
            "\n❌ The reply used the whole token budget or reported no usage, so it may be cut off."
        )
        return False
    if "paris" not in content.lower():
        print("\n❌ The reply does not answer the question (expected Paris).")
        return False
    return True


def migration_checklist() -> None:
    """Provide a migration checklist for developers."""
    print("\n✅ Migration Checklist")
    print("-" * 50)

    checklist_items = [
        ("Install Venice AI SDK", "pip install venice-py"),
        ("Update imports", "from venice_ai import VeniceClient  (or SyncVeniceClient)"),
        ("Set up API key", "export VENICE_API_KEY='your-key'"),
        ("Update message format", "SystemMessage / UserMessage from venice_ai.types.api"),
        ("Choose models", f"{RESOLVE_CALL} and the other resolve_*() helpers"),
        ("Update parameters", "max_tokens → max_completion_tokens"),
        (
            "Check for truncation",
            "finish_reason 'length', or completion tokens at the cap: some models report a "
            "cut-off reply as 'stop'",
        ),
        ("Update error handling", "Catch venice_ai.exceptions (AuthenticationError, ...)"),
        ("Performance testing", "Verify latency and throughput"),
    ]

    print("\n📋 Step-by-Step Migration:")
    for i, (step, details) in enumerate(checklist_items, 1):
        print(f"\n{i}. {step}")
        print(f"   → {details}")

    print("\n🔍 Common Issues & Solutions:")
    print("\n   ❓ Model not found")
    print("   → Pick the model with resolve_chat(...) or list_traits() instead of a fixed name")
    print("\n   ❓ Reply cut off or empty")
    print("   → Reasoning models spend tokens thinking: pass exclude_reasoning=True to")
    print("     resolve_chat() for direct answers, or raise max_completion_tokens")
    print("\n   ❓ Sync vs Async")
    print("   → Use SyncVeniceClient for blocking code, VeniceClient with await otherwise")


async def main() -> int:
    """Run all compatibility and migration examples.

    Returns ``0`` when the chat migration ran and no live section failed,
    ``1`` on any failure, and ``77`` when no chat model fits.
    """
    print("🚀 Venice AI API Compatibility & Migration Guide")
    print("=" * 60)

    async with VeniceClient() as client:
        try:
            text_compat = (await client.models.list_compatibility(type="text")).data
            resolved_chat = await client.models.resolve_chat(
                prefer="cheapest", exclude_reasoning=True
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no chat model in the catalog answers directly ({e})")
            return EXIT_SKIPPED
        except VeniceError as e:
            print(f"❌ Could not load the compatibility map or resolve a chat model: {e}")
            return 1

        # Each live section returns True (passed), False (failed) or None (skipped).
        results: list[tuple[str, bool | None]] = [
            ("chat_migration", await chat_migration(text_compat, resolved_chat)),
            ("embedding_migration", await embedding_migration(client)),
        ]
        parameter_mapping_guide()
        results.append(("practical_migration_example", await practical_migration_example(client)))
        migration_checklist()

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} live sections failed: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"\n✅ Chat migration verified; skipped: {', '.join(skipped)}")
        return 0

    print("\n✨ Migration guide completed!")
    print("\n💡 Key Takeaways:")
    print("   - The request and response shapes follow the OpenAI chat format")
    print("   - Choose models with resolve_*(); aliases exist only for old code")
    print("   - Typed Pydantic messages and responses")
    print("   - Async by default, with SyncVeniceClient for blocking code")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
