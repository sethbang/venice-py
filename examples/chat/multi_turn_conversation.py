#!/usr/bin/env python3
"""
Venice AI SDK - Multi-turn Conversation
========================================

This example demonstrates how to maintain context across multiple conversation turns.
Learn how to build conversational AI that remembers previous interactions.

Every reply is checked before it goes into the history: a reply cut off at
``max_completion_tokens`` would poison every later turn, so the section stops
and the script exits non-zero instead. ``finish_reason == "length"`` is the
usual signal, but some models report a cut-off reply as ``"stop"``, so the
completion-token count is compared with the cap as well.

Each conversation sets its own system message, so every request leaves
Venice's own system prompt out (``include_venice_system_prompt=False``).
"""

import asyncio
import sys
from collections import Counter

from venice_ai import Conversation, NoMatchingModelError, VeniceClient
from venice_ai.types.api import (
    AssistantMessage,
    ChatCompletionResponse,
    SystemMessage,
    UserMessage,
)
from venice_ai.types.api.requests import VeniceParameters

NO_VENICE_PROMPT = VeniceParameters(include_venice_system_prompt=False)


def reply_ok(response: ChatCompletionResponse, cap: int) -> bool:
    """Return ``False`` (and say why) if a reply is truncated or empty.

    ``cap`` is the request's ``max_completion_tokens``.
    """
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    if finish_reason == "length" or used is None or used >= cap:
        print(
            "   ❌ Reply may be cut off at max_completion_tokens "
            f"(finish_reason={finish_reason}, {used} of {cap} completion tokens)"
        )
        return False
    if not (response.text or "").strip():
        print("   ❌ Model returned no visible reply")
        return False
    return True


async def simple_conversation(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate a simple multi-turn conversation with context preservation.

    Uses :class:`venice_ai.Conversation` — a thin wrapper that manages the
    message list and exposes ``add_user`` / ``add_response`` for the
    canonical loop. For variations (sliding-window history, mid-conversation
    context reset, etc.) see the other functions in this file.
    """
    print("💬 Simple Multi-turn Conversation")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    # System message sets the context
    conv = Conversation(
        system="You are a helpful assistant. Keep your responses brief and friendly."
    )

    # Turns 2 and 3 only make sense if the model remembers turn 1.
    conversation_turns = [
        "I'm learning Python. Name one built-in data structure that's good for lookups by key.",
        "In two or three sentences, why is it fast for that?",
        "Show me a three-line code example that uses it.",
    ]

    for turn_num, user_input in enumerate(conversation_turns, 1):
        print(f"\n🔄 Turn {turn_num}")
        print(f"👤 User: {user_input}")

        conv.add_user(user_input)

        response = await client.chat.completions.create(
            model=chat_model,
            messages=conv.messages,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=1000,
            temperature=0.7,
        )

        print(f"🤖 Assistant: {response.text or ''}")
        if not reply_ok(response, 1000):
            return False

        # add_response unwraps the choice and appends an AssistantMessage.
        conv.add_response(response)

    # Count what is actually in the history rather than what we meant to add.
    roles = Counter(type(message).__name__ for message in conv.messages)
    print(f"\n📝 Conversation history: {len(conv.messages)} messages")
    print(f"   System messages: {roles['SystemMessage']}")
    print(f"   User messages: {roles['UserMessage']}")
    print(f"   Assistant messages: {roles['AssistantMessage']}")
    expected_turns = len(conversation_turns)
    if roles["UserMessage"] != expected_turns or roles["AssistantMessage"] != expected_turns:
        print("   ❌ History does not hold one user and one assistant message per turn")
        return False
    return True


async def conversation_with_personality(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate conversation with a persona set by the system message."""
    print("\n🎭 Conversation with Personality")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    # Initialize with personality-defining system message
    messages: list[SystemMessage | UserMessage | AssistantMessage] = [
        SystemMessage(
            content=(
                "You are a pirate captain assistant. Speak like a pirate and make nautical "
                "references. Answer in at most three sentences."
            ),
            name="Captain",
        )
    ]

    # Ask questions about different topics
    questions = [
        "How do I learn programming?",
        "What's a good way to stay focused while studying?",
    ]

    for question in questions:
        print(f"\n👤 User: {question}")

        messages.append(UserMessage(content=question))

        response = await client.chat.completions.create(
            model=chat_model,
            messages=messages,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=800,
            temperature=0.9,  # Higher temperature for more creative responses
        )

        print(f"🏴‍☠️ Pirate Assistant: {response.text or ''}")
        if not reply_ok(response, 800):
            return False

        messages.append(AssistantMessage.from_response(response))
    return True


async def context_window_management(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate managing context window by limiting conversation history."""
    print("\n🪟 Context Window Management")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    # Short numeric answers keep the demo focused on what the model can still see.
    system_message = SystemMessage(
        content="You are a math tutor. Reply with just the resulting number, nothing else."
    )

    # Keep only last N conversation turns (sliding window)
    MAX_HISTORY_PAIRS = 2  # Keep last 2 user-assistant pairs
    conversation_history: list[UserMessage | AssistantMessage] = []

    # Series of math questions. Turns 2 and 3 only work if the previous answer
    # is still in the window; by turn 4 the first question has been trimmed.
    questions = [
        "What is 15 + 27?",
        "Multiply the answer by 2.",  # Refers to previous
        "Now divide that by 3.",  # Refers to previous
        "What exactly was my first question?",  # Turn 1 has left the window by now
    ]
    expected_numbers = ["42", "84", "28"]
    replies: list[str] = []

    for turn_num, question in enumerate(questions, 1):
        print(f"\n🔄 Turn {turn_num}")
        print(f"👤 User: {question}")

        # Build messages with system + limited history
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [system_message]
        messages.extend(conversation_history)
        messages.append(UserMessage(content=question))
        visible = [m.content for m in conversation_history if isinstance(m, UserMessage)]
        print(f"   (model can see earlier questions: {visible})")

        response = await client.chat.completions.create(
            model=chat_model,
            messages=messages,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=100,
            temperature=0.0,
        )

        print(f"🤖 Assistant: {response.text or ''}")
        if not reply_ok(response, 100):
            return False
        replies.append(response.text or "")

        # Add to conversation history
        conversation_history.append(UserMessage(content=question))
        conversation_history.append(AssistantMessage.from_response(response))

        # Trim history to maintain window size (keep last N pairs)
        # Each pair = 2 messages (user + assistant)
        max_messages = MAX_HISTORY_PAIRS * 2
        if len(conversation_history) > max_messages:
            conversation_history = conversation_history[-max_messages:]

        print(
            f"📊 History size: {len(conversation_history)} messages ({len(conversation_history) // 2} pairs)"
        )
    print(
        "\n💡 By turn 4 the first question has been trimmed from the window, so the model "
        "can only see turns 2 and 3."
    )

    ok = True
    for turn_num, (reply, number) in enumerate(zip(replies, expected_numbers, strict=False), 1):
        if number not in reply:
            print(f"   ❌ Turn {turn_num} should answer {number}; the chain lost its context")
            ok = False
    # The trimmed question can't be quoted back; seeing it would mean the window
    # did not actually drop turn 1.
    first_question_seen = all(n in replies[3] for n in ("15", "27"))
    if first_question_seen:
        print("   ❌ Turn 4 quoted the first question, which should have left the window")
        ok = False
    if ok:
        print("   ✅ Each step used the previous answer, and turn 1 was out of view by turn 4")
    return ok


async def conversation_with_context_reset(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate resetting conversation context, and show that it took effect.

    The first conversation mentions a detail the model cannot guess (a team
    name). Before the reset the model can recall it; after the reset it cannot.
    """
    print("\n🔄 Conversation with Context Reset")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    team_name = "Marigold"

    async def ask(
        messages: list[SystemMessage | UserMessage | AssistantMessage], question: str, label: str
    ) -> str | None:
        """Send one turn, print it, and append the reply; ``None`` on a bad reply."""
        print(f"👤 User: {question}")
        messages.append(UserMessage(content=question))
        response = await client.chat.completions.create(
            model=chat_model,
            messages=messages,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=800,
            temperature=0.3,
        )
        print(f"🤖 {label}: {response.text or ''}\n")
        if not reply_ok(response, 800):
            return None
        messages.append(AssistantMessage.from_response(response))
        return response.text or ""

    # First conversation topic
    print("\n📚 Topic 1: Science")
    messages: list[SystemMessage | UserMessage | AssistantMessage] = [
        SystemMessage(content="You are a science teacher. Answer in two or three sentences.")
    ]
    science_questions = [
        f"My lab group is called Team {team_name}. What is photosynthesis?",
        "In two sentences, why is it important?",
        "What is my lab group called?",
    ]
    reply = None
    for question in science_questions:
        reply = await ask(messages, question, "Science Teacher")
        if reply is None:
            return False
    recalled_before = team_name.lower() in (reply or "").lower()
    print(f"   Before the reset the model recalls the team name: {recalled_before}")

    # Reset context and switch topics: start a fresh message list.
    print("\n🔄 Resetting context and switching topics...\n")
    print("🎨 Topic 2: Art")
    messages = [
        SystemMessage(
            content=(
                "You are an art historian. Answer in two or three sentences. If you don't "
                "know something about the user, say so."
            )
        )
    ]
    art_questions = [
        "Who painted the Mona Lisa?",
        # "their" only resolves through the previous turn of the new conversation.
        "What was special about their technique?",
        # Only the discarded conversation contained this.
        "What is my lab group called?",
    ]
    for question in art_questions:
        reply = await ask(messages, question, "Art Historian")
        if reply is None:
            return False
    recalled_after = team_name.lower() in (reply or "").lower()
    print(f"   After the reset the model recalls the team name: {recalled_after}")

    if not recalled_before:
        print("❌ The model could not recall the team name even with full context")
        return False
    if recalled_after:
        print("❌ The team name survived the reset")
        return False
    print("✅ Reset confirmed: the team name was available before the reset and gone after it")
    return True


async def interactive_conversation_example(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate pattern for interactive user input (simulated)."""
    print("\n💻 Interactive Conversation Pattern")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    # Initialize conversation
    messages: list[SystemMessage | UserMessage | AssistantMessage] = [
        SystemMessage(
            content=(
                "You are a helpful coding assistant. Give brief, practical advice with at most "
                "one short code snippet."
            )
        )
    ]

    # Simulated user inputs (in real app, these would come from input())
    simulated_inputs = [
        "In a few sentences, how do I read a file in Python?",
        "exit",  # Exit command
    ]

    print("💡 Type 'exit' to end the conversation (simulated)\n")

    for user_input in simulated_inputs:
        # Simulate user input
        print(f"👤 User: {user_input}")

        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        # Add user message
        messages.append(UserMessage(content=user_input))

        # Get response
        response = await client.chat.completions.create(
            model=chat_model,
            messages=messages,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=1000,
            temperature=0.7,
        )

        # Display response
        print(f"🤖 Assistant: {response.text or ''}\n")
        if not reply_ok(response, 1000):
            return False

        # Add to history
        messages.append(AssistantMessage.from_response(response))
    return True


async def main() -> int:
    """Run all multi-turn conversation examples; return a process exit code."""
    print("🚀 Venice AI Multi-turn Conversation Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        # The cheapest model that answers directly, so short replies are not
        # spent on hidden reasoning.
        try:
            chat_model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no non-reasoning chat model ({e})")
            return 77

        results: list[tuple[str, bool]] = [
            ("simple_conversation", await simple_conversation(client, chat_model)),
            (
                "conversation_with_personality",
                await conversation_with_personality(client, chat_model),
            ),
            ("context_window_management", await context_window_management(client, chat_model)),
            (
                "conversation_with_context_reset",
                await conversation_with_context_reset(client, chat_model),
            ),
            (
                "interactive_conversation_example",
                await interactive_conversation_example(client, chat_model),
            ),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print("\n✨ Multi-turn conversation examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Maintaining conversation history")
    print("   - Context preservation across turns")
    print("   - System messages for personality")
    print("   - Managing context window size")
    print("   - Resetting conversation context")
    print("   - Interactive conversation patterns")
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
