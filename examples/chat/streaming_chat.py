#!/usr/bin/env python3
"""
Venice AI SDK - Streaming Chat Completions
==========================================

This example demonstrates streaming chat completions using the SDK's high-level
streaming API. Learn how to handle real-time token streaming, track usage, and
implement various streaming patterns.

Features Demonstrated:
    - Manual chunk iteration (``async for chunk in stream``)
    - collect_with_deltas(): live deltas plus the assembled final response
    - collect(): only the assembled final response
    - Usage on streams (``stream()`` always requests it, no ``stream_options`` needed)
    - Concurrent streaming
    - Multi-turn streaming conversations
    - Typed error handling during streams, and detecting truncated streams

``text_deltas()`` is the lightest option when you only display text: it yields
the text and nothing else, so it cannot tell you whether the answer finished.
Every section below checks for truncation and treats an answer that hit
``max_completion_tokens`` as a failure, except the one that deliberately
demonstrates truncation. ``finish_reason == "length"`` is the usual signal,
but some models report a cut-off answer as ``"stop"``, so the completion-token
count from the final chunk's usage is compared with the cap as well.
"""

import asyncio
import sys
import time

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import AuthenticationError, NotFoundError
from venice_ai.types.api import (
    AssistantMessage,
    ChatCompletionResponse,
    SystemMessage,
    UserMessage,
)
from venice_ai.types.api.requests import VeniceParameters

# Our own messages carry every instruction these requests need, so Venice's
# system prompt is left out.
NO_VENICE_PROMPT = VeniceParameters(include_venice_system_prompt=False)


def cut_off(finish_reason: str | None, completion_tokens: int | None, cap: int) -> bool:
    """True if an answer stopped at the cap, or if usage is missing so that can't be ruled out."""
    return finish_reason == "length" or completion_tokens is None or completion_tokens >= cap


def finish_ok(response: ChatCompletionResponse | None, cap: int) -> bool:
    """Print ``finish_reason`` for a streamed response; ``False`` if unusable.

    ``cap`` is the request's ``max_completion_tokens``.
    """
    if response is None:
        print("   ❌ Stream ended without a final response")
        return False
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    print(f"   ↳ finish_reason: {finish_reason} ({used} of {cap} completion tokens)")
    if cut_off(finish_reason, used, cap):
        print("   ❌ Answer may be cut off at max_completion_tokens")
        return False
    if not (response.text or "").strip():
        print("   ❌ Stream produced no visible text")
        return False
    return True


# =============================================================================
# Streaming Examples
# =============================================================================


async def manual_chunk_iteration(client: VeniceClient, chat_model: str) -> bool:
    """Iterate raw ``ChatCompletionChunk`` objects yourself.

    Each chunk carries a text delta (``chunk.text``); the last content chunk
    carries ``finish_reason`` and the final chunk carries ``usage``.
    """
    print("🌊 Manual Chunk Iteration")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    print("\n🤖 Assistant (streaming): ", end="", flush=True)

    stream = await client.chat.completions.stream(
        model=chat_model,
        messages=[
            UserMessage(
                content="Write a short story about a robot learning to paint. Make it exactly 3 sentences."
            )
        ],
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=300,
        temperature=0.8,
    )

    chunk_count = 0
    text = ""
    finish_reason: str | None = None
    usage = None
    async with stream:
        async for chunk in stream:
            chunk_count += 1
            print(chunk.text, end="", flush=True)
            text += chunk.text
            if chunk.choices and chunk.choices[0].finish_reason is not None:
                finish_reason = chunk.choices[0].finish_reason
            if chunk.usage is not None:
                usage = chunk.usage
    print()

    print(f"\n   📦 {chunk_count} chunks, finish_reason: {finish_reason}")
    if usage is not None:
        print(f"   📊 Usage from the final chunk: {usage.total_tokens} total tokens")
    if cut_off(finish_reason, usage.completion_tokens if usage else None, 300):
        print("   ❌ Answer may be cut off at max_completion_tokens")
        return False
    if not text.strip() or finish_reason is None:
        print("   ❌ Stream did not complete with an answer")
        return False
    return True


async def streaming_with_collect(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate ``collect_with_deltas()`` — live deltas and the final response.

    ``ChatStream.collect_with_deltas()`` yields each text delta as it arrives
    AND populates ``stream.final_response`` once iteration completes — one
    consumption gets you both signals, no duplicate request needed.
    (For display-only streaming use ``text_deltas()``; for "give me only the
    final response, don't render deltas" use ``collect()``.)
    """
    print("\n📦 Streaming with collect_with_deltas()")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    messages: list[SystemMessage | UserMessage] = [
        SystemMessage(content="You are a helpful assistant that provides concise answers."),
        UserMessage(content="What are the three primary colors of light? Answer in one sentence."),
    ]

    print("\n🤖 Assistant: ", end="", flush=True)
    # stream() always asks the server for usage on the final chunk, so no
    # stream_options argument is needed to get token counts.
    stream = await client.chat.completions.stream(
        model=chat_model,
        messages=messages,
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=200,
        temperature=0.3,
    )
    async with stream:
        async for text in stream.collect_with_deltas():
            print(text, end="", flush=True)
    print()

    # final_response is populated once iteration completes.
    response = stream.final_response
    ok = finish_ok(response, 200)
    if response is not None and response.usage:
        print("\n📊 Token Usage:")
        print(f"   Prompt tokens: {response.usage.prompt_tokens}")
        print(f"   Completion tokens: {response.usage.completion_tokens}")
        print(f"   Total tokens: {response.usage.total_tokens}")
    return ok


async def concurrent_streams(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate handling multiple concurrent streams."""
    print("\n🔀 Concurrent Streaming")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    prompts = [
        "Tell me a short joke about programming.",
        "Give me one fun fact about space, in one sentence.",
    ]

    async def process_stream(prompt: str) -> tuple[ChatCompletionResponse, float]:
        """Consume a single stream and return its response and wall time."""
        wall_start = time.perf_counter()
        stream = await client.chat.completions.stream(
            model=chat_model,
            messages=[UserMessage(content=prompt)],
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=300,
            temperature=0.8,
        )
        async with stream:
            response = await stream.collect()
        return response, time.perf_counter() - wall_start

    print("\n🚀 Starting concurrent streams...")

    start_time = time.perf_counter()
    # return_exceptions=True keeps one failed stream from cancelling the others;
    # every exception in the results still counts as a failure below.
    results = await asyncio.gather(*(process_stream(p) for p in prompts), return_exceptions=True)
    total_time = time.perf_counter() - start_time

    ok = True
    for index, result in enumerate(results):
        print(f"\n📝 Stream {index + 1} - {prompts[index]}")
        if isinstance(result, BaseException):
            print(f"   ❌ {type(result).__name__}: {result}")
            ok = False
            continue
        response, wall_time = result
        clean_content = (response.text or "").replace("\n", " ").strip()
        print(f"   Response: {clean_content}")
        print(f"   Duration: {wall_time:.3f}s")
        ok = finish_ok(response, 300) and ok

    print(f"\n⏱️ Total wall time for {len(prompts)} concurrent streams: {total_time:.3f}s")
    return ok


async def error_handling_in_streams(client: VeniceClient, chat_model: str) -> bool:
    """Trigger real, typed errors during streaming and detect a truncated stream.

    Each case states the outcome it expects; the section fails if any case
    behaves differently (for example, if no error is raised at all).
    """
    print("\n⚠️ Error Handling in Streams")
    print("-" * 40)

    ok = True

    # Case 1: a model ID that does not exist -> NotFoundError (HTTP 404).
    print("\n🧪 Test: unknown model")
    try:
        stream = await client.chat.completions.stream(
            model="this-model-does-not-exist",
            messages=[UserMessage(content="Say hello")],
            max_completion_tokens=20,
        )
        async with stream:
            await stream.collect()
        print("   ❌ Expected NotFoundError, but the stream succeeded")
        ok = False
    except NotFoundError as e:
        print(f"   ✅ Caught {type(e).__name__} (HTTP {e.status_code})")

    # Case 2: an invalid API key on an authenticated endpoint -> AuthenticationError (HTTP 401).
    print("\n🧪 Test: invalid API key")
    try:
        async with VeniceClient(api_key="invalid-api-key-for-demo") as bad_client:
            stream = await bad_client.chat.completions.stream(
                model=chat_model,
                messages=[UserMessage(content="Say hello")],
                max_completion_tokens=20,
            )
            async with stream:
                await stream.collect()
        print("   ❌ Expected AuthenticationError, but the stream succeeded")
        ok = False
    except AuthenticationError as e:
        print(f"   ✅ Caught {type(e).__name__} (HTTP {e.status_code})")

    # Case 3: a deliberately tiny token cap. A truncated stream raises nothing.
    # The signals are finish_reason == "length" and, because some models
    # report a cut-off answer as "stop", a completion count that reached the cap.
    truncation_cap = 20
    print(f"\n🧪 Test: truncation (max_completion_tokens={truncation_cap} for a long answer)")
    stream = await client.chat.completions.stream(
        model=chat_model,
        messages=[UserMessage(content="Count from 1 to 1000, separated by commas.")],
        venice_parameters=NO_VENICE_PROMPT,
        max_completion_tokens=truncation_cap,
        temperature=0.0,
    )
    async with stream:
        response = await stream.collect()
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    print(f"   Content: {(response.text or '').strip()}")
    print(f"   Finish reason: {finish_reason}, completion tokens: {used} of {truncation_cap}")
    if finish_reason == "length" or (used is not None and used >= truncation_cap):
        print("   ✅ Detected truncation: this answer is incomplete and must not be used as-is")
        if finish_reason != "length":
            print(f"      (finish_reason said {finish_reason!r}; the token count showed the cut)")
    else:
        print("   ❌ Expected a truncated answer: finish_reason='length' or the cap reached")
        ok = False

    return ok


async def multi_turn_streaming(client: VeniceClient, chat_model: str) -> bool:
    """Demonstrate multi-turn conversation with streaming."""
    print("\n💬 Multi-turn Streaming Conversation")
    print("-" * 40)
    print(f"📍 Using model: {chat_model}")

    # Initialize conversation
    messages: list[SystemMessage | UserMessage | AssistantMessage] = [
        SystemMessage(
            content="You are a helpful tutor teaching about planets. Answer in one or two sentences."
        )
    ]

    # Conversation turns; "it" and "the second largest" only make sense with history.
    user_inputs = [
        "What is the largest planet?",
        "How many moons does it have?",
        "What about the second largest planet?",
    ]

    for turn, user_input in enumerate(user_inputs, 1):
        print(f"\n🔄 Turn {turn}")
        print(f"👤 User: {user_input}")

        messages.append(UserMessage(content=user_input))

        print("🤖 Assistant: ", end="", flush=True)

        stream = await client.chat.completions.stream(
            model=chat_model,
            messages=messages,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=250,
            temperature=0.5,
        )
        async with stream:
            async for text in stream.collect_with_deltas():
                print(text, end="", flush=True)

        print()

        # A missing or truncated turn would corrupt the history the next turn
        # relies on, so stop the conversation instead of continuing without it.
        if not finish_ok(stream.final_response, 250):
            return False
        assert stream.final_response is not None
        messages.append(AssistantMessage.from_response(stream.final_response))

    print(f"\n📝 Final conversation length: {len(messages)} messages")
    return True


async def main() -> int:
    """Run all streaming examples; return a process exit code."""
    print("🚀 Venice AI Streaming Chat Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        # The cheapest model that answers directly: a reasoning model would spend
        # the small caps below (20 tokens in the truncation case) on hidden thinking.
        try:
            chat_model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no non-reasoning chat model ({e})")
            return 77

        results: list[tuple[str, bool]] = [
            ("manual_chunk_iteration", await manual_chunk_iteration(client, chat_model)),
            ("streaming_with_collect", await streaming_with_collect(client, chat_model)),
            ("concurrent_streams", await concurrent_streams(client, chat_model)),
            ("error_handling_in_streams", await error_handling_in_streams(client, chat_model)),
            ("multi_turn_streaming", await multi_turn_streaming(client, chat_model)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print("\n✨ Streaming examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Manual chunk iteration with finish_reason and usage")
    print("   - collect_with_deltas() for live display plus the final response")
    print("   - collect() for assembled responses")
    print("   - Concurrent stream processing with per-stream failure counting")
    print("   - Typed errors (NotFoundError, AuthenticationError) and truncation detection")
    print("   - Multi-turn conversations with streaming")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Stream interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
