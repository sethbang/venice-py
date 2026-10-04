"""
Venice AI SDK - Production Async Patterns

This example demonstrates production-ready asynchronous patterns for the Venice AI SDK:

1. Async context managers
2. Concurrent request handling
3. Timeouts and task cancellation
4. Error handling in async code
5. Streaming with async iteration
6. Bounded concurrency with client.gather()
7. Background tasks
8. Scoped retry policies with client.with_retries()

Every pattern checks its own result, including what each answer says: the
prompts ask for short, checkable answers, so a refusal or an off-topic reply
fails the pattern. The script exits non-zero if any pattern did not behave as
described.

The model is the cheapest one that answers directly
(``resolve_chat(prefer="cheapest", exclude_reasoning=True)``): a reasoning
model would spend these small completion budgets thinking. Every request
passes ``include_venice_system_prompt=False``, so only the prompts shown here
are billed.

Requirements:
    pip install venice-py
    export VENICE_API_KEY="your-api-key"
"""

import asyncio
import contextlib
import re
import sys
import time
from collections.abc import Awaitable
from typing import TypeVar

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    APIConnectionError,
    NoMatchingModelError,
    NotFoundError,
    VeniceError,
)
from venice_ai.middleware.retry import RetryOptions
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import UserMessage, VeniceParameters

T = TypeVar("T")

# A deliberately bogus model id used to make one request fail. It is a
# failure-trigger fixture, not a usage example: a model that does not exist
# cannot be resolved with client.models.resolve_*().
INVALID_MODEL = "venice-nonexistent-model"

# Bill only the prompts shown here, not the system prompt Venice adds by default.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

# The words Pattern 6 expects back for 1 to 5.
NUMBER_WORDS = ("one", "two", "three", "four", "five")


def check_answer(response: ChatCompletionResponse, label: str, expected: str) -> bool:
    """Print a one-line answer summary and report whether it answers the prompt.

    A ``finish_reason`` of ``"length"`` means the answer was cut off at
    ``max_completion_tokens``; an empty answer fails too, and so does one that
    does not contain *expected* as a whole word (case-insensitive), such as a
    refusal, or "none" when "one" is expected.
    """
    finish_reason = response.choices[0].finish_reason if response.choices else None
    text = (response.text or "").strip().replace("\n", " ")
    print(f"   {label}: {text}")
    print(f"      finish_reason={finish_reason}")
    if finish_reason == "length" or not text:
        print("      ❌ answer was truncated or empty")
        return False
    if not re.search(rf"\b{re.escape(expected)}\b", text, re.IGNORECASE):
        print(f"      ❌ answer does not contain {expected!r}")
        return False
    return True


async def resolve_direct_model(client: VeniceClient) -> str:
    """The cheapest chat model that answers without a reasoning phase."""
    return await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)


# =============================================================================
# Pattern 1: Async Context Managers
# =============================================================================


async def example_context_managers() -> bool:
    """
    Demonstrate proper resource management with async context managers.

    Best practices:
    - Always use async with for automatic cleanup
    - If you cannot, close the client in a finally block
    """
    print("=" * 60)
    print("Pattern 1: Async Context Managers")
    print("=" * 60)

    print("\n✅ Using async context manager for automatic cleanup:")
    async with VeniceClient() as client:
        model = await resolve_direct_model(client)
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Reply with the single word: hello")],
            max_completion_tokens=60,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        ok = check_answer(response, "Response", "hello")
    print("   ✓ Client closed on leaving the block")

    print("\n⚠️  Manual management (for comparison):")
    client = VeniceClient()
    try:
        model = await resolve_direct_model(client)
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Reply with the single word: goodbye")],
            max_completion_tokens=60,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        ok = check_answer(response, "Response", "goodbye") and ok
    finally:
        await client.close()
        print("   ✓ Client closed in finally")

    return ok


# =============================================================================
# Pattern 2: Concurrent Request Handling
# =============================================================================


async def example_concurrent_requests(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate efficient concurrent API request handling.

    Best practices:
    - Use asyncio.gather() for concurrent requests
    - Handle individual task failures with return_exceptions=True
    - Count the exceptions: gather returns them as list items, it does not raise
    """
    print("\n" + "=" * 60)
    print("Pattern 2: Concurrent Request Handling")
    print("=" * 60)

    # Each prompt has a short answer the check can recognise.
    prompts = [
        ("What is the capital of Japan? Answer in one word.", "Tokyo"),
        ("What is 12 times 12? Reply with the number only.", "144"),
        ("Which planet is known as the Red Planet? Answer in one word.", "Mars"),
    ]

    print(f"\n🚀 Executing {len(prompts)} requests concurrently...")
    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=150,
                venice_parameters=OWN_PROMPT_ONLY,
            )
            for prompt, _ in prompts
        ),
        return_exceptions=True,
    )
    print(f"   Wall time: {time.perf_counter() - started:.2f}s")

    ok = True
    for (prompt, expected), result in zip(prompts, results, strict=True):
        print(f"\n   Prompt: {prompt}")
        if isinstance(result, BaseException):
            print(f"   ❌ {type(result).__name__}: {result}")
            ok = False
            continue
        ok = check_answer(result, "Answer", expected) and ok
        if result.usage:
            print(
                f"      tokens: {result.usage.prompt_tokens} prompt + "
                f"{result.usage.completion_tokens} completion"
            )

    print(
        "\n   ℹ️  include_venice_system_prompt=False keeps Venice's own system prompt out,"
        "\n      so the prompt tokens above are only the question asked."
    )
    return ok


# =============================================================================
# Pattern 3: Timeouts and Task Cancellation
# =============================================================================


async def example_task_cancellation(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate timeouts and task cancellation, and show the client is still
    usable afterwards.

    Best practices:
    - Use asyncio.timeout() to bound how long you wait
    - Re-raise CancelledError unless you started the cancellation yourself
    - Clean up in finally blocks so cancellation never leaks resources
    """
    print("\n" + "=" * 60)
    print("Pattern 3: Timeouts & Task Cancellation")
    print("=" * 60)

    async def slow_request() -> ChatCompletionResponse:
        return await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Write a haiku about patience.")],
            max_completion_tokens=150,
            venice_parameters=OWN_PROMPT_ONLY,
        )

    ok = True

    # A 10ms budget is far below any real round trip, so this must time out.
    print("\n⏱️  Request with a 10ms timeout:")
    try:
        async with asyncio.timeout(0.01):
            await slow_request()
        print("   ❌ Request finished inside 10ms; the timeout never fired")
        ok = False
    except TimeoutError:
        # asyncio.timeout raises the built-in TimeoutError for its own
        # deadline. The client's HTTP timeout raises APITimeoutError instead.
        print("   ✅ TimeoutError raised; the in-flight request was cancelled")

    print("\n🚫 Cancelling a running task:")
    task = asyncio.create_task(slow_request())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
        print("   ❌ Task completed before it could be cancelled")
        ok = False
    except asyncio.CancelledError:
        # This coroutine requested the cancellation, so it is safe to absorb.
        print(f"   ✅ Task cancelled (task.cancelled() = {task.cancelled()})")

    print("\n🔁 Follow-up request on the same client:")
    response = await client.chat.completions.create(
        model=model,
        messages=[UserMessage(content="Reply with the single word: ready")],
        max_completion_tokens=60,
        venice_parameters=OWN_PROMPT_ONLY,
    )
    ok = check_answer(response, "Response", "ready") and ok
    return ok


# =============================================================================
# Pattern 4: Error Handling in Async Code
# =============================================================================


async def example_async_error_handling(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate per-task error handling with asyncio.gather().

    Best practices:
    - Catch specific exception classes
    - Use return_exceptions=True so one failure does not abort the batch
    - Inspect every result: a failure is a list item, not a raised exception
    """
    print("\n" + "=" * 60)
    print("Pattern 4: Async Error Handling")
    print("=" * 60)
    print("\n🛡️  One valid request and one for a model that does not exist:")

    results = await asyncio.gather(
        client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Reply with the single word: valid")],
            max_completion_tokens=60,
            venice_parameters=OWN_PROMPT_ONLY,
        ),
        client.chat.completions.create(
            model=INVALID_MODEL,
            messages=[UserMessage(content="This call should fail")],
            max_completion_tokens=60,
            venice_parameters=OWN_PROMPT_ONLY,
        ),
        return_exceptions=True,
    )

    valid, invalid = results
    ok = True

    if isinstance(valid, BaseException):
        print(f"❌ Valid task failed: {type(valid).__name__}: {valid}")
        ok = False
    else:
        print("✅ Task 1 succeeded")
        ok = check_answer(valid, "Response", "valid") and ok

    if isinstance(invalid, NotFoundError):
        print(f"✅ Task 2 failed as expected with {type(invalid).__name__}")
        print(f"   {invalid}")
    elif isinstance(invalid, BaseException):
        print(f"❌ Task 2 raised {type(invalid).__name__}, expected NotFoundError: {invalid}")
        ok = False
    else:
        print("❌ Task 2 succeeded against a model that does not exist")
        ok = False

    return ok


# =============================================================================
# Pattern 5: Streaming with Async Iteration
# =============================================================================


async def example_async_streaming(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate async streaming with proper iteration.

    Best practices:
    - Use async for to iterate over streams
    - Record finish_reason from the final chunk
    - Process chunks incrementally
    """
    print("\n" + "=" * 60)
    print("Pattern 5: Async Streaming")
    print("=" * 60)
    print("\n📡 Streaming response:")

    stream = await client.chat.completions.create(
        model=model,
        messages=[UserMessage(content="Count from 1 to 5, separated by spaces.")],
        max_completion_tokens=100,
        venice_parameters=OWN_PROMPT_ONLY,
        stream=True,
    )

    chunks = 0
    text = ""
    finish_reason = None
    print("   ", end="", flush=True)
    async for chunk in stream:
        if chunk.text:
            chunks += 1
            text += chunk.text
            print(chunk.text, end="", flush=True)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
    print()
    print(f"   {chunks} content chunks, finish_reason={finish_reason}")

    if finish_reason == "length" or re.findall(r"\d+", text)[:5] != ["1", "2", "3", "4", "5"]:
        print("   ❌ Stream was truncated or did not count 1 to 5 in order")
        return False
    return True


# =============================================================================
# Pattern 6: Bounded Concurrency with client.gather()
# =============================================================================


async def example_bounded_concurrency(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate ``client.gather()`` for bounded-concurrency request batching.

    Best practices:
    - Use ``client.gather(awaitables, max_concurrency=N)`` instead of
      hand-rolled ``asyncio.Semaphore`` + ``asyncio.gather()`` loops
    - It accepts awaitables across any modality (chat, image, embeddings…)
    - ``return_exceptions=True`` (the default) returns failures as list items,
      so count them
    """
    print("\n" + "=" * 60)
    print("Pattern 6: Bounded Concurrency with client.gather()")
    print("=" * 60)

    max_concurrency = 3
    in_flight = 0
    peak = 0

    async def tracked(awaitable: Awaitable[T]) -> T:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            return await awaitable
        finally:
            in_flight -= 1

    print(f"\n🚦 Executing 5 requests with max {max_concurrency} concurrent:")
    results = await client.gather(
        [
            tracked(
                client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content=f"Spell the number {i + 1} as an English word.")],
                    max_completion_tokens=60,
                    venice_parameters=OWN_PROMPT_ONLY,
                )
            )
            for i in range(5)
        ],
        max_concurrency=max_concurrency,
    )

    errors = [r for r in results if isinstance(r, BaseException)]
    ok = not errors
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            print(f"   ❌ Request {i + 1}: {type(result).__name__}: {result}")
        else:
            ok = check_answer(result, f"Request {i + 1}", NUMBER_WORDS[i]) and ok

    print(f"\n   {len(results) - len(errors)}/{len(results)} succeeded")
    print(f"   Peak requests in flight: {peak} (limit {max_concurrency})")
    if peak > max_concurrency:
        print("   ❌ Concurrency limit was exceeded")
        ok = False
    return ok


# =============================================================================
# Pattern 7: Background Tasks
# =============================================================================


async def example_background_tasks(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate running background work alongside a request.

    Best practices:
    - Use asyncio.create_task() for background work
    - Keep a reference to the task and cancel it when you are done
    - Await the cancelled task so its cleanup runs
    """
    print("\n" + "=" * 60)
    print("Pattern 7: Background Tasks")
    print("=" * 60)

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.1)
            ticks += 1

    print("\n🔄 Starting a 100ms heartbeat, then making a request...")
    bg_task = asyncio.create_task(heartbeat())
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                UserMessage(content="What colour is a clear daytime sky? Answer in one word.")
            ],
            max_completion_tokens=60,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        ticks_during_request = ticks
    finally:
        bg_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bg_task

    ok = check_answer(response, "Main request", "blue")
    print(f"   Heartbeat ticked {ticks_during_request} times while the request was in flight")
    print(f"   Background task cancelled: {bg_task.cancelled()}")
    if ticks_during_request == 0:
        print("   ❌ The background task never ran concurrently with the request")
        ok = False
    return ok


# =============================================================================
# Pattern 8: Scoped Retry Policies
# =============================================================================


async def example_retry_pattern(client: VeniceClient, model: str) -> bool:
    """
    Demonstrate ``client.with_retries()`` for a scoped retry policy.

    The SDK's retry middleware resends a request only when that is safe for
    its endpoint. A chat completion is resent after a connection failure, a
    502 or 503, and at most once after a 500; a 504 or a read timeout is not
    resent, because the model may already have produced (and billed) the
    answer. A failure to connect is retried for every endpoint, since the
    request never left the client. Client errors such as a 404 for an unknown
    model are never retried. ``on_retry`` receives the 0-based index of the
    attempt that failed.

    Best practices:
    - Configure retries with RetryOptions instead of hand-written loops
    - Use on_retry to log or count retry attempts
    - Do not retry 4xx errors
    """
    print("\n" + "=" * 60)
    print("Pattern 8: Scoped Retries with client.with_retries()")
    print("=" * 60)

    retries: list[tuple[int, float, str]] = []

    max_attempts = 3

    def on_retry(attempt: int, delay: float, error: Exception | None) -> None:
        reason = type(error).__name__ if error else "retryable status"
        retries.append((attempt, delay, reason))
        print(f"   🔄 Retry {attempt + 1}/{max_attempts} in {delay:.2f}s ({reason})")

    options = RetryOptions(
        max_attempts=max_attempts, base_delay=0.2, max_delay=2.0, on_retry=on_retry
    )

    ok = True
    async with client.with_retries(options):
        print("\n✅ Normal request inside the retry scope:")
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Reply with the single word: done")],
            max_completion_tokens=60,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        ok = check_answer(response, "Response", "done")
        print(f"      retries needed: {len(retries)}")

        print("\n🧪 Request for an unknown model inside the same scope:")
        retries_before = len(retries)
        try:
            await client.chat.completions.create(
                model=INVALID_MODEL,
                messages=[UserMessage(content="This call should fail")],
                max_completion_tokens=60,
                venice_parameters=OWN_PROMPT_ONLY,
            )
            print("   ❌ Request unexpectedly succeeded")
            ok = False
        except NotFoundError as e:
            retried = len(retries) - retries_before
            print(f"   ✅ {type(e).__name__} raised after {retried} retries (4xx is not retried)")
            if retried:
                ok = False

    # A transient failure: nothing listens on this local port, so every
    # attempt fails to connect and is retried until the budget runs out. The
    # client gets a placeholder key: the real one is never sent to a
    # non-Venice URL.
    print("\n🧪 Request to an unreachable host, with the same retry policy:")
    retries_before = len(retries)
    async with (
        VeniceClient(api_key="placeholder-key", base_url="http://127.0.0.1:9/api/v1") as offline,
        offline.with_retries(options),
    ):
        try:
            await offline.chat.completions.create(
                model=model,
                messages=[UserMessage(content="This call cannot connect")],
                max_completion_tokens=60,
                venice_parameters=OWN_PROMPT_ONLY,
            )
            print("   ❌ Request unexpectedly succeeded")
            ok = False
        except APIConnectionError as e:
            retried = len(retries) - retries_before
            print(f"   ✅ {type(e).__name__} raised after {retried} retries")
            if retried != max_attempts:
                print(f"   ❌ Expected {max_attempts} retries")
                ok = False

    return ok


# =============================================================================
# Best Practices Summary
# =============================================================================


def show_best_practices() -> None:
    """Display async programming best practices."""
    print("\n" + "=" * 60)
    print("Async Programming Best Practices")
    print("=" * 60)

    practices = [
        (
            "Resource Management",
            [
                "✅ Always use async with for VeniceClient",
                "✅ Ensure proper cleanup in finally blocks",
                "✅ Cancel background tasks on shutdown and await them",
            ],
        ),
        (
            "Concurrency",
            [
                "✅ Use asyncio.gather() for multiple requests",
                "✅ Bound concurrency with client.gather(max_concurrency=N)",
                "✅ Count the exceptions gather() returns as results",
            ],
        ),
        (
            "Error Handling",
            [
                "✅ Catch specific exception classes",
                "✅ Use client.with_retries(RetryOptions(...)) for transient failures",
                "✅ Log all errors with context",
            ],
        ),
        (
            "Streaming",
            [
                "✅ Use async for to iterate streams",
                "✅ Check finish_reason on the final chunk",
                "✅ Process chunks incrementally",
            ],
        ),
        (
            "Performance",
            [
                "✅ Reuse one client (and its connection pool) across requests",
                "✅ Batch similar requests",
                "✅ Bound every wait with asyncio.timeout()",
            ],
        ),
    ]

    for category, items in practices:
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   {item}")


# =============================================================================
# Main Example Runner
# =============================================================================


async def main() -> int:
    """Run all async pattern examples; return 0 only if every pattern succeeded."""
    print("=" * 60)
    print("Venice AI SDK - Production Async Patterns")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    results.append(("Context Managers", await example_context_managers()))

    # One shared client for the remaining patterns: it reuses a single
    # connection pool, which is what a long-running service should do.
    async with VeniceClient() as client:
        model = await resolve_direct_model(client)
        print(f"\nUsing model: {model}")
        results.append(("Concurrent Requests", await example_concurrent_requests(client, model)))
        results.append(("Timeouts & Cancellation", await example_task_cancellation(client, model)))
        results.append(("Async Error Handling", await example_async_error_handling(client, model)))
        results.append(("Async Streaming", await example_async_streaming(client, model)))
        results.append(("Bounded Concurrency", await example_bounded_concurrency(client, model)))
        results.append(("Background Tasks", await example_background_tasks(client, model)))
        results.append(("Scoped Retries", await example_retry_pattern(client, model)))

    show_best_practices()

    failed = [name for name, ok in results if not ok]
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(results) - len(failed)}/{len(results)} patterns succeeded")
        for name, ok in results:
            print(f"   {'✓' if ok else '✗'} {name}")
    else:
        print(f"✅ All {len(results)} async patterns succeeded")
    print("=" * 60)

    print("\n🔑 Key Takeaways:")
    print("   1. Always use async with for automatic resource cleanup")
    print("   2. Use client.gather() for bounded-concurrency batching")
    print("   3. Count the exceptions gather() returns instead of assuming success")
    print("   4. Use client.with_retries() for scoped retry policies")
    print("   5. Bound waits with asyncio.timeout() and cancel what you start")
    print("   6. Use async for to iterate over streams and check finish_reason")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except NoMatchingModelError as e:
        # The catalog has no model of the kind this example needs.
        print(f"SKIPPED: {e}")
        sys.exit(77)
    except VeniceError as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
