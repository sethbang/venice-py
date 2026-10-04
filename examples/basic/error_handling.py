#!/usr/bin/env python3
"""
Venice AI SDK - Error Handling Examples
=======================================

This example demonstrates comprehensive error handling patterns for the Venice AI SDK.
Every error section triggers the error it describes for real (against the live
API, an unreachable host or a local stand-in server), so you can see exactly
which exception class the SDK raises:

- ``AuthenticationError`` from a request made with a bad API key
- ``NotFoundError`` from a request for a model that does not exist
- ``APIConnectionError`` after the SDK's own retries against an unreachable host
- ``APITimeoutError`` when a request outlives the client's timeout, with or
  without a rate limiter attached (shown against a local stand-in server)
- Which requests the SDK resends after a server error: the retry policy is
  billing-aware, so a paid generation request is never resent once the server
  may have processed it (shown against a local stand-in server, so it costs
  nothing)
- A fallback chain that skips an unavailable model
- Logging request context when a call fails

Exit status: ``0`` when every section behaved as described, ``1`` when any
section failed, and ``77`` (with a ``SKIPPED:`` line) when the catalog has no
chat model that answers directly. In that case the live sections print
``Section skipped:``, the local stand-in sections still run, and a failure in
one of them still exits ``1``.
"""

import asyncio
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from itertools import pairwise

from aiohttp import web

from venice_ai import (
    NoMatchingModelError,
    RetryClass,
    RetryOptions,
    SimpleRateLimiter,
    VeniceClient,
    classify_request,
)
from venice_ai.exceptions import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    InvalidRequestError,
    ModelGoneError,
    NotFoundError,
    RateLimitError,
    VeniceError,
)
from venice_ai.types.api import ChatCompletionResponse, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# Plenty for a one-line answer from a model that answers directly.
MAX_TOKENS = 256

# Send only our own messages, without Venice's default system prompt.
NO_SYSTEM_PROMPT = VeniceParameters(include_venice_system_prompt=False)

# A port nothing listens on, used to show retries against an unreachable host.
UNREACHABLE_BASE_URL = "http://127.0.0.1:9/api/v1"

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77

# The timeout demo's client gives up after this many seconds; its stand-in
# server answers only after STALL_SECONDS.
DEMO_TIMEOUT_SECONDS = 0.5
STALL_SECONDS = 3.0


def cut_off(response: ChatCompletionResponse) -> bool:
    """``True`` when a reply may have stopped at the token cap.

    Some backends report ``finish_reason="stop"`` even when the reply ran into
    ``max_completion_tokens``, so a reply that used the whole budget counts as
    cut off as well, and so does one with no usage to check.
    """
    usage = response.usage
    if usage is None or usage.completion_tokens >= MAX_TOKENS:
        return True
    return bool(response.choices) and response.choices[0].finish_reason == "length"


async def cheapest_direct_model(client: VeniceClient) -> str:
    """The cheapest chat model that answers directly, without a reasoning phase.

    The cheapest model overall can be a reasoning model, and these demos want
    short, direct answers, hence ``exclude_reasoning=True``.
    """
    return await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)


def missing_model_id(real_model: str) -> str:
    """A model ID that cannot exist, derived from a real one (a typo)."""
    return f"{real_model}-typo"


async def basic_error_handling(chat_model: str) -> bool:
    """Happy path, then a deliberate AuthenticationError and NotFoundError.

    Returns ``True`` if the happy-path call produced a complete answer and each
    deliberate error surfaced as the expected exception class.
    """
    print("🛡️ Basic Error Handling")
    print("-" * 30)

    ok = True

    # --- Happy-path branch ---
    async with VeniceClient() as client:
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Say hello in one short sentence.")],
            max_completion_tokens=MAX_TOKENS,
            venice_parameters=NO_SYSTEM_PROMPT,
        )
        content = response.text or ""
        finish_reason = response.choices[0].finish_reason
        if cut_off(response) or not content.strip():
            print(f"❌ Happy path gave an unusable reply (finish_reason={finish_reason})")
            ok = False
        else:
            print(f"✅ Success ({chat_model}, finish_reason={finish_reason}): {content.strip()}")

    # --- AuthenticationError branch: deliberately use a bad key ---
    print("\n🔐 Triggering AuthenticationError (bad API key)...")
    async with VeniceClient(api_key="sk-invalid-key-for-demo") as bad_client:
        try:
            await bad_client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content="Hello!")],
                max_completion_tokens=10,
            )
            print("❌ Expected AuthenticationError but the request succeeded")
            ok = False
        except AuthenticationError as e:
            print(f"🔐 Authentication failed (as expected): status={e.status_code}, {e}")
            print("   ✓ This is how AuthenticationError surfaces from the SDK")

    # --- NotFoundError branch: ask for a model that doesn't exist ---
    print("\n🔍 Triggering NotFoundError (nonexistent model)...")
    async with VeniceClient() as client:
        try:
            await client.chat.completions.create(
                model=missing_model_id(chat_model),
                messages=[UserMessage(content="Hi")],
                max_completion_tokens=10,
            )
            print("❌ Expected NotFoundError but the request succeeded")
            ok = False
        except NotFoundError as e:
            print(f"🔍 Not found (as expected): status={e.status_code}, {e}")
            print("   ✓ This is how NotFoundError surfaces from the SDK")

    return ok


async def removed_model_lifecycle(current_model: str) -> bool:
    """Show how to handle a model that has left the catalog.

    A model's identifier moves through a lifecycle:

    1. **Active** — routable, returns ``200``.
    2. **Deprecated** — still routable; Venice auto-routes to a replacement and
       advertises the sunset via ``deprecation_info`` response headers.
    3. **Retired (410 Gone)** — no longer routable but still *recognised*; the
       SDK maps this to :class:`ModelGoneError`. This is the "migrate to a
       replacement" signal.
    4. **Removed (404 Not Found)** — dropped from the catalog entirely, now
       indistinguishable from a model that never existed; the SDK maps this to
       :class:`NotFoundError`.

    Which stage a stale ID is in drifts over time, so production code should
    handle both ``ModelGoneError`` and ``NotFoundError`` the same way: pick a
    current model. There is no way to know a retired (410) model ahead of time,
    so this run exercises stage 4 live; the 410 handler is identical in shape.

    Returns ``True`` if the stale ID surfaced as ModelGoneError or NotFoundError.
    """
    print("\n🪦 Removed-model lifecycle (410 Gone / 404 Not Found)")
    print("-" * 30)
    print("   ℹ️  Exercises the 404 stage live; 410 needs a recently retired model")

    async with VeniceClient() as client:
        stale_model = missing_model_id(current_model)
        try:
            await client.chat.completions.create(
                model=stale_model,
                messages=[UserMessage(content="Hi")],
                max_completion_tokens=10,
            )
            print(f"❌ Expected the request to fail for stale model '{stale_model}'")
            return False
        except ModelGoneError as e:
            print(f"🪦 Model gone (410): {e}")
        except NotFoundError as e:
            print(f"🔍 Model not found (404): {e}")

        print(
            "   💡 Migrate: client.models.resolve_chat(prefer='cheapest', exclude_reasoning=True)"
        )
        print(f"      currently returns {current_model}")
        return True


async def retry_with_backoff(chat_model: str) -> bool:
    """Show the SDK's built-in retries firing against an unreachable host.

    ``client.with_retries(RetryOptions(...))`` sets the retry policy for every
    call inside the block. Connection errors are retried with exponential
    backoff; once the attempts are used up the SDK raises
    :class:`APIConnectionError`. The ``on_retry`` hook records each retry so
    the backoff is visible.

    Returns ``True`` if the SDK retried the configured number of times and then
    raised ``APIConnectionError``.
    """
    print("\n🔄 Retry with Exponential Backoff")
    print("-" * 30)

    retries: list[tuple[int, float, str]] = []

    def on_retry(attempt: int, delay: float, error: Exception | None) -> None:
        retries.append((attempt, delay, type(error).__name__ if error else "status"))
        print(f"   ⏱️ Retry {attempt + 1}: waiting {delay:.2f}s after {retries[-1][2]}")

    options = RetryOptions(max_attempts=3, base_delay=0.25, on_retry=on_retry)

    print(f"🌐 Sending a request to an unreachable host ({UNREACHABLE_BASE_URL})")
    # A placeholder key: the real one is never sent to a non-Venice URL.
    async with VeniceClient(
        api_key="placeholder-key", base_url=UNREACHABLE_BASE_URL
    ) as offline_client:
        started = time.monotonic()
        try:
            async with offline_client.with_retries(options):
                await offline_client.chat.completions.create(
                    model=chat_model,
                    messages=[UserMessage(content="Count to 3")],
                    max_completion_tokens=50,
                )
            print("❌ Expected APIConnectionError but the request succeeded")
            return False
        except APIConnectionError as e:
            elapsed = time.monotonic() - started
            print(f"🌐 {type(e).__name__} after {len(retries)} retries in {elapsed:.2f}s: {e}")

    if len(retries) != options.max_attempts:
        print(f"❌ Expected {options.max_attempts} retries, saw {len(retries)}")
        return False
    delays = [d for _, d, _ in retries]
    if any(later <= earlier for earlier, later in pairwise(delays)):
        print(f"❌ Backoff did not grow between attempts: {[round(d, 2) for d in delays]}")
        return False
    print("   ✓ Backoff grew between attempts:", [round(d, 2) for d in delays])

    # 429 is not a RetryOptions status: the client's rate limiter owns it
    # (SimpleRateLimiter(max_retries=...)), so RetryOptions rejects it.
    try:
        RetryOptions(retry_status_codes={429, 503})
    except ValueError as e:
        print(f"   ✓ RetryOptions refuses 429: {e}")
        return True
    print("❌ RetryOptions accepted 429 in retry_status_codes")
    return False


async def request_timeout() -> bool:
    """Show a request that outlives the client's timeout.

    A local stand-in server takes longer to answer than the client waits, so
    every call times out. The SDK raises :class:`APITimeoutError` whether or
    not a rate limiter wraps the request. Its message warns that a request the
    server received may still have been processed and billed, which is why a
    timed-out chat call is not resent automatically.

    Returns ``True`` if both clients raised ``APITimeoutError`` after sending
    the request once.
    """
    print("\n⏳ Request timeout (local stand-in server, no charges)")
    print("-" * 30)

    attempts: Counter[str] = Counter()

    async def slow(request: web.Request) -> web.Response:
        attempts[request.path] += 1
        await asyncio.sleep(STALL_SECONDS)
        return web.json_response({"error": "answered too late"}, status=500)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", slow)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base_url = f"http://127.0.0.1:{runner.addresses[0][1]}/api/v1"

    ok = True
    try:
        for label, limiter in (
            ("no rate limiter", None),
            ("SimpleRateLimiter", SimpleRateLimiter()),
        ):
            attempts.clear()
            async with VeniceClient(
                api_key="sk-local-stand-in",
                base_url=base_url,
                timeout=DEMO_TIMEOUT_SECONDS,
                rate_limiter=limiter,
            ) as local:
                started = time.monotonic()
                try:
                    await local.chat.completions.create(
                        model="stand-in-model",
                        messages=[UserMessage(content="Hi")],
                        max_completion_tokens=10,
                    )
                    print(f"❌ {label}: expected APITimeoutError, the request succeeded")
                    ok = False
                    continue
                except APITimeoutError as e:
                    elapsed = time.monotonic() - started
                    sent = attempts["/api/v1/chat/completions"]
                    mark = "✓" if sent == 1 else "❌"
                    print(f"   {mark} {label}: APITimeoutError after {elapsed:.2f}s, sent {sent}x")
                    print(f"     {e}")
                    ok = ok and sent == 1
    finally:
        await runner.cleanup()
    return ok


async def billing_aware_retries() -> bool:
    """Show which requests the SDK resends after a server error.

    Venice bills a generation once it has been queued or run, and its paid
    endpoints accept no idempotency key, so resending a request the server
    already processed could charge twice. The SDK's retry policy therefore
    classifies every request (:func:`classify_request`) and retries by class:

    - ``IDEMPOTENT`` (GETs, free quote/retrieve calls): any transient failure.
    - ``INFERENCE`` (chat, responses, embeddings): connection failures, 502
      and 503, and a 500 at most ``max_inference_500_retries`` times.
    - ``PAID`` (image, video, music, speech and other generation POSTs): only
      when the connection never opened, or a 503 that cannot have followed
      processing. A 500 is surfaced at once.

    A local stand-in server answers every request with a 500 and counts the
    attempts, so the policy is visible without spending anything.

    Returns ``True`` if each class was attempted the number of times the
    client's own retry policy says it should be.
    """
    print("\n💳 Billing-aware retries (local stand-in server, no charges)")
    print("-" * 30)

    attempts: Counter[str] = Counter()

    async def always_500(request: web.Request) -> web.Response:
        attempts[request.path] += 1
        return web.json_response({"error": "simulated server error"}, status=500)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", always_500)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    try:
        async with VeniceClient(
            api_key="sk-local-stand-in", base_url=f"http://127.0.0.1:{port}/api/v1"
        ) as local:
            policy = local.retry_options
            if policy is None:
                print("❌ The client has no SDK retry policy installed")
                return False
            print(
                f"\n   Default policy: max_attempts={policy.max_attempts}, "
                f"base_delay={policy.base_delay}s, max_delay={policy.max_delay}s, "
                f"max_inference_500_retries={policy.max_inference_500_retries}"
            )

            calls = [
                (
                    "GET",
                    "/api/v1/api_keys/rate_limits",
                    RetryClass.IDEMPOTENT,
                    1 + policy.max_attempts,
                    lambda: local.api_keys.get_rate_limits(),
                ),
                (
                    "POST",
                    "/api/v1/chat/completions",
                    RetryClass.INFERENCE,
                    1 + min(policy.max_attempts, policy.max_inference_500_retries),
                    lambda: local.chat.completions.create(
                        model="stand-in-model",
                        messages=[UserMessage(content="Hi")],
                        max_completion_tokens=10,
                    ),
                ),
                (
                    "POST",
                    "/api/v1/image/generate",
                    RetryClass.PAID,
                    1,
                    lambda: local.image.create(model="stand-in-model", prompt="a lighthouse"),
                ),
            ]

            ok = True
            for method, path, request_class, expected, send in calls:
                # classify_request is the function the retry middleware uses.
                classified = classify_request(method, path, {})
                if classified is not request_class:
                    print(f"❌ {method} {path} classified {classified.value}, not {request_class}")
                    ok = False
                    continue
                try:
                    await send()
                    print(f"❌ {path}: expected InternalServerError, the request succeeded")
                    ok = False
                    continue
                except InternalServerError:
                    pass
                seen = attempts[path]
                mark = "✓" if seen == expected else "❌"
                print(
                    f"   {mark} {method:4} {path:30} {request_class.value:10} "
                    f"sent {seen}x (expected {expected}), then InternalServerError"
                )
                ok = ok and seen == expected
    finally:
        await runner.cleanup()

    if ok:
        print("   ✓ The paid request was sent once; only safe requests were resent")
    return ok


async def graceful_degradation(primary: str) -> bool:
    """Fall back through a chain of models until one gives a usable answer.

    The chain starts with a model that does not exist, so the fallback path
    runs every time. A reply only counts once it finished normally and
    actually answers the question.

    Returns ``True`` once a model in the chain answers correctly.
    """
    print("\n🎯 Graceful Degradation")
    print("-" * 30)

    async with VeniceClient() as client:
        models_to_try = [missing_model_id(primary), primary]

        for model in models_to_try:
            print(f"🔄 Trying model: {model}")
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content="What is 2+2? Reply with just the number.")],
                    max_completion_tokens=MAX_TOKENS,
                    venice_parameters=NO_SYSTEM_PROMPT,
                )
            except (NotFoundError, ModelGoneError):
                print("   ↪ Model not available, trying next...")
                continue
            except RateLimitError:
                print("   ↪ Model rate limited, trying next...")
                continue

            content = (response.text or "").strip()
            finish_reason = response.choices[0].finish_reason
            if cut_off(response) or not re.search(r"\b4\b", content):
                print(f"   ↪ Unusable reply (finish_reason={finish_reason}): {content!r}")
                continue

            print(f"✅ Answered by {model} (finish_reason={finish_reason})")
            print(f"📝 Response: {content}")
            return True

        print("❌ Every model in the chain failed")
        return False


async def error_context_handling(chat_model: str) -> bool:
    """Log the request context alongside a failure.

    A mistyped model name triggers a real ``NotFoundError``; the handler then
    records what was being attempted so the failure can be debugged later.

    Returns ``True`` if the error fired and its context was captured.
    """
    print("\n🎭 Error Context Handling")
    print("-" * 30)

    async with VeniceClient() as client:
        conversation_history = [
            UserMessage(content="What's your name?"),
        ]
        requested_model = missing_model_id(chat_model)

        try:
            await client.chat.completions.create(
                model=requested_model,
                messages=conversation_history,
                max_completion_tokens=50,
            )
            print("❌ Expected the mistyped model to fail")
            return False

        except VeniceError as e:
            print(f"📝 Caught {type(e).__name__} (expected); logging its context")

            error_context = {
                "model": requested_model,
                "message_count": len(conversation_history),
                "last_message": conversation_history[-1].content,
                "error_type": type(e).__name__,
                "status_code": getattr(e, "status_code", None),
                "error_message": str(e),
            }

            print("🔍 Error context for debugging:")
            for key, value in error_context.items():
                print(f"   {key}: {value}")

            # Provide helpful suggestions
            if isinstance(e, AuthenticationError):
                print("💡 Suggestion: Check your VENICE_API_KEY environment variable")
            elif isinstance(e, RateLimitError):
                print("💡 Suggestion: Reduce request frequency or upgrade your plan")
            elif isinstance(e, (NotFoundError, ModelGoneError)):
                print("💡 Suggestion: Pick a current model with client.models.resolve_chat()")
            elif isinstance(e, InvalidRequestError):
                print("💡 Suggestion: Check the request parameters")

            return isinstance(e, NotFoundError)


async def main() -> int:
    """Demonstrate comprehensive error handling patterns.

    Returns ``1`` if any section failed. Otherwise returns ``0`` when every
    section ran, and ``77`` when the catalog has no chat model to run the live
    sections on (the local stand-in sections still run first, so a failure in
    one of them is never hidden behind the skip).
    """
    print("🚀 Venice AI Error Handling Examples")
    print("=" * 50)

    # The live sections share one model: the cheapest that answers directly.
    # Resolving it is a free catalog read.
    chat_model: str | None
    no_model_reason = ""
    try:
        async with VeniceClient() as client:
            chat_model = await cheapest_direct_model(client)
    except NoMatchingModelError as e:
        chat_model = None
        no_model_reason = f"no chat model in the catalog answers directly ({e})"

    live_sections: list[tuple[str, Callable[[str], Awaitable[bool]]]] = [
        ("basic_error_handling", basic_error_handling),
        ("removed_model_lifecycle", removed_model_lifecycle),
        ("retry_with_backoff", retry_with_backoff),
        ("graceful_degradation", graceful_degradation),
        ("error_context_handling", error_context_handling),
    ]
    local_sections: list[tuple[str, Callable[[], Awaitable[bool]]]] = [
        ("request_timeout", request_timeout),
        ("billing_aware_retries", billing_aware_retries),
    ]

    # Each section records True (passed), False (failed) or None (skipped).
    results: list[tuple[str, bool | None]] = []
    for name, live in live_sections:
        if chat_model is None:
            print(f"\nSection skipped: {name} needs a chat model; {no_model_reason}")
            results.append((name, None))
            continue
        try:
            results.append((name, await live(chat_model)))
        except VeniceError as e:
            print(f"❌ {name} raised an unexpected {type(e).__name__}: {e}")
            results.append((name, False))
    for name, local in local_sections:
        try:
            results.append((name, await local()))
        except VeniceError as e:
            print(f"❌ {name} raised an unexpected {type(e).__name__}: {e}")
            results.append((name, False))

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if failed:
        print(
            f"\n❌ {len(failed)} of {len(results)} error-handling examples failed: "
            f"{', '.join(failed)}"
        )
        return 1
    if skipped:
        print(
            f"\nSKIPPED: the live error sections did not run ({', '.join(skipped)}); "
            "only the local stand-in sections passed"
        )
        return EXIT_SKIPPED

    print("\n✨ Error handling examples completed!")
    print("\n💡 Key takeaways:")
    print("   - Always use specific exception types for targeted handling")
    print("   - The SDK resends only what is safe: reads always, chat/embeddings on")
    print("     transient errors, paid generation requests never once processing began")
    print("   - Tune the policy per block with client.with_retries(RetryOptions(...))")
    print("   - A timeout raises APITimeoutError; the request may still have been billed")
    print("   - Have fallback strategies for graceful degradation")
    print("   - Trust a reply only if finish_reason is 'stop' AND it stayed under the")
    print("     token cap: some backends report a cut-off reply as 'stop'")
    print("   - Log error context for effective debugging")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error in main: {e}", file=sys.stderr)
        sys.exit(1)
