#!/usr/bin/env python3
"""
Venice AI SDK - Advanced Error Recovery
========================================

This example demonstrates advanced error recovery patterns.
Learn how to build resilient applications with retry strategies and error handling patterns.

Every error this script talks about is triggered for real, and none of them
is billed:

- a connection failure against a closed local port, retried by the SDK's retry
  middleware (the retries are observed through ``RetryOptions.on_retry``);
- 500, 502 and 504 responses from a small local server standing in for the
  API, which show that the SDK decides per endpoint whether a failed request
  may be resent (see "Retry classes" below);
- a response that stalls past the client's timeout, from the same local
  server, which arrives as ``APITimeoutError`` with or without a rate limiter
  in front;
- an unknown model name, classified as ``NotFoundError`` and not retried;
- an invalid API key on an authenticated endpoint, classified as
  ``AuthenticationError`` and not retried.

Retry classes
-------------
Resending a request is only safe when the first attempt cannot have been
billed, or when doing it twice costs nothing. The SDK classifies every request
(``venice_ai.classify_request``) into a ``RetryClass``:

- ``IDEMPOTENT``: GETs and free POSTs (quotes, status reads, billing reads).
  Retried on any transient failure.
- ``INFERENCE``: chat completions, responses and embeddings. Retried after a
  connection failure, a 502 or 503, and at most once after a 500 (these 500s
  are not billed but can repeat deterministically). A 504 or a read timeout is
  not resent: the answer may already have been produced and billed.
- ``PAID``: every other POST (image, video, audio and music generation, key
  creation, top-ups). Resent only when the connection never opened, or on a
  503 that means "at capacity" or carries ``Retry-After``.

The retry control is ``RetryOptions``: pass one to ``VeniceClient(retry_options=...)``
for the client default, or scope one to a block with ``client.with_retries(...)``.
A policy is validated when it is built and frozen afterwards; derive a variant
with ``dataclasses.replace()``, which validates it again. Rate limits (429) are
not part of it: ``RetryOptions`` rejects 429 in ``retry_status_codes``, and
429s are retried by a rate limiter instead.

A request timeout is ``APITimeoutError``, a ``VeniceError``. It does not
subclass the built-in ``TimeoutError``, so ``except TimeoutError`` around a
request does not catch it.
"""

import asyncio
import collections
import contextlib
import dataclasses
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from aiohttp import web

from venice_ai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    NoMatchingModelError,
    NotFoundError,
    RateLimitError,
    RetryOptions,
    SimpleRateLimiter,
    UserMessage,
    VeniceClient,
    VeniceError,
    classify_request,
)

# A local port nothing listens on: connecting fails immediately and costs nothing.
UNREACHABLE_BASE_URL = "http://127.0.0.1:9/api/v1"
# A name that is not in the model catalog, used to provoke a 404.
UNKNOWN_MODEL = "no-such-model-error-recovery-demo"
# How long the stand-in API holds a "stalls" answer, and the client timeout
# that gives up on it first.
STALL_SECONDS = 2.0
SHORT_TIMEOUT_SECONDS = 0.5


class RetryRecorder:
    """``on_retry`` callback that records every retry the middleware schedules."""

    def __init__(self) -> None:
        self.retries: list[tuple[int, float, str]] = []

    def __call__(self, attempt: int, delay: float, error: Exception | None) -> None:
        cause = type(error).__name__ if error is not None else "retryable status"
        self.retries.append((attempt, delay, cause))
        print(f"   ↻ retry #{attempt + 1} in {delay:.2f}s after {cause}")


def retry_configuration() -> bool:
    """Show how a ``RetryOptions`` policy is built and validated. No API calls.

    Fails if ``RetryOptions`` accepts 429 in ``retry_status_codes`` (directly
    or through ``dataclasses.replace()``), accepts a list where it takes a
    set, or lets a field be assigned after construction.
    """
    print("🔄 Retry Strategy Configuration")
    print("-" * 40)

    retry_options = RetryOptions(max_attempts=3, base_delay=1.0, max_delay=30.0)
    print(f"max_attempts (retries after the first try): {retry_options.max_attempts}")
    print(
        f"base_delay: {retry_options.base_delay}s, exponential_base: {retry_options.exponential_base}"
    )
    print(
        f"max_delay cap: {retry_options.max_delay}s, jitter_factor: {retry_options.jitter_factor}"
    )
    print(f"Retried status codes: {sorted(retry_options.retry_status_codes)}")
    print(f"Retried exceptions: {[e.__name__ for e in retry_options.retry_exceptions]}")
    print(
        f"A chat/responses/embeddings 500 is retried at most "
        f"{retry_options.max_inference_500_retries}x; a Retry-After longer than "
        f"{retry_options.max_retry_after:.0f}s surfaces the error instead of waiting."
    )
    defaults = RetryOptions()
    print(
        f"Defaults (RetryOptions()): {defaults.max_attempts} retries, "
        f"{defaults.base_delay}s first delay, {defaults.max_delay}s cap"
    )
    print("Whether a listed status is retried also depends on the request's retry class:")
    for method, path in (
        ("GET", "/api/v1/models"),
        ("POST", "/api/v1/video/quote"),
        ("POST", "/api/v1/chat/completions"),
        ("POST", "/api/v1/image/generate"),
        ("POST", "/api/v1/video/queue"),
    ):
        print(f"   {method:<4} {path:<26} -> {classify_request(method, path, {}).value}")
    print("A policy is checked when it is built and cannot change afterwards:")
    ok = True
    checks: list[tuple[str, type[Exception], Callable[[], object]]] = [
        (
            "429 in retry_status_codes",
            ValueError,
            lambda: RetryOptions(retry_status_codes={429, 503}),
        ),
        (
            "429 added through dataclasses.replace()",
            ValueError,
            lambda: dataclasses.replace(retry_options, retry_status_codes={429}),
        ),
        (
            "a list for retry_status_codes (it takes a set)",
            TypeError,
            lambda: RetryOptions(retry_status_codes=[500, 503]),  # type: ignore[arg-type]
        ),
        (
            "assigning max_attempts after construction",
            dataclasses.FrozenInstanceError,
            lambda: setattr(retry_options, "max_attempts", 5),
        ),
    ]
    for label, expected, attempt in checks:
        try:
            attempt()
        except expected as e:
            print(f"   ✓ {label}: {type(e).__name__}: {e}")
        else:
            print(f"   ❌ {label} was accepted; expected {expected.__name__}")
            ok = False
    variant = dataclasses.replace(retry_options, max_attempts=5)
    print(
        f"   ✓ dataclasses.replace(retry_options, max_attempts=5) -> max_attempts="
        f"{variant.max_attempts}; the original still has {retry_options.max_attempts}"
    )
    if variant.max_attempts != 5 or retry_options.max_attempts != 3:
        print("   ❌ replace() did not produce an independent variant")
        ok = False
    print("A client built with VeniceClientFactory handles rate limits in its rate")
    print("limiter, which honors Retry-After; a bare VeniceClient raises RateLimitError")
    print("straight away unless you pass rate_limiter=SimpleRateLimiter(max_retries=...).")
    print()
    print("Use it as the client default:   VeniceClient(retry_options=retry_options)")
    print("Or scope it to a block:         async with client.with_retries(retry_options): ...")
    return ok


async def observed_retries() -> bool:
    """Trigger a retryable connection failure and watch the SDK retry it.

    The client points at a closed local port, so every attempt fails with a
    connection error. The section passes only if the middleware retried exactly
    ``max_attempts`` times and then surfaced ``APIConnectionError``.
    """
    print("\n🔁 Observed Retries (connection failure)")
    print("-" * 40)

    recorder = RetryRecorder()
    policy = RetryOptions(max_attempts=2, base_delay=0.2, jitter_factor=0.0, on_retry=recorder)
    print(
        f"Target: {UNREACHABLE_BASE_URL} (closed port), policy max_attempts={policy.max_attempts}"
    )

    # A placeholder key: the real one is never sent to a non-Venice URL.
    async with VeniceClient(
        api_key="placeholder-key", base_url=UNREACHABLE_BASE_URL, retry_options=policy
    ) as client:
        start = time.perf_counter()
        try:
            await client.chat.completions.create(
                model=UNKNOWN_MODEL,
                messages=[UserMessage(content="Hello")],
                max_completion_tokens=16,
            )
        except APIConnectionError as e:
            elapsed = time.perf_counter() - start
            print(f"   ✓ APIConnectionError after {elapsed:.2f}s: {e}")
        else:
            print("   ❌ The request unexpectedly succeeded against a closed port")
            return False

    if len(recorder.retries) != policy.max_attempts:
        print(f"   ❌ Expected {policy.max_attempts} retries, observed {len(recorder.retries)}")
        return False
    print(f"   ✓ The middleware retried {len(recorder.retries)} times, then gave up")
    return True


async def scoped_retry_override() -> bool:
    """Use ``client.with_retries()`` to swap the policy for one block of calls.

    ``with_retries`` replaces the retry policy for the calls inside the block
    and restores the client default afterwards. ``RetryOptions`` already
    implements exponential backoff, capping and jitter, so there is no need for
    a hand-written ``for attempt in range(...)`` loop.

    Each policy has its own recorder, so the retries show which policy handled
    each call. The calls go to the closed local port, which fails every
    attempt for free. The section passes only if the default policy handled
    the calls before and after the block and the scoped policy handled the
    call inside it.
    """
    print("\n🎯 Scoped Retry Policy Override (with_retries)")
    print("-" * 40)

    default_recorder = RetryRecorder()
    scoped_recorder = RetryRecorder()
    default_policy = RetryOptions(
        max_attempts=1, base_delay=0.1, jitter_factor=0.0, on_retry=default_recorder
    )
    scoped_policy = RetryOptions(
        max_attempts=3, base_delay=0.1, jitter_factor=0.0, on_retry=scoped_recorder
    )

    async def attempt(client: VeniceClient) -> tuple[int, int]:
        """Make one doomed call; return the retries each recorder saw for it."""
        before = len(default_recorder.retries), len(scoped_recorder.retries)
        # Every attempt fails to connect; the retry counts are what matter here.
        with contextlib.suppress(APIConnectionError):
            await client.chat.completions.create(
                model=UNKNOWN_MODEL,
                messages=[UserMessage(content="Hello")],
                max_completion_tokens=16,
            )
        return (
            len(default_recorder.retries) - before[0],
            len(scoped_recorder.retries) - before[1],
        )

    expected = {
        "before the block": (default_policy.max_attempts, 0),
        "inside with_retries": (0, scoped_policy.max_attempts),
        "after the block": (default_policy.max_attempts, 0),
    }
    observed: dict[str, tuple[int, int]] = {}

    # A placeholder key: the real one is never sent to a non-Venice URL.
    async with VeniceClient(
        api_key="placeholder-key", base_url=UNREACHABLE_BASE_URL, retry_options=default_policy
    ) as client:
        print(f"Client default policy: max_attempts={default_policy.max_attempts}")
        observed["before the block"] = await attempt(client)

        print(f"\nInside with_retries(RetryOptions(max_attempts={scoped_policy.max_attempts})):")
        async with client.with_retries(scoped_policy):
            observed["inside with_retries"] = await attempt(client)

        print("\nAfter the block:")
        observed["after the block"] = await attempt(client)

    ok = True
    print()
    for phase, (default_count, scoped_count) in observed.items():
        match = (default_count, scoped_count) == expected[phase]
        ok = ok and match
        print(
            f"   {'✓' if match else '❌'} {phase}: {default_count} default-policy retries, "
            f"{scoped_count} scoped-policy retries (expected {expected[phase]})"
        )
    if ok:
        print("   ✓ with_retries applied only inside its block, then the default came back")
    return ok


@contextlib.asynccontextmanager
async def stand_in_api() -> AsyncIterator[tuple[str, collections.Counter[str]]]:
    """Run a local server that answers like a failing Venice API.

    Yields its base URL and a counter of requests received per route, so the
    caller can see how many attempts each call really made. Responses:

    - ``POST /chat/completions`` with model ``fails-500``: always 500;
      ``fails-504``: always 504; any model starting ``flaky-502``: 502 once,
      then a normal reply; any model starting ``stalls``: no answer for
      ``STALL_SECONDS``, then 500.
    - ``POST /image/generate``: always 500.
    - ``GET /models``: always 500.
    """
    received: collections.Counter[str] = collections.Counter()

    async def chat(request: web.Request) -> web.Response:
        model = (await request.json())["model"]
        received[f"chat {model}"] += 1
        if model.startswith("stalls"):
            await asyncio.sleep(STALL_SECONDS)
        flaky = model.startswith("flaky-502")
        if flaky and received[f"chat {model}"] > 1:
            return web.json_response(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "recovered"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            )
        status = 502 if flaky else {"fails-504": 504}.get(model, 500)
        return web.json_response({"error": f"stand-in failure {status}"}, status=status)

    async def image(request: web.Request) -> web.Response:
        received["image"] += 1
        return web.json_response({"error": "stand-in failure 500"}, status=500)

    async def models(request: web.Request) -> web.Response:
        received["models"] += 1
        return web.json_response({"error": "stand-in failure 500"}, status=500)

    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", chat)
    app.router.add_post("/api/v1/image/generate", image)
    app.router.add_get("/api/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}/api/v1", received
    finally:
        await runner.cleanup()


async def retry_classes() -> bool:
    """Show that the same policy resends some failed requests and not others.

    One client with ``RetryOptions(max_attempts=3)`` sends five calls to a
    local stand-in API. The server counts the attempts it receives, and the
    section passes only if each call made the number of attempts its retry
    class allows. A sixth call wraps a request that fails once and then
    succeeds in ``client.with_retries(...)``: the scoped policy, not the
    client default, must handle its retry. Nothing here reaches Venice, so
    nothing is billed.
    """
    print("\n🧭 Retry Classes (per-endpoint resend rules)")
    print("-" * 40)

    recorder = RetryRecorder()
    policy = RetryOptions(max_attempts=3, base_delay=0.05, jitter_factor=0.0, on_retry=recorder)
    hello = [UserMessage(content="Hello")]

    async with (
        stand_in_api() as (base_url, received),
        # A placeholder key: the real one is never sent to a non-Venice URL.
        VeniceClient(api_key="placeholder-key", base_url=base_url, retry_options=policy) as client,
    ):
        cases: list[tuple[str, str, Callable[[], Awaitable[object]], int, bool]] = [
            (
                "GET /models, 500 (IDEMPOTENT)",
                "models",
                lambda: client.models.list(),
                policy.max_attempts + 1,
                False,
            ),
            (
                "chat, 500 (INFERENCE: one retry)",
                "chat fails-500",
                lambda: client.chat.completions.create(model="fails-500", messages=hello),
                1 + policy.max_inference_500_retries,
                False,
            ),
            (
                "chat, 504 (INFERENCE: may be billed, not resent)",
                "chat fails-504",
                lambda: client.chat.completions.create(model="fails-504", messages=hello),
                1,
                False,
            ),
            (
                "chat, 502 then OK (INFERENCE: retried, succeeds)",
                "chat flaky-502",
                lambda: client.chat.completions.create(model="flaky-502", messages=hello),
                2,
                True,
            ),
            (
                "image generate, 500 (PAID: never resent)",
                "image",
                lambda: client.image.create(model="stand-in-image-model", prompt="a lighthouse"),
                1,
                False,
            ),
        ]

        ok = True
        for label, route, call, expected_attempts, should_succeed in cases:
            print(f"\n▶ {label}")
            succeeded = False
            try:
                result = await call()
                succeeded = True
                print(f"   ✓ Succeeded: {getattr(result, 'text', None)!r}")
            except APIError as e:
                print(f"   {type(e).__name__} (status {e.status_code})")
            attempts = received[route]
            match = attempts == expected_attempts and succeeded == should_succeed
            ok = ok and match
            print(
                f"   {'✓' if match else '❌'} {attempts} attempt(s) reached the server "
                f"(expected {expected_attempts}"
                f"{', then success' if should_succeed else ''})"
            )

        # with_retries around a call that succeeds after one transient failure.
        scoped_recorder = RetryRecorder()
        scoped = RetryOptions(
            max_attempts=1, base_delay=0.05, jitter_factor=0.0, on_retry=scoped_recorder
        )
        default_retries_before = len(recorder.retries)
        print(f"\n▶ chat, 502 then OK, inside with_retries(max_attempts={scoped.max_attempts})")
        succeeded = False
        try:
            async with client.with_retries(scoped):
                result = await client.chat.completions.create(
                    model="flaky-502-scoped", messages=hello
                )
            succeeded = True
            print(f"   ✓ Succeeded: {result.text!r}")
        except APIError as e:
            print(f"   {type(e).__name__} (status {e.status_code})")
        attempts = received["chat flaky-502-scoped"]
        scoped_retries = len(scoped_recorder.retries)
        default_retries = len(recorder.retries) - default_retries_before
        match = succeeded and attempts == 2 and scoped_retries == 1 and default_retries == 0
        ok = ok and match
        print(
            f"   {'✓' if match else '❌'} {attempts} attempt(s), {scoped_retries} retry by the "
            f"scoped policy, {default_retries} by the client default "
            "(expected 2 attempts, 1 scoped retry, 0 default, then success)"
        )

    print(f"\n   The client default policy's on_retry saw {len(recorder.retries)} retries")
    return ok


async def timeout_errors() -> bool:
    """Show that a request timeout is ``APITimeoutError``, not ``TimeoutError``.

    The stand-in API holds its answer longer than the client's timeout, once
    for a bare client and once for a client with a ``SimpleRateLimiter`` in
    front. Both must raise ``APITimeoutError`` (a ``VeniceError``), neither
    may be caught by ``except TimeoutError``, and the chat request must not be
    resent: an inference request that timed out may already have been billed.
    Nothing here reaches Venice.
    """
    print("\n⏰ Timeouts (APITimeoutError, not TimeoutError)")
    print("-" * 40)

    ok = True
    hello = [UserMessage(content="Hello")]
    async with stand_in_api() as (base_url, received):
        for label, limiter in (
            ("bare client", None),
            ("client with SimpleRateLimiter", SimpleRateLimiter(max_retries=1)),
        ):
            model = f"stalls-{len(received)}"
            print(f"\n▶ {label}, timeout={SHORT_TIMEOUT_SECONDS}s, server stalls {STALL_SECONDS}s")
            raised: BaseException | None = None
            # A placeholder key: the real one is never sent to a non-Venice URL.
            async with VeniceClient(
                api_key="placeholder-key",
                base_url=base_url,
                timeout=SHORT_TIMEOUT_SECONDS,
                rate_limiter=limiter,
            ) as client:
                try:
                    await client.chat.completions.create(model=model, messages=hello)
                except TimeoutError as e:
                    print(f"   ❌ A built-in TimeoutError escaped: {e!r}")
                    raised = e
                except APITimeoutError as e:
                    print(f"   ✓ APITimeoutError: {e}")
                    raised = e
                except VeniceError as e:
                    print(f"   ❌ {type(e).__name__} instead of a timeout: {e}")
                    raised = e
            attempts = received[f"chat {model}"]
            match = isinstance(raised, APITimeoutError) and attempts == 1
            ok = ok and match
            print(
                f"   {'✓' if match else '❌'} {type(raised).__name__ if raised else 'no error'}, "
                f"{attempts} attempt(s) reached the server (expected APITimeoutError, 1)"
            )
    print("\n   Catch APITimeoutError (or VeniceError) around requests; a bare")
    print("   `except TimeoutError` only sees your own deadlines, such as asyncio.timeout().")
    return ok


async def classify(
    label: str, call: Callable[[], Awaitable[object]], expected: type[Exception]
) -> bool:
    """Run ``call`` and report which error class the SDK raised.

    The ``except`` blocks classify the terminal error after the SDK has finished
    retrying; they never drive retries themselves. The section passes only if
    the raised error is an instance of ``expected``.
    """
    print(f"\n▶ {label}")
    raised: Exception | None = None
    try:
        await call()
    except AuthenticationError as e:
        print(f"   🔑 AuthenticationError (not retryable, fix the API key): {e}")
        raised = e
    except NotFoundError as e:
        print(f"   🔍 NotFoundError (not retryable, fix the model or endpoint): {e}")
        raised = e
    except RateLimitError as e:
        print(f"   ⏱️ RateLimitError (back off, then retry later): {e}")
        raised = e
    except APITimeoutError as e:
        print(f"   ⏰ APITimeoutError (retries exhausted): {e}")
        raised = e
    except APIConnectionError as e:
        print(f"   🌐 APIConnectionError (retries exhausted): {e}")
        raised = e
    except APIError as e:
        print(f"   🚨 {type(e).__name__} (status {e.status_code}): {e}")
        raised = e
    except VeniceError as e:
        print(f"   ⚠️ {type(e).__name__}: {e}")
        raised = e

    if raised is None:
        print(f"   ❌ Expected {expected.__name__}, but the call succeeded")
        return False
    if not isinstance(raised, expected):
        print(f"   ❌ Expected {expected.__name__}, got {type(raised).__name__}")
        return False
    print(f"   ✓ Classified as {expected.__name__} as expected")
    return True


async def comprehensive_error_handling(chat_model: str) -> bool:
    """Trigger non-retryable errors and classify them.

    Both requests go through ``with_retries`` with a recorder attached, which
    shows that the SDK does not waste retries on errors a retry cannot fix.
    """
    print("\n🛡️ Comprehensive Error Handling")
    print("-" * 40)

    recorder = RetryRecorder()
    policy = RetryOptions(max_attempts=3, base_delay=1.0, jitter_factor=0.2, on_retry=recorder)
    hello = [UserMessage(content="Hello")]

    async with VeniceClient() as client, client.with_retries(policy):
        unknown_model_ok = await classify(
            f"Unknown model {UNKNOWN_MODEL!r}",
            lambda: client.chat.completions.create(
                model=UNKNOWN_MODEL, messages=hello, max_completion_tokens=16
            ),
            NotFoundError,
        )

    bad_key = "invalid-api-key-error-recovery-demo"
    async with VeniceClient(api_key=bad_key) as bad_client, bad_client.with_retries(policy):
        bad_key_ok = await classify(
            "Invalid API key on chat completions (an authenticated endpoint)",
            lambda: bad_client.chat.completions.create(
                model=chat_model, messages=hello, max_completion_tokens=16
            ),
            AuthenticationError,
        )

    if recorder.retries:
        print(f"\n   ❌ Non-retryable errors were retried {len(recorder.retries)} times")
        return False
    print("\n   ✓ Neither error was retried: 4xx client errors fail fast")
    return unknown_model_ok and bad_key_ok


def error_recovery_best_practices() -> bool:
    """Print error recovery guidance. No API calls."""
    print("\n💡 Error Recovery Best Practices")
    print("-" * 40)

    print("1. 🎯 Classify errors:")
    print("   Retryable: APIConnectionError and 5xx, when the request's retry class")
    print("   allows it; the SDK already did that before the error reached you")
    print("   Check before resending: APITimeoutError and 504 on a paid or inference")
    print("   call, since the server may have finished (and billed) the work")
    print("   APITimeoutError is a VeniceError, not a TimeoutError: catch it by name")
    print("   Back off first: RateLimitError (429); honor its Retry-After")
    print("   Fix, don't retry: AuthenticationError, InvalidRequestError,")
    print("   PermissionDeniedError, NotFoundError")
    print()
    print("2. 🔄 Let RetryOptions own the backoff:")
    print("   - Exponential backoff with jitter and a max_delay cap")
    print("   - Fewer retries for interactive paths, more for background jobs")
    print("   - Use on_retry to count and log retries")
    print()
    print("3. ⏱️ Set timeouts to match the workload:")
    print("   - Interactive: 5-10s; background: 30-60s; batch: 120s+")
    print()
    print("4. 🎭 Plan fallbacks: cached responses, degraded functionality, another model")
    print()
    print("💡 Configuration Example:")
    print("   ```python")
    print("   from venice_ai import RetryOptions, VeniceClient")
    print("   client = VeniceClient(")
    print("       timeout=30.0,")
    print("       retry_options=RetryOptions(max_attempts=3, base_delay=1.0, max_delay=30.0),")
    print("   )")
    print("   async with client.with_retries(RetryOptions(max_attempts=5)):")
    print("       ...  # calls in this block use the stricter policy")
    print("   ```")
    return True


async def main() -> int:
    """Run all error recovery examples and return the process exit code."""
    print("=" * 60)
    print("Venice AI SDK - Advanced Error Recovery Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat(prefer="cheapest")
    print(f"Chat model: {chat_model}\n")

    results: list[tuple[str, bool]] = [
        ("retry_configuration", retry_configuration()),
        ("observed_retries", await observed_retries()),
        ("scoped_retry_override", await scoped_retry_override()),
        ("retry_classes", await retry_classes()),
        ("timeout_errors", await timeout_errors()),
        ("comprehensive_error_handling", await comprehensive_error_handling(chat_model)),
        ("error_recovery_best_practices", error_recovery_best_practices()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
    else:
        print(f"✅ All {len(results)} sections passed")
    print("=" * 60)
    if not failed:
        print()
        print("📚 Next Steps:")
        print("   - Review examples/basic/error_handling.py for the basics")
        print("   - Pick a RetryOptions policy per workload")
        print("   - Track retry counts with on_retry and alert on spikes")

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
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
