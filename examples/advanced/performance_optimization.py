#!/usr/bin/env python3
"""
Venice AI SDK - Performance Optimization
=========================================

This example demonstrates performance optimization techniques.
Learn how to maximize throughput and minimize latency for your applications.

The live sections measure sequential vs concurrent requests, timing each
request on the wire with an ``aiohttp.TraceConfig``, and streaming vs
non-streaming time to first token. Any failed, truncated or wrong answer fails
its section, as does a concurrent batch whose requests were not all on the
wire at once, and the script exits 1 if any section failed. Requests pass
``include_venice_system_prompt=False``, so only the prompts shown here are
billed and timed.
"""

import asyncio
import os
import re
import sys
import time
from types import SimpleNamespace

import aiohttp

from venice_ai import (
    ChatCompletionResponse,
    NoMatchingModelError,
    RetryOptions,
    UserMessage,
    VeniceClient,
    VeniceParameters,
    create_production_config,
)
from venice_ai.middleware import create_retry_middleware

# Bill only the prompts shown here, not the system prompt Venice adds by default.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

PRODUCTION_SNIPPET = """\
import os
from venice_ai import VeniceClientFactory, create_production_config

# BackendType.REDIS + RateLimiterMode.ADAPTIVE: rate-limit state is shared
# by every process that points at the same Redis.
# VENICE_BACKEND__REDIS__REDIS_URL is the SDK's own setting and wins;
# VENICE_REDIS_URL is the fallback.
redis_url = os.environ.get("VENICE_BACKEND__REDIS__REDIS_URL") or os.environ["VENICE_REDIS_URL"]
config = create_production_config(
    redis_url=redis_url,
    max_concurrent_executions=100,  # adaptive scheduler concurrency
    max_queue_size=5000,            # per-model queue bound (backpressure)
)
client = VeniceClientFactory.create_client(config=config, account_id="my-service")
"""


def connection_pooling_optimization() -> bool:
    """Explain connection pool settings. No API calls."""
    print("🔌 Connection Pooling Optimization")
    print("-" * 40)

    print("✅ HTTP Connection Pool Settings:")
    print()
    print("1. max_connections:")
    print("   - Size of the connection pool")
    print("   - Also the cap on concurrent requests from one client")
    print("   - Raise it when many requests run at once")
    print()
    print("2. Reuse one client:")
    print("   ✓ Pooled connections skip the TCP and TLS handshakes")
    print("   ✓ Lower latency per request")
    print("   ✗ A new client per request throws the pool away")
    print()
    print("💡 Example Configuration:")
    print("   ```python")
    print("   HttpClientConfig(timeout=30.0, max_connections=100)")
    print("   ```")
    return True


def check_completion(label: str, response: ChatCompletionResponse) -> bool:
    """Fail a non-streaming response that was truncated or came back empty."""
    finish = response.choices[0].finish_reason
    if finish == "length" or not (response.text or "").strip():
        print(f"   ❌ {label}: truncated or empty (finish_reason={finish})")
        return False
    return True


def check_count(label: str, response: ChatCompletionResponse, n: int) -> bool:
    """Fail a "count from 1 to n" answer that is cut off or counts wrongly."""
    if not check_completion(label, response):
        return False
    digits = re.findall(r"\d", response.text or "")
    if digits != [str(i) for i in range(1, n + 1)]:
        print(f"   ❌ {label}: expected 1 to {n}, got {(response.text or '').strip()!r}")
        return False
    return True


class WireTimer:
    """Time every HTTP request from the moment it is sent to its response.

    An ``aiohttp.TraceConfig`` on the session sees each request after it has
    a connection and its headers are written, so the time a request spends
    waiting for a free pooled connection is not counted. Timing around
    ``create()`` instead would count that wait, and a batch the pool
    serializes would still look concurrent.
    """

    def __init__(self) -> None:
        self.intervals: list[tuple[float, float]] = []
        self.config = aiohttp.TraceConfig()
        self.config.on_request_headers_sent.append(self._sent)
        self.config.on_request_end.append(self._answered)

    async def _sent(
        self,
        session: aiohttp.ClientSession,
        ctx: SimpleNamespace,
        params: aiohttp.TraceRequestHeadersSentParams,
    ) -> None:
        ctx.sent_at = time.perf_counter()

    async def _answered(
        self,
        session: aiohttp.ClientSession,
        ctx: SimpleNamespace,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        self.intervals.append((ctx.sent_at, time.perf_counter()))

    def take(self) -> list[tuple[float, float]]:
        """Return the intervals recorded so far and start a new batch."""
        taken, self.intervals = self.intervals, []
        return taken


def peak_overlap(intervals: list[tuple[float, float]]) -> int:
    """The largest number of (start, end) intervals open at one instant."""
    # At equal timestamps an end sorts before a start, so back-to-back
    # requests do not count as overlapping.
    events = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    peak = open_now = 0
    for _, step in events:
        open_now += step
        peak = max(peak, open_now)
    return peak


async def concurrent_requests_pattern() -> bool:
    """Compare sequential and concurrent execution of the same requests.

    Every request is timed on the wire by a :class:`WireTimer` attached to
    the client's HTTP session. The section fails unless the sequential
    requests never overlapped and all the concurrent ones were on the wire
    at the same instant. The speedup over the sequential run is reported,
    not checked, together with the measured times that explain it.
    """
    print("\n⚡ Concurrent Request Patterns")
    print("-" * 40)

    timer = WireTimer()
    request_count = 3
    # A session of your own lets you attach instrumentation such as a
    # TraceConfig. The client authenticates each request it sends through the
    # session, so the session needs no credentials. This one carries the SDK's
    # retry middleware, and its connector's limit is the pool size, which caps
    # how many requests can be on the wire at once.
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100),
        middlewares=[create_retry_middleware(RetryOptions())],
        trace_configs=[timer.config],
    )

    async with session, VeniceClient(http_client=session, timeout=30.0) as client:
        chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        print(f"✅ Model: {chat_model}")
        timer.take()  # Drop the catalog request.

        async def make_single_request(index: int) -> ChatCompletionResponse:
            return await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content=f"Count from 1 to {index}, digits only.")],
                max_completion_tokens=32,
                venice_parameters=OWN_PROMPT_ONLY,
            )

        print("\n📊 Performance Comparison:")

        print("\n1. Sequential (one at a time):")
        start = time.perf_counter()
        sequential: list[ChatCompletionResponse] = []
        for i in range(request_count):
            sequential.append(await make_single_request(i + 1))
        sequential_time = time.perf_counter() - start
        sequential_wire = timer.take()
        print(f"   Total time: {sequential_time:.2f}s")

        print("\n2. Concurrent (asyncio.gather):")
        start = time.perf_counter()
        gathered = await asyncio.gather(
            *(make_single_request(i + 1) for i in range(request_count)),
            return_exceptions=True,
        )
        concurrent_time = time.perf_counter() - start
        concurrent_wire = timer.take()
        concurrent = [r for r in gathered if isinstance(r, ChatCompletionResponse)]
        errors = [r for r in gathered if isinstance(r, BaseException)]
        print(f"   Total time: {concurrent_time:.2f}s")
        for error in errors:
            print(f"   ❌ Request failed: {type(error).__name__}: {error}")

    ok = not errors
    for i, response in enumerate(sequential):
        ok = check_count(f"sequential request {i + 1}", response, i + 1) and ok
    for i, response in enumerate(gathered):
        if isinstance(response, ChatCompletionResponse):
            ok = check_count(f"concurrent request {i + 1}", response, i + 1) and ok
    print(
        f"   Completed: {len(sequential)}/{request_count} sequential, "
        f"{len(concurrent)}/{request_count} concurrent"
    )

    for name, wire in (("sequential", sequential_wire), ("concurrent", concurrent_wire)):
        times = ", ".join(f"{end - start:.2f}s" for start, end in wire)
        print(f"   On the wire, {name}: {times}; at most {peak_overlap(wire)} at once")
    if len(sequential_wire) != request_count or len(concurrent_wire) != request_count:
        print(f"   ❌ Expected {request_count} timed requests per batch (a retry adds one)")
        ok = False
    elif peak_overlap(sequential_wire) != 1:
        print("   ❌ The sequential requests overlapped on the wire")
        ok = False
    elif peak_overlap(concurrent_wire) != request_count:
        print("   ❌ The concurrent requests were not all on the wire at once")
        ok = False

    if ok:
        speedup = sequential_time / concurrent_time
        slowest = max(end - start for start, end in concurrent_wire)
        wire_sum = sum(end - start for start, end in sequential_wire)
        print(f"\n   ⚡ Speedup over the sequential run: {speedup:.1f}x")
        print(
            f"   A concurrent batch lasts about as long as its slowest request "
            f"({slowest:.2f}s on the wire);\n"
            f"   a sequential run lasts about the sum of its requests ({wire_sum:.2f}s)."
        )
        if speedup <= 1:
            if slowest >= wire_sum:
                print(
                    "   ℹ️  Not faster this time: the slowest concurrent request alone was on\n"
                    "      the wire longer than all the sequential requests together."
                )
            else:
                print(
                    "   ℹ️  Not faster this time: time spent off the wire made the difference "
                    f"({concurrent_time - slowest:.2f}s\n"
                    f"      in the concurrent run vs {sequential_time - wire_sum:.2f}s "
                    "in the sequential one)."
                )
            print("      One small batch is a noisy sample; measure with more requests.")
        limits = concurrent[-1].response_rate_limits
        if limits and limits.remaining_requests is not None:
            print(
                f"   📉 Rate limit: {limits.remaining_requests}/{limits.limit_requests} "
                "requests left in this window"
            )

    print("\n💡 Best Practice:")
    print("   - Use asyncio.gather() for independent requests")
    print("   - Bound concurrency with asyncio.Semaphore or max_connections")
    print("   - Watch response_rate_limits to stay under the limit")
    return ok


def batching_strategies() -> bool:
    """Explain request batching. No API calls."""
    print("\n📦 Request Batching Strategies")
    print("-" * 40)

    print("✅ Batching Approaches:")
    print()
    print("1. Size-based: send a batch once N requests are waiting")
    print("2. Time-based: send whatever arrived in the last T seconds")
    print("3. Hybrid: whichever of N or T comes first")
    print()
    print("💡 Example Implementation:")
    print("   ```python")
    print("   async def process_in_batches(coros, batch_size=10):")
    print("       results = []")
    print("       for i in range(0, len(coros), batch_size):")
    print("           batch = coros[i : i + batch_size]")
    print("           results += await asyncio.gather(*batch, return_exceptions=True)")
    print("       return results")
    print("   ```")
    print()
    print("📊 Benefits:")
    print("   ✓ Bounded concurrency and memory")
    print("   ✓ Easier rate limit management")
    print("   ✓ Failures are collected per request, not lost")
    return True


async def streaming_for_large_responses() -> bool:
    """Compare time to first token with and without streaming."""
    print("\n🌊 Streaming for Large Responses")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        prompt = (
            "Write two short paragraphs (about 120 words in total) about a robot learning to paint."
        )
        max_tokens = 600

        print("1. Non-Streaming (wait for the complete response):")
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content=prompt)],
            max_completion_tokens=max_tokens,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        ns_total = time.perf_counter() - start
        ns_finish = response.choices[0].finish_reason
        print(f"   Time to first token: {ns_total:.2f}s (== time to complete)")
        print(f"   Characters returned: {len(response.text or '')}")
        print(f"   finish_reason:       {ns_finish}")
        ok = check_completion("non-streaming", response)

        print("\n2. Streaming (progressive updates):")
        start = time.perf_counter()
        first_token_at: float | None = None
        s_chars = 0
        chunk_count = 0
        s_finish: str | None = None
        stream = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content=prompt)],
            max_completion_tokens=max_tokens,
            venice_parameters=OWN_PROMPT_ONLY,
            stream=True,
        )
        async for chunk in stream:
            chunk_count += 1
            if chunk.text:
                if first_token_at is None:
                    first_token_at = time.perf_counter() - start
                s_chars += len(chunk.text)
            if chunk.choices and chunk.choices[0].finish_reason:
                s_finish = chunk.choices[0].finish_reason
        s_total = time.perf_counter() - start
        print(f"   Time to first token: {first_token_at or s_total:.2f}s")
        print(f"   Total time:          {s_total:.2f}s")
        print(f"   Chunks received:     {chunk_count}")
        print(f"   Characters returned: {s_chars}")
        print(f"   finish_reason:       {s_finish}")
        if first_token_at is None or s_finish in (None, "length"):
            print(f"   ❌ streaming: truncated or empty (finish_reason={s_finish})")
            ok = False

        if ok and first_token_at is not None:
            delta = ns_total - first_token_at
            if delta >= 0:
                print(
                    f"\n⚡ Streaming showed the first token {delta:.2f}s sooner "
                    f"({delta / ns_total * 100:.0f}% of the non-streaming wait)"
                )
            else:
                print(
                    f"\n🐢 Streaming showed the first token {-delta:.2f}s later than "
                    "the full non-streaming response (network variance)"
                )
    return ok


def memory_management() -> bool:
    """Explain memory-efficient patterns. No API calls."""
    print("\n💾 Memory Management")
    print("-" * 40)

    print("1. Stream long responses: handle chunks as they arrive")
    print("2. Bound in-flight work: asyncio.Semaphore or batches, not one giant gather()")
    print("3. Close clients: `async with VeniceClient() as client:` releases the pool")
    print("4. Keep only what you need from each response, not the whole object")
    print("5. max_connections bounds the pool; each connection holds buffers")
    return True


def performance_monitoring() -> bool:
    """Explain what to measure. No API calls."""
    print("\n📊 Performance Monitoring")
    print("-" * 40)

    print("✅ Key Metrics to Track:")
    print("   - Latency percentiles (p50, p95, p99) per model")
    print("   - Throughput: requests and tokens per second")
    print("   - Error rate by exception class")
    print("   - Rate limit headroom: remaining requests and tokens")
    print()
    print("💡 Example Monitoring:")
    print("   ```python")
    print("   start = time.perf_counter()")
    print("   response = await client.chat.completions.create(...)")
    print("   metrics.record_latency(time.perf_counter() - start)")
    print("   metrics.record_tokens(response.usage.total_tokens)")
    print("   limits = response.response_rate_limits")
    print("   if limits and limits.remaining_requests is not None:")
    print("       metrics.record_rate_limit_headroom(limits.remaining_requests)")
    print("   ```")
    return True


def production_configuration() -> bool:
    """Show a distributed production configuration and check that it builds."""
    print("\n🏭 Production Configuration")
    print("-" * 40)

    print("✅ Several processes sharing one rate limit (Redis + ADAPTIVE):")
    print()
    print("```python")
    print(PRODUCTION_SNIPPET, end="")
    print("```")

    # Build the same config the snippet shows. Building it does not contact Redis.
    config = create_production_config(
        redis_url="redis://redis.internal:6379/0",
        max_concurrent_executions=100,
        max_queue_size=5000,
    )
    assert config.rate_limiter is not None
    print(
        f"\n   ✓ Config builds: backend={config.backend.backend_type.value}, "
        f"rate_limiter={config.rate_limiter.mode.value}"
    )
    print("   Run it against a live Redis with examples/advanced/redis_backend.py.")
    print("   Measure throughput and latency under your own rate limits before sizing.")
    return True


async def main() -> int:
    """Run all performance optimization examples and return the process exit code."""
    print("=" * 60)
    print("Venice AI SDK - Performance Optimization Examples")
    print("=" * 60)

    results = [
        ("connection_pooling_optimization", connection_pooling_optimization()),
        ("concurrent_requests_pattern", await concurrent_requests_pattern()),
        ("batching_strategies", batching_strategies()),
        ("streaming_for_large_responses", await streaming_for_large_responses()),
        ("memory_management", memory_management()),
        ("performance_monitoring", performance_monitoring()),
        ("production_configuration", production_configuration()),
    ]
    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        print("=" * 60)
        return 1
    print(f"✅ All {len(results)} sections passed")
    print("=" * 60)
    print()
    print("📚 Key Takeaways:")
    print("   1. Reuse one client so connections are pooled")
    print("   2. Use asyncio.gather() for independent requests, with bounded concurrency")
    print("   3. Stream long responses to cut time to first token")
    print("   4. Monitor latency, errors and rate-limit headroom in production")
    return 0


if __name__ == "__main__":
    if not os.getenv("VENICE_API_KEY"):
        print(
            "ERROR: VENICE_API_KEY is not set. Export it (e.g. via .env) and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)
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
