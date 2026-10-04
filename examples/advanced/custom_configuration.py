#!/usr/bin/env python3
"""
Venice AI SDK - Advanced Custom Configuration
==============================================

This example demonstrates advanced client configuration options.
Learn how to fine-tune Venice AI for specific use cases and environments.

Two ways to build a client:

- ``VeniceClient()`` directly: retry middleware with default ``RetryOptions``,
  a 120s timeout and no client-side rate limiter.
- ``VeniceClientFactory.create_client(config=VeniceAIConfig(...))``: the same
  client wired from a config object, including a rate limiter chosen by
  ``RateLimiterConfig.mode``.

Every section reads the settings back from the built client (``timeout``,
``rate_limiter``, ``retry_options`` and ``connection_limits``), checks them
against what was configured, and makes one request with it. Requests use the
cheapest chat model that answers directly and omit the Venice system prompt,
so each costs a small fraction of a cent.
"""

import asyncio
import sys
from typing import Any

from venice_ai import (
    BackendConfig,
    BackendType,
    HttpClientConfig,
    NoMatchingModelError,
    RateLimiterConfig,
    RateLimiterMode,
    RetryOptions,
    UserMessage,
    VeniceAIConfig,
    VeniceClient,
    VeniceClientFactory,
    VeniceError,
)
from venice_ai.types.api import VeniceParameters


def memory_config(**sections: Any) -> VeniceAIConfig:
    """Build a ``VeniceAIConfig`` with an explicit in-memory backend.

    Naming the backend keeps a Redis URL in the environment
    (``VENICE_BACKEND__REDIS__REDIS_URL``) from being merged into a config that
    does not use Redis.
    """
    return VeniceAIConfig(backend=BackendConfig(backend_type=BackendType.MEMORY), **sections)


async def check_reply(client: VeniceClient, prompt: str) -> bool:
    """Send one short chat request and fail on a truncated or empty reply."""
    # exclude_reasoning: a reasoning model would spend this small budget thinking.
    chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    try:
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content=prompt)],
            max_completion_tokens=64,
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )
    except VeniceError as e:
        # One section's failed request fails that section; the others still
        # run. A timeout shorter than the model's current latency arrives here
        # as APITimeoutError, with or without a rate limiter in front.
        print(f"   ❌ {chat_model}: {type(e).__name__}: {e}")
        return False
    finish = response.choices[0].finish_reason
    text = (response.text or "").strip()
    print(f"   🤖 {chat_model}: {text!r} (finish_reason={finish})")
    if finish == "length" or not text:
        print("   ❌ The reply was truncated or empty")
        return False
    return True


def describe_client(
    client: VeniceClient,
    *,
    timeout: float,
    max_retries: int,
    pool_limit: int,
    exponential_base: float | None = None,
) -> bool:
    """Print the settings read back from a built client and check them.

    ``client.retry_options`` is the retry policy the client resolved
    (defaults filled in; frozen, so a variant is made with
    ``dataclasses.replace()``) and
    ``client.connection_limits`` the pool limits of its HTTP session. Returns False if any value differs from the expected one.
    An *exponential_base* equal to the default would pass whether or not the
    configured value reached the client, so it must differ from the default.
    """
    limiter = client.rate_limiter
    retry = client.retry_options
    pool = client.connection_limits
    print(f"   - Timeout: {client.timeout:.0f}s")
    print(f"   - Rate limiter: {type(limiter).__name__ if limiter else 'none'}")
    if retry is None:
        print("   ❌ The client reports no retry policy")
        return False
    print(
        f"   - Retry policy: {retry.max_attempts} retries, first delay {retry.base_delay}s "
        f"x{retry.exponential_base} per retry, capped at {retry.max_delay}s; a chat 500 is "
        f"retried at most {min(retry.max_attempts, retry.max_inference_500_retries)}x"
    )
    per_host = pool.limit_per_host or "no"
    print(f"   - Connection pool: {pool.limit} connections in total, {per_host} per-host cap")

    expected = {
        "timeout": (client.timeout, timeout),
        "retries": (retry.max_attempts, max_retries),
        "pool limit": (pool.limit, pool_limit),
    }
    if exponential_base is not None:
        default_base = RetryOptions().exponential_base
        if exponential_base == default_base:
            print(
                f"   ❌ backoff multiplier {exponential_base} is the default; the check proves nothing"
            )
            return False
        expected["backoff multiplier"] = (retry.exponential_base, exponential_base)
    wrong = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    for name, (got, want) in wrong.items():
        print(f"   ❌ {name}: the client has {got}, expected {want}")
    return not wrong


async def minimal_configuration() -> bool:
    """Demonstrate the bare client."""
    print("🎯 Minimal Configuration")
    print("-" * 40)

    async with VeniceClient() as client:
        print("✅ VeniceClient() with no arguments:")
        defaults = RetryOptions()
        ok = describe_client(
            client, timeout=120.0, max_retries=defaults.max_attempts, pool_limit=1000
        )
        print("   - Retries depend on the endpoint: a paid generation request is never")
        print("     resent once the server may have started it")
        print("   - A 429 raises RateLimitError; nothing queues or backs off for you")
        return await check_reply(client, "Say hello in five words or fewer.") and ok


async def custom_http_settings() -> bool:
    """Configure the HTTP client: timeout, pool size and retry policy."""
    print("\n🌐 Custom HTTP Settings")
    print("-" * 40)

    config = memory_config(
        http_client=HttpClientConfig(
            timeout=60.0,  # Longer timeout for slow connections
            max_connections=100,  # Pool size, which is also the request concurrency cap
            max_retries=2,  # Retries after the first attempt
            retry_backoff_factor=3.0,  # Delay triples with each retry (default 2.0)
        ),
    )
    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Custom HTTP client configured")
        ok = describe_client(
            client,
            timeout=config.http_client.timeout,
            max_retries=config.http_client.max_retries,
            pool_limit=config.http_client.max_connections,
            exponential_base=config.http_client.retry_backoff_factor,
        )
        return await check_reply(client, "Reply with a one-line greeting.") and ok


async def rate_limiter_modes() -> bool:
    """Compare the rate limiter modes a config can select."""
    print("\n⚙️ Rate Limiter Modes")
    print("-" * 40)

    print("\n📋 RateLimiterMode.SIMPLE (the default):")
    print("   - In-process; reacts to 429 responses")
    print("   - Exponential backoff with jitter, honors Retry-After")
    print("   - No external dependencies")

    config = memory_config(
        rate_limiter=RateLimiterConfig(mode=RateLimiterMode.SIMPLE, max_retries=3)
    )
    client = VeniceClientFactory.create_client(config=config)

    async with client:
        ok = describe_client(
            client,
            timeout=config.http_client.timeout,
            max_retries=config.http_client.max_retries,
            pool_limit=config.http_client.max_connections,
        )
        assert config.rate_limiter is not None
        print(f"   - Configured 429 retries (rate limiter): {config.rate_limiter.max_retries}")
        ok = await check_reply(client, "Name one primary color.") and ok

    print("\n📋 RateLimiterMode.ADAPTIVE:")
    print("   - Shares rate-limit state across processes through Redis")
    print("   - Reads the x-ratelimit-* response headers to pace requests")
    print("   - Runs the scheduler configured by SchedulerConfig")
    print("   - Needs Redis and the [adaptive] extra; see examples/advanced/redis_backend.py")
    return ok


async def high_concurrency_config() -> bool:
    """Show a configuration tuned for many concurrent requests from one process."""
    print("\n🏭 High-Concurrency Configuration")
    print("-" * 40)

    config = memory_config(
        http_client=HttpClientConfig(
            timeout=45.0,
            max_connections=200,
            max_retries=3,
        ),
        rate_limiter=RateLimiterConfig(mode=RateLimiterMode.SIMPLE),
    )
    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ High-concurrency configuration loaded")
        ok = describe_client(
            client,
            timeout=config.http_client.timeout,
            max_retries=config.http_client.max_retries,
            pool_limit=config.http_client.max_connections,
        )
        print("\n📊 Configuration Summary:")
        print(f"   Backend: {config.backend.backend_type.value}")
        assert config.rate_limiter is not None
        print(f"   Rate limiter mode: {config.rate_limiter.mode.value}")
        print("\n💡 Best for a single process sending many requests at once.")
        print("   For several processes sharing one rate limit, use create_production_config()")
        print("   with Redis (see examples/advanced/redis_backend.py).")
        return await check_reply(client, "Reply with the single word: ready") and ok


async def development_config() -> bool:
    """Show a configuration that fails fast while debugging."""
    print("\n🔧 Development Configuration")
    print("-" * 40)

    config = memory_config(
        http_client=HttpClientConfig(
            timeout=120.0,  # Room for stepping through code in a debugger
            max_connections=10,
            max_retries=0,  # Surface every error immediately
        ),
    )
    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Development configuration loaded")
        ok = describe_client(
            client,
            timeout=config.http_client.timeout,
            max_retries=config.http_client.max_retries,
            pool_limit=config.http_client.max_connections,
        )
        print("\n📊 Configuration Summary:")
        print(f"   Backend: {config.backend.backend_type.value} (no external dependencies)")
        print("   0 retries: errors surface on the first failure")
        return await check_reply(client, "Reply with the single word: debugging") and ok


async def main() -> int:
    """Run all custom configuration examples and return the process exit code."""
    print("🚀 Venice AI Advanced Configuration Examples")
    print("=" * 60)

    results = [
        ("minimal_configuration", await minimal_configuration()),
        ("custom_http_settings", await custom_http_settings()),
        ("rate_limiter_modes", await rate_limiter_modes()),
        ("high_concurrency_config", await high_concurrency_config()),
        ("development_config", await development_config()),
    ]
    failed = [name for name, ok in results if not ok]

    print()
    if failed:
        print(f"❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print(f"✨ All {len(results)} configuration examples passed")
    print("\n💡 Key concepts demonstrated:")
    print("   - The bare client vs a factory-built client")
    print("   - HTTP client tuning: timeout, pool size, retry policy")
    print("   - Reading the resolved settings back: retry_options, connection_limits")
    print("   - The SIMPLE in-process rate limiter (ADAPTIVE is in redis_backend.py)")
    print("   - High-concurrency and development settings")
    return 0


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
        print(f"\n❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
