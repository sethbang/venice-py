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

The sections read the settings back from the built client (``timeout``,
``rate_limiter``, ``retry_options`` and ``connection_limits``) and check them
against what was configured. ``client_options`` runs against a local server
standing in for a proxy and the API, and ``configuration_presets`` builds
clients and sends nothing, so those two are free and need no API key. Every
other section makes one request. Requests use the cheapest chat model that
answers directly and omit the Venice system prompt.

Cost: 5 short chat requests, a small fraction of a cent in total.

Exit status: ``0`` when every section passed, ``1`` when any failed, ``77``
(with a ``SKIPPED:`` line) when the catalog has no chat model to use.
"""

import asyncio
import contextlib
import os
import sys
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from aiohttp import web

from venice_ai import (
    BackendConfig,
    BackendType,
    HttpClientConfig,
    NoMatchingModelError,
    RateLimiterConfig,
    RateLimiterMode,
    RetryOptions,
    SimpleRateLimiter,
    UserMessage,
    VeniceAIConfig,
    VeniceClient,
    VeniceClientFactory,
    VeniceError,
    create_developer_client,
)
from venice_ai.presets import (
    create_development_config,
    create_production_config,
    create_testing_config,
)
from venice_ai.presets.production import (
    create_production_config_conservative,
    create_production_config_high_throughput,
)
from venice_ai.types.api import VeniceParameters

# Plenty for the short replies these sections ask for.
REPLY_MAX_TOKENS = 64

# Sections that send nothing, or send only to a local server, need no real key.
PLACEHOLDER_KEY = "placeholder-key-custom-configuration"

# Nothing listens on this local port. A request that reaches the stand-in went
# through the proxy; one that tried to go direct would fail to connect.
UNREACHABLE_BASE_URL = "http://127.0.0.1:9/api/v1"


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
            max_completion_tokens=REPLY_MAX_TOKENS,
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
    # Some backends report finish_reason="stop" at the token cap, so a reply that
    # used the whole budget, or has no usage to check, counts as truncated too.
    usage = response.usage
    capped = usage is None or usage.completion_tokens >= REPLY_MAX_TOKENS
    if capped or finish == "length" or not text:
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

    # 5 differs from the default of 3, so reading it back proves it was applied.
    simple_retries = 5
    if RateLimiterConfig().max_retries == simple_retries:
        print(f"   ❌ max_retries={simple_retries} is the default; the check proves nothing")
        return False
    config = memory_config(
        rate_limiter=RateLimiterConfig(mode=RateLimiterMode.SIMPLE, max_retries=simple_retries)
    )
    client = VeniceClientFactory.create_client(config=config)

    async with client:
        ok = describe_client(
            client,
            timeout=config.http_client.timeout,
            max_retries=config.http_client.max_retries,
            pool_limit=config.http_client.max_connections,
        )
        limiter = client.rate_limiter
        if not isinstance(limiter, SimpleRateLimiter) or limiter.max_retries != simple_retries:
            print(
                f"   ❌ Expected a SimpleRateLimiter with {simple_retries} retries, got {limiter}"
            )
            ok = False
        else:
            print(f"   - 429 retries (read from the client's limiter): {limiter.max_retries}")
        ok = await check_reply(client, "Name one primary color.") and ok

    print("\n📋 RateLimiterMode.DISABLED:")
    print("   - No limiter logic: a 429 raises RateLimitError at once, with no backoff")
    print("   - The SDK warns when it builds this client, and logs at CRITICAL")
    config = memory_config(rate_limiter=RateLimiterConfig(mode=RateLimiterMode.DISABLED))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        disabled_client = VeniceClientFactory.create_client(config=config)
    async with disabled_client:
        name = type(disabled_client.rate_limiter).__name__
        print(f"   - Rate limiter on the client: {name}")
        if name != "NoOpRateLimiter":
            print(f"   ❌ Expected NoOpRateLimiter, got {name}")
            ok = False
    if not any("DISABLED" in str(w.message) for w in caught):
        print("   ❌ Expected a warning that rate limiting is disabled")
        ok = False

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
    """Show a configuration with no retries at all, for debugging.

    Two layers retry independently: ``HttpClientConfig.max_retries`` covers
    connection failures and 5xx responses, and ``RateLimiterConfig.max_retries``
    covers 429. Both must be 0 for every error to surface on the first failure.
    """
    print("\n🔧 Development Configuration")
    print("-" * 40)

    config = memory_config(
        http_client=HttpClientConfig(
            timeout=120.0,  # Room for stepping through code in a debugger
            max_connections=10,
            max_retries=0,  # No resends after a connection failure or 5xx
        ),
        rate_limiter=RateLimiterConfig(
            mode=RateLimiterMode.SIMPLE,
            max_retries=0,  # No resends after a 429
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
        limiter_retries = getattr(client.rate_limiter, "max_retries", None)
        print(f"   - 429 retries (rate limiter): {limiter_retries}")
        if limiter_retries != 0:
            print(f"   ❌ 429 retries: the client has {limiter_retries}, expected 0")
            ok = False
        print("\n📊 Configuration Summary:")
        print(f"   Backend: {config.backend.backend_type.value} (no external dependencies)")
        print("   0 retries in both layers: errors, 429 included, surface on the first failure")
        return await check_reply(client, "Reply with the single word: debugging") and ok


@contextlib.asynccontextmanager
async def stand_in_proxy() -> AsyncIterator[tuple[str, list[dict[str, str]]]]:
    """Run a local server that plays both the forward proxy and the API behind it.

    Yields its URL and one record per request received: the ``Host`` the client
    was aiming at, the path, and the custom header. Every request gets an empty
    model catalog back.
    """
    received: list[dict[str, str]] = []

    async def answer(request: web.Request) -> web.Response:
        received.append(
            {
                "host": request.headers.get("Host", ""),
                "path": request.url.path,
                "trace": request.headers.get("X-Example-Trace", ""),
            }
        )
        return web.json_response({"object": "list", "type": "text", "data": []})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}", received
    finally:
        await runner.cleanup()


@contextlib.contextmanager
def proxy_environment(proxy_url: str) -> Iterator[None]:
    """Set the proxy variables ``trust_env`` reads, and restore them afterwards."""
    names = ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy")
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update(HTTP_PROXY=proxy_url, http_proxy=proxy_url, NO_PROXY="", no_proxy="")
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def reached_stand_in(label: str, received: list[dict[str, str]], trace: str = "") -> bool:
    """Check the one request a client just sent arrived at the stand-in, aimed at the API."""
    if len(received) != 1:
        print(f"   ❌ {label}: expected 1 request at the stand-in, got {len(received)}")
        return False
    seen = received.pop()
    print(f"   - {label}: Host {seen['host']}, path {seen['path']}")
    if seen["host"] != "127.0.0.1:9" or seen["path"] != "/api/v1/models":
        print(f"   ❌ {label}: the request was not aimed at the configured API")
        return False
    if seen["trace"] != trace:
        print(f"   ❌ {label}: X-Example-Trace was {seen['trace']!r}, expected {trace!r}")
        return False
    return True


async def client_options() -> bool:
    """Show the constructor options for proxies, headers and connection limits.

    Runs against a local server only, with a placeholder key: nothing is billed
    and no API key is needed. ``proxy=`` and ``trust_env=True`` are proved by
    aiming the client at a port nothing listens on and seeing the request arrive
    at the proxy. The limits are read back from ``client.connection_limits``.
    """
    print("\n🔀 Proxies, Headers and Connection Limits (local stand-in, free)")
    print("-" * 40)

    ok = True
    async with stand_in_proxy() as (proxy_url, received):
        try:
            async with VeniceClient(
                api_key=PLACEHOLDER_KEY,
                base_url=UNREACHABLE_BASE_URL,
                proxy=proxy_url,
                headers={"X-Example-Trace": "from-headers-option"},
            ) as client:
                await client.models.list()
            ok = reached_stand_in("proxy= and headers=", received, "from-headers-option") and ok

            # trust_env=True takes the proxy from HTTP_PROXY, as curl and other tools do.
            with proxy_environment(proxy_url):
                async with VeniceClient(
                    api_key=PLACEHOLDER_KEY, base_url=UNREACHABLE_BASE_URL, trust_env=True
                ) as client:
                    await client.models.list()
            ok = reached_stand_in("trust_env=True", received) and ok
        except VeniceError as e:
            print(f"   ❌ The request did not go through the proxy: {type(e).__name__}: {e}")
            ok = False

    # connector_limit_per_host caps connections to one host; the total stays at its default.
    async with VeniceClient(api_key=PLACEHOLDER_KEY, connector_limit_per_host=3) as client:
        limits = client.connection_limits
        print(
            f"   - connector_limit_per_host=3: {limits.limit} total, {limits.limit_per_host} per host"
        )
        if limits.limit_per_host != 3:
            print("   ❌ The per-host limit did not reach the connector")
            ok = False

    # http_transport_options goes to aiohttp.TCPConnector, and its limits win
    # over connector_limit and connector_limit_per_host.
    async with VeniceClient(
        api_key=PLACEHOLDER_KEY,
        connector_limit_per_host=3,
        http_transport_options={"limit": 7, "limit_per_host": 2},
    ) as client:
        limits = client.connection_limits
        print(
            f"   - http_transport_options: {limits.limit} total, {limits.limit_per_host} per host"
        )
        if (limits.limit, limits.limit_per_host) != (7, 2):
            print("   ❌ http_transport_options did not reach the connector")
            ok = False
    return ok


async def configuration_presets() -> bool:
    """Show the ready-made configurations next to the hand-built ones above.

    Nothing is sent: the development, testing and developer clients are built
    with a placeholder key and their settings read back, and the production
    variants are built as configs. A production config contacts Redis only once
    a client starts using it; ``examples/advanced/redis_backend.py`` runs one.
    """
    print("\n📦 Configuration Presets (nothing sent)")
    print("-" * 40)

    ok = True
    for name, config in (
        ("create_development_config()", create_development_config(enable_debug=False)),
        ("create_testing_config()", create_testing_config()),
    ):
        print(f"\n✅ {name}")
        async with VeniceClientFactory.create_client(
            config=config, api_key=PLACEHOLDER_KEY
        ) as client:
            ok = (
                describe_client(
                    client,
                    timeout=config.http_client.timeout,
                    max_retries=config.http_client.max_retries,
                    pool_limit=config.http_client.max_connections,
                )
                and ok
            )

    # A client rather than a config: the development preset, with a shorter
    # timeout and fewer retries so that failures surface quickly.
    print("\n✅ create_developer_client()")
    async with create_developer_client(api_key=PLACEHOLDER_KEY) as client:
        ok = describe_client(client, timeout=30.0, max_retries=1, pool_limit=20) and ok

    # The production variants live in venice_ai.presets.production, and all of
    # them need a Redis URL. This host is a placeholder; building contacts nothing.
    redis_url = "redis://redis.internal:6379/0"
    print("\n✅ Production presets (Redis URL required)")
    for name, config in (
        ("create_production_config", create_production_config(redis_url=redis_url)),
        (
            "create_production_config_high_throughput",
            create_production_config_high_throughput(redis_url=redis_url),
        ),
        (
            "create_production_config_conservative",
            create_production_config_conservative(redis_url=redis_url),
        ),
    ):
        http = config.http_client
        print(
            f"   - {name}: timeout {http.timeout:.0f}s, "
            f"{http.max_connections} connections, {http.max_retries} retries"
        )
        limiter = config.rate_limiter
        if config.backend.backend_type != BackendType.REDIS or (
            limiter is None or limiter.mode != RateLimiterMode.ADAPTIVE
        ):
            print(f"   ❌ {name} should pair the Redis backend with the ADAPTIVE rate limiter")
            ok = False
    return ok


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
        ("client_options", await client_options()),
        ("configuration_presets", await configuration_presets()),
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
    print("   - Proxies, default headers and connection limits")
    print("   - The development, testing and production presets")
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
