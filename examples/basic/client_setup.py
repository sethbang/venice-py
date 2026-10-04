#!/usr/bin/env python3
"""
Venice AI SDK - Client Setup Examples
====================================

This example demonstrates different ways to configure and set up the Venice AI client
for various use cases and environments.

Every live section verifies the client against ``api_keys.get_rate_limits()``,
a free endpoint that requires a valid API key. (``models.list()`` is public, so
it would succeed even with a wrong key and prove nothing about the setup.)

The API root comes from ``base_url=``, ``VeniceAIConfig.api_base_url`` or the
``VENICE_API_BASE_URL`` environment variable, in the form
``https://api.venice.ai/api/v1`` (version path included). The sections print the
root each client actually uses.

The production section needs a reachable Redis server. The Redis URL is the
first of these that is set:

1. ``VENICE_BACKEND__REDIS__REDIS_URL`` — the SDK's own setting
   (``VeniceAIConfig.backend.redis.redis_url``), so it always wins;
2. ``VENICE_REDIS_URL`` — a convenience name the examples also read;
3. ``REDIS_URL`` — the conventional name many hosts set.

Without one the section prints a ``Section skipped:`` line. The script still
exits 0 when every other section passed, since Redis is optional.
"""

import asyncio
import os
import sys

from venice_ai import VeniceAIConfig, VeniceClient, VeniceClientFactory
from venice_ai.core.config import (
    BackendConfig,
    BackendType,
    HttpClientConfig,
    RateLimiterConfig,
    RateLimiterMode,
    RedisBackendConfig,
)
from venice_ai.exceptions import AuthenticationError, VeniceError

# Checked in this order: the SDK's own setting first, then the example fallbacks.
REDIS_URL_ENV_VARS = ("VENICE_BACKEND__REDIS__REDIS_URL", "VENICE_REDIS_URL", "REDIS_URL")


async def verify_api_key(client: VeniceClient, indent: str = "   ") -> bool:
    """Make one authenticated call so the check proves the key works."""
    try:
        limits = await client.api_keys.get_rate_limits()
    except VeniceError as e:
        print(f"{indent}❌ Authenticated call failed: {type(e).__name__}: {e}")
        return False
    print(f"{indent}🔐 API key accepted (tier: {limits.data.apiTier.id})")
    return True


async def basic_setup() -> bool:
    """Basic client setup with API key from the environment."""
    print("🔧 Basic Client Setup")
    print("-" * 30)

    # Method 1: Direct instantiation (reads VENICE_API_KEY from environment)
    async with VeniceClient() as client:
        print("✅ Basic client created")
        print(f"   - API root: {client.base_url}")
        print(f"   - Request timeout: {client.timeout}s")
        return await verify_api_key(client)


async def explicit_key_setup() -> bool:
    """Setup with an explicit API key (useful for testing or multi-key setups)."""
    print("\n🔑 Explicit Key Setup")
    print("-" * 30)

    # Method 2: Pass the API key explicitly, e.g. one loaded from a secrets store.
    api_key = os.environ.get("VENICE_API_KEY")
    if not api_key:
        print("❌ VENICE_API_KEY is not set")
        return False

    async with VeniceClient(api_key=api_key) as client:
        print("✅ Client created with explicit API key")
        return await verify_api_key(client)


async def custom_configuration() -> bool:
    """Setup with a custom VeniceAIConfig built through the factory."""
    print("\n⚙️ Custom Configuration Setup")
    print("-" * 30)

    config = VeniceAIConfig(
        # In-process rate-limit state (no external services needed)
        backend=BackendConfig(backend_type=BackendType.MEMORY),
        # HTTP client settings
        http_client=HttpClientConfig(timeout=60.0, max_connections=50),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Custom configured client created")
        print(f"   - API root: {client.base_url} (config: {config.api_base_url})")
        print(f"   - Backend: {config.backend.backend_type.value}")
        print(f"   - Rate limiter: {type(client.rate_limiter).__name__}")
        print(f"   - Request timeout: {client.timeout}s")
        # Read the settings back from the client rather than from the config,
        # so the output shows what the client actually uses.
        limits = client.connection_limits
        print(
            f"   - Connection pool: {limits.limit} total, {limits.limit_per_host or 'no'} per host"
        )
        retry = client.retry_options
        if retry is not None:
            print(f"   - Retries: up to {retry.max_attempts} after the first attempt")
        if limits.limit != config.http_client.max_connections:
            print(
                f"   ❌ Expected a pool of {config.http_client.max_connections} connections, "
                f"the client has {limits.limit}"
            )
            return False
        if client.base_url.rstrip("/") != config.api_base_url.rstrip("/"):
            print("   ❌ The client does not use the API root the config names")
            return False
        return await verify_api_key(client)


def configured_redis_url() -> str | None:
    """Return the first Redis URL set in ``REDIS_URL_ENV_VARS`` order, if any."""
    for name in REDIS_URL_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


async def redis_reachable(redis_url: str) -> bool:
    """Ping Redis so the section never claims a backend it cannot reach."""
    try:
        import redis.asyncio as redis_asyncio
        from redis.exceptions import RedisError
    except ImportError:
        print("   The redis package is missing: pip install 'venice-py[adaptive]'")
        return False

    probe = redis_asyncio.Redis.from_url(redis_url, socket_connect_timeout=2.0)
    try:
        await probe.ping()
        return True
    except (RedisError, OSError) as e:
        print(f"   Redis ping failed: {type(e).__name__}: {e}")
        return False
    finally:
        await probe.aclose()


async def production_setup() -> bool | None:
    """Production-ready client setup with shared rate-limit state in Redis.

    Redis is only contacted by the ADAPTIVE rate limiter, so the backend and the
    rate limiter are configured together. Without a reachable Redis the section
    is skipped (an infrastructure prerequisite, not a code failure) and returns
    ``None``.
    """
    print("\n🏭 Production Setup")
    print("-" * 30)

    redis_url = configured_redis_url()
    if not redis_url:
        print(
            "Section skipped: production_setup needs Redis; "
            f"set one of {', '.join(REDIS_URL_ENV_VARS)}"
        )
        return None
    if not await redis_reachable(redis_url):
        print(
            "Section skipped: production_setup needs Redis, and the configured one is unreachable"
        )
        print("   (examples/advanced/redis_backend.py shows the full Redis walkthrough)")
        return None

    config = VeniceAIConfig(
        # Redis holds the rate-limit state shared by every process using this key
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(redis_url=redis_url, max_connections=20),
        ),
        # The ADAPTIVE limiter is the component that reads and writes Redis
        rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url=redis_url),
        http_client=HttpClientConfig(timeout=30.0, max_connections=100),
    )

    # account_id scopes the Redis keys, so deployments sharing one Redis
    # instance do not collide.
    client = VeniceClientFactory.create_client(config=config, account_id="client-setup-example")

    async with client:
        print("✅ Production client created")
        print(f"   - Backend: {config.backend.backend_type.value} (reachable)")
        print(f"   - Rate limiter: {type(client.rate_limiter).__name__}")
        print(f"   - Max connections: {config.http_client.max_connections}")
        return await verify_api_key(client)


async def test_setup() -> bool:
    """Test-optimized client setup.

    The test client carries a placeholder key, so an authenticated call must be
    rejected with :class:`AuthenticationError`. That rejection proves the
    request reached the API; a 2xx would mean the placeholder was never sent.
    """
    print("\n🧪 Test Setup")
    print("-" * 30)

    from venice_ai import create_test_venice_client

    async with create_test_venice_client(api_key="test-key", enable_redis=False) as client:
        print("✅ Test client created")
        print(f"   - Request timeout: {client.timeout}s")
        print(f"   - Rate limiter: {type(client.rate_limiter).__name__}")
        try:
            await client.api_keys.get_rate_limits()
        except AuthenticationError as e:
            print(f"   📋 Placeholder key rejected as expected: {type(e).__name__}")
            return True
        print("   ❌ The placeholder key was accepted; expected AuthenticationError")
        return False


async def main() -> int:
    """Demonstrate various client setup methods.

    Returns ``1`` if any section failed, otherwise ``0``. The Redis-backed
    production section is optional: when it is skipped the other sections
    still validate the client, so the script exits ``0`` after saying so.
    """
    print("🚀 Venice AI Client Setup Examples")
    print("=" * 50)

    sections = [
        ("basic_setup", basic_setup),
        ("explicit_key_setup", explicit_key_setup),
        ("custom_configuration", custom_configuration),
        ("production_setup", production_setup),
        ("test_setup", test_setup),
    ]

    # Each section returns True (passed), False (failed) or None (skipped).
    results: list[tuple[str, bool | None]] = []
    for name, section in sections:
        try:
            results.append((name, await section()))
        except VeniceError as e:
            print(f"❌ {name} failed: {type(e).__name__}: {e}")
            results.append((name, False))

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    passed = len(results) - len(failed) - len(skipped)

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} setup examples failed: {', '.join(failed)}")
        return 1

    if skipped:
        print(f"\n✅ {passed} setup examples passed, {len(skipped)} skipped: {', '.join(skipped)}")
    else:
        print("\n✨ All client setup examples completed!")
    print("\n💡 Choose the setup method that best fits your use case:")
    print("   - Basic: Simple applications")
    print("   - Explicit key: Keys loaded from a secrets store")
    print("   - Custom: Specific requirements")
    print("   - Production: Shared rate-limit state across processes (needs Redis)")
    print("   - Test: Unit/integration testing")
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
