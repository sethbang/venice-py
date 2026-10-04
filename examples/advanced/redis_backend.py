#!/usr/bin/env python3
"""
Venice AI SDK - Redis Backend Configuration
============================================

Demonstrates the *correct* wiring for a Redis-backed Venice client and
self-verifies that the SDK's rate limiter state really lives in Redis.

Key correctness note
--------------------
``BackendConfig(backend_type=BackendType.REDIS, ...)`` alone is **not** enough
to make the SDK use Redis at runtime: it also requires
``RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url=...)``. Without
ADAPTIVE mode the SDK uses the in-memory SimpleRateLimiter and Redis is never
contacted. The config validator flags that combination as an error (see
``venice_ai.validation.config_validator``); the production preset wires both
pieces and is the canonical entrypoint.

Run
---
Start Redis (any reachable instance works; localhost shown for demo)::

    docker run -d --name venice-redis -p 6379:6379 redis:7-alpine

Then::

    poetry run python examples/advanced/redis_backend.py

The Redis URL comes from the first of ``VENICE_BACKEND__REDIS__REDIS_URL``
(the variable ``VeniceAIConfig`` itself reads), ``VENICE_REDIS_URL`` and
``REDIS_URL`` that is set, falling back to ``redis://localhost:6379``.

Exit codes: 0 when the limiter state was verified in Redis, 1 on a failure,
and 77 (with a ``SKIPPED:`` line) when a prerequisite is missing: the
``redis`` or ``adaptive-rate-limiter`` package (``pip install "venice-py[adaptive]"``)
or a reachable Redis.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import sys
from typing import cast

# Exit code for "a prerequisite is missing", so a test runner can tell a skip
# from a failure.
EXIT_SKIPPED = 77

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:
    print('SKIPPED: the redis package is not installed (pip install "venice-py[adaptive]")')
    sys.exit(EXIT_SKIPPED)
# ADAPTIVE mode also needs the adaptive-rate-limiter package from the same extra.
if importlib.util.find_spec("adaptive_rate_limiter") is None:
    print(
        "SKIPPED: the adaptive-rate-limiter package is not installed "
        '(pip install "venice-py[adaptive]")'
    )
    sys.exit(EXIT_SKIPPED)

# Line-buffer stdout so our prints show up in chronological order with any
# stderr output the package may emit (rather than appearing in a chunk at
# the end after stderr already flushed).
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from venice_ai.exceptions import NoMatchingModelError  # noqa: E402
from venice_ai.factory import VeniceClientFactory  # noqa: E402
from venice_ai.presets import create_production_config  # noqa: E402
from venice_ai.types.api import UserMessage, VeniceParameters  # noqa: E402
from venice_ai.validation.config_validator import validate_config  # noqa: E402

REDIS_URL_ENV_VARS = ("VENICE_BACKEND__REDIS__REDIS_URL", "VENICE_REDIS_URL", "REDIS_URL")
DEFAULT_REDIS_URL = "redis://localhost:6379"

# The adaptive limiter scopes its Redis keys by account id.
ACCOUNT_ID = "redis-example"


def resolve_redis_url() -> str:
    """Return the Redis URL from the environment, naming where it came from."""
    for name in REDIS_URL_ENV_VARS:
        value = os.getenv(name)
        if value:
            print(f"Redis URL from {name}")
            return value
    print(f"Redis URL: none of {', '.join(REDIS_URL_ENV_VARS)} is set; using the default")
    return DEFAULT_REDIS_URL


def assert_redis_reachable(redis_url: str) -> redis.Redis:
    """Ping Redis up-front; exit with a SKIPPED message if it is unreachable.

    Returns a sync ``redis.Redis`` client used later to inspect the keys the
    SDK wrote.
    """
    print(f"Checking Redis connectivity at {redis_url} ...")
    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2.0)
    try:
        client.ping()
    except RedisError as exc:
        print(
            f"\nSKIPPED: cannot reach Redis at {redis_url}: {exc}\n"
            "\nStart a local Redis (Docker):\n"
            "    docker run -d --name venice-redis -p 6379:6379 redis:7-alpine\n"
            f"\nOr point one of {', '.join(REDIS_URL_ENV_VARS)} at a reachable instance."
        )
        sys.exit(EXIT_SKIPPED)
    print("   Redis reachable.")
    return client


def _b64(value: str) -> str:
    """Encode a key component the way the adaptive limiter does."""
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def model_state_key(model: str) -> str:
    """The Redis hash holding the adaptive limiter's state for one model."""
    return f"rl:{{{_b64(ACCOUNT_ID)}|{_b64(model)}}}:state"


async def run_redis_backed_request(redis_url: str, verifier: redis.Redis) -> bool:
    """Build a Redis-backed config, issue one request, then verify the limiter
    state for this account and model was written to Redis."""

    print("\nBuilding production config (BackendType.REDIS + RateLimiterMode.ADAPTIVE)")
    # The production preset is the canonical way to get this right. It wires:
    #   BackendConfig(backend_type=BackendType.REDIS, redis=RedisBackendConfig(...))
    # AND
    #   RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url=redis_url)
    config = create_production_config(
        redis_url=redis_url,
        max_concurrent_executions=10,
        max_queue_size=100,
        # Localhost is fine for a local demo but rejected by default in
        # production mode.
        _allow_localhost_for_testing=True,
    )

    # With both pieces wired the validator reports no errors.
    validation = validate_config(config)
    print(
        f"   Config validation: errors={len(validation.errors)} warnings={len(validation.warnings)}"
    )
    for warning in validation.warnings:
        print(f"   WARNING: {warning}")
    if validation.errors:
        for err in validation.errors:
            print(f"   ERROR: {err}", file=sys.stderr)
        return False

    client = VeniceClientFactory.create_client(
        config=config,
        api_key=os.environ["VENICE_API_KEY"],
        account_id=ACCOUNT_ID,
    )

    async with client:
        chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        state_key = model_state_key(chat_model)
        # Start from a clean slate so the check below only sees this run's writes.
        verifier.delete(state_key)

        print(f"\nIssuing chat completion via {chat_model} ...")
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Reply with exactly one word: OK")],
            max_completion_tokens=16,
            # Only this prompt is billed, not Venice's own system prompt.
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )
        finish = response.choices[0].finish_reason
        text = (response.text or "").strip()
        print(f"   Response: {text!r} (finish_reason={finish})")
        if finish == "length" or not text:
            print("FAIL: the reply was truncated or empty.", file=sys.stderr)
            return False

    # Verification: the limiter state for this account and model is in Redis.
    state = {
        k.decode(): v.decode()
        for k, v in cast(dict[bytes, bytes], verifier.hgetall(state_key)).items()
    }
    keys = sorted(k.decode() for k in verifier.scan_iter(match=f"rl:{{{_b64(ACCOUNT_ID)}|*"))
    print(f"\nRedis keys for account {ACCOUNT_ID!r}:")
    for key in keys:
        print(f"   - {key}")

    if not state:
        print(
            f"\nFAIL: {state_key} was not written.\n"
            "      The client is not keeping its rate limiter state in Redis; check\n"
            "      that the config uses BackendType.REDIS with RateLimiterMode.ADAPTIVE.",
            file=sys.stderr,
        )
        return False

    print(f"\nState in {state_key}:")
    for field in ("lim_req", "rem_req", "lim_tok", "rem_tok", "vrf_req", "vrf_tok"):
        print(f"   {field} = {state.get(field, '(not set)')}")
    if state.get("vrf_req") != "1":
        print(
            "\nFAIL: the request limit was not verified from response headers (vrf_req != 1).",
            file=sys.stderr,
        )
        return False

    print("\nSUCCESS: the adaptive limiter stored header-verified state in Redis.")
    return True


async def main() -> int:
    print("=" * 60)
    print("Venice AI SDK - Redis Backend Example")
    print("=" * 60)

    redis_url = resolve_redis_url()

    # Confirm Redis is reachable before configuring the SDK against it;
    # otherwise failures get buried in async stacks.
    verifier = assert_redis_reachable(redis_url)

    ok = await run_redis_backed_request(redis_url, verifier)

    print("\n" + "=" * 60)
    print("Done." if ok else "Failed.")
    print("=" * 60)
    return 0 if ok else 1


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
        print("\nInterrupted.")
        sys.exit(130)
    except NoMatchingModelError as exc:
        # The catalog has no model of the kind this example needs.
        print(f"SKIPPED: {exc}")
        sys.exit(EXIT_SKIPPED)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
