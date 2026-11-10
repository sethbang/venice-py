#!/usr/bin/env python3
"""
Venice AI SDK - Configuration from Environment Variables
========================================================

``VeniceAIConfig`` reads ``VENICE_``-prefixed variables, with a double underscore for
each nested level, so ``VENICE_HTTP_CLIENT__TIMEOUT`` sets ``http_client.timeout``.
This needs the optional ``pydantic-settings`` package
(``pip install 'venice-py[enterprise]'``).

This script shows which ways of building a client apply the value. With the variable
set:

- ``VeniceClientFactory.create_client(config=config)`` applies it.
- ``VeniceClient(config=config)`` applies it.
- ``VeniceClient()`` keeps its 120s default for every request.

With nothing setting the timeout, the two ways of using a config differ:

- ``VeniceClient(config=config)`` keeps the client's own 120s.
- ``VeniceClientFactory.create_client(config=config)`` uses the config's 30s field
  default, which also bounds streams. Pass a timeout if a long stream needs more.

The variable is changed only while the script runs, and restored afterwards.

Cost: free. Nothing is sent: the clients are built with a placeholder key and their
settings read back, so no API key is needed.

Exit status: ``0`` when every value matched, ``1`` when any differed, ``77`` (with a
``SKIPPED:`` line) when ``pydantic-settings`` is not installed.
"""

import asyncio
import contextlib
import importlib.util
import os
import sys
from collections.abc import Iterator

from venice_ai import VeniceAIConfig, VeniceClient, VeniceClientFactory

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77

TIMEOUT_VARIABLE = "VENICE_HTTP_CLIENT__TIMEOUT"
#: Differs from every default, so seeing it proves the variable was read.
CONFIGURED_TIMEOUT = 7.0
#: A client's timeout when nothing configures it.
DEFAULT_TIMEOUT = 120.0
#: ``HttpClientConfig.timeout``'s field default, which the factory applies.
CONFIG_FIELD_DEFAULT = 30.0

PLACEHOLDER_KEY = "placeholder-key-env-configuration"


@contextlib.contextmanager
def environment_variable(name: str, value: str | None) -> Iterator[None]:
    """Set or unset one environment variable for a block, then restore its old state."""
    saved = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def check(label: str, got: float, expected: float) -> bool:
    """Print one timeout and whether it is the expected value."""
    mark = "✓" if got == expected else "❌"
    print(f"   {mark} {label}: {got}s (expected {expected}s)")
    return got == expected


async def factory_timeout(config: VeniceAIConfig, expected: float) -> bool:
    """Check the timeout of a client built by the factory from ``config``."""
    client = VeniceClientFactory.create_client(config=config, api_key=PLACEHOLDER_KEY)
    async with client:
        return check("VeniceClientFactory.create_client(config=config)", client.timeout, expected)


async def direct_timeout(config: VeniceAIConfig, expected: float) -> bool:
    """Check the timeout of a client built directly from ``config``."""
    async with VeniceClient(api_key=PLACEHOLDER_KEY, config=config) as client:
        return check("VeniceClient(config=config)", client.timeout, expected)


async def main() -> int:
    """Show which ways of building a client honour the environment variable."""
    print("🌱 Configuration from Environment Variables")
    print("=" * 50)

    if importlib.util.find_spec("pydantic_settings") is None:
        print(
            "SKIPPED: reading VENICE_* variables needs pydantic-settings "
            "(pip install 'venice-py[enterprise]')"
        )
        return EXIT_SKIPPED

    print(f"{TIMEOUT_VARIABLE}={CONFIGURED_TIMEOUT}")
    with environment_variable(TIMEOUT_VARIABLE, str(CONFIGURED_TIMEOUT)):
        config = VeniceAIConfig()
        results = [
            check(
                "VeniceAIConfig().http_client.timeout",
                config.http_client.timeout,
                CONFIGURED_TIMEOUT,
            ),
            await factory_timeout(config, CONFIGURED_TIMEOUT),
            await direct_timeout(config, CONFIGURED_TIMEOUT),
        ]
        async with VeniceClient(api_key=PLACEHOLDER_KEY) as client:
            results.append(check("VeniceClient()", client.timeout, DEFAULT_TIMEOUT))

    print(f"\n{TIMEOUT_VARIABLE} unset")
    with environment_variable(TIMEOUT_VARIABLE, None):
        config = VeniceAIConfig()
        results.append(await factory_timeout(config, CONFIG_FIELD_DEFAULT))
        results.append(await direct_timeout(config, DEFAULT_TIMEOUT))

    if not all(results):
        print("\n❌ A client's timeout differs from what this script describes")
        return 1

    print("\n💡 A timeout you set applies whichever way the client is built. With none set, the")
    print("   factory's 30s also bounds a stream, so give a long stream a longer timeout.")
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
