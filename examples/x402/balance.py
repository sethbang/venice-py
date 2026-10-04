#!/usr/bin/env python3
"""
Venice AI SDK - x402: Wallet Balance
====================================

Demonstrates ``client.x402.balance(auth=...)`` — Venice's wallet-based
billing endpoint that returns the current prepaid USDC balance for the
wallet authenticated via SIWE (Sign-In-With-Ethereum / EIP-4361).

**Install the optional extra** first::

    pip install 'venice-py[x402]'

This pulls ``eth-account`` and ``siwe`` for EIP-4361 message signing.

**Private key safety:** Read the key from an environment variable or a
secure vault — never hardcode it. This example reads
``VENICE_X402_TEST_PRIVATE_KEY`` (the same variable the ``venice-py health
--wallet`` CLI uses), falling back to ``X402_WALLET_PRIVATE_KEY``. Use a
dedicated test wallet, not your main key.
"""

import asyncio
import os
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError

WALLET_KEY_ENV_VARS = ("VENICE_X402_TEST_PRIVATE_KEY", "X402_WALLET_PRIVATE_KEY")

# Exit codes: 0 = balance read and verified, 1 = a failure, 77 = skipped
# because the x402 extra or a wallet key is missing.
EXIT_SKIPPED = 77


def _wallet_private_key() -> tuple[str, str] | None:
    """Return ``(env_var_name, private_key)`` for the first variable that is set."""
    for name in WALLET_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


async def show_balance() -> bool | None:
    """Read the wallet balance with SIWE auth.

    Returns ``True`` on success, ``None`` when the optional extra or wallet key
    is not configured, and ``False`` when the balance lookup fails.
    """
    try:
        from venice_ai.auth.x402 import X402Auth
    except ImportError as e:
        print(f"SKIPPED: the x402 extra is not installed (pip install 'venice-py[x402]'): {e}")
        return None

    found = _wallet_private_key()
    if found is None:
        print(f"SKIPPED: set {' or '.join(WALLET_KEY_ENV_VARS)} to a test wallet's key")
        return None

    env_name, private_key = found
    auth = X402Auth(private_key=private_key)
    print(f"🔑 Wallet: {auth.wallet_address} (from {env_name})")

    async with VeniceClient() as client:
        try:
            balance = await client.x402.balance(auth=auth)
        except VeniceError as e:
            print(f"❌ Balance lookup failed: {e}")
            return False

    data = balance.data
    print("\n💰 x402 Balance")
    print("-" * 30)
    print(f"   Address:          {data.walletAddress}")
    print(f"   Balance (USD):    ${data.balanceUsd:,.4f}")
    print(f"   Can consume?      {data.canConsume}")
    if data.minimumTopUpUsd is not None:
        print(f"   Minimum top-up:   ${data.minimumTopUpUsd:,.2f}")
    if data.suggestedTopUpUsd is not None:
        print(f"   Suggested top-up: ${data.suggestedTopUpUsd:,.2f}")
    if data.diemBalanceUsd is not None:
        print(f"   Diem balance:     ${data.diemBalanceUsd:,.4f}")

    if data.walletAddress.lower() != auth.wallet_address.lower():
        print("❌ The server reported a different wallet than the one that signed in")
        return False
    return True


async def main() -> int:
    print("🚀 Venice AI x402 — Balance Example")
    print("=" * 50)
    outcome = await show_balance()
    if outcome is None:
        return EXIT_SKIPPED
    if not outcome:
        print("\n❌ Balance example failed.")
        return 1
    print("\n✨ Done.")
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
