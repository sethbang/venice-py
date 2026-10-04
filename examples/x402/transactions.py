#!/usr/bin/env python3
"""
Venice AI SDK - x402: Transaction Ledger
========================================

Demonstrates ``client.x402.transactions(auth=...)`` — Venice's wallet
ledger endpoint. Returns the current balance plus a page of past ledger
entries (top-ups and usage debits) for the SIWE-authenticated wallet.

**Install the optional extra** first::

    pip install 'venice-py[x402]'

**Private key safety:** Read the key from an environment variable or a
secure vault — never hardcode. This example reads
``VENICE_X402_TEST_PRIVATE_KEY`` (the same variable the ``venice-py health
--wallet`` CLI uses), falling back to ``X402_WALLET_PRIVATE_KEY``. Use a
dedicated test wallet when experimenting.
"""

import asyncio
import os
import sys
from datetime import datetime

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError

WALLET_KEY_ENV_VARS = ("VENICE_X402_TEST_PRIVATE_KEY", "X402_WALLET_PRIVATE_KEY")
SHOW_ENTRIES = 10
# A small page size so the full walk below really crosses page boundaries;
# in production use the default (100, the server maximum).
WALK_PAGE_SIZE = 5

# Exit codes: 0 = ledger read and verified, 1 = a failure, 77 = skipped
# because the x402 extra or a wallet key is missing.
EXIT_SKIPPED = 77


def _wallet_private_key() -> tuple[str, str] | None:
    """Return ``(env_var_name, private_key)`` for the first variable that is set."""
    for name in WALLET_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def _usd(amount: float) -> str:
    """Format a signed USD amount as ``-$0.0003`` / ``+$5.0000``."""
    sign = "-" if amount < 0 else "+"
    return f"{sign}${abs(amount):,.4f}"


async def list_transactions() -> bool | None:
    """Fetch and print the x402 wallet ledger.

    Returns ``True`` when the ledger was read and is consistent, ``None`` when
    the optional extra or the wallet key is absent, and ``False`` when a lookup
    fails or the ledger contradicts itself.
    """
    try:
        from venice_ai.auth.x402 import X402Auth
    except ImportError as e:
        print(f"SKIPPED: the x402 extra is not installed (pip install 'venice-py[x402]'): {e}")
        return None

    found = _wallet_private_key()
    if found is None:
        print(
            f"SKIPPED: set {' or '.join(WALLET_KEY_ENV_VARS)} to a test wallet's key; "
            "never commit a production private key"
        )
        return None

    env_name, private_key = found
    auth = X402Auth(private_key=private_key)
    print(f"🔑 Wallet: {auth.wallet_address} (from {env_name})")

    async with VeniceClient() as client:
        try:
            # One page, to show the response shape and its pagination block.
            result = await client.x402.transactions(auth=auth, limit=WALK_PAGE_SIZE)
            # Every page: iter_transactions follows pagination.hasMore for you.
            entries = [
                entry
                async for entry in client.x402.iter_transactions(
                    auth=auth, page_size=WALK_PAGE_SIZE
                )
            ]
        except VeniceError as e:
            print(f"❌ Transactions lookup failed: {e}")
            return False

    data = result.data
    print("\n📜 Transaction Ledger")
    print("-" * 30)
    print(f"   Current balance:  ${data.currentBalance:.4f}")
    print(
        f"   First page:       {len(data.transactions)} entries (limit={data.pagination.limit}, "
        f"offset={data.pagination.offset}, hasMore={data.pagination.hasMore})"
    )
    print(f"   Full walk:        {len(entries)} entries (page_size={WALK_PAGE_SIZE})")

    if not entries:
        print("\n   (no entries yet — try top_up.py first)")
        return True

    ok = True
    ids = [entry.id for entry in entries]
    if len(set(ids)) != len(ids):
        print("❌ The walk returned the same entry on more than one page")
        ok = False

    # Sort explicitly rather than rely on the server's ordering.
    newest_first = sorted(entries, key=lambda e: datetime.fromisoformat(e.createdAt), reverse=True)
    if abs(newest_first[0].balanceAfter - data.currentBalance) > 1e-6:
        print(
            f"❌ The newest entry's balance after (${newest_first[0].balanceAfter:,.4f}) does not "
            f"match the current balance (${data.currentBalance:,.4f})"
        )
        ok = False
    # Each entry's balance after should be the previous one plus its amount.
    oldest_first = newest_first[::-1]
    for before, entry in zip(oldest_first, oldest_first[1:], strict=False):
        if abs(before.balanceAfter + entry.amount - entry.balanceAfter) > 1e-6:
            print(f"❌ Entry {entry.id} does not follow from the one before it")
            ok = False

    shown = newest_first[:SHOW_ENTRIES]
    print(f"\n   Recent entries (newest first, showing {len(shown)} of {len(entries)}):")
    for entry in shown:
        print(
            f"   • {entry.createdAt}  {entry.type:<10} "
            f"{_usd(entry.amount)}  (balance after: ${entry.balanceAfter:,.4f})"
        )
        if entry.modelId:
            print(f"     model: {entry.modelId}")
        if entry.requestId:
            print(f"     request: {entry.requestId}")

    return ok


async def main() -> int:
    """Run the x402 transactions demo.

    Returns ``0`` if the ledger was read and is consistent, ``1`` if a lookup
    failed or the ledger contradicts itself, and ``77`` when the extra or a
    wallet key is missing.
    """
    print("🚀 Venice AI x402 — Transactions Example")
    print("=" * 50)

    outcome = await list_transactions()
    if outcome is None:
        return EXIT_SKIPPED
    if not outcome:
        print("\n❌ Transactions example failed.")
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
