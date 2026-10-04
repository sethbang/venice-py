#!/usr/bin/env python3
"""
Venice AI SDK - x402: Solana Settlement
=======================================

Top up your Venice prepaid USDC ledger from a **Solana** wallet, using the
x402 "exact" SVM settlement path. This mirrors the EVM/Base flow
(``client.x402.top_up_with``) but settles in USDC on Solana mainnet.

Two pieces:

1. ``SolanaX402Auth(private_key=...)`` — wraps a base58 Solana secret key and
   builds the base64 ``X-402-Payment`` envelope (a partially-signed
   ``VersionedTransaction``; Venice's facilitator sponsors the network fee).
2. ``client.x402.top_up_with_solana(auth=..., amount_usdc=..., max_amount_usdc=...)``
   — runs the full probe → sign → submit handshake in one call.

**This moves real funds.** USDC on Solana mainnet is real money and on-chain
settlement is irreversible. So this example is a **dry run by default**: it
derives the wallet address, reads its on-chain USDC balance and its Venice
prepaid balance, and checks that Venice offers a Solana payment option, but does
**not** submit a top-up unless you explicitly opt in by setting::

    export X402_SOLANA_DO_TOPUP=1

The minimum top-up is $5. Install the extra first:

    pip install 'venice-py[x402-solana]'    # pulls solders

Credentials (never commit): set ``VENICE_X402_SOLANA_TEST_PRIVATE_KEY`` to the
base58 secret key. Optionally set ``VENICE_X402_SOLANA_RPC_URL`` to a private
RPC endpoint (the public default rate-limits).
"""

import asyncio
import os
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import aiohttp

from venice_ai import VeniceClient
from venice_ai.exceptions import PaymentRequiredError, VeniceError

if TYPE_CHECKING:
    from venice_ai.auth.x402_solana import SolanaX402Auth

# Exit codes: 0 = the wallet's on-chain balance and Venice's side were both
# verified (or an opted-in top-up settled), 1 = a failure, 77 = skipped: the
# extra or the wallet key is missing, or the on-chain balance could not be read.
EXIT_SKIPPED = 77

TOPUP_AMOUNT_USDC = 5.0  # Venice's documented minimum


def _rpc_host(rpc_url: str) -> str:
    """Scheme and host only: private RPC URLs often embed an API key."""
    parts = urlsplit(rpc_url)
    return f"{parts.scheme}://{parts.hostname}"


async def _usdc_balance(rpc_url: str, owner: str, mint: str) -> float | None:
    """Read the wallet's on-chain USDC balance via Solana JSON-RPC.

    Returns the summed UI amount across the owner's USDC token accounts, or
    None if the read fails (network error, HTTP error, or a JSON-RPC ``error``
    body such as the public endpoint's rate limit). Read-only — no signing.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            owner,
            {"mint": mint},
            {"encoding": "jsonParsed"},
        ],
    }
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session,
            session.post(rpc_url, json=payload) as resp,
        ):
            resp.raise_for_status()
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        # aiohttp error messages echo the full URL, so print only the error type.
        status = getattr(e, "status", None)
        detail = f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__
        print(f"   ⚠️ Could not read on-chain balance from {_rpc_host(rpc_url)}: {detail}")
        return None

    # JSON-RPC reports failures in an "error" member with HTTP 200, so an
    # absent "result" must not be read as an empty (zero-balance) wallet.
    if "error" in data or "result" not in data:
        print(f"   ⚠️ Solana RPC returned an error instead of a balance: {data.get('error')}")
        return None

    total = 0.0
    for acct in data["result"].get("value") or []:
        info = acct["account"]["data"]["parsed"]["info"]["tokenAmount"]
        total += float(info.get("uiAmount") or 0.0)
    return total


async def check_venice_side(auth: "SolanaX402Auth") -> bool:
    """Read the wallet's Venice prepaid balance and find the Solana payment option.

    Both calls are free: the balance read is signed in with the wallet (SIWS),
    and an empty top-up POST answers 402 with the accepted payment options.
    """
    # Imported lazily so the module loads without the x402-solana extra;
    # solana_top_up() has already imported it under its ImportError guard.
    from venice_ai.auth.x402_solana import USDC_SOLANA_MAINNET, is_solana_mainnet

    async with VeniceClient() as client:
        try:
            balance = await client.x402.balance(auth=auth)
        except VeniceError as e:
            print(f"   ❌ Venice prepaid balance lookup failed: {e}")
            return False
        print(f"   Venice prepaid balance: ${balance.data.balanceUsd:,.4f}")

        try:
            await client.x402.top_up()
        except PaymentRequiredError as e:
            body = e.body
        except VeniceError as e:
            print(f"   ❌ Discovery failed with {type(e).__name__} instead of a 402: {e}")
            return False
        else:
            print("   ❌ The server credited a top-up without a payment header")
            return False

    accepts = body.get("accepts") if isinstance(body, dict) else None
    solana = next((o for o in accepts or [] if is_solana_mainnet(o.get("network"))), None)
    if solana is None:
        print(f"   ❌ Venice offered no Solana mainnet payment option: {body!r}")
        return False
    if solana.get("asset") != USDC_SOLANA_MAINNET:
        print(f"   ❌ Unexpected Solana asset {solana.get('asset')!r}; expected USDC")
        return False
    minimum = int(str(solana.get("amount", 0))) / 10**6
    print(f"   Venice accepts USDC on Solana mainnet, minimum top-up ${minimum:,.2f}")
    return True


async def solana_top_up() -> bool | None:
    """Derive the wallet, read its USDC balance, and (opt-in) top up.

    Returns ``True`` when the dry run verified both the on-chain balance and
    Venice's side (or an opted-in top-up settled), ``False`` on a failure, and
    ``None`` when a prerequisite is missing: the extra, the wallet key, or a
    readable on-chain balance.
    """
    print("🪙 x402 Solana settlement")
    print("-" * 40)

    secret = os.environ.get("VENICE_X402_SOLANA_TEST_PRIVATE_KEY")
    if not secret:
        print("SKIPPED: set VENICE_X402_SOLANA_TEST_PRIVATE_KEY (base58 secret) to run")
        return None

    # The Solana helpers require the [x402-solana] extra (solders).
    try:
        from venice_ai.auth.x402_solana import (
            DEFAULT_SOLANA_RPC_URL,
            USDC_SOLANA_MAINNET,
            SolanaX402Auth,
        )
    except ImportError:
        print("SKIPPED: install the extra with  pip install 'venice-py[x402-solana]'")
        return None

    auth = SolanaX402Auth(private_key=secret)
    rpc_url = os.environ.get("VENICE_X402_SOLANA_RPC_URL", DEFAULT_SOLANA_RPC_URL)
    print(f"   Wallet:  {auth.wallet_address}")
    print(f"   RPC:     {_rpc_host(rpc_url)}")

    balance = await _usdc_balance(rpc_url, auth.wallet_address, USDC_SOLANA_MAINNET)
    if balance is not None:
        print(f"   On-chain USDC balance: ${balance:,.4f}")
    short = balance is not None and balance < TOPUP_AMOUNT_USDC

    # Venice's side is checked even when the RPC read failed: a real failure
    # there outranks a skip.
    if not await check_venice_side(auth):
        return False

    opted_in = os.environ.get("X402_SOLANA_DO_TOPUP") == "1"
    if balance is None:
        # The on-chain read is this example's own safety check before settling.
        if opted_in:
            # A requested top-up that did not happen is a failure, not a skip.
            print("\n   ❌ Refusing to settle: the wallet balance could not be verified.")
            return False
        print(
            "SKIPPED: the wallet's on-chain USDC balance could not be read; set "
            "VENICE_X402_SOLANA_RPC_URL to a reachable RPC endpoint"
        )
        return None

    # Safety gate: only settle real funds when explicitly opted in.
    if not opted_in:
        print(f"\n   💤 Dry run — not submitting. This would top up ${TOPUP_AMOUNT_USDC:.2f}")
        print("      (real, irreversible USDC settlement on Solana mainnet).")
        if short:
            print(
                f"      ⚠️ The wallet holds less than ${TOPUP_AMOUNT_USDC:.2f}; "
                "fund it before opting in."
            )
        print("      Set X402_SOLANA_DO_TOPUP=1 to execute it.")
        return True

    if short:
        print(
            f"\n   ❌ Wallet holds ${balance:,.4f}, below the "
            f"${TOPUP_AMOUNT_USDC:.2f} top-up — fund it first."
        )
        return False

    print(f"\n   💸 Submitting a ${TOPUP_AMOUNT_USDC:.2f} top-up (real funds)…")
    async with VeniceClient() as client:  # Bearer auth (VENICE_API_KEY) routes the request
        try:
            result = await client.x402.top_up_with_solana(
                auth=auth,
                amount_usdc=TOPUP_AMOUNT_USDC,
                max_amount_usdc=TOPUP_AMOUNT_USDC,  # refuse to sign for more than this
                rpc_url=rpc_url,
            )
        except (VeniceError, ValueError, RuntimeError) as e:
            print(f"   ❌ Top-up failed ({type(e).__name__}): {e}")
            return False

    data = result.data
    print("   ✅ Top-up settled on-chain")
    print(f"      Wallet:      {data.walletAddress}")
    print(f"      Credited:    ${data.amountCredited:,.4f}")
    print(f"      New balance: ${data.newBalance:,.4f}")
    print(f"      Payment ID:  {data.paymentId}")
    return True


async def main() -> int:
    print("🚀 Venice AI x402 — Solana Settlement Example")
    print("=" * 50)

    outcome = await solana_top_up()
    if outcome is None:
        return EXIT_SKIPPED
    if not outcome:
        print("\n❌ Solana settlement example failed.")
        return 1

    print("\n✨ Done.")
    print("\n💡 Key concepts demonstrated:")
    print("   - SolanaX402Auth(private_key=...) → base58 wallet address")
    print("   - Reading on-chain USDC balance before settling")
    print("   - client.x402.balance(auth=SolanaX402Auth) and the 402 discovery envelope")
    print("   - client.x402.top_up_with_solana(amount_usdc=, max_amount_usdc=) (opt-in only)")
    print("   - Dry-run-by-default safety gate for irreversible on-chain spend")
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
