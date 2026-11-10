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
   — runs the full probe → sign → submit handshake in one call. ``amount_usdc``
   must equal the amount the server requires, or it raises ``ValueError`` before
   signing.

**This moves real funds.** USDC on Solana mainnet is real money and on-chain
settlement is irreversible. So this example is a **dry run by default**: it
derives the wallet address, reads its on-chain USDC balance and its Venice
prepaid balance, checks that Venice offers a Solana payment option, and signs a
payment locally to verify it, but does **not** submit a top-up unless you
explicitly opt in by setting::

    export X402_SOLANA_DO_TOPUP=1

The local signature is over a placeholder blockhash no validator accepts, with
the facilitator's signature slot left empty, so that transaction could never
land. It is never printed or sent, and the signing itself makes no request.

Venice's documented minimum top-up is $5. Install the extra first:

    pip install 'venice-py[x402-solana]'    # pulls solders

Credentials (never commit): set ``VENICE_X402_SOLANA_TEST_PRIVATE_KEY`` to the
base58 secret key. Optionally set ``VENICE_X402_SOLANA_RPC_URL`` to a private
RPC endpoint (the public default rate-limits).

No API key is needed: ``VeniceClient(api_key="", auth=auth)`` authenticates with
the wallet alone.

If the opted-in submit times out, drops or gets a 5xx or an unreadable reply, the
payment may still have settled, and each run signs a fresh payment, so a re-run
can pay again. The example reports that as an unknown outcome, apart from errors
raised before anything was signed, such as a failed RPC read.
"""

import asyncio
import base64
import json
import os
import sys
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import aiohttp

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, PaymentRequiredError, VeniceError

if TYPE_CHECKING:
    from venice_ai.auth.x402_solana import SolanaX402Auth

# Exit codes: 0 = the wallet's on-chain balance, Venice's side and a locally
# signed payment were verified (or an opted-in top-up settled), 1 = a failure or an unknown
# settlement outcome, 77 = skipped: the extra or the wallet key is missing, or
# the on-chain balance could not be read.
EXIT_SKIPPED = 77

TOPUP_AMOUNT_USDC = 5.0  # Venice's documented minimum
USDC_DECIMALS = 6

# What the opted-in flow reads from the chain, and the dry run does not: the SPL
# Token program that owns the USDC mint.
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKmA6xZ4Cc6vE1WoEhaXXS"


def _rpc_host(rpc_url: str) -> str:
    """Scheme and host only: private RPC URLs often embed an API key."""
    parts = urlsplit(rpc_url)
    return f"{parts.scheme}://{parts.hostname}"


def _transport_error(e: Exception) -> str:
    """Error type and status only: aiohttp messages echo the full RPC URL."""
    status = getattr(e, "status", None)
    return f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__


def _recording_auth(private_key: str) -> "tuple[SolanaX402Auth, list[bool]]":
    """Build the wallet, plus a list that gains an entry each time it signs a payment."""
    from venice_ai.auth.x402_solana import SolanaX402Auth

    signed: list[bool] = []

    class RecordingAuth(SolanaX402Auth):
        """Note each signed payment: ``top_up_with_solana`` probes, signs and submits in one call."""

        def build_payment_header(self, *args: Any, **kwargs: Any) -> str:
            header = super().build_payment_header(*args, **kwargs)
            signed.append(True)
            return header

    return RecordingAuth(private_key=private_key), signed


def _report_top_up_error(e: Exception, signed: bool) -> None:
    """Print a failed top-up by what it means for the funds.

    Before signing, nothing can have been spent. After signing, a 4xx is the
    server refusing the payment, and anything else may have settled.
    """
    detail = f"{type(e).__name__}: {e}"
    if not signed:
        print(f"   ❌ Nothing was signed or spent: {detail}")
    elif isinstance(e, APIError) and e.status_code is not None and 400 <= e.status_code < 500:
        print(f"   ❌ The server rejected the signed payment: {detail}")
    else:
        print(f"   ⚠️ {detail}")
        print("      Outcome unknown, check balance.py and transactions.py before retrying:")
        print("      the payment may have settled, and a re-run signs a new one. Both scripts")
        print("      read an EVM wallet, so for this Solana wallet call")
        print("      client.x402.balance(auth=auth) and client.x402.transactions(auth=auth).")


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
        print(
            f"   ⚠️ Could not read on-chain balance from {_rpc_host(rpc_url)}: {_transport_error(e)}"
        )
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


async def check_venice_side(auth: "SolanaX402Auth") -> dict[str, Any] | None:
    """Read the wallet's Venice prepaid balance and find the Solana payment option.

    Both calls are free: the balance read is signed in with the wallet (SIWS),
    and an empty top-up POST answers 402 with the accepted payment options.

    Returns the Solana entry of those options, or ``None`` after a failure.
    """
    # Imported lazily so the module loads without the x402-solana extra;
    # solana_top_up() has already imported it under its ImportError guard.
    from venice_ai.auth.x402_solana import USDC_SOLANA_MAINNET, is_solana_mainnet

    # api_key="" signs in with the wallet alone, even when VENICE_API_KEY is set.
    async with VeniceClient(api_key="", auth=auth) as client:
        try:
            balance = await client.x402.balance(auth=auth)
        except VeniceError as e:
            print(f"   ❌ Venice prepaid balance lookup failed: {e}")
            return None
        print(f"   Venice prepaid balance: ${balance.data.balanceUsd:,.4f}")

        try:
            await client.x402.top_up()
        except PaymentRequiredError as e:
            body = e.body
        except VeniceError as e:
            print(f"   ❌ Discovery failed with {type(e).__name__} instead of a 402: {e}")
            return None
        else:
            print("   ❌ The server credited a top-up without a payment header")
            return None

    accepts = body.get("accepts") if isinstance(body, dict) else None
    solana = next((o for o in accepts or [] if is_solana_mainnet(o.get("network"))), None)
    if solana is None:
        print(f"   ❌ Venice offered no Solana mainnet payment option: {body!r}")
        return None
    if solana.get("asset") != USDC_SOLANA_MAINNET:
        print(f"   ❌ Unexpected Solana asset {solana.get('asset')!r}; expected USDC")
        return None
    required = int(str(solana.get("amount", 0))) / 10**USDC_DECIMALS
    print(f"   Venice accepts USDC on Solana mainnet, and requires ${required:,.2f}")
    return solana


def sign_without_submitting(auth: "SolanaX402Auth", requirement: dict[str, Any]) -> bool:
    """Build the partially signed payment locally and verify it. Sends nothing.

    The all-zero blockhash is never a recent one, and the facilitator's signature
    slot stays empty, so this transaction cannot land even if a copy leaked. It
    is still never printed, stored or sent; only its decoded fields are checked.
    The opted-in flow reads a real blockhash from the RPC endpoint instead.
    """
    from solders.hash import Hash
    from solders.message import to_bytes_versioned
    from solders.pubkey import Pubkey
    from solders.signature import Signature
    from solders.transaction import VersionedTransaction

    print("\n   🖊️ Local signing check (nothing is sent)")
    amount = int(str(requirement.get("amount", 0)))

    def sign(req: dict[str, Any], cap: int) -> str:
        return auth.build_payment_header(
            requirement=req,
            recent_blockhash=str(Hash.default()),
            mint_decimals=USDC_DECIMALS,
            token_program=SPL_TOKEN_PROGRAM,
            max_amount_units=cap,
        )

    def refused(req: dict[str, Any], cap: int) -> bool:
        try:
            sign(req, cap)
        except ValueError:
            return True
        return False

    try:
        header = sign(requirement, int(TOPUP_AMOUNT_USDC * 10**USDC_DECIMALS))
    except ValueError as e:
        print(f"      ❌ build_payment_header refused to sign: {e}")
        return False

    envelope = json.loads(base64.b64decode(header))
    tx = VersionedTransaction.from_bytes(base64.b64decode(envelope["payload"]["transaction"]))
    message = tx.message
    signed_bytes = to_bytes_versioned(message)
    wallet = Pubkey.from_string(auth.wallet_address)
    fee_payer = str(requirement["extra"]["feePayer"])
    transfer = bytes(message.instructions[2].data)
    print(f"      payer {auth.wallet_address} → payTo {requirement.get('payTo')}")
    print(f"      value {amount / 10**USDC_DECIMALS:,.2f} USDC, network fee paid by {fee_payer}")

    checks = {
        "the wallet's signature verifies": tx.signatures[1].verify(wallet, signed_bytes),
        "a copy with one byte changed does not verify": not tx.signatures[1].verify(
            wallet, signed_bytes[:-1] + bytes([signed_bytes[-1] ^ 1])
        ),
        "the facilitator pays the fee, and its signature slot is empty": (
            str(message.account_keys[0]) == fee_payer and tx.signatures[0] == Signature.default()
        ),
        "the four instructions of the exact scheme, transferring the quoted amount": (
            len(message.instructions) == 4 and int.from_bytes(transfer[1:9], "little") == amount
        ),
        "the blockhash is the placeholder, so it cannot land": (
            message.recent_blockhash == Hash.default()
        ),
        "the SDK refuses to sign above the cap": refused(requirement, amount - 1),
        "the SDK refuses to sign for another cluster": refused(
            {**requirement, "network": "solana:devnet"}, amount
        ),
    }
    for label, passed in checks.items():
        print(f"      {'✅' if passed else '❌'} {label}")
    return all(checks.values())


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
        )
    except ImportError:
        print("SKIPPED: install the extra with  pip install 'venice-py[x402-solana]'")
        return None

    auth, signed = _recording_auth(secret)
    rpc_url = os.environ.get("VENICE_X402_SOLANA_RPC_URL", DEFAULT_SOLANA_RPC_URL)
    print(f"   Wallet:  {auth.wallet_address}")
    print(f"   RPC:     {_rpc_host(rpc_url)}")

    balance = await _usdc_balance(rpc_url, auth.wallet_address, USDC_SOLANA_MAINNET)
    if balance is not None:
        print(f"   On-chain USDC balance: ${balance:,.4f}")
    short = balance is not None and balance < TOPUP_AMOUNT_USDC

    # Venice's side is checked even when the RPC read failed: a real failure
    # there outranks a skip.
    requirement = await check_venice_side(auth)
    if requirement is None:
        return False
    # top_up_with_solana pays exactly the required amount and raises ValueError on any other.
    required = int(str(requirement.get("amount", 0))) / 10**USDC_DECIMALS
    if required != TOPUP_AMOUNT_USDC:
        print(
            f"   ❌ Server requires ${required:,.2f}, and top_up_with_solana pays exactly that: "
            f"it refuses this example's ${TOPUP_AMOUNT_USDC:.2f}"
        )
        return False

    opted_in = os.environ.get("X402_SOLANA_DO_TOPUP") == "1"
    # Only when not opted in: _recording_auth notes every signature, and an entry
    # from this check would make a later error before the real signing read as
    # "the payment may have settled".
    if not opted_in and not sign_without_submitting(auth, requirement):
        return False

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
    async with VeniceClient(api_key="", auth=auth) as client:
        try:
            result = await client.x402.top_up_with_solana(
                auth=auth,
                amount_usdc=TOPUP_AMOUNT_USDC,
                max_amount_usdc=TOPUP_AMOUNT_USDC,  # refuse to sign for more than this
                rpc_url=rpc_url,
            )
        except (VeniceError, ValueError, RuntimeError) as e:
            _report_top_up_error(e, bool(signed))
            return False

    data = result.data
    print("   ✅ Top-up settled on-chain")
    print(f"      Wallet:      {data.walletAddress}")
    print(f"      Credited:    ${data.amountCredited:,.4f}")
    print(f"      New balance: ${data.newBalance:,.4f}")
    print(f"      Payment ID:  {data.paymentId}")
    checks = {
        "a positive amount was credited": data.amountCredited > 0,
        # Base58 is case-sensitive, so this is an exact match.
        "the credit went to this wallet": data.walletAddress == auth.wallet_address,
    }
    for label, passed in checks.items():
        print(f"      {'✅' if passed else '❌'} {label}")
    if not all(checks.values()):
        print("   ❌ The top-up settled, but the reply does not show this wallet credited.")
        print("      Do not re-run: a re-run pays again. Check with")
        print("      client.x402.balance(auth=auth) and client.x402.transactions(auth=auth).")
        return False
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
    if os.environ.get("X402_SOLANA_DO_TOPUP") != "1":
        print("   - SolanaX402Auth.build_payment_header: a payment signed and verified locally")
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
