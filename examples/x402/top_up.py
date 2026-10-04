#!/usr/bin/env python3
"""
Venice AI SDK - x402: Top-Up
============================

Demonstrates Venice's x402 payment-channel top-up endpoint
(``POST /x402/top-up``), which credits the prepaid USDC ledger of a wallet.

Two flows:

1. **Discover payment requirements** — ``client.x402.top_up()`` with no payment
   header returns ``402 Payment Required``. The SDK raises
   :class:`PaymentRequiredError` whose ``body`` is the x402 v2 discovery
   envelope: one ``accepts`` entry per supported chain, each with the network
   (CAIP-2 id), the asset contract, and the amount in the asset's atomic units
   (USDC has 6 decimals, so ``5000000`` is $5.00). The discovery probe returns
   the same envelope whether or not the API key is valid.

2. **Pay and settle** — ``client.x402.top_up_with(auth=X402Auth(...),
   amount_usdc=..., max_amount_usdc=...)`` runs the whole probe → sign → submit
   handshake for USDC on Base: it reads the requirement, refuses to sign for
   more than ``max_amount_usdc``, signs an EIP-3009 transfer authorization with
   your wallet, and submits it. (``top_up_with_solana`` is the Solana
   equivalent; see ``solana_settlement.py``.)

**This moves real funds.** Settlement is an irreversible on-chain USDC
transfer, so the paid flow is a **dry run by default**. Opt in with::

    export X402_DO_TOPUP=1

The dry run still exercises the signing step: it builds the payment header
locally with ``X402Auth.build_payment_header``, then recovers the signer from
the EIP-712 signature (using the token domain the server advertised) and checks
it is this wallet. The authorization it signs is already expired, and it is
never printed or submitted, so it cannot move funds.

Requirements for the paid flow: ``pip install 'venice-py[x402]'`` and a wallet
key in ``VENICE_X402_TEST_PRIVATE_KEY`` (or ``X402_WALLET_PRIVATE_KEY``) whose
wallet holds at least the top-up amount in USDC on Base.
"""

import asyncio
import base64
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from venice_ai import VeniceClient
from venice_ai.exceptions import PaymentRequiredError, VeniceError

if TYPE_CHECKING:
    from venice_ai.auth.x402 import X402Auth

# Exit codes: 0 = discovery and the signing check (or the opted-in top-up)
# verified, 1 = a failure, 77 = skipped because the x402 extra or a wallet key
# is missing, so only discovery ran.
EXIT_SKIPPED = 77

TOPUP_AMOUNT_USDC = 5.0  # Venice's documented minimum
USDC_DECIMALS = 6
WALLET_KEY_ENV_VARS = ("VENICE_X402_TEST_PRIVATE_KEY", "X402_WALLET_PRIVATE_KEY")

# CAIP-2 chain ids used in x402 envelopes: "<namespace>:<chain reference>".
BASE_MAINNET = "eip155:8453"
CHAIN_NAMES = {
    BASE_MAINNET: "Base mainnet",
    "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp": "Solana mainnet",
}


def _usdc(atomic: object) -> float:
    return int(str(atomic)) / 10**USDC_DECIMALS


def _wallet_private_key() -> tuple[str, str] | None:
    """Return ``(env_var_name, private_key)`` for the first variable that is set."""
    for name in WALLET_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


async def discover_payment_requirements() -> list[dict[str, Any]] | None:
    """Empty POST — the server answers 402 with the payment requirements.

    Returns the ``accepts`` list, or ``None`` if discovery failed.
    """
    print("🔍 Discover x402 Payment Requirements")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            result = await client.x402.top_up()
        except PaymentRequiredError as e:
            body = e.body
        except VeniceError as e:
            print(f"❌ Discovery failed with {type(e).__name__} instead of a 402: {e}")
            return None
        else:
            print(
                "❌ The server credited a top-up without a payment header: "
                f"{result.data.amountCredited}"
            )
            return None

    if not isinstance(body, dict) or not body.get("accepts"):
        print(f"❌ 402 received, but the body has no 'accepts' list: {body!r}")
        return None
    accepts = body["accepts"]

    print(f"📬 402 Payment Required — x402 version {body.get('x402Version')}")
    print(f"   {len(accepts)} accepted payment option(s):")
    for option in accepts:
        network = option.get("network")
        chain = CHAIN_NAMES.get(network, "unrecognized chain")
        print(
            f"   • {chain} ({network}): ${_usdc(option.get('amount', 0)):,.2f} "
            f"of asset {option.get('asset')}"
        )
        print(
            f"     scheme={option.get('scheme')}, pay to {option.get('payTo')}, "
            f"signature valid for {option.get('maxTimeoutSeconds')}s"
        )

    print("\n   Full discovery envelope:")
    print(json.dumps(body, indent=2))
    return accepts


def _recover_signer(envelope: dict[str, Any], requirement: dict[str, Any]) -> str:
    """Recover the address that signed the envelope's EIP-3009 authorization.

    The EIP-712 typed data is rebuilt independently of the SDK: the token
    domain's ``name`` and ``version`` come from the server's requirement
    (``extra``), the chain id from its CAIP-2 network and the verifying
    contract from its asset. A match therefore also shows the SDK signed over
    the same domain Venice's facilitator will verify against.
    """
    from eth_account import Account
    from eth_account.messages import encode_typed_data

    authorization = envelope["payload"]["authorization"]
    extra = requirement.get("extra") or {}
    typed_data = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce", "type": "bytes32"},
            ],
        },
        "primaryType": "TransferWithAuthorization",
        "domain": {
            "name": extra.get("name"),
            "version": extra.get("version"),
            "chainId": int(str(requirement["network"]).partition(":")[2]),
            "verifyingContract": requirement["asset"],
        },
        "message": {
            "from": authorization["from"],
            "to": authorization["to"],
            "value": int(authorization["value"]),
            "validAfter": int(authorization["validAfter"]),
            "validBefore": int(authorization["validBefore"]),
            "nonce": authorization["nonce"],
        },
    }
    return Account.recover_message(
        encode_typed_data(full_message=typed_data), signature=envelope["payload"]["signature"]
    )


def sign_without_submitting(auth: "X402Auth", requirement: dict[str, Any]) -> bool:
    """Build the EIP-3009 payment header locally and verify its signature.

    The authorization is signed with a validity window that ended an hour ago,
    so even a leaked copy could never settle. It is still never printed,
    stored or sent; only its decoded fields are checked.
    """
    if not (requirement.get("extra") or {}).get("name"):
        print("   ❌ The requirement does not advertise its EIP-712 token domain (extra.name)")
        return False

    max_units = int(TOPUP_AMOUNT_USDC * 10**USDC_DECIMALS)
    signed_at = datetime.now(UTC) - timedelta(hours=1)
    try:
        header = auth.build_payment_header(
            requirement, max_amount_units=max_units, valid_for_seconds=60, now=signed_at
        )
    except ValueError as e:
        print(f"   ❌ build_payment_header refused to sign: {e}")
        return False

    envelope = json.loads(base64.b64decode(header))
    authorization = envelope["payload"]["authorization"]
    signer = _recover_signer(envelope, requirement)
    valid_before = int(authorization["validBefore"])
    expected_before = int(signed_at.timestamp()) + 60

    print("   🖊️ Payment header signed locally (not submitted, not shown):")
    print(f"      payer {authorization['from']} → payTo {authorization['to']}")
    print(
        f"      value {_usdc(authorization['value']):,.2f} USDC, nonce {authorization['nonce'][:10]}…"
    )
    # The signature binds the signed fields, so a copy with the amount raised
    # by one unit must recover to some other address.
    tampered = json.loads(json.dumps(envelope))
    tampered["payload"]["authorization"]["value"] = str(int(authorization["value"]) + 1)
    tampered_signer = _recover_signer(tampered, requirement)

    domain = requirement["extra"]
    checks = {
        f"signature recovers to this wallet over the server's token domain "
        f"({domain['name']} v{domain.get('version')})": signer.lower()
        == auth.wallet_address.lower(),
        "a copy with the amount changed does not verify": tampered_signer.lower()
        != auth.wallet_address.lower(),
        "validity window is the requested 60s, and already over": valid_before == expected_before
        and valid_before < time.time(),
    }
    for label, passed in checks.items():
        print(f"      {'✅' if passed else '❌'} {label}")
    return all(checks.values())


async def top_up_from_wallet(accepts: list[dict[str, Any]]) -> bool | None:
    """Sign locally (dry run) or, opted in, settle a real USDC top-up on Base.

    Returns ``None`` when the x402 extra or a wallet key is missing.
    """
    print("\n💳 Top Up From a Wallet (USDC on Base)")
    print("-" * 40)

    try:
        from venice_ai.auth.x402 import X402Auth
    except ImportError:
        print("SKIPPED: the x402 extra is not installed (pip install 'venice-py[x402]')")
        return None

    found = _wallet_private_key()
    if found is None:
        print(f"SKIPPED: set {' or '.join(WALLET_KEY_ENV_VARS)} to a test wallet's key")
        return None

    env_name, private_key = found
    auth = X402Auth(private_key=private_key)
    print(f"   Wallet: {auth.wallet_address} (from {env_name})")

    base = next((o for o in accepts if o.get("network") == BASE_MAINNET), None)
    if base is None:
        print(f"❌ The server offered no {CHAIN_NAMES[BASE_MAINNET]} option to pay with")
        return False
    required = _usdc(base.get("amount", 0))
    if required > TOPUP_AMOUNT_USDC:
        print(
            f"❌ Server requires ${required:,.2f}, above this example's ${TOPUP_AMOUNT_USDC:.2f} cap"
        )
        return False
    print(f"   Server minimum on Base: ${required:,.2f} (within the ${TOPUP_AMOUNT_USDC:.2f} cap)")

    if os.environ.get("X402_DO_TOPUP") != "1":
        print(f"\n   💤 Dry run — not submitting. This would top up ${TOPUP_AMOUNT_USDC:.2f}")
        print("      (real, irreversible USDC settlement on Base).")
        if not sign_without_submitting(auth, base):
            return False
        print("      Nothing was settled. Set X402_DO_TOPUP=1 to execute the top-up.")
        return True

    print(f"\n   💸 Submitting a ${TOPUP_AMOUNT_USDC:.2f} top-up (real funds)…")
    async with VeniceClient() as client:
        try:
            result = await client.x402.top_up_with(
                auth=auth,
                amount_usdc=TOPUP_AMOUNT_USDC,
                max_amount_usdc=TOPUP_AMOUNT_USDC,  # refuse to sign for more than this
            )
        except (VeniceError, ValueError, RuntimeError) as e:
            print(f"   ❌ Top-up failed ({type(e).__name__}): {e}")
            return False

    data = result.data
    print("   ✅ Top-up settled")
    print(f"      Wallet:      {data.walletAddress}")
    print(f"      Credited:    ${data.amountCredited:,.4f}")
    print(f"      New balance: ${data.newBalance:,.4f}")
    print(f"      Payment ID:  {data.paymentId}")
    return True


async def main() -> int:
    print("🚀 Venice AI x402 — Top-Up Example")
    print("=" * 50)

    accepts = await discover_payment_requirements()
    if accepts is None:
        print("\n❌ Payment-requirement discovery failed.")
        return 1

    outcome = await top_up_from_wallet(accepts)
    if outcome is None:
        print("   Discovery succeeded; the signing step needs the extra and a wallet key.")
        return EXIT_SKIPPED
    if not outcome:
        print("\n❌ Top-up failed.")
        return 1

    print("\n✨ Done.")
    print("\n💡 Key concepts demonstrated:")
    print("   - Reading the x402 v2 discovery envelope from PaymentRequiredError.body")
    print("   - Decoding atomic USDC amounts (6 decimals) and CAIP-2 network ids")
    print("   - X402Auth.build_payment_header: signing an EIP-3009 transfer locally")
    print("   - Recovering the EIP-712 signer with eth_account to verify the signature")
    print("   - client.x402.top_up_with(auth=, amount_usdc=, max_amount_usdc=) (opt-in only)")
    print("   - Confirm the new balance afterwards with examples/x402/balance.py")
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
