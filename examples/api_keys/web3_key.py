#!/usr/bin/env python3
"""
Venice AI SDK - Minting an API Key with a Wallet
================================================

This example walks the wallet flow behind ``/api_keys/generate_web3_key``:

1. ``client.api_keys.get_web3_token()`` fetches a short-lived token from Venice
   (``{success, data: {token}}``). The token expires after 15 minutes.
2. The wallet signs the raw token with ``personal_sign``. Here the wallet is a
   throwaway one generated for this run, and the signature is verified locally.
3. ``client.api_keys.create_web3_key()`` posts the address, signature and token.
   Only this step can create anything, so it runs only with
   ``VENICE_RUN_WEB3_MINT=1``.

Venice documents that it mints a key only for a wallet holding staked VVV on Base.
A throwaway wallet holds none, so with the opt-in set the expected answer is a 400
saying so, and a real mint stays unshown: it needs a staked wallet, which this
example never loads. Should a mint ever succeed, the new key belongs to the
wallet's own account, which ``VENICE_API_KEY`` cannot manage, so it is requested as
an ADMIN key with a one-day expiry and a zero USD limit, and it deletes itself
before the script exits. That contradicts the docs, so the script then exits 1 even
when the deletion works, and it prints the key's id. No key secret or wallet key is
printed.

No API key is used. The client signs in as the throwaway wallet
(``VeniceClient(api_key="", auth=...)``), so no real credential is sent even when
``VENICE_API_KEY`` is set.

Install the optional extra first::

    pip install 'venice-py[x402]'

Cost: nothing is expected. Neither endpoint has a listed price, the spec marks both
as needing no authentication, signing is local, and the wallet has no funds to bill.

Exit status: ``0`` when every section that ran verified, ``1`` on a failure, and
``77`` with a ``SKIPPED:`` line when the x402 extra is missing.
"""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from venice_ai import VeniceClient
from venice_ai.exceptions import InvalidRequestError, VeniceError
from venice_ai.types.api import Web3CreateApiKeyRequest
from venice_ai.types.api.requests.api_keys import ConsumptionLimit

if TYPE_CHECKING:
    from eth_account.signers.local import LocalAccount

MINT_OPT_IN_ENV_VAR = "VENICE_RUN_WEB3_MINT"
EXIT_SKIPPED = 77


def sign_token(wallet: "LocalAccount", token: str) -> str:
    """``personal_sign`` over the raw token string, as hex."""
    from eth_account.messages import encode_defunct

    signed = wallet.sign_message(encode_defunct(text=token))
    return "0x" + signed.signature.hex().removeprefix("0x")


def signer_of(token: str, signature: str) -> str:
    """The address that produced ``signature`` over ``token``."""
    from eth_account import Account
    from eth_account.messages import encode_defunct

    return Account.recover_message(encode_defunct(text=token), signature=signature)


def signature_checks_out(wallet: "LocalAccount", token: str, signature: str) -> bool:
    """Check the signature recovers to the wallet, and not when the token differs."""
    recovered = signer_of(token, signature)
    print(f"   Recovered signer: {recovered}")
    if recovered != wallet.address:
        print(f"   ❌ The signature recovers to {recovered}, not {wallet.address}")
        return False
    if signer_of(token + "x", signature) == wallet.address:
        print("   ❌ The signature also verifies for a different token")
        return False
    print("   ✅ The signature matches this wallet and this token only")
    return True


async def delete_minted_key(secret: str, key_id: str) -> bool:
    """Delete a minted key with its own secret, and confirm it is gone."""
    async with VeniceClient(api_key=secret) as owner:
        try:
            result = await owner.api_keys.delete(api_key_id=key_id)
        except VeniceError as e:
            print(f"   ❌ Could not delete minted key {key_id}: {e}")
            return False
    if not result.success:
        print(f"   ❌ Deleting minted key {key_id} reported success=False")
        return False
    print(f"   ✅ Deleted minted key {key_id}")
    return True


async def mint_with_throwaway_wallet(
    client: VeniceClient, wallet: "LocalAccount", token: str, signature: str
) -> bool:
    """Post the signed token. A throwaway wallet should be refused for having no stake."""
    print("\n🪙 Asking Venice to Mint a Key (opted in)")
    print("-" * 50)

    request = Web3CreateApiKeyRequest(
        # Only an ADMIN key can delete a key on the wallet's own account.
        apiKeyType="ADMIN",
        address=wallet.address,
        signature=signature,
        token=token,
        description="venice-sdk-example-web3",
        consumptionLimit=ConsumptionLimit(usd=0),
        limitPeriod="LIFETIME",
        expiresAt=(datetime.now(UTC) + timedelta(days=1)).date().isoformat(),
    )
    try:
        minted = await client.api_keys.create_web3_key(web3_key_request=request)
    except InvalidRequestError as e:
        print(f"   ✅ Refused, and no key exists: {e}")
        print("   A real mint needs a wallet with staked VVV, so it stays unshown here.")
        return True
    except VeniceError as e:
        print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
        return False

    key = minted.data
    print(f"   ❌ A wallet with no stake was given key {key.id}, against the docs. Deleting it.")
    await delete_minted_key(key.apiKey, key.id)
    return False


async def main() -> int:
    """Run the wallet key flow. Returns the process exit status."""
    print("🚀 Venice AI Web3 API Key Flow")
    print("=" * 50)

    try:
        from eth_account import Account

        from venice_ai.auth.x402 import X402Auth
    except ImportError as e:
        print(f"SKIPPED: the x402 extra is not installed (pip install 'venice-py[x402]'): {e}")
        return EXIT_SKIPPED

    # Generated here and held only in memory, so no real wallet is ever involved.
    wallet = Account.create()
    auth = X402Auth(private_key=wallet.key.hex())

    # api_key="" keeps VENICE_API_KEY out of these requests.
    async with VeniceClient(api_key="", auth=auth) as client:
        print("\n🎟️ Fetching a Token")
        print("-" * 50)
        try:
            response = await client.api_keys.get_web3_token()
        except VeniceError as e:
            print(f"❌ get_web3_token() failed: {type(e).__name__}: {e}")
            return 1
        token = response.data.token
        print(f"   success: {response.success}")
        print(f"   token:   {len(token)} characters, {token.count('.') + 1} dot-separated parts")
        if not response.success or token.count(".") != 2:
            print("   ❌ Expected success=True and a three-part JWT")
            return 1

        print("\n✍️ Signing It Locally")
        print("-" * 50)
        print(f"   Throwaway wallet: {wallet.address}")
        signature = sign_token(wallet, token)
        if not signature_checks_out(wallet, token, signature):
            return 1

        if os.environ.get(MINT_OPT_IN_ENV_VAR) != "1":
            print(f"\nSection skipped: minting runs only with {MINT_OPT_IN_ENV_VAR}=1")
        elif not await mint_with_throwaway_wallet(client, wallet, token, signature):
            return 1

    print("\n✨ Web3 key flow completed.")
    print("\n💡 Key concepts demonstrated:")
    print("   - Fetching the short-lived token with get_web3_token()")
    print("   - Signing the raw token with personal_sign and verifying the signer")
    print("   - create_web3_key() needs a wallet with staked VVV on Base")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except VeniceError as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
