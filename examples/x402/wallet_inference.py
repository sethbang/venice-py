#!/usr/bin/env python3
"""
Venice AI SDK - x402: Wallet-Paid Inference
===========================================

**The trap: with ``VENICE_API_KEY`` set, ``VeniceClient(auth=auth)`` silently
pays with the API key, not the wallet.** A resolved API key beats a wallet auth,
and nothing warns you. Pass ``api_key=""`` to stop the lookup::

    VeniceClient(api_key="", auth=auth)   # the wallet signs and pays
    VeniceClient(auth=auth)               # the wallet pays only if VENICE_API_KEY is unset

This example demonstrates keyless inference: no API key, a fresh Sign-In-With-X
envelope signed by the wallet on every request, and the cost debited from the
wallet's prepaid USDC ledger. The ledger is funded beforehand with
``top_up.py``; individual requests make no on-chain transfer. The response's
``x402_balance_remaining`` reads what is left on the ledger when the response
carries ``X-Balance-Remaining``. Venice declares that header but may not send it,
so this example reads the ledger itself to see the charge.

What it does, and what it costs:

1. **Which credential wins** (free, offline). Sends one request each way to a
   small server on 127.0.0.1, signed by a throwaway wallet, and checks the
   headers that arrived. Runs without a wallet key.
2. **Sign without paying** (free). Reads the wallet's ledger balance, picks the
   cheapest chat model, estimates the call, then signs a sign-in envelope that
   has already expired and checks its signature locally. The envelope is never
   sent or printed in full.
3. **The paid call** (opt-in). One short chat completion, sent once with retries
   off. It spends a small fraction of a cent from the wallet's prepaid ledger
   (the estimate is printed first), and only with::

       export X402_RUN_INFERENCE=1

Without the opt-in, or without a wallet key, the example prints a ``SKIPPED:``
line and exits ``77``, because no inference was paid for.

Requirements: ``pip install 'venice-py[x402]'`` and a wallet key in
``VENICE_X402_TEST_PRIVATE_KEY`` (or ``X402_WALLET_PRIVATE_KEY``). Use a dedicated
test wallet, never your main key. No API key is needed.
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import warnings
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest import mock

from aiohttp import web

from venice_ai import NoMatchingModelError, RetryOptions, UserMessage, VeniceClient
from venice_ai.exceptions import PaymentRequiredError, VeniceError
from venice_ai.types.api.requests import VeniceParameters

if TYPE_CHECKING:
    from venice_ai.auth.x402 import X402Auth

# Exit codes: 0 = the paid call ran, its answer was right and the ledger did not grow, 1 = a
# failure, 77 = skipped: the extra, a wallet key, a chat model or the opt-in is
# missing, so no inference was paid for.
EXIT_SKIPPED = 77

WALLET_KEY_ENV_VARS = ("VENICE_X402_TEST_PRIVATE_KEY", "X402_WALLET_PRIVATE_KEY")
SIGN_IN_HEADER = "X-Sign-In-With-X"

QUESTION = "What is 2 + 2? Answer with the digit only."
EXPECTED_ANSWER = "4"
MAX_TOKENS = 16
NO_VENICE_PROMPT = VeniceParameters(include_venice_system_prompt=False)
# The paid call is refused above this estimate: this is a one-line question, so
# a higher figure means the catalog's cheapest model is not what it should be.
MAX_ESTIMATE_USD = 0.01
# The ledger and the header are both rounded by the server.
BALANCE_TOLERANCE_USD = 0.000001


@contextlib.asynccontextmanager
async def _header_recorder() -> AsyncIterator[tuple[str, list[dict[str, str]]]]:
    """Run a stand-in API on 127.0.0.1 that records each request's headers."""
    seen: list[dict[str, str]] = []

    async def models(request: web.Request) -> web.Response:
        seen.append(dict(request.headers))
        return web.json_response({"object": "list", "type": "text", "data": []})

    app = web.Application()
    app.router.add_get("/api/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0][:2]
        yield f"http://{host}:{port}/api/v1", seen
    finally:
        await runner.cleanup()


@contextlib.contextmanager
def _sdk_warnings() -> Iterator[list[str]]:
    """Collect what the SDK warns about, through ``warnings`` or its logger."""
    found: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            found.append(record.getMessage())

    collector = Collector(level=logging.WARNING)
    sdk_logger = logging.getLogger("venice_ai")
    sdk_logger.addHandler(collector)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield found
            found.extend(str(w.message) for w in caught if issubclass(w.category, UserWarning))
    finally:
        sdk_logger.removeHandler(collector)


async def which_credential_wins(auth_class: "type[X402Auth]") -> bool:
    """Check which header each constructor sends. Free: it never leaves this machine.

    The wallet is a throwaway and the key a placeholder, so no real credential
    reaches even the local server. Only ``VENICE_API_KEY`` is overridden, and
    only while the first client is constructed, since the key is read there.
    """
    from eth_account import Account

    print("🪤 Which Credential Wins (local server, throwaway wallet)")
    print("-" * 50)

    auth = auth_class(private_key=Account.create().key.hex())
    placeholder = "placeholder-key"
    async with _header_recorder() as (base_url, seen):
        with _sdk_warnings() as warned:
            with mock.patch.dict(os.environ, {"VENICE_API_KEY": placeholder}):
                trapped = VeniceClient(auth=auth, base_url=base_url)
            async with trapped:
                await trapped.models.list()

        async with VeniceClient(api_key="", auth=auth, base_url=base_url) as keyless:
            await keyless.models.list()

    if len(seen) != 2:
        print(f"   ❌ Expected 2 requests at the local server, saw {len(seen)}")
        return False
    bearer, wallet = seen
    envelope = wallet.get(SIGN_IN_HEADER)
    signed_by = json.loads(base64.b64decode(envelope)).get("address") if envelope else None
    checks = {
        "VeniceClient(auth=auth) with VENICE_API_KEY set sends the Bearer key": (
            bearer.get("Authorization") == f"Bearer {placeholder}"
        ),
        "... and no wallet envelope": SIGN_IN_HEADER not in bearer,
        "... and the SDK does not warn about it": not warned,
        'VeniceClient(api_key="", auth=auth) sends no Authorization header': (
            "Authorization" not in wallet
        ),
        "... and an envelope signed by the wallet": signed_by == auth.wallet_address,
    }
    for label, passed in checks.items():
        print(f"   {'✅' if passed else '❌'} {label}")
    return all(checks.values())


def _load_wallet(auth_class: "type[X402Auth]") -> "X402Auth | None":
    """Build the wallet from the first key variable that is set."""
    for name in WALLET_KEY_ENV_VARS:
        private_key = os.environ.get(name)
        if private_key:
            auth = auth_class(private_key=private_key)
            print(f"🔑 Wallet: {auth.wallet_address} (from {name})")
            return auth
    return None


def sign_without_paying(auth: "X402Auth") -> bool:
    """Sign a sign-in envelope like the one a paid call carries, and verify it.

    The envelope expired before it was signed, so even a leaked copy could not
    authenticate. Its message is shown; its signature is not.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct

    print("\n🖊️ Sign Without Paying")
    print("-" * 50)

    header = auth.build_header(now=datetime.now(UTC) - timedelta(seconds=auth.ttl_seconds + 3600))
    envelope = json.loads(base64.b64decode(header))
    message = envelope["message"]
    print("   Every request carries a fresh envelope like this, because a nonce works once:")
    for line in message.splitlines():
        print(f"      {line}")

    def signer_of(text: str) -> str:
        return Account.recover_message(encode_defunct(text=text), signature=envelope["signature"])

    expires = next(
        line.removeprefix("Expiration Time: ")
        for line in message.splitlines()
        if line.startswith("Expiration Time: ")
    )
    wallet = auth.wallet_address.lower()
    checks = {
        "the signature recovers to this wallet": signer_of(message).lower() == wallet,
        "a copy with the text changed does not verify": (
            signer_of(message.replace("Nonce:", "Nonce: 0")).lower() != wallet
        ),
        "the envelope names this wallet": envelope["address"].lower() == wallet,
        "the envelope has already expired": datetime.fromisoformat(expires) < datetime.now(UTC),
    }
    for label, passed in checks.items():
        print(f"      {'✅' if passed else '❌'} {label}")
    return all(checks.values())


async def inference_flow(auth: "X402Auth") -> int:
    """Read the ledger, estimate, sign locally and, opted in, pay for one chat."""
    print("\n💳 Wallet-Paid Inference")
    print("-" * 50)

    # api_key="" stops the key lookup: without it, a VENICE_API_KEY in the
    # environment would pay for this call instead of the wallet.
    async with VeniceClient(api_key="", auth=auth) as client:
        try:
            before = await client.x402.balance(auth=auth)
            model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
            messages = [UserMessage(content=QUESTION)]
            estimate = await client.chat.completions.estimate_cost(
                model=model,
                messages=messages,
                expected_completion_tokens=MAX_TOKENS,
                venice_parameters=NO_VENICE_PROMPT,
            )
        except NoMatchingModelError as e:
            print(f"\nSKIPPED: the catalog lists no non-reasoning chat model ({e})")
            return EXIT_SKIPPED
        except (VeniceError, ValueError) as e:
            print(f"   ❌ A free read failed ({type(e).__name__}): {e}")
            return 1

        if before.data.walletAddress.lower() != auth.wallet_address.lower():
            print("   ❌ The server reported a different wallet than the one that signed in")
            return 1

        ledger = before.data.balanceUsd
        cost = float(estimate.total_cost_usd)
        print(f"   Prepaid ledger: ${ledger:,.6f} (can consume: {before.data.canConsume})")
        print(f"   Model:          {model}")
        print(f"   Estimate:       ${cost:.6f} for up to {MAX_TOKENS} completion tokens")

        if not sign_without_paying(auth):
            return 1

        if os.environ.get("X402_RUN_INFERENCE") != "1":
            print("\n   💤 Not sent. The paid call would be POST /chat/completions with that")
            print(f"      envelope in {SIGN_IN_HEADER} and no Authorization header.")
            print("\nSKIPPED: set X402_RUN_INFERENCE=1 to pay for one short chat from the ledger")
            return EXIT_SKIPPED

        # A requested call that did not happen is a failure, not a skip.
        if not before.data.canConsume or ledger < cost:
            print("\n   ❌ Not sent: the ledger cannot cover this call. Fund it with top_up.py.")
            return 1
        if cost > MAX_ESTIMATE_USD:
            print(f"\n   ❌ Not sent: the estimate is above the ${MAX_ESTIMATE_USD} cap.")
            return 1

        print("\n   💸 Sending one paid chat, retries off…")
        try:
            async with client.with_retries(RetryOptions(max_attempts=0)):
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    venice_parameters=NO_VENICE_PROMPT,
                    max_completion_tokens=MAX_TOKENS,
                )
        except PaymentRequiredError as e:
            print(f"   ❌ The ledger refused the call, so it was not billed: {e}")
            return 1
        except VeniceError as e:
            print(f"   ❌ {type(e).__name__}: {e}")
            print("      It may still have been billed: check balance.py before running again.")
            return 1

        try:
            after = await client.x402.balance(auth=auth)
        except VeniceError as e:
            print(f"   ❌ The call was billed, but the balance read failed: {e}")
            return 1

    answer = response.text or ""
    remaining = response.x402_balance_remaining
    print(f"   Answer:              {answer!r}")
    if remaining is not None:
        print(f"   x402_balance_remaining: ${remaining:,.6f} (ledger before: ${ledger:,.6f})")
    print(f"   Ledger after:        ${after.data.balanceUsd:,.6f}")
    if remaining is None:
        print("   ℹ️  The response has no X-Balance-Remaining, so the ledger is the only reading.")

    checks = {
        "the answer is correct": EXPECTED_ANSWER in answer,
        "any remaining balance reported is not above the ledger before the call": (
            remaining is None or remaining <= ledger + BALANCE_TOLERANCE_USD
        ),
        "the ledger did not grow": after.data.balanceUsd <= ledger + BALANCE_TOLERANCE_USD,
    }
    for label, passed in checks.items():
        print(f"   {'✅' if passed else '❌'} {label}")
    return 0 if all(checks.values()) else 1


async def main() -> int:
    print("🚀 Venice AI x402 — Wallet-Paid Inference Example")
    print("=" * 50)

    try:
        from venice_ai.auth.x402 import X402Auth
    except ImportError as e:
        print(f"SKIPPED: the x402 extra is not installed (pip install 'venice-py[x402]'): {e}")
        return EXIT_SKIPPED

    if not await which_credential_wins(X402Auth):
        print("\n❌ Wallet-paid inference example failed.")
        return 1

    print()
    auth = _load_wallet(X402Auth)
    if auth is None:
        print(f"SKIPPED: set {' or '.join(WALLET_KEY_ENV_VARS)} to a test wallet's key")
        return EXIT_SKIPPED

    code = await inference_flow(auth)
    if code == 1:
        print("\n❌ Wallet-paid inference example failed.")
    if code:
        return code

    print("\n✨ Done.")
    print("\n💡 Key concepts demonstrated:")
    print('   - VeniceClient(api_key="", auth=auth): the wallet pays even with VENICE_API_KEY set')
    print("   - A fresh Sign-In-With-X envelope on every request")
    print("   - Estimating first, then one paid call with retries off")
    print("   - response.x402_balance_remaining: the ledger after the call, if the header is sent")
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
