#!/usr/bin/env python3
"""
Venice AI SDK - Crypto JSON-RPC Proxy Examples
==============================================

Venice exposes a JSON-RPC 2.0 proxy over a set of supported blockchains via the
``client.crypto`` resource. This example demonstrates the full surface:

- ``client.crypto.networks()``   — discover the supported network slugs
- ``client.crypto.rpc(...)``      — forward a single read-only JSON-RPC call
- ``client.crypto.batch_rpc(...)``— forward 2-3 read-only calls in one batch

No models are involved — this is a thin pass-through to on-chain RPC nodes, so
there is no resolver call here (a chat/image model is never needed).

Key idea: ``networks()`` is *authoritative*. A slug taken from its return value
cannot 400 as "Unsupported RPC network", whereas a hardcoded guess can. So we
discover first, then feed the chosen slug into both ``rpc()`` and ``batch_rpc()``
— the crypto analog of resolver-based model selection.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
  (``networks()`` is public, but ``rpc()``/``batch_rpc()`` bill credits and need a key.)
"""

import asyncio
import sys
from typing import TypeGuard

from venice_ai import (
    PaymentRequiredError,
    PermissionDeniedError,
    VeniceClient,
)
from venice_ai.exceptions import VeniceError

# Exit codes: 0 = networks listed and at least one RPC demo verified (the
# other may print "Section skipped:"), 1 = a failure, 77 = both RPC demos were
# skipped (proxy not enabled on this account, or no EVM network to call).
EXIT_SKIPPED = 77

# Read-only EVM methods that take no params and are cheap to call. These are the
# canonical "is the chain reachable" probes — safe to run repeatedly.
EVM_READ_METHODS = ("eth_chainId", "eth_blockNumber", "eth_gasPrice")

# Chain IDs of well-known networks, so eth_chainId can be checked against the
# slug it was sent to. Slugs not listed here are checked for a hex value only.
KNOWN_CHAIN_IDS = {
    "ethereum-mainnet": 1,
    "ethereum-sepolia": 11155111,
    "base-mainnet": 8453,
    "arbitrum-mainnet": 42161,
    "optimism-mainnet": 10,
    "polygon-mainnet": 137,
}


def _describe_quantity(method: str, value: object) -> str:
    """Decode an EVM hex quantity into a readable number for display."""
    if not (isinstance(value, str) and value.startswith("0x")):
        return repr(value)
    number = int(value, 16)
    if method == "eth_gasPrice":
        return f"{value} = {number:,} wei ({number / 1e9:,.4f} gwei)"
    return f"{value} = {number:,}"


def _select_evm_network(networks: list[str]) -> str | None:
    """Pick an EVM-compatible network slug from the authoritative list.

    The read-only methods used below (``eth_*``) are EVM-specific, so we must not
    land on a non-EVM chain (e.g. a Solana slug). We prefer Ethereum mainnet, then
    any Ethereum-flavored slug, then any slug whose name hints at EVM compatibility.
    Returns ``None`` if nothing suitable is found.
    """
    if not networks:
        return None
    # Exact mainnet match first.
    if "ethereum-mainnet" in networks:
        return "ethereum-mainnet"
    # Any Ethereum-flavored slug next.
    for slug in networks:
        if "ethereum" in slug.lower():
            return slug
    # Fall back to other common EVM chains by name hint.
    evm_hints = ("base", "arbitrum", "optimism", "polygon", "bsc", "avalanche")
    for slug in networks:
        if any(hint in slug.lower() for hint in evm_hints):
            return slug
    return None


async def list_networks(client: VeniceClient) -> tuple[bool, list[str]]:
    """Discover the supported crypto RPC networks.

    Returns ``(ok, networks)``. ``networks()`` is a public endpoint, so this is
    the most likely demo to succeed regardless of account entitlements.
    """
    print("🌐 Supported Crypto RPC Networks")
    print("-" * 40)

    try:
        networks = await client.crypto.networks()
    except VeniceError as e:
        print(f"❌ Failed to list networks: {e}")
        return False, []

    print(f"✅ Proxy supports {len(networks)} network(s):")
    for slug in networks:
        print(f"   • {slug}")
    return True, networks


def _check_chain_id(value: object, network: str) -> TypeGuard[str]:
    """Return True if ``value`` is a hex chain ID matching ``network`` (when known)."""
    if not (isinstance(value, str) and value.startswith("0x")):
        print(f"   ❌ eth_chainId returned {value!r}, not a hex quantity")
        return False
    expected = KNOWN_CHAIN_IDS.get(network)
    if expected is not None and int(value, 16) != expected:
        print(f"   ❌ eth_chainId is {int(value, 16)}, but {network} is chain {expected}")
        return False
    return True


async def single_rpc_call(client: VeniceClient, network: str) -> bool | None:
    """Forward a single read-only JSON-RPC call (``eth_chainId``).

    Returns ``True`` on success, ``None`` when the crypto proxy is not enabled on
    this account (403, a clear skip), and ``False`` when the key cannot pay (402:
    the account balance or the key's own spend limit is exhausted),
    any other error occurs, or the chain ID does not match the network.
    """
    print("\n🔗 Single JSON-RPC Call")
    print("-" * 40)
    print(f"📍 Network: {network}")
    print("🛠️  Method: eth_chainId (read-only, no params)")

    try:
        # rpc() takes the network slug, a JSON-RPC method, optional params and id.
        # eth_chainId returns the chain ID as a hex string (e.g. "0x1" for mainnet).
        # rpc() attaches a random Idempotency-Key to each call, so the server
        # replays rather than re-runs any automatic retry of it. Pass your own
        # idempotency_key= to deduplicate across separate calls.
        resp = await client.crypto.rpc(
            network=network,
            method="eth_chainId",
            params=[],
            id=1,
        )
    except PermissionDeniedError as e:
        print(f"Section skipped: crypto RPC is not enabled on this account ({e})")
        return None
    except PaymentRequiredError as e:
        print(f"❌ rpc() was refused for payment: this key cannot spend enough ({e}).")
        return False
    except VeniceError as e:
        print(f"❌ rpc() failed: {e}")
        return False

    # JSON-RPC returns HTTP 200 even for per-request errors, so check .error first
    # (.result can legitimately be None for some methods).
    if resp.error is not None:
        print(f"❌ JSON-RPC error {resp.error.code}: {resp.error.message}")
        return False

    chain_id_hex = resp.result
    print("✅ Call succeeded:")
    print(f"   id:       {resp.id}")
    print(f"   result:   {chain_id_hex}")
    if not _check_chain_id(chain_id_hex, network):
        return False
    print(f"   decimal:  {int(chain_id_hex, 16)}")

    # The typed billing accessors are surfaced from the HTTP response headers.
    if resp.rpc_credits is not None:
        print(f"   credits:  {resp.rpc_credits}")
    if resp.rpc_cost_usd is not None:
        print(f"   cost USD: {resp.rpc_cost_usd:.8f}")
    return True


async def batch_rpc_call(client: VeniceClient, network: str) -> bool | None:
    """Forward a batch of 3 read-only JSON-RPC calls in a single request.

    Returns ``True`` on success, ``None`` when the proxy is not enabled (403, a
    clear skip), and ``False`` on 402 or any other failure.
    """
    print("\n📦 Batch JSON-RPC Call")
    print("-" * 40)
    print(f"📍 Network: {network}")
    print(f"🛠️  Methods: {', '.join(EVM_READ_METHODS)}")

    # Each request carries an id so we can correlate responses back to methods —
    # JSON-RPC does not guarantee that responses come back in request order.
    id_to_method = {i + 1: method for i, method in enumerate(EVM_READ_METHODS)}
    requests = [
        {"method": method, "params": [], "id": rpc_id} for rpc_id, method in id_to_method.items()
    ]

    try:
        batch = await client.crypto.batch_rpc(network=network, requests=requests)
    except PermissionDeniedError as e:
        print(f"Section skipped: crypto RPC is not enabled on this account ({e})")
        return None
    except PaymentRequiredError as e:
        print(f"❌ batch_rpc() was refused for payment: this key cannot spend enough ({e}).")
        return False
    except VeniceError as e:
        print(f"❌ batch_rpc() failed: {e}")
        return False

    print(f"✅ Batch returned {len(batch)} response(s):")
    ok = True
    returned_ids = {item.id for item in batch}
    missing = [m for rpc_id, m in id_to_method.items() if rpc_id not in returned_ids]
    if missing:
        print(f"   ❌ No response for: {', '.join(missing)}")
        ok = False
    # Correlate by id rather than position (ordering is not guaranteed).
    for item in batch:
        method = id_to_method.get(item.id, "<unknown>") if item.id is not None else "<unknown>"
        if item.error is not None:
            # A per-item RPC error does not fail the whole batch — report and continue.
            print(f"   • {method} (id={item.id}): ❌ error {item.error.code}: {item.error.message}")
            ok = False
        else:
            print(f"   • {method} (id={item.id}): {_describe_quantity(method, item.result)}")
            if method == "eth_chainId" and not _check_chain_id(item.result, network):
                ok = False

    # Billing headers cover the whole batch and live on the wrapper.
    if batch.rpc_credits is not None:
        print(f"   batch credits:  {batch.rpc_credits}")
    if batch.rpc_cost_usd is not None:
        print(f"   batch cost USD: {batch.rpc_cost_usd:.8f}")
    return ok


async def main() -> int:
    """Run the crypto RPC proxy demos.

    Returns ``1`` if any demo failed. Otherwise returns ``0`` when at least
    one RPC demo succeeded (a skipped one prints ``Section skipped:``), and
    ``77`` when neither RPC demo could run (403: the proxy is not enabled, or
    no EVM network is listed): listing networks alone is not the feature.
    """
    print("🚀 Venice AI Crypto JSON-RPC Proxy Examples")
    print("=" * 50)

    results: list[tuple[str, bool | None]] = []

    async with VeniceClient() as client:
        # 1) Discover networks (authoritative — feeds the slug into the RPC calls).
        networks_ok, networks = await list_networks(client)
        results.append(("list_networks", networks_ok))

        network = _select_evm_network(networks)
        if network is None:
            # No EVM-compatible slug to exercise eth_* against, so neither RPC
            # demo can run. A network listing alone is not the feature: exit 77.
            print("\nSection skipped: no EVM-compatible network slug for rpc()/batch_rpc()")
            results.append(("single_rpc_call", None))
            results.append(("batch_rpc_call", None))
        else:
            print(f"\n🎯 Selected network for RPC demos: {network}")
            # 2) Single JSON-RPC call.
            results.append(("single_rpc_call", await single_rpc_call(client, network)))
            # 3) Batch JSON-RPC call.
            results.append(("batch_rpc_call", await batch_rpc_call(client, network)))

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    passed = {name for name, ok in results if ok}

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1
    if not passed & {"single_rpc_call", "batch_rpc_call"}:
        print(
            f"\nSKIPPED: neither RPC demo could run ({', '.join(skipped)}); see the reasons above"
        )
        return EXIT_SKIPPED

    if skipped:
        print(f"\n✨ Crypto RPC proxy demos completed: {', '.join(sorted(passed))}")
        print(f"   ({', '.join(skipped)} skipped; see the reason above)")
    else:
        print("\n✨ Crypto RPC proxy examples completed!")
    # Name only what the demos that ran actually showed.
    concepts = [
        ("list_networks", "Discovering supported chains with client.crypto.networks()"),
        ("single_rpc_call", "Single read-only JSON-RPC via client.crypto.rpc()"),
        ("batch_rpc_call", "Batched read-only JSON-RPC via client.crypto.batch_rpc()"),
        ("batch_rpc_call", "Correlating batch items by id"),
    ]
    print("\n💡 Key concepts demonstrated:")
    for name, concept in concepts:
        if name in passed:
            print(f"   - {concept}")
    print("   - Checking .error before .result")
    print("   - Reading typed billing headers (.rpc_credits / .rpc_cost_usd)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have internet connection.", file=sys.stderr)
        sys.exit(1)
