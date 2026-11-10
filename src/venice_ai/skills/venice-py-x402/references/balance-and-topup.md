# Balance reads and topup flow

Sourced from `src/venice_ai/resources/x402.py` and validated end-to-end against the live API. The full topup script lives at `scripts/topup_eip3009.py`.

## Read-only ledger queries

Both balance and transactions endpoints take an `X402Auth` (SIWE) and the underlying HTTP request also uses your normal `VENICE_API_KEY`:

```python
from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth

async with VeniceClient() as client:
    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])

    # Balance — current prepaid USDC ledger for this wallet
    balance = await client.x402.balance(auth=auth)
    print(f"USD: ${balance.data.balanceUsd}")
    print(f"Can consume: {balance.data.canConsume}")
    print(f"Min top-up: ${balance.data.minimumTopUpUsd}")

    # Transactions — paginated history
    txns = await client.x402.transactions(auth=auth, limit=20, offset=0)
    # txns.data is an X402TransactionsData envelope; the list lives under .transactions
    for t in txns.data.transactions:
        print(f"{t.createdAt} {t.type} ${t.amount}")
```

**Important: balance is at `balance.data.balanceUsd`, not `balance.usd`.** The response has a `data` envelope wrapping a typed `X402BalanceData`:

```
X402BalanceResponse{
  success: bool,
  data: X402BalanceData{
    balanceUsd:        float,
    canConsume:        bool,
    minimumTopUpUsd:   float,
    walletAddress:     str,
    ...
  }
}
```

## Recommended: `client.x402.top_up_with(...)` (SDK ≥ 2.0.0)

```python
result = await client.x402.top_up_with(
    auth=X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"]),
    amount_usdc=5.0,
    max_amount_usdc=5.0,    # safety cap; defaults to amount_usdc
)
print(result.data.amountCredited, result.data.paymentId)
```

`top_up_with` performs the full probe → sign → submit flow internally:
1. POSTs `/x402/top-up` with no header → catches `PaymentRequiredError`.
2. Picks the first `"exact"` requirement on Base mainnet from `e.body["accepts"]`.
3. Validates the requirement against `amount_usdc` and `max_amount_usdc` (raises `ValueError` before signing if the server requires more than `max_amount_usdc`, or if `amount_usdc` differs from the server's required amount, in either direction).
4. Builds the EIP-3009 payment header via `auth.build_payment_header(...)`.
5. Re-POSTs with the signed header; returns `X402TopUpResponse{success, data: {walletAddress, amountCredited, newBalance, paymentId}}`.

Use this for the common case. The lower-level path below is for finer control (custom validation, alternate signing, tooling).

## Solana: `client.x402.top_up_with_solana(...)`

The Solana equivalent settles USDC on Solana mainnet via the x402 "exact" SVM
path. Requires the `[x402-solana]` extra (`solders`).

```python
from venice_ai.auth.x402_solana import SolanaX402Auth

result = await client.x402.top_up_with_solana(
    auth=SolanaX402Auth(private_key=os.environ["SOLANA_SECRET"]),  # base58 secret
    amount_usdc=5.0,         # documented minimum
    max_amount_usdc=5.0,     # safety cap; defaults to amount_usdc
    rpc_url=None,            # VENICE_X402_SOLANA_RPC_URL env, else https://api.mainnet-beta.solana.com
)
print(result.data.amountCredited, result.data.paymentId)
```

Flow differences vs the EVM path:
1. Probes `/x402/top-up`; picks the `accepts` entry whose `network` is the Solana
   mainnet id — the CAIP-2 `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp` the live
   challenge sends (the legacy bare `"solana"` is also accepted; any other
   `solana:*` cluster is rejected fail-closed, before signing).
2. Fetches recent blockhash + USDC mint decimals/token-program over Solana
   JSON-RPC (`fetch_solana_tx_context`).
3. Builds a **partially-signed `VersionedTransaction`** (not EIP-3009 typed
   data) — Venice's facilitator co-signs as `feePayer` and sponsors the network
   fee, so the wallet needs USDC but not SOL for gas.
4. Sends the x402 **V2** envelope (`{x402Version, payload, accepted}`).

Returns the same `X402TopUpResponse{success, data: {walletAddress, amountCredited,
newBalance, paymentId}}` shape. **Real, irreversible mainnet settlement** — read
the wallet's on-chain USDC balance and gate behind explicit authorization first
(see `examples/x402/solana_settlement.py`, which is dry-run by default).
`SolanaX402Auth.build_payment_header(requirement=...)` is the lower-level seam.

## Lower-level: the topup flow — two-step probe + sign

The `client.x402.top_up(*, payment_header=None)` endpoint follows the x402 v2 protocol:

1. Call with no `payment_header` → server returns HTTP 402 with structured payment requirements. SDK raises `PaymentRequiredError(body=<requirements>)`.
2. Build a signed `X-402-Payment` header (base64 JSON envelope wrapping an EIP-3009 USDC `transferWithAuthorization` signature). Re-call `top_up(payment_header=<signed>)`.
3. Server settles the EIP-3009 transfer on-chain and credits the prepaid ledger.

### What the 402 body looks like (validated 2026-05-05)

```json
{
  "x402Version": 2,
  "accepts": [
    {
      "protocol": "x402",
      "version":  2,
      "network":  "eip155:8453",
      "asset":    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
      "amount":   "5000000",
      "payTo":    "0x2670B922ef37C7Df47158725C0CC407b5382293F"
    }
  ]
}
```

Fields:
- `network` — CAIP-2 chain ID. Currently `eip155:8453` (Base mainnet).
- `asset` — ERC-20 token contract. Currently the canonical USDC on Base.
- `amount` — amount in token base units (string). USDC has 6 decimals, so `"5000000"` = 5 USDC.
- `payTo` — Venice's settlement address. **Always read this from the response; don't hardcode.**

### Building the X-402-Payment header

Use the SDK seam `X402Auth.build_payment_header(...)`. It builds the EIP-712
`transferWithAuthorization` typed data, signs it, and returns the base64-encoded
**v2** envelope — `{x402Version, payload, accepted}`, with the chosen requirement
wrapped under `accepted` (including `scheme` + `maxTimeoutSeconds`). The flat
top-level `{x402Version, scheme, network, payload}` shape — and an `accepted`
block missing `maxTimeoutSeconds` — are both rejected with a 400 by Venice's V2
facilitator. This mirrors the Solana V2 envelope described above.

```python
from venice_ai.auth.x402 import X402Auth

auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])

# `req` is the chosen requirement from e.body["accepts"][i] (see the 402 body above).
header = auth.build_payment_header(
    req,
    max_amount_units=10_000_000,   # $10 cap; refuses to sign above this
)
```

`build_payment_header` validates `network` / `asset` / `amount` BEFORE signing
(refusing payloads that deviate from expectations), so the manual validation
checklist below is what it enforces internally.

### Submitting the signed payment

```python
result = await client.x402.top_up(payment_header=header)
# result is X402TopUpResponse{
#   success:        bool,
#   data: {
#     walletAddress: str,
#     amountCredited: float,
#     newBalance:    float,
#     paymentId:     str,
#   }
# }
print(f"Topped up ${result.data.amountCredited}; new balance ${result.data.newBalance}")
```

The server settles the EIP-3009 transfer on-chain and credits the ledger. The `paymentId` is returned for your records.

## Validation checklist before signing

Before signing the payment, **always validate the requirement** matches what you expected:

```python
# Validate before signing — never blindly trust the server
EXPECTED_NETWORK = "eip155:8453"
EXPECTED_USDC    = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
MAX_AMOUNT_UNITS = 10_000_000                # $10 cap

assert req["network"] == EXPECTED_NETWORK,        f"unexpected network {req['network']}"
assert req["asset"].lower() == EXPECTED_USDC.lower(), f"unexpected asset {req['asset']}"
assert int(req["amount"]) <= MAX_AMOUNT_UNITS,    f"server wants ${int(req['amount'])/1e6}, over our cap"
```

Without these checks, a misbehaving (or compromised) server could ask you to sign a transfer to an attacker-controlled address. The validation is cheap; do it.

## When the EIP-3009 transferWithAuthorization fails

| Failure mode | Cause | Fix |
|---|---|---|
| `INSUFFICIENT_FUNDS` | Wallet balance < amount | Top up the wallet's USDC on-chain first |
| `EXPIRED_SIGNATURE` | `validBefore` is in the past | Recompute with a fresh `validBefore = time.time() + 600` |
| `INVALID_SIGNATURE` | Domain/message mismatch | Check `chainId`, `verifyingContract`, `name`, `version` in the typed data |
| `NONCE_USED` | Same nonce was already settled | Generate a fresh nonce per call |
| Server still returns 402 | Settlement failed mid-flight | Retry with a fresh signature (new nonce + validBefore) |

## Idempotency

EIP-3009 nonces are single-use — Venice will reject a replayed payment header. **Always sign a fresh nonce per topup.** The script above generates a random 32-byte nonce on every call.

## Idempotent server-side ID

The `paymentId` returned in `X402TopUpResponse.data.paymentId` is your settlement record. Save it; if you need to dispute a charge or correlate with on-chain logs, it's the lookup key.

## Common bugs

- **Reading `balance.usd` instead of `balance.data.balanceUsd`** — the `data` envelope is real.
- **Hardcoding `payTo` instead of reading from the 402** — Venice's settlement address may rotate.
- **Skipping validation of `network` / `asset` / `amount`** — accepting whatever the server asks is unsafe.
- **Reusing a `nonce`** — second use is rejected. Generate fresh per signature.
- **`validBefore` too short** (< 60s) — clock skew between client and server can cause expiration before the request reaches Venice. 600s is a safe default.
- **`X402Auth.payment_header(amount_usd=...)` (fictional method)** — there is no such method. The correct call is `X402Auth.build_payment_header(requirement, max_amount_units=...)` (it signs the EIP-3009 `transferWithAuthorization` typed data internally and returns the base64 `X-402-Payment` envelope), or end-to-end via `client.x402.top_up_with(auth=auth, amount_usdc=...)`.
- **`PaymentRequiredError.payment_instructions`** — wrong attr. Use `e.body`.

## Related references

- `siwe-auth.md` — `X402Auth` builds both the SIWE auth header (read-only ledger) and the EIP-3009 payment envelope.
- `wallet-security.md` — managing the private key safely.
- `agent-frameworks.md` — how Coinbase Agentkit / Eliza / x402-axios automate this flow.
- `venice-py-x402/scripts/topup_eip3009.py` — the validated, runnable script.
