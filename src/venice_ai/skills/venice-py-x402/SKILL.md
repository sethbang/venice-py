---
name: venice-py-x402
description: Use Venice AI through the x402 micropayment protocol — pay-per-request with on-chain stablecoins, SIWE auth, no traditional API key required. Use this skill — NOT the generic `x402` skill or `goldrush-x402` — whenever the user mentions x402 in the context of Venice, says "Venice without an API key", "Venice from an autonomous agent", "SIWE for Venice", "venice top up", "check x402 balance on venice", "venice wallet billing", "top up Venice from a Solana wallet", "Venice x402 Solana settlement", or is wiring Venice into Coinbase Agentkit / Eliza / x402-axios / OpenClaw / NanoClaw / Hermes / a custom on-chain agent. Covers `client.x402.balance / top_up / transactions`, SIWE token generation via `venice_ai.auth.x402.X402Auth`, the lazy `eth-account` / `siwe` install dance (plus Solana USDC settlement via `SolanaX402Auth` / `client.x402.top_up_with_solana` / the `[x402-solana]` extra), and the three modes (Venice-key client reading x402 ledger, Venice-as-x402-service from a non-Venice client, hybrid). Use this — not generic x402 protocol theory — because Venice's x402 surface has its own auth header semantics, top-up endpoints, and balance/transaction model that the generic skill won't know about. For the chat/image/etc. API surface itself, also load `venice-py` and / or `venice-py-multimodal`.
---

# Venice AI via x402 (wallet-based micropayments)

> _Unofficial, community-maintained — not affiliated with or endorsed by Venice AI._

x402 (HTTP 402 Payment Required) lets autonomous agents pay for Venice API calls in stablecoins on-chain instead of using a traditional API key. This skill covers the Venice-specific surface: `client.x402.*` for ledger ops, `venice_ai.auth.x402.X402Auth` for SIWE-based auth, and the patterns for wiring Venice into agent frameworks.

## When to use this skill (vs the generic `x402` skill)

- **This skill** — Venice's specific x402 surface: balance/top-up/transactions on `api.venice.ai`, SIWE auth for Venice, Venice-from-agent-frameworks.
- **Generic `x402`** skill — protocol-level details, x402 bazaar, paying any x402-enabled service generically.
- **`goldrush-x402`** — entirely different service (GoldRush blockchain data), don't confuse.

For the actual API surface (chat, images, etc.) once you're authenticated, load `venice-py` and/or `venice-py-multimodal` alongside this skill.

## Optional dependencies

x402 features depend on `eth-account` and `siwe` packages, which are NOT in the SDK's core install. The user installs the extra:

```bash
pip install 'venice-py[x402]'
```

If the dependency isn't installed, `from venice_ai.auth.x402 import X402Auth` raises `ImportError`. Code that reaches for these features should fail fast with a clear message.

**Direct-wheel / file:// installs:** when `venice-py` is installed via a direct path (e.g. `pip install /path/to/venice_py-2.2.0-py3-none-any.whl` or `pip install -e /path/to/the/checkout`) instead of from PyPI, pip's resolution of the `[x402]` extras spec is unreliable across pip versions — the extras may quietly be skipped. If `from venice_ai.auth.x402 import X402Auth` raises `ImportError` after a direct-wheel install, fall back to installing the deps directly:

```bash
pip install eth-account siwe
```

This matches the contents of the `[project.optional-dependencies] x402 = ["eth-account", "siwe"]` block in `pyproject.toml`. The PyPI install path (`pip install 'venice-py[x402]'`) is the supported one — the direct-wheel workaround only matters when consuming the SDK from a sibling project.

**Solana settlement** uses a separate extra — `pip install 'venice-py[x402-solana]'` (pulls `solders`). `from venice_ai.auth.x402_solana import SolanaX402Auth` raises `ImportError` without it. The EVM (`X402Auth`, `[x402]`) and Solana (`SolanaX402Auth`, `[x402-solana]`) paths are independent — install whichever chain you settle on.

## The three modes

### Mode 1 — Venice-key client reading the x402 ledger

You have a Venice API key, but you also want to read your prepaid USDC balance or your transaction ledger.

```python
from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth

PRIVATE_KEY = "0x..."           # 0x-prefixed; never commit; load from secret store

async with VeniceClient() as client:
    # X402Auth derives the wallet address from the private key — DON'T pass wallet_address.
    auth = X402Auth(private_key=PRIVATE_KEY)
    print(f"Wallet: {auth.wallet_address}")            # checksummed address property

    balance = await client.x402.balance(auth=auth)
    # The balance response has a `data` envelope:
    # X402BalanceResponse{success, data: X402BalanceData{balanceUsd, canConsume, minimumTopUpUsd, ...}}
    print(f"Prepaid USDC balance: ${balance.data.balanceUsd}")

    txns = await client.x402.transactions(auth=auth, limit=20)
    # txns.data is an X402TransactionsData envelope; the list lives under .transactions
    for t in txns.data.transactions:
        print(f"{t.createdAt} {t.type} ${t.amount}")
```

`X402Auth(private_key=..., chain_id=8453, ttl_seconds=600)` is kw-only. There is **no `wallet_address=` kwarg** — the address is derived (`auth.wallet_address` property). `auth.build_header()` produces the base64 `X-Sign-In-With-X` value the SDK sends, but the SDK calls it for you — only reach for it if crafting raw HTTP. Full ctor / lifecycle details: `references/siwe-auth.md`.

### Mode 2 — No Venice API key: top up once, then SIWE-auth each request

The cleanest way to use Venice without an API key is **two phases**:

1. **Bootstrap (once)**: top up the wallet's Venice prepaid ledger (see Mode 3).
2. **Per-request**: pass `auth=X402Auth(private_key=...)` to `VeniceClient` — the SDK signs and attaches the `X-Sign-In-With-X` header on every request, debiting the prepaid balance.

```python
import os
from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth
from venice_ai.types.api import UserMessage

async def chat_via_siwe(question: str) -> str:
    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
    async with VeniceClient(auth=auth) as client:        # NO api_key — SIWE only
        response = await client.chat.completions.create(
            model=await client.models.resolve_chat(),
            messages=[UserMessage(content=question)],
        )
        return response.text
```

That's the entire mode-2 chat flow on SDK ≥ 2.0.0. The SDK signs a fresh SIWE envelope for every request — Venice treats the nonce as single-use, so envelopes are never reused.

Things to know:
- The wallet must have non-zero prepaid balance — see Mode 3 for `client.x402.top_up_with(...)`.
- Each chat / image / etc. call debits the ledger at Venice's posted rates. Monitor with `client.x402.balance(auth=auth)` or `response.x402_balance_remaining` (the `X-Balance-Remaining` header Venice sends on wallet-authenticated requests).
- When both `api_key=` and `auth=` are passed to `VeniceClient`, the API key wins for default Bearer auth; the auth instance is retained for explicit per-call `auth=` kwargs (e.g., `client.x402.balance(auth=auth)`).
- Per-call `headers={"X-Sign-In-With-X": ...}` overrides the default envelope if you need to sign one yourself (rare).

#### "True" pay-per-request x402 (no prepaid balance)

If you want to pay per-call from on-chain USDC instead of pre-funding the ledger, you need a wrapping HTTP client that handles the 402 → sign → retry handshake (`x402-axios` for JS, `x402` Python library for full scheme registration). That path is more complex and has higher per-call latency than prepaid + SIWE; most autonomous agents are better served by topping up the ledger periodically. See `references/agent-frameworks.md` for the framework-integration entry points.

#### Pre-2.0 SDK fallback

If you can't upgrade past SDK 2.0.0, the `VeniceClient` constructor used to require `api_key`. The fallback was to drop to `aiohttp` / `httpx` and attach `X-Sign-In-With-X` manually — see `references/siwe-auth.md` for that pattern.

### Mode 3 — Hybrid: Venice-key client + autonomous top-up

Long-running agents authenticate via API key but periodically top up the prepaid balance from an on-chain wallet. Use `client.x402.top_up_with(...)` — one call, full probe → sign → submit flow.

```python
from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth

async with VeniceClient() as client:                # uses VENICE_API_KEY for the route
    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
    result = await client.x402.top_up_with(
        auth=auth,
        amount_usdc=5.0,
        max_amount_usdc=5.0,                        # safety cap; defaults to amount_usdc
    )
    print(f"Credited ${result.data.amountCredited}; new balance ${result.data.newBalance}")
```

`top_up_with` internally: probes `/x402/top-up` (no header) → catches `PaymentRequiredError` → validates the first Base-mainnet `"exact"` requirement against `amount_usdc` / `max_amount_usdc` → signs EIP-3009 `TransferWithAuthorization` with the wallet's private key → re-POSTs with the `X-402-Payment` header → returns `X402TopUpResponse{success, data: {walletAddress, amountCredited, newBalance, paymentId}}`. `max_amount_usdc` defaults to `amount_usdc` (refuses to sign if the server asks for more).

For the lower-level path (`auth.build_payment_header(requirement)`), the 402-body schema, and `PaymentRequiredError` semantics (`e.body`, NOT `e.payment_instructions`), see `references/balance-and-topup.md`. A standalone script `scripts/topup_eip3009.py` demos the manual flow — **don't run without explicit per-transaction authorization**, it moves real funds.

### Mode 3 (Solana) — top up from a Solana wallet

Same hybrid pattern, settling USDC on **Solana** instead of Base. Use
`SolanaX402Auth` + `client.x402.top_up_with_solana(...)`:

```python
import os
from venice_ai import VeniceClient
from venice_ai.auth.x402_solana import SolanaX402Auth

async with VeniceClient() as client:                # Bearer auth (VENICE_API_KEY) routes the request
    auth = SolanaX402Auth(private_key=os.environ["SOLANA_SECRET"])  # base58 secret
    print(auth.wallet_address)                      # base58 pubkey, derived
    result = await client.x402.top_up_with_solana(
        auth=auth,
        amount_usdc=5.0,                            # Venice's documented minimum
        max_amount_usdc=5.0,                        # refuse to sign for more; defaults to amount_usdc
        rpc_url=None,                               # VENICE_X402_SOLANA_RPC_URL env, else mainnet-beta
    )
    print(f"Credited ${result.data.amountCredited}; new balance ${result.data.newBalance}")
```

`top_up_with_solana` runs the x402 v2 probe → sign → submit handshake for the
"exact" SVM path: it picks the `accepts` entry whose `network` is the Solana
mainnet id — the CAIP-2 `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp` the live
challenge sends, or the legacy bare `"solana"` (any other `solana:*` cluster is
rejected) — fetches blockhash/mint context over JSON-RPC, builds a
partially-signed `VersionedTransaction` (Venice's facilitator sponsors the
network fee via `feePayer`), and sends the **V2** `X-402-Payment` envelope.
Settlement is on **Solana mainnet** in real USDC — **irreversible**; gate it
behind explicit authorization and read the wallet's balance first. `SolanaX402Auth`
is kw-only with **no `wallet_address=`** (derived; read `auth.wallet_address`).

## Wiring Venice into agent frameworks

- **Coinbase Agentkit** — Venice is registered as a tool provider; the Agentkit signer handles x402 auth automatically. See `references/agent-frameworks.md` and Venice docs at `docs.venice.ai/guides/integrations/ai-agents.md`.
- **Eliza** — There's a Venice provider in the Eliza ecosystem; it accepts either an API key or a wallet config. Same docs.
- **OpenClaw / NanoClaw / Hermes** — Self-hosted bot frameworks pre-configured for Venice + x402. Pointers in references.
- **x402-axios** (JS) — most common JS path; takes a viem account, handles 402 transparently.

These integrations are evolving. Don't trust a code snippet from training data — check the upstream repo's README for current shapes.

## Pitfalls AI assistants reliably get wrong

1. **Passing `wallet_address=` to `X402Auth`** — no such kwarg; address is derived. Use `auth.wallet_address` (property) to read it.
2. **Calling a fictional `auth.payment_header(amount_usd=...)`** — doesn't exist. `X402Auth` signs SIWE only. For payment headers use `auth.build_payment_header(requirement)` or, for the common case, `client.x402.top_up_with(...)`.
3. **Treating `PaymentRequiredError` as transient** — it's terminal. The structured requirements live on `e.body` (NOT `e.payment_instructions` — `venice-py lint` flags this as V601). Sign a payment, then retry the original op.
4. **Reusing one signed SIWE envelope across calls** — Venice rejects a repeated nonce with `401 This nonce has already been used`, so a cached envelope makes every request after the first fail. Reuse the `X402Auth` *instance*, but call `build_header()` per request. `VeniceClient(auth=auth)` already does this for you, and re-signs on retries too.
5. **Forgetting the `[x402]` extra** — `from venice_ai.auth.x402 import X402Auth` ImportError is a setup smell. (Solana settlement needs the separate `[x402-solana]` extra.)
6. **Committing the private key** — the wallet IS the agent's credentials; see `references/wallet-security.md`.
7. **Confusing the EVM and Solana auth classes** — `X402Auth` (EVM, `0x` key, `[x402]`) and `SolanaX402Auth` (base58 key, `[x402-solana]`) are different classes with different top-up methods (`top_up_with` vs `top_up_with_solana`). Neither takes `wallet_address=`.

## References

- `references/siwe-auth.md` — SIWE token mechanics for Venice (message format, expiry, header placement, single-use nonces)
- `references/agent-frameworks.md` — wiring patterns for Coinbase Agentkit, Eliza, x402-axios, OpenClaw, Hermes, NanoClaw
- `references/balance-and-topup.md` — read flows, write flows, structured `PaymentRequiredError` handling
- `references/wallet-security.md` — key management, scoped keys, dev vs prod isolation

## Scripts

- `scripts/topup_eip3009.py` — the validated, runnable EIP-3009 top-up flow. **Don't run without explicit per-transaction authorization** — it moves real funds.

## Examples to read

In the SDK repo at `examples/x402/`:
- `balance.py` — wallet balance via `X402Auth`
- `transactions.py` — transaction history pagination
- `top_up.py` — top-up flow with error handling
- `solana_settlement.py` — Solana USDC top-up via `SolanaX402Auth` (dry-run by default)

In Venice's external docs (under `docs.venice.ai/guides/`):
- `guides/integrations/ai-agents.md` — overall AI agents framing
- `guides/integrations/crypto-rpc-agents.md` — on-chain agent patterns
- `guides/getting-started/generating-api-key-agent.md` — autonomous key creation via VVV staking
- `guides/integrations/x402-venice-api.md` — protocol details
