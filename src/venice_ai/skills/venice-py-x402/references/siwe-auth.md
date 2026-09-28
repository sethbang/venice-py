# SIWE auth for Venice (mode-2 deep dive)

Sourced from `src/venice_ai/auth/x402.py` — and validated end-to-end against `api.venice.ai`.

## What `X402Auth` actually does

`X402Auth(private_key=...)` produces a Sign-In-With-X (SIWE / EIP-4361) header that proves wallet ownership to Venice. Venice validates the signature server-side and treats the wallet as the authenticated principal. The header goes in `X-Sign-In-With-X` (NOT `Authorization: Bearer`).

**`X402Auth` builds two distinct signatures.** It signs the SIWE `X-Sign-In-With-X` auth header via `build_header()` for read-only ledger queries (`balance`, `transactions`) and for SIWE-authenticated chat/image/etc. requests when you have prepaid balance. It also signs the EIP-3009 `X-402-Payment` payment envelope via `build_payment_header(requirement, ...)` for top-ups. The two are different signatures (EIP-191/SIWE over a sign-in message vs EIP-712 `transferWithAuthorization` over a USDC transfer) but both live on `X402Auth`; the high-level `client.x402.top_up_with(...)` wraps the payment side (see `balance-and-topup.md` and the SKILL.md mode-3 example).

## Constructor

```python
from venice_ai.auth.x402 import X402Auth

auth = X402Auth(
    private_key=os.environ["WALLET_PRIVATE_KEY"],   # required — 0x-prefixed or bare 64 hex chars
    chain_id=8453,                                  # default — Base mainnet
    ttl_seconds=600,                                # default — 10 min SIWE TTL
)
```

**There is no `wallet_address=` kwarg.** The address is derived from the private key. Read it via the `auth.wallet_address` property (returns the EIP-55 checksummed address):

```python
print(auth.wallet_address)        # "0x1234567890AbCdeF1234567890abCDeF12345678"
```

If you have the private key, you have the address — that's why passing both is redundant.

## Header lifecycle

`auth.build_header(nonce=None, now=None)` produces the base64-encoded `X-Sign-In-With-X` header value. The structure (decoded):

```json
{
  "address":   "0x1234567890AbCdeF1234567890abCDeF12345678",
  "message":   "outerface.venice.ai wants you to sign in...\n...nonce: <hex>...",
  "signature": "0x<hex>",
  "timestamp": 1714939200000,
  "chainId":   8453
}
```

The SIWE message conforms to EIP-4361. The signature is over the EIP-191 personal_sign hash of the message. Server-side, Venice (a) validates the signature recovers to `address`, (b) checks the message hasn't expired (per `expiration_time`), (c) checks `domain` matches `outerface.venice.ai`.

### Token TTL and the single-use nonce

Every call to `build_header()` mints a fresh nonce and signs a new envelope. The signed message stays *valid* for `ttl_seconds` (default 600), but that is an expiry bound, not a reuse window: **Venice accepts a given nonce exactly once.** Replaying an envelope — even seconds after signing it, well inside the TTL — returns `401`: `This nonce has already been used` on inference endpoints, or `Invalid Sign-in-with-x signature` on the `/x402/*` reads.

So sign one envelope per request:

```python
headers = {"X-Sign-In-With-X": auth.build_header()}    # per request, not per session
```

Reuse the `X402Auth` *instance* freely — it holds the key and derives the address, and building one is the part worth keeping. It is the *envelope* that must never be shared between requests. Signing is a single local elliptic-curve operation, negligible beside the network round trip it authenticates, so there is nothing here worth optimising away.

`VeniceClient(auth=...)` signs one envelope per request for you, retries included. The retry middleware replays the same request object on a retryable 5xx (`500`, `502`, `503`, `504` by default), so the SDK re-signs the `X-Sign-In-With-X` header in place before each retried attempt. A wallet-authenticated call that hits a transient 5xx retries normally.

That applies to a per-call wallet, and to rate-limit retries as well as 5xx ones. `client.x402.balance(auth=other_wallet)` signs a fresh envelope for every attempt with `other_wallet`, never the wallet the client was constructed with. If you build the `X-Sign-In-With-X` header yourself and pass it in `headers=`, the SDK leaves it untouched — it is yours to refresh, so drive those retries yourself.

## Calling Venice via SIWE (mode 2 in practice)

Validated flow:

```python
import asyncio, os
import aiohttp
from venice_ai.auth.x402 import X402Auth


async def chat_via_siwe(question: str) -> str:
    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])

    async with aiohttp.ClientSession() as http:
        # Models catalog accepts SIWE auth (free read). Each request signs
        # its own envelope — the nonce is single-use.
        async with http.get(
            "https://api.venice.ai/api/v1/models",
            headers={"X-Sign-In-With-X": auth.build_header()},
            params={"type": "text"},
        ) as r:
            r.raise_for_status()
            data = await r.json()
            model_id = data["data"][0]["id"]

        # Chat completion — SIWE auth, debited from prepaid balance
        async with http.post(
            "https://api.venice.ai/api/v1/chat/completions",
            headers={
                "X-Sign-In-With-X": auth.build_header(),      # a new envelope, not the one above
                "Content-Type": "application/json",
                # NO Authorization: Bearer header
            },
            json={"model": model_id, "messages": [{"role": "user", "content": question}]},
        ) as r:
            r.raise_for_status()
            data = await r.json()
            return data["choices"][0]["message"]["content"]
```

Validated 2026-05-05 against `api.venice.ai`: the call succeeded with `X-Sign-In-With-X` alone, debited the prepaid ledger by ~$0.0015 for a small completion, and returned a normal `chat.completions` response.

## Use `VeniceClient(auth=X402Auth(...))` for SIWE-only mode

On SDK ≥ 2.0.0, the cleanest Mode-2 path is to pass `auth=` directly to `VeniceClient`:

```python
async with VeniceClient(auth=X402Auth(private_key=...)) as client:
    response = await client.chat.completions.create(...)
```

The SDK skips the `Authorization: Bearer` header (since no `api_key` is provided) and signs a fresh `X-Sign-In-With-X` envelope for every request, retries included, so no nonce is ever replayed.

When both `api_key=` and `auth=` are passed, the API key wins for default request auth; the auth instance is retained so callers can still pass it explicitly to per-call `auth=` kwargs (e.g., `client.x402.balance(auth=auth)`).

## Pre-2.0 fallback: drop to aiohttp / httpx

If you can't upgrade past SDK 2.0.0, the constructor used to require `api_key` and the only mode-2 path was to drop to a raw HTTP client. The pattern looked like:

```python
auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])

async with aiohttp.ClientSession() as http:
    # 1. Discover a chat model (the public catalog accepts SIWE auth).
    #    build_header() is called per request: the nonce is single-use.
    async with http.get(
        "https://api.venice.ai/api/v1/models",
        headers={"X-Sign-In-With-X": auth.build_header()},
        params={"type": "text"},
    ) as r:
        r.raise_for_status()
        data = await r.json()
        model_id = data["data"][0]["id"]

    # 2. Chat completion via SIWE — no Authorization: Bearer header
    async with http.post(
        "https://api.venice.ai/api/v1/chat/completions",
        headers={"X-Sign-In-With-X": auth.build_header(), "Content-Type": "application/json"},
        json={"model": model_id, "messages": [{"role": "user", "content": question}]},
    ) as r:
        r.raise_for_status()
        data = await r.json()
        return data["choices"][0]["message"]["content"]
```

You'd sign an envelope per request yourself and keep each nonce single-use. On modern SDKs this is unnecessary — the `auth=` constructor param handles it.

## What goes wrong if SIWE fails

Common error responses from `outerface.venice.ai`:

| Server response | Meaning |
|---|---|
| 401 with `code: "INVALID_SIGNATURE"` | Signature didn't recover to the claimed address. Check the private key. |
| 401 with `code: "EXPIRED_SIGNATURE"` | The SIWE message's `expirationTime` is in the past. Refresh the header. |
| 401 with `code: "INVALID_CHAIN_ID"` | `chainId` in the header doesn't match Venice's expected chain. Default 8453 (Base). |
| 401 with `code: "INVALID_DOMAIN"` | The SIWE message's `domain` field is not `outerface.venice.ai`. Don't override the domain. |
| 401 `This nonce has already been used` (`code: "X402_SIGN_IN_NONCE_REUSED"`) | An envelope was replayed on an inference endpoint. Sign a fresh one per request — the nonce is single-use, independent of TTL. |
| 401 `Invalid Sign-in-with-x signature`, no `code`, on `/x402/*` reads | Often the same replay: the x402 read endpoints report a reused envelope this way rather than naming the nonce. If the key is correct and the first request with that envelope succeeded, sign a fresh one per request. |
| 402 with structured `topUpInstructions` | Auth succeeded; prepaid balance is exhausted. Top up — see `balance-and-topup.md`. |

## Common bugs

- **Passing `wallet_address=` to `X402Auth`** — there's no such kwarg. The address is derived.
- **Reusing one `build_header()` envelope across requests** — the nonce is single-use, so the second request returns a `401` (see the table above for how each endpoint words it). Keep the `X402Auth` instance; sign a new envelope each time. (Holding an envelope past `ttl_seconds` also 401s, on expiry — but reuse fails long before that.)
- **Sending `Authorization: Bearer <api_key>` AND `X-Sign-In-With-X`** — confuses the server's auth pipeline. Pick one. (For `client.x402.balance`, the SDK sends the API key for the HTTP request and SIWE in `X-Sign-In-With-X` to identify the wallet — that's correct because the read endpoint requires API-key access to the route, plus SIWE to scope to a wallet. Don't try to reason about it from first principles; trust the SDK there.)
- **Signing the EIP-191 message manually** when you have `X402Auth` already — that class does the right thing.
- **Treating `auth.build_header()` output as plaintext JSON** — it's base64-encoded. Decode with `base64.b64decode(...)` then `json.loads(...)` if you want to inspect.

## Related references

- `balance-and-topup.md` — the EIP-3009 payment payload (different from SIWE auth).
- `agent-frameworks.md` — Coinbase Agentkit / Eliza / x402-axios alternatives that handle SIWE for you.
- `wallet-security.md` — managing the private key safely.
