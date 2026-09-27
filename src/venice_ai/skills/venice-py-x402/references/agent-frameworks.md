# Wiring Venice into agent frameworks

Sourced from Venice's external docs under `docs.venice.ai/guides/` (`guides/integrations/` and `guides/getting-started/`) and the integrations validated during skill development. Each section is a pointer + minimal wiring sketch — full setup is in the framework's own docs.

## Coinbase Agentkit

[Agentkit](https://github.com/coinbase/agentkit) gives an agent a wallet (CDP-managed or self-custody) and a registry of "actions" (tools). Venice can be wired in two ways:

### Option A — Venice with API key (read-only Coinbase signer)

Use Coinbase Agentkit for **on-chain actions only**, and let Venice handle LLM calls via your normal `VENICE_API_KEY`. Top-up of the Venice prepaid ledger is performed manually (or via `topup_eip3009.py`-style script triggered when the balance dips).

```python
# pseudocode — verify against current Agentkit docs
from coinbase_agentkit import AgentKit, CdpWalletProvider
from venice_ai import VeniceClient

agent = AgentKit(wallet_provider=CdpWalletProvider(...))
client = VeniceClient(api_key=os.environ["VENICE_API_KEY"])

# LLM via Venice; on-chain via Agentkit's wallet
```

### Option B — Venice via x402 from the agent's wallet

The agent uses ITS OWN wallet (managed by Agentkit) to pay Venice per-call. Top-up the Venice ledger from the agent's wallet, then SIWE-auth subsequent chat calls (see `siwe-auth.md` and the topup example in `balance-and-topup.md`).

```python
# pseudocode
private_key = await agent.wallet.export_private_key()    # agentkit-specific call
auth = X402Auth(private_key=private_key)

# Top up if needed (paying from the agent's USDC balance):
balance = await client.x402.balance(auth=auth)           # client uses your API key for this read
if balance.data.balanceUsd < 5:
    await topup_via_eip3009(private_key=private_key, ...)  # see balance-and-topup.md

# Subsequent calls use SIWE auth, debit the agent's prepaid ledger
```

The autonomous-agent story: the agent's wallet IS the agent's identity AND its payment instrument. Top-up is just one of the wallet's actions.

External doc: `docs.venice.ai/guides/integrations/ai-agents.md` covers Agentkit wiring in the broader AI-agents context.

## Eliza

[Eliza](https://github.com/elizaos/eliza) is a multi-agent framework with a Venice provider plugin. Configure via the agent's character file:

```json
{
  "name": "MyAgent",
  "modelProvider": "venice",
  "settings": {
    "secrets": {
      "VENICE_API_KEY": "..."
    }
  }
}
```

For x402-paid mode (no API key), the Eliza Venice provider also accepts a wallet config (`VENICE_WALLET_PRIVATE_KEY` env var as of writing). The provider handles SIWE auth internally — you don't construct `X402Auth` directly. Check the current Eliza Venice plugin docs for env-var names; they evolve.

External doc: `docs.venice.ai/guides/integrations/ai-agents.md` discusses Eliza in the broader agents context.

## x402-axios (JavaScript / TypeScript)

For JS/TS apps, [x402-axios](https://github.com/x402-foundation/x402-axios) is the standard wallet-aware HTTP client. Wraps `axios` with automatic 402-handling:

```typescript
// pseudocode — TS
import { withPaymentInterceptor } from "x402-axios";
import axios from "axios";
import { privateKeyToAccount } from "viem/accounts";

const account = privateKeyToAccount(process.env.WALLET_PRIVATE_KEY as `0x${string}`);
const http = withPaymentInterceptor(axios.create({ baseURL: "https://api.venice.ai/api/v1" }), account);

const response = await http.post("/chat/completions", {
  model: "...",
  messages: [{ role: "user", content: "..." }],
});
```

The interceptor catches HTTP 402, signs the EIP-3009 USDC payment, and retries with `X-PAYMENT`. The Python equivalent is the `x402` package (v2.9+) — see `balance-and-topup.md` for the structure.

## OpenClaw / NanoClaw / Hermes

These are higher-level Venice-native agent frameworks (WhatsApp/Telegram/Discord bots, persistent-memory agents). They handle the Venice + wallet plumbing internally — you configure character + tools and they take care of the rest.

| Framework | Use case | Source |
|---|---|---|
| OpenClaw | Multi-channel chat bots (Discord, Telegram, WhatsApp) | [openclaw](https://github.com/openclaw) |
| NanoClaw | Lightweight personal assistant | [nanoclaw](https://github.com/nanoclaw) (verify current home) |
| Hermes (Nous Research) | Persistent-memory agents | Nous Research org |

External docs under `docs.venice.ai/guides/integrations/`: `openclaw-bot.md`, `nanoclaw-venice.md`, `hermes-agent.md`.

## LangChain / LlamaIndex / CrewAI

Venice is OpenAI-API-compatible at the chat-completions level (with quirks — see `venice_parameters` in the main skill). For frameworks that accept a custom `base_url`:

```python
# LangChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="...",                                # MUST be a Venice model id; resolve dynamically before this line
    api_key=os.environ["VENICE_API_KEY"],
    base_url="https://api.venice.ai/api/v1",
)
```

```python
# LlamaIndex
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="...",
    api_key=os.environ["VENICE_API_KEY"],
    api_base="https://api.venice.ai/api/v1",
)
```

These integrations don't go through `VeniceClient` — you lose the typed exception classes, `_response` metadata, and `venice_parameters`. They're fine for prototyping; for production, the native `VeniceClient` is more robust.

For CrewAI specifically, point its `llm` config at the OpenAI-compat base URL with your Venice API key.

## Custom HTTP path (autonomous agents)

For full control, drop down to `aiohttp` / `httpx` and use the SIWE-auth pattern documented in `siwe-auth.md`. The cost is ~$0.0015 per chat completion debited from the prepaid ledger.

```python
# Top up once (see balance-and-topup.md), then for each call:
auth = X402Auth(private_key=...)
header = auth.build_header()                    # sign one per request; the nonce is single-use

async with aiohttp.ClientSession() as http:
    async with http.post(
        "https://api.venice.ai/api/v1/chat/completions",
        headers={"X-Sign-In-With-X": header, "Content-Type": "application/json"},
        json={"model": ..., "messages": [...]},
    ) as r:
        ...
```

## Picking a framework

| If you want | Pick |
|---|---|
| A bot in a chat platform | OpenClaw (multi-channel) |
| Persistent agents with memory | Hermes / Nous Research |
| Multi-agent orchestration | Eliza or CrewAI |
| Tool-calling agents in Python | Native `VeniceClient.run_with_tools` (no framework) |
| TypeScript / JS app | x402-axios for HTTP, or LangChain.js with OpenAI-compat base |
| Coinbase / on-chain agents | Coinbase Agentkit + native Venice client |

## Common bugs

- **Hardcoding model IDs in framework configs** — no `resolve_*()` in LangChain/LlamaIndex configs. Resolve once at startup, then pass the string.
- **Mixing `Authorization: Bearer <api_key>` and SIWE auth in the same client** — pick one auth model per request.
- **Treating x402-axios's payment retry as transparent forever** — each call signs a payment payload; if the wallet runs out, you get 402s on every call. Top up.
- **Storing `WALLET_PRIVATE_KEY` in framework config files committed to git** — the wallet IS the agent's identity. Treat it like a production credential.

## Related references

- `siwe-auth.md` — the SIWE-auth flow under the hood (mode 2).
- `balance-and-topup.md` — wallet ↔ Venice prepaid ledger flows (mode 3).
- `wallet-security.md` — operational hygiene for the wallet.
- `venice-py/SKILL.md` — the chat / image / etc. surface that frameworks call into.
