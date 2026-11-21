# Client-side end-to-end encrypted (E2EE) chat

`chat.completions.create(..., e2ee=True)` encrypts your messages in-process to the attested model key and decrypts the reply locally, so Venice's infrastructure sees user/system content, assistant history and `reasoning_content` only as ciphertext. Model ID, token counts, roles and timing stay in the clear, and so does every request parameter other than message content: `stop`, `response_format` (including a JSON schema), `prompt_cache_key`, `user`, `metadata`, `seed`, `temperature` and the rest. Keep confidential text out of them. Each response chunk's `id`, `object`, `model`, `created` and `system_fingerprint` are protocol metadata, not encrypted. Ciphertext length reveals plaintext length.

What the attestation binds: the quote's REPORTDATA binds the enclave's signing address and the request nonce. The SDK checks that the model encryption key derives to that signing address and raises `TeeAttestationError` if it does not.

Consume or close every E2EE stream, preferably with `async with`; the session private key is held until it is.

## Install and pick a model

```bash
pip install 'venice-py[e2ee]'
```

```python
model = await client.models.resolve_chat(require_e2ee=True)
```

Always pass `require_e2ee=True` for an E2EE call. The default ranking does not filter on E2EE, so it can return an `e2ee-*` model or a model without E2EE, and `prefer="cheapest"` skips `e2ee-*` models unless `require_e2ee=True`. Do not combine it with `require_function_calling` or `require_vision`; E2EE calls refuse tools and non-text content.

## One call, then a multi-turn loop

Pass the previous `response.choices[0].message` back in `messages`; `ChatMessageParam` accepts it. The assistant turn is encrypted client-side like the user turn.

```python
import asyncio
from venice_ai import VeniceClient
from venice_ai.types.api import ChatMessageParam, SystemMessage, UserMessage


async def main() -> None:
    async with VeniceClient() as client:
        model = await client.models.resolve_chat(require_e2ee=True)
        history: list[ChatMessageParam] = [SystemMessage(content="You are concise.")]
        for question in [
            "Pick one random noun and reply with only that word.",
            "What exact word did you reply with?",
            "Spell it backwards.",
        ]:
            history.append(UserMessage(content=question))
            response = await client.chat.completions.create(
                model=model,
                messages=history,
                e2ee=True,
                stream=False,
                max_completion_tokens=512,
            )
            reply = response.choices[0].message
            print(reply.content)
            history.append(reply)


asyncio.run(main())
```

## Streaming

```python
stream = await client.chat.completions.stream(
    model=model,
    messages=history,
    e2ee=True,
    max_completion_tokens=512,
)
async with stream:
    async for text in stream.collect_with_deltas():
        print(text, end="", flush=True)
final = stream.final_response
if final is not None:
    print(final.choices[0].message.reasoning_content)  # decrypted, None if the model did not reason
```

`reasoning_content` from reasoning models arrives decrypted, in streamed deltas and in the collected message.

## Reading the trust level

`response.e2ee` (or `stream.e2ee`) is a `TeeTrust` from `venice_ai.tee`:

```python
trust = response.e2ee
if trust is not None:
    print(trust.model, trust.verification, trust.server_verified)
```

Fields: `model`, `verification` (`"baseline"` or `"full_quote"`), `verifier`, `server_verified`, `nonce`, `model_public_key`, `signing_address`. `"baseline"` trusts Venice's server-side verified claim. `"full_quote"` means a client-side quote verifier was supplied.

Every E2EE call that uses the baseline verifier emits a `UserWarning` saying so. Python's default warning filter shows it once per call site for direct `create()` calls and once per process through `stream()`, `run_with_tools()` or the sync client; run with `-W always::UserWarning` to see every one.

## Refused before any network call

`create()` raises `InvalidRequestError` for:

- `developer` messages (use `system`), `tool` messages, assistant tool calls
- message keys other than `role`, `content` and assistant `reasoning_content` (for example `name`, `cache_control`)
- non-text content
- `logprobs` / `top_logprobs`
- tools, web search and web scraping
- `e2ee=True` together with `venice_parameters` `enable_e2ee=False`

An `enable_e2ee` value that `VeniceParameters` validates as `True`, such as `1` or `"true"`, turns E2EE on, the same as `e2ee=True`. A non-`bool` `enable_e2ee` that skipped validation (from `VeniceParameters.model_construct` or a subclass that loosens the field) raises `InvalidRequestError` before any request, unless E2EE is on.

## TeeEncryptionError

Any text in the response outside the encrypted fields and the listed protocol metadata raises `TeeEncryptionError` instead of being passed through: an unencrypted `content` or `reasoning_content` delta, `tool_calls`, `reasoning_encrypted`, `thought_signature`, choice `logprobs`, a delta `role` other than `assistant`, a `finish_reason` that is not protocol vocabulary (`stop`, `length`, ...), or an unknown field carrying text.

The protocol metadata that passes in the clear: `finish_reason`, the `assistant` role, `usage` counts, the usage chunk's numeric `cost`, a choice's `matched_stop` and `stop_reason` as a token id or one of your request's own `stop` strings, and each chunk's `id`, `object`, `model`, `created` and `system_fingerprint`.

```python
import sys

from venice_ai.tee import TeeEncryptionError

try:
    response = await client.chat.completions.create(
        model=model, messages=history, e2ee=True, stream=False
    )
except TeeEncryptionError as exc:
    print(f"unencrypted field in reply: {exc}", file=sys.stderr)
    raise
```

## TEE-only mode

An `e2ee-*` model called WITHOUT `e2ee=True` answers normally and raises no error. The prompt and reply are then not encrypted between client and enclave. Pass `e2ee=True` on every call that needs encryption.

## Full quote verification

Install `venice-py[e2ee-verify]`, build a `DcapTdxVerifier` and pass `e2ee=TeeOptions(verifier=verifier)` (both from `venice_ai.tee`) to verify the Intel TDX quote client-side. See `examples/chat/tee_e2ee.py`.

## Handling full-verify failures

A `DcapTdxVerifier` that rejects a quote raises a typed `TeeAttestationError` subclass (all from `venice_ai.exceptions` and `venice_ai.tee`). The error arrives unwrapped from `tee.get_attestation(verifier=)`, `tee.open_session(verifier=)` and `chat.completions.create(e2ee=TeeOptions(verifier=...))`, so catch the class instead of matching message text. With `fail_closed=False` failures stay strings in the result plus a warning.

| Error | Fires when | Attributes | What to do |
| --- | --- | --- | --- |
| `TeeTcbStatusError` | The platform's TCB status is not acceptable under `tcb_policy` | `status`, `advisory_ids`, `policy` | Surface to the user. Fresh collateral does not help; the platform needs a firmware or microcode update. Use `tcb_policy="advisory"` only if the user accepts the advisories. |
| `TeeCollateralMismatchError` | The verifier's fixed collateral is for another FMSPC (`reason == "fmspc"`) or PCK CA (`"pck_ca"`), or a `collateral_provider` returned collateral that does not match the key it was given | `reason`, `collateral_key`, `quote_key` | For a provider, check how your store is keyed (use `CollateralKey.from_quote`). Otherwise fetch collateral for `quote_key`, or use `collateral_provider` so one verifier serves several platforms. |
| `TeeCollateralExpiredError` | A collateral item is past its `nextUpdate` | `key`, `item` (`tcb_info`, `qe_identity`, `pck_crl` or `root_ca_crl`), `next_update`, `raw_quote` | Re-fetch collateral from `raw_quote` and rebuild the verifier or update the provider's cache. |
| `TeeCollateralUnavailableError` | A `collateral_provider` returned `None` for this quote | `key`, `raw_quote` | Fetch collateral from `raw_quote`, store it under `key`, retry once. |

All four are also `TeeError`, so an existing `except TeeError` or `except TeeAttestationError` keeps catching them. An untyped verifier failure is still wrapped in a plain `TeeAttestationError`.

`verify()` is synchronous and never touches the network. To serve quotes from several platforms, or refresh collateral without rebuilding the verifier, pass `collateral_provider=fn` instead of `collateral=` (the two are mutually exclusive). `fn(key, now_secs)` is sync and returns collateral or `None`. Key your cache with `CollateralKey.from_quote(raw_quote)` (from `venice_ai.tee`), which is the same rule the verifier uses, and decide expiry from the `now_secs` you are given.

```python
import json
from datetime import datetime

import dcap_qvl
from cryptography import x509
from venice_ai.exceptions import TeeCollateralExpiredError, TeeCollateralUnavailableError
from venice_ai.tee import CollateralKey, DcapTdxVerifier

cache: dict[CollateralKey, tuple[object, int]] = {}

def expires_at(collateral) -> int:
    def stamp(text: str) -> int:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())

    crls = (collateral.pck_crl, collateral.root_ca_crl)
    return min(
        stamp(json.loads(collateral.tcb_info)["nextUpdate"]),
        stamp(json.loads(collateral.qe_identity)["nextUpdate"]),
        *(int(x509.load_der_x509_crl(crl).next_update_utc.timestamp()) for crl in crls),
    )

def provider(key: CollateralKey, now_secs: int) -> object | None:
    entry = cache.get(key)
    if entry is None or entry[1] <= now_secs:
        return None  # a miss, or expired: the next call takes the fetch path
    return entry[0]

verifier = DcapTdxVerifier(collateral_provider=provider)

async def open_session():
    try:
        return await client.tee.open_session(model=model, verifier=verifier)
    except (TeeCollateralUnavailableError, TeeCollateralExpiredError) as err:
        collateral = await dcap_qvl.get_collateral("https://pccs.phala.network", err.raw_quote)
        cache[err.key] = (collateral, expires_at(collateral))
        return await client.tee.open_session(model=model, verifier=verifier)  # retry once
```

Collateral expires when the earliest of its four items does: `tcb_info` and `qe_identity` (the `nextUpdate` in their JSON) and `pck_crl` and `root_ca_crl` (the CRL's `next_update_utc`). The PCK CRL is usually first. `expires_at` takes the minimum so the provider returns `None` once any of them has passed, and the next call takes the fetch path. `verify()` checks every validity window itself, so an entry kept too long raises `TeeCollateralExpiredError` instead; the handler above refreshes that too.

The `/responses` endpoint does not support E2EE models; use `chat.completions.create(..., e2ee=True)`.
