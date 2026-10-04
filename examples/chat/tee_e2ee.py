#!/usr/bin/env python3
"""
Venice AI SDK - TEE Client-Side End-to-End Encryption (E2EE)
============================================================

Demonstrates Venice confidential-compute (TEE) chat with client-side E2EE.
Messages are encrypted in-process to the attested model key and the streamed
response deltas are decrypted locally — Venice's infrastructure never sees the
plaintext of your ``user`` / ``system`` content.

Requires the ``[e2ee]`` extra (pulls in ``cryptography``)::

    pip install 'venice-py[e2ee]'

For the optional full client-side TDX quote verification step, also install the
``[e2ee-verify]`` extra (pulls in ``dcap-qvl``; arm64-macOS wheel available)::

    pip install 'venice-py[e2ee-verify]'

Features Demonstrated:
    - Discovery of E2EE-capable models from the catalog, falling back to the
      next model when a confidential-compute node is temporarily unavailable
    - client.tee.get_attestation(model=...) → fail-closed baseline verification
    - Full client-side Intel TDX quote verification via DcapTdxVerifier,
      attached to the chat calls with e2ee=TeeOptions(verifier=...)
    - Streaming encrypted chat, consuming decrypted text deltas
    - A stream=False call returning a reassembled (decrypted) response

Security note:
    The *baseline* attestation verifier TRUSTS Venice's server-side ``verified``
    claim. ``DcapTdxVerifier`` (shown below) removes that trust by verifying the
    Intel TDX quote client-side, but by default proves only GENUINE non-debug
    TDX hardware + a self-consistent dstack workload (Tier B) — not that it is
    the legitimate Venice image, which requires an independently-pinned
    reference (``expected_compose_hash`` / ``expected_measurements``, Tier A).
"""

import asyncio
import math
import sys
from dataclasses import dataclass

from venice_ai import VeniceClient, model_price
from venice_ai.exceptions import APIError, TeeAttestationError, TeeError
from venice_ai.tee import DcapTdxVerifier, TeeOptions
from venice_ai.tee.types import TeeAttestation
from venice_ai.types.api import SystemMessage, TextModelSpec, UserMessage

MAX_COMPLETION_TOKENS = 512


def reached_cap(response: object) -> bool:
    """True if a reply used all ``MAX_COMPLETION_TOKENS``, or reported no usage to check.

    Some models report a reply cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked as well. Encrypted replies carry
    usage in the clear, both streamed and not.
    """
    usage = getattr(response, "usage", None)
    used = usage.completion_tokens if usage is not None else None
    print(f"   📊 Completion tokens: {used} of {MAX_COMPLETION_TOKENS}")
    return used is None or used >= MAX_COMPLETION_TOKENS


class NodeUnavailableError(Exception):
    """A confidential-compute node answered with a transient server error."""


def _is_transient(error: APIError) -> bool:
    """True for errors that another E2EE model may not share (5xx, overload)."""
    status = error.status_code
    return status is not None and (status >= 500 or status == 429)


@dataclass
class DemoResult:
    """Outcome of one encrypted-chat run."""

    ok: bool
    full_verification: bool
    skipped: bool = False


async def build_full_verifier(attestation: TeeAttestation) -> DcapTdxVerifier | None:
    """Build a ``DcapTdxVerifier`` for *attestation*, or ``None`` if it cannot run.

    Full verification cannot run when the attestation carries no Intel TDX
    quote, the ``[e2ee-verify]`` extra (``dcap-qvl``) is not installed, or the
    PCCS collateral fetch fails. Those are skipped with an explanation. This
    only BUILDS the verifier; whether the quote passes is decided by
    ``verify()``, whose rejection the caller must treat as a failure.
    """
    print("\n🔎 Full client-side TDX verification ([e2ee-verify])...")
    quote = attestation.intel_quote
    if not isinstance(quote, (str, bytes)):
        print("   ⏭️  The attestation carries no Intel TDX quote to verify.")
        print("      The chats below use baseline verification only.")
        return None
    try:
        # with_fetched_collateral makes the ONE no-auth PCCS call; verify()
        # itself is fully offline.
        return await DcapTdxVerifier.with_fetched_collateral(quote)
    except TeeError as e:
        # Raised when dcap-qvl is missing; the message carries the install hint.
        print(f"   ⏭️  Full TDX verification is unavailable: {e}")
    except (ValueError, RuntimeError) as e:
        # dcap-qvl raises ValueError for an unparseable quote / FMSPC and
        # RuntimeError when the PCCS request fails.
        print(f"   ⏭️  Could not fetch TDX collateral ({type(e).__name__}: {e})")
    print("      The chats below use baseline verification only (the SDK warns")
    print("      once per call about that limitation).")
    return None


async def discover_e2ee_models(client: VeniceClient) -> list[str]:
    """Return every E2EE-capable text model, non-reasoning models first, then by price.

    Uses the catalog's ``capabilities.supportsE2EE`` flag. Non-reasoning models
    come first because they answer the short prompts below without spending
    the token budget on hidden reasoning; within each group the cheapest model
    is tried first, so falling back past an unavailable node stays cheap too.
    """
    models = await client.models.list(type="text")
    candidates: list[tuple[bool, float, str]] = []
    for m in models.data:
        if not isinstance(m.model_spec, TextModelSpec):
            continue
        caps = m.model_spec.capabilities
        if caps is not None and caps.supportsE2EE:
            price = model_price(m.model_dump())
            candidates.append(
                (bool(caps.supportsReasoning), price if price is not None else math.inf, m.id)
            )
    return [model_id for _reasoning, _price, model_id in sorted(candidates)]


async def run_encrypted_demo(client: VeniceClient, model: str) -> DemoResult:
    """Attest *model*, verify its quote, then run streaming and non-streaming E2EE chat.

    Returns a failed result when full quote verification rejects the enclave
    or an encrypted reply is truncated or wrong. Raises
    ``NodeUnavailableError`` on a transient server error so the caller can try
    another model, and lets ``TeeAttestationError`` propagate: a failed
    attestation check is a security failure, never something to route around.
    """
    try:
        # --- (1) Attestation ----------------------------------------------------
        # Fetch and BASELINE-verify the enclave's attestation. fail_closed=True
        # (the default) means any failed check raises TeeAttestationError, so if
        # this returns, the attestation passed every baseline check.
        # This is a standalone demonstration of the attestation surface; each
        # encrypted create() below re-attests when it opens its session.
        print("\n🪪  Fetching + baseline-verifying attestation...")
        attestation = await client.tee.get_attestation(model=model)
        print(f"   verified (server claim): {attestation.verified}")
        print("   ✅ Attestation is BASELINE-verified (nonce echo + report-data")
        print("      binding + TDX debug-flag checks all passed, fail-closed).")
        print("   ⚠️  SECURITY LIMITATION: baseline verification TRUSTS Venice's")
        print("      server-side 'verified' claim and does NOT perform full")
        print("      client-side Intel TDX quote verification.")

        # --- (1b) Full client-side TDX verification (optional) ----------------
        # The [e2ee-verify] extra adds DcapTdxVerifier, which verifies the raw
        # Intel TDX quote itself (signature → pinned Intel SGX Root CA + TCB
        # status + QE identity + debug-flag), binds the E2EE key to the enclave
        # (REPORTDATA), and replays the event log to the quoted RTMRs — entirely
        # client-side, instead of trusting Venice's 'verified' flag. It runs on
        # Apple Silicon (dcap-qvl ships an arm64 wheel).
        verifier = await build_full_verifier(attestation)
        e2ee: bool | TeeOptions = True
        full_verification = False
        if verifier is not None:
            # verify() is fail-closed: a rejected quote raises TeeError, which
            # fails the demo. A verifier that rejected this enclave must never
            # be quietly replaced by baseline trust.
            try:
                verified = verifier.verify(attestation)
            except TeeError as e:
                print(f"   ❌ Full TDX quote verification REJECTED this enclave: {e}")
                return DemoResult(ok=False, full_verification=True)
            if not verified:
                print("   ❌ Full TDX quote verification did not accept this enclave")
                return DemoResult(ok=False, full_verification=True)
            result = verifier.last_result or {}
            full_verification = True
            print(f"   full quote verified: {verified}")
            print(f"   TCB status: {result.get('tcb_status')!r}  (fail-closed; reject-by-default)")
            print(f"   workload_identity_pinned: {result.get('workload_identity_pinned')}")
            print("   ℹ️  Tier B: this proves GENUINE non-debug Intel TDX hardware +")
            print("      a self-consistent dstack workload — but NOT that it is the")
            print("      legitimate Venice image. For per-dimension Tier A, pin an")
            print("      INDEPENDENTLY-obtained reference, e.g.:")
            print("        DcapTdxVerifier(collateral=..., expected_compose_hash='<ref>')")
            # Attach the verifier so every encrypted session below runs full
            # quote verification on its own attestation, not just this one.
            e2ee = TeeOptions(verifier=verifier)
            print("   🔗 Attaching it to the encrypted chats via e2ee=TeeOptions(verifier=...)")

        # --- (2) Streaming encrypted chat -------------------------------------
        # stream(e2ee=...) returns a ChatStream. collect_with_deltas()
        # yields already-DECRYPTED text (decryption happens locally per chunk)
        # and assembles the final response so we can check its finish reason.
        # venice_parameters.include_venice_system_prompt is not set here: E2EE
        # mode has no Venice system prompt (Venice would have to encrypt it
        # client-side), and the SDK always sends the switch off on an
        # encrypted request, so only the messages below are billed.
        messages: list[SystemMessage | UserMessage] = [
            SystemMessage(content="You are a concise, helpful assistant."),
            UserMessage(content="In one sentence, what does end-to-end encryption protect?"),
        ]
        print("\n🤖 Assistant (E2EE streaming): ", end="", flush=True)
        stream = await client.chat.completions.stream(
            model=model,
            messages=messages,
            e2ee=e2ee,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.3,
        )
        async with stream:
            async for text in stream.collect_with_deltas():
                print(text, end="", flush=True)
        print()
        streamed = stream.final_response
        stream_finish = streamed.choices[0].finish_reason if streamed and streamed.choices else None
        streamed_text = (streamed.text if streamed else "") or ""
        print(f"   🏁 Finish reason: {stream_finish}")

        # --- (3) Non-streaming encrypted chat ---------------------------------
        # stream=False still runs the encrypted wire flow under the hood, but
        # the SDK reassembles the decrypted deltas into a normal
        # ChatCompletionResponse for you.
        print("\n📦 Non-streaming E2EE response:")
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Reply with exactly: ENCRYPTED OK")],
            e2ee=e2ee,
            stream=False,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.0,
        )
    except TeeAttestationError:
        raise
    except APIError as e:
        if _is_transient(e):
            raise NodeUnavailableError(f"{type(e).__name__}: {e}") from e
        raise

    reply = (response.text or "").strip()
    finish = response.choices[0].finish_reason if response.choices else None
    print(f"   {reply}")
    print(f"   🏁 Finish reason: {finish}")

    ok = True
    for label, result in (("streamed", streamed), ("non-streaming", response)):
        if reached_cap(result):
            print(f"   ❌ The {label} E2EE reply may be cut off (cap reached or no usage)")
            ok = False
    if stream_finish != "stop" or not streamed_text.strip():
        print("   ❌ The streamed E2EE answer was empty or cut off")
        ok = False
    if finish != "stop" or "ENCRYPTED OK" not in reply.upper():
        print("   ❌ The non-streaming E2EE reply was not the requested text")
        ok = False
    return DemoResult(ok=ok, full_verification=full_verification)


async def tee_e2ee_demo() -> DemoResult:
    """Discover E2EE models, then attest one and run encrypted chat.

    Tries each E2EE-capable model in turn when a node answers with a transient
    server error (individual confidential-compute nodes can be down while
    others serve). The result is ``skipped`` when no E2EE model or the
    ``[e2ee]`` extra is available, and not ``ok`` when the demo ran and failed
    or every candidate was unavailable.
    """
    print("🔐 Venice TEE End-to-End Encryption")
    print("-" * 40)

    async with VeniceClient() as client:
        candidates = await discover_e2ee_models(client)
        if not candidates:
            # No E2EE-capable model is available to this account. That's an
            # entitlement / availability condition, not a code error.
            print("   ⏭️  No E2EE-capable model is available on this account.")
            print("      Confidential-compute models are gated; nothing to demo.")
            return DemoResult(ok=True, full_verification=False, skipped=True)
        print(f"🔎 Found {len(candidates)} E2EE-capable model(s)")

        for model in candidates:
            print(f"\n📍 Using E2EE model: {model}")
            try:
                return await run_encrypted_demo(client, model)
            except NodeUnavailableError as e:
                print(f"\n   ⚠️  {model} is unavailable right now ({e}); trying the next one.")
            except ImportError as e:
                # The [e2ee] extra (cryptography) is not installed, so the
                # encrypting session cannot be opened.
                print("\n   ⏭️  Encrypted chat needs the [e2ee] extra.")
                print(f"      ({e})")
                print("      Install it with: pip install 'venice-py[e2ee]'")
                return DemoResult(ok=True, full_verification=False, skipped=True)

    print("\n❌ Every E2EE-capable model was unavailable.")
    return DemoResult(ok=False, full_verification=False)


async def main() -> int:
    """Run the TEE E2EE example.

    Returns ``0`` only if the demo succeeded, ``77`` if it was skipped (no
    E2EE model on this account, or the ``[e2ee]`` extra is missing), and ``1``
    if a load-bearing step failed, so neither a failure nor a skip is masked
    by the success banner.
    """
    print("🚀 Venice AI TEE E2EE Example")
    print("=" * 50)

    result = await tee_e2ee_demo()

    if not result.ok:
        print("\n❌ TEE E2EE example failed.")
        return 1
    if result.skipped:
        print("\nSKIPPED: TEE E2EE example skipped (nothing was encrypted).")
        return 77

    print("\n✨ TEE E2EE example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Discovering E2EE models via capabilities.supportsE2EE")
    print("   - client.tee.get_attestation(model=...) → fail-closed baseline verify")
    if result.full_verification:
        print("   - DcapTdxVerifier full client-side TDX quote verification ([e2ee-verify])")
        print("   - e2ee=TeeOptions(verifier=...) to verify every encrypted session")
    print("   - Streaming E2EE chat with locally decrypted deltas")
    print("   - Non-streaming E2EE chat with a reassembled response")
    if result.full_verification:
        print("   - Tier B vs caller-pinned Tier A verification")
    else:
        print("   ⚠️  Only baseline verification ran: Venice's 'verified' claim was trusted.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except TeeError as e:
        # Attestation is fail-closed: a failed baseline check, or a session
        # rejected by the attached verifier, raises and MUST surface loudly
        # (non-zero exit), never silently degrade.
        print(f"\n❌ TEE verification failed ({type(e).__name__}): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
