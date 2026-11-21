"""Resource-phase tests for ``venice_ai.resources.tee.Tee`` and ``client.tee`` wiring.

The transport is mocked (``_client.get`` returns a parsed
:class:`TeeAttestation`). Two attestation sources:

* the live fixture ``fixtures/attestation_gemma.json`` — used to assert
  :meth:`get_attestation` hits the endpoint and returns a verified attestation;
  its ``request_nonce`` is fed back as the sent nonce so verification passes.
* a synthetic, verification-passing attestation built around a model keypair we
  control — used for the :meth:`open_session` encrypt/decrypt round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# cryptography is the optional [e2ee] extra; open_session (keypair gen) needs it.
ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee resource tests require the [e2ee] extra (cryptography)",
)

from venice_ai.exceptions import TeeAttestationError  # noqa: E402
from venice_ai.resources.tee import Tee  # noqa: E402
from venice_ai.tee import _crypto  # noqa: E402
from venice_ai.tee._attestation import NONCE_LEN_HEX, _keccak256  # noqa: E402
from venice_ai.tee._constants import (  # noqa: E402
    HEADER_CLIENT_PUB_KEY,
    HEADER_MODEL_PUB_KEY,
)
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.tee.types import TeeAttestation  # noqa: E402

_CURVE = ec.SECP256K1()
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "attestation_gemma.json"


@pytest.fixture
def raw_attestation() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture
def tee() -> Tee:
    return Tee(MagicMock())  # type: ignore[arg-type]


def _synthetic_attestation() -> tuple[dict[str, Any], ec.EllipticCurvePrivateKey, str]:
    """A verification-passing attestation around a model key we control.

    reportData = signing_address(40hex) || "00"*12 || nonce(64hex) = 128 hex.
    """
    model_priv = ec.generate_private_key(_CURVE)
    model_pub_hex = _crypto.uncompressed_hex(model_priv.public_key())
    # The address the key derives to, as verify_attestation's key binding requires.
    addr = _keccak256(bytes.fromhex(model_pub_hex)[1:])[-20:].hex()
    nonce = "cd" * 32
    report_data = addr + ("00" * 12) + nonce
    raw = {
        "verified": True,
        "signing_public_key": model_pub_hex,
        "signing_address": "0x" + addr,
        "signing_algo": "ecdsa",
        "nonce": nonce,
        "server_verification": {
            "tdx": {
                "reportData": report_data,
                "measurements": {"tdAttributes": "0000000000000000"},
            }
        },
    }
    return raw, model_priv, nonce


# --- get_attestation ---------------------------------------------------------


@pytest.mark.asyncio
async def test_get_attestation_calls_endpoint_and_returns_verified(
    tee: Tee, raw_attestation: dict[str, Any]
) -> None:
    attestation = TeeAttestation.model_validate(raw_attestation)
    tee._client.get = AsyncMock(return_value=attestation)  # type: ignore[attr-defined]

    sent_nonce = raw_attestation["request_nonce"]
    result = await tee.get_attestation(model="e2ee-gemma-3-27b-p", nonce=sent_nonce)

    # Hit the right endpoint with model + nonce params.
    call = tee._client.get.call_args  # type: ignore[attr-defined]
    assert call.args[0] == "tee/attestation"
    assert call.kwargs["params"] == {"model": "e2ee-gemma-3-27b-p", "nonce": sent_nonce}
    assert call.kwargs["cast_to"] is TeeAttestation

    # Verified (no raise) and sent_nonce recorded.
    assert result is attestation
    assert result.verified is True
    assert result.sent_nonce == sent_nonce


@pytest.mark.asyncio
async def test_get_attestation_generates_a_64hex_nonce_when_omitted(
    tee: Tee, raw_attestation: dict[str, Any]
) -> None:
    # Echo whatever nonce the resource generated so verification passes.
    def _echo(path: str, *, params: dict[str, str], cast_to: Any) -> TeeAttestation:
        raw = dict(raw_attestation)
        raw["nonce"] = params["nonce"]
        sv = json.loads(json.dumps(raw_attestation["server_verification"]))
        addr = raw_attestation["signing_address"].lower().removeprefix("0x")
        sv["tdx"]["reportData"] = addr + ("00" * 12) + params["nonce"]
        raw["server_verification"] = sv
        return TeeAttestation.model_validate(raw)

    tee._client.get = AsyncMock(side_effect=_echo)  # type: ignore[attr-defined]
    result = await tee.get_attestation(model="e2ee-gemma-3-27b-p")

    generated = tee._client.get.call_args.kwargs["params"]["nonce"]  # type: ignore[attr-defined]
    assert len(generated) == NONCE_LEN_HEX
    bytes.fromhex(generated)  # valid hex
    assert result.sent_nonce == generated


@pytest.mark.asyncio
async def test_get_attestation_fails_closed_on_bad_verification(tee: Tee) -> None:
    raw, _model_priv, nonce = _synthetic_attestation()
    raw["verified"] = False  # break it
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    with pytest.raises(TeeAttestationError):
        await tee.get_attestation(model="e2ee-x", nonce=nonce)


# --- open_session ------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_session_round_trips_encrypt_decrypt(tee: Tee) -> None:
    raw, model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    session = await tee.open_session(model="e2ee-x", nonce=nonce)
    assert isinstance(session, TeeSession)

    headers = session.request_headers()
    # Header model key matches the attestation's signing key.
    assert headers[HEADER_MODEL_PUB_KEY] == raw["signing_public_key"]

    # Request direction: encrypt -> the model private key decrypts it.
    blob = session.encrypt_message("secret")
    assert _crypto.decrypt_chunk(model_priv, blob) == "secret"

    # Response direction: server encrypts to the SESSION pub -> decrypt_chunk.
    server_blob = _crypto.encrypt_message(headers[HEADER_CLIENT_PUB_KEY], "reply")
    assert session.decrypt_chunk(server_blob) == "reply"


@pytest.mark.asyncio
async def test_open_session_attaches_trust(tee: Tee) -> None:
    raw, _model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    session = await tee.open_session(model="e2ee-x", nonce=nonce)
    trust = session.trust
    assert trust is not None
    assert trust.model == "e2ee-x"
    assert trust.verification == "baseline"
    assert trust.verifier is None
    assert trust.server_verified is True
    assert trust.nonce == nonce
    assert trust.model_public_key == raw["signing_public_key"]
    assert trust.signing_address == raw["signing_address"]

    session.close()
    assert session.trust is trust

    full = await tee.open_session(model="e2ee-x", nonce=nonce, verifier=_SpyVerifier(result=True))
    assert full.trust is not None
    assert full.trust.verification == "full_quote"
    assert full.trust.verifier == "_SpyVerifier"


@pytest.mark.asyncio
async def test_open_session_fails_closed_on_bad_attestation(tee: Tee) -> None:
    raw, _model_priv, nonce = _synthetic_attestation()
    raw["verified"] = False
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    with pytest.raises(TeeAttestationError):
        await tee.open_session(model="e2ee-x", nonce=nonce)


# --- client wiring -----------------------------------------------------------


# --- FullQuoteVerifier wiring ------------------------------------------------


class _SpyVerifier:
    """Minimal :class:`FullQuoteVerifier` that records what it saw at verify-time.

    Captures ``attestation.sent_nonce`` at the *moment* ``verify`` is invoked so
    a regression that moves the ``sent_nonce`` assignment back after the verify
    call would surface as a captured ``None``.
    """

    def __init__(self, *, result: bool = True, raises: bool = False) -> None:
        self._result = result
        self._raises = raises
        self.called = False
        self.seen_sent_nonce: str | None = None

    def verify(self, attestation: TeeAttestation) -> bool:
        self.called = True
        self.seen_sent_nonce = attestation.sent_nonce
        if self._raises:
            raise RuntimeError("boom")
        return self._result


@pytest.mark.asyncio
async def test_verifier_sees_populated_sent_nonce_before_running(tee: Tee) -> None:
    """REGRESSION: the resource must set ``sent_nonce`` BEFORE verification runs.

    A ``FullQuoteVerifier`` reads ``attestation.sent_nonce`` for the REPORTDATA
    key binding; if the resource assigned it *after* invoking verification the
    verifier would see ``None``. This locks the ordering in ``resources/tee.py``.
    """
    raw, _model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    spy = _SpyVerifier(result=True)

    result = await tee.get_attestation(model="e2ee-x", nonce=nonce, verifier=spy)

    assert spy.called is True
    assert spy.seen_sent_nonce == nonce  # NOT None — populated before verify ran
    assert result.sent_nonce == nonce


@pytest.mark.asyncio
async def test_get_attestation_runs_verifier_and_passes_on_true(tee: Tee) -> None:
    raw, _model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    spy = _SpyVerifier(result=True)

    result = await tee.get_attestation(model="e2ee-x", nonce=nonce, verifier=spy)

    assert spy.called is True
    assert result.verified is True


@pytest.mark.asyncio
async def test_get_attestation_fails_closed_when_verifier_returns_false(tee: Tee) -> None:
    raw, _model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    spy = _SpyVerifier(result=False)

    with pytest.raises(TeeAttestationError):
        await tee.get_attestation(model="e2ee-x", nonce=nonce, verifier=spy)
    assert spy.called is True


@pytest.mark.asyncio
async def test_get_attestation_fails_closed_when_verifier_raises(tee: Tee) -> None:
    raw, _model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )
    spy = _SpyVerifier(raises=True)

    with pytest.raises(TeeAttestationError):
        await tee.get_attestation(model="e2ee-x", nonce=nonce, verifier=spy)
    assert spy.called is True


@pytest.mark.asyncio
async def test_open_session_runs_verifier_and_surfaces_failure(tee: Tee) -> None:
    """``open_session`` (always fail-closed) must run the verifier and raise on False."""
    raw, _model_priv, nonce = _synthetic_attestation()
    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(raw)
    )

    # Passing verifier -> session opens.
    spy_ok = _SpyVerifier(result=True)
    session = await tee.open_session(model="e2ee-x", nonce=nonce, verifier=spy_ok)
    assert isinstance(session, TeeSession)
    assert spy_ok.called is True
    assert spy_ok.seen_sent_nonce == nonce

    # Failing verifier -> fail-closed.
    spy_bad = _SpyVerifier(result=False)
    with pytest.raises(TeeAttestationError):
        await tee.open_session(model="e2ee-x", nonce=nonce, verifier=spy_bad)
    assert spy_bad.called is True


# --- DcapTdxVerifier end-to-end through the resource -------------------------


@pytest.mark.asyncio
async def test_get_attestation_runs_real_dcap_verifier(tee: Tee) -> None:
    """End-to-end: ``DcapTdxVerifier(collateral=..., now_secs=...)`` via the resource.

    Reuses the live-captured fixture corpus + collateral snapshot from
    ``test_verify``; ``now_secs`` is frozen to the capture epoch so the
    collateral validity window passes. Proves the named verifier surfaces its
    result through ``get_attestation`` and saw a populated ``sent_nonce``.
    """
    dcap_qvl = pytest.importorskip("dcap_qvl", reason="requires the [e2ee-verify] extra (dcap-qvl)")
    from venice_ai.tee import DcapTdxVerifier
    from venice_ai.tee._constants import INTEL_SGX_ROOT_CA_DER

    from .fixtures import CAPTURE_EPOCH, load_attestation, load_collateral_json

    slug = "gemma_3_27b_p"
    raw = load_attestation(slug)
    payload = raw["attestation"]
    nonce = raw["_client_nonce"]
    collateral = dcap_qvl.QuoteCollateralV3.from_json(load_collateral_json(slug))

    tee._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=TeeAttestation.model_validate(payload)
    )
    verifier = DcapTdxVerifier(
        collateral=collateral,
        root_ca_der=INTEL_SGX_ROOT_CA_DER,
        now_secs=CAPTURE_EPOCH,
    )

    result = await tee.get_attestation(model="e2ee-x", nonce=nonce, verifier=verifier)

    assert result.sent_nonce == nonce
    assert verifier.last_result is not None
    assert verifier.last_result["checks"]["signature_chain"] is True
    assert verifier.last_result["checks"]["reportdata_binding"] is True


def test_tee_attached_to_async_client() -> None:
    from venice_ai import VeniceClient

    # No request is made, so no aiohttp session is opened (nothing to close).
    client = VeniceClient(api_key="test-key")
    assert isinstance(client.tee, Tee)


def test_tee_attached_to_sync_client() -> None:
    from venice_ai._sync_client import SyncVeniceClient

    with SyncVeniceClient(api_key="test-key") as client:
        # The sync proxy wraps the async resource; attribute access must resolve.
        proxy = client.tee
        assert proxy is not None
        # The wrapped target is the async Tee resource.
        assert isinstance(proxy._target, Tee)  # type: ignore[attr-defined]
