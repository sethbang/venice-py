"""``TeeTrust`` construction and its carriage on ``TeeSession``."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee session tests require the [e2ee] extra (cryptography)",
)

pytest.importorskip("Crypto.Hash.keccak", reason="key binding needs the [e2ee] extra")

from venice_ai.tee._attestation import _keccak256  # noqa: E402
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.tee.types import TeeAttestation, TeeTrust  # noqa: E402


def _attestation() -> TeeAttestation:
    from venice_ai.tee import _crypto

    pub = _crypto.uncompressed_hex(_crypto.generate_session_keypair().public_key())
    address = _keccak256(bytes.fromhex(pub)[1:])[-20:].hex()
    return TeeAttestation.model_validate(
        {
            "verified": True,
            "signing_public_key": pub,
            "signing_address": "0x" + address,
            "signing_algo": "ecdsa",
            "nonce": "cd" * 32,
        }
    )


class _StubVerifier:
    def verify(self, attestation: TeeAttestation) -> bool:
        return True


def test_tee_trust_from_attestation_baseline_and_full():
    att = _attestation()
    baseline = TeeTrust.from_attestation(att, model="e2ee-x", verifier=None)
    assert baseline.verification == "baseline"
    assert baseline.verifier is None
    assert baseline.model == "e2ee-x"
    assert baseline.server_verified is True
    assert baseline.nonce == att.nonce
    assert baseline.model_public_key == att.signing_public_key
    assert baseline.signing_address == att.signing_address

    full = TeeTrust.from_attestation(att, model="e2ee-x", verifier=_StubVerifier())
    assert full.verification == "full_quote"
    assert full.verifier == "_StubVerifier"


def test_session_trust_defaults_to_none():
    session = TeeSession.from_attestation(_attestation())
    assert session.trust is None


def test_session_trust_survives_close():
    att = _attestation()
    trust = TeeTrust.from_attestation(att, model="e2ee-x", verifier=None)
    session = TeeSession.from_attestation(att, trust=trust)
    assert session.trust is trust
    session.close()
    assert session.trust is trust


def test_tee_trust_from_attestation_without_model_key_raises():
    from venice_ai.exceptions import TeeEncryptionError

    att = _attestation().model_copy(update={"signing_public_key": None})
    with pytest.raises(TeeEncryptionError, match="signing_public_key"):
        TeeTrust.from_attestation(att, model="e2ee-x", verifier=None)
