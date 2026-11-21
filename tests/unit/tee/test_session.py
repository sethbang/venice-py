"""Session-phase tests for ``venice_ai.tee._session.TeeSession``.

These exercise the SESSION keypair / header / encrypt / decrypt surface offline.
The load-bearing trap pinned here: ``X-Venice-TEE-Client-Pub-Key`` is the
**SESSION** public key (stable across calls, private half decrypts the response),
**not** a per-message ephemeral key (which is fresh per :meth:`encrypt_message`
and differs from the header and from message to message).
"""

from __future__ import annotations

import pytest

# cryptography is the optional [e2ee] extra; these tests require it.
ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee session tests require the [e2ee] extra (cryptography)",
)

from venice_ai.exceptions import TeeEncryptionError  # noqa: E402
from venice_ai.tee import _crypto  # noqa: E402
from venice_ai.tee._constants import (  # noqa: E402
    HEADER_CLIENT_PUB_KEY,
    HEADER_MODEL_PUB_KEY,
    HEADER_SIGNING_ALGO,
    UNCOMPRESSED_PUBKEY_LEN,
)
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.tee.types import TeeAttestation  # noqa: E402

_CURVE = ec.SECP256K1()
_EPH_PREFIX_HEX = UNCOMPRESSED_PUBKEY_LEN * 2  # 130 hex chars


def _model_keypair() -> tuple[ec.EllipticCurvePrivateKey, str]:
    """Return a (private, 130-hex public) model keypair we control."""
    priv = ec.generate_private_key(_CURVE)
    pub_hex = _crypto.uncompressed_hex(priv.public_key())
    return priv, pub_hex


def _build_session() -> tuple[TeeSession, ec.EllipticCurvePrivateKey, str]:
    """Build a session against a model keypair we own (so we can decrypt requests)."""
    model_priv, model_pub_hex = _model_keypair()
    session = TeeSession(
        session_private_key=ec.generate_private_key(_CURVE),
        model_public_key_hex=model_pub_hex,
        signing_algo="ecdsa",
    )
    return session, model_priv, model_pub_hex


# --- request_headers ---------------------------------------------------------


def test_request_headers_has_three_confirmed_names() -> None:
    session, _model_priv, model_pub_hex = _build_session()
    headers = session.request_headers()
    assert set(headers) == {
        HEADER_CLIENT_PUB_KEY,
        HEADER_MODEL_PUB_KEY,
        HEADER_SIGNING_ALGO,
    }
    assert headers[HEADER_MODEL_PUB_KEY] == model_pub_hex
    assert headers[HEADER_SIGNING_ALGO] == "ecdsa"


def test_client_pub_key_is_the_session_pub_not_a_per_message_ephemeral() -> None:
    """THE trap: Client-Pub-Key == SESSION pub, stable, != any message prefix."""
    session, _model_priv, _model_pub_hex = _build_session()
    headers = session.request_headers()
    client_pub = headers[HEADER_CLIENT_PUB_KEY]

    # It equals the session's public key.
    assert client_pub == session.session_public_key_hex

    # It is stable across calls (a per-message ephemeral would change).
    assert session.request_headers()[HEADER_CLIENT_PUB_KEY] == client_pub

    # Encrypting two messages yields two DIFFERENT ephemeral prefixes, and
    # NEITHER equals the header Client-Pub-Key.
    blob1 = session.encrypt_message("first")
    blob2 = session.encrypt_message("second")
    prefix1 = blob1[:_EPH_PREFIX_HEX]
    prefix2 = blob2[:_EPH_PREFIX_HEX]
    assert prefix1 != prefix2  # fresh ephemeral per message
    assert prefix1 != client_pub
    assert prefix2 != client_pub


# --- round-trip --------------------------------------------------------------


def test_encrypt_message_round_trips_to_the_model_private_key() -> None:
    """Request direction: session.encrypt_message -> model_priv decrypts it."""
    session, model_priv, _model_pub_hex = _build_session()
    blob = session.encrypt_message("confidential prompt")
    # The server (model) side decrypts a request with its OWN private key against
    # the per-message ephemeral pub embedded in the blob.
    assert _crypto.decrypt_chunk(model_priv, blob) == "confidential prompt"


def test_decrypt_chunk_round_trips_from_a_server_ephemeral() -> None:
    """Response direction: server encrypts to SESSION pub -> session.decrypt_chunk."""
    session, _model_priv, _model_pub_hex = _build_session()
    session_pub = session.request_headers()[HEADER_CLIENT_PUB_KEY]
    # Simulate the server encrypting a response chunk to the SESSION public key.
    server_blob = _crypto.encrypt_message(session_pub, "decrypted answer")
    assert session.decrypt_chunk(server_blob) == "decrypted answer"


def test_from_attestation_builds_a_working_session() -> None:
    """from_attestation pulls model_pub + algo off the attestation and works E2E."""
    pytest.importorskip("Crypto.Hash.keccak", reason="key binding needs the [e2ee] extra")
    from venice_ai.tee._attestation import _keccak256

    _model_priv, model_pub_hex = _model_keypair()
    address = _keccak256(bytes.fromhex(model_pub_hex)[1:])[-20:].hex()
    attestation = TeeAttestation(
        verified=True,
        signing_public_key=model_pub_hex,
        signing_address="0x" + address,
        signing_algo="ecdsa",
        nonce="00" * 32,
    )
    session = TeeSession.from_attestation(attestation)
    headers = session.request_headers()
    assert headers[HEADER_MODEL_PUB_KEY] == model_pub_hex
    assert headers[HEADER_SIGNING_ALGO] == "ecdsa"
    # Round-trip the response direction through the generated SESSION key.
    server_blob = _crypto.encrypt_message(headers[HEADER_CLIENT_PUB_KEY], "ok")
    assert session.decrypt_chunk(server_blob) == "ok"


# --- close / context manager -------------------------------------------------


def test_close_drops_secret_and_blocks_further_use() -> None:
    session, _model_priv, _model_pub_hex = _build_session()
    session.close()
    with pytest.raises(TeeEncryptionError):
        session.request_headers()
    with pytest.raises(TeeEncryptionError):
        session.encrypt_message("nope")
    with pytest.raises(TeeEncryptionError):
        session.decrypt_chunk("00" * 100)


def test_context_manager_closes_on_exit() -> None:
    session, _model_priv, _model_pub_hex = _build_session()
    with session as ctx:
        assert ctx is session
        ctx.request_headers()  # works inside the block
    with pytest.raises(TeeEncryptionError):
        session.request_headers()  # closed on exit
