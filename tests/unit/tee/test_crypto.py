"""Core-phase tests for ``venice_ai.tee._crypto``.

These exercise the pure, synchronous, network-free crypto primitives offline.
The crypto is **not symmetric in the obvious way**: the request path encrypts to
the model's public key with a fresh per-message ephemeral keypair, while the
response path is decrypted with the SESSION private key against a *server*
ephemeral public key. The response-format tests therefore **simulate the server
side** locally (a fake server-ephemeral keypair encrypts to the SESSION pub).

The pinned known-answer test (``test_encrypt_message_pinned_kat``) freezes a
literal ``content_hex`` computed *independently* of the implementation (mirroring
the live probe's algorithm with fixed inputs). This pins, all at once:
``info=b"ecdsa_encryption"``, ``salt=None``, raw-X (unhashed) ECDH, AES-256-GCM
with the tag appended, ``AAD=None``, and the ``eph_pub(65) || nonce(12) || ct``
wire ordering. A regression in any of those breaks the literal.
"""

from __future__ import annotations

import pytest

# cryptography is the optional [e2ee] extra; these tests require it.
ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee crypto tests require the [e2ee] extra (cryptography)",
)
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402
from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # noqa: E402

from venice_ai.exceptions import TeeEncryptionError  # noqa: E402
from venice_ai.tee import _crypto  # noqa: E402
from venice_ai.tee._constants import (  # noqa: E402
    HKDF_INFO,
    MIN_ENCRYPTED_HEX_LEN,
    UNCOMPRESSED_PUBKEY_LEN,
)

_CURVE = ec.SECP256K1()

# --- Pinned KAT inputs (frozen literals; see module docstring) ---------------

# A fixed model private key (server side) and a fixed ephemeral private key
# (per-message, client side). Both are deterministic scalars.
_KAT_MODEL_PRIV_INT = 0x1111111111111111111111111111111111111111111111111111111111111111
_KAT_EPH_PRIV_INT = 0x2222222222222222222222222222222222222222222222222222222222222222
_KAT_GCM_NONCE = bytes.fromhex("0c0c0c0c0c0c0c0c0c0c0c0c")
_KAT_PLAINTEXT = "PROBE_OK"

# The model public key for the fixed model private key (130-hex, uncompressed).
_KAT_MODEL_PUB_HEX = (
    "044f355bdcb7cc0af728ef3cceb9615d90684bb5b2ca5f859ab0f0b704075871aa"
    "385b6b1b8ead809ca67454d9683fcf2ba03456d6fe2c4abe2b07f0fbdbb2f1c1"
)

# The EXACT content_hex produced by encrypting _KAT_PLAINTEXT to _KAT_MODEL_PUB
# with the fixed ephemeral key + fixed gcm nonce. Computed independently of the
# implementation under test (mirroring the live probe). DO NOT recompute this in
# the assert from the same primitives -- that would make the KAT vacuous.
_KAT_CONTENT_HEX = (
    "04466d7fcae563e5cb09a0d1870bb580344804617879a14949cf22285f1bae3f27"
    "6728176c3c6431f8eeda4538dc37c865e2784f3a9e77d044f33e407797e1278a"
    "0c0c0c0c0c0c0c0c0c0c0c0c"
    "61fcb8d3782019da4a51df38195ea647df38e4b1b4cf7af6"
)


def _uncompressed_hex(pub: object) -> str:
    raw: bytes = pub.public_bytes(  # type: ignore[attr-defined]
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    return raw.hex()


def _server_derive_key(priv: object, peer_pub: object) -> bytes:
    """Reference KDF mirroring the probe, used to simulate the server side."""
    shared = priv.exchange(ec.ECDH(), peer_pub)  # type: ignore[attr-defined]
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=HKDF_INFO).derive(shared)


# --- KAT ----------------------------------------------------------------------


def test_encrypt_message_pinned_kat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed model key + fixed ephemeral + fixed nonce → the exact frozen hex."""
    fixed_eph = ec.derive_private_key(_KAT_EPH_PRIV_INT, _CURVE)
    monkeypatch.setattr(_crypto, "generate_ephemeral_keypair", lambda: fixed_eph)
    monkeypatch.setattr(_crypto.os, "urandom", lambda n: _KAT_GCM_NONCE)

    out = _crypto.encrypt_message(_KAT_MODEL_PUB_HEX, _KAT_PLAINTEXT)

    assert out == _KAT_CONTENT_HEX
    assert out == out.lower()
    assert "0x" not in out


def test_ecdh_shared_is_32_raw_bytes() -> None:
    """ECDH shared secret is exactly 32 bytes (raw big-endian X, unhashed)."""
    a = ec.derive_private_key(_KAT_MODEL_PRIV_INT, _CURVE)
    b = ec.derive_private_key(_KAT_EPH_PRIV_INT, _CURVE)
    shared = a.exchange(ec.ECDH(), b.public_key())
    assert len(shared) == 32
    # The derived AES key must be the HKDF of the raw X, not the raw X itself.
    key = _crypto.derive_key(a, b.public_key())
    assert len(key) == 32
    assert bytes(key) != shared  # HKDF-transformed, not the raw shared secret


# --- Request roundtrip (client encrypts; simulated server decrypts) -----------


@pytest.mark.parametrize(
    "plaintext",
    ["hello", "", "a" * 5000, "unicode: café 🐉 日本語", "PROBE_OK"],
)
def test_request_roundtrip(plaintext: str) -> None:
    """encrypt_message → parse on the server side → decrypt → original."""
    model_priv = ec.generate_private_key(_CURVE)
    model_pub_hex = _uncompressed_hex(model_priv.public_key())

    content_hex = _crypto.encrypt_message(model_pub_hex, plaintext)

    # Wire sanity: lowercase, no prefix, even, >= min length.
    assert content_hex == content_hex.lower()
    assert not content_hex.startswith("0x")
    assert len(content_hex) % 2 == 0
    assert len(content_hex) >= MIN_ENCRYPTED_HEX_LEN

    # Simulate the server: split eph_pub || nonce || ct, derive, decrypt.
    raw = bytes.fromhex(content_hex)
    server_eph_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        _CURVE, raw[:UNCOMPRESSED_PUBKEY_LEN]
    )
    gcm_nonce = raw[UNCOMPRESSED_PUBKEY_LEN : UNCOMPRESSED_PUBKEY_LEN + 12]
    ct = raw[UNCOMPRESSED_PUBKEY_LEN + 12 :]
    key = _server_derive_key(model_priv, server_eph_pub)
    recovered = AESGCM(key).decrypt(gcm_nonce, ct, None).decode()

    assert recovered == plaintext


# --- Response roundtrip (simulated server encrypts; client decrypts) ----------


def test_response_roundtrip() -> None:
    """Server ephemeral key encrypts to the SESSION pub; client decrypts."""
    session_priv = _crypto.generate_session_keypair()
    session_pub_hex = _crypto.uncompressed_hex(session_priv.public_key())

    # Server side: fresh ephemeral key, ECDH to the session pub, AES-GCM.
    server_eph = ec.generate_private_key(_CURVE)
    server_eph_pub = server_eph.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    session_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        _CURVE, bytes.fromhex(session_pub_hex)
    )
    key = _server_derive_key(server_eph, session_pub)
    gcm_nonce = b"\x01" * 12
    token = "the lazy dog"
    ct = AESGCM(key).encrypt(gcm_nonce, token.encode(), None)
    chunk_hex = (server_eph_pub + gcm_nonce + ct).hex()

    assert _crypto.decrypt_chunk(session_priv, chunk_hex) == token


# --- looks_encrypted boundaries -----------------------------------------------


def test_looks_encrypted_boundaries() -> None:
    hexchars = "0123456789abcdef"
    just_under = (hexchars * 100)[: MIN_ENCRYPTED_HEX_LEN - 1]  # 185
    exactly = (hexchars * 100)[:MIN_ENCRYPTED_HEX_LEN]  # 186

    assert len(just_under) == 185
    assert len(exactly) == 186

    assert _crypto.looks_encrypted(just_under) is False  # too short
    assert _crypto.looks_encrypted(exactly) is True  # at the boundary
    assert _crypto.looks_encrypted(exactly[:-1] + "z") is False  # non-hex
    assert _crypto.looks_encrypted(exactly.upper()) is False  # uppercase wire never used
    assert _crypto.looks_encrypted(exactly + "a") is False  # odd length
    assert _crypto.looks_encrypted("") is False  # empty
    assert _crypto.looks_encrypted("hello world, this is plain text") is False


def test_encrypt_output_looks_encrypted_for_empty_plaintext() -> None:
    """Even an empty plaintext yields a blob that passes looks_encrypted."""
    model_priv = ec.generate_private_key(_CURVE)
    out = _crypto.encrypt_message(_uncompressed_hex(model_priv.public_key()), "")
    # 65 (eph) + 12 (nonce) + 16 (tag) = 93 bytes = 186 hex.
    assert len(out) == MIN_ENCRYPTED_HEX_LEN
    assert _crypto.looks_encrypted(out) is True


# --- Key normalization --------------------------------------------------------


def test_public_key_from_hex_normalizes_128_to_130() -> None:
    """A 128-hex point (no 0x04 prefix) is accepted and normalized."""
    priv = ec.generate_private_key(_CURVE)
    full_hex = _uncompressed_hex(priv.public_key())  # 130, 04-prefixed
    assert full_hex.startswith("04")
    body_128 = full_hex[2:]  # strip the 04 → 128 hex

    key_from_130 = _crypto.public_key_from_hex(full_hex)
    key_from_128 = _crypto.public_key_from_hex(body_128)

    assert _crypto.uncompressed_hex(key_from_128) == full_hex
    assert _crypto.uncompressed_hex(key_from_130) == full_hex


def test_public_key_from_hex_rejects_off_curve() -> None:
    """An off-curve / malformed point is rejected."""
    bad = "04" + "00" * 64  # (0, 0) is not on secp256k1
    with pytest.raises((ValueError, TeeEncryptionError)):
        _crypto.public_key_from_hex(bad)


def test_uncompressed_hex_is_130_lowercase() -> None:
    priv = _crypto.generate_session_keypair()
    h = _crypto.uncompressed_hex(priv.public_key())
    assert len(h) == UNCOMPRESSED_PUBKEY_LEN * 2  # 130
    assert h.startswith("04")
    assert h == h.lower()


# --- Negative crypto (fail closed; never emit corrupt plaintext) --------------


def test_decrypt_chunk_tampered_tag_raises() -> None:
    """Flipping a ciphertext byte → GCM tag failure → TeeEncryptionError."""
    session_priv = _crypto.generate_session_keypair()
    session_pub_hex = _crypto.uncompressed_hex(session_priv.public_key())

    server_eph = ec.generate_private_key(_CURVE)
    server_eph_pub = server_eph.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    session_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        _CURVE, bytes.fromhex(session_pub_hex)
    )
    key = _server_derive_key(server_eph, session_pub)
    gcm_nonce = b"\x02" * 12
    ct = AESGCM(key).encrypt(gcm_nonce, b"secret", None)

    raw = bytearray(server_eph_pub + gcm_nonce + ct)
    raw[-1] ^= 0xFF  # tamper the tag

    with pytest.raises(TeeEncryptionError):
        _crypto.decrypt_chunk(session_priv, raw.hex())


def test_decrypt_chunk_wrong_key_raises() -> None:
    """Decrypting with the wrong session key → tag failure → TeeEncryptionError."""
    right = _crypto.generate_session_keypair()
    wrong = _crypto.generate_session_keypair()
    right_pub_hex = _crypto.uncompressed_hex(right.public_key())

    server_eph = ec.generate_private_key(_CURVE)
    server_eph_pub = server_eph.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    right_pub = ec.EllipticCurvePublicKey.from_encoded_point(_CURVE, bytes.fromhex(right_pub_hex))
    key = _server_derive_key(server_eph, right_pub)
    gcm_nonce = b"\x03" * 12
    ct = AESGCM(key).encrypt(gcm_nonce, b"secret", None)
    chunk_hex = (server_eph_pub + gcm_nonce + ct).hex()

    with pytest.raises(TeeEncryptionError):
        _crypto.decrypt_chunk(wrong, chunk_hex)


def test_decrypt_chunk_malformed_hex_raises() -> None:
    with pytest.raises(TeeEncryptionError):
        _crypto.decrypt_chunk(_crypto.generate_session_keypair(), "not-valid-hex!!")


def test_decrypt_chunk_too_short_raises() -> None:
    with pytest.raises(TeeEncryptionError):
        _crypto.decrypt_chunk(_crypto.generate_session_keypair(), "abcd")


# --- Errors carry no plaintext ------------------------------------------------


def _assert_no_plaintext(error: BaseException, secret: str) -> None:
    assert secret not in str(error)
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__ is True


def test_encrypt_failure_does_not_chain_plaintext() -> None:
    model_pub = _crypto.uncompressed_hex(ec.generate_private_key(_CURVE).public_key())
    with pytest.raises(TeeEncryptionError, match="UnicodeEncodeError") as excinfo:
        _crypto.encrypt_message(model_pub, "SECRETXYZ \ud800")
    _assert_no_plaintext(excinfo.value, "SECRETXYZ")


def test_decrypt_failure_does_not_chain_plaintext() -> None:
    session_priv = _crypto.generate_session_keypair()
    server_eph = ec.generate_private_key(_CURVE)
    server_eph_pub = server_eph.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    key = _server_derive_key(server_eph, session_priv.public_key())
    gcm_nonce = b"\x04" * 12
    ct = AESGCM(key).encrypt(gcm_nonce, b"SECRETXYZ \xe2\x82", None)
    chunk_hex = (server_eph_pub + gcm_nonce + ct).hex()
    with pytest.raises(TeeEncryptionError, match="UnicodeDecodeError") as excinfo:
        _crypto.decrypt_chunk(session_priv, chunk_hex)
    _assert_no_plaintext(excinfo.value, "SECRETXYZ")
