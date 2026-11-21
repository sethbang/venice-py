"""Pure, synchronous, network-free crypto primitives for Venice TEE E2EE.

This module owns the secp256k1 ECDH + HKDF-SHA256 + AES-256-GCM primitives used
by the client-side end-to-end-encryption path (``e2ee-*`` chat models). Every
function here is deterministic given its inputs (modulo the per-message
ephemeral key and the GCM nonce, both drawn from the OS RNG), takes no network
I/O, and is shared verbatim between the sync and async clients.

``cryptography`` is an **optional** dependency (the ``[e2ee]`` extra). This
module must stay import-time dependency-free so that ``import venice_ai.tee``
succeeds on a bare install; the actual ``cryptography`` import happens lazily
inside :func:`_require_crypto` / the primitive functions, which raise a clear,
actionable :class:`ImportError` when the extra is not installed.

Wire format (live-verified against the Venice API on 2026-06-04):
the encrypted content of every request message and every response chunk is the
lowercase hex of ``ephemeral_pub(65) || gcm_nonce(12) || ciphertext+tag``.

Crypto: secp256k1 ECDH whose shared secret is the **raw 32-byte big-endian X**
coordinate (NOT hashed), fed through ``HKDF-SHA256(salt=None,
info=b"ecdsa_encryption", length=32)`` to an AES-256 key, then AES-256-GCM with
a 12-byte random nonce and ``AAD=None`` (16-byte tag appended).

**Zeroization is best-effort and honest.** Derived secret ``bytearray``s (the
ECDH raw-X and the AES key) are overwritten with zeros after use. The immutable
intermediates that ``cryptography`` hands back (the ``exchange()`` return value,
the HKDF output, and ``AESGCM``'s internal copy of the key) cannot be wiped from
Python, and the private-key *objects* expose no wipeable buffer. We do not claim
otherwise.
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from ..exceptions import TeeEncryptionError
from ._constants import (
    GCM_NONCE_LEN,
    HKDF_INFO,
    HKDF_LENGTH,
    MIN_ENCRYPTED_HEX_LEN,
    UNCOMPRESSED_PUBKEY_LEN,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cryptography.hazmat.primitives.asymmetric.ec import (
        EllipticCurvePrivateKey,
        EllipticCurvePublicKey,
    )

_INSTALL_HINT = (
    "Venice TEE end-to-end encryption requires the ``cryptography`` package, "
    "which is an optional dependency. Install it with: "
    "pip install 'venice-py[e2ee]'"
)

#: Byte offset where the GCM nonce begins within a decoded wire blob.
_GCM_NONCE_OFFSET = UNCOMPRESSED_PUBKEY_LEN
#: Byte offset where the ciphertext+tag begins within a decoded wire blob.
_CT_OFFSET = UNCOMPRESSED_PUBKEY_LEN + GCM_NONCE_LEN


def _require_crypto() -> ModuleType:
    """Import and return the ``cryptography`` top-level module, lazily.

    Returns:
        The imported ``cryptography`` module.

    Raises:
        ImportError: If ``cryptography`` is not installed, with a hint to
            install the ``[e2ee]`` extra.
    """
    try:
        import cryptography  # noqa: PLC0415  (lazy by design — optional dep)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    # Typed local so the return is `ModuleType` whether or not cryptography's
    # types resolve (it is `Any` when the optional [e2ee] extra is absent, e.g.
    # in CI's lint env). cast() would flip to a redundant-cast error when the
    # extra IS installed; this idiom is clean in both conditions.
    mod: ModuleType = cryptography
    return mod


def _ec_module() -> Any:
    """Lazily import the ``cryptography`` EC asymmetric module."""
    _require_crypto()
    from cryptography.hazmat.primitives.asymmetric import ec  # noqa: PLC0415

    return ec


# --- Keypairs -----------------------------------------------------------------


def generate_session_keypair() -> EllipticCurvePrivateKey:
    """Generate a SESSION secp256k1 private key.

    One per :meth:`chat.completions.create` call. Its **public** half goes in the
    ``X-Venice-TEE-Client-Pub-Key`` header; its **private** half is retained to
    decrypt the streamed response. Never use it to encrypt a request message.
    """
    ec = _ec_module()
    return cast("EllipticCurvePrivateKey", ec.generate_private_key(ec.SECP256K1()))


def generate_ephemeral_keypair() -> EllipticCurvePrivateKey:
    """Generate a fresh PER-MESSAGE ephemeral secp256k1 private key.

    Used **only** to encrypt one request message; its public half becomes the
    65-byte prefix of that message's ciphertext. It must never escape
    :func:`encrypt_message` and must never be placed in a request header.
    """
    ec = _ec_module()
    return cast("EllipticCurvePrivateKey", ec.generate_private_key(ec.SECP256K1()))


# --- Public-key (de)serialization --------------------------------------------


def uncompressed_hex(public_key: EllipticCurvePublicKey) -> str:
    """Serialize a public key to uncompressed SEC1 lowercase hex (130 chars).

    Layout: ``0x04 || X(32) || Y(32)`` → 65 bytes → 130 lowercase hex chars.
    """
    _require_crypto()
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

    # Typed local: `str` whether or not cryptography's types resolve (see _require_crypto).
    encoded: str = public_key.public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    ).hex()
    return encoded


def public_key_from_hex(hex_str: str) -> EllipticCurvePublicKey:
    """Parse an uncompressed secp256k1 public key from hex.

    Accepts either the full 130-hex form (``04``-prefixed) or the bare 128-hex
    point body, in which case the ``04`` prefix is prepended. ``from_encoded_point``
    validates that the point lies on the curve, so off-curve / malformed input is
    rejected.

    Raises:
        ValueError: If the hex is malformed or the point is not on the curve.
    """
    ec = _ec_module()
    normalized = hex_str.strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    # Normalize a bare 128-hex point body to the 130-hex uncompressed form.
    if len(normalized) == (UNCOMPRESSED_PUBKEY_LEN - 1) * 2:
        normalized = "04" + normalized
    raw = bytes.fromhex(normalized)
    return cast(
        "EllipticCurvePublicKey",
        ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), raw),
    )


# --- Key agreement ------------------------------------------------------------


def derive_key(priv: EllipticCurvePrivateKey, peer_pub: EllipticCurvePublicKey) -> bytearray:
    """Derive the AES-256 key via secp256k1 ECDH + HKDF-SHA256.

    The ECDH shared secret is the **raw 32-byte big-endian X** coordinate (NOT
    hashed); it is fed through ``HKDF-SHA256(salt=None, info=b"ecdsa_encryption",
    length=32)`` to produce the AES-256 key.

    The intermediate raw-X is held in a ``bytearray`` and zeroed after the HKDF
    step. The returned key is a mutable ``bytearray`` so the caller can zero it
    after building the cipher.
    """
    _require_crypto()
    from cryptography.hazmat.primitives import hashes  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import ec  # noqa: PLC0415
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # noqa: PLC0415

    shared = bytearray(priv.exchange(ec.ECDH(), peer_pub))  # raw 32-byte X
    try:
        key = HKDF(
            algorithm=hashes.SHA256(),
            length=HKDF_LENGTH,
            salt=None,
            info=HKDF_INFO,
        ).derive(bytes(shared))
        return bytearray(key)
    finally:
        # Overwrite the raw shared secret; the immutable copy taken by bytes()
        # above cannot be wiped (honest limitation).
        for i in range(len(shared)):
            shared[i] = 0


# --- Encrypt / decrypt --------------------------------------------------------


def _content_free_error(prefix: str, exc: Exception) -> TeeEncryptionError:
    """A :class:`TeeEncryptionError` naming only the type of ``exc``.

    ``str(exc)`` and the exception object itself can carry message text (a
    ``UnicodeEncodeError`` / ``UnicodeDecodeError`` keeps the whole input on
    ``.object``), so the replacement quotes neither. Callers raise it outside
    the ``except`` block so it has no ``__cause__`` and no ``__context__``.
    """
    error = TeeEncryptionError(f"{prefix} ({type(exc).__name__}).")
    error.__suppress_context__ = True
    return error


def encrypt_message(model_pub: str, plaintext: str) -> str:
    """Encrypt a request message to the model's public key.

    A fresh per-message ephemeral keypair is generated (via the module-level
    :func:`generate_ephemeral_keypair`); the ECDH/HKDF-derived AES-256-GCM key
    encrypts the UTF-8 plaintext with a random 12-byte nonce and ``AAD=None``.

    Returns:
        Lowercase hex of ``ephemeral_pub(65) || gcm_nonce(12) || ciphertext+tag``.

    Raises:
        TeeEncryptionError: If the model public key is malformed or encryption
            fails.
    """
    _require_crypto()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

    try:
        eph = generate_ephemeral_keypair()
        eph_pub_hex = uncompressed_hex(eph.public_key())
        peer = public_key_from_hex(model_pub)
        key = derive_key(eph, peer)
        try:
            gcm_nonce = os.urandom(GCM_NONCE_LEN)
            ct = AESGCM(bytes(key)).encrypt(gcm_nonce, plaintext.encode("utf-8"), None)
        finally:
            for i in range(len(key)):
                key[i] = 0
        encrypted_hex: str = bytes.fromhex(eph_pub_hex).hex() + gcm_nonce.hex() + ct.hex()
        return encrypted_hex
    except Exception as exc:  # noqa: BLE001 - fail closed on any crypto failure
        failure = _content_free_error("TEE message encryption failed", exc)
    raise failure


def decrypt_chunk(session_priv: EllipticCurvePrivateKey, content_hex: str) -> str:
    """Decrypt one streamed response chunk with the SESSION private key.

    The wire blob is ``server_ephemeral_pub(65) || gcm_nonce(12) || ciphertext+tag``;
    the SESSION private key + the server ephemeral public key derive the same
    AES-256-GCM key the server used.

    Fail-closed: any malformed blob, off-curve point, or GCM tag mismatch raises
    :class:`TeeEncryptionError`. Corrupt plaintext is never emitted.

    Raises:
        TeeEncryptionError: On any decode / key-agreement / authentication failure.
    """
    _require_crypto()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

    try:
        raw = bytes.fromhex(content_hex)
        if len(raw) < _CT_OFFSET + 16:  # need at least eph(65)+nonce(12)+tag(16)
            raise ValueError("encrypted chunk too short")
        server_eph_pub = public_key_from_hex(raw[:_GCM_NONCE_OFFSET].hex())
        gcm_nonce = raw[_GCM_NONCE_OFFSET:_CT_OFFSET]
        ct = raw[_CT_OFFSET:]
        key = derive_key(session_priv, server_eph_pub)
        try:
            plaintext: str = AESGCM(bytes(key)).decrypt(gcm_nonce, ct, None).decode("utf-8")
            return plaintext
        finally:
            for i in range(len(key)):
                key[i] = 0
    except Exception as exc:  # noqa: BLE001 - fail closed; never emit corrupt plaintext
        failure = _content_free_error("TEE response chunk decryption failed", exc)
    raise failure


# --- Heuristic ----------------------------------------------------------------


def looks_encrypted(s: str) -> bool:
    """Shape check: could ``s`` be an encrypted wire blob?

    True iff ``s`` is all-lowercase hex, even-length, and at least
    :data:`MIN_ENCRYPTED_HEX_LEN` (186) characters — the minimum for
    ``eph_pub(65) || nonce(12) || tag(16)`` = 93 bytes. This only gates
    decryption: a value that fails the check is rejected by callers, and one
    that passes is still authenticated by :func:`decrypt_chunk`.
    """
    if len(s) < MIN_ENCRYPTED_HEX_LEN or len(s) % 2 != 0:
        return False
    return all(c in "0123456789abcdef" for c in s)
