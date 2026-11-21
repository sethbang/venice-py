"""TEE E2EE session: the SESSION keypair + the three request headers.

A :class:`TeeSession` is the stateful client side of one Venice confidential
chat conversation. It is built from a **verified**
:class:`~venice_ai.tee.types.TeeAttestation` and owns:

* the **SESSION keypair** — generated once per session. Its *public* half goes
  in the ``X-Venice-TEE-Client-Pub-Key`` header on every request; its *private*
  half decrypts the streamed response (:meth:`decrypt_chunk`). **It is never
  used to encrypt a request message** — that is the load-bearing trap. Per-
  message encryption (:meth:`encrypt_message`) uses a *fresh* ephemeral keypair
  each call (see :func:`venice_ai.tee._crypto.encrypt_message`), so the 65-byte
  public prefix of an encrypted message differs from the session pub and from
  every other message's prefix.
* the **model public key** (``attestation.signing_public_key``) — the model's
  ECDH/encryption key, used as the peer key when encrypting request messages and
  echoed back to the server in the ``X-Venice-TEE-Model-Pub-Key`` header.
* the **signing algorithm** (``attestation.signing_algo``, always ``"ecdsa"``)
  for the ``X-Venice-TEE-Signing-Algo`` header.

``cryptography`` is an optional dependency (the ``[e2ee]`` extra). Building a
session generates the SESSION keypair, which requires it; the lazy import lives
in :mod:`venice_ai.tee._crypto` and raises a clear, actionable
:class:`ImportError` when the extra is absent. Baseline *attestation* does not
require it — only encryption does.

**Zeroization is best-effort and honest** (mirrors :mod:`_crypto`). On
:meth:`close` the reference to the SESSION private-key object is dropped; the
``EllipticCurvePrivateKey`` object exposes no wipeable buffer, so we do not claim
to wipe it from memory. Post-close use of :meth:`encrypt_message` /
:meth:`decrypt_chunk` / :meth:`request_headers` raises a clear error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import TeeAttestationError, TeeEncryptionError
from . import _crypto
from ._attestation import _check_key_binding
from ._constants import (
    HEADER_CLIENT_PUB_KEY,
    HEADER_MODEL_PUB_KEY,
    HEADER_SIGNING_ALGO,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import TracebackType

    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey

    from .types import TeeAttestation, TeeTrust

__all__ = ["TeeSession"]


class TeeSession:
    """One client-side Venice TEE E2EE session.

    Holds the SESSION keypair and the model encryption key from a verified
    attestation. Build via :meth:`venice_ai.resources.tee.Tee.open_session` (or
    directly with :meth:`from_attestation`); use as a context manager so secrets
    are dropped on exit::

        with client.tee.open_session(model="e2ee-...") as session:
            headers = session.request_headers()
            blob = session.encrypt_message("hello")
            # ... stream the response; for each encrypted delta:
            text = session.decrypt_chunk(delta_hex)

    .. warning::
        **Security limitation.** A :class:`TeeSession` is only as trustworthy as
        the attestation it is built from, and that attestation is **baseline-
        verified**: the verifier trusts Venice's server-side ``verified`` claim,
        checks the nonce / report-data binding and that the model key derives to
        the attested signing address, but does **not** perform full client-side
        Intel TDX / NVIDIA quote verification. A malicious Venice operator
        forging a self-consistent attestation would not be detected. Messages
        you encrypt with this session are confidential against a passive network
        observer, not against a hostile enclave operator. Supply a
        :class:`venice_ai.tee.FullQuoteVerifier` to
        :meth:`venice_ai.resources.tee.Tee.open_session` if your threat model
        requires it.
    """

    def __init__(
        self,
        *,
        session_private_key: EllipticCurvePrivateKey,
        model_public_key_hex: str,
        signing_algo: str,
        trust: TeeTrust | None = None,
    ) -> None:
        """Construct a session from an already-generated SESSION keypair.

        Prefer :meth:`from_attestation`; this initialiser is for callers that
        manage the keypair themselves (e.g. tests).

        Args:
            session_private_key: The SESSION ``secp256k1`` private key. Its
                public half is the ``X-Venice-TEE-Client-Pub-Key``; its private
                half decrypts responses.
            model_public_key_hex: The model's 130-hex uncompressed ECDH/signing
                public key (``attestation.signing_public_key``).
            signing_algo: The signing algorithm (``"ecdsa"``).
            trust: The attestation trust level this session was opened with.
        """
        self._trust = trust
        self._session_private_key: EllipticCurvePrivateKey | None = session_private_key
        self._session_public_key_hex = _crypto.uncompressed_hex(session_private_key.public_key())
        self._model_public_key_hex = model_public_key_hex
        self._signing_algo = signing_algo

    @classmethod
    def from_attestation(
        cls, attestation: TeeAttestation, *, trust: TeeTrust | None = None
    ) -> TeeSession:
        """Build a session by generating a fresh SESSION keypair for ``attestation``.

        The attestation must already be **verified** (the resource layer verifies
        fail-closed before calling this). The model key and signing algorithm are
        taken from the attestation.

        Whatever verification the caller ran, this method itself checks that
        ``attestation.signing_public_key`` derives to
        ``attestation.signing_address`` before any session key is generated, so a
        session never encrypts to a key the attested address does not bind. It
        does not repeat the other attestation checks (server claim, nonce,
        report-data binding, debug flag); those need the request context that
        :meth:`venice_ai.resources.tee.Tee.get_attestation` has.

        Args:
            attestation: A verified :class:`~venice_ai.tee.types.TeeAttestation`.
            trust: The trust level to carry on the session, if known.

        Returns:
            A new :class:`TeeSession`.

        Raises:
            TeeEncryptionError: If the attestation has no ``signing_public_key``.
            TeeAttestationError: If ``signing_public_key`` is malformed or does
                not derive to ``signing_address``. The message quotes no key
                material.
            ImportError: If the ``[e2ee]`` extra is not installed: ``pycryptodome``
                for the key-binding check, or ``cryptography`` for the SESSION
                keypair (both raised lazily).
        """
        if attestation.signing_public_key is None:
            raise TeeEncryptionError(
                "Cannot establish a TEE E2EE session: the attestation has no "
                "signing_public_key (model public key). This model does not "
                "publish a model key for end-to-end encryption."
            )
        key_failure = _check_key_binding(
            attestation.signing_public_key, attestation.signing_address
        )
        if key_failure is not None:
            raise TeeAttestationError(key_failure)
        session_priv = _crypto.generate_session_keypair()
        return cls(
            session_private_key=session_priv,
            model_public_key_hex=attestation.signing_public_key,
            signing_algo=attestation.signing_algo,
            trust=trust,
        )

    @property
    def trust(self) -> TeeTrust | None:
        """The attestation trust level this session was opened with.

        Holds no key material and stays readable after :meth:`close`.
        """
        return self._trust

    @property
    def session_public_key_hex(self) -> str:
        """The SESSION public key as 130-hex (the ``X-Venice-TEE-Client-Pub-Key``)."""
        return self._session_public_key_hex

    @property
    def model_public_key_hex(self) -> str:
        """The model's 130-hex ECDH/signing public key from the attestation."""
        return self._model_public_key_hex

    def _require_open(self) -> EllipticCurvePrivateKey:
        if self._session_private_key is None:
            raise TeeEncryptionError(
                "TeeSession is closed; its SESSION private key has been dropped. "
                "Open a new session via client.tee.open_session(...)."
            )
        return self._session_private_key

    def request_headers(self) -> dict[str, str]:
        """Build the three ``X-Venice-TEE-*`` request headers for this session.

        Returns a dict with:

        * ``X-Venice-TEE-Client-Pub-Key`` — the **SESSION** public key (130-hex).
          This is **not** a per-message ephemeral key; the same value is returned
          on every call for the life of the session, and its private half is what
          decrypts the response.
        * ``X-Venice-TEE-Model-Pub-Key`` — the model's ``signing_public_key``.
        * ``X-Venice-TEE-Signing-Algo`` — the signing algorithm (``"ecdsa"``).

        Raises:
            TeeEncryptionError: If the session has been closed.
        """
        self._require_open()
        return {
            HEADER_CLIENT_PUB_KEY: self._session_public_key_hex,
            HEADER_MODEL_PUB_KEY: self._model_public_key_hex,
            HEADER_SIGNING_ALGO: self._signing_algo,
        }

    def encrypt_message(self, plaintext: str) -> str:
        """Encrypt one request message to the model's public key.

        Delegates to :func:`venice_ai.tee._crypto.encrypt_message`, which
        generates a **fresh per-message ephemeral keypair** (never the SESSION
        key) and returns lowercase hex of
        ``ephemeral_pub(65) || gcm_nonce(12) || ciphertext+tag``.

        Args:
            plaintext: The UTF-8 text to encrypt: a user, system or assistant
                ``content``, or an assistant ``reasoning_content``.

        Returns:
            The encrypted wire blob as lowercase hex.

        Raises:
            TeeEncryptionError: If the session is closed or encryption fails.
        """
        self._require_open()
        return _crypto.encrypt_message(self._model_public_key_hex, plaintext)

    def decrypt_chunk(self, content_hex: str) -> str:
        """Decrypt one streamed response chunk with the SESSION private key.

        Delegates to :func:`venice_ai.tee._crypto.decrypt_chunk`. Fail-closed:
        any malformed blob or GCM tag mismatch raises.

        Args:
            content_hex: The encrypted ``choices[0].delta.content`` hex blob.

        Returns:
            The decrypted UTF-8 chunk text.

        Raises:
            TeeEncryptionError: If the session is closed or decryption fails.
        """
        priv = self._require_open()
        return _crypto.decrypt_chunk(priv, content_hex)

    def close(self) -> None:
        """Drop the SESSION private key reference (best-effort secret hygiene).

        After :meth:`close`, :meth:`request_headers`, :meth:`encrypt_message` and
        :meth:`decrypt_chunk` raise :class:`~venice_ai.exceptions.TeeEncryptionError`.

        The underlying ``EllipticCurvePrivateKey`` object exposes no wipeable
        buffer, so this drops the reference and lets it be garbage-collected; it
        does **not** zero the key bytes in memory (honest limitation, matching
        :mod:`venice_ai.tee._crypto`).
        """
        self._session_private_key = None

    def __enter__(self) -> TeeSession:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self.close()
