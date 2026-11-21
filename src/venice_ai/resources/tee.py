"""Venice TEE (Trusted Execution Environment) attestation + session resource.

Wraps ``GET /tee/attestation`` and bootstraps a client-side E2EE
:class:`~venice_ai.tee._session.TeeSession` for the confidential ``e2ee-*`` chat
path. Attached as ``client.tee`` on both the async :class:`~venice_ai._client.VeniceClient`
and (via the sync proxy) the :class:`~venice_ai._sync_client.SyncVeniceClient`.

Two entry points:

* :meth:`Tee.get_attestation` — fetch and **baseline-verify** an attestation
  (fail-closed). An attestation that publishes a model key is also checked for
  key binding, which needs Keccak-256 from ``pycryptodome`` (the ``[e2ee]``
  extra); an attestation without a key verifies on a bare install.
* :meth:`Tee.open_session` — :meth:`get_attestation` then build a
  :class:`TeeSession` (generates the SESSION keypair). The keypair generation
  needs ``cryptography``; the lazy import + actionable ``ImportError`` (the
  ``pip install 'venice-py[e2ee]'`` hint) live in :mod:`venice_ai.tee._crypto`.

.. warning::
    Baseline attestation trusts Venice's server-side ``verified`` claim and does
    not perform full client-side Intel TDX / NVIDIA quote verification on its own.
    For full offline Intel TDX verification, pass a
    :class:`venice_ai.tee.DcapTdxVerifier` (the ``[e2ee-verify]`` extra) via
    ``verifier=``. See :func:`venice_ai.tee._attestation.verify_attestation` for
    the limitation and the :class:`FullQuoteVerifier` extension point.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .._resource import APIResource
from ..tee._attestation import (
    NONCE_LEN_BYTES,
    FullQuoteVerifier,
    preflight_nonce,
    verify_attestation,
)
from ..tee._session import TeeSession
from ..tee.types import TeeAttestation, TeeSignatureResponse, TeeTrust

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

__all__ = ["Tee"]


def _generate_nonce() -> str:
    """Generate a fresh 32-byte (64-hex) attestation nonce."""
    return os.urandom(NONCE_LEN_BYTES).hex()


class Tee(APIResource["VeniceClient"]):
    """Venice confidential-compute (TEE) attestation + E2EE session bootstrap.

    Example::

        async with VeniceClient() as client:
            # Verify the enclave and open an encrypted session in one step.
            session = await client.tee.open_session(model="e2ee-gemma-3-27b-p")
            with session:
                headers = session.request_headers()
                blob = session.encrypt_message("Hello, confidential world.")
                # ... POST /chat/completions (stream=True) with `headers` and
                # the encrypted message content; decrypt each streamed delta:
                #     text = session.decrypt_chunk(delta_hex)
    """

    async def get_attestation(
        self,
        *,
        model: str,
        nonce: str | None = None,
        fail_closed: bool = True,
        verifier: FullQuoteVerifier | None = None,
    ) -> TeeAttestation:
        """Fetch and baseline-verify a TEE attestation for ``model``.

        Calls ``GET /tee/attestation?model=&nonce=`` (a free endpoint), verifies
        the response fail-closed against the client-generated nonce (server
        ``verified`` claim, nonce echo, report-data binding, TDX debug-flag, and
        that ``signing_public_key`` derives to ``signing_address``), and records
        the sent nonce on the returned attestation.

        REPORTDATA binds the signing address and the nonce, not the model key, so
        the key-binding check is what ties the encryption key to the attested
        enclave. It needs Keccak-256 from ``pycryptodome`` (the ``[e2ee]`` extra)
        and runs whenever the attestation carries a key.

        Args:
            model: The ``e2ee-*`` model id to attest (e.g. ``"e2ee-gemma-3-27b-p"``).
            nonce: Optional 32-byte (64 lowercase hex) freshness nonce. When
                omitted, a cryptographically random one is generated. The Venice
                API rejects nonces that are not exactly 32 bytes.
            fail_closed: When ``True`` (default), any failed verification check
                raises :class:`~venice_ai.exceptions.TeeAttestationError`. When
                ``False``, failures are collected and a :class:`UserWarning` is
                emitted instead (the attestation is still returned).
            verifier: Optional :class:`FullQuoteVerifier` for full client-side
                quote verification (baseline ships none).

        Returns:
            A verified :class:`~venice_ai.tee.types.TeeAttestation` with
            ``sent_nonce`` populated.

        Raises:
            TeeAttestationError: If ``fail_closed`` and verification fails (or the
                supplied ``nonce`` is malformed), including when
                ``signing_public_key`` does not derive to ``signing_address``.
            ImportError: If the attestation carries a model key and
                ``pycryptodome`` (the ``[e2ee]`` extra) is not installed.
            APIError: For HTTP-level failures.
        """
        sent_nonce = preflight_nonce(nonce) if nonce is not None else _generate_nonce()
        attestation = await self._client.get(
            "tee/attestation",
            params={"model": model, "nonce": sent_nonce},
            cast_to=TeeAttestation,
        )
        # Populate sent_nonce BEFORE verification so a FullQuoteVerifier reading
        # attestation.sent_nonce (REPORTDATA binding) sees the real nonce
        # rather than None. Baseline equality already uses the local sent_nonce.
        attestation.sent_nonce = sent_nonce
        verify_attestation(
            attestation,
            sent_nonce,
            fail_closed=fail_closed,
            verifier=verifier,
        )
        return attestation

    async def get_signature(self, *, model: str, request_id: str) -> TeeSignatureResponse:
        """Fetch the TEE signature proving a completion was produced by the enclave.

        Calls ``GET /tee/signature?model=&request_id=``. The returned
        :class:`~venice_ai.tee.types.TeeSignatureResponse` carries the
        ``signature`` over ``text`` by ``signing_address`` (verify against the
        model's :meth:`get_attestation`) plus the signed attestation ``receipt``.

        :param model: The TEE model id the completion ran on (an ``e2ee-*`` /
            TEE-attestation-capable model).
        :param request_id: The completion id to attest (the chat completion ``id``).
        :return: The parsed :class:`~venice_ai.tee.types.TeeSignatureResponse`.
        """
        signature: TeeSignatureResponse = await self._client.get(
            "tee/signature",
            params={"model": model, "request_id": request_id},
            cast_to=TeeSignatureResponse,
        )
        return signature

    async def open_session(
        self,
        *,
        model: str,
        nonce: str | None = None,
        verifier: FullQuoteVerifier | None = None,
    ) -> TeeSession:
        """Verify ``model``'s attestation (fail-closed) and open an E2EE session.

        Convenience over :meth:`get_attestation` + :meth:`TeeSession.from_attestation`.
        Always verifies fail-closed (an unverified enclave must never back an
        encrypting session). Generating the SESSION keypair requires the
        ``[e2ee]`` extra; if ``cryptography`` is missing, the lazy import in
        :mod:`venice_ai.tee._crypto` raises an :class:`ImportError` with the
        ``pip install 'venice-py[e2ee]'`` hint.

        Args:
            model: The ``e2ee-*`` model id.
            nonce: Optional 32-byte (64-hex) nonce; generated when omitted.
            verifier: Optional :class:`FullQuoteVerifier`.

        Returns:
            A :class:`~venice_ai.tee._session.TeeSession` ready to produce request
            headers and encrypt/decrypt messages. Use it as a context manager so
            its SESSION private key is dropped on exit.

        Raises:
            TeeAttestationError: If verification fails (always fail-closed here),
                including when ``signing_public_key`` does not derive to the
                attested ``signing_address``.
            TeeEncryptionError: If the verified attestation publishes no model
                public key (``signing_public_key``) to encrypt to.
            ImportError: If the ``[e2ee]`` extra is not installed.
            APIError: For HTTP-level failures.
        """
        attestation = await self.get_attestation(
            model=model,
            nonce=nonce,
            fail_closed=True,
            verifier=verifier,
        )
        trust = TeeTrust.from_attestation(attestation, model=model, verifier=verifier)
        return TeeSession.from_attestation(attestation, trust=trust)
