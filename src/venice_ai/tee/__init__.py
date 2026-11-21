"""Venice TEE (Trusted Execution Environment) end-to-end encryption.

This package implements client-side support for Venice's confidential-compute
E2EE chat path (``e2ee-*`` models): attestation verification and per-message
encryption/decryption over secp256k1 ECDH + HKDF-SHA256 + AES-256-GCM.

``cryptography`` and ``pycryptodome`` are **optional** dependencies (the
``[e2ee]`` extra); this package imports cleanly on a bare install and only
requires them lazily: ``cryptography`` when an encryption path is used (see
:func:`_require_crypto`), ``pycryptodome`` when an attestation that carries a
model key is verified (its Keccak-256 key-binding check).

.. warning::
    The baseline attestation verifier trusts Venice's server-side ``verified``
    claim. Full client-side TDX / NVIDIA quote verification is NOT performed by
    the baseline. For full **offline** Intel TDX verification, supply a
    :class:`DcapTdxVerifier` (the ``[e2ee-verify]`` extra) via the
    :class:`FullQuoteVerifier` extension point; NVIDIA GPU attestation is not yet
    shipped.

Public symbols are re-exported here: the typed TEE exceptions
(:class:`TeeError`, :class:`TeeAttestationError`, :class:`TeeEncryptionError`,
:class:`TeeTcbStatusError`, :class:`TeeCollateralMismatchError`,
:class:`TeeCollateralExpiredError`, :class:`TeeCollateralUnavailableError`),
:class:`CollateralKey`, the attestation and quote verifiers (:class:`FullQuoteVerifier`,
:class:`DcapTdxVerifier`), the encrypted session (:class:`TeeSession`),
:class:`TeeOptions`, :class:`TeeTrust`, and the :func:`_require_crypto` helper.
"""

from __future__ import annotations

from ..exceptions import (
    TeeAttestationError,
    TeeCollateralExpiredError,
    TeeCollateralMismatchError,
    TeeCollateralUnavailableError,
    TeeEncryptionError,
    TeeError,
    TeeTcbStatusError,
)
from ._attestation import FullQuoteVerifier
from ._crypto import _require_crypto
from ._session import TeeSession
from ._verify import DcapTdxVerifier
from .types import CollateralKey, TeeOptions, TeeTrust

__all__ = [
    "TeeError",
    "TeeAttestationError",
    "TeeEncryptionError",
    "TeeTcbStatusError",
    "TeeCollateralMismatchError",
    "TeeCollateralExpiredError",
    "TeeCollateralUnavailableError",
    "CollateralKey",
    "FullQuoteVerifier",
    "DcapTdxVerifier",
    "TeeOptions",
    "TeeTrust",
    "TeeSession",
    "_require_crypto",
]
