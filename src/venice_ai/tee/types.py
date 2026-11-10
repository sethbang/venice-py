"""Typed models for the Venice TEE (Trusted Execution Environment) surface.

NOTE: The ``GET /tee/attestation`` and ``GET /tee/signature`` endpoints have no
formal OpenAPI/swagger definition. The shapes modeled here are derived from live
probes of the API rather than a published schema, so additional provider-specific
keys may appear; ``extra='allow'`` preserves them on ``model_extra``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from ..core.models.base import VeniceBaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._attestation import FullQuoteVerifier


@dataclass(frozen=True, slots=True)
class TeeOptions:
    """Structured form of the ``e2ee=`` argument to ``chat.completions.create``.

    ``e2ee=True`` is the simple path; pass a :class:`TeeOptions` instead when you
    need to control the attestation freshness nonce or supply a full client-side
    quote verifier. Both fields are forwarded to
    :meth:`venice_ai.resources.tee.Tee.open_session`.

    Attributes:
        nonce: Optional 32-byte (64 lowercase hex) attestation freshness nonce.
            Generated automatically when ``None``.
        verifier: Optional :class:`~venice_ai.tee._attestation.FullQuoteVerifier`
            for full client-side Intel TDX / NVIDIA quote verification (the
            baseline ships none).
    """

    nonce: str | None = None
    verifier: FullQuoteVerifier | None = None


class TeeAttestation(VeniceBaseModel):
    """Parsed response from ``GET /tee/attestation``.

    Captures the fields the baseline verifier needs plus the retained raw
    attestation evidence (Intel TDX quote, NVIDIA payload, server-side
    verification report) for an optional :class:`FullQuoteVerifier` extension.

    The live response carries many provider-specific keys (``info``,
    ``event_log``, ``tee_provider``, ``tee_hardware``, ``upstream_model``,
    ``nonce_source``, ``candidates_*``, ...). We override the inherited
    ``extra="forbid"`` with ``extra="allow"`` so future / provider-specific
    fields are retained rather than rejected.

    Attributes:
        verified: The server's verification claim. The baseline verifier
            fails closed if this is ``False``.
        signing_public_key: 130-hex uncompressed model key. This IS the model's
            ECDH/encryption key (no separate enc key). May be ``None`` for plain
            TEE models that do not publish a model signing key.
        signing_address: 20-byte Ethereum-style address derived from the
            signing key, used in the report-data binding.
        signing_algo: Always ``"ecdsa"``.
        nonce: The nonce echoed by the server; the verifier checks it equals
            :attr:`sent_nonce`.
        sent_nonce: The nonce the client sent (NOT part of the wire response;
            populated by the verifier for the equality / binding check).
        intel_quote / quote: Raw Intel TDX quote hex (retained for full
            quote verification).
        nvidia_payload: Raw NVIDIA GPU attestation payload (retained).
        server_verification: The server's own verification report (retained).
        vm_config: Raw VM configuration blob (retained).
    """

    # extra="allow" overrides VeniceBaseModel's inherited extra="forbid".
    model_config = ConfigDict(extra="allow")

    verified: bool
    signing_public_key: str | None = None
    signing_address: str
    signing_algo: str
    nonce: str

    # Set by the verifier; absent from the wire response.
    sent_nonce: str | None = None

    # Retained raw attestation evidence — shapes vary by model/provider
    # (some are JSON strings, some are dicts), so type them loosely.
    intel_quote: Any | None = None
    nvidia_payload: Any | None = None
    quote: Any | None = None
    server_verification: Any | None = None
    vm_config: Any | None = None


class TeeReceiptEvent(VeniceBaseModel):
    """A single entry in a TEE receipt's ``event_log``.

    Fields vary by provider: some entries carry a ``seq`` and some do not, and
    event-specific fields such as ``target_route_id`` are kept in ``model_extra``.
    """

    model_config = ConfigDict(extra="allow")

    seq: int | None = None
    type: str
    body_hash: str | None = None


class TeeReceiptSignature(VeniceBaseModel):
    """The signature block on a TEE receipt (``ecdsa-secp256k1`` over the receipt)."""

    model_config = ConfigDict(extra="allow")

    algo: str
    key_id: str
    value: str


class TeeReceipt(VeniceBaseModel):
    """Signed attestation receipt binding a specific request to the attested workload."""

    model_config = ConfigDict(extra="allow")

    api_version: str | None = None
    receipt_id: str | None = None
    chat_id: str | None = None
    workload_id: str | None = None
    workload_keyset_digest: str | None = None
    endpoint: str | None = None
    method: str | None = None
    served_at: int | None = None
    event_log: list[TeeReceiptEvent] = Field(default_factory=list)
    #: A structured block from some providers, a bare hex string from others.
    signature: TeeReceiptSignature | str | None = None


class TeeSignatureVerification(VeniceBaseModel):
    """Pointer to the attestation endpoint for verifying the signature's chain of trust."""

    model_config = ConfigDict(extra="allow")

    attestation_endpoint: str | None = None
    description: str | None = None


class TeeSignatureResponse(VeniceBaseModel):
    """Parsed response from ``GET /tee/signature``.

    Proves a specific completion (:attr:`requested_request_id`) was produced by
    the attested enclave: :attr:`signature` is over :attr:`text` by
    :attr:`signing_address` (verify against the model's attestation), and the
    signed :attr:`receipt` records the request's event log. ``extra="allow"``
    so provider-specific / future fields are retained, not rejected.
    """

    model_config = ConfigDict(extra="allow")

    text: str
    signature: str
    signing_address: str
    api_version: str | None = None
    signing_algo: str | None = None
    receipt: TeeReceipt | None = None
    model: str | None = None
    requested_request_id: str | None = None
    tee_provider: str | None = None
    upstream_model: str | None = None
    verification: TeeSignatureVerification | None = None


class TeeVerificationResult(VeniceBaseModel):
    """Outcome of baseline attestation verification.

    Returned by :func:`venice_ai.tee._attestation.verify_attestation`. When
    ``fail_closed=True`` (the default) any failed check raises
    :class:`~venice_ai.exceptions.TeeAttestationError` instead of returning a
    result with :attr:`ok` ``False``; when ``fail_closed=False`` the failures are
    collected here and a :class:`UserWarning` is emitted.

    Attributes:
        ok: ``True`` only if every baseline check passed.
        checks: Per-check pass/fail map (e.g. ``"verified"``, ``"nonce_echo"``,
            ``"reportdata_binding"``, ``"debug_flag"``, ``"full_quote"``).
        failures: Human-readable messages for the checks that did not pass.
        signing_address: The lowercase, ``0x``-stripped signing address that the
            report-data binding was checked against.
        model_public_key: The model's 65-byte uncompressed ECDH/signing public
            key (the bytes form of ``signing_public_key``), or ``None`` if it
            could not be decoded.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ok: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    signing_address: str | None = None
    model_public_key: bytes | None = None
