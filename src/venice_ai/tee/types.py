"""Typed models for the Venice TEE (Trusted Execution Environment) surface.

NOTE: The ``GET /tee/attestation`` and ``GET /tee/signature`` endpoints have no
formal OpenAPI/swagger definition. The shapes modeled here are derived from live
probes of the API rather than a published schema, so additional provider-specific
keys may appear; ``extra='allow'`` preserves them on ``model_extra``.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field

from ..core.models.base import VeniceBaseModel
from ..exceptions import TeeAttestationError, TeeEncryptionError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._attestation import FullQuoteVerifier

#: One PEM certificate's base64 body.
_PEM_CERTIFICATE = re.compile(
    r"-----BEGIN CERTIFICATE-----(.+?)-----END CERTIFICATE-----", re.DOTALL
)


def _key_from_pem_chain(
    chain_pem: str | bytes, fmspc: str, *, ca_index: int
) -> CollateralKey | None:
    """The one rule that turns a PEM chain and an FMSPC into a :class:`CollateralKey`.

    The PCK CA is the certificate at ``ca_index`` of the chain (index 1 of a quote's
    PCK chain, index 0 of a collateral's CRL issuer chain). Returns ``None`` when the
    chain has no certificate there.
    """
    text = chain_pem.decode("ascii", "replace") if isinstance(chain_pem, bytes) else chain_pem
    certificates = _PEM_CERTIFICATE.findall(text)
    if len(certificates) <= ca_index:
        return None
    der = base64.b64decode("".join(certificates[ca_index].split()), validate=True)
    return CollateralKey(fmspc=fmspc.upper(), pck_ca_sha256=hashlib.sha256(der).hexdigest())


@dataclass(frozen=True, slots=True)
class CollateralKey:
    """Identifies which Intel collateral a quote needs: its FMSPC and PCK CA.

    Collateral (TCB info, QE identity, CRLs) is per platform family (``fmspc``) and
    per PCK CA, so this pair is the natural cache key for an application that
    stores collateral. Build it with :meth:`from_quote`; that is the only rule the
    verifier uses too, so a key made by the application always matches.

    Attributes:
        fmspc: Upper-case hex FMSPC, e.g. ``"90C06F000000"``.
        pck_ca_sha256: Lower-case hex SHA-256 of the PCK CA certificate's DER.
    """

    fmspc: str
    pck_ca_sha256: str

    @classmethod
    def from_quote(cls, raw_quote: bytes | str) -> CollateralKey:
        """Compute the key for a raw TDX quote (bytes, or a hex string).

        Raises:
            TeeError: ``dcap-qvl`` is not installed (the ``[e2ee-verify]`` extra).
            TeeAttestationError: The quote cannot be parsed or embeds no PCK chain.
        """
        from ._verify import _require_dcap  # noqa: PLC0415  (optional dep; avoids a cycle)

        dcap = _require_dcap()
        try:
            raw = bytes.fromhex(raw_quote) if isinstance(raw_quote, str) else bytes(raw_quote)
            quote = dcap.parse_quote(raw)
            chain = quote.cert_chain_pem_bytes()
            if not chain:
                raise TeeAttestationError(
                    "TEE quote embeds no PCK certificate chain (fail-closed)."
                )
            key = _key_from_pem_chain(chain, quote.fmspc(), ca_index=1)
        except (ValueError, TypeError) as exc:
            raise TeeAttestationError(f"TEE quote could not be read: {exc}") from exc
        if key is None:
            raise TeeAttestationError("TEE quote embeds no PCK CA certificate (fail-closed).")
        return key


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
        verifier: Optional :class:`venice_ai.tee.FullQuoteVerifier` for full
            client-side Intel TDX / NVIDIA quote verification (the baseline
            ships none).
    """

    nonce: str | None = None
    verifier: FullQuoteVerifier | None = None


@dataclass(frozen=True, slots=True)
class TeeTrust:
    """How much the attestation behind an E2EE session was trusted.

    Read it from ``ChatStream.e2ee`` or ``ChatCompletionResponse.e2ee`` to see what
    was verified for an E2EE request. It holds only public attestation fields, no
    key material.

    Attributes:
        model: The model the session was opened for.
        verification: ``"baseline"`` or ``"full_quote"``. ``"baseline"`` trusts
            Venice's server-side ``verified`` claim: the SDK checks the nonce,
            the report-data binding (REPORTDATA binds the signing address and
            the nonce) and that ``model_public_key`` derives to
            ``signing_address``, but does NOT verify the Intel TDX or NVIDIA
            quote client-side. ``"full_quote"`` means a
            :class:`venice_ai.tee.FullQuoteVerifier` you supplied also accepted
            the attestation; the key-binding check runs on both levels.
        verifier: The class name of the supplied verifier, or ``None`` for
            ``"baseline"``.
        server_verified: The attestation's ``verified`` claim from Venice.
        nonce: The nonce the attestation echoed (checked against the nonce sent).
        model_public_key: The model's 130-hex public key that requests are
            encrypted to. When this trust level comes from
            :meth:`venice_ai.resources.tee.Tee.get_attestation` or
            :meth:`venice_ai.resources.tee.Tee.open_session`, verification
            checked that the key derives to :attr:`signing_address`. A
            :class:`TeeTrust` built by hand, or from an attestation that was not
            verified, carries no such guarantee.
        signing_address: The attested 20-byte signing address: bound into
            REPORTDATA with the nonce, and the Keccak-256 address of
            :attr:`model_public_key`.
    """

    model: str
    verification: Literal["baseline", "full_quote"]
    verifier: str | None
    server_verified: bool
    nonce: str
    model_public_key: str
    signing_address: str

    @classmethod
    def from_attestation(
        cls,
        attestation: TeeAttestation,
        *,
        model: str,
        verifier: FullQuoteVerifier | None,
    ) -> TeeTrust:
        """Build the trust level from a verified attestation.

        Args:
            attestation: The verified attestation.
            model: The model id the session was opened for.
            verifier: The full-quote verifier used, or ``None`` for baseline.

        Returns:
            A frozen :class:`TeeTrust`.

        Raises:
            TeeEncryptionError: If the attestation has no ``signing_public_key``.
        """
        if attestation.signing_public_key is None:
            raise TeeEncryptionError(
                "Cannot establish a TEE E2EE session: the attestation has no "
                "signing_public_key (model public key). This model does not "
                "publish a model key for end-to-end encryption."
            )
        return cls(
            model=model,
            verification="baseline" if verifier is None else "full_quote",
            verifier=None if verifier is None else type(verifier).__name__,
            server_verified=attestation.verified,
            nonce=attestation.nonce,
            model_public_key=attestation.signing_public_key,
            signing_address=attestation.signing_address,
        )


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
            ``"reportdata_binding"``, ``"debug_flag"``, ``"key_binding"``,
            ``"full_quote"``). ``"key_binding"`` is absent when the attestation
            carries no ``signing_public_key``.
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
