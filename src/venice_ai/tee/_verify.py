"""Full client-side Intel TDX quote verification (``[e2ee-verify]`` extra).

The baseline verifier (:mod:`venice_ai.tee._attestation`) trusts Venice's
server-side ``verified`` claim. :class:`DcapTdxVerifier` closes that gap: it
verifies the raw Intel TDX quote's ECDSA signature and PCK certificate chain to
a **pinned** Intel SGX Root CA, evaluates the FMSPC TCB status against
Intel-signed collateral, confirms the enclave is non-debug, that the signing
address and nonce are bound into REPORTDATA, that the event log replays to the
quoted RTMRs, and that the dstack ``app_compose`` hashes into the quoted
``mr_config_id`` — all OFFLINE.

REPORTDATA does not contain the E2EE encryption key itself. The key is tied to
the enclave by :func:`venice_ai.tee._attestation.verify_attestation`, which
checks that ``signing_public_key`` derives (Keccak-256) to ``signing_address``
before any full-quote verifier runs; this verifier binds the same top-level
``signing_address`` into REPORTDATA.

**Security tier.** By default this proves the model runs on a *genuine,
non-debug Intel TDX enclave* running a *self-consistent dstack workload* (Tier
B). It does **not** independently confirm this is the legitimate Venice
image/app unless the caller supplies ``expected_measurements`` /
``expected_compose_hash`` from a source independent of the Venice endpoint
(per-dimension Tier A).

**Wire schemas.** Both the original dstack layout and the current
``attestation.evidence`` layout are accepted (see
:mod:`venice_ai.tee._evidence`). The current wire carries neither
``app_compose`` nor ``compose_hash``, so compose identity cannot be
established from the attestation alone; ``last_result["checks"]
["compose_binding"]`` is then the string ``"unavailable"`` rather than
``True``. Tier B on that wire therefore proves genuine non-debug TDX
hardware, a signing address bound into the enclave (and, through the key
binding, the encryption key that derives to it), and an event log consistent
with the quoted RTMRs — but not that the workload matches a given compose.
Pin ``expected_measurements["mr_config_id"]`` for that dimension. Because a
check value may be a string, test results with ``is True``, never for
truthiness.

**The #1 correctness gate.** ``dcap_qvl.parse_quote`` performs **no** signature
check, so every policy byte (debug bit, REPORTDATA, RTMRs, ``mr_config_id``) is
trustworthy ONLY after ``verify_with_root_ca`` has PASSED for the same raw
quote. :meth:`DcapTdxVerifier.verify` enforces that ordering: the crypto gate
runs first and any failure aborts before a single parsed byte is consulted.

``dcap-qvl`` is an **optional** dependency (the ``[e2ee-verify]`` extra). This
module imports cleanly on a bare install; the import happens lazily inside
:func:`_require_dcap`, which raises a clear :class:`TeeError` with an install
hint when the extra is absent. **It never silently skips** — a silent skip would
degrade to baseline trust and defeat the entire feature (fail-closed).
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import UTC, datetime
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from ..exceptions import (
    TeeAttestationError,
    TeeCollateralExpiredError,
    TeeCollateralMismatchError,
    TeeCollateralUnavailableError,
    TeeError,
    TeeTcbStatusError,
)
from ._constants import INTEL_SGX_ROOT_CA_DER, PHALA_PCCS_URL
from ._evidence import SCHEMA_EVIDENCE, NormalizedEvidence, normalize
from .types import CollateralKey, _key_from_pem_chain

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from .types import TeeAttestation

_INSTALL_HINT = (
    "Venice full client-side TDX quote verification requires the ``dcap-qvl`` "
    "package, which is an optional dependency. Install it with: "
    "pip install 'venice-py[e2ee-verify]'"
)

#: REPORTDATA layout = signing_address(20B) || 12 zero bytes || sent_nonce(32B).
_ADDR_LEN = 20
_ZERO_PAD = b"\x00" * 12
_NONCE_LEN = 32
_REPORTDATA_LEN = _ADDR_LEN + 12 + _NONCE_LEN

#: Bit 0 (little-endian) of TD_ATTRIBUTES is ``TUD.DEBUG``.
_TDX_DEBUG_BIT = 0x1

#: ``mr_config_id`` embeds the dstack compose hash as ``0x01 || sha256 || pad``.
_COMPOSE_PREFIX = b"\x01"


def _lenient_key(chain_pem: str | bytes, fmspc: str, *, ca_index: int) -> CollateralKey:
    """Like :func:`_key_from_pem_chain`, but an unusable chain gives an empty fingerprint.

    The FMSPC comparison must still be able to refuse when a chain is corrupt; an
    empty fingerprint never matches, so the PCK CA comparison refuses too.
    """
    try:
        key = _key_from_pem_chain(chain_pem, fmspc, ca_index=ca_index)
    except ValueError:
        key = None
    return key or CollateralKey(fmspc=fmspc.upper(), pck_ca_sha256="")


def _collateral_key_from_quote(quote: Any) -> CollateralKey | None:
    """The key for an already-parsed quote, or ``None`` when it embeds no PCK chain."""
    embedded = getattr(quote, "cert_chain_pem_bytes", lambda: None)()
    if not embedded:
        return None
    return _key_from_pem_chain(embedded, quote.fmspc(), ca_index=1)


def _require_dcap() -> ModuleType:
    """Import and return the ``dcap_qvl`` module, lazily.

    Returns:
        The imported ``dcap_qvl`` module.

    Raises:
        TeeError: If ``dcap-qvl`` is not installed, with the install hint for
            the ``[e2ee-verify]`` extra. **Never** silently skips.
    """
    try:
        import dcap_qvl  # noqa: PLC0415  (lazy by design — optional dep)
    except ImportError as exc:
        raise TeeError(_INSTALL_HINT) from exc
    return cast(ModuleType, dcap_qvl)


# Bound at import only if the extra is installed; ``verify`` re-checks via
# ``_require_dcap`` so the install-hint path is exercised even when this is None.
# Tests monkeypatch ``verify_mod.dcap_qvl``; ``verify`` calls ``_require_dcap``,
# which returns this same module object, so the patched attributes take effect.
def _maybe_import_dcap() -> Any:
    try:
        import dcap_qvl  # noqa: PLC0415

        return dcap_qvl
    except ImportError:  # pragma: no cover - bare-install path
        return None


dcap_qvl: Any = _maybe_import_dcap()


class DcapTdxVerifier:
    """Full client-side Intel TDX quote verifier (:class:`FullQuoteVerifier`).

    Holds collateral / pinned root / measurement policy as constructor state and
    reads the evidence off the attestation. :meth:`verify` is **synchronous and
    offline** (it never touches the network); collateral is *injected*, either
    directly, via the async :meth:`with_fetched_collateral` factory, or per quote
    through ``collateral_provider``.

    **Reusing one verifier.** Collateral describes a platform type (its FMSPC) and
    a PCK CA, not one machine, so a verifier can check quotes from every machine
    of that type, including a re-attestation that lands on another machine. Each
    quote is checked against the PCK certificate chain it embeds, validated to the
    pinned root; a chain attached to the collateral belongs to the machine it was
    fetched for and is not used. With fixed ``collateral``, a quote from another
    FMSPC or PCK CA raises :class:`~venice_ai.exceptions.TeeCollateralMismatchError`,
    and collateral past its validity window is refused too, which limits how long
    one verifier lasts.

    **Collateral per quote.** To serve quotes from several FMSPCs or PCK CAs, or to
    refresh collateral without rebuilding the verifier, pass ``collateral_provider``
    instead of ``collateral``. :meth:`verify` calls it as ``provider(key, now_secs)``
    with the quote's :class:`~venice_ai.tee.types.CollateralKey` and the epoch
    seconds it checks validity windows against, and verifies with what it returns.
    The returned collateral goes through the same FMSPC and PCK CA checks as fixed
    collateral, and the crypto gate still decides. A provider must be synchronous
    and must not block on the network: :meth:`verify` never fetches. Return
    ``None`` for a cache miss; :meth:`verify` then raises
    :class:`~venice_ai.exceptions.TeeCollateralUnavailableError`, which carries the
    key and the raw quote so the application can fetch collateral asynchronously
    and retry. Any exception the provider raises propagates unchanged. Decide
    expiry from the ``now_secs`` the provider receives, not ``time.time()``::

        import json
        from datetime import datetime

        import dcap_qvl
        from cryptography import x509
        from venice_ai.exceptions import (
            TeeCollateralExpiredError,
            TeeCollateralUnavailableError,
        )
        from venice_ai.tee import CollateralKey, DcapTdxVerifier

        PCCS_URL = "https://pccs.phala.network"  # the with_fetched_collateral default

        cache: dict[CollateralKey, tuple[object, int]] = {}

        def expires_at(collateral) -> int:
            def stamp(text: str) -> int:
                return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())

            crls = (collateral.pck_crl, collateral.root_ca_crl)
            return min(
                stamp(json.loads(collateral.tcb_info)["nextUpdate"]),
                stamp(json.loads(collateral.qe_identity)["nextUpdate"]),
                *(int(x509.load_der_x509_crl(crl).next_update_utc.timestamp()) for crl in crls),
            )

        def provider(key: CollateralKey, now_secs: int) -> object | None:
            entry = cache.get(key)
            if entry is None or entry[1] <= now_secs:
                return None
            return entry[0]

        verifier = DcapTdxVerifier(collateral_provider=provider)

        async def verify(attestation) -> bool:
            try:
                return verifier.verify(attestation)
            except (TeeCollateralUnavailableError, TeeCollateralExpiredError) as err:
                collateral = await dcap_qvl.get_collateral(PCCS_URL, err.raw_quote)
                cache[err.key] = (collateral, expires_at(collateral))
                return verifier.verify(attestation)

    ``expires_at`` keeps collateral until the earliest of its TCB info, QE identity,
    PCK CRL and root CA CRL is next updated; :meth:`verify` still checks every
    validity window itself, and raises
    :class:`~venice_ai.exceptions.TeeCollateralExpiredError` for an entry the
    provider kept too long. With a provider, ``err.key`` is always set.
    ``err.key`` equals ``CollateralKey.from_quote(err.raw_quote)``, so either can
    key the store.

    **Checks may not be bool.** The ``"checks"`` mapping in :attr:`last_result`
    usually holds ``bool`` values, but ``checks["compose_binding"]`` may instead
    be the string ``"unavailable"`` when the wire cannot establish compose
    identity (see :meth:`verify`). Compare with ``is True``, never for
    truthiness — the string ``"unavailable"`` is truthy in Python.

    Args:
        collateral: An Intel-signed :class:`dcap_qvl.QuoteCollateralV3` (TCB info
            / QE identity / CRLs) for the quote's FMSPC. ``None`` means no
            collateral is available and :meth:`verify` will **reject** — it never
            downgrades to baseline trust. Mutually exclusive with
            ``collateral_provider``.
        collateral_provider: A synchronous callable ``(key, now_secs)`` returning
            the collateral for a quote's :class:`~venice_ai.tee.types.CollateralKey`,
            or ``None`` when it has none. See *Collateral per quote* above.
        root_ca_der: The pinned Intel SGX Root CA in DER. Defaults to the baked
            :data:`~venice_ai.tee._constants.INTEL_SGX_ROOT_CA_DER`; override for
            the (rare) Intel root-rotation path.
        now_secs: Wall-clock epoch seconds for collateral validity-window checks.
            Defaults to the real current time; tests inject a frozen value. TCB
            info has a validity window, so this MUST be real in production.
        expected_measurements: Optional dict of hex reference values keyed by
            ``mrtd`` / ``rtmr0..3`` / ``mr_config_id``. Each supplied value is
            compared against the verified quote field (per-dimension Tier A).
        expected_compose_hash: Optional hex dstack compose hash; compared against
            the verified ``mr_config_id`` embedding (per-dimension Tier A).
        tcb_policy: ``"reject"`` (default) rejects any non-``UpToDate`` status;
            ``"advisory"`` additionally accepts ``SWHardeningNeeded`` /
            ``ConfigurationAndSWHardeningNeeded`` but surfaces ``advisory_ids``
            and emits a :class:`UserWarning`. ``OutOfDate`` / ``Revoked`` /
            ``ConfigurationNeeded`` are rejected under both policies.
    """

    #: The ONLY status accepted unconditionally — the single load-bearing enum
    #: string, pinned against real collateral in the tests. The policy is
    #: accept-list (not reject-list) by design: an unknown / renamed / future
    #: status fails closed (over-rejects), never silently passes.
    ACCEPT_TCB_STATUS = "UpToDate"

    #: Documentary only — the known non-``UpToDate`` statuses dcap-qvl can emit.
    #: NOT consulted for the decision (the accept-list above is); listed so the
    #: docstring/policy reasoning is auditable. Do not key logic off this tuple.
    REJECT_TCB_STATUSES = (
        "OutOfDate",
        "Revoked",
        "ConfigurationNeeded",
        "SWHardeningNeeded",
        "ConfigurationAndSWHardeningNeeded",
        "OutOfDateConfigurationNeeded",
    )

    #: Statuses the ``"advisory"`` policy tolerates (with advisory surfacing).
    _ADVISORY_TOLERATED = ("SWHardeningNeeded", "ConfigurationAndSWHardeningNeeded")

    def __init__(
        self,
        *,
        collateral: Any | None = None,
        collateral_provider: Callable[[CollateralKey, int], Any | None] | None = None,
        root_ca_der: bytes = INTEL_SGX_ROOT_CA_DER,
        now_secs: int | None = None,
        expected_measurements: dict[str, str] | None = None,
        expected_compose_hash: str | None = None,
        tcb_policy: str = "reject",
    ) -> None:
        if tcb_policy not in ("reject", "advisory"):
            raise TeeError(f"tcb_policy must be 'reject' or 'advisory', got {tcb_policy!r}.")
        if collateral is not None and collateral_provider is not None:
            raise TeeError("Pass either collateral or collateral_provider, not both.")
        self._collateral = collateral
        self._collateral_provider = collateral_provider
        self._root_ca_der = root_ca_der
        self._now_secs = now_secs
        self._expected_measurements = {
            k.lower(): v.lower() for k, v in (expected_measurements or {}).items()
        }
        self._expected_compose_hash = (
            expected_compose_hash.lower() if expected_compose_hash else None
        )
        self._tcb_policy = tcb_policy
        #: Populated by the most recent :meth:`verify` call (structured result).
        self.last_result: dict[str, Any] | None = None

    @classmethod
    async def with_fetched_collateral(
        cls,
        probe_quote: str | bytes,
        *,
        pccs_url: str = PHALA_PCCS_URL,
        **kwargs: Any,
    ) -> DcapTdxVerifier:
        """Fetch collateral via the (no-auth) PCCS, then build a sync verifier.

        This is the **only** network touch in the class; :meth:`verify` itself
        stays offline. Airgapped callers should construct directly with a
        ``QuoteCollateralV3.from_json(...)`` snapshot instead.

        Args:
            probe_quote: A raw TDX quote (hex string or bytes) whose FMSPC drives
                the collateral fetch.
            pccs_url: The no-auth PCCS base URL (defaults to Phala's mirror).
            **kwargs: Forwarded to the constructor (minus ``collateral``).

        Returns:
            A :class:`DcapTdxVerifier` with freshly fetched collateral.

        Raises:
            TeeError: If ``collateral_provider`` is in ``kwargs``; a verifier takes
                fixed collateral or a provider, not both.
        """
        if "collateral_provider" in kwargs:
            raise TeeError(
                "with_fetched_collateral builds a verifier with fixed collateral; "
                "construct DcapTdxVerifier(collateral_provider=...) directly instead."
            )
        dcap = _require_dcap()
        raw = bytes.fromhex(probe_quote) if isinstance(probe_quote, str) else bytes(probe_quote)
        collateral = await dcap.get_collateral(pccs_url, raw)
        kwargs.pop("collateral", None)
        return cls(collateral=collateral, **kwargs)

    def _now(self) -> int:
        return self._now_secs if self._now_secs is not None else int(time.time())

    def verify(self, attestation: TeeAttestation) -> bool:
        """Return ``True`` iff the full TDX quote chain + policy all validate.

        Fail-closed: any failed step raises :class:`TeeError`. Returns ``True``
        only when every step passes. Records a structured result in
        :attr:`last_result`; note that ``last_result["checks"]["compose_binding"]``
        may be the string ``"unavailable"`` rather than a bool (when the wire
        cannot establish compose identity) — test it with ``is True``, not
        truthiness.
        """
        dcap = _require_dcap()
        checks: dict[str, bool | str] = {}

        # (0) NORMALIZE — read schema-independent inputs. Uses intel_quote (NOT
        # quote): only intel_quote's REPORTDATA binds the signing address.
        evidence = normalize(attestation)
        raw_quote = evidence.raw_quote

        # (1) STRUCTURAL PARSE (no signature trust yet).
        try:
            quote = dcap.parse_quote(raw_quote)
        except Exception as exc:  # noqa: BLE001 - any parse failure is fatal
            raise TeeError(f"TDX quote could not be parsed (fail-closed): {exc!r}") from exc
        if not quote.is_tdx():
            raise TeeError("TDX quote is not a TDX quote (fail-closed).")

        # (2) THE TRUST GATE — signature + PCK chain to pinned root + TCB + QE id.
        # Every parsed byte below is trustworthy ONLY because this passes here.
        now = self._now()
        collateral_key: CollateralKey | None
        if self._collateral_provider is not None:
            collateral_key = self._provider_key(quote)
            candidate = self._collateral_provider(collateral_key, now)
            if candidate is None:
                raise TeeCollateralUnavailableError(
                    f"collateral_provider has no collateral for FMSPC {collateral_key.fmspc} "
                    f"and PCK CA {collateral_key.pck_ca_sha256} (fail-closed).",
                    key=collateral_key,
                    raw_quote=raw_quote,
                )
            if not isinstance(candidate, dcap.QuoteCollateralV3):
                raise TeeAttestationError(
                    "collateral_provider returned "
                    f"{type(candidate).__name__}, not a dcap_qvl.QuoteCollateralV3 "
                    "(fail-closed)."
                )
        elif self._collateral is None:
            raise TeeError(
                "No Intel-signed collateral available for full quote verification; "
                "refusing to downgrade to baseline trust (fail-closed). Supply "
                "collateral or use DcapTdxVerifier.with_fetched_collateral(...)."
            )
        else:
            candidate = self._collateral
            try:
                collateral_key = _collateral_key_from_quote(quote)
            except Exception:  # noqa: BLE001 - the key is informational on this path
                collateral_key = None
        collateral = self._collateral_for_quote(
            dcap, quote, candidate, from_provider=self._collateral_provider is not None
        )
        self._check_collateral_window(collateral, collateral_key, raw_quote, now)
        try:
            verified = dcap.verify_with_root_ca(raw_quote, collateral, self._root_ca_der, now)
        except Exception as exc:  # noqa: BLE001 - any crypto failure is fatal
            raise TeeError(
                f"TDX quote signature/chain verification failed (fail-closed): {exc!r}"
            ) from exc
        checks["signature_chain"] = True

        # (3) TCB-STATUS POLICY.
        advisory_ids = list(getattr(verified, "advisory_ids", None) or [])
        self._evaluate_tcb_status(verified.status, advisory_ids)
        checks["tcb_status"] = True

        report = quote.report

        # (4) DEBUG-BIT — from the now-verified report.
        self._check_debug_bit(report)
        checks["debug_bit"] = True

        # (5) REPORTDATA binding of the signing address and nonce.
        self._check_reportdata(report, evidence)
        checks["reportdata_binding"] = True

        # (6) EVENT-LOG SHA384 replay == rt_mr0..3 (self-consistency).
        self._check_rtmr_replay(report, evidence)
        checks["rtmr_replay"] = True

        # (7) COMPOSE binding, when the wire carries the inputs for it.
        compose_hash = self._check_compose_binding(report, evidence)
        checks["compose_binding"] = True if compose_hash is not None else "unavailable"

        # (8) MEASUREMENT POLICY (Tier A only when caller supplied references).
        workload_identity_pinned = self._check_expected(report, compose_hash)
        checks["workload_identity_pinned"] = workload_identity_pinned

        # fmspc() reads the PCK chain embedded in the quote, which collateral can
        # supply in its place, so it can fail on a quote that passed the gate.
        try:
            fmspc = quote.fmspc() if hasattr(quote, "fmspc") else None
        except ValueError as exc:
            raise TeeError(
                "This TDX quote carries no embedded PCK chain, so its FMSPC could not be "
                f"read (fail-closed): {exc}"
            ) from exc

        self.last_result = {
            "checks": checks,
            "schema": evidence.schema,
            "tcb_status": verified.status,
            "advisory_ids": advisory_ids,
            "workload_identity_pinned": workload_identity_pinned,
            "fmspc": fmspc,
            "collateral_key": collateral_key,
            "unsigned_metadata": {
                "os_image_hash": evidence.os_image_hash,
                "repo_commit": evidence.repo_commit,
            },
        }
        return True

    # --- step helpers --------------------------------------------------------

    def _provider_key(self, quote: Any) -> CollateralKey:
        """The quote's collateral key for ``collateral_provider``, or refuse."""
        try:
            embedded = getattr(quote, "cert_chain_pem_bytes", lambda: None)()
            key = _collateral_key_from_quote(quote) if embedded else None
        except Exception as exc:  # noqa: BLE001 - an unreadable chain is fatal
            raise TeeAttestationError(
                "This TDX quote's embedded PCK chain could not be read, so "
                f"collateral_provider cannot be given its FMSPC and PCK CA (fail-closed): {exc}"
            ) from exc
        if not embedded:
            raise TeeAttestationError(
                "This TDX quote carries no embedded PCK chain, so collateral_provider cannot "
                "be given its FMSPC and PCK CA (fail-closed)."
            )
        if key is None:
            raise TeeAttestationError(
                "This TDX quote has no PCK CA certificate in its embedded chain, so "
                "collateral_provider cannot be given its PCK CA (fail-closed)."
            )
        return key

    @staticmethod
    def _check_collateral_window(
        collateral: Any, key: CollateralKey | None, raw_quote: bytes, now: int
    ) -> None:
        """Raise :class:`TeeCollateralExpiredError` for collateral past its ``nextUpdate``.

        This only refuses earlier and more specifically than the crypto gate, which
        still decides every acceptance. An item that cannot be read is skipped so
        ``verify_with_root_ca`` rules on it; the only exception raised is the typed
        expiry, for the item with the earliest ``nextUpdate``.
        """
        if not hasattr(collateral, "to_json"):
            return
        try:
            stored = json.loads(collateral.to_json())
        except Exception:  # noqa: BLE001 - unreadable collateral is left to dcap
            return
        if not isinstance(stored, dict):
            return

        expired: list[tuple[datetime, str]] = []
        for item in ("tcb_info", "qe_identity", "pck_crl", "root_ca_crl"):
            try:
                parsed: datetime | None
                if item in ("tcb_info", "qe_identity"):
                    text = json.loads(stored[item])["nextUpdate"]
                    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                else:
                    from cryptography import x509

                    parsed = x509.load_der_x509_crl(bytes.fromhex(stored[item])).next_update_utc
                if parsed is None:
                    continue
                next_update = parsed
                if next_update.tzinfo is None:
                    next_update = next_update.replace(tzinfo=UTC)
                # dcap refuses a CRL at exactly its nextUpdate; TCB info and QE identity
                # keep a strict comparison.
                if item in ("pck_crl", "root_ca_crl"):
                    is_expired = now >= next_update.timestamp()
                else:
                    is_expired = now > next_update.timestamp()
            except Exception:  # noqa: BLE001  # nosec B112 - an unreadable item is left to dcap
                continue
            if is_expired:
                expired.append((next_update, item))
        if not expired:
            return
        next_update, item = min(expired)
        raise TeeCollateralExpiredError(
            f"Attestation collateral {item} expired at {next_update.isoformat()} "
            "(fail-closed); fetch fresh collateral for this quote.",
            key=key,
            item=item,  # type: ignore[arg-type]
            next_update=next_update,
            raw_quote=raw_quote,
        )

    def _collateral_for_quote(
        self, dcap: ModuleType, quote: Any, candidate: Any, *, from_provider: bool
    ) -> Any:
        """Return the collateral to verify ``quote`` with, or refuse a mismatch.

        ``candidate`` is the fixed ``collateral`` or, when ``from_provider``, what
        ``collateral_provider`` returned for this quote; both get the same checks,
        and only the advice in a mismatch error differs.

        ``dcap_qvl.get_collateral`` attaches the probe machine's PCK chain to the
        collateral, and ``verify_with_root_ca`` prefers it to the chain in the
        quote, so a quote from any other machine fails on its signature. Dropping
        the attached chain makes the library use the quote's own, which it
        validates to the pinned root. A quote without an embedded chain keeps the
        collateral as it is.

        Nothing here trusts the parsed quote: these comparisons can only refuse,
        and the crypto gate still decides. They exist so a quote that needs other
        collateral says so, instead of failing with a generic error or, for the PCK
        CA, being checked against the CRL of a CA that did not issue its PCK.
        """
        collateral: Any = candidate
        embedded = getattr(quote, "cert_chain_pem_bytes", lambda: None)()
        if not embedded or not hasattr(collateral, "to_json"):
            return collateral

        try:
            stored = json.loads(collateral.to_json())
            collateral_fmspc = json.loads(stored["tcb_info"])["fmspc"]
            quote_fmspc = quote.fmspc()
            collateral_key = _lenient_key(
                stored["pck_crl_issuer_chain"], collateral_fmspc, ca_index=0
            )
            quote_key = _lenient_key(embedded, quote_fmspc, ca_index=1)
        except Exception as exc:  # noqa: BLE001 - unreadable collateral is fatal
            raise TeeError(
                f"TEE collateral could not be read for this quote (fail-closed): {exc!r}"
            ) from exc

        if from_provider:
            needs_own = (
                "collateral_provider returned collateral that does not match the key it "
                "was given; check how its store is keyed (fail-closed)."
            )
        else:
            needs_own = (
                "Build a verifier for it with DcapTdxVerifier.with_fetched_collateral(...) "
                "(fail-closed)."
            )
        if collateral_key.fmspc != quote_key.fmspc:
            raise TeeCollateralMismatchError(
                f"TEE collateral is for FMSPC {collateral_fmspc}, but this quote is from "
                f"FMSPC {quote_fmspc}. {needs_own}",
                reason="fmspc",
                collateral_key=collateral_key,
                quote_key=quote_key,
            )
        # A PCK chain runs leaf, PCK CA, root; a CRL issuer chain starts at the PCK CA.
        if (
            not collateral_key.pck_ca_sha256
            or collateral_key.pck_ca_sha256 != quote_key.pck_ca_sha256
        ):
            raise TeeCollateralMismatchError(
                f"TEE collateral's revocation data is for a different PCK CA than the one "
                f"that issued this quote's certificate. {needs_own}",
                reason="pck_ca",
                collateral_key=collateral_key,
                quote_key=quote_key,
            )

        stored.pop("pck_certificate_chain", None)
        return dcap.QuoteCollateralV3.from_json(json.dumps(stored))

    def _evaluate_tcb_status(self, status: str, advisory_ids: list[str]) -> None:
        if status == self.ACCEPT_TCB_STATUS:
            return
        if self._tcb_policy == "advisory" and status in self._ADVISORY_TOLERATED:
            warnings.warn(
                f"TEE TCB status is {status!r} (accepted under advisory policy); "
                f"advisory IDs: {advisory_ids or 'none'}.",
                UserWarning,
                stacklevel=3,
            )
            return
        raise TeeTcbStatusError(
            f"TEE TCB status {status!r} is not acceptable under the "
            f"{self._tcb_policy!r} policy (fail-closed).",
            status=status,
            advisory_ids=advisory_ids,
            policy=self._tcb_policy,
        )

    def _check_debug_bit(self, report: Any) -> None:
        td_attributes = bytes(report.td_attributes)
        if int.from_bytes(td_attributes, "little") & _TDX_DEBUG_BIT:
            raise TeeError("TDX enclave is in DEBUG mode (TUD.DEBUG set); refusing (fail-closed).")

    def _check_reportdata(self, report: Any, evidence: NormalizedEvidence) -> None:
        sent_nonce = evidence.sent_nonce
        if not isinstance(sent_nonce, str) or not sent_nonce:
            raise TeeError(
                "TEE attestation sent_nonce is missing; cannot verify the REPORTDATA "
                "key binding (fail-closed)."
            )
        try:
            nonce_bytes = bytes.fromhex(sent_nonce.lower())
        except ValueError as exc:
            raise TeeError("TEE attestation sent_nonce is not valid hex (fail-closed).") from exc
        if len(nonce_bytes) != _NONCE_LEN:
            raise TeeError(f"TEE attestation sent_nonce must be {_NONCE_LEN} bytes (fail-closed).")
        addr = evidence.signing_address.lower().removeprefix("0x")
        try:
            addr_bytes = bytes.fromhex(addr)
        except ValueError as exc:
            raise TeeError(
                "TEE attestation signing_address is not valid hex (fail-closed)."
            ) from exc
        if len(addr_bytes) != _ADDR_LEN:
            raise TeeError(
                f"TEE attestation signing_address must be {_ADDR_LEN} bytes (fail-closed)."
            )
        expected = addr_bytes + _ZERO_PAD + nonce_bytes
        report_data = bytes(report.report_data)
        if report_data[:_REPORTDATA_LEN] != expected:
            raise TeeError(
                "TEE REPORTDATA binding mismatch: report_data != signing_address || "
                "zero-pad || sent_nonce (fail-closed)."
            )

    def _check_rtmr_replay(self, report: Any, evidence: NormalizedEvidence) -> None:
        # Replay the complete event log against the quoted RTMRs. This is what
        # makes the (unsigned) log trustworthy: it must reproduce signed values.
        event_log = evidence.event_log
        rtmrs = [report.rt_mr0, report.rt_mr1, report.rt_mr2, report.rt_mr3]
        for imr in range(4):
            acc = b"\x00" * 48
            for entry in event_log:
                if not isinstance(entry, dict) or entry.get("imr") != imr:
                    continue
                digest = entry.get("digest")
                if not isinstance(digest, str):
                    raise TeeError("TEE event_log entry has a non-string digest (fail-closed).")
                try:
                    acc = hashlib.sha384(acc + bytes.fromhex(digest)).digest()
                except ValueError as exc:
                    raise TeeError("TEE event_log digest is not valid hex (fail-closed).") from exc
            if acc != bytes(rtmrs[imr]):
                raise TeeError(f"TEE event-log replay does not reproduce rt_mr{imr} (fail-closed).")

    def _check_compose_binding(self, report: Any, evidence: NormalizedEvidence) -> str | None:
        """Verify the compose binding, or return ``None`` when unverifiable.

        The current (``SCHEMA_EVIDENCE``) wire carries neither ``app_compose``
        nor ``compose_hash`` at all, so compose identity cannot be established
        from the attestation; that is reported as unavailable rather than
        passing, and callers needing this dimension must pin ``mr_config_id``
        via ``expected_measurements``. On the dstack wire these fields are
        mandatory: their absence indicates a malformed or tampered attestation,
        not a schema limitation, so it fails closed instead of downgrading.
        """
        if evidence.schema == SCHEMA_EVIDENCE:
            return None
        if evidence.compose_hash is None:
            raise TeeError("TEE attestation info.compose_hash is missing (fail-closed).")
        if evidence.app_compose is None:
            raise TeeError("TEE attestation info.tcb_info.app_compose is missing (fail-closed).")
        compose_hash = evidence.compose_hash.lower()
        app_compose = evidence.app_compose
        computed = hashlib.sha256(app_compose.encode()).hexdigest()
        if computed != compose_hash:
            raise TeeError(
                "TEE compose binding mismatch: SHA256(app_compose) != compose_hash (fail-closed)."
            )
        # compose_hash must be embedded in mr_config_id as 0x01 || compose_hash || pad.
        try:
            ch_bytes = bytes.fromhex(compose_hash)
        except ValueError as exc:
            raise TeeError("TEE compose_hash is not valid hex (fail-closed).") from exc
        mr_config_id = bytes(report.mr_config_id)
        expected = _COMPOSE_PREFIX + ch_bytes
        if (
            len(mr_config_id) < len(expected)
            or mr_config_id[: len(expected)] != expected
            or any(mr_config_id[len(expected) :])
        ):
            raise TeeError(
                "TEE compose_hash is not embedded in mr_config_id as "
                "0x01 || compose_hash || zero-pad (fail-closed)."
            )
        return compose_hash

    def _check_expected(self, report: Any, compose_hash: str | None) -> bool:
        """Compare any caller-supplied references; return whether any were pinned."""
        pinned = False
        field_map = {
            "mrtd": report.mr_td,
            "rtmr0": report.rt_mr0,
            "rtmr1": report.rt_mr1,
            "rtmr2": report.rt_mr2,
            "rtmr3": report.rt_mr3,
            "mr_config_id": report.mr_config_id,
        }
        for key, expected_hex in self._expected_measurements.items():
            if key not in field_map:
                raise TeeError(f"Unknown expected_measurements key {key!r} (fail-closed).")
            actual = bytes(field_map[key]).hex()
            if actual != expected_hex:
                raise TeeError(
                    f"TEE measurement mismatch for {key!r}: expected does not match the "
                    "verified quote field (fail-closed)."
                )
            pinned = True
        if self._expected_compose_hash is not None:
            if compose_hash is None:
                raise TeeError(
                    "expected_compose_hash was supplied but this attestation does not "
                    "carry a compose hash to compare against (fail-closed)."
                )
            if self._expected_compose_hash != compose_hash:
                raise TeeError(
                    "TEE expected_compose_hash mismatch against the verified quote (fail-closed)."
                )
            pinned = True
        return pinned
