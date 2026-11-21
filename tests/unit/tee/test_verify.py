"""Tests for ``DcapTdxVerifier`` (full client-side Intel TDX quote verification).

REJECTION PATHS ARE LOAD-BEARING: a verifier whose entire purpose is "don't
trust the server" is only meaningful if every fail-closed path actually bites.
These tests come first; the happy path is last.

All tests run OFFLINE against the four committed live captures + their matching
collateral snapshots, with ``now_secs`` frozen to :data:`CAPTURE_EPOCH` so the
collateral validity windows pass deterministically.

Two-layer rejection structure (mirrors the design doc §7):

* **Crypto-layer** rejections byte-flip the REAL fixtures so
  ``verify_with_root_ca`` fails (signature / PCK-chain / stale collateral /
  truncated quote). These need no stubbing.
* **Policy-layer** rejections (debug bit set, bad TCB status, measurement /
  nonce / compose mismatch) cannot be produced by tampering a *signed* field
  without first breaking the signature, so they stub ``dcap_qvl.parse_quote`` /
  ``dcap_qvl.verify_with_root_ca`` (the synthetic path the doc calls out) to
  exercise the policy gates downstream of a *passing* crypto gate.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import venice_ai.tee._verify as verify_mod
from venice_ai.exceptions import (
    TeeAttestationError,
    TeeCollateralExpiredError,
    TeeCollateralMismatchError,
    TeeCollateralUnavailableError,
    TeeError,
    TeeTcbStatusError,
)
from venice_ai.tee._attestation import verify_attestation
from venice_ai.tee._constants import INTEL_SGX_ROOT_CA_DER
from venice_ai.tee._verify import DcapTdxVerifier, _collateral_key_from_quote
from venice_ai.tee.types import _PEM_CERTIFICATE, CollateralKey, TeeAttestation

from .fixtures import (
    CAPTURE_EPOCH,
    EVIDENCE_CAPTURE_EPOCH,
    FIXTURE_SLUGS,
    load_attestation,
    load_collateral_json,
)

dcap_qvl = pytest.importorskip("dcap_qvl")


# --------------------------------------------------------------------------- #
# Fixture plumbing
# --------------------------------------------------------------------------- #


def _build_attestation(slug: str) -> tuple[TeeAttestation, str]:
    """Return a populated ``TeeAttestation`` + the client nonce for ``slug``.

    Mirrors the live wiring after the ``resources/tee.py`` fix: ``sent_nonce``
    is populated before the verifier runs.
    """
    raw = load_attestation(slug)
    payload = raw["attestation"]
    nonce = raw["_client_nonce"]
    att = TeeAttestation.model_validate(payload)
    att.sent_nonce = nonce
    return att, nonce


def _collateral(slug: str) -> Any:
    return dcap_qvl.QuoteCollateralV3.from_json(load_collateral_json(slug))


def _verifier(slug: str, **kwargs: Any) -> DcapTdxVerifier:
    """A verifier wired to ``slug``'s collateral, frozen at capture epoch."""
    kwargs.setdefault("collateral", _collateral(slug))
    kwargs.setdefault("now_secs", CAPTURE_EPOCH)
    return DcapTdxVerifier(**kwargs)


# --------------------------------------------------------------------------- #
# T8 prep: PIN the exact TCB-status enum strings against the real library,
# rather than hardcoding them blind in the implementation tests.
# --------------------------------------------------------------------------- #


def test_accept_tcb_status_enum_string_is_pinned_against_real_collateral() -> None:
    """The single load-bearing enum string (``"UpToDate"``) is pinned to reality.

    The policy is accept-list, not reject-list: ``_evaluate_tcb_status`` accepts
    ONLY ``ACCEPT_TCB_STATUS`` (plus the advisory-tolerated set under advisory
    policy) and rejects everything else. So the only string whose exact spelling
    can cause an unsound *accept* is ``ACCEPT_TCB_STATUS`` — pin it against the
    real collateral. ``REJECT_TCB_STATUSES`` is documentary; a wrong entry there
    could only over-reject, never under-accept.
    """
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    raw_quote = bytes.fromhex(att.intel_quote)
    vr = dcap_qvl.verify_with_root_ca(
        raw_quote, _collateral(slug), INTEL_SGX_ROOT_CA_DER, CAPTURE_EPOCH
    )
    # The genuine fixture's status equals the one (and only) accepted string.
    assert vr.status == DcapTdxVerifier.ACCEPT_TCB_STATUS == "UpToDate"


# --------------------------------------------------------------------------- #
# CRYPTO-LAYER rejections (byte-flip the real fixture; gate must fail)
# --------------------------------------------------------------------------- #


def test_tampered_signature_byte_rejects() -> None:
    """T1: flip a byte inside the ISV/QE report signature region → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    raw = bytearray(bytes.fromhex(att.intel_quote))
    raw[650] ^= 0xFF  # inside the signed ECDSA region (empirically verified)
    att.intel_quote = raw.hex()
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_wrong_root_ca_rejects() -> None:
    """T2: a bogus root CA → PCK chain cannot anchor → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    bad_root = b"\x30\x82\x00\x10" + b"\x00" * 12
    with pytest.raises(TeeError):
        _verifier(slug, root_ca_der=bad_root).verify(att)


def test_stale_collateral_now_secs_past_next_update_rejects() -> None:
    """T3: ``now_secs`` ten years past capture → collateral expired → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    far_future = CAPTURE_EPOCH + 10 * 365 * 86400
    with pytest.raises(TeeError):
        _verifier(slug, now_secs=far_future).verify(att)


def test_truncated_quote_rejects() -> None:
    """T4: a truncated quote → ``parse_quote`` fails → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    att.intel_quote = att.intel_quote[:200]  # too short to parse
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_missing_intel_quote_rejects() -> None:
    """Absent ``intel_quote`` → reject (never skip)."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    att.intel_quote = None
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_odd_length_intel_quote_rejects() -> None:
    """Odd-length hex → cannot decode → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    att.intel_quote = att.intel_quote + "a"  # odd length
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


# --------------------------------------------------------------------------- #
# POLICY-LAYER rejections (stub the crypto gate to PASS, then exercise policy)
# --------------------------------------------------------------------------- #


@dataclass
class _FakeReport:
    td_attributes: bytes
    report_data: bytes
    mr_td: bytes
    mr_config_id: bytes
    rt_mr0: bytes
    rt_mr1: bytes
    rt_mr2: bytes
    rt_mr3: bytes


@dataclass
class _FakeQuote:
    report: _FakeReport

    def is_tdx(self) -> bool:
        return True


@dataclass
class _FakeVerified:
    status: str
    advisory_ids: list[str]


def _real_report(slug: str) -> _FakeReport:
    """The genuine parsed report for ``slug`` as a mutable fake."""
    att, _ = _build_attestation(slug)
    q = dcap_qvl.parse_quote(bytes.fromhex(att.intel_quote))
    r = q.report
    return _FakeReport(
        td_attributes=bytes(r.td_attributes),
        report_data=bytes(r.report_data),
        mr_td=bytes(r.mr_td),
        mr_config_id=bytes(r.mr_config_id),
        rt_mr0=bytes(r.rt_mr0),
        rt_mr1=bytes(r.rt_mr1),
        rt_mr2=bytes(r.rt_mr2),
        rt_mr3=bytes(r.rt_mr3),
    )


def _stub_crypto(
    monkeypatch: pytest.MonkeyPatch,
    report: _FakeReport,
    *,
    status: str = "UpToDate",
    advisory_ids: list[str] | None = None,
) -> None:
    """Make ``parse_quote``/``verify_with_root_ca`` succeed with given report+status."""
    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda raw: _FakeQuote(report))
    monkeypatch.setattr(
        verify_mod.dcap_qvl,
        "verify_with_root_ca",
        lambda raw, col, root, now: _FakeVerified(status, advisory_ids or []),
    )


def test_debug_bit_set_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: TUD.DEBUG bit (bit 0, little-endian) set in td_attributes → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    report = _real_report(slug)
    report.td_attributes = (0x1).to_bytes(8, "little")  # debug bit set
    _stub_crypto(monkeypatch, report)
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_tcb_status_out_of_date_rejects_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: OutOfDate → reject under default ('reject') policy."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug), status="OutOfDate")
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_tcb_status_swhardening_rejects_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: SWHardeningNeeded → reject by default."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug), status="SWHardeningNeeded")
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_tcb_status_swhardening_accepted_in_advisory_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: advisory mode accepts SWHardeningNeeded but surfaces advisory_ids."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(
        monkeypatch,
        _real_report(slug),
        status="SWHardeningNeeded",
        advisory_ids=["INTEL-SA-00999"],
    )
    v = _verifier(slug, tcb_policy="advisory")
    with pytest.warns(UserWarning):
        assert v.verify(att) is True
    assert v.last_result is not None
    assert v.last_result["checks"]["tcb_status"] is True
    assert v.last_result["advisory_ids"] == ["INTEL-SA-00999"]


def test_tcb_status_revoked_rejects_even_in_advisory_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advisory mode must NOT accept Revoked/OutOfDate (only SWHardening*)."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug), status="Revoked")
    with pytest.raises(TeeError):
        _verifier(slug, tcb_policy="advisory").verify(att)


def test_wrong_expected_mrtd_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6: caller-supplied expected mrtd mismatch → fail-closed."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug))
    with pytest.raises(TeeError):
        _verifier(slug, expected_measurements={"mrtd": "00" * 48}).verify(att)


def test_wrong_nonce_in_reportdata_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """T7: sent_nonce different from the one bound in REPORTDATA → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug))
    att.sent_nonce = "ff" * 32  # not the nonce bound into the report
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_missing_nonce_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sent_nonce → cannot recompute the binding → reject (not skip)."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug))
    att.sent_nonce = None
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_corrupt_event_log_digest_rejects_rtmr_replay() -> None:
    """T9: corrupting an event-log digest (imr 0..2) → RTMR replay mismatch.

    Targets imr 0 (always replayable) so the check bites regardless of the
    runtime-extended rt_mr3 nuance.
    """
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    info = copy.deepcopy(att.info)
    tcb = info["tcb_info"]
    el = tcb["event_log"]
    for entry in el:
        if entry.get("imr") == 0:
            d = bytearray(bytes.fromhex(entry["digest"]))
            d[0] ^= 0xFF
            entry["digest"] = d.hex()
            break
    att.info = info
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_mutated_app_compose_rejects_compose_hash() -> None:
    """T10: altering app_compose so SHA256 != compose_hash → reject."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    info = copy.deepcopy(att.info)
    info["tcb_info"]["app_compose"] = info["tcb_info"]["app_compose"] + "tampered"
    att.info = info
    with pytest.raises(TeeError):
        _verifier(slug).verify(att)


def test_missing_compose_hash_on_dstack_wire_rejects() -> None:
    """Regression: on the dstack wire, ``info.compose_hash`` is mandatory.

    Its absence must raise fail-closed. Stripping it must NOT be a way to
    downgrade this check to a passing ``"unavailable"`` — that string is only
    legitimate on the current (``SCHEMA_EVIDENCE``) wire, which genuinely never
    carries this field. On the dstack wire, absence means tampering or a
    malformed attestation.
    """
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    info = copy.deepcopy(att.info)
    del info["compose_hash"]
    att.info = info
    v = _verifier(slug)
    with pytest.raises(TeeError, match="compose_hash is missing"):
        v.verify(att)
    # Never reached a result — in particular, never a passing "unavailable".
    assert v.last_result is None


def test_missing_app_compose_on_dstack_wire_rejects() -> None:
    """Regression: on the dstack wire, ``info.tcb_info.app_compose`` is
    mandatory. Same rationale as ``test_missing_compose_hash_on_dstack_wire_rejects``.
    """
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    info = copy.deepcopy(att.info)
    del info["tcb_info"]["app_compose"]
    att.info = info
    v = _verifier(slug)
    with pytest.raises(TeeError, match="app_compose is missing"):
        v.verify(att)
    assert v.last_result is None


def test_collateral_none_and_no_fetch_rejects() -> None:
    """T11: collateral=None and not fetched → reject (never downgrade to baseline)."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    v = DcapTdxVerifier(collateral=None, now_secs=CAPTURE_EPOCH)
    with pytest.raises(TeeError):
        v.verify(att)


def test_require_dcap_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """T12: dcap-qvl missing → _require_dcap raises with the install hint; never silent-skip."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dcap_qvl":
            raise ImportError("no module named dcap_qvl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TeeError) as exc:
        verify_mod._require_dcap()
    assert "venice-py[e2ee-verify]" in str(exc.value)


# --------------------------------------------------------------------------- #
# HAPPY PATH (green last)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", FIXTURE_SLUGS)
def test_happy_path_all_fixtures_verify_tier_b(slug: str) -> None:
    """T13: all four captures verify True at Tier B with frozen now_secs.

    Includes the near-ai ``glm_5_1`` (keys off dstack structure, not provider)
    and the two captures where ``intel_quote != quote``.
    """
    att, _ = _build_attestation(slug)
    v = _verifier(slug)
    assert v.verify(att) is True
    assert v.last_result is not None
    # Tier B: no expected_* supplied → workload identity is NOT independently pinned.
    assert v.last_result["workload_identity_pinned"] is False
    assert v.last_result["checks"]["signature_chain"] is True
    assert v.last_result["checks"]["reportdata_binding"] is True
    assert v.last_result["checks"]["rtmr_replay"] is True
    assert v.last_result["checks"]["compose_binding"] is True


def test_intel_quote_is_the_bound_field_not_quote() -> None:
    """The verifier MUST consume ``intel_quote`` (the key-binding quote), not ``quote``.

    On ``gpt_oss_120b_p`` the two differ; corrupting ``quote`` must NOT affect
    the verdict, while the genuine ``intel_quote`` still verifies True.
    """
    slug = "gpt_oss_120b_p"
    att, _ = _build_attestation(slug)
    assert att.intel_quote != att.quote  # precondition: they differ on this model
    att.quote = "deadbeef"  # corrupt the field we must NOT read
    v = _verifier(slug)
    assert v.verify(att) is True


def test_expected_compose_hash_match_pins_workload_identity() -> None:
    """T15: caller pins expected_compose_hash matching the fixture → True + pinned True."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    compose_hash = att.info["tcb_info"]["compose_hash"]
    v = _verifier(slug, expected_compose_hash=compose_hash)
    assert v.verify(att) is True
    assert v.last_result is not None
    assert v.last_result["workload_identity_pinned"] is True


def test_expected_compose_hash_mismatch_rejects() -> None:
    """A wrong expected_compose_hash → fail-closed (Tier-A dimension mismatch)."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    v = _verifier(slug, expected_compose_hash="ab" * 32)
    with pytest.raises(TeeError):
        v.verify(att)


def test_expected_measurements_mrtd_match_pins_workload_identity() -> None:
    """Caller pins the genuine mrtd → True + workload_identity_pinned True."""
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    q = dcap_qvl.parse_quote(bytes.fromhex(att.intel_quote))
    mrtd = bytes(q.report.mr_td).hex()
    v = _verifier(slug, expected_measurements={"mrtd": mrtd})
    assert v.verify(att) is True
    assert v.last_result is not None
    assert v.last_result["workload_identity_pinned"] is True


def test_protocol_conformance() -> None:
    """DcapTdxVerifier satisfies the shipped FullQuoteVerifier Protocol."""
    from venice_ai.tee import FullQuoteVerifier

    v = _verifier("gemma_3_27b_p")
    assert isinstance(v, FullQuoteVerifier)


# --------------------------------------------------------------------------- #
# T14: WIRING — sent_nonce is populated BEFORE the verifier runs, and a
# verifier flows through to checks["full_quote"]. Locks the resources/tee.py fix.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_verifier_sees_populated_sent_nonce_before_running() -> None:
    """Regression for the §5 wiring fix: at verify-time sent_nonce is NOT None."""
    from unittest.mock import AsyncMock, MagicMock

    from venice_ai.resources.tee import Tee

    slug = "gemma_3_27b_p"
    raw = load_attestation(slug)
    payload = raw["attestation"]
    sent = raw["_client_nonce"]
    # Make the server-echoed nonce match so the baseline nonce_echo passes.
    payload = dict(payload)
    payload["nonce"] = sent
    attestation = TeeAttestation.model_validate(payload)

    seen: dict[str, Any] = {}

    class _Spy:
        def verify(self, att: TeeAttestation) -> bool:
            seen["sent_nonce"] = att.sent_nonce
            return True

    tee = Tee(MagicMock())
    tee._client.get = AsyncMock(return_value=attestation)  # type: ignore[attr-defined]
    result = await tee.get_attestation(model="e2ee-gemma-3-27b-p", nonce=sent, verifier=_Spy())

    assert seen["sent_nonce"] == sent  # populated BEFORE the verifier ran
    assert result.sent_nonce == sent


@pytest.mark.asyncio
async def test_with_fetched_collateral_call_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async factory fetches via PCCS with the verified ``(url, raw_quote)``
    call shape and returns a constructed, offline-ready verifier.

    Locks the ``dcap_qvl.get_collateral(pccs_url, raw_quote)`` signature (the one
    network touch) without hitting the network. ``verify`` stays offline.
    """
    from venice_ai.tee._constants import PHALA_PCCS_URL

    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    real_collateral = _collateral(slug)
    captured: dict[str, Any] = {}

    async def _fake_get_collateral(url: str, raw: bytes) -> Any:
        captured["url"] = url
        captured["raw"] = raw
        return real_collateral

    monkeypatch.setattr(verify_mod.dcap_qvl, "get_collateral", _fake_get_collateral)

    v = await DcapTdxVerifier.with_fetched_collateral(att.intel_quote, now_secs=CAPTURE_EPOCH)

    assert captured["url"] == PHALA_PCCS_URL  # default no-auth mirror
    assert captured["raw"] == bytes.fromhex(att.intel_quote)
    # The returned verifier is offline-ready: it verifies the real fixture True.
    assert v.verify(att) is True


# --------------------------------------------------------------------------- #
# Reusing one verifier across machines (real crypto, offline)
# --------------------------------------------------------------------------- #

# Two captures from different machines on the same FMSPC and PCK CA.
_MACHINE_A = "gemma_3_27b_p"
_MACHINE_B = "gpt_oss_120b_p"
# The same machine as _MACHINE_A, serving another model.
_MACHINE_A_AGAIN = "qwen3_vl_30b_a3b_p"
# A capture on another FMSPC.
_OTHER_FMSPC = "live_evidence"


def _quote(slug: str) -> Any:
    att, _nonce = _build_attestation(slug)
    return dcap_qvl.parse_quote(bytes.fromhex(att.intel_quote))


def test_fixtures_cover_two_machines_on_one_fmspc() -> None:
    """Pins what these tests rely on, and how dcap_qvl exposes platform identity."""
    a, b, again = _quote(_MACHINE_A), _quote(_MACHINE_B), _quote(_MACHINE_A_AGAIN)

    assert a.fmspc() == b.fmspc() == again.fmspc()
    assert a.ca() == b.ca()
    assert a.pck_extension().ppid != b.pck_extension().ppid
    assert a.pck_extension().platform_instance_id != b.pck_extension().platform_instance_id
    assert a.pck_extension().ppid == again.pck_extension().ppid
    assert _quote(_OTHER_FMSPC).fmspc() != a.fmspc()


def test_one_verifier_accepts_quotes_from_two_machines_on_one_fmspc() -> None:
    verifier = _verifier(_MACHINE_A)

    for slug in (_MACHINE_A, _MACHINE_B, _MACHINE_A):
        att, _nonce = _build_attestation(slug)
        assert verifier.verify(att) is True
        assert verifier.last_result is not None
        assert verifier.last_result["checks"]["signature_chain"] is True


def test_reuse_does_not_change_the_callers_collateral() -> None:
    collateral = _collateral(_MACHINE_A)
    before = collateral.to_json()
    att, _nonce = _build_attestation(_MACHINE_B)

    assert _verifier(_MACHINE_A, collateral=collateral).verify(att) is True

    assert collateral.to_json() == before


def test_collateral_without_a_pck_chain_verifies() -> None:
    stored = json.loads(load_collateral_json(_MACHINE_A))
    del stored["pck_certificate_chain"]
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))
    att, _nonce = _build_attestation(_MACHINE_B)

    assert _verifier(_MACHINE_A, collateral=collateral).verify(att) is True


def test_quote_from_another_machine_is_still_checked_against_the_pinned_root() -> None:
    # A root that still parses, holding another public key: the byte after the
    # marker of an uncompressed EC point is inside the key.
    key_at = INTEL_SGX_ROOT_CA_DER.index(b"\x03\x42\x00\x04") + 4
    wrong_root = bytearray(INTEL_SGX_ROOT_CA_DER)
    wrong_root[key_at] ^= 0xFF
    att, _nonce = _build_attestation(_MACHINE_B)

    with pytest.raises(TeeError, match="fail-closed"):
        _verifier(_MACHINE_A, root_ca_der=bytes(wrong_root)).verify(att)


def test_tampered_quote_from_another_machine_is_rejected() -> None:
    att, _nonce = _build_attestation(_MACHINE_B)
    raw = bytearray(bytes.fromhex(att.intel_quote))
    raw[700] ^= 0xFF
    tampered = att.model_copy(update={"intel_quote": raw.hex()})

    with pytest.raises(TeeError, match="fail-closed"):
        _verifier(_MACHINE_A).verify(tampered)


def test_verified_quote_with_no_embedded_chain_raises_tee_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fmspc()`` reads the chain in the quote; collateral can verify one without it."""

    class _NoChainQuote(_FakeQuote):
        def fmspc(self) -> str:
            raise ValueError("no embedded PCK chain")

    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    report = _real_report(slug)
    _stub_crypto(monkeypatch, report)
    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda raw: _NoChainQuote(report))

    with pytest.raises(TeeError, match="FMSPC"):
        _verifier(slug).verify(att)


def test_quote_on_another_fmspc_is_refused_and_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_run(*_args: Any) -> None:
        pytest.fail("verified with collateral for another FMSPC")

    monkeypatch.setattr(verify_mod.dcap_qvl, "verify_with_root_ca", must_not_run)
    att, _nonce = _build_attestation(_OTHER_FMSPC)

    with pytest.raises(TeeCollateralMismatchError, match="FMSPC") as excinfo:
        _verifier(_MACHINE_A).verify(att)

    assert _quote(_OTHER_FMSPC).fmspc() in str(excinfo.value)
    assert "with_fetched_collateral" in str(excinfo.value)
    err = excinfo.value
    assert err.reason == "fmspc"
    assert err.quote_key.fmspc == _quote(_OTHER_FMSPC).fmspc().upper()
    assert err.collateral_key.fmspc == json.loads(_collateral(_MACHINE_A).tcb_info)["fmspc"]


def test_collateral_from_another_pck_ca_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(*_args: Any) -> None:
        pytest.fail("verified with revocation data for another PCK CA")

    monkeypatch.setattr(verify_mod.dcap_qvl, "verify_with_root_ca", must_not_run)
    stored = json.loads(load_collateral_json(_MACHINE_A))
    # A chain whose first certificate is not the one that issued the quote's PCK.
    root_only = stored["pck_crl_issuer_chain"].split("-----END CERTIFICATE-----")[1]
    stored["pck_crl_issuer_chain"] = f"{root_only.strip()}\n-----END CERTIFICATE-----\n"
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))
    att, _nonce = _build_attestation(_MACHINE_B)

    with pytest.raises(TeeError, match="PCK CA"):
        _verifier(_MACHINE_A, collateral=collateral).verify(att)


def test_pck_ca_mismatch_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(*_args: Any) -> None:
        pytest.fail("verified with revocation data for another PCK CA")

    monkeypatch.setattr(verify_mod.dcap_qvl, "verify_with_root_ca", must_not_run)
    stored = json.loads(load_collateral_json(_MACHINE_A))
    other = json.loads(load_collateral_json(_OTHER_FMSPC))
    swapped = _PEM_CERTIFICATE.findall(other["tcb_info_issuer_chain"])[0]
    rest = _PEM_CERTIFICATE.findall(stored["pck_crl_issuer_chain"])[1:]
    stored["pck_crl_issuer_chain"] = "".join(
        f"-----BEGIN CERTIFICATE-----{body}-----END CERTIFICATE-----\n" for body in [swapped, *rest]
    )
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))
    att, _nonce = _build_attestation(_MACHINE_B)

    with pytest.raises(TeeCollateralMismatchError, match="PCK CA") as excinfo:
        _verifier(_MACHINE_A, collateral=collateral).verify(att)

    err = excinfo.value
    assert err.reason == "pck_ca"
    assert err.quote_key == CollateralKey.from_quote(bytes.fromhex(att.intel_quote))
    assert err.collateral_key.fmspc == err.quote_key.fmspc
    assert err.collateral_key.pck_ca_sha256 != err.quote_key.pck_ca_sha256


def test_tcb_status_rejection_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(
        monkeypatch, _real_report(slug), status="OutOfDate", advisory_ids=["INTEL-SA-00837"]
    )
    with pytest.raises(TeeTcbStatusError) as excinfo:
        _verifier(slug).verify(att)
    err = excinfo.value
    assert isinstance(err, TeeError)
    assert err.status == "OutOfDate"
    assert err.advisory_ids == ["INTEL-SA-00837"]
    assert err.policy == "reject"


def test_advisory_status_rejected_under_reject_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    _stub_crypto(monkeypatch, _real_report(slug), status="SWHardeningNeeded")
    with pytest.raises(TeeTcbStatusError) as excinfo:
        _verifier(slug, tcb_policy="reject").verify(att)
    assert excinfo.value.policy == "reject"
    assert excinfo.value.status == "SWHardeningNeeded"

    with pytest.warns(UserWarning):
        assert _verifier(slug, tcb_policy="advisory").verify(att) is True


def test_collateral_key_from_quote() -> None:
    att, _ = _build_attestation("glm_5_1")
    raw = bytes.fromhex(att.intel_quote)
    quote = dcap_qvl.parse_quote(raw)
    second = _PEM_CERTIFICATE.findall(quote.cert_chain_pem_bytes().decode("ascii"))[1]
    expected = hashlib.sha256(base64.b64decode("".join(second.split()))).hexdigest()

    key = CollateralKey.from_quote(raw)

    assert key.fmspc == "90C06F000000"
    assert key.pck_ca_sha256 == expected


def test_collateral_key_from_quote_public() -> None:
    att, _ = _build_attestation(_MACHINE_A)
    raw = bytes.fromhex(att.intel_quote)

    key = CollateralKey.from_quote(raw)

    assert key == CollateralKey.from_quote(raw.hex())
    assert {key: 1}[CollateralKey.from_quote(raw)] == 1
    assert key == _collateral_key_from_quote(dcap_qvl.parse_quote(raw))


def test_collateral_key_from_quote_without_chain_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Bare:
        def cert_chain_pem_bytes(self) -> None:
            return None

        def fmspc(self) -> str:
            return "90c06f000000"

    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda raw: _Bare())
    assert _collateral_key_from_quote(_Bare()) is None
    with pytest.raises(TeeAttestationError, match="no PCK certificate chain"):
        CollateralKey.from_quote(b"\x00")


def test_collateral_key_from_quote_unparseable_raises() -> None:
    with pytest.raises(TeeAttestationError):
        CollateralKey.from_quote(b"\x00\x01")
    with pytest.raises(TeeAttestationError):
        CollateralKey.from_quote("not hex")


def test_collateral_key_from_quote_leaf_only_chain_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    att, _ = _build_attestation(_MACHINE_A)
    raw = bytes.fromhex(att.intel_quote)
    real = dcap_qvl.parse_quote(raw)
    leaf = _PEM_CERTIFICATE.findall(real.cert_chain_pem_bytes().decode("ascii"))[0]

    class _LeafOnly:
        def cert_chain_pem_bytes(self) -> bytes:
            return f"-----BEGIN CERTIFICATE-----{leaf}-----END CERTIFICATE-----\n".encode()

        def fmspc(self) -> str:
            return real.fmspc()

    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda _raw: _LeafOnly())
    with pytest.raises(TeeAttestationError, match="no PCK CA certificate"):
        CollateralKey.from_quote(raw)


def test_fmspc_mismatch_is_reported_even_with_a_corrupt_issuer_chain() -> None:
    stored = json.loads(load_collateral_json(_MACHINE_A))
    stored["pck_crl_issuer_chain"] = "-----BEGIN CERTIFICATE-----!!!-----END CERTIFICATE-----"
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))
    att, _nonce = _build_attestation(_OTHER_FMSPC)

    with pytest.raises(TeeCollateralMismatchError) as excinfo:
        _verifier(_MACHINE_A, collateral=collateral).verify(att)
    assert excinfo.value.reason == "fmspc"
    assert excinfo.value.collateral_key.pck_ca_sha256 == ""


def test_typed_errors_survive_copy_and_pickle() -> None:
    att, _ = _build_attestation(_MACHINE_A)
    key = CollateralKey.from_quote(bytes.fromhex(att.intel_quote))
    other = CollateralKey(fmspc="00AA", pck_ca_sha256="ff")
    when = datetime(2026, 1, 2, tzinfo=UTC)
    errors: list[TeeAttestationError] = [
        TeeTcbStatusError("m", status="OutOfDate", advisory_ids=["x"], policy="reject"),
        TeeCollateralMismatchError("m", reason="pck_ca", collateral_key=other, quote_key=key),
        TeeCollateralExpiredError("m", key=key, item="pck_crl", next_update=when, raw_quote=b"q"),
        TeeCollateralUnavailableError("m", key=key, raw_quote=b"q"),
    ]
    for err in errors:
        for clone in (pickle.loads(pickle.dumps(err)), copy.deepcopy(err)):  # noqa: S301
            assert type(clone) is type(err)
            assert str(clone) == "m"
            assert vars(clone) == vars(err)


# --------------------------------------------------------------------------- #
# collateral_provider: collateral chosen per quote
# --------------------------------------------------------------------------- #

_PROVIDER_SLUG = "glm_5_1"


def _raw_quote(slug: str) -> bytes:
    att, _nonce = _build_attestation(slug)
    return bytes.fromhex(att.intel_quote)


class _Provider:
    """A dict-backed provider that records every call."""

    def __init__(self, *slugs: str) -> None:
        self.store = {CollateralKey.from_quote(_raw_quote(s)): _collateral(s) for s in slugs}
        self.calls: list[tuple[CollateralKey, int]] = []

    def __call__(self, key: CollateralKey, now_secs: int) -> Any:
        self.calls.append((key, now_secs))
        return self.store.get(key)


def test_collateral_and_provider_are_mutually_exclusive() -> None:
    with pytest.raises(TeeError, match="collateral_provider"):
        DcapTdxVerifier(collateral=_collateral(_PROVIDER_SLUG), collateral_provider=_Provider())


@pytest.mark.asyncio
async def test_with_fetched_collateral_refuses_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    async def must_not_fetch(*_args: Any) -> None:
        pytest.fail("fetched collateral for a verifier that cannot use it")

    monkeypatch.setattr(verify_mod.dcap_qvl, "get_collateral", must_not_fetch)
    with pytest.raises(TeeError, match="collateral_provider"):
        await DcapTdxVerifier.with_fetched_collateral(
            _raw_quote(_PROVIDER_SLUG), collateral_provider=_Provider()
        )


def test_provider_supplies_collateral_per_quote() -> None:
    provider = _Provider(_PROVIDER_SLUG)
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    verifier = DcapTdxVerifier(collateral_provider=provider, now_secs=CAPTURE_EPOCH)

    assert verifier.verify(att) is True

    assert len(provider.calls) == 1
    key = provider.calls[0][0]
    assert key.fmspc == "90C06F000000"
    assert key == CollateralKey.from_quote(_raw_quote(_PROVIDER_SLUG))
    assert verifier.last_result is not None
    assert verifier.last_result["collateral_key"] == key
    assert verifier.last_result["checks"]["signature_chain"] is True


def test_provider_routes_two_fmspcs_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(_PROVIDER_SLUG, _OTHER_FMSPC)
    verifier = DcapTdxVerifier(collateral_provider=provider)
    seen: list[CollateralKey] = []

    for slug, epoch in ((_PROVIDER_SLUG, CAPTURE_EPOCH), (_OTHER_FMSPC, EVIDENCE_CAPTURE_EPOCH)):
        monkeypatch.setattr(verify_mod.time, "time", lambda epoch=epoch: float(epoch))
        att, _nonce = _build_attestation(slug)
        assert verifier.verify(att) is True
        assert verifier.last_result is not None
        assert verifier.last_result["checks"]["signature_chain"] is True
        seen.append(verifier.last_result["collateral_key"])
        assert verifier._collateral is None

    assert seen[0] != seen[1]
    assert seen[0].fmspc != seen[1].fmspc
    assert [call[0] for call in provider.calls] == seen
    assert [call[1] for call in provider.calls] == [CAPTURE_EPOCH, EVIDENCE_CAPTURE_EPOCH]


def test_provider_receives_verifier_clock() -> None:
    provider = _Provider(_PROVIDER_SLUG)
    att, _nonce = _build_attestation(_PROVIDER_SLUG)

    DcapTdxVerifier(collateral_provider=provider, now_secs=CAPTURE_EPOCH).verify(att)

    assert provider.calls[0][1] == CAPTURE_EPOCH


def test_store_key_matches_miss_key() -> None:
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    raw = bytes.fromhex(att.intel_quote)
    verifier = DcapTdxVerifier(collateral_provider=lambda _k, _n: None, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeCollateralUnavailableError) as excinfo:
        verifier.verify(att)

    assert CollateralKey.from_quote(raw) == excinfo.value.key
    assert CollateralKey.from_quote(excinfo.value.raw_quote) == excinfo.value.key


def test_provider_cache_miss_raises_unavailable() -> None:
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    verifier = DcapTdxVerifier(collateral_provider=_Provider(), now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeCollateralUnavailableError) as excinfo:
        verifier.verify(att)

    err = excinfo.value
    assert isinstance(err, TeeAttestationError)
    assert err.key.fmspc == "90C06F000000"
    assert err.raw_quote == bytes.fromhex(att.intel_quote)
    assert verifier.last_result is None


def test_provider_wrong_collateral_raises_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(*_args: Any) -> None:
        pytest.fail("verified with collateral for another FMSPC")

    monkeypatch.setattr(verify_mod.dcap_qvl, "verify_with_root_ca", must_not_run)
    wrong = _collateral(_OTHER_FMSPC)
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    verifier = DcapTdxVerifier(collateral_provider=lambda _k, _n: wrong, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeCollateralMismatchError) as excinfo:
        verifier.verify(att)

    assert excinfo.value.reason == "fmspc"
    assert excinfo.value.quote_key == CollateralKey.from_quote(bytes.fromhex(att.intel_quote))


def test_provider_arbitrary_exception_fails_closed() -> None:
    def broken(_key: CollateralKey, _now: int) -> Any:
        raise KeyError("cache backend down")

    raw = load_attestation(_PROVIDER_SLUG)
    att = TeeAttestation.model_validate(raw["attestation"])
    verifier = DcapTdxVerifier(collateral_provider=broken, now_secs=CAPTURE_EPOCH)

    with pytest.raises(KeyError):
        verifier.verify(att)

    with pytest.raises(TeeAttestationError) as excinfo:
        verify_attestation(att, raw["_client_nonce"], verifier=verifier)
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_provider_quote_without_chain_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoChainQuote(_FakeQuote):
        def fmspc(self) -> str:
            raise ValueError("no embedded PCK chain")

    def must_not_call(_key: CollateralKey, _now: int) -> Any:
        pytest.fail("collateral_provider called without a key")

    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    report = _real_report(slug)
    _stub_crypto(monkeypatch, report)
    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda raw: _NoChainQuote(report))
    verifier = DcapTdxVerifier(collateral_provider=must_not_call, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeAttestationError, match="collateral_provider"):
        verifier.verify(att)


def test_provider_quote_with_unreadable_chain_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadChainQuote(_FakeQuote):
        def cert_chain_pem_bytes(self) -> bytes:
            return b"-----BEGIN CERTIFICATE-----!!!-----END CERTIFICATE-----" * 2

        def fmspc(self) -> str:
            return "90c06f000000"

    def must_not_call(_key: CollateralKey, _now: int) -> Any:
        pytest.fail("collateral_provider called without a key")

    slug = "gemma_3_27b_p"
    att, _ = _build_attestation(slug)
    report = _real_report(slug)
    _stub_crypto(monkeypatch, report)
    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda raw: _BadChainQuote(report))
    verifier = DcapTdxVerifier(collateral_provider=must_not_call, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeAttestationError, match="collateral_provider"):
        verifier.verify(att)


def test_static_collateral_records_collateral_key() -> None:
    att, _nonce = _build_attestation(_MACHINE_B)
    verifier = _verifier(_MACHINE_A)

    assert verifier.verify(att) is True

    assert verifier.last_result is not None
    assert verifier.last_result["collateral_key"] == CollateralKey.from_quote(
        bytes.fromhex(att.intel_quote)
    )


def test_neither_collateral_nor_provider_keeps_refusal_message() -> None:
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    with pytest.raises(TeeError, match="No Intel-signed collateral available"):
        DcapTdxVerifier(now_secs=CAPTURE_EPOCH).verify(att)


def _must_not_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(*_args: Any) -> None:
        pytest.fail("reached the crypto gate")

    monkeypatch.setattr(verify_mod.dcap_qvl, "verify_with_root_ca", must_not_run)


def _swapped_pck_ca_collateral(slug: str) -> Any:
    stored = json.loads(load_collateral_json(slug))
    other = json.loads(load_collateral_json(_OTHER_FMSPC))
    swapped = _PEM_CERTIFICATE.findall(other["tcb_info_issuer_chain"])[0]
    rest = _PEM_CERTIFICATE.findall(stored["pck_crl_issuer_chain"])[1:]
    stored["pck_crl_issuer_chain"] = "".join(
        f"-----BEGIN CERTIFICATE-----{body}-----END CERTIFICATE-----\n" for body in [swapped, *rest]
    )
    return dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))


def test_provider_wrong_pck_ca_raises_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _must_not_verify(monkeypatch)
    wrong = _swapped_pck_ca_collateral(_PROVIDER_SLUG)
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    verifier = DcapTdxVerifier(collateral_provider=lambda _k, _n: wrong, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeCollateralMismatchError) as excinfo:
        verifier.verify(att)

    err = excinfo.value
    assert err.reason == "pck_ca"
    assert err.quote_key == CollateralKey.from_quote(bytes.fromhex(att.intel_quote))
    assert err.collateral_key.fmspc == err.quote_key.fmspc
    assert err.collateral_key.pck_ca_sha256 != err.quote_key.pck_ca_sha256


@pytest.mark.parametrize(
    ("wrong", "reason"),
    [
        (lambda: _collateral(_OTHER_FMSPC), "fmspc"),
        (lambda: _swapped_pck_ca_collateral(_PROVIDER_SLUG), "pck_ca"),
    ],
)
def test_provider_mismatch_hint_points_at_the_store(
    monkeypatch: pytest.MonkeyPatch, wrong: Any, reason: str
) -> None:
    _must_not_verify(monkeypatch)
    collateral = wrong()
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    verifier = DcapTdxVerifier(
        collateral_provider=lambda _k, _n: collateral, now_secs=CAPTURE_EPOCH
    )

    with pytest.raises(TeeCollateralMismatchError) as excinfo:
        verifier.verify(att)

    message = str(excinfo.value)
    assert excinfo.value.reason == reason
    assert "collateral_provider returned collateral that does not match the key" in message
    assert "check how its store is keyed" in message
    assert "with_fetched_collateral" not in message


def test_static_mismatch_hint_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _must_not_verify(monkeypatch)
    att, _nonce = _build_attestation(_OTHER_FMSPC)

    with pytest.raises(TeeCollateralMismatchError) as excinfo:
        _verifier(_MACHINE_A).verify(att)

    assert str(excinfo.value).endswith(
        "Build a verifier for it with DcapTdxVerifier.with_fetched_collateral(...) (fail-closed)."
    )
    assert "collateral_provider" not in str(excinfo.value)


class _HasToJson:
    def to_json(self) -> str:
        return "{}"


@pytest.mark.parametrize("returned", [object(), {"tcb_info": "{}"}, "collateral", _HasToJson()])
def test_provider_non_collateral_return_refused(
    monkeypatch: pytest.MonkeyPatch, returned: Any
) -> None:
    _must_not_verify(monkeypatch)
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    verifier = DcapTdxVerifier(collateral_provider=lambda _k, _n: returned, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeAttestationError, match="QuoteCollateralV3"):
        verifier.verify(att)
    assert verifier.last_result is None


def test_provider_leaf_only_chain_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    real = dcap_qvl.parse_quote(bytes.fromhex(att.intel_quote))
    leaf = _PEM_CERTIFICATE.findall(real.cert_chain_pem_bytes().decode("ascii"))[0]

    class _LeafOnlyQuote(_FakeQuote):
        def cert_chain_pem_bytes(self) -> bytes:
            return f"-----BEGIN CERTIFICATE-----{leaf}-----END CERTIFICATE-----\n".encode()

        def fmspc(self) -> str:
            return real.fmspc()

    def must_not_call(_key: CollateralKey, _now: int) -> Any:
        pytest.fail("collateral_provider called without a key")

    report = _real_report(_PROVIDER_SLUG)
    _stub_crypto(monkeypatch, report)
    monkeypatch.setattr(verify_mod.dcap_qvl, "parse_quote", lambda raw: _LeafOnlyQuote(report))
    verifier = DcapTdxVerifier(collateral_provider=must_not_call, now_secs=CAPTURE_EPOCH)

    with pytest.raises(TeeAttestationError, match="no PCK CA certificate in its embedded chain"):
        verifier.verify(att)


class _FmspcBrokenQuote:
    """A real parsed quote whose ``fmspc()`` raises something other than ``ValueError``."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def fmspc(self) -> str:
        raise RuntimeError("fmspc unavailable")


def test_static_key_failure_still_raises_tee_error(monkeypatch: pytest.MonkeyPatch) -> None:
    att, _nonce = _build_attestation(_MACHINE_A)
    real_parse = dcap_qvl.parse_quote
    monkeypatch.setattr(
        verify_mod.dcap_qvl, "parse_quote", lambda raw: _FmspcBrokenQuote(real_parse(raw))
    )

    with pytest.raises(TeeError, match="could not be read"):
        _verifier(_MACHINE_A).verify(att)


def test_provider_key_failure_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    att, _nonce = _build_attestation(_PROVIDER_SLUG)
    real_parse = dcap_qvl.parse_quote
    monkeypatch.setattr(
        verify_mod.dcap_qvl, "parse_quote", lambda raw: _FmspcBrokenQuote(real_parse(raw))
    )

    def must_not_call(_key: CollateralKey, _now: int) -> Any:
        pytest.fail("collateral_provider called without a key")

    verifier = DcapTdxVerifier(collateral_provider=must_not_call, now_secs=CAPTURE_EPOCH)
    with pytest.raises(TeeAttestationError, match="could not be read") as excinfo:
        verifier.verify(att)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# --------------------------------------------------------------------------- #
# Expired collateral
# --------------------------------------------------------------------------- #

_EXPIRY_SLUG = "glm_5_1"


def _fixture_next_updates() -> dict[str, datetime]:
    from cryptography import x509

    stored = json.loads(load_collateral_json(_EXPIRY_SLUG))
    out: dict[str, datetime] = {}
    for item in ("tcb_info", "qe_identity"):
        text = json.loads(stored[item])["nextUpdate"]
        out[item] = datetime.fromisoformat(text.replace("Z", "+00:00"))
    for item in ("pck_crl", "root_ca_crl"):
        crl = x509.load_der_x509_crl(bytes.fromhex(stored[item]))
        out[item] = crl.next_update_utc
    return out


def test_expired_collateral_is_typed() -> None:
    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    now = int(next_updates["pck_crl"].timestamp()) + 1

    with pytest.raises(TeeCollateralExpiredError) as excinfo:
        _verifier(_EXPIRY_SLUG, now_secs=now).verify(att)

    err = excinfo.value
    assert err.item == "pck_crl"
    assert err.next_update == next_updates["pck_crl"]
    assert err.key is not None
    assert err.key.fmspc == "90C06F000000"
    assert err.raw_quote


def test_expired_reports_earliest_item() -> None:
    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    now = int(datetime(2026, 7, 4, 0, 30, tzinfo=UTC).timestamp())
    assert next_updates["pck_crl"] < next_updates["qe_identity"]
    assert next_updates["qe_identity"].timestamp() < now

    with pytest.raises(TeeCollateralExpiredError) as excinfo:
        _verifier(_EXPIRY_SLUG, now_secs=now).verify(att)

    assert excinfo.value.item == "pck_crl"


def test_expired_check_uses_provider_collateral() -> None:
    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    now = int(next_updates["pck_crl"].timestamp()) + 1
    verifier = DcapTdxVerifier(
        collateral_provider=lambda _key, _now: _collateral(_EXPIRY_SLUG), now_secs=now
    )

    with pytest.raises(TeeCollateralExpiredError) as excinfo:
        verifier.verify(att)

    assert excinfo.value.item == "pck_crl"
    assert excinfo.value.key is not None


def test_collateral_within_window_verifies() -> None:
    att, _nonce = _build_attestation(_EXPIRY_SLUG)

    assert _verifier(_EXPIRY_SLUG).verify(att) is True


def test_unparseable_collateral_dates_skip_typed_expiry() -> None:
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    stored = json.loads(load_collateral_json(_EXPIRY_SLUG))
    tcb_info = json.loads(stored["tcb_info"])
    tcb_info["nextUpdate"] = "garbage"
    stored["tcb_info"] = json.dumps(tcb_info)
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))

    with pytest.raises(TeeError) as excinfo:
        DcapTdxVerifier(collateral=collateral, now_secs=CAPTURE_EPOCH).verify(att)

    assert not isinstance(excinfo.value, TeeCollateralExpiredError)


def test_root_ca_crl_expiry_is_reported() -> None:
    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    stored = json.loads(load_collateral_json(_EXPIRY_SLUG))
    stored["pck_crl"] = "00"
    for item in ("tcb_info", "qe_identity"):
        body = json.loads(stored[item])
        body["nextUpdate"] = "2099-01-01T00:00:00Z"
        stored[item] = json.dumps(body)
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))
    now = int(next_updates["root_ca_crl"].timestamp()) + 1

    with pytest.raises(TeeCollateralExpiredError) as excinfo:
        DcapTdxVerifier(collateral=collateral, now_secs=now).verify(att)

    assert excinfo.value.item == "root_ca_crl"
    assert excinfo.value.next_update == next_updates["root_ca_crl"]


def test_crl_is_expired_at_its_exact_next_update() -> None:
    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    now = int(next_updates["pck_crl"].timestamp())

    with pytest.raises(TeeCollateralExpiredError) as excinfo:
        _verifier(_EXPIRY_SLUG, now_secs=now).verify(att)

    assert excinfo.value.item == "pck_crl"


@pytest.mark.parametrize("item", ["tcb_info", "qe_identity"])
def test_json_item_is_not_typed_expired_at_its_exact_next_update(item: str) -> None:
    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    stored = json.loads(load_collateral_json(_EXPIRY_SLUG))
    earliest = next_updates["pck_crl"] - timedelta(days=1)
    for name in ("tcb_info", "qe_identity"):
        body = json.loads(stored[name])
        body["nextUpdate"] = (
            earliest.strftime("%Y-%m-%dT%H:%M:%SZ") if name == item else "2099-01-01T00:00:00Z"
        )
        stored[name] = json.dumps(body)
    collateral = dcap_qvl.QuoteCollateralV3.from_json(json.dumps(stored))

    # The edited JSON breaks its Intel signature, so dcap refuses; only the typed
    # expiry must stay out at the exact second.
    with pytest.raises(TeeError) as excinfo:
        DcapTdxVerifier(collateral=collateral, now_secs=int(earliest.timestamp())).verify(att)

    assert not isinstance(excinfo.value, TeeCollateralExpiredError)


def test_json_item_expiry_is_still_typed_when_crl_parsing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cryptography import x509

    next_updates = _fixture_next_updates()
    att, _nonce = _build_attestation(_EXPIRY_SLUG)
    now = int(datetime(2026, 7, 4, 0, 30, tzinfo=UTC).timestamp())
    assert next_updates["pck_crl"] < next_updates["qe_identity"]

    def boom(_data: bytes) -> Any:
        raise ValueError("CRL parsing unavailable")

    monkeypatch.setattr(x509, "load_der_x509_crl", boom)

    with pytest.raises(TeeCollateralExpiredError) as excinfo:
        _verifier(_EXPIRY_SLUG, now_secs=now).verify(att)

    json_items = {k: v for k, v in next_updates.items() if k in ("tcb_info", "qe_identity")}
    expected = min(json_items, key=lambda k: json_items[k])
    assert excinfo.value.item == expected
    assert excinfo.value.next_update == json_items[expected]
