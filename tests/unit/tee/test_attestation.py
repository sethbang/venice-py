"""Unit tests for baseline TEE attestation verification.

The fixture ``fixtures/attestation_gemma.json`` is a real ``GET /tee/attestation``
response captured live (2026-06-04, ``e2ee-gemma-3-27b-p``); only the giant raw
``intel_quote`` / ``nvidia_payload`` blobs were truncated for file size. Its
``server_verification.tdx.reportData`` is the genuine
``signing_address || zero-pad || nonce`` binding, so the happy path exercises
the real wire shape.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from venice_ai.exceptions import TeeAttestationError, TeeTcbStatusError
from venice_ai.tee._attestation import (
    FullQuoteVerifier,
    preflight_nonce,
    verify_attestation,
)
from venice_ai.tee.types import TeeAttestation, TeeVerificationResult

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "attestation_gemma.json"


@pytest.fixture
def raw_attestation() -> dict[str, Any]:
    """A fresh deep copy of the live attestation dict for each test."""
    return json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture
def sent_nonce(raw_attestation: dict[str, Any]) -> str:
    """The client-generated nonce that produced the fixture (its echo)."""
    nonce = raw_attestation["request_nonce"]
    assert isinstance(nonce, str)
    return nonce


def _build(raw: dict[str, Any]) -> TeeAttestation:
    return TeeAttestation.model_validate(raw)


# --- A1: fixture sanity ------------------------------------------------------


def test_fixture_is_real_shape(raw_attestation: dict[str, Any]) -> None:
    sv = raw_attestation["server_verification"]
    assert raw_attestation["verified"] is True
    assert raw_attestation["signing_algo"] == "ecdsa"
    assert len(raw_attestation["signing_public_key"]) == 130
    assert "reportData" in sv["tdx"]
    assert "tdAttributes" in sv["tdx"]["measurements"]


# --- A2: happy path ----------------------------------------------------------


def test_happy_path_passes_baseline(raw_attestation: dict[str, Any], sent_nonce: str) -> None:
    att = _build(raw_attestation)
    result = verify_attestation(att, sent_nonce)
    assert isinstance(result, TeeVerificationResult)
    assert result.ok is True
    assert result.failures == []
    assert result.checks == {
        "verified": True,
        "nonce_echo": True,
        "reportdata_binding": True,
        "debug_flag": True,
        "key_binding": True,
    }
    # signing_address normalized (lowercase, no 0x); model key decoded to 65 bytes.
    assert result.signing_address == raw_attestation["signing_address"].lower().removeprefix("0x")
    assert result.model_public_key is not None
    assert len(result.model_public_key) == 65
    assert result.model_public_key[0] == 0x04


# --- A3: nonce echo mismatch -------------------------------------------------


def test_nonce_echo_mismatch_raises(raw_attestation: dict[str, Any]) -> None:
    # Supply a DIFFERENT client nonce than the fixture's echo. This is only
    # constructible because verify takes an externally-supplied sent_nonce.
    att = _build(raw_attestation)
    other_nonce = "f" * 64
    with pytest.raises(TeeAttestationError, match="nonce echo"):
        verify_attestation(att, other_nonce)


# --- A4: REPORTDATA binding (flip the address portion, keep nonce intact) ----


def test_flipped_reportdata_address_byte_raises(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raw = copy.deepcopy(raw_attestation)
    rd = raw["server_verification"]["tdx"]["reportData"]
    # Flip the first nibble of the address portion; leave nonce echo untouched
    # so the nonce-echo check passes and the binding check is what fires.
    first = "0" if rd[0] != "0" else "1"
    raw["server_verification"]["tdx"]["reportData"] = first + rd[1:]
    att = _build(raw)
    with pytest.raises(TeeAttestationError, match="report-data binding mismatch"):
        verify_attestation(att, sent_nonce)


def test_missing_reportdata_fails_closed(raw_attestation: dict[str, Any], sent_nonce: str) -> None:
    # Absence of evidence must fail closed, not silently pass.
    raw = copy.deepcopy(raw_attestation)
    del raw["server_verification"]["tdx"]["reportData"]
    att = _build(raw)
    with pytest.raises(TeeAttestationError, match="reportData"):
        verify_attestation(att, sent_nonce)


def test_missing_server_verification_fails_closed(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raw = copy.deepcopy(raw_attestation)
    raw["server_verification"] = None
    att = _build(raw)
    with pytest.raises(TeeAttestationError, match="server_verification"):
        verify_attestation(att, sent_nonce)


# --- A5: verified:false ------------------------------------------------------


def test_verified_false_raises(raw_attestation: dict[str, Any], sent_nonce: str) -> None:
    raw = copy.deepcopy(raw_attestation)
    raw["verified"] = False
    att = _build(raw)
    with pytest.raises(TeeAttestationError, match="verified=false"):
        verify_attestation(att, sent_nonce)


# --- A6: TDX debug flag ------------------------------------------------------


def test_debug_flag_set_raises(raw_attestation: dict[str, Any], sent_nonce: str) -> None:
    raw = copy.deepcopy(raw_attestation)
    # Set bit 0 (TUD.DEBUG) of the little-endian tdAttributes: first byte 0x01.
    raw["server_verification"]["tdx"]["measurements"]["tdAttributes"] = "0100001000000000"
    att = _build(raw)
    with pytest.raises(TeeAttestationError, match="DEBUG mode"):
        verify_attestation(att, sent_nonce)


def test_missing_td_attributes_fails_closed(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raw = copy.deepcopy(raw_attestation)
    del raw["server_verification"]["tdx"]["measurements"]["tdAttributes"]
    att = _build(raw)
    with pytest.raises(TeeAttestationError, match="tdAttributes"):
        verify_attestation(att, sent_nonce)


# --- A7: nonce preflight ------------------------------------------------------


def test_preflight_rejects_short_nonce() -> None:
    with pytest.raises(TeeAttestationError, match="32 bytes"):
        preflight_nonce("ab" * 16)  # 16 bytes / 32 hex


def test_preflight_rejects_non_hex_nonce() -> None:
    with pytest.raises(TeeAttestationError, match="hexadecimal"):
        preflight_nonce("z" * 64)


def test_preflight_lowercases_valid_nonce() -> None:
    upper = "AB" * 32
    assert preflight_nonce(upper) == upper.lower()


def test_verify_rejects_short_nonce(raw_attestation: dict[str, Any]) -> None:
    att = _build(raw_attestation)
    with pytest.raises(TeeAttestationError, match="32 bytes"):
        verify_attestation(att, "ab" * 16)


# --- A8: fail_closed=False ----------------------------------------------------


def test_fail_closed_false_returns_not_ok_and_warns(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raw = copy.deepcopy(raw_attestation)
    raw["verified"] = False
    att = _build(raw)
    with pytest.warns(UserWarning, match="fail_closed=False"):
        result = verify_attestation(att, sent_nonce, fail_closed=False)
    assert result.ok is False
    assert result.checks["verified"] is False
    assert any("verified=false" in f for f in result.failures)
    # Other checks still ran and passed.
    assert result.checks["nonce_echo"] is True
    assert result.checks["reportdata_binding"] is True


def test_fail_closed_false_collects_multiple_failures(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raw = copy.deepcopy(raw_attestation)
    raw["verified"] = False
    raw["server_verification"]["tdx"]["measurements"]["tdAttributes"] = "0100000000000000"
    att = _build(raw)
    with pytest.warns(UserWarning):
        result = verify_attestation(att, sent_nonce, fail_closed=False)
    assert result.ok is False
    assert result.checks["verified"] is False
    assert result.checks["debug_flag"] is False
    assert len(result.failures) >= 2


# --- FullQuoteVerifier extension point ---------------------------------------


def test_full_quote_verifier_invoked_on_pass(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    calls: list[TeeAttestation] = []

    class _Verifier:
        def verify(self, attestation: TeeAttestation) -> bool:
            calls.append(attestation)
            return True

    att = _build(raw_attestation)
    result = verify_attestation(att, sent_nonce, verifier=_Verifier())
    assert result.ok is True
    assert result.checks["full_quote"] is True
    assert len(calls) == 1
    assert isinstance(_Verifier(), FullQuoteVerifier)
    # Retained raw evidence must reach the extension point for a future verifier.
    assert calls[0].intel_quote is not None
    assert calls[0].nvidia_payload is not None


def test_full_quote_verifier_false_raises(raw_attestation: dict[str, Any], sent_nonce: str) -> None:
    class _Verifier:
        def verify(self, attestation: TeeAttestation) -> bool:
            return False

    att = _build(raw_attestation)
    with pytest.raises(TeeAttestationError, match="returned False"):
        verify_attestation(att, sent_nonce, verifier=_Verifier())


def test_full_quote_verifier_not_run_when_baseline_fails(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raw = copy.deepcopy(raw_attestation)
    raw["verified"] = False
    att = _build(raw)

    class _Verifier:
        called = False

        def verify(self, attestation: TeeAttestation) -> bool:  # pragma: no cover
            type(self).called = True
            return True

    v = _Verifier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verify_attestation(att, sent_nonce, fail_closed=False, verifier=v)
    assert _Verifier.called is False


def _tcb_error() -> TeeTcbStatusError:
    return TeeTcbStatusError("x", status="OutOfDate", advisory_ids=[], policy="reject")


def test_typed_verifier_error_propagates_unwrapped(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    raised = _tcb_error()

    class _Verifier:
        def verify(self, attestation: TeeAttestation) -> bool:
            raise raised

    att = _build(raw_attestation)
    with pytest.raises(TeeTcbStatusError) as excinfo:
        verify_attestation(att, sent_nonce, verifier=_Verifier())
    assert excinfo.value is raised
    assert excinfo.value.status == "OutOfDate"


def test_untyped_verifier_error_still_wrapped(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    boom = ValueError("boom")

    class _Verifier:
        def verify(self, attestation: TeeAttestation) -> bool:
            raise boom

    att = _build(raw_attestation)
    with pytest.raises(TeeAttestationError) as excinfo:
        verify_attestation(att, sent_nonce, verifier=_Verifier())
    assert type(excinfo.value) is TeeAttestationError
    assert str(excinfo.value).startswith("TEE full-quote verifier raised:")
    assert excinfo.value.__cause__ is boom


def test_typed_error_not_raised_when_not_fail_closed(
    raw_attestation: dict[str, Any], sent_nonce: str
) -> None:
    class _Verifier:
        def verify(self, attestation: TeeAttestation) -> bool:
            raise _tcb_error()

    att = _build(raw_attestation)
    with pytest.warns(UserWarning):
        result = verify_attestation(att, sent_nonce, fail_closed=False, verifier=_Verifier())
    assert result.ok is False
    assert result.checks["full_quote"] is False
    assert any("Full-quote verifier raised" in f for f in result.failures)
