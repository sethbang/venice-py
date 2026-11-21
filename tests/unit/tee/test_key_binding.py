"""The model encryption key must derive to the attested signing address.

REPORTDATA binds ``signing_address || zero-pad || nonce``; it does not bind
``signing_public_key``. Requests are encrypted to ``signing_public_key``, so
:func:`verify_attestation` checks that
``keccak256(signing_public_key[1:])[-20:] == signing_address``. Without that
check a relayed genuine quote, address and nonce with a substituted key would
pass both the baseline and a full-quote verifier.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="key-binding tests generate keys with the [e2ee] extra (cryptography)",
)
pytest.importorskip("Crypto.Hash.keccak", reason="key binding needs the [e2ee] extra")

from venice_ai.exceptions import TeeAttestationError, TeeEncryptionError  # noqa: E402
from venice_ai.resources.tee import Tee  # noqa: E402
from venice_ai.tee import _attestation, _crypto  # noqa: E402
from venice_ai.tee._attestation import verify_attestation  # noqa: E402
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.tee.types import TeeAttestation  # noqa: E402

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "attestation_gemma.json"

_KECCAK256_EMPTY = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"


@pytest.fixture
def raw() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture
def nonce(raw: dict[str, Any]) -> str:
    value = raw["request_nonce"]
    assert isinstance(value, str)
    return value


def _attacker_key_hex() -> str:
    return _crypto.uncompressed_hex(ec.generate_private_key(ec.SECP256K1()).public_key())


def _substituted(raw: dict[str, Any]) -> dict[str, Any]:
    """The genuine fixture with a valid, on-curve key that is not the attested one."""
    tampered = dict(raw)
    tampered["signing_public_key"] = _attacker_key_hex()
    return tampered


class _AcceptingVerifier:
    """A full-quote verifier stub that accepts everything."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, attestation: TeeAttestation) -> bool:
        self.calls += 1
        return True


def _tee_returning(raw: dict[str, Any]) -> Tee:
    client = MagicMock()
    client.get = AsyncMock(side_effect=lambda *a, **k: TeeAttestation.model_validate(raw))
    return Tee(client)


# --- Keccak-256 --------------------------------------------------------------


def test_keccak256_known_answer() -> None:
    assert _attestation._keccak256(b"").hex() == _KECCAK256_EMPTY


def test_keccak256_is_not_sha3() -> None:
    # SHA3-256 pads differently; using it would reject every genuine attestation.
    assert hashlib.sha3_256(b"").hexdigest() != _KECCAK256_EMPTY


def test_fixture_key_derives_to_fixture_address(raw: dict[str, Any]) -> None:
    key = bytes.fromhex(raw["signing_public_key"])
    derived = _attestation._keccak256(key[1:])[-20:].hex()
    assert derived == raw["signing_address"].lower().removeprefix("0x")


# --- verify_attestation ------------------------------------------------------


def test_genuine_fixture_passes_key_binding(raw: dict[str, Any], nonce: str) -> None:
    result = verify_attestation(TeeAttestation.model_validate(raw), nonce)
    assert result.ok is True
    assert result.checks["key_binding"] is True


@pytest.mark.parametrize(
    "address_form",
    [
        lambda a: a.lower(),
        lambda a: a.upper().replace("0X", "0x"),
        lambda a: a.lower().removeprefix("0x"),
        lambda a: a.upper().removeprefix("0X"),
    ],
    ids=["lower", "upper", "lower-no-0x", "upper-no-0x"],
)
def test_address_case_and_prefix_variations_pass(
    raw: dict[str, Any], nonce: str, address_form: Any
) -> None:
    raw["signing_address"] = address_form(raw["signing_address"])
    result = verify_attestation(TeeAttestation.model_validate(raw), nonce)
    assert result.checks["key_binding"] is True


@pytest.mark.parametrize(
    "key_form",
    [
        lambda k: k.upper(),
        lambda k: "0x" + k,
        lambda k: k[2:],
    ],
    ids=["upper", "0x-prefixed", "bare-point-body"],
)
def test_key_encoding_variations_pass(raw: dict[str, Any], nonce: str, key_form: Any) -> None:
    raw["signing_public_key"] = key_form(raw["signing_public_key"])
    result = verify_attestation(TeeAttestation.model_validate(raw), nonce)
    assert result.checks["key_binding"] is True


def test_substituted_key_fails_closed(raw: dict[str, Any], nonce: str) -> None:
    with pytest.raises(
        TeeAttestationError, match="does not derive to the attested signing_address"
    ):
        verify_attestation(TeeAttestation.model_validate(_substituted(raw)), nonce)


def test_substituted_key_fails_with_full_quote_verifier(raw: dict[str, Any], nonce: str) -> None:
    verifier = _AcceptingVerifier()
    with pytest.raises(
        TeeAttestationError, match="does not derive to the attested signing_address"
    ):
        verify_attestation(
            TeeAttestation.model_validate(_substituted(raw)), nonce, verifier=verifier
        )
    assert verifier.calls == 0


def test_substituted_key_error_quotes_no_key_material(raw: dict[str, Any], nonce: str) -> None:
    tampered = _substituted(raw)
    with pytest.raises(TeeAttestationError) as info:
        verify_attestation(TeeAttestation.model_validate(tampered), nonce)
    assert tampered["signing_public_key"] not in str(info.value)
    assert tampered["signing_public_key"][2:20] not in str(info.value)


def test_substituted_key_recorded_when_not_fail_closed(raw: dict[str, Any], nonce: str) -> None:
    with pytest.warns(UserWarning, match="signing_address"):
        result = verify_attestation(
            TeeAttestation.model_validate(_substituted(raw)), nonce, fail_closed=False
        )
    assert result.ok is False
    assert result.checks["key_binding"] is False


@pytest.mark.parametrize(
    "bad_key",
    ["zz" * 65, "04" + "ab" * 10, "05" + "ab" * 64, "04" + "ab" * 65],
    ids=["non-hex", "short", "bad-prefix", "long"],
)
def test_malformed_key_fails_closed(raw: dict[str, Any], nonce: str, bad_key: str) -> None:
    raw["signing_public_key"] = bad_key
    with pytest.raises(TeeAttestationError, match="signing_public_key is malformed"):
        verify_attestation(TeeAttestation.model_validate(raw), nonce)


def test_absent_key_skips_key_binding(raw: dict[str, Any], nonce: str) -> None:
    raw["signing_public_key"] = None
    result = verify_attestation(TeeAttestation.model_validate(raw), nonce)
    assert result.ok is True
    assert "key_binding" not in result.checks


def test_missing_pycryptodome_raises_install_hint(
    raw: dict[str, Any], nonce: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "Crypto", None)
    monkeypatch.setitem(sys.modules, "Crypto.Hash", None)
    monkeypatch.setitem(sys.modules, "Crypto.Hash.keccak", None)
    with pytest.raises(ImportError, match=r"pip install 'venice-py\[e2ee\]'"):
        verify_attestation(TeeAttestation.model_validate(raw), nonce)


# --- client.tee --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("with_verifier", [False, True], ids=["baseline", "full-quote"])
async def test_get_attestation_refuses_substituted_key(
    raw: dict[str, Any], nonce: str, with_verifier: bool
) -> None:
    tee = _tee_returning(_substituted(raw))
    verifier = _AcceptingVerifier() if with_verifier else None
    with pytest.raises(TeeAttestationError, match="signing_address"):
        await tee.get_attestation(model="e2ee-x", nonce=nonce, fail_closed=True, verifier=verifier)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_verifier", [False, True], ids=["baseline", "full-quote"])
async def test_open_session_refuses_substituted_key(
    raw: dict[str, Any], nonce: str, with_verifier: bool
) -> None:
    tee = _tee_returning(_substituted(raw))
    verifier = _AcceptingVerifier() if with_verifier else None
    with pytest.raises(TeeAttestationError, match="signing_address"):
        await tee.open_session(model="e2ee-x", nonce=nonce, verifier=verifier)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_verifier", [False, True], ids=["baseline", "full-quote"])
async def test_open_session_accepts_genuine_fixture(
    raw: dict[str, Any], nonce: str, with_verifier: bool
) -> None:
    tee = _tee_returning(raw)
    verifier = _AcceptingVerifier() if with_verifier else None
    session = await tee.open_session(model="e2ee-x", nonce=nonce, verifier=verifier)
    with session:
        assert session.trust is not None
        assert session.trust.model_public_key == raw["signing_public_key"]
        assert session.trust.verification == ("full_quote" if with_verifier else "baseline")


# --- TeeSession.from_attestation ---------------------------------------------


def test_from_attestation_builds_a_session_from_the_genuine_fixture(raw: dict[str, Any]) -> None:
    session = TeeSession.from_attestation(TeeAttestation.model_validate(raw))
    with session:
        assert session.model_public_key_hex == raw["signing_public_key"]


def test_from_attestation_refuses_a_substituted_key(raw: dict[str, Any]) -> None:
    tampered = _substituted(raw)
    with pytest.raises(
        TeeAttestationError, match="does not derive to the attested signing_address"
    ) as info:
        TeeSession.from_attestation(TeeAttestation.model_validate(tampered))
    assert tampered["signing_public_key"] not in str(info.value)
    assert tampered["signing_public_key"][2:20] not in str(info.value)


@pytest.mark.parametrize(
    "bad_key",
    ["zz" * 65, "04" + "ab" * 10, "05" + "ab" * 64, "04" + "ab" * 65],
    ids=["non-hex", "short", "bad-prefix", "long"],
)
def test_from_attestation_refuses_a_malformed_key(raw: dict[str, Any], bad_key: str) -> None:
    raw["signing_public_key"] = bad_key
    with pytest.raises(TeeAttestationError, match="signing_public_key is malformed"):
        TeeSession.from_attestation(TeeAttestation.model_validate(raw))


def test_from_attestation_absent_key_still_raises_encryption_error(raw: dict[str, Any]) -> None:
    raw["signing_public_key"] = None
    with pytest.raises(TeeEncryptionError, match="signing_public_key"):
        TeeSession.from_attestation(TeeAttestation.model_validate(raw))


def test_from_attestation_refuses_before_generating_a_session_key(
    raw: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    generate = MagicMock(side_effect=AssertionError("session key generated"))
    monkeypatch.setattr(_crypto, "generate_session_keypair", generate)
    with pytest.raises(TeeAttestationError):
        TeeSession.from_attestation(TeeAttestation.model_validate(_substituted(raw)))
    generate.assert_not_called()


@pytest.mark.asyncio
async def test_from_attestation_refuses_a_key_accepted_without_fail_closed(
    raw: dict[str, Any], nonce: str
) -> None:
    tee = _tee_returning(_substituted(raw))
    with pytest.warns(UserWarning, match="signing_address"):
        attestation = await tee.get_attestation(model="e2ee-x", nonce=nonce, fail_closed=False)
    with pytest.raises(TeeAttestationError, match="signing_address"):
        TeeSession.from_attestation(attestation)
