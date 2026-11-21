"""Scaffold-phase tests for the ``venice_ai.tee`` package.

These verify the package imports on a *bare* install, that the typed
exceptions exist with the right MRO, that the confirmed protocol constants
have their live-verified values, and that :class:`TeeAttestation` parses a
**full** live attestation payload (extra keys retained, not rejected).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import venice_ai.tee as tee
from venice_ai.exceptions import VeniceError
from venice_ai.tee import (
    TeeAttestationError,
    TeeEncryptionError,
    TeeError,
)

# A complete attestation payload as returned live by
# GET /tee/attestation?model=e2ee-...&nonce=<64-hex> on 2026-06-04.
# The giant hex quotes are truncated for readability; the *set of keys* is what
# matters here (it must exercise extra="allow" overriding the inherited
# extra="forbid").
SAMPLE_ATTESTATION: dict = {
    "signing_address": "0x337F14bbAeAdDfD6f7C9A0722f3D06574674C426",
    "signing_algo": "ecdsa",
    "request_nonce": "593db2ea60de42b7f1d65293bf08d6206f089396f03ed1a4668af1b1b9b57cef",
    "intel_quote": "040002008100000000000000939a7233f79c4ca9940a0db3957f060705",
    "nvidia_payload": '{"nonce": "593db2ea60de42b7f1d65293bf08d6206f089396f03ed1a4668af1b1b9b57cef"}',
    "info": {
        "app_id": "abc",
        "instance_id": "def",
        "vm_config": "{}",
    },
    "quote": "040002008100000000000000939a7233f79c4ca9940a0db3957f060705",
    "event_log": [{"imr": 0, "event_type": 2147483659, "digest": "09abe9"}],
    "vm_config": '{"os_image_hash":"021bf66a7c9fd4a05031b8fa688834948874631c2ad5b9a2d566b4421b817271"}',
    "signing_public_key": (
        "04b94aa42e7d246bc8b3763858fa5d57e899f9b11bbd5a57583b892a2c532eafc7"
        "ed0d52a9c7c3339488efd9ee1234567890abcdef1234567890abcdef1234567890"
    ),
    "verified": True,
    "model": "e2ee-venice-uncensored-24b-p",
    "nonce": "593db2ea60de42b7f1d65293bf08d6206f089396f03ed1a4668af1b1b9b57cef",
    "nonce_source": "client",
    "tee_provider": "phala",
    "tee_hardware": "intel-tdx",
    "upstream_model": "phala/uncensored-24b",
    "server_verification": {
        "tdx": {"valid": True},
        "nvidia": {"valid": True},
        "signingAddressBinding": True,
        "nonceBinding": True,
    },
    "candidates_evaluated": 1,
    "candidates_available": 1,
}


def test_package_imports_on_bare_install() -> None:
    """``import venice_ai.tee`` must succeed without the ``[e2ee]`` extra."""
    assert tee is not None
    assert hasattr(tee, "_require_crypto")


def test_exceptions_exist_and_subclass() -> None:
    assert issubclass(TeeError, VeniceError)
    assert issubclass(TeeAttestationError, TeeError)
    assert issubclass(TeeEncryptionError, TeeError)
    # And re-exported from the package
    assert tee.TeeError is TeeError
    assert tee.TeeAttestationError is TeeAttestationError
    assert tee.TeeEncryptionError is TeeEncryptionError


def test_constants_have_confirmed_values() -> None:
    from venice_ai.tee import _constants as c

    assert c.HKDF_INFO == b"ecdsa_encryption"
    assert c.HKDF_LENGTH == 32
    assert c.CURVE_NAME == "secp256k1"
    assert c.MIN_ENCRYPTED_HEX_LEN == 186
    assert c.GCM_NONCE_LEN == 12
    assert c.UNCOMPRESSED_PUBKEY_LEN == 65
    assert c.HEADER_CLIENT_PUB_KEY == "X-Venice-TEE-Client-Pub-Key"
    assert c.HEADER_MODEL_PUB_KEY == "X-Venice-TEE-Model-Pub-Key"
    assert c.HEADER_SIGNING_ALGO == "X-Venice-TEE-Signing-Algo"


def test_attestation_parses_full_payload_and_retains_extras() -> None:
    from venice_ai.tee.types import TeeAttestation

    att = TeeAttestation.model_validate(SAMPLE_ATTESTATION)

    # Modeled fields
    assert att.verified is True
    assert att.signing_public_key.startswith("04")
    assert att.signing_address == "0x337F14bbAeAdDfD6f7C9A0722f3D06574674C426"
    assert att.signing_algo == "ecdsa"
    assert att.nonce == SAMPLE_ATTESTATION["nonce"]
    # Retained raw fields (loosely typed)
    assert att.intel_quote is not None
    assert att.nvidia_payload is not None
    assert att.quote is not None
    assert att.server_verification is not None
    assert att.vm_config is not None
    # sent_nonce defaults to None and is settable later by the verifier
    assert att.sent_nonce is None
    att.sent_nonce = SAMPLE_ATTESTATION["nonce"]
    assert att.sent_nonce == SAMPLE_ATTESTATION["nonce"]

    # extra="allow" must override the inherited extra="forbid": the unmodeled
    # keys (info, event_log, tee_provider, ...) must NOT raise and must be
    # retained on the model.
    extras = att.model_dump()
    for key in ("info", "event_log", "tee_provider", "tee_hardware", "upstream_model"):
        assert key in extras, f"extra key {key!r} was dropped — extra='allow' did not take effect"


def test_extra_forbid_did_not_leak() -> None:
    """A trimmed-only sample would parse either way; an unknown junk key proves allow."""
    from venice_ai.tee.types import TeeAttestation

    payload = dict(SAMPLE_ATTESTATION)
    payload["totally_unexpected_future_field"] = {"x": 1}
    att = TeeAttestation.model_validate(payload)
    assert att.model_dump().get("totally_unexpected_future_field") == {"x": 1}


def test_no_top_level_cryptography_import_in_tee_package() -> None:
    """A bare install must import ``venice_ai.tee``; ``cryptography`` may only be
    imported lazily inside functions, never at module top level.

    (``cryptography`` is installed in this dev env, so an import test alone would
    not catch a stray top-level import — this AST check does.)
    """
    pkg_dir = Path(tee.__file__).parent
    offenders: list[str] = []
    for py in pkg_dir.glob("*.py"):
        module = ast.parse(py.read_text())
        for node in ast.iter_child_nodes(module):  # top-level statements only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "cryptography" or alias.name.startswith("cryptography."):
                        offenders.append(f"{py.name}: import {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "cryptography" or node.module.startswith("cryptography."))
            ):
                offenders.append(f"{py.name}: from {node.module} import ...")
    assert not offenders, f"top-level cryptography imports found: {offenders}"


def test_require_crypto_returns_module_when_installed() -> None:
    """In this env cryptography is installed, so ``_require_crypto`` returns it."""
    mod = tee._require_crypto()
    assert mod is not None
    assert getattr(mod, "__name__", "").startswith("cryptography")


def test_require_crypto_raises_helpful_importerror_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When cryptography is unavailable, ``_require_crypto`` raises ImportError
    with the ``pip install venice-py[e2ee]`` hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("No module named 'cryptography'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"venice-py\[e2ee\]"):
        tee._require_crypto()


def test_tee_exports_typed_errors() -> None:
    import venice_ai.exceptions as exc_mod

    for name in (
        "TeeTcbStatusError",
        "TeeCollateralMismatchError",
        "TeeCollateralExpiredError",
        "TeeCollateralUnavailableError",
    ):
        cls = getattr(tee, name)
        assert cls is getattr(exc_mod, name)
        assert name in exc_mod.__all__
        assert issubclass(cls, TeeAttestationError)
    assert tee.CollateralKey.__name__ == "CollateralKey"
