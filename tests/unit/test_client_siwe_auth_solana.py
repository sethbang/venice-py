"""Mode-2 (SIWX-only) auth path on :class:`VeniceClient` with a **Solana** wallet.

The Solana parallel of ``test_client_siwe_auth.py``: Venice enabled SIWX
inference authentication for Solana wallets, so the client's default-header
path (used to authenticate inference when no API key is set) must accept a
:class:`~venice_ai.auth.x402_solana.SolanaX402Auth`, not only the EVM
:class:`~venice_ai.auth.x402.X402Auth`.

Skips if the ``[x402-solana]`` extra (``solders``) is not installed.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Gated on the Solana extra only — this path never touches eth_account/siwe.
pytest.importorskip("solders", reason="x402-solana extra not installed")

from solders.keypair import Keypair  # noqa: E402

from venice_ai import VeniceClient  # noqa: E402
from venice_ai.auth.x402_solana import SolanaX402Auth  # noqa: E402


@pytest.fixture
def solana_auth() -> SolanaX402Auth:
    """A random throwaway Solana keypair — never funded, never reused."""
    return SolanaX402Auth(private_key=str(Keypair()))


def test_solana_auth_exposes_ttl_seconds(solana_auth: SolanaX402Auth) -> None:
    """SolanaX402Auth exposes ttl_seconds (its SIWX message TTL), mirroring
    X402Auth. It bounds how long a signed envelope stays *valid*, not how long
    one may be reused — the nonce is single-use."""
    assert solana_auth.ttl_seconds == 600


def test_default_siwe_supports_solana_auth(solana_auth: SolanaX402Auth) -> None:
    """Mode 2 with a Solana wallet: the default SIWX header builds without
    crashing (previously raised AttributeError on the missing ttl_seconds)."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=solana_auth)
    header = client._default_siwe_header()
    assert header is not None
    assert isinstance(header, str)
    # base64 SIWX envelope — well over 200 bytes.
    assert len(header) > 200


def test_consecutive_solana_envelopes_carry_different_nonces(
    solana_auth: SolanaX402Auth,
) -> None:
    """The envelopes must differ in the nonce specifically.

    Ed25519 signing is deterministic, so two differing envelopes only prove the
    signed *message* changed. A static nonce paired with a ticking ``Issued At``
    would satisfy that while every request after the first still came back
    ``401 This nonce has already been used``.
    """
    import base64
    import json
    import re

    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=solana_auth)

    def nonce_of(header: str | None) -> str:
        assert header is not None
        message = json.loads(base64.b64decode(header))["message"]
        found = re.search(r"^Nonce: (\S+)$", message, re.M)
        assert found is not None, f"no nonce in SIWX message: {message!r}"
        return found.group(1)

    nonces = {nonce_of(client._default_siwe_header()) for _ in range(5)}

    assert len(nonces) == 5
