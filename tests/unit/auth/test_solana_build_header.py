"""TDD: SolanaX402Auth.build_header — the SIGN-IN-WITH-X auth header (audit MED #11).

Client-side correctness only (structure + a verifiable ed25519 signature over the
SIWS message). Server acceptance is confirmed separately by a live /x402/balance
probe with the funded test key.
"""

import base64
import json
import re

import pytest

solders = pytest.importorskip("solders")
from solders.keypair import Keypair  # noqa: E402
from solders.signature import Signature  # noqa: E402

from venice_ai.auth.x402_solana import SolanaX402Auth  # noqa: E402


def _auth():
    kp = Keypair()
    return SolanaX402Auth(private_key=str(kp)), kp


def test_build_header_structure_and_fields():
    auth, kp = _auth()
    obj = json.loads(base64.b64decode(auth.build_header(nonce="0123456789abcdef")))
    assert obj["address"] == auth.wallet_address == str(kp.pubkey())
    assert obj["type"] == "ed25519"
    assert obj["chainId"].startswith("solana:")
    assert isinstance(obj["timestamp"], int)
    assert "wants you to sign in with your Solana account:" in obj["message"]
    assert "Nonce: 0123456789abcdef" in obj["message"]


def test_build_header_signature_verifies():
    auth, kp = _auth()
    obj = json.loads(base64.b64decode(auth.build_header()))
    sig = Signature.from_string(obj["signature"])
    assert sig.verify(kp.pubkey(), obj["message"].encode("utf-8"))


def test_client_signs_x402_reads_with_a_solana_auth():
    # client.x402.balance/transactions name the wallet and let the client sign
    # it, so that signer resolution must accept a SolanaX402Auth and not only
    # the EVM X402Auth. It returns the callable rather than a header because
    # each retried attempt calls it again for a fresh nonce.
    import os
    from unittest.mock import patch

    from venice_ai import VeniceClient

    auth, _ = _auth()
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)

    resign = client._resolve_siwe_resigner(None, auth)
    assert resign is not None

    # Assert on the nonce, not on the envelopes differing: Ed25519 signing is
    # deterministic, so inequality alone would also be satisfied by a static
    # nonce paired with a ticking ``Issued At``.
    def nonce_of(header: str) -> str:
        message = json.loads(base64.b64decode(header))["message"]
        found = re.search(r"^Nonce: (\S+)$", message, re.M)
        assert found is not None, f"no nonce in SIWX message: {message!r}"
        return found.group(1)

    assert len({nonce_of(resign()) for _ in range(5)}) == 5
