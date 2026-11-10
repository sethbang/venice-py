"""Unit tests for :mod:`venice_ai.auth.x402_solana` (Solana x402 settlement).

Deterministic, no network. Validates the byte-exact "exact" SVM scheme:
envelope shape, the four instructions and their order/program ids, the
TransferChecked data bytes, ATA derivation, feePayer placement, and the
partial-signing / round-trip semantics.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web

# Skip the whole module cleanly if the optional extra is missing.
solders = pytest.importorskip("solders")

from solders.hash import Hash  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402
from solders.signature import Signature  # noqa: E402
from solders.transaction import VersionedTransaction  # noqa: E402

from venice_ai import RetryOptions, VeniceClient  # noqa: E402
from venice_ai.auth.x402_solana import (  # noqa: E402
    MEMO_PROGRAM_ID,
    SOLANA_MAINNET_CAIP2,
    USDC_SOLANA_MAINNET,
    SolanaX402Auth,
    _derive_ata,
)
from venice_ai.types.api.x402 import X402TopUpData, X402TopUpResponse  # noqa: E402

# A fixed facilitator + recipient mirroring Venice's live 402 response.
_FEE_PAYER = "BFK9TLC3edb13K6v4YyH3DwPb5DSUpkWvb7XnqCL9b4F"
_PAY_TO = "8qUL23aSj7mDWdoLMXGHFvnVCT9wd7jXcysiekroADEL"
_USDC_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
_COMPUTE_BUDGET = "ComputeBudget111111111111111111111111111111"
_ASSOCIATED_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"

# A deterministic recent blockhash for stable goldens (a valid base58 hash).
_BLOCKHASH = str(Hash.default())

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(b: bytes) -> str:
    """Minimal base58 encoder (avoids a runtime dep on a base58 package)."""
    n = int.from_bytes(b, "big")
    out = ""
    while n > 0:
        n, r = divmod(n, 58)
        out = _B58_ALPHABET[r] + out
    pad = len(b) - len(b.lstrip(b"\x00"))
    return _B58_ALPHABET[0] * pad + out


@pytest.fixture
def keypair() -> Keypair:
    """A fixed, deterministic test keypair (seeded; never used on-chain)."""
    # 32-byte seed → deterministic keypair.
    return Keypair.from_seed(bytes(range(32)))


@pytest.fixture
def auth(keypair: Keypair) -> SolanaX402Auth:
    return SolanaX402Auth(private_key=_b58encode(bytes(keypair)))


@pytest.fixture
def requirement() -> dict:
    """The legacy bare-network (``network="solana"``) Solana 402 requirement shape.

    Venice's *current* live challenge advertises the Solana requirement under
    the CAIP-2 mainnet id instead (see
    :data:`venice_ai.auth.x402_solana.SOLANA_MAINNET_CAIP2` and
    ``test_top_up_with_solana_picks_caip2_accepts_entry`` below, which exercises
    that shape). Both spellings are accepted by :func:`is_solana_mainnet`, so
    this fixture still exercises the signing / envelope logic end-to-end for
    the legacy shape.
    """
    return {
        "protocol": "x402",
        "version": 2,
        "network": "solana",
        "asset": USDC_SOLANA_MAINNET,
        "amount": "5000000",
        "payTo": _PAY_TO,
        "extra": {
            "name": "USD Coin",
            "version": "2",
            "feePayer": _FEE_PAYER,
            "memo": "venice-topup-test",
        },
    }


def _build(auth: SolanaX402Auth, requirement: dict, **overrides) -> str:
    kwargs = {
        "requirement": requirement,
        "recent_blockhash": _BLOCKHASH,
        "mint_decimals": 6,
        "token_program": _USDC_TOKEN_PROGRAM,
    }
    kwargs.update(overrides)
    return auth.build_payment_header(**kwargs)


def _decode_envelope(header: str) -> dict:
    return json.loads(base64.b64decode(header).decode("utf-8"))


def _decode_tx(header: str) -> VersionedTransaction:
    env = _decode_envelope(header)
    return VersionedTransaction.from_bytes(base64.b64decode(env["payload"]["transaction"]))


# ── Memo program id ─────────────────────────────────────────────────────────


def test_memo_program_id_round_trips() -> None:
    """The canonical Memo v2 id parses to a 32-byte pubkey."""
    pk = Pubkey.from_string(MEMO_PROGRAM_ID)
    assert len(bytes(pk)) == 32
    assert str(pk) == MEMO_PROGRAM_ID
    # The common placeholder typo must NOT parse as a valid pubkey.
    with pytest.raises(ValueError):
        Pubkey.from_string("MemoSq4gq4qj4qj4qj4qj4qj4qj4qj4qj4qj4qj4qj")


# ── ImportError hint ────────────────────────────────────────────────────────


def test_import_error_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-importing the module without solders surfaces the install hint."""
    import builtins
    import importlib
    import sys

    # Evict the module under test and any cached solders submodules so the
    # try/except import block re-runs.
    monkeypatch.delitem(sys.modules, "venice_ai.auth.x402_solana", raising=False)
    for name in [m for m in list(sys.modules) if m == "solders" or m.startswith("solders.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "solders" or name.startswith("solders."):
            raise ImportError("No module named 'solders' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match=r"venice-py\[x402-solana\]"):
        importlib.import_module("venice_ai.auth.x402_solana")


# ── wallet_address ──────────────────────────────────────────────────────────


def test_wallet_address_matches_keypair(auth: SolanaX402Auth, keypair: Keypair) -> None:
    assert auth.wallet_address == str(keypair.pubkey())


# ── Envelope shape ──────────────────────────────────────────────────────────


def test_envelope_shape(auth: SolanaX402Auth, requirement: dict) -> None:
    # x402 V2 PaymentPayload: {x402Version, payload, accepted}. The chosen
    # requirement is echoed under ``accepted`` (canonical PaymentRequirements:
    # scheme/network/asset/amount/payTo/maxTimeoutSeconds/extra — protocol &
    # version stripped). Verified against Venice's live facilitator (a missing
    # ``maxTimeoutSeconds`` or a top-level flat shape is rejected 400).
    env = _decode_envelope(_build(auth, requirement))
    assert env["x402Version"] == 2
    assert isinstance(env["payload"]["transaction"], str)
    assert env["payload"]["transaction"]  # non-empty base64
    accepted = env["accepted"]
    assert accepted["scheme"] == "exact"
    assert accepted["network"] == "solana"  # verbatim echo (requirement fixture is bare)
    assert accepted["maxTimeoutSeconds"] == 300  # required by V2 PaymentRequirements
    assert "protocol" not in accepted and "version" not in accepted
    assert "scheme" not in env and "network" not in env  # not at top level


# ── Transaction: instruction count / order / program ids ────────────────────


def test_four_instructions_in_order(auth: SolanaX402Auth, requirement: dict) -> None:
    tx = _decode_tx(_build(auth, requirement))
    msg = tx.message
    keys = list(msg.account_keys)
    instrs = msg.instructions
    assert len(instrs) == 4

    programs = [str(keys[ix.program_id_index]) for ix in instrs]
    assert programs[0] == _COMPUTE_BUDGET  # set unit limit
    assert programs[1] == _COMPUTE_BUDGET  # set unit price
    assert programs[2] == _USDC_TOKEN_PROGRAM  # TransferChecked
    assert programs[3] == MEMO_PROGRAM_ID  # memo


def test_compute_budget_instruction_data(auth: SolanaX402Auth, requirement: dict) -> None:
    tx = _decode_tx(_build(auth, requirement))
    instrs = tx.message.instructions
    # SetComputeUnitLimit: tag 2 + u32 LE 20000.
    assert instrs[0].data == bytes([2]) + (20000).to_bytes(4, "little")
    # SetComputeUnitPrice: tag 3 + u64 LE 1.
    assert instrs[1].data == bytes([3]) + (1).to_bytes(8, "little")


def test_transfer_checked_data_bytes(auth: SolanaX402Auth, requirement: dict) -> None:
    tx = _decode_tx(_build(auth, requirement))
    transfer = tx.message.instructions[2]
    # 12 + amount(u64 LE) + decimals.
    assert transfer.data == bytes([12]) + (5_000_000).to_bytes(8, "little") + bytes([6])


def test_transfer_checked_accounts(
    auth: SolanaX402Auth, requirement: dict, keypair: Keypair
) -> None:
    tx = _decode_tx(_build(auth, requirement))
    msg = tx.message
    keys = list(msg.account_keys)
    transfer = msg.instructions[2]
    acct_keys = [str(keys[i]) for i in transfer.accounts]

    mint = Pubkey.from_string(USDC_SOLANA_MAINNET)
    token_prog = Pubkey.from_string(_USDC_TOKEN_PROGRAM)
    expected_src = _derive_ata(keypair.pubkey(), token_prog, mint)
    expected_dst = _derive_ata(Pubkey.from_string(_PAY_TO), token_prog, mint)

    assert acct_keys[0] == str(expected_src)  # source ATA
    assert acct_keys[1] == USDC_SOLANA_MAINNET  # mint
    assert acct_keys[2] == str(expected_dst)  # dest ATA
    assert acct_keys[3] == str(keypair.pubkey())  # owner = payer wallet


def test_memo_instruction_data(auth: SolanaX402Auth, requirement: dict) -> None:
    tx = _decode_tx(_build(auth, requirement))
    memo = tx.message.instructions[3]
    assert memo.data == b"venice-topup-test"
    assert list(memo.accounts) == []  # memo has no accounts


def test_memo_random_when_absent(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["extra"].pop("memo")
    tx = _decode_tx(_build(auth, requirement))
    memo_data = tx.message.instructions[3].data
    assert len(memo_data) == 32  # hexlify(16 random bytes) → 32 hex chars
    bytes.fromhex(memo_data.decode("ascii"))  # valid hex


# ── feePayer placement ──────────────────────────────────────────────────────


def test_fee_payer_is_payer_and_not_in_instructions(
    auth: SolanaX402Auth, requirement: dict
) -> None:
    tx = _decode_tx(_build(auth, requirement))
    msg = tx.message
    keys = list(msg.account_keys)
    # account_keys[0] is the message payer = feePayer.
    assert str(keys[0]) == _FEE_PAYER
    fee_payer_index = 0
    for ix in msg.instructions:
        assert fee_payer_index not in list(ix.accounts), (
            "feePayer must not appear in any instruction's accounts"
        )


# ── Partial signing / round-trip ────────────────────────────────────────────


def test_signature_slot_zero_is_placeholder(auth: SolanaX402Auth, requirement: dict) -> None:
    tx = _decode_tx(_build(auth, requirement))
    assert tx.signatures[0] == Signature.default()  # 64 zero bytes
    assert bytes(tx.signatures[0]) == bytes(64)


def test_client_signature_verifies_over_deserialized_message(
    auth: SolanaX402Auth, requirement: dict, keypair: Keypair
) -> None:
    tx = _decode_tx(_build(auth, requirement))
    # Reconstruct the signed payload from the DESERIALIZED message (catches
    # serialization drift): 0x80 wire-version prefix + bytes(message).
    msg_bytes = bytes([0x80]) + bytes(tx.message)
    client_sig = tx.signatures[1]
    assert client_sig.verify(keypair.pubkey(), msg_bytes)


def test_two_signatures_present(auth: SolanaX402Auth, requirement: dict) -> None:
    tx = _decode_tx(_build(auth, requirement))
    assert len(tx.signatures) == 2


def test_signing_formula_matches_solders_authoritative_sign() -> None:
    """Non-circular guard for the ``bytes([0x80]) + bytes(message)`` payload.

    The other signing tests reconstruct the signed payload with the SAME
    formula the implementation uses, so they only prove self-consistency. This
    test instead delegates to solders' own (= solana-sdk's) authoritative
    transaction signing and asserts our hand-rolled payload produces a
    byte-identical signature. If ``bytes(MessageV0)`` ever started including
    the 0x80 wire-version prefix (giving a double prefix), this catches it even
    though every other test would still pass — exactly the silent on-chain
    signature failure the spec warned about.
    """
    from solders.instruction import Instruction
    from solders.message import MessageV0

    kp = Keypair.from_seed(bytes(range(32)))
    ix = Instruction(Pubkey.from_string(MEMO_PROGRAM_ID), b"x", [])
    # Client is the fee payer here so signatures[0] is the client's slot.
    msg = MessageV0.try_compile(kp.pubkey(), [ix], [], Hash.default())

    # The first message byte is the header's num_required_signatures, NOT the
    # 0x80 wire-version prefix — proves bytes(MessageV0) omits the prefix.
    assert bytes(msg)[0] != 0x80

    authoritative = VersionedTransaction(msg, [kp])
    manual = kp.sign_message(bytes([0x80]) + bytes(msg))
    assert authoritative.signatures[0] == manual


# ── Amount / decimals encoding edge cases ───────────────────────────────────


@pytest.mark.parametrize(
    "amount,decimals",
    [
        (1, 0),
        (5_000_000, 6),
        (2**64 - 1, 9),  # max u64
        (123456789, 255),  # max decimals byte
    ],
)
def test_amount_decimals_encoding(
    auth: SolanaX402Auth, requirement: dict, amount: int, decimals: int
) -> None:
    requirement["amount"] = str(amount)
    tx = _decode_tx(_build(auth, requirement, mint_decimals=decimals))
    transfer = tx.message.instructions[2]
    assert transfer.data == bytes([12]) + amount.to_bytes(8, "little") + bytes([decimals])


# ── Validation (refuse to sign deviations) ──────────────────────────────────


def test_rejects_non_solana_network(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["network"] = "eip155:8453"
    with pytest.raises(ValueError, match="network mismatch"):
        _build(auth, requirement)


def test_rejects_missing_fee_payer(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["extra"].pop("feePayer")
    with pytest.raises(ValueError, match="feePayer"):
        _build(auth, requirement)


def test_rejects_amount_over_cap(auth: SolanaX402Auth, requirement: dict) -> None:
    with pytest.raises(ValueError, match="exceeds cap"):
        _build(auth, requirement, max_amount_units=1_000_000)


def test_rejects_non_positive_amount(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["amount"] = "0"
    with pytest.raises(ValueError, match="must be positive"):
        _build(auth, requirement)


def test_rejects_oversize_memo(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["extra"]["memo"] = "x" * 257
    with pytest.raises(ValueError, match="memo"):
        _build(auth, requirement)


def test_rejects_bad_decimals(auth: SolanaX402Auth, requirement: dict) -> None:
    with pytest.raises(ValueError, match="mint_decimals"):
        _build(auth, requirement, mint_decimals=256)


_OTHER_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"


def test_rejects_asset_that_is_not_usdc(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["asset"] = _OTHER_MINT
    with pytest.raises(ValueError, match="asset mismatch") as exc_info:
        _build(auth, requirement)
    assert _OTHER_MINT in str(exc_info.value)
    assert USDC_SOLANA_MAINNET in str(exc_info.value)


def test_validate_asset_none_means_usdc(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["asset"] = _OTHER_MINT
    with pytest.raises(ValueError, match="asset mismatch"):
        _build(auth, requirement, validate_asset=None)


def test_asset_comparison_is_case_sensitive(auth: SolanaX402Auth, requirement: dict) -> None:
    """Base58 is case-sensitive: a re-cased mint is a different address, unlike EVM hex."""
    requirement["asset"] = USDC_SOLANA_MAINNET.swapcase()
    with pytest.raises(ValueError, match="asset mismatch"):
        _build(auth, requirement)


def test_validate_asset_allows_the_named_mint(auth: SolanaX402Auth, requirement: dict) -> None:
    requirement["asset"] = _OTHER_MINT
    tx = _decode_tx(_build(auth, requirement, validate_asset=_OTHER_MINT))
    keys = list(tx.message.account_keys)
    transfer = tx.message.instructions[2]
    assert str(keys[transfer.accounts[1]]) == _OTHER_MINT


def test_validate_asset_replaces_the_usdc_default(auth: SolanaX402Auth, requirement: dict) -> None:
    with pytest.raises(ValueError, match="asset mismatch"):
        _build(auth, requirement, validate_asset=_OTHER_MINT)


# ── Resource wiring: X402.top_up_with_solana ────────────────────────────────


def _payment_required_error(body: dict):
    from unittest.mock import MagicMock

    from venice_ai.exceptions import PaymentRequiredError

    fake_response = MagicMock()
    fake_response.status_code = 402
    return PaymentRequiredError("Payment required", response=fake_response, body=body)


@pytest.fixture
def x402_resource():
    from unittest.mock import MagicMock

    from venice_ai.resources.x402 import X402

    client = MagicMock()
    client._get_session = AsyncMock(return_value=MagicMock())
    return X402(client)


@pytest.mark.asyncio
async def test_top_up_with_solana_happy_path(
    x402_resource, auth: SolanaX402Auth, requirement: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    from venice_ai.types.api.x402 import X402TopUpData, X402TopUpResponse

    probe_err = _payment_required_error({"x402Version": 2, "accepts": [requirement]})
    settled = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress=auth.wallet_address,
            amountCredited=5.0,
            newBalance=5.0,
            paymentId="x402-sol-test",
        ),
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])

    # Patch the live RPC fetch to return deterministic context (no network).
    async def _fake_ctx(rpc_url, mint, *, http, **_kwargs):
        return _BLOCKHASH, 6, _USDC_TOKEN_PROGRAM

    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", _fake_ctx)

    result = await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)

    assert result.data.amountCredited == 5.0
    assert x402_resource.top_up.call_count == 2
    first, second = x402_resource.top_up.call_args_list
    assert first.kwargs == {}  # probe carries no header
    header = second.kwargs["payment_header"]
    env = _decode_envelope(header)
    assert env["accepted"]["network"] == "solana"
    assert env["payload"]["transaction"]


@pytest.mark.asyncio
async def test_top_up_with_solana_picks_solana_accepts_entry(
    x402_resource, auth: SolanaX402Auth, requirement: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch must select the network=='solana' entry, ignoring EVM entries."""
    from unittest.mock import AsyncMock

    from venice_ai.types.api.x402 import X402TopUpData, X402TopUpResponse

    evm_entry = {
        "network": "eip155:8453",
        "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "amount": "5000000",
        "payTo": "0xdead",
    }
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [evm_entry, requirement]})
    settled = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress=auth.wallet_address,
            amountCredited=5.0,
            newBalance=5.0,
            paymentId="x402-sol-test",
        ),
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])

    async def _fake_ctx(rpc_url, mint, *, http, **_kwargs):
        assert mint == USDC_SOLANA_MAINNET  # used the solana entry's asset
        return _BLOCKHASH, 6, _USDC_TOKEN_PROGRAM

    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", _fake_ctx)

    result = await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)
    assert result.data.amountCredited == 5.0


@pytest.mark.asyncio
async def test_top_up_with_solana_picks_caip2_accepts_entry(
    x402_resource, auth: SolanaX402Auth, requirement: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch must select the CAIP-2 network entry, ignoring EVM entries.

    Venice's live 402 challenge advertises Solana under the CAIP-2 id (see
    :mod:`venice_ai.auth.x402_solana`'s module docstring), not the bare
    ``"solana"`` string covered by
    ``test_top_up_with_solana_picks_solana_accepts_entry`` above. Reverting the
    selection in ``resources/x402.py`` to a bare ``== "solana"`` check would
    fail this test (no unit test previously covered that path).
    """
    from unittest.mock import AsyncMock

    from venice_ai.auth.x402_solana import SOLANA_MAINNET_CAIP2
    from venice_ai.types.api.x402 import X402TopUpData, X402TopUpResponse

    caip2_requirement = {**requirement, "network": SOLANA_MAINNET_CAIP2}
    evm_entry = {
        "network": "eip155:8453",
        "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "amount": "5000000",
        "payTo": "0xdead",
    }
    probe_err = _payment_required_error(
        {"x402Version": 2, "accepts": [evm_entry, caip2_requirement]}
    )
    settled = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress=auth.wallet_address,
            amountCredited=5.0,
            newBalance=5.0,
            paymentId="x402-sol-test",
        ),
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])

    async def _fake_ctx(rpc_url, mint, *, http, **_kwargs):
        assert mint == USDC_SOLANA_MAINNET  # used the CAIP-2 entry's asset
        return _BLOCKHASH, 6, _USDC_TOKEN_PROGRAM

    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", _fake_ctx)

    result = await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)
    assert result.data.amountCredited == 5.0
    _, second = x402_resource.top_up.call_args_list
    header = second.kwargs["payment_header"]
    env = _decode_envelope(header)
    assert env["accepted"]["network"] == SOLANA_MAINNET_CAIP2  # the CAIP-2 entry was selected


@pytest.mark.asyncio
async def test_top_up_with_solana_no_solana_entry_raises(
    x402_resource, auth: SolanaX402Auth
) -> None:
    from unittest.mock import AsyncMock

    evm_entry = {"network": "eip155:8453", "asset": "0x0", "amount": "5000000", "payTo": "0x0"}
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [evm_entry]})
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])

    with pytest.raises(RuntimeError, match="no Solana mainnet requirement"):
        await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)


@pytest.mark.parametrize(
    ("amount_usdc", "max_amount_usdc", "server_units", "pattern"),
    [
        (10.0, None, "5000000", r"amount_usdc=\$10\.0 differs .*\$5\.0 USDC"),
        (3.0, 10.0, "5000000", r"amount_usdc=\$3\.0 differs .*\$5\.0 USDC"),
        (5.0, 10.0, "7000000", r"amount_usdc=\$5\.0 differs .*\$7\.0 USDC"),
        (5.0, 5.0, "10000000", r"Server requires \$10\.0 USDC"),
    ],
)
@pytest.mark.asyncio
async def test_top_up_with_solana_refuses_amount_mismatch_before_rpc_or_signing(
    x402_resource,
    auth: SolanaX402Auth,
    requirement: dict,
    monkeypatch: pytest.MonkeyPatch,
    amount_usdc: float,
    max_amount_usdc: float | None,
    server_units: str,
    pattern: str,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    requirement["amount"] = server_units
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [requirement]})
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])
    fetch = AsyncMock(return_value=(_BLOCKHASH, 6, _USDC_TOKEN_PROGRAM))
    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", fetch)
    sign = MagicMock(side_effect=AssertionError("must not sign"))
    monkeypatch.setattr(auth, "build_payment_header", sign)

    with pytest.raises(ValueError, match=pattern):
        await x402_resource.top_up_with_solana(
            auth=auth, amount_usdc=amount_usdc, max_amount_usdc=max_amount_usdc
        )

    fetch.assert_not_called()
    sign.assert_not_called()
    assert x402_resource.top_up.call_count == 1


@pytest.mark.asyncio
async def test_top_up_with_solana_signs_exactly_the_requirement_under_a_higher_cap(
    x402_resource, auth: SolanaX402Auth, requirement: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    from venice_ai.types.api.x402 import X402TopUpData, X402TopUpResponse

    probe_err = _payment_required_error({"x402Version": 2, "accepts": [requirement]})
    settled = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress=auth.wallet_address,
            amountCredited=5.0,
            newBalance=5.0,
            paymentId="x402-sol-test",
        ),
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])
    monkeypatch.setattr(
        "venice_ai.auth.x402_solana.fetch_solana_tx_context",
        AsyncMock(return_value=(_BLOCKHASH, 6, _USDC_TOKEN_PROGRAM)),
    )

    await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0, max_amount_usdc=10.0)

    tx = _decode_tx(x402_resource.top_up.call_args_list[1].kwargs["payment_header"])
    assert tx.message.instructions[2].data[1:9] == (5_000_000).to_bytes(8, "little")


@pytest.mark.asyncio
async def test_top_up_with_solana_refuses_a_requirement_that_is_not_usdc(
    x402_resource, auth: SolanaX402Auth, requirement: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-call path takes no validate_asset, so a non-USDC mint never settles."""
    from unittest.mock import AsyncMock

    requirement["asset"] = _OTHER_MINT
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [requirement]})
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])
    fetch = AsyncMock(return_value=(_BLOCKHASH, 6, _USDC_TOKEN_PROGRAM))
    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", fetch)

    with pytest.raises(ValueError, match="no USDC requirement on Solana mainnet") as exc_info:
        await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)

    assert "validate_asset" not in str(exc_info.value)
    assert _OTHER_MINT in str(exc_info.value)
    assert x402_resource.top_up.call_count == 1
    fetch.assert_not_awaited()


@pytest.mark.parametrize("network", ["solana", SOLANA_MAINNET_CAIP2])
@pytest.mark.asyncio
async def test_top_up_with_solana_takes_the_usdc_entry_behind_another_mint(
    x402_resource,
    auth: SolanaX402Auth,
    requirement: dict,
    monkeypatch: pytest.MonkeyPatch,
    network: str,
) -> None:
    from unittest.mock import AsyncMock

    from venice_ai.types.api.x402 import X402TopUpData, X402TopUpResponse

    requirement["network"] = network
    other_mint = {**requirement, "asset": _OTHER_MINT, "amount": "9000000"}
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [other_mint, requirement]})
    settled = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress=auth.wallet_address,
            amountCredited=5.0,
            newBalance=5.0,
            paymentId="x402-sol-test",
        ),
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])
    fetched: list[str] = []

    async def _fake_ctx(rpc_url, mint, *, http, **_kwargs):
        fetched.append(mint)
        return _BLOCKHASH, 6, _USDC_TOKEN_PROGRAM

    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", _fake_ctx)

    await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)

    assert fetched == [USDC_SOLANA_MAINNET]
    envelope = _decode_envelope(x402_resource.top_up.call_args_list[1].kwargs["payment_header"])
    assert envelope["accepted"]["asset"] == USDC_SOLANA_MAINNET


@pytest.mark.asyncio
async def test_top_up_with_solana_skips_usdc_on_another_network(
    x402_resource, auth: SolanaX402Auth, requirement: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    requirement["asset"] = _OTHER_MINT
    devnet = {**requirement, "network": "solana-devnet", "asset": USDC_SOLANA_MAINNET}
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [devnet, requirement]})
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])
    fetch = AsyncMock(return_value=(_BLOCKHASH, 6, _USDC_TOKEN_PROGRAM))
    monkeypatch.setattr("venice_ai.auth.x402_solana.fetch_solana_tx_context", fetch)

    with pytest.raises(ValueError, match="no USDC requirement on Solana mainnet"):
        await x402_resource.top_up_with_solana(auth=auth, amount_usdc=5.0)

    fetch.assert_not_awaited()


# ---------------------------------------------------------------------------
# The RPC call goes through the client's session
# ---------------------------------------------------------------------------

_API_KEY = "sk-SENTINELAPIKEY"
_RPC_SECRET = "RPCSECRET"
_RPC_URL = f"http://rpc.example/v2/{_RPC_SECRET}/?api-key={_RPC_SECRET}"


def _rpc_answer(method: str) -> dict:
    if method == "getLatestBlockhash":
        return {"result": {"value": {"blockhash": _BLOCKHASH}}}
    mint = base64.b64encode(bytes(44) + bytes([6]) + bytes(37)).decode()
    return {"result": {"value": {"owner": _USDC_TOKEN_PROGRAM, "data": [mint, "base64"]}}}


@asynccontextmanager
async def _rpc_host(seen: list[web.Request]) -> AsyncIterator[str]:
    """A loopback host answering Solana RPC calls, used directly or as an HTTP proxy."""

    async def answer(request: web.Request) -> web.Response:
        seen.append(request)
        return web.json_response(_rpc_answer((await request.json())["method"]))

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", answer)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


def _stub_top_up(client: VeniceClient, requirement: dict, auth: SolanaX402Auth) -> None:
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [requirement]})
    settled = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress=auth.wallet_address,
            amountCredited=5.0,
            newBalance=5.0,
            paymentId="x402-sol-test",
        ),
    )
    client.x402.top_up = AsyncMock(side_effect=[probe_err, settled])  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_solana_rpc_goes_through_the_clients_proxy_without_the_api_key(
    auth: SolanaX402Auth, requirement: dict
) -> None:
    seen: list[web.Request] = []
    async with (
        _rpc_host(seen) as proxy,
        VeniceClient(api_key=_API_KEY, base_url="http://127.0.0.1:9/api/v1", proxy=proxy) as client,
    ):
        _stub_top_up(client, requirement, auth)
        result = await client.x402.top_up_with_solana(auth=auth, amount_usdc=5.0, rpc_url=_RPC_URL)

    assert result.success
    assert [r.host for r in seen] == ["rpc.example", "rpc.example"]
    for request in seen:
        assert "Authorization" not in request.headers
        assert "X-API-Key" not in request.headers
        assert _API_KEY not in str(request.headers)


@pytest.mark.asyncio
async def test_solana_rpc_uses_a_session_the_caller_supplied(
    auth: SolanaX402Auth, requirement: dict
) -> None:
    seen: list[web.Request] = []
    async with (
        _rpc_host(seen) as rpc,
        aiohttp.ClientSession() as session,
        VeniceClient(api_key=_API_KEY, base_url=f"{rpc}/api/v1", http_client=session) as client,
    ):
        _stub_top_up(client, requirement, auth)
        with patch.object(session, "post", wraps=session.post) as post:
            await client.x402.top_up_with_solana(auth=auth, amount_usdc=5.0, rpc_url=f"{rpc}/rpc")
        assert not session.closed

    assert post.call_count == 2
    assert all("Authorization" not in r.headers for r in seen)


@pytest.mark.parametrize("status", [401, 503])
@pytest.mark.asyncio
async def test_solana_rpc_failure_through_the_clients_session_stays_url_free(
    auth: SolanaX402Auth,
    requirement: dict,
    status: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    async def refuse(_request: web.Request) -> web.Response:
        return web.Response(status=status, text=f"refused {_RPC_SECRET}")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", refuse)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    rpc = f"http://127.0.0.1:{runner.addresses[0][1]}/v2/{_RPC_SECRET}/?api-key={_RPC_SECRET}"
    try:
        async with VeniceClient(
            api_key=_API_KEY,
            base_url="http://127.0.0.1:9/api/v1",
            retry_options=RetryOptions(max_attempts=1, base_delay=0.001, max_delay=0.002),
        ) as client:
            _stub_top_up(client, requirement, auth)
            with pytest.raises(RuntimeError) as excinfo:
                await client.x402.top_up_with_solana(auth=auth, amount_usdc=5.0, rpc_url=rpc)
    finally:
        await runner.cleanup()

    error = excinfo.value
    assert f"HTTP {status}" in str(error)
    assert error.__cause__ is None and error.__context__ is None
    for secret in (_RPC_SECRET, _API_KEY):
        assert secret not in f"{error!s} {error!r}"
        assert secret not in caplog.text
