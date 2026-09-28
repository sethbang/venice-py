"""
x402 Solana ("exact" scheme, SVM) settlement for Venice's ``/x402/top-up``
endpoint.

Venice's live 402 challenge (``x402Version 2``) advertises a Solana payment
requirement whose ``network`` is the CAIP-2 mainnet id
:data:`SOLANA_MAINNET_CAIP2` (``"solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"``);
the bare string ``"solana"`` is also accepted for compatibility with older
challenges. The requirement's ``network`` is echoed back verbatim in the
settlement envelope. The envelope mirrors the EVM one in
:mod:`venice_ai.auth.x402` but carries a base64-encoded, partially-signed
Solana :class:`~solders.transaction.VersionedTransaction` instead of an
EIP-712 authorization::

    { "x402Version": 2, "payload": { "transaction": "<base64 VersionedTransaction>" },
      "accepted": { "scheme": "exact", "network": "<echoed verbatim>", ... } }

The transaction implements the upstream x402 "exact" scheme for SVM:

* message ``payer`` = the requirement's ``extra.feePayer`` (the facilitator
  pays gas; the client only signs the SPL transfer it authorizes), and
* exactly four instructions: ComputeBudget set-unit-limit, ComputeBudget
  set-unit-price, SPL ``TransferChecked``, SPL Memo.

The client signs only its own slot; the feePayer slot (index 0) is left as a
64-zero-byte placeholder for the facilitator to fill in before submission.

Usage::

    from venice_ai.auth.x402_solana import SolanaX402Auth

    auth = SolanaX402Auth(private_key="<base58 secret>")
    # auth.wallet_address — base58 pubkey derived from the key
    header = auth.build_payment_header(
        requirement=req,            # the selected Solana accepts entry (bare "solana" or CAIP-2)
        recent_blockhash=blockhash,
        mint_decimals=6,
        token_program="Tokenkeg...",
    )

This module is optional and requires ``pip install 'venice-py[x402-solana]'``.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

try:
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.signature import Signature
    from solders.transaction import VersionedTransaction
except ImportError as exc:  # pragma: no cover - import-time only
    raise ImportError(
        "The x402 Solana settlement helpers require the ``x402-solana`` extra. "
        "Install it with: pip install 'venice-py[x402-solana]'"
    ) from exc

if TYPE_CHECKING:
    from solders.pubkey import Pubkey as _Pubkey

__all__ = [
    "SolanaX402Auth",
    "USDC_SOLANA_MAINNET",
    "SOLANA_MAINNET_CAIP2",
    "is_solana_mainnet",
    "MEMO_PROGRAM_ID",
    "DEFAULT_SOLANA_RPC_URL",
    "fetch_solana_tx_context",
]

#: CAIP-2 chain id for Solana mainnet-beta (genesis hash reference). Venice's
#: live 402 challenge advertises the Solana requirement under this id.
SOLANA_MAINNET_CAIP2 = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"

#: The bare network string Venice used before it moved to CAIP-2 ids.
_SOLANA_BARE = "solana"


def is_solana_mainnet(network: str | None) -> bool:
    """Return whether ``network`` identifies Solana mainnet for x402 settlement.

    Accepts exactly two spellings: the bare ``"solana"`` string and the pinned
    :data:`SOLANA_MAINNET_CAIP2` id. Any other value — including other
    ``solana:<genesis>`` clusters such as devnet — returns ``False``.

    The match is deliberately an accept-list rather than a ``solana:`` prefix
    test. This guards a real-funds settlement path: a misconfigured or hostile
    facilitator offering a different cluster must not be able to steer a USDC
    transfer away from mainnet.
    """
    return network in (_SOLANA_BARE, SOLANA_MAINNET_CAIP2)


# Default Solana RPC endpoint; override via ``VENICE_X402_SOLANA_RPC_URL`` or
# the ``rpc_url`` parameter on the resource method.
DEFAULT_SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"

# ── Sign-In-With-X (SIWS) message fields ───────────────────────────────────
# Mirror the EVM ``X402Auth`` SIWE constants so the wallet-auth header is built
# consistently across chains. (Live-verified against /x402/balance.)
_SIWX_DOMAIN = "outerface.venice.ai"
_SIWX_URI = "https://outerface.venice.ai"
_SIWX_STATEMENT = "Sign in to Venice API"
_SIWX_VERSION = "1"
_SIWX_TTL_SECONDS = 600  # 10 minutes, per the docs' recommended SIWX TTL
# CAIP-2 chain identity for Solana mainnet (per the x402 docs / live 402 challenge).
# Same value as SOLANA_MAINNET_CAIP2 above; aliased (not redefined) to avoid drift.
_SOLANA_CAIP2 = SOLANA_MAINNET_CAIP2

# ── Canonical Solana program ids ───────────────────────────────────────────
# ComputeBudget program — native, fixed id.
_COMPUTE_BUDGET_PROGRAM_ID = "ComputeBudget111111111111111111111111111111"

# Associated Token Account program — used to derive the source/dest ATAs.
_ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"

# SPL Memo program (v2). Canonical mainnet id, verified against the official
# SPL/Solana program docs (https://www.solana-program.com/docs/memo, redirect
# from https://spl.solana.com/memo) and round-tripped via Pubkey.from_string.
# NOTE: a common typo'd placeholder
# ``MemoSq4gq4qj4qj4qj...`` does NOT parse as a 32-byte pubkey — do not use it.
MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"

# USDC mint on Solana mainnet (the asset Venice's live 402 advertises). The
# token program and decimals are read live from chain via getAccountInfo —
# this constant is only the default/expected asset for validation.
USDC_SOLANA_MAINNET = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# SPL Token TransferChecked instruction discriminator.
_SPL_TRANSFER_CHECKED = 12
# ComputeBudget instruction discriminators.
_CB_SET_UNIT_LIMIT = 2
_CB_SET_UNIT_PRICE = 3
# Conservative compute-budget values matching the upstream x402 SVM scheme.
_COMPUTE_UNIT_LIMIT = 20_000
_COMPUTE_UNIT_PRICE = 1

# MessageV0 wire-format version prefix. ``bytes(MessageV0)`` omits this byte
# (it starts at the message header), but the signed payload includes it.
_MESSAGE_V0_PREFIX = 0x80

_MAX_MEMO_BYTES = 256


def _derive_ata(owner: _Pubkey, token_program: _Pubkey, mint: _Pubkey) -> _Pubkey:
    """Derive the Associated Token Account address for ``(owner, mint)``.

    Manual derivation (no spl-token helper):
    ``find_program_address([owner, token_program, mint], ATA_PROGRAM_ID)``.
    """
    ata, _bump = Pubkey.find_program_address(
        [bytes(owner), bytes(token_program), bytes(mint)],
        Pubkey.from_string(_ASSOCIATED_TOKEN_PROGRAM_ID),
    )
    return ata


class SolanaX402Auth:
    """Builds base64 ``X-402-Payment`` envelopes for Venice's Solana x402 flow.

    :param private_key: A base58-encoded Solana secret key (the 64-byte
        keypair, as produced by ``solana-keygen`` / wallet exports).
        **Never share or commit this.**
    """

    def __init__(self, *, private_key: str) -> None:
        self._keypair = Keypair.from_base58_string(private_key)

    @property
    def wallet_address(self) -> str:
        """Base58 public key (wallet address) derived from the secret key."""
        return str(self._keypair.pubkey())

    @property
    def ttl_seconds(self) -> int:
        """SIWX token TTL in seconds — the validity window baked into the
        message :meth:`build_header` signs (default 600), mirroring
        :attr:`~venice_ai.auth.x402.X402Auth.ttl_seconds`.

        This bounds how long a signed envelope stays *valid*; it is not a
        reuse window. Venice treats the nonce as single-use, so every request
        carries its own freshly signed envelope.
        """
        return _SIWX_TTL_SECONDS

    def build_header(self, *, nonce: str | None = None, now: datetime | None = None) -> str:
        """Build the base64-encoded ``X-Sign-In-With-X`` (SIGN-IN-WITH-X) header value.

        Signs a Solana SIWX (Sign-In-With-X) message with the wallet's Ed25519
        key so wallet-authenticated reads (``client.x402.balance`` /
        ``transactions``) work for Solana, mirroring
        :meth:`venice_ai.auth.x402.X402Auth.build_header` for EVM.

        :param nonce: Optional 16-character hex nonce; a random one is generated
            if omitted.
        :param now: Override the ``issuedAt`` timestamp (UTC), for testing.
        """
        if nonce is None:
            nonce = secrets.token_hex(8)  # 16 hex chars
        elif len(nonce) != 16:
            raise ValueError("nonce must be a 16-character hex string")

        if now is None:
            now = datetime.now(UTC)
        issued_at = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        expiration = (now + timedelta(seconds=_SIWX_TTL_SECONDS)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        address = self.wallet_address
        # Solana SIWS message (per Venice docs): a CAIP-122-style block that the
        # server parses for domain/nonce/expiry validation.
        message = (
            f"{_SIWX_DOMAIN} wants you to sign in with your Solana account:\n"
            f"{address}\n\n"
            f"{_SIWX_STATEMENT}\n\n"
            f"URI: {_SIWX_URI}\n"
            f"Version: {_SIWX_VERSION}\n"
            f"Chain ID: {_SOLANA_CAIP2}\n"
            f"Nonce: {nonce}\n"
            f"Issued At: {issued_at}\n"
            f"Expiration Time: {expiration}"
        )
        signature = self._keypair.sign_message(message.encode("utf-8"))

        header_obj: dict[str, Any] = {
            "address": address,
            "message": message,
            "signature": str(signature),  # base58 (Ed25519)
            "timestamp": int(now.timestamp() * 1000),
            "chainId": _SOLANA_CAIP2,
            "type": "ed25519",
        }
        return base64.b64encode(
            json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

    def build_payment_header(
        self,
        *,
        requirement: dict[str, Any],
        recent_blockhash: str,
        mint_decimals: int,
        token_program: str,
        max_amount_units: int | None = None,
    ) -> str:
        """Build the base64 ``X-402-Payment`` v2 envelope (PURE, no I/O).

        Constructs the four-instruction Solana ``VersionedTransaction`` for the
        x402 "exact" SVM scheme, partial-signs the client's slot, and base64
        encodes the v2 payment envelope. Performs no network I/O — the caller
        supplies ``recent_blockhash``, ``mint_decimals`` and ``token_program``
        (fetched live via :func:`fetch_solana_tx_context`).

        Validates ``network``/``asset``/``amount`` BEFORE signing — refuses to
        sign payloads that deviate from expectations. This is a security
        control: a misbehaving server could otherwise ask you to authorize a
        transfer to an attacker-controlled address.

        :param requirement: A single ``accepts`` entry whose ``network`` is
            either the bare string ``"solana"`` or
            :data:`SOLANA_MAINNET_CAIP2`; the value is echoed back verbatim.
            Expected keys: ``asset`` (mint), ``amount`` (string base units),
            ``payTo`` (recipient owner), and ``extra.feePayer`` (the
            facilitator). Optional ``extra.memo`` (UTF-8, <=256 bytes) — a
            random 16-byte hex memo is generated when absent. Optional
            ``scheme`` (defaults ``"exact"``).
        :param recent_blockhash: Base58 recent blockhash for the message.
        :param mint_decimals: The mint's decimals (byte 44 of the mint account
            data), encoded into the ``TransferChecked`` instruction.
        :param token_program: The mint's owning token program id (base58).
        :param max_amount_units: Optional cap in token base units; refuses to
            sign if ``int(requirement["amount"])`` exceeds it. ``None``
            disables the cap (not recommended in production).

        :return: Base64-encoded v2 ``X-402-Payment`` envelope string.

        :raises ValueError: On validation mismatch (network/asset/amount) or a
            malformed requirement.
        """
        # ── Validate the requirement shape (refuse to sign deviations) ──────
        network = requirement.get("network")
        if not is_solana_mainnet(network):
            raise ValueError(
                f"network mismatch: requirement has {network!r}, expected "
                f"{_SOLANA_BARE!r} or {SOLANA_MAINNET_CAIP2!r}; refusing to sign."
            )

        asset = requirement.get("asset")
        if not asset:
            raise ValueError(f"requirement missing 'asset': {requirement!r}")

        pay_to = requirement.get("payTo")
        if not pay_to:
            raise ValueError(f"requirement missing 'payTo': {requirement!r}")

        extra = requirement.get("extra") or {}
        if not isinstance(extra, dict):
            raise ValueError(f"requirement 'extra' must be an object: {extra!r}")
        fee_payer = extra.get("feePayer")
        if not fee_payer:
            raise ValueError(f"requirement missing 'extra.feePayer' (facilitator): {requirement!r}")

        amount_raw = requirement.get("amount")
        if amount_raw is None:
            raise ValueError(f"requirement missing 'amount': {requirement!r}")
        try:
            amount = int(amount_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"requirement amount {amount_raw!r} is not an integer string") from e
        if amount <= 0:
            raise ValueError(f"requirement amount must be positive, got {amount}")
        if max_amount_units is not None and amount > max_amount_units:
            raise ValueError(
                f"requirement amount {amount} exceeds cap {max_amount_units}; refusing to sign."
            )

        if not (0 <= mint_decimals <= 255):
            raise ValueError(f"mint_decimals out of range: {mint_decimals}")

        # ── Resolve pubkeys ────────────────────────────────────────────────
        token_program_pk = Pubkey.from_string(str(token_program))
        mint_pk = Pubkey.from_string(str(asset))
        pay_to_pk = Pubkey.from_string(str(pay_to))
        fee_payer_pk = Pubkey.from_string(str(fee_payer))
        owner_pk = self._keypair.pubkey()

        source_ata = _derive_ata(owner_pk, token_program_pk, mint_pk)
        dest_ata = _derive_ata(pay_to_pk, token_program_pk, mint_pk)

        # ── Build the four instructions (order is significant) ──────────────
        cb_program = Pubkey.from_string(_COMPUTE_BUDGET_PROGRAM_ID)
        memo_program = Pubkey.from_string(MEMO_PROGRAM_ID)

        ix_unit_limit = Instruction(
            cb_program,
            bytes([_CB_SET_UNIT_LIMIT]) + _COMPUTE_UNIT_LIMIT.to_bytes(4, "little"),
            [],
        )
        ix_unit_price = Instruction(
            cb_program,
            bytes([_CB_SET_UNIT_PRICE]) + _COMPUTE_UNIT_PRICE.to_bytes(8, "little"),
            [],
        )
        ix_transfer = Instruction(
            token_program_pk,
            bytes([_SPL_TRANSFER_CHECKED]) + amount.to_bytes(8, "little") + bytes([mint_decimals]),
            [
                AccountMeta(source_ata, False, True),  # source ATA (writable, non-signer)
                AccountMeta(mint_pk, False, False),  # mint (readonly)
                AccountMeta(dest_ata, False, True),  # dest ATA (writable, non-signer)
                AccountMeta(owner_pk, True, False),  # owner = payer wallet (readonly SIGNER)
            ],
        )

        memo_value = extra.get("memo")
        if memo_value:
            memo_bytes = str(memo_value).encode("utf-8")
            if len(memo_bytes) > _MAX_MEMO_BYTES:
                raise ValueError(f"extra.memo is {len(memo_bytes)} bytes; max {_MAX_MEMO_BYTES}")
        else:
            memo_bytes = binascii.hexlify(os.urandom(16))  # 32 hex bytes
        ix_memo = Instruction(memo_program, memo_bytes, [])

        # ── Compile MessageV0 with payer = feePayer (facilitator) ───────────
        blockhash = Hash.from_string(str(recent_blockhash))
        message = MessageV0.try_compile(
            fee_payer_pk,
            [ix_unit_limit, ix_unit_price, ix_transfer, ix_memo],
            [],
            blockhash,
        )

        # ── Partial-sign: feePayer placeholder at index 0, client at 1 ──────
        # ``bytes(MessageV0)`` omits the 0x80 wire-version prefix; the signed
        # payload includes it. Signers are ordered first in account_keys, so
        # account_keys[0] == feePayer and account_keys[1] == client wallet.
        msg_bytes = bytes([_MESSAGE_V0_PREFIX]) + bytes(message)
        client_sig = self._keypair.sign_message(msg_bytes)
        sigs = [Signature.default(), client_sig]
        tx = VersionedTransaction.populate(message, sigs)
        tx_b64 = base64.b64encode(bytes(tx)).decode("ascii")

        # x402 V2 payment envelope: the chosen requirement goes under
        # ``accepted`` (the V2 decoder validates ``accepted`` as a required
        # field — omitting it yields "could not extract payment info"). It must
        # match the canonical x402 ``PaymentRequirements`` shape: drop Venice's
        # challenge-metadata fields (``protocol``/``version``) and ensure
        # ``scheme`` is present, otherwise the schema validation fails with
        # "invalid payment header format". The inner SVM payload is
        # ``{transaction}`` (base64, standard alphabet).
        accepted = {k: v for k, v in requirement.items() if k not in ("protocol", "version")}
        accepted.setdefault("scheme", "exact")
        # V2 PaymentRequirements requires maxTimeoutSeconds; Venice's Solana
        # challenge omits it, so default to 300s (matching Venice's Base example).
        accepted.setdefault("maxTimeoutSeconds", 300)
        envelope: dict[str, Any] = {
            "x402Version": 2,
            "payload": {"transaction": tx_b64},
            "accepted": accepted,
        }
        return base64.b64encode(json.dumps(envelope, separators=(",", ":")).encode("utf-8")).decode(
            "ascii"
        )


async def fetch_solana_tx_context(
    rpc_url: str,
    mint: str,
    *,
    http: Any,
) -> tuple[str, int, str]:
    """Fetch the live transaction context needed to build a Solana payment.

    Performs two raw JSON-RPC calls over the supplied aiohttp session:

    1. ``getLatestBlockhash`` → recent blockhash (base58 str) for the message.
    2. ``getAccountInfo(mint, {encoding:"base64"})`` → the mint account's
       ``owner`` (the token program id) and ``data``; ``decimals`` is byte 44
       of the base64-decoded mint account data.

    :param rpc_url: Solana JSON-RPC endpoint.
    :param mint: The mint (asset) base58 address from the 402 requirement.
    :param http: An ``aiohttp.ClientSession`` to issue the POSTs with.

    :return: ``(recent_blockhash, mint_decimals, token_program)``.

    :raises RuntimeError: If the RPC returns an error or unexpected shape.
    """

    async def _rpc(method: str, params: list[Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        async with http.post(rpc_url, json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()
        if "error" in body:
            raise RuntimeError(f"Solana RPC {method} error: {body['error']!r}")
        result = body.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Solana RPC {method} unexpected result: {body!r}")
        return result

    blockhash_result = await _rpc("getLatestBlockhash", [{"commitment": "finalized"}])
    value = blockhash_result.get("value") or {}
    recent_blockhash = value.get("blockhash")
    if not recent_blockhash:
        raise RuntimeError(
            f"Solana RPC getLatestBlockhash returned no blockhash: {blockhash_result!r}"
        )

    account_result = await _rpc(
        "getAccountInfo", [mint, {"encoding": "base64", "commitment": "finalized"}]
    )
    account_value = account_result.get("value")
    if not isinstance(account_value, dict):
        raise RuntimeError(f"mint account {mint!r} not found on chain: {account_result!r}")
    token_program = account_value.get("owner")
    if not token_program:
        raise RuntimeError(f"mint account {mint!r} has no owner: {account_value!r}")
    data_field = account_value.get("data")
    # data is [base64_string, "base64"] for base64 encoding.
    data_b64 = data_field[0] if isinstance(data_field, list) else data_field
    if not data_b64:
        raise RuntimeError(f"mint account {mint!r} has no data: {account_value!r}")
    mint_data = base64.b64decode(data_b64)
    if len(mint_data) <= 44:
        raise RuntimeError(f"mint account data too short ({len(mint_data)} bytes) to read decimals")
    mint_decimals = mint_data[44]

    return str(recent_blockhash), int(mint_decimals), str(token_program)
