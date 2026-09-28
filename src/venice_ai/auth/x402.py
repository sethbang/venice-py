"""
x402 Sign-In-With-X (SIWE / EIP-4361) authentication for Venice's
``/x402/*`` endpoints.

The API requires an ``X-Sign-In-With-X`` header — a base64-encoded JSON
payload proving ownership of an Ethereum wallet on Base (chain 8453).

Usage::

    from venice_ai.auth.x402 import X402Auth

    auth = X402Auth(private_key="0xabc...")
    # auth.wallet_address — checksummed address derived from the key
    # auth.build_header() — returns the base64 value for ``X-Sign-In-With-X``

This module is optional and requires ``pip install 'venice-py[x402]'``.
"""

from __future__ import annotations

import base64
import json
import secrets
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any, cast

try:
    from eth_account import Account
    from eth_account.messages import encode_typed_data

    # siwe compiles its ABNF grammar at import time, and the ``abnf`` package
    # reports the RFC core rules it re-declares as ``GrammarWarning``s. The
    # filter is scoped to this import so user warning settings stay untouched.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"rule '.*' redefines", module=r"abnf\.")
        from siwe import SiweMessage
except ImportError as exc:  # pragma: no cover - import-time only
    raise ImportError(
        "The x402 auth helpers require the ``x402`` extra. "
        "Install it with: pip install 'venice-py[x402]'"
    ) from exc

__all__ = ["X402Auth", "USDC_BASE_MAINNET"]

# Constants required by the Venice x402 SIWE spec.
# See api-reference/endpoint/x402/balance.md.
_SIWE_DOMAIN = "outerface.venice.ai"
_SIWE_URI = "https://outerface.venice.ai"
_SIWE_STATEMENT = "Sign in to Venice API"
_SIWE_VERSION = "1"
_BASE_CHAIN_ID = 8453

# USDC contract on Base mainnet. The EIP-712 domain values
# (``USD Coin`` / ``"2"``) are USDC-specific; other ERC-20s would need
# different domain parameters. Today Venice's x402 settlement only
# accepts USDC on Base, so :meth:`X402Auth.build_payment_header` hardcodes
# this domain. Future extensions can take ``eip712_domain_*`` overrides.
USDC_BASE_MAINNET = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_USDC_EIP712_NAME = "USD Coin"
_USDC_EIP712_VERSION = "2"


class X402Auth:
    """Builds ``X-Sign-In-With-X`` tokens for Venice's x402 endpoints.

    :param private_key: Ethereum private key as hex string (``0x``-prefixed
        or bare 64 hex chars). **Never share or commit this.**
    :param chain_id: Chain ID; defaults to Base mainnet (``8453``).
    :param ttl_seconds: SIWE message TTL from ``issuedAt`` to ``expirationTime``.
        Defaults to 600 s (10 min) per the Venice docs.
    """

    def __init__(
        self,
        *,
        private_key: str,
        chain_id: int = _BASE_CHAIN_ID,
        ttl_seconds: int = 600,
    ) -> None:
        self._account = Account.from_key(private_key)
        self._chain_id = chain_id
        self._ttl_seconds = ttl_seconds

    @property
    def wallet_address(self) -> str:
        """Checksummed Ethereum address derived from the private key."""
        return cast(str, self._account.address)

    @property
    def ttl_seconds(self) -> int:
        """SIWE token TTL in seconds (default 600)."""
        return self._ttl_seconds

    @property
    def chain_id(self) -> int:
        """EVM chain ID this auth was constructed for (default 8453, Base mainnet)."""
        return self._chain_id

    def build_header(self, *, nonce: str | None = None, now: datetime | None = None) -> str:
        """Build the base64-encoded ``X-Sign-In-With-X`` header value.

        :param nonce: Optional 16-character hex nonce; a cryptographically
            random one is generated if omitted.
        :param now: Override the ``issuedAt`` timestamp (UTC). Useful for
            testing / cassette recording. Defaults to ``datetime.now(UTC)``.
        """
        if nonce is None:
            nonce = secrets.token_hex(8)  # 16 hex chars
        elif len(nonce) != 16:
            raise ValueError("nonce must be a 16-character hex string")

        if now is None:
            now = datetime.now(UTC)

        issued_at = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        expiration = (now + timedelta(seconds=self._ttl_seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # The python siwe library takes snake_case field names; the HTTP
        # header JSON uses camelCase (see _header_obj below). SiweMessage's
        # field types are Pydantic NewTypes / Enums that accept plain strings
        # at runtime through validators, so cast for the type checker.
        message = SiweMessage(
            domain=_SIWE_DOMAIN,
            address=cast(Any, self.wallet_address),
            statement=_SIWE_STATEMENT,
            uri=_SIWE_URI,
            version=cast(Any, _SIWE_VERSION),
            chain_id=self._chain_id,
            nonce=nonce,
            issued_at=cast(Any, issued_at),
            expiration_time=cast(Any, expiration),
        )
        prepared = message.prepare_message()

        signed = self._account.sign_message(_encode_siwe_text(prepared))
        signature_hex = signed.signature.hex()
        if not signature_hex.startswith("0x"):
            signature_hex = "0x" + signature_hex

        header_obj: dict[str, Any] = {
            "address": self.wallet_address,
            "message": prepared,
            "signature": signature_hex,
            "timestamp": int(now.timestamp() * 1000),
            "chainId": self._chain_id,
        }
        return base64.b64encode(
            json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

    def build_payment_header(
        self,
        requirement: dict[str, Any],
        *,
        validate_network: str = "eip155:8453",
        validate_asset: str | None = None,
        max_amount_units: int | None = None,
        valid_for_seconds: int = 600,
        nonce: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Build the base64-encoded ``X-402-Payment`` v2 envelope from a 402 requirement.

        Constructs the EIP-712 typed data for USDC ``transferWithAuthorization``,
        signs with this auth's private key, and base64-encodes the v2 payment
        envelope ready for :meth:`venice_ai.resources.x402.X402.top_up`.

        Validates ``network``, ``asset``, and ``amount`` BEFORE signing —
        refuses to sign payloads that deviate from expectations. This is a
        security control: a misbehaving or malicious server could otherwise
        ask you to sign a transfer to an attacker-controlled address.

        Currently supports **only USDC on Base mainnet** (``eip155:8453``).
        Other tokens / chains would need different EIP-712 domain parameters
        (``name``, ``version``); extending this helper to accept overrides is
        future work.

        :param requirement: A single requirement dict from
            ``PaymentRequiredError.body["accepts"][i]``. Expected keys:
            ``network`` (CAIP-2 chain id), ``asset`` (ERC-20 contract
            address), ``amount`` (string in token base units), ``payTo``
            (recipient address). Optional ``scheme`` (defaults to ``"exact"``).
        :param validate_network: Refuse if ``requirement["network"]`` doesn't
            match. Default ``"eip155:8453"`` (Base mainnet).
        :param validate_asset: Optional asset contract address (case-insensitive).
            When ``None`` (default), the asset is checked against
            :data:`USDC_BASE_MAINNET`. Pass an explicit address to allow
            another asset; you'll also need EIP-712 domain support which is
            not currently exposed.
        :param max_amount_units: Optional cap in token base units (USDC has
            6 decimals, so ``5_000_000`` = 5 USDC). Refuses to sign if
            ``int(requirement["amount"])`` exceeds this. ``None`` disables
            the cap (not recommended in production).
        :param valid_for_seconds: ``validBefore`` window from ``now``.
            Default 600s (10 min) — long enough to cover network round-trip
            and server processing, short enough to limit replay surface.
        :param nonce: Optional 32-byte hex nonce (with or without ``0x``
            prefix). Generated cryptographically when ``None``. Override only
            for testing / cassette recording. **Re-using a nonce is
            rejected server-side** — generate a fresh one per call.
        :param now: Optional UTC datetime override for ``validBefore``
            computation. Defaults to ``datetime.now(UTC)``. Override for
            testing.

        :return: Base64-encoded v2 ``X-402-Payment`` envelope string.

        :raises ValueError: On validation mismatch (network, asset, amount)
            or malformed ``requirement`` (missing required keys).
        :raises ImportError: If the ``[x402]`` extra isn't installed (raised
            at module import time, not here, but documented for completeness).

        Example::

            from venice_ai.auth.x402 import X402Auth
            from venice_ai.exceptions import PaymentRequiredError

            auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
            try:
                await client.x402.top_up()
            except PaymentRequiredError as e:
                requirement = e.body["accepts"][0]
                header = auth.build_payment_header(
                    requirement,
                    max_amount_units=10_000_000,  # cap at $10
                )
                result = await client.x402.top_up(payment_header=header)
                print(result.data.amountCredited)
        """
        # ── Validate the requirement shape ─────────────────────────────
        network = requirement.get("network")
        if network != validate_network:
            raise ValueError(
                f"network mismatch: requirement has {network!r}, expected {validate_network!r}"
            )
        chain_family, _, chain_id_str = str(network).partition(":")
        if chain_family != "eip155" or not chain_id_str:
            raise ValueError(
                f"unsupported network family: {network!r} (only eip155:* CAIP-2 chains are supported)"
            )
        chain_id = int(chain_id_str)

        asset = requirement.get("asset")
        if not asset:
            raise ValueError(f"requirement missing 'asset': {requirement!r}")
        expected_asset = validate_asset if validate_asset is not None else USDC_BASE_MAINNET
        if str(asset).lower() != expected_asset.lower():
            raise ValueError(
                f"asset mismatch: requirement has {asset!r}, expected {expected_asset!r} "
                "(currently only USDC on Base is supported by build_payment_header; pass "
                "validate_asset= to override the validation)"
            )

        pay_to = requirement.get("payTo")
        if not pay_to:
            raise ValueError(f"requirement missing 'payTo': {requirement!r}")

        amount_raw = requirement.get("amount")
        if amount_raw is None:
            raise ValueError(f"requirement missing 'amount': {requirement!r}")
        try:
            value_units = int(amount_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"requirement amount {amount_raw!r} is not an integer string") from e
        if value_units <= 0:
            raise ValueError(f"requirement amount must be positive, got {value_units}")
        if max_amount_units is not None and value_units > max_amount_units:
            raise ValueError(
                f"requirement amount {value_units} (= ${value_units / 1_000_000} USDC) "
                f"exceeds cap {max_amount_units} (= ${max_amount_units / 1_000_000} USDC); "
                "refusing to sign."
            )

        # ── Build EIP-712 typed data (USDC TransferWithAuthorization) ──
        if nonce is None:
            nonce_hex = "0x" + secrets.token_hex(32)
        else:
            stripped = nonce[2:] if nonce.startswith("0x") else nonce
            if len(stripped) != 64:
                raise ValueError(
                    "nonce must be 32 bytes hex (64 hex chars, with or without 0x prefix)"
                )
            try:
                int(stripped, 16)
            except ValueError as e:
                raise ValueError(f"nonce {nonce!r} is not valid hex") from e
            nonce_hex = "0x" + stripped

        if now is None:
            now = datetime.now(UTC)
        valid_after = 0
        valid_before = int(now.timestamp()) + valid_for_seconds

        typed_data: dict[str, Any] = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "TransferWithAuthorization": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "validAfter", "type": "uint256"},
                    {"name": "validBefore", "type": "uint256"},
                    {"name": "nonce", "type": "bytes32"},
                ],
            },
            "primaryType": "TransferWithAuthorization",
            "domain": {
                "name": _USDC_EIP712_NAME,
                "version": _USDC_EIP712_VERSION,
                "chainId": chain_id,
                "verifyingContract": str(asset),
            },
            "message": {
                "from": self.wallet_address,
                "to": str(pay_to),
                "value": value_units,
                "validAfter": valid_after,
                "validBefore": valid_before,
                "nonce": nonce_hex,
            },
        }

        signed = self._account.sign_message(encode_typed_data(full_message=typed_data))
        sig_hex = signed.signature.hex()
        if not sig_hex.startswith("0x"):
            sig_hex = "0x" + sig_hex

        # x402 V2 payment envelope: the chosen requirement goes under
        # ``accepted`` (the V2 decoder validates ``accepted`` as a required
        # field — omitting it yields "could not extract payment info"). It must
        # match the canonical x402 ``PaymentRequirements`` shape: drop Venice's
        # challenge-metadata fields (``protocol``/``version``) and ensure
        # ``scheme`` is present, otherwise the schema validation fails with
        # "invalid payment header format". Mirrors the Solana settlement
        # envelope; the flat top-level ``{scheme, network}`` shape and an
        # ``accepted`` block missing ``maxTimeoutSeconds`` are both rejected
        # 400 by Venice's shared V2 facilitator.
        accepted = {k: v for k, v in requirement.items() if k not in ("protocol", "version")}
        accepted.setdefault("scheme", "exact")
        accepted["network"] = validate_network
        # V2 PaymentRequirements requires maxTimeoutSeconds; default to 300s
        # (matching Venice's Base example) when the challenge omits it.
        accepted.setdefault("maxTimeoutSeconds", 300)
        envelope: dict[str, Any] = {
            "x402Version": 2,
            "payload": {
                "signature": sig_hex,
                "authorization": {
                    "from": self.wallet_address,
                    "to": str(pay_to),
                    "value": str(value_units),
                    "validAfter": str(valid_after),
                    "validBefore": str(valid_before),
                    "nonce": nonce_hex,
                },
            },
            "accepted": accepted,
        }
        return base64.b64encode(json.dumps(envelope, separators=(",", ":")).encode("utf-8")).decode(
            "ascii"
        )


def _encode_siwe_text(text: str) -> Any:
    """Wrap a SIWE prepared message in the EIP-191 encoding that
    ``Account.sign_message`` expects."""
    from eth_account.messages import encode_defunct

    return encode_defunct(text=text)
