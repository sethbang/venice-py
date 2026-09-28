"""
Unit tests for :class:`venice_ai.resources.x402.X402` and
:class:`venice_ai.auth.x402.X402Auth`.

The tests use a deterministic throwaway private key — NOT a real wallet.
Nothing here touches real funds; every network call is mocked.
"""

import base64
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip the whole module if the x402 extra isn't installed.
pytest.importorskip("eth_account", reason="x402 extra not installed")
pytest.importorskip("siwe", reason="x402 extra not installed")

from venice_ai.auth.x402 import USDC_BASE_MAINNET, X402Auth
from venice_ai.exceptions import PaymentRequiredError
from venice_ai.resources.x402 import X402
from venice_ai.types.api.x402 import (
    X402BalanceData,
    X402BalanceResponse,
    X402TopUpData,
    X402TopUpResponse,
    X402TransactionsData,
    X402TransactionsPagination,
    X402TransactionsResponse,
)

# A deterministic, never-used-in-production throwaway key. Do NOT put anything
# of value on the derived wallet. Generated once for this test suite.
_THROWAWAY_KEY = "0x" + "a" * 63 + "b"


@pytest.fixture
def auth() -> X402Auth:
    return X402Auth(private_key=_THROWAWAY_KEY)


@pytest.fixture
def x402_resource() -> X402:
    client = MagicMock()
    return X402(client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# X402Auth
# ---------------------------------------------------------------------------


def test_x402_auth_exposes_wallet_address(auth: X402Auth) -> None:
    assert auth.wallet_address.startswith("0x")
    assert len(auth.wallet_address) == 42  # 0x + 40 hex


def test_x402_auth_build_header_is_base64_json(auth: X402Auth) -> None:
    header = auth.build_header(
        nonce="0123456789abcdef",
        now=datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
    )
    payload = json.loads(base64.b64decode(header).decode("utf-8"))

    assert payload["address"] == auth.wallet_address
    assert payload["chainId"] == 8453
    assert payload["signature"].startswith("0x")
    assert isinstance(payload["timestamp"], int)
    assert "Sign in to Venice API" in payload["message"]
    assert "outerface.venice.ai" in payload["message"]


def test_x402_auth_rejects_bad_nonce(auth: X402Auth) -> None:
    with pytest.raises(ValueError):
        auth.build_header(nonce="short")


def test_x402_auth_exposes_ttl_seconds_and_chain_id() -> None:
    a = X402Auth(private_key=_THROWAWAY_KEY)
    assert a.ttl_seconds == 600
    assert a.chain_id == 8453

    a2 = X402Auth(private_key=_THROWAWAY_KEY, ttl_seconds=1200, chain_id=1)
    assert a2.ttl_seconds == 1200
    assert a2.chain_id == 1


# ---------------------------------------------------------------------------
# X402Auth.build_payment_header
# ---------------------------------------------------------------------------


def _good_requirement() -> dict:
    """A canonical USDC-on-Base requirement matching what Venice's 402 returns."""
    return {
        "x402Version": 2,
        "protocol": "x402",
        "version": 2,
        "network": "eip155:8453",
        "asset": USDC_BASE_MAINNET,
        "amount": "5000000",
        "payTo": "0x2670B922ef37C7Df47158725C0CC407b5382293F",
        "scheme": "exact",
    }


def test_build_payment_header_envelope_shape(auth: X402Auth) -> None:
    """The signed envelope is base64(JSON) with the v2 schema fields."""
    header = auth.build_payment_header(
        _good_requirement(),
        max_amount_units=5_000_000,
        nonce="0x" + "1" * 64,
        now=datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC),
    )
    envelope = json.loads(base64.b64decode(header).decode("utf-8"))

    assert envelope["x402Version"] == 2
    # V2 PaymentPayload: scheme/network live under ``accepted`` (see
    # test_build_payment_header_uses_v2_accepted_wrapper), not top-level.
    assert envelope["accepted"]["scheme"] == "exact"
    assert envelope["accepted"]["network"] == "eip155:8453"

    auth_block = envelope["payload"]["authorization"]
    assert auth_block["from"] == auth.wallet_address
    assert auth_block["to"] == "0x2670B922ef37C7Df47158725C0CC407b5382293F"
    assert auth_block["value"] == "5000000"
    assert auth_block["validAfter"] == "0"
    assert auth_block["validBefore"].isdigit()
    assert auth_block["nonce"] == "0x" + "1" * 64

    # Signature is 65-byte hex (130 chars + 0x)
    sig = envelope["payload"]["signature"]
    assert sig.startswith("0x")
    assert len(sig) == 132


def test_build_payment_header_uses_v2_accepted_wrapper(auth: X402Auth) -> None:
    """The EVM envelope is the x402 V2 PaymentPayload {x402Version, payload, accepted}.

    The chosen requirement is echoed under ``accepted`` (canonical
    PaymentRequirements: scheme/network/asset/amount/payTo/maxTimeoutSeconds/extra
    — protocol & version stripped). Mirrors the Solana settlement fix
    (744cc16): Venice's shared V2 facilitator rejects the old flat
    ``{x402Version, scheme, network, payload}`` shape and any ``accepted``
    block missing the required ``maxTimeoutSeconds`` with a 400.
    """
    header = auth.build_payment_header(
        _good_requirement(),
        max_amount_units=5_000_000,
        nonce="0x" + "1" * 64,
        now=datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC),
    )
    envelope = json.loads(base64.b64decode(header).decode("utf-8"))

    assert envelope["x402Version"] == 2
    # The flat top-level scheme/network must be gone (rejected by the V2 decoder).
    assert "scheme" not in envelope
    assert "network" not in envelope

    accepted = envelope["accepted"]
    assert accepted["scheme"] == "exact"
    assert accepted["network"] == "eip155:8453"  # CAIP-2, unchanged (X402-03 is correct)
    assert accepted["asset"] == USDC_BASE_MAINNET
    assert accepted["amount"] == "5000000"
    assert accepted["payTo"] == "0x2670B922ef37C7Df47158725C0CC407b5382293F"
    assert accepted["maxTimeoutSeconds"] == 300  # required by V2 PaymentRequirements
    # Venice's challenge-metadata fields are stripped.
    assert "protocol" not in accepted
    assert "version" not in accepted

    # The signed authorization stays under payload (unchanged).
    auth_block = envelope["payload"]["authorization"]
    assert auth_block["from"] == auth.wallet_address
    assert envelope["payload"]["signature"].startswith("0x")


def test_build_payment_header_signature_is_deterministic(auth: X402Auth) -> None:
    """Same key + nonce + now → same signature (deterministic ECDSA precondition)."""
    args = {
        "max_amount_units": 5_000_000,
        "nonce": "0x" + "2" * 64,
        "now": datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    }
    h1 = auth.build_payment_header(_good_requirement(), **args)
    h2 = auth.build_payment_header(_good_requirement(), **args)
    assert h1 == h2


def test_build_payment_header_random_nonce_when_omitted(auth: X402Auth) -> None:
    h1 = auth.build_payment_header(_good_requirement(), max_amount_units=5_000_000)
    h2 = auth.build_payment_header(_good_requirement(), max_amount_units=5_000_000)
    # Different nonces → different headers
    assert h1 != h2


def test_build_payment_header_rejects_wrong_network(auth: X402Auth) -> None:
    bad = {**_good_requirement(), "network": "eip155:1"}  # Ethereum mainnet, not Base
    with pytest.raises(ValueError, match="network mismatch"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_unsupported_chain_family(auth: X402Auth) -> None:
    bad = {**_good_requirement(), "network": "solana:101"}
    with pytest.raises(ValueError, match="network mismatch"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_wrong_asset(auth: X402Auth) -> None:
    bad = {**_good_requirement(), "asset": "0x" + "b" * 40}
    with pytest.raises(ValueError, match="asset mismatch"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_accepts_validate_asset_override(auth: X402Auth) -> None:
    """Caller can override the default USDC-on-Base asset check."""
    other_token = "0x" + "c" * 40
    req = {**_good_requirement(), "asset": other_token}
    header = auth.build_payment_header(
        req,
        validate_asset=other_token,
        max_amount_units=5_000_000,
        nonce="0x" + "3" * 64,
        now=datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC),
    )
    # Just assert it produced a header (signature will be invalid for non-USDC,
    # but the helper doesn't enforce token type beyond validate_asset).
    envelope = json.loads(base64.b64decode(header).decode("utf-8"))
    assert envelope["payload"]["authorization"]["to"] == _good_requirement()["payTo"]


def test_build_payment_header_rejects_over_cap(auth: X402Auth) -> None:
    with pytest.raises(ValueError, match="exceeds cap"):
        auth.build_payment_header(_good_requirement(), max_amount_units=1_000_000)


def test_build_payment_header_rejects_missing_pay_to(auth: X402Auth) -> None:
    bad = {k: v for k, v in _good_requirement().items() if k != "payTo"}
    with pytest.raises(ValueError, match="payTo"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_missing_amount(auth: X402Auth) -> None:
    bad = {k: v for k, v in _good_requirement().items() if k != "amount"}
    with pytest.raises(ValueError, match="amount"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_missing_asset(auth: X402Auth) -> None:
    bad = {k: v for k, v in _good_requirement().items() if k != "asset"}
    with pytest.raises(ValueError, match="asset"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_zero_amount(auth: X402Auth) -> None:
    bad = {**_good_requirement(), "amount": "0"}
    with pytest.raises(ValueError, match="positive"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_non_integer_amount(auth: X402Auth) -> None:
    bad = {**_good_requirement(), "amount": "five"}
    with pytest.raises(ValueError, match="not an integer"):
        auth.build_payment_header(bad, max_amount_units=5_000_000)


def test_build_payment_header_rejects_bad_nonce_length(auth: X402Auth) -> None:
    with pytest.raises(ValueError, match="32 bytes hex"):
        auth.build_payment_header(_good_requirement(), max_amount_units=5_000_000, nonce="0xshort")


def test_build_payment_header_rejects_non_hex_nonce(auth: X402Auth) -> None:
    with pytest.raises(ValueError, match="not valid hex"):
        auth.build_payment_header(_good_requirement(), max_amount_units=5_000_000, nonce="g" * 64)


def test_build_payment_header_valid_for_seconds_reflected_in_validBefore(
    auth: X402Auth,
) -> None:
    fixed_now = datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC)
    fixed_ts = int(fixed_now.timestamp())
    header = auth.build_payment_header(
        _good_requirement(),
        max_amount_units=5_000_000,
        valid_for_seconds=300,
        nonce="0x" + "4" * 64,
        now=fixed_now,
    )
    envelope = json.loads(base64.b64decode(header).decode("utf-8"))
    assert envelope["payload"]["authorization"]["validBefore"] == str(fixed_ts + 300)


def test_build_payment_header_no_cap_signs_anything(auth: X402Auth) -> None:
    """When max_amount_units=None, no cap is enforced (default)."""
    huge = {**_good_requirement(), "amount": "1000000000000"}  # $1M USDC
    header = auth.build_payment_header(
        huge,
        max_amount_units=None,  # explicitly disable cap
        nonce="0x" + "5" * 64,
        now=datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC),
    )
    envelope = json.loads(base64.b64decode(header).decode("utf-8"))
    assert envelope["payload"]["authorization"]["value"] == "1000000000000"


# ---------------------------------------------------------------------------
# X402.balance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_balance_uses_wallet_address_in_path(x402_resource: X402, auth: X402Auth) -> None:
    fake = X402BalanceResponse(
        success=True,
        data=X402BalanceData(
            walletAddress=auth.wallet_address,
            balanceUsd=12.5,
            canConsume=True,
            minimumTopUpUsd=5,
            suggestedTopUpUsd=10,
            diemBalanceUsd=None,
        ),
    )
    x402_resource._client.get = AsyncMock(return_value=fake)  # type: ignore[attr-defined]

    result = await x402_resource.balance(auth=auth)

    assert isinstance(result, X402BalanceResponse)
    assert result.data.walletAddress == auth.wallet_address

    call = x402_resource._client.get.call_args  # type: ignore[attr-defined]
    assert call.args[0] == f"x402/balance/{auth.wallet_address}"
    assert call.kwargs["cast_to"] is X402BalanceResponse
    # The wallet is named rather than pre-signed into headers, so the client can
    # sign a fresh envelope for every attempt. The header on the wire is covered
    # in tests/unit/test_client_siwe_retry_resign.py.
    assert call.kwargs["siwe_auth"] is auth
    assert "headers" not in call.kwargs


# ---------------------------------------------------------------------------
# X402.transactions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transactions_uses_wallet_address_in_path(
    x402_resource: X402, auth: X402Auth
) -> None:
    fake = X402TransactionsResponse(
        success=True,
        data=X402TransactionsData(
            walletAddress=auth.wallet_address,
            currentBalance=12.35,
            transactions=[],
            pagination=X402TransactionsPagination(limit=50, offset=0, hasMore=False),
        ),
    )
    x402_resource._client.get = AsyncMock(return_value=fake)  # type: ignore[attr-defined]

    result = await x402_resource.transactions(auth=auth)

    assert isinstance(result, X402TransactionsResponse)
    assert result.data.walletAddress == auth.wallet_address

    call = x402_resource._client.get.call_args  # type: ignore[attr-defined]
    assert call.args[0] == f"x402/transactions/{auth.wallet_address}"
    assert call.kwargs["siwe_auth"] is auth
    # No pagination params by default — server default (limit=50, offset=0) applies.
    assert call.kwargs.get("params") is None


@pytest.mark.asyncio
async def test_transactions_forwards_pagination_params(x402_resource: X402, auth: X402Auth) -> None:
    fake = X402TransactionsResponse(
        success=True,
        data=X402TransactionsData(
            walletAddress=auth.wallet_address,
            currentBalance=0.0,
            transactions=[],
            pagination=X402TransactionsPagination(limit=25, offset=100, hasMore=False),
        ),
    )
    x402_resource._client.get = AsyncMock(return_value=fake)  # type: ignore[attr-defined]

    await x402_resource.transactions(auth=auth, limit=25, offset=100)

    call = x402_resource._client.get.call_args  # type: ignore[attr-defined]
    assert call.kwargs["params"] == {"limit": 25, "offset": 100}


@pytest.mark.asyncio
async def test_transactions_forwards_only_limit(x402_resource: X402, auth: X402Auth) -> None:
    fake = X402TransactionsResponse(
        success=True,
        data=X402TransactionsData(
            walletAddress=auth.wallet_address,
            currentBalance=0.0,
            transactions=[],
            pagination=X402TransactionsPagination(limit=10, offset=0, hasMore=False),
        ),
    )
    x402_resource._client.get = AsyncMock(return_value=fake)  # type: ignore[attr-defined]

    await x402_resource.transactions(auth=auth, limit=10)

    call = x402_resource._client.get.call_args  # type: ignore[attr-defined]
    assert call.kwargs["params"] == {"limit": 10}


# ---------------------------------------------------------------------------
# X402.top_up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_up_empty_post_uses_bearer(x402_resource: X402) -> None:
    fake = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress="0xtest",
            amountCredited=10,
            newBalance=22.5,
            paymentId="payment_test",
        ),
    )
    x402_resource._client.post = AsyncMock(return_value=fake)  # type: ignore[attr-defined]

    result = await x402_resource.top_up()

    assert result.data.amountCredited == 10

    call = x402_resource._client.post.call_args  # type: ignore[attr-defined]
    assert call.args[0] == "x402/top-up"
    assert call.kwargs["json_data"] == {}
    # No X-402-Payment header when not provided; Bearer auth is the default.
    assert call.kwargs["headers"] is None


@pytest.mark.asyncio
async def test_top_up_with_payment_header(x402_resource: X402) -> None:
    fake = X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress="0xtest",
            amountCredited=5,
            newBalance=17.5,
            paymentId="payment_test2",
        ),
    )
    x402_resource._client.post = AsyncMock(return_value=fake)  # type: ignore[attr-defined]

    await x402_resource.top_up(payment_header="eyJ4NDAyVmVyc2lvbiI6Mn0=")

    call = x402_resource._client.post.call_args  # type: ignore[attr-defined]
    assert call.kwargs["headers"] == {"X-402-Payment": "eyJ4NDAyVmVyc2lvbiI6Mn0="}


# ---------------------------------------------------------------------------
# X402.top_up_with
# ---------------------------------------------------------------------------


def _payment_required_error(body: dict | None) -> PaymentRequiredError:
    """Build a PaymentRequiredError with the structured body Venice's 402 returns."""
    fake_response = MagicMock()
    fake_response.status_code = 402
    return PaymentRequiredError(
        "Payment required",
        response=fake_response,
        body=body,
    )


def _settle_response(amount: float = 5.0) -> X402TopUpResponse:
    return X402TopUpResponse(
        success=True,
        data=X402TopUpData(
            walletAddress="0xtest",
            amountCredited=amount,
            newBalance=amount,
            paymentId=f"x402-test-{int(amount)}",
        ),
    )


@pytest.mark.asyncio
async def test_top_up_with_happy_path(x402_resource: X402, auth: X402Auth) -> None:
    probe_err = _payment_required_error({"x402Version": 2, "accepts": [_good_requirement()]})
    settled = _settle_response(amount=5.0)
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])  # type: ignore[method-assign]

    result = await x402_resource.top_up_with(auth=auth, amount_usdc=5.0)

    assert result.data.amountCredited == 5.0
    assert result.data.paymentId == "x402-test-5"

    assert x402_resource.top_up.call_count == 2  # type: ignore[attr-defined]
    first, second = x402_resource.top_up.call_args_list  # type: ignore[attr-defined]
    # Probe: no payment_header
    assert first.kwargs == {}
    # Settle: signed payment_header
    assert "payment_header" in second.kwargs
    assert isinstance(second.kwargs["payment_header"], str)
    # Verify the header is valid base64-JSON with the auth's wallet
    envelope = json.loads(base64.b64decode(second.kwargs["payment_header"]).decode())
    assert envelope["payload"]["authorization"]["from"] == auth.wallet_address


@pytest.mark.asyncio
async def test_top_up_with_caps_via_max_amount(x402_resource: X402, auth: X402Auth) -> None:
    """Server requires more than max_amount_usdc → ValueError, no second call."""
    probe_err = _payment_required_error(
        {"x402Version": 2, "accepts": [{**_good_requirement(), "amount": "10000000"}]}
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Server requires"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=5.0, max_amount_usdc=5.0)

    # Only the probe ran; no signing, no settle.
    assert x402_resource.top_up.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_top_up_with_rejects_below_minimum(x402_resource: X402, auth: X402Auth) -> None:
    """amount_usdc < server minimum → ValueError, no signing."""
    probe_err = _payment_required_error(
        {"x402Version": 2, "accepts": [_good_requirement()]}  # server requires 5 USDC
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])  # type: ignore[method-assign]

    # Setting max_amount_usdc=10 keeps the cap check passing so we exercise
    # the below-minimum branch specifically.
    with pytest.raises(ValueError, match="below server minimum"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=3.0, max_amount_usdc=10.0)

    assert x402_resource.top_up.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_top_up_with_rejects_non_positive_amount(x402_resource: X402, auth: X402Auth) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=-1.0)


@pytest.mark.asyncio
async def test_top_up_with_rejects_inverted_max(x402_resource: X402, auth: X402Auth) -> None:
    """max_amount_usdc < amount_usdc is nonsense — fail fast."""
    with pytest.raises(ValueError, match="cannot be less than"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=10.0, max_amount_usdc=5.0)


@pytest.mark.asyncio
async def test_top_up_with_unexpected_success_raises(x402_resource: X402, auth: X402Auth) -> None:
    """If the empty probe somehow succeeds, treat as a server bug."""
    unexpected = _settle_response(amount=0.0)
    x402_resource.top_up = AsyncMock(side_effect=[unexpected])  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="succeeded unexpectedly"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=5.0)


@pytest.mark.asyncio
async def test_top_up_with_no_accepts_in_402_body(x402_resource: X402, auth: X402Auth) -> None:
    """402 body lacking 'accepts' → RuntimeError."""
    probe_err = _payment_required_error({"x402Version": 2})  # no 'accepts' key
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no 'accepts'"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=5.0)


@pytest.mark.asyncio
async def test_top_up_with_no_matching_requirement(x402_resource: X402, auth: X402Auth) -> None:
    """All accepts entries are on a different chain → RuntimeError."""
    probe_err = _payment_required_error(
        {
            "x402Version": 2,
            "accepts": [
                {**_good_requirement(), "network": "eip155:1"},  # ETH mainnet, not Base
                {**_good_requirement(), "network": "solana:101"},
            ],
        }
    )
    x402_resource.top_up = AsyncMock(side_effect=[probe_err])  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no 'exact' requirement on eip155:8453"):
        await x402_resource.top_up_with(auth=auth, amount_usdc=5.0)


@pytest.mark.asyncio
async def test_top_up_with_handles_alternate_scheme_names(
    x402_resource: X402, auth: X402Auth
) -> None:
    """The matcher accepts 'evm-exact' and 'evm/exact' as scheme variants."""
    probe_err = _payment_required_error(
        {
            "x402Version": 2,
            "accepts": [{**_good_requirement(), "scheme": "evm-exact"}],
        }
    )
    settled = _settle_response(amount=5.0)
    x402_resource.top_up = AsyncMock(side_effect=[probe_err, settled])  # type: ignore[method-assign]

    result = await x402_resource.top_up_with(auth=auth, amount_usdc=5.0)
    assert result.data.amountCredited == 5.0


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------


def test_client_exposes_x402_namespace() -> None:
    from venice_ai import create_test_venice_client

    client = create_test_venice_client(api_key="vn_unit_test_X402Wiring_12345678")
    try:
        assert isinstance(client.x402, X402)
    finally:
        import asyncio

        asyncio.run(client.close())
