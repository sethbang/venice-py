"""
Unit tests for :class:`venice_ai.resources.crypto.Crypto`.

Exercises the two ``/crypto/rpc/*`` endpoints documented at
``api-reference/endpoint/crypto/``: networks discovery and JSON-RPC
forwarder (single + batch).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from venice_ai.resources.crypto import Crypto
from venice_ai.types.api.crypto import (
    BatchJsonRpcResponse,
    CryptoNetworksResponse,
    JsonRpcRequest,
    JsonRpcResponse,
)


def _make_fake_aiohttp_response(
    *,
    body: Any,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a MagicMock(spec=aiohttp.ClientResponse) that mimics the bits
    used by ``Crypto.batch_rpc``: ``await response.json()``, ``release()``,
    and a ``headers`` mapping. ``isinstance`` against ``aiohttp.ClientResponse``
    succeeds because of the ``spec`` argument.
    """
    fake = MagicMock(spec=aiohttp.ClientResponse)
    fake.json = AsyncMock(return_value=body)
    fake.release = MagicMock()
    fake.headers = headers if headers is not None else {}
    return fake


@pytest.fixture
def crypto() -> Crypto:
    client = MagicMock()
    return Crypto(client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# networks()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_networks_returns_slug_list(crypto: Crypto) -> None:
    crypto._client.get = AsyncMock(  # type: ignore[attr-defined]
        return_value=CryptoNetworksResponse(
            networks=["ethereum-mainnet", "base-mainnet", "polygon-mainnet"],
        )
    )
    slugs = await crypto.networks()

    assert slugs == ["ethereum-mainnet", "base-mainnet", "polygon-mainnet"]
    call_args = crypto._client.get.call_args  # type: ignore[attr-defined]
    assert call_args.args[0] == "crypto/rpc/networks"
    assert call_args.kwargs["cast_to"] is CryptoNetworksResponse


# ---------------------------------------------------------------------------
# rpc()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rpc_posts_jsonrpc_request_to_network(crypto: Crypto) -> None:
    crypto._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=JsonRpcResponse(jsonrpc="2.0", id=1, result="0x1", error=None)
    )
    result = await crypto.rpc(
        network="ethereum-mainnet",
        method="eth_chainId",
        params=[],
        id=1,
    )

    assert result.result == "0x1"
    assert result.error is None
    call_args = crypto._client.post.call_args  # type: ignore[attr-defined]
    assert call_args.args[0] == "crypto/rpc/ethereum-mainnet"
    assert call_args.kwargs["json_data"] == {
        "method": "eth_chainId",
        "jsonrpc": "2.0",
        "params": [],
        "id": 1,
    }
    assert call_args.kwargs["cast_to"] is JsonRpcResponse
    # No idempotency header by default
    key = call_args.kwargs["headers"]["Idempotency-Key"]
    assert key.startswith("venice-py-")


@pytest.mark.asyncio
async def test_rpc_omits_optional_fields_when_none(crypto: Crypto) -> None:
    crypto._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=JsonRpcResponse(jsonrpc="2.0", id=1, result=None, error=None)
    )
    await crypto.rpc(network="ethereum-mainnet", method="net_version", params=None, id=None)

    body = crypto._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    # exclude_none should drop params and id
    assert body == {"method": "net_version", "jsonrpc": "2.0"}


@pytest.mark.asyncio
async def test_rpc_attaches_idempotency_key(crypto: Crypto) -> None:
    crypto._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=JsonRpcResponse(jsonrpc="2.0", id=1, result="0x1", error=None)
    )
    await crypto.rpc(
        network="ethereum-mainnet",
        method="eth_chainId",
        params=[],
        id=1,
        idempotency_key="abc-123_DEF",
    )

    headers = crypto._client.post.call_args.kwargs["headers"]  # type: ignore[attr-defined]
    assert headers == {"Idempotency-Key": "abc-123_DEF"}


@pytest.mark.asyncio
async def test_rpc_rejects_invalid_idempotency_key(crypto: Crypto) -> None:
    with pytest.raises(ValueError, match="idempotency_key"):
        await crypto.rpc(
            network="ethereum-mainnet",
            method="eth_chainId",
            idempotency_key="not valid (spaces!)",
        )


@pytest.mark.asyncio
async def test_rpc_surfaces_jsonrpc_error_object(crypto: Crypto) -> None:
    crypto._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=JsonRpcResponse.model_validate(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32602, "message": "invalid params"},
            }
        )
    )
    result = await crypto.rpc(network="ethereum-mainnet", method="eth_getBalance", params=["bad"])

    assert result.result is None
    assert result.error is not None
    assert result.error.code == -32602
    assert result.error.message == "invalid params"


# ---------------------------------------------------------------------------
# batch_rpc()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_rpc_sends_json_array(crypto: Crypto) -> None:
    crypto._client._request = AsyncMock(  # type: ignore[attr-defined]
        return_value=_make_fake_aiohttp_response(
            body=[
                {"jsonrpc": "2.0", "id": 1, "result": "0x1"},
                {"jsonrpc": "2.0", "id": 2, "result": "0x14b0a4d"},
            ]
        )
    )
    results = await crypto.batch_rpc(
        network="ethereum-mainnet",
        requests=[
            {"method": "eth_chainId", "params": [], "id": 1},
            JsonRpcRequest(method="eth_blockNumber", params=[], id=2),
        ],
    )

    assert isinstance(results, BatchJsonRpcResponse)
    assert len(results) == 2
    assert results[0].id == 1
    assert results[0].result == "0x1"
    assert results[1].id == 2
    assert results[1].result == "0x14b0a4d"
    # Indexing/iteration mirror the underlying list.
    assert [item.id for item in results] == [1, 2]
    assert results.responses[0].result == "0x1"

    call_kwargs = crypto._client._request.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["method"] == "POST"
    assert call_kwargs["path"] == "crypto/rpc/ethereum-mainnet"
    assert call_kwargs["raw_response"] is True
    body = call_kwargs["json_data"]
    assert isinstance(body, list)
    assert body[0] == {"method": "eth_chainId", "jsonrpc": "2.0", "params": [], "id": 1}
    assert body[1] == {"method": "eth_blockNumber", "jsonrpc": "2.0", "params": [], "id": 2}


@pytest.mark.asyncio
async def test_batch_rpc_rejects_empty(crypto: Crypto) -> None:
    with pytest.raises(ValueError, match="at least one"):
        await crypto.batch_rpc(network="ethereum-mainnet", requests=[])


@pytest.mark.asyncio
async def test_batch_rpc_rejects_oversize(crypto: Crypto) -> None:
    too_many: list[dict[str, Any]] = [{"method": "eth_chainId", "id": i} for i in range(101)]
    with pytest.raises(ValueError, match="at most 100"):
        await crypto.batch_rpc(network="ethereum-mainnet", requests=too_many)


@pytest.mark.asyncio
async def test_batch_rpc_rejects_non_list_response(crypto: Crypto) -> None:
    crypto._client._request = AsyncMock(  # type: ignore[attr-defined]
        return_value=_make_fake_aiohttp_response(
            body={"jsonrpc": "2.0", "id": 1, "result": "0x1"},
        )
    )
    with pytest.raises(TypeError, match="batch response"):
        await crypto.batch_rpc(
            network="ethereum-mainnet",
            requests=[{"method": "eth_chainId", "id": 1}],
        )


@pytest.mark.asyncio
async def test_batch_rpc_attaches_idempotency_key(crypto: Crypto) -> None:
    crypto._client._request = AsyncMock(  # type: ignore[attr-defined]
        return_value=_make_fake_aiohttp_response(
            body=[{"jsonrpc": "2.0", "id": 1, "result": "0x1"}]
        )
    )
    await crypto.batch_rpc(
        network="ethereum-mainnet",
        requests=[{"method": "eth_chainId", "id": 1}],
        idempotency_key="batch-key-1",
    )

    headers = crypto._client._request.call_args.kwargs["headers"]  # type: ignore[attr-defined]
    assert headers == {"Idempotency-Key": "batch-key-1"}


# ---------------------------------------------------------------------------
# Response-header surfacing (rpc + batch_rpc)
# ---------------------------------------------------------------------------


def _attach_fake_headers(model: Any, headers: dict[str, str]) -> None:
    """Mimic the post-validation _response attachment performed by the client."""
    fake_response = MagicMock()
    fake_response.headers = headers
    model._response = fake_response


@pytest.mark.asyncio
async def test_rpc_surfaces_response_headers(crypto: Crypto) -> None:
    """``JsonRpcResponse`` exposes the four crypto-proxy billing headers."""
    response = JsonRpcResponse(jsonrpc="2.0", id=1, result="0x1", error=None)
    _attach_fake_headers(
        response,
        {
            "x-venice-rpc-credits": "25",
            "x-venice-rpc-cost-usd": "0.00001563",
            "x-request-id": "req_abc123",
            "idempotent-replayed": "true",
        },
    )
    crypto._client.post = AsyncMock(return_value=response)  # type: ignore[attr-defined]

    result = await crypto.rpc(network="ethereum-mainnet", method="eth_chainId", params=[], id=1)

    assert result.rpc_credits == 25
    assert result.rpc_cost_usd == pytest.approx(0.00001563)
    assert result.venice_request_id == "req_abc123"
    assert result.idempotent_replayed is True


@pytest.mark.asyncio
async def test_rpc_response_header_defaults_when_missing(crypto: Crypto) -> None:
    """All four accessors degrade to ``None``/``False`` when headers are absent."""
    response = JsonRpcResponse(jsonrpc="2.0", id=1, result="0x1", error=None)
    crypto._client.post = AsyncMock(return_value=response)  # type: ignore[attr-defined]

    result = await crypto.rpc(network="ethereum-mainnet", method="eth_chainId")

    assert result.rpc_credits is None
    assert result.rpc_cost_usd is None
    assert result.venice_request_id is None
    assert result.idempotent_replayed is False


@pytest.mark.asyncio
async def test_batch_rpc_surfaces_response_headers(crypto: Crypto) -> None:
    """``BatchJsonRpcResponse`` carries the (summed) billing headers."""
    crypto._client._request = AsyncMock(  # type: ignore[attr-defined]
        return_value=_make_fake_aiohttp_response(
            body=[
                {"jsonrpc": "2.0", "id": 1, "result": "0x1"},
                {"jsonrpc": "2.0", "id": 2, "result": "0x2"},
            ],
            headers={
                "x-venice-rpc-credits": "20",
                "x-venice-rpc-cost-usd": "0.00001250",
                "x-request-id": "req_batch_1",
                "idempotent-replayed": "false",
            },
        )
    )
    results = await crypto.batch_rpc(
        network="ethereum-mainnet",
        requests=[
            {"method": "eth_chainId", "id": 1},
            {"method": "eth_blockNumber", "id": 2},
        ],
    )

    assert results.rpc_credits == 20
    assert results.rpc_cost_usd == pytest.approx(0.00001250)
    assert results.venice_request_id == "req_batch_1"
    assert results.idempotent_replayed is False
