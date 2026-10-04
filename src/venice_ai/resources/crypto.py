"""
Venice AI Crypto RPC proxy resource.

Wraps the two ``/crypto/rpc/*`` endpoints documented at
``api-reference/endpoint/crypto/``:

- ``GET  /crypto/rpc/networks`` — public list of supported network slugs
- ``POST /crypto/rpc/{network}`` — JSON-RPC 2.0 forwarder (single + batch)

The proxy bills per credit (``baseCredits[chain] x methodTier``); credit and
rate-limit detail lives in the endpoint docs. This resource is a thin pass-through
— ``params`` and ``result`` are forwarded to / from the upstream chain unchanged.

Idempotency:
    Every ``rpc()`` / ``batch_rpc()`` call carries an ``Idempotency-Key``: the
    caller's ``idempotency_key``, or a fresh random key when none is given. The
    key stays the same across the client's automatic retries, so a retried call
    is deduplicated by the proxy instead of billed twice. Replaying within 24h
    with the same key + same body returns the cached response with the
    ``Idempotent-Replayed: true`` header. Same key + different body returns 400.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import aiohttp

from .._resource import APIResource
from ..types.api.crypto import (
    BatchJsonRpcResponse,
    CryptoNetworksResponse,
    JsonRpcRequest,
    JsonRpcResponse,
)
from ..utils.errors import read_body

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

__all__ = ["Crypto"]

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
_MAX_BATCH_SIZE = 100


def _idempotency_headers(idempotency_key: str | None) -> dict[str, str]:
    """Build the ``Idempotency-Key`` header, generating a key when none is given.

    The proxy enforces ``[A-Za-z0-9_-]{1,255}``; we mirror that client-side so
    a 400 surfaces as a ``ValueError`` at the call site instead of an HTTP error.
    """
    if idempotency_key is None:
        return {"Idempotency-Key": f"venice-py-{uuid.uuid4().hex}"}
    if not _IDEMPOTENCY_KEY_PATTERN.match(idempotency_key):
        raise ValueError(
            "idempotency_key must match [A-Za-z0-9_-]{1,255} (Venice proxy constraint)"
        )
    return {"Idempotency-Key": idempotency_key}


class Crypto(APIResource["VeniceClient"]):
    """Crypto RPC proxy: discovery + JSON-RPC 2.0 forwarder.

    Example::

        async with VeniceClient() as client:
            # Discover supported networks
            slugs = await client.crypto.networks()

            # Single call
            resp = await client.crypto.rpc(
                network="ethereum-mainnet",
                method="eth_chainId",
                params=[],
                id=1,
            )
            chain_id = resp.result  # "0x1"

            # Batch call (up to 100 items)
            results = await client.crypto.batch_rpc(
                network="ethereum-mainnet",
                requests=[
                    {"method": "eth_chainId", "params": [], "id": 1},
                    {"method": "eth_blockNumber", "params": [], "id": 2},
                ],
            )
    """

    async def networks(self) -> list[str]:
        """List supported crypto RPC network slugs.

        Public endpoint — does not require authentication. The list is
        authoritative: slugs not in it return ``400 Unsupported RPC network``
        from ``rpc()``.

        :return: Sorted Venice-side network slugs (e.g. ``"ethereum-mainnet"``).
        :rtype: list[str]
        """
        response = await self._client.get("crypto/rpc/networks", cast_to=CryptoNetworksResponse)
        return response.networks

    async def rpc(
        self,
        *,
        network: str,
        method: str,
        params: list[Any] | dict[str, Any] | None = None,
        id: int | str | None = 1,
        idempotency_key: str | None = None,
    ) -> JsonRpcResponse:
        """Forward a single JSON-RPC 2.0 call to a supported chain.

        :param network: Venice-side network slug (e.g. ``"ethereum-mainnet"``).
            See :meth:`networks` for the live list.
        :type network: str
        :param method: JSON-RPC method name (e.g. ``"eth_chainId"``).
        :type method: str
        :param params: Method parameters. Shape is method-dependent and forwarded
            to the upstream chain unchanged.
        :type params: list[Any] | dict[str, Any] | None
        :param id: Caller-supplied request ID echoed back in the response.
            Defaults to ``1``.
        :type id: int | str | None
        :param idempotency_key: Idempotency key (``[A-Za-z0-9_-]{1,255}``).
            Same key + same body within 24h replays the cached response. When
            omitted, a random key is generated for this call, so the client's
            automatic retries are deduplicated; pass your own to deduplicate
            across separate calls.
        :type idempotency_key: str | None

        :return: :class:`JsonRpcResponse`. On per-request failure, ``error`` is
            populated and HTTP status is still 200 — check ``response.error``.
        :rtype: JsonRpcResponse

        :raises ValueError: If ``idempotency_key`` violates the proxy pattern.
        :raises venice_ai.exceptions.APIError: For HTTP-level failures (e.g.
            400 unsupported network, 429 rate-limited).
        """
        body = JsonRpcRequest(method=method, params=params, id=id).model_dump(exclude_none=True)
        headers = _idempotency_headers(idempotency_key)
        return await self._client.post(
            f"crypto/rpc/{network}",
            json_data=body,
            headers=headers,
            cast_to=JsonRpcResponse,
        )

    async def batch_rpc(
        self,
        *,
        network: str,
        requests: Sequence[JsonRpcRequest | dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> BatchJsonRpcResponse:
        """Forward a JSON-RPC 2.0 batch (1–100 items) to a supported chain.

        Each item is validated independently; if any method is unsupported the
        entire batch is rejected with 400 and every offending name is listed in
        the error message.

        Per-item RPC errors do NOT fail the batch — successful items still return
        ``result``, failed items return ``error``. Per the docs, RPC-level errors
        in batch responses bill at 5 credits each rather than the full method tier.

        :param network: Venice-side network slug.
        :type network: str
        :param requests: Up to 100 JSON-RPC requests. Each may be a
            :class:`JsonRpcRequest` instance or a plain dict — dicts are
            validated through :class:`JsonRpcRequest` before being sent.
        :type requests: Sequence[JsonRpcRequest | dict[str, Any]]
        :param idempotency_key: Optional idempotency key, same semantics as
            :meth:`rpc`.
        :type idempotency_key: str | None

        :return: :class:`BatchJsonRpcResponse` whose ``responses`` list mirrors
            the input order at the wire level. JSON-RPC does not guarantee
            response ordering — use each item's ``id`` field to correlate.
            HTTP-level billing headers (``rpc_credits``, ``rpc_cost_usd``,
            ``venice_request_id``, ``idempotent_replayed``) are exposed on the
            wrapper since they cover the whole batch.
        :rtype: BatchJsonRpcResponse

        :raises ValueError: If ``requests`` is empty, exceeds 100 items, or
            ``idempotency_key`` violates the proxy pattern.
        """
        if not requests:
            raise ValueError("batch_rpc requires at least one request")
        if len(requests) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_rpc accepts at most {_MAX_BATCH_SIZE} requests (got {len(requests)})"
            )

        body: list[dict[str, Any]] = []
        for item in requests:
            if isinstance(item, JsonRpcRequest):
                body.append(item.model_dump(exclude_none=True))
            else:
                body.append(JsonRpcRequest.model_validate(item).model_dump(exclude_none=True))

        headers = _idempotency_headers(idempotency_key)
        # Batch responses are JSON arrays — cast_to expects a single model, so
        # we ask the client for the raw aiohttp response and assemble the
        # wrapper ourselves so HTTP-level billing headers stay attached.
        response = await self._client._request(
            method="POST",
            path=f"crypto/rpc/{network}",
            json_data=body,  # type: ignore[arg-type]  # batch payload is JSON-RPC array, not dict
            headers=headers,
            raw_response=True,
        )
        if not isinstance(response, aiohttp.ClientResponse):
            raise TypeError(
                f"Expected aiohttp.ClientResponse for batch RPC, got {type(response).__name__}"
            )
        try:
            await read_body(response)
            raw = await response.json()
        finally:
            response.release()
        if not isinstance(raw, list):
            raise TypeError(f"Expected JSON-RPC batch response (list), got {type(raw).__name__}")

        wrapper = BatchJsonRpcResponse(
            responses=[JsonRpcResponse.model_validate(item) for item in raw]
        )
        wrapper._response = response  # billing-header surfacing
        return wrapper
