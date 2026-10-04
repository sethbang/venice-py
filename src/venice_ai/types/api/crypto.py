"""
Crypto RPC proxy models for Venice AI API.

Covers ``GET /crypto/rpc/networks`` (network discovery) and
``POST /crypto/rpc/{network}`` (JSON-RPC 2.0 forwarder, single + batch).

The proxy speaks JSON-RPC 2.0 verbatim — request and response shapes match
the upstream chain. Pricing/rate-limit detail is documented at
``api-reference/endpoint/crypto/rpc``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.models.common import VeniceBaseModel


class JsonRpcRequest(BaseModel):
    """A single JSON-RPC 2.0 request envelope.

    Mirrors the body shape accepted by ``POST /crypto/rpc/{network}``. ``params``
    accepts the upstream chain's parameter list (or object); the proxy does not
    reshape it.
    """

    model_config = ConfigDict(populate_by_name=True)

    method: str = Field(
        ...,
        description=(
            "JSON-RPC method name. See the proxy's supported-methods table for "
            "1x/2x/4x credit tiers."
        ),
    )
    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC protocol version.")
    params: list[Any] | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Method parameters. Shape is method-dependent — passed through to "
            "the upstream chain unchanged."
        ),
    )
    id: int | str | None = Field(
        default=None,
        description="Caller-supplied request ID echoed back. Required to correlate batch items.",
    )


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object.

    Present on per-request failure. The HTTP status is still 200 in that case —
    callers must inspect ``error`` on each response item.
    """

    model_config = ConfigDict(extra="allow")

    code: int = Field(..., description="JSON-RPC error code (e.g. -32602 for invalid params).")
    message: str = Field(..., description="Human-readable error message.")
    data: Any | None = Field(
        default=None, description="Optional error payload from the upstream chain."
    )


def _rpc_credits(model: VeniceBaseModel) -> int | None:
    """Parse ``X-Venice-RPC-Credits`` from a VeniceBaseModel's response headers."""
    from venice_ai.utils.parsing import safe_int

    headers = model.headers
    return safe_int(headers.get("x-venice-rpc-credits")) if headers else None


def _rpc_cost_usd(model: VeniceBaseModel) -> float | None:
    """Parse ``X-Venice-RPC-Cost-USD`` from a VeniceBaseModel's response headers."""
    from venice_ai.utils.parsing import safe_float

    headers = model.headers
    return safe_float(headers.get("x-venice-rpc-cost-usd")) if headers else None


def _venice_request_id(model: VeniceBaseModel) -> str | None:
    """Read ``X-Request-ID`` from a VeniceBaseModel's response headers."""
    headers = model.headers
    return headers.get("x-request-id") if headers else None


def _idempotent_replayed(model: VeniceBaseModel) -> bool:
    """``True`` iff ``Idempotent-Replayed: true`` is present in response headers."""
    headers = model.headers
    if not headers:
        return False
    return headers.get("idempotent-replayed", "").lower() == "true"


class JsonRpcResponse(VeniceBaseModel):
    """A single JSON-RPC 2.0 response envelope.

    Exactly one of ``result`` or ``error`` is populated; ``result`` may legitimately
    be ``None`` for some methods, so check ``error`` first to disambiguate.

    Inherits header-extraction helpers from :class:`VeniceBaseModel`. The four
    crypto-proxy billing headers (documented at
    ``api-reference/endpoint/crypto/rpc.md``) are surfaced via the typed
    accessors :attr:`rpc_credits`, :attr:`rpc_cost_usd`,
    :attr:`venice_request_id`, and :attr:`idempotent_replayed`.
    """

    model_config = ConfigDict(extra="allow")

    jsonrpc: str = Field(..., description="JSON-RPC protocol version.")
    id: int | str | None = Field(
        default=None, description="Echoed request ID. ``None`` only for malformed requests."
    )
    result: Any | None = Field(default=None, description="Method result on success.")
    error: JsonRpcError | None = Field(
        default=None, description="Error object on per-request failure."
    )

    @property
    def rpc_credits(self) -> int | None:
        """Credits charged (``X-Venice-RPC-Credits``)."""
        return _rpc_credits(self)

    @property
    def rpc_cost_usd(self) -> float | None:
        """Dollar cost to 8 decimal places (``X-Venice-RPC-Cost-USD``)."""
        return _rpc_cost_usd(self)

    @property
    def venice_request_id(self) -> str | None:
        """Correlation ID (``X-Request-ID``).

        Distinct from :attr:`VeniceBaseModel.request_id`, which reads the
        Cloudflare ``cf-ray`` header.
        """
        return _venice_request_id(self)

    @property
    def idempotent_replayed(self) -> bool:
        """``True`` if the response came from the idempotency cache."""
        return _idempotent_replayed(self)


class BatchJsonRpcResponse(VeniceBaseModel):
    """Wrapper for a JSON-RPC 2.0 batch response.

    The Venice proxy returns one HTTP response per batch, so the four billing
    headers (``X-Venice-RPC-Credits``, ``X-Venice-RPC-Cost-USD``,
    ``X-Request-ID``, ``Idempotent-Replayed``) cover the whole batch and are
    surfaced here rather than on individual items. Iterate or index this
    wrapper to access per-item ``result``/``error`` payloads.
    """

    model_config = ConfigDict(extra="allow")

    responses: list[JsonRpcResponse] = Field(
        ..., description="One JsonRpcResponse per item in the original batch request."
    )

    @property
    def rpc_credits(self) -> int | None:
        """Sum of credits charged across the batch (``X-Venice-RPC-Credits``)."""
        return _rpc_credits(self)

    @property
    def rpc_cost_usd(self) -> float | None:
        """Sum of dollar cost across the batch (``X-Venice-RPC-Cost-USD``)."""
        return _rpc_cost_usd(self)

    @property
    def venice_request_id(self) -> str | None:
        """Correlation ID (``X-Request-ID``)."""
        return _venice_request_id(self)

    @property
    def idempotent_replayed(self) -> bool:
        """``True`` if the batch was served from the idempotency cache."""
        return _idempotent_replayed(self)

    def __iter__(self) -> Any:
        return iter(self.responses)

    def __len__(self) -> int:
        return len(self.responses)

    def __getitem__(self, index: int) -> JsonRpcResponse:
        return self.responses[index]


class CryptoNetworksResponse(BaseModel):
    """Response body for ``GET /crypto/rpc/networks``.

    Public discovery endpoint — no auth required. The list is authoritative;
    slugs not in this list return ``400 Unsupported RPC network`` from the
    proxy.
    """

    model_config = ConfigDict(extra="allow")

    networks: list[str] = Field(..., description="Sorted Venice-side network slugs.")


__all__ = [
    "BatchJsonRpcResponse",
    "CryptoNetworksResponse",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
]
