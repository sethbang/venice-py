"""TDD: Tee.get_signature + TeeSignatureResponse (audit MED #12).

Response shape captured live from GET /tee/signature (model e2ee-gpt-oss-120b-p,
Phala TEE provider) — see out/audit-2026-06-19/probes/tee_signature.json.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.tee import Tee
from venice_ai.tee.types import TeeReceiptSignature, TeeSignatureResponse

LIVE_SHAPE = {
    "api_version": "aci/1",
    "text": "73559fb84516e36bc5371a5418cf35c98d8b9071",
    "signature": "0x2d8fafad90245391e4d035d9c6035a759f9eb9aa",
    "signing_address": "0xffb22d95ed9b929e876cb4199e6cbb88e2a3d3aa",
    "signing_algo": "ecdsa",
    "receipt": {
        "api_version": "aci/1",
        "receipt_id": "rcpt-278a1efedf88d3b304804a18",
        "chat_id": "chatcmpl-e62c4a55-85cb-497a-ba0a-1891b21",
        "workload_id": "sha256:3def476b72026924f9d88f7b339b2e211",
        "workload_keyset_digest": "sha256:dbcfcdece78f009bc1f4b46a4bf1197b9",
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "served_at": 1781948632,
        "event_log": [{"seq": 0, "type": "request.received", "body_hash": "sha256:73559fb84516"}],
        "signature": {
            "algo": "ecdsa-secp256k1",
            "key_id": "dstack-kms-receipt-v1",
            "value": "5fd579bb382233b81ae35af74c5a88a217732a39",
        },
    },
    "model": "e2ee-gpt-oss-120b-p",
    "requested_request_id": "chatcmpl-e62c4a55-85cb-497a-ba0a-1891b21",
    "tee_provider": "phala",
    "upstream_model": "openai/gpt-oss-120b",
    "verification": {
        "attestation_endpoint": "/api/v1/tee/attestation",
        "description": "Verify the chain of trust",
    },
}


def test_parses_live_signature_shape():
    r = TeeSignatureResponse.model_validate(LIVE_SHAPE)
    assert r.signature.startswith("0x")
    assert r.signing_algo == "ecdsa"
    assert r.tee_provider == "phala"
    assert r.receipt is not None
    assert isinstance(r.receipt.signature, TeeReceiptSignature)
    assert r.receipt.signature.algo == "ecdsa-secp256k1"
    assert r.receipt.event_log[0].type == "request.received"


def test_tolerates_unknown_fields():
    r = TeeSignatureResponse.model_validate({**LIVE_SHAPE, "future_field": 1})
    assert (r.model_extra or {}).get("future_field") == 1


@pytest.fixture
def tee_resource() -> Tee:
    client = Mock()
    client.get = AsyncMock(return_value=TeeSignatureResponse.model_validate(LIVE_SHAPE))
    return Tee(client)


@pytest.mark.asyncio
async def test_get_signature_calls_endpoint(tee_resource: Tee):
    await tee_resource.get_signature(
        model="e2ee-gpt-oss-120b-p", request_id="chatcmpl-e62c4a55-85cb-497a-ba0a-1891b21"
    )
    call = cast(Any, tee_resource._client.get).call_args
    assert call.args[0] == "tee/signature"
    assert call.kwargs["params"] == {
        "model": "e2ee-gpt-oss-120b-p",
        "request_id": "chatcmpl-e62c4a55-85cb-497a-ba0a-1891b21",
    }
    assert call.kwargs["cast_to"] is TeeSignatureResponse


def test_parses_a_receipt_without_seq_and_with_a_bare_signature():
    """Another shape seen live: events carry no ``seq`` and the signature is a hex string."""
    receipt = {
        **LIVE_SHAPE["receipt"],
        "event_log": [
            {"type": "request.received", "body_hash": "sha256:dab0"},
            {"type": "route.selected", "target_route_id": "phala-1"},
        ],
        "signature": "3e58e7442b1c023b09e3ce5e",
    }
    r = TeeSignatureResponse.model_validate({**LIVE_SHAPE, "receipt": receipt})
    assert r.receipt is not None
    assert r.receipt.signature == "3e58e7442b1c023b09e3ce5e"
    assert [e.seq for e in r.receipt.event_log] == [None, None]
    assert (r.receipt.event_log[1].model_extra or {}).get("target_route_id") == "phala-1"
