"""Live regression: the verifier must handle the wire Venice serves today.

The schema-drift defect this guards against was invisible to the offline suite,
because the committed captures were taken before the wire changed. This test
fetches a real attestation and asserts the normalizer recognizes it and the full
verifier accepts it. Skipped without credentials.
"""

from __future__ import annotations

import os

import pytest

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError
from venice_ai.tee._evidence import detect_schema, normalize
from venice_ai.tee.types import TeeAttestation

pytest.importorskip("dcap_qvl")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("VENICE_API_KEY"),
        reason="live TEE schema check requires VENICE_API_KEY",
    ),
]


async def _live_attestation(client: VeniceClient) -> TeeAttestation:
    """Fetch an attestation from the first e2ee-* node that serves one.

    An individual TEE node can be unreachable (502) while attestation as a
    whole is healthy, and which node sits first in the model list is not this
    test's concern — the wire schema is. So a 5xx moves on to the next
    candidate; every other failure, a malformed attestation included,
    propagates.
    """
    models = await client.models.list(type="text")
    candidates = [m.id for m in models.data if m.id.startswith("e2ee-")]
    if not candidates:
        pytest.skip("no e2ee-* model entitled on this account")

    unavailable: list[str] = []
    for model in candidates:
        try:
            return await client.tee.get_attestation(model=model)
        except APIError as exc:
            if exc.status_code is None or exc.status_code < 500:
                raise
            unavailable.append(f"{model} ({exc.status_code})")
    pytest.skip(f"no e2ee-* node served an attestation: {', '.join(unavailable)}")


async def test_live_attestation_schema_is_recognized() -> None:
    async with VeniceClient() as client:
        attestation = await _live_attestation(client)

        # Fails loudly if Venice changes the wire again.
        schema = detect_schema(attestation)
        evidence = normalize(attestation)
        assert evidence.raw_quote, "live attestation carried no usable quote"
        assert evidence.event_log, f"live attestation ({schema}) carried an empty event log"
        assert all("imr" in e and "digest" in e for e in evidence.event_log)


async def test_live_attestation_fully_verifies() -> None:
    from venice_ai.tee import DcapTdxVerifier

    async with VeniceClient() as client:
        attestation = await _live_attestation(client)

        verifier = await DcapTdxVerifier.with_fetched_collateral(attestation.intel_quote)
        assert verifier.verify(attestation) is True
        result = verifier.last_result
        assert result is not None
        assert result["checks"]["signature_chain"] is True
        assert result["checks"]["reportdata_binding"] is True
        assert result["checks"]["rtmr_replay"] is True
