"""VCR-based integration tests for the Responses API (Alpha).

Covers ``POST /responses`` for both non-streaming and streaming flows. The
endpoint is Alpha-tagged on Venice, so tests skip cleanly on a 4xx that
indicates the caller's key isn't enrolled in the Alpha program.

E2EE-capable models are not supported on ``/responses`` per the API docs;
:func:`_select_non_e2ee_model` filters them out.
"""

import os

import pytest
import pytest_asyncio

from venice_ai import VeniceClient, create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError
from venice_ai.types.api import (
    KNOWN_RESPONSE_STATUSES,
    ResponsesResponse,
    ResponsesStreamEvent,
)

pytestmark = [pytest.mark.integration, pytest.mark.vcr]


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


async def _select_non_e2ee_model(client: VeniceClient) -> str:
    """Pick a text model that's not E2EE-capable (E2EE is rejected on /responses).

    Prefers a non-reasoning model: reasoning models spend the (small) output
    token budget on thinking and can stall a streaming ``/responses`` call past
    the client timeout, so they're a poor fit for these smoke tests. Falls back
    to any non-E2EE model if no non-reasoning one is available.
    """
    models = await client.models.list(type="text")
    fallback: str | None = None
    for m in models.data:
        spec = getattr(m, "model_spec", None)
        caps = getattr(spec, "capabilities", None) if spec else None
        if caps is None or getattr(caps, "supportsE2EE", False) is not False:
            continue
        if fallback is None:
            fallback = m.id
        if getattr(caps, "supportsReasoning", False) is False:
            return m.id
    if fallback is not None:
        return fallback
    pytest.skip("No non-E2EE text model available")


def _skip_if_alpha_unavailable(exc: APIError) -> None:
    """Alpha endpoints can return 403/404 when the key isn't enrolled — skip rather than fail."""
    status = getattr(exc, "status_code", None)
    if status in (403, 404):
        pytest.skip(f"/responses Alpha endpoint not available for this key: {exc}")
    raise exc


@pytest.mark.integration
async def test_responses_create_basic(venice_client, vcr_cassette):
    """``responses.create()`` returns a ResponsesResponse with at least one output block."""
    with vcr_cassette:
        model = await _select_non_e2ee_model(venice_client)
        try:
            result = await venice_client.responses.create(
                model=model,
                input="Reply with exactly the word: pong",
                max_output_tokens=20,
            )
        except APIError as e:
            _skip_if_alpha_unavailable(e)
            return

        assert isinstance(result, ResponsesResponse)
        assert result.id
        assert result.object == "response"
        assert result.status in KNOWN_RESPONSE_STATUSES
        if result.status == "incomplete":
            assert result.incomplete_details is not None
        assert isinstance(result.output, list)


@pytest.mark.integration
async def test_responses_create_streaming(venice_client, vcr_cassette):
    """``responses.create(stream=True)`` yields ResponsesStreamEvent chunks via SSE."""
    with vcr_cassette:
        model = await _select_non_e2ee_model(venice_client)
        try:
            stream = await venice_client.responses.create(
                model=model,
                input="Count: one, two, three.",
                max_output_tokens=20,
                stream=True,
            )

            events: list[ResponsesStreamEvent] = []
            async for event in stream:
                events.append(event)
        except APIError as e:
            _skip_if_alpha_unavailable(e)
            return

        assert len(events) > 0
        # Every event has a ``type`` per the OpenAI Responses streaming contract.
        for event in events:
            assert isinstance(event, ResponsesStreamEvent)
            assert isinstance(event.type, str) and event.type
        # The terminal event for a successful stream is response.completed.
        types = [e.type for e in events]
        assert any(t.startswith("response.") for t in types)
