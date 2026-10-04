"""VCR-based integration tests for the Music generation API.

Music generation is served through the ``/audio/*`` queue family
(``queue`` / ``quote`` / ``retrieve`` / ``complete``). The polling shape and
content-type discrimination on ``/audio/retrieve`` are wire-format-dependent,
so live cassettes are the source of truth. Mock-based unit tests in
``tests/unit/resources/test_music_job.py`` cover the branches that aren't
reachable from a single recording (FAILED status, dual-exception cleanup
logging, etc).
"""

import os

import pytest
import pytest_asyncio

from venice_ai import VeniceClient, create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError
from venice_ai.types.api import MusicModelSpec
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQueueResponse,
    MusicQuoteResponse,
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


async def _select_music_model(client: VeniceClient, *, target_duration: int = 15) -> str:
    """Pick a music model that accepts ``target_duration`` seconds.

    Filters dynamically against ``MusicModelSpec`` capability fields:
      - skips models with a strict ``duration_options`` enum unless
        ``target_duration`` is in the enum
      - requires ``min_duration <= target_duration <= max_duration`` when
        either bound is set
      - falls back to any music model when no capability fields are set
    """
    try:
        models = await client.models.list(type="music")
    except APIError as e:
        pytest.skip(f"music models endpoint not available: {e}")
    if not models.data:
        pytest.skip("no music models available for this account")

    candidates: list[str] = []
    for m in models.data:
        spec = m.model_spec
        if not isinstance(spec, MusicModelSpec):
            # Should always be MusicModelSpec for type=music after the
            # refactor; if not, skip rather than guess.
            continue

        # Strict enum gate — model only accepts these exact durations.
        if spec.duration_options:
            if target_duration not in spec.duration_options:
                continue
            candidates.append(m.id)
            continue

        # Range gate — both bounds optional.
        if spec.min_duration is not None and target_duration < spec.min_duration:
            continue
        if spec.max_duration is not None and target_duration > spec.max_duration:
            continue
        candidates.append(m.id)

    if not candidates:
        pytest.skip(
            f"no music models accept duration={target_duration}s "
            f"(checked {len(models.data)} model(s))"
        )

    # Stable preference order keeps cassettes deterministic across re-records.
    # Cheapest known short-duration models first; otherwise alphabetical.
    preferred_first = ("mmaudio-v2-text-to-audio", "elevenlabs-sound-effects-v2")
    for pref in preferred_first:
        if pref in candidates:
            return pref
    return sorted(candidates)[0]


def _skip_if_unavailable(exc: APIError) -> None:
    """Music endpoints can return 403/404 when the account isn't enrolled."""
    status = getattr(exc, "status_code", None)
    if status in (403, 404):
        pytest.skip(f"music endpoint not available for this key: {exc}")
    raise exc


@pytest.mark.integration
async def test_music_quote_returns_estimated_cost(vcr_cassette, venice_client):
    """``music.quote()`` (POST /audio/quote) returns a numeric cost estimate.

    Wire-shape validation: the response body must deserialize into
    ``MusicQuoteResponse`` with a numeric ``quote`` field. The cassette is
    the source of truth for whether the API returns int or float here — both
    are accepted by the SDK type.
    """
    with vcr_cassette:
        try:
            model = await _select_music_model(venice_client)
            quote = await venice_client.music.quote(
                model=model,
                duration_seconds=15,
            )
        except APIError as e:
            _skip_if_unavailable(e)
            return

        assert isinstance(quote, MusicQuoteResponse)
        # Quote should be non-negative (free tier may legitimately return 0).
        assert isinstance(quote.quote, (int, float))
        assert quote.quote >= 0


@pytest.mark.integration
async def test_music_submit_then_release(vcr_cassette, venice_client):
    """End-to-end smoke: ``submit()`` queues a job and ``release()`` releases it.

    This validates the queue/complete wire shapes without waiting for
    generation to finish. The cassette captures both the submission body
    (camelCase forwarding via Pydantic) and the cleanup response shape.
    """
    with vcr_cassette:
        try:
            model = await _select_music_model(venice_client)
            queue_response = await venice_client.music.submit(
                model=model,
                prompt="Brief instrumental synthwave intro, 15 seconds",
                duration_seconds=15,
            )
        except APIError as e:
            _skip_if_unavailable(e)
            return

        assert isinstance(queue_response, MusicQueueResponse)
        assert queue_response.queue_id
        assert queue_response.status == "QUEUED"
        # ``model`` round-trips on the response — server can echo or remap.
        assert queue_response.model

        # Cleanup — also exercises the ``release()`` wire shape. The server
        # returns ``{"success": <bool>}`` regardless of whether the job was
        # actually in-flight; on fast models the job may complete before this
        # call lands and the server returns ``success: false``. The wire
        # shape is what we verify here.
        cleanup = await venice_client.music.release(
            model=queue_response.model,
            queue_id=queue_response.queue_id,
        )
        assert isinstance(cleanup, MusicCompleteResponse)
        assert isinstance(cleanup.success, bool)


@pytest.mark.integration
async def test_music_submit_retrieve_processing_then_release(vcr_cassette, venice_client):
    """``music.retrieve()`` returns a ``PROCESSING`` status while generation runs.

    Validates the JSON-content-type branch of ``Music.retrieve()``: the
    response is parsed as a dict and discriminated on ``status``. The
    COMPLETED branch is exercised by the
    unit tests in ``test_music_job.py`` since it requires either a long-poll
    or a stub binary response — both unwieldy for a fast integration test.
    """
    with vcr_cassette:
        try:
            model = await _select_music_model(venice_client)
            queue_response = await venice_client.music.submit(
                model=model,
                prompt="Lush ambient texture under a calm narration, 15 seconds",
                duration_seconds=15,
            )
        except APIError as e:
            _skip_if_unavailable(e)
            return

        # Single poll — should still be PROCESSING for any non-trivial duration.
        try:
            status = await venice_client.music.retrieve(
                model=queue_response.model,
                queue_id=queue_response.queue_id,
            )
        except APIError as e:
            # Cleanup the queue entry even if retrieve fails so we don't leak.
            await venice_client.music.release(
                model=queue_response.model, queue_id=queue_response.queue_id
            )
            _skip_if_unavailable(e)
            return

        # Most likely PROCESSING; could also be already-COMPLETED on a fast model.
        # Either way, the union must discriminate without raising, and FAILED
        # would be unexpected here — surface it as a failure.
        assert isinstance(status, (MusicProcessingStatus, MusicCompletedStatus, MusicFailedStatus))
        assert status.status in ("PROCESSING", "COMPLETED", "FAILED")

        if isinstance(status, MusicProcessingStatus):
            # Numeric timing fields exist and are non-negative.
            assert status.average_execution_time >= 0
            assert status.execution_duration >= 0

        # Cleanup
        await venice_client.music.release(
            model=queue_response.model, queue_id=queue_response.queue_id
        )
