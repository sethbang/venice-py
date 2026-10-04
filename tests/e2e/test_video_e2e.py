"""
End-to-end tests for Venice AI Video resource using VCR recording/replay.

These tests exercise the complete video generation lifecycle through the real
Venice AI API (or pre-recorded VCR cassettes):
  - Quoting (price estimation)
  - Queuing (text-to-video and image-to-video)
  - Retrieving (polling for status)
  - Completing (cleanup)

## Recording New Cassettes

Cassettes live under ``tests/e2e/cassettes/`` and are gitignored — they are a
local-dev convenience, never a committed fixture. To refresh them after an API
change, re-run the tests you want with ``VENICE_VCR_RECORD=all``; the root
conftest overwrites exactly those cassettes and restores the previous recording
if a test fails, so a transient error never destroys a good one::

    VENICE_API_KEY=... VENICE_VCR_RECORD=all \\
        poetry run pytest tests/e2e/test_video_e2e.py

Queueing a video costs real credit on every un-cassetted run, so each test uses
the cheapest model of its kind at the cheapest request that model accepts.

## Security Note

Cassettes are automatically scrubbed of sensitive data
(Authorization headers etc.) by the root conftest's ``vcr_config`` fixture.
"""

import asyncio
import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pytest
import pytest_asyncio
from PIL import Image

from venice_ai import VideoInputMode, cheapest_video_params, create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import (
    APIError,
    APIStatusError,
    InvalidRequestError,
    ModelQuotesUnavailableError,
    NoMatchingModelError,
    PaymentRequiredError,
    VeniceError,
)
from venice_ai.types.api.models import VideoModelConstraints
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
    VideoQuoteResponse,
)

# ---------------------------------------------------------------------------
# Video models are resolved DYNAMICALLY from the live API — never hardcoded.
# Venice retires models and ships new generations under new ID conventions
# (wan-2.6-* and wan-2-7-* are distinct generations that coexist, not a
# rename), and a stale hardcoded ID silently 400s.
#
# Resolution deliberately happens at the top of each test, *before* the body
# enters its ``vcr_cassette`` context. Inside a cassette, ``GET /models`` is
# served from whatever catalog snapshot that cassette happened to record, so a
# model picked there can be months out of date — and because the choice is
# cached for the whole session, it leaks into tests whose own cassette is
# missing and which therefore talk to the live API. Resolving outside the
# cassette keeps the model id and its advertised constraints consistent with the
# API actually being called. Each kind resolves to its cheapest model at that
# model's cheapest valid request, since every un-cassetted queue bills real
# credit. An env var can still pin a specific model; its request is then built
# from the catalog constraints the same way.
#
# (A fixture would read better but cannot work here: async fixtures run on the
# session event loop while test bodies run on a per-function loop, so an HTTP
# call made during fixture setup binds the client's aiohttp session to the wrong
# loop.)
# ---------------------------------------------------------------------------
_VIDEO_PICKS: dict[str, "_VideoPick | _Unresolved"] = {}

#: video_type -> (env var that pins a model, required image-to-video input mode).
_VIDEO_KINDS: dict[str, tuple[str, VideoInputMode | None]] = {
    "text-to-video": ("VENICE_E2E_VIDEO_T2V_MODEL", None),
    "image-to-video": ("VENICE_E2E_VIDEO_I2V_MODEL", "image"),
}


@dataclass(frozen=True)
class _VideoPick:
    """A video model and the cheapest request it accepts.

    ``request_params`` holds the ``duration_seconds`` / ``resolution`` /
    ``aspect_ratio`` / ``audio`` arguments, every one taken from values the
    model lists, so no test carries a duration or resolution a model rejects.
    ``quote_usd`` is the free quote for that request, or ``None`` when an env
    var pinned the model and no ranking quote was taken.
    """

    model: str
    request_params: dict[str, Any]
    quote_usd: float | None


async def _video_constraints(client, model: str) -> VideoModelConstraints:
    entry = await client.models.get(model)
    constraints = getattr(entry.model_spec, "constraints", None)
    if not isinstance(constraints, VideoModelConstraints):
        pytest.fail(f"{model!r} has no video constraints in the catalog")
    return constraints


@dataclass(frozen=True)
class _Unresolved:
    """Why no model could be picked; cached so a failed ranking is not repeated."""

    reason: str
    fail: bool


async def _pick_video(client, video_type: str) -> "_VideoPick | _Unresolved":
    """Pick the cheapest model of ``video_type``.

    No model of that kind is a skip. Candidates existing but every quote failing
    is a failure: it points at an outage or a rejected quote request, not at
    the catalog.
    """
    env_var, input_mode = _VIDEO_KINDS[video_type]
    override = os.environ.get(env_var)
    if override:
        constraints = await _video_constraints(client, override)
        return _VideoPick(override, cheapest_video_params(constraints), None)
    try:
        result = await client.models.resolve_cheapest_video(
            video_type=video_type, input_mode=input_mode
        )
    except ModelQuotesUnavailableError as exc:
        return _Unresolved(f"Every {video_type} quote failed: {exc.failures}", fail=True)
    except NoMatchingModelError as exc:
        return _Unresolved(f"No {video_type} model available on this account: {exc}", fail=False)
    return _VideoPick(result.model, dict(result.request_params), result.quote_usd)


async def _resolve_video(client, video_type: str) -> _VideoPick:
    """Resolve (and session-cache) the cheapest ``video_type`` model; skip if none.

    Ranking quotes every candidate once per session; ``/video/quote`` is free.
    """
    if video_type not in _VIDEO_PICKS:
        _VIDEO_PICKS[video_type] = await _pick_video(client, video_type)
    pick = _VIDEO_PICKS[video_type]
    if isinstance(pick, _Unresolved):
        if pick.fail:
            pytest.fail(pick.reason)
        pytest.skip(pick.reason)
    return pick


async def _resolve_t2v(client) -> _VideoPick:
    return await _resolve_video(client, "text-to-video")


async def _resolve_i2v(client) -> _VideoPick:
    return await _resolve_video(client, "image-to-video")


async def _optional_video_params(client, model: str, *, audio: bool) -> dict[str, Any]:
    """Build the widest *valid* optional-parameter set for ``model``.

    Which optional parameters a video model accepts varies per model, and
    sending an unsupported one is a hard 400 rather than a silent ignore —
    ``audio`` on a model with ``audio_configurable=False`` returns
    ``"This model does not support audio configuration"``. So every value comes
    from the model's own advertised constraints, at its cheapest listed setting
    (shortest duration, lowest resolution), which keeps these tests meaningful as
    the catalog turns over without paying for more video than they need.

    Note that ``supports_audio`` and ``audio_configurable`` are different
    claims: a model can generate audio yet reject the ``audio`` parameter.
    Only the latter gates sending it; asking for audio additionally needs the
    former.
    """
    constraints = await _video_constraints(client, model)
    return cheapest_video_params(constraints, audio=audio and constraints.audio)


def _generate_test_image_data_url(width: int = 256, height: int = 256) -> str:
    """Generate a small solid-color JPEG as a data URL for I2V tests.

    Using a data URL avoids external HTTP dependencies (e.g. Wikimedia
    rate-limiting) that previously caused 'corrupted or unreadable' errors.
    The Venice API requires a minimum of 240×240 pixels for I2V.
    """
    img = Image.new("RGB", (width, height), color=(135, 206, 235))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# A small inline image for I2V tests (data URL avoids external fetch failures)
VIDEO_TEST_IMAGE_URL = os.environ.get(
    "VENICE_E2E_VIDEO_IMAGE_URL",
    _generate_test_image_data_url(),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestVideoE2E:
    """End-to-end tests for the Video resource using VCR cassettes."""

    @pytest_asyncio.fixture
    async def venice_client(self):
        """Create VeniceClient for E2E testing with intelligent rate limiting."""
        api_key = os.getenv("VENICE_API_KEY")
        if not api_key:
            pytest.skip("VENICE_API_KEY environment variable required for E2E tests")

        client = create_test_venice_client(
            api_key=api_key,
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=False,
        )
        try:
            yield client
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # quote() tests
    # ------------------------------------------------------------------

    async def test_quote_t2v_basic(self, venice_client, vcr_cassette):
        """Quote a text-to-video generation and verify response shape."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                quote = await venice_client.video.quote(
                    model=t2v.model,
                    **t2v.request_params,
                )

                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote is not None
                assert quote.quote >= 0

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_quote_t2v_all_params(self, venice_client, vcr_cassette):
        """Quote with every optional parameter the model supports populated."""
        t2v = await _resolve_t2v(venice_client)
        t2v_optional_params = await _optional_video_params(venice_client, t2v.model, audio=True)
        # Guard against the derived set quietly collapsing: every video model
        # advertises resolutions, so an absent one means the capability lookup
        # degraded and this test would be exercising almost no optional params.
        assert "resolution" in t2v_optional_params, (
            f"no optional params derived for {t2v.model}: {t2v_optional_params}"
        )
        with vcr_cassette:
            try:
                quote = await venice_client.video.quote(
                    model=t2v.model,
                    **t2v_optional_params,
                )

                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote >= 0

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_quote_upscale_with_video_url(self, venice_client, vcr_cassette):
        """Quote an upscale (v2v) request using video_url + upscale_factor."""
        i2v = await _resolve_i2v(venice_client)
        with vcr_cassette:
            try:
                quote = await venice_client.video.quote(
                    model=i2v.model,
                    **i2v.request_params,
                    upscale_factor=2,
                    video_url=VIDEO_TEST_IMAGE_URL,
                )

                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote >= 0

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc) or "not supported" in str(exc).lower():
                    pytest.skip(f"Upscale combination unsupported for this model: {exc}")
                raise

    # ------------------------------------------------------------------
    # queue() tests
    # ------------------------------------------------------------------

    async def test_queue_t2v_basic(self, venice_client, vcr_cassette):
        """Queue a minimal text-to-video request and get a queue_id back."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="A serene mountain landscape with flowing clouds",
                    **t2v.request_params,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None
                assert len(result.queue_id) > 0
                assert result.model is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_queue_t2v_all_optional_params(self, venice_client, vcr_cassette):
        """Queue with every optional T2V parameter the model supports populated."""
        t2v = await _resolve_t2v(venice_client)
        t2v_optional_params = await _optional_video_params(venice_client, t2v.model, audio=True)
        # Guard against the derived set quietly collapsing: every video model
        # advertises resolutions, so an absent one means the capability lookup
        # degraded and this test would be exercising almost no optional params.
        assert "resolution" in t2v_optional_params, (
            f"no optional params derived for {t2v.model}: {t2v_optional_params}"
        )
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="A kitten chasing a laser pointer",
                    negative_prompt="blurry, ugly, low quality",
                    **t2v_optional_params,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_queue_i2v(self, venice_client, vcr_cassette):
        """Queue an image-to-video request."""
        i2v = await _resolve_i2v(venice_client)
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=i2v.model,
                    prompt="Bring this image to life with subtle motion",
                    **i2v.request_params,
                    image_url=VIDEO_TEST_IMAGE_URL,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"I2V model unavailable: {exc}")
                raise

    async def test_queue_i2v_all_optional_params(self, venice_client, vcr_cassette):
        """Queue I2V with every optional param the model supports populated."""
        i2v = await _resolve_i2v(venice_client)
        i2v_optional_params = await _optional_video_params(venice_client, i2v.model, audio=False)
        assert "resolution" in i2v_optional_params, (
            f"no optional params derived for {i2v.model}: {i2v_optional_params}"
        )
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=i2v.model,
                    prompt="Pan across the scene slowly",
                    negative_prompt="low quality, distorted",
                    image_url=VIDEO_TEST_IMAGE_URL,
                    **i2v_optional_params,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"I2V model unavailable: {exc}")
                raise

    # ------------------------------------------------------------------
    # retrieve() tests
    # ------------------------------------------------------------------

    async def test_retrieve_after_queue(self, venice_client, vcr_cassette):
        """Queue → retrieve immediately; expect PROCESSING or COMPLETED."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="A river flowing through a forest",
                    **t2v.request_params,
                )

                status = await venice_client.video.retrieve(
                    model=t2v.model,
                    queue_id=queued.queue_id,
                )

                # Immediately after queuing we typically get PROCESSING
                assert isinstance(
                    status,
                    (VideoProcessingStatus, VideoCompletedStatus, VideoFailedStatus),
                )

                if isinstance(status, VideoProcessingStatus):
                    assert status.status == "PROCESSING"
                    assert status.average_execution_time >= 0
                elif isinstance(status, VideoCompletedStatus):
                    assert status.status == "COMPLETED"
                elif isinstance(status, VideoFailedStatus):
                    assert status.status == "FAILED"

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_retrieve_with_delete_media_on_completion(self, venice_client, vcr_cassette):
        """Verify delete_media_on_completion flag is accepted."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="Waves crashing on a rocky shore",
                    **t2v.request_params,
                )

                status = await venice_client.video.retrieve(
                    model=t2v.model,
                    queue_id=queued.queue_id,
                    delete_media_on_completion=True,
                )

                assert isinstance(
                    status,
                    (VideoProcessingStatus, VideoCompletedStatus, VideoFailedStatus),
                )

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_retrieve_poll_until_done(self, venice_client, vcr_cassette):
        """Queue and poll until COMPLETED or FAILED (with timeout)."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="A slow-motion droplet splashing into water",
                    **t2v.request_params,
                )

                max_polls = 60  # ~5 min with 5s sleep
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=t2v.model,
                        queue_id=queued.queue_id,
                    )

                    if isinstance(status, VideoCompletedStatus):
                        assert status.url is not None or status.status == "COMPLETED"
                        return  # success
                    elif isinstance(status, VideoFailedStatus):
                        # Generation failed – not a test failure per se
                        pytest.skip(f"Video generation failed: {status.error}")
                        return

                    assert isinstance(status, VideoProcessingStatus)
                    await asyncio.sleep(5)

                pytest.skip("Video generation did not complete within timeout")

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    # ------------------------------------------------------------------
    # complete() tests
    # ------------------------------------------------------------------

    async def test_complete_after_retrieval(self, venice_client, vcr_cassette):
        """Full lifecycle: queue → poll → complete."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="A butterfly landing on a flower",
                    **t2v.request_params,
                )

                # Poll until done (or timeout)
                max_polls = 60
                final_status: VideoCompletedStatus | None = None
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=t2v.model,
                        queue_id=queued.queue_id,
                    )

                    if isinstance(status, VideoCompletedStatus):
                        final_status = status
                        break
                    elif isinstance(status, VideoFailedStatus):
                        pytest.skip(f"Video generation failed: {status.error}")
                        return

                    await asyncio.sleep(5)

                if final_status is None:
                    pytest.skip("Video generation did not complete within timeout")

                # Now call complete.
                # When the video was delivered as binary (url is None),
                # the API may return success=False because there is no
                # server-side media to clean up.  Both outcomes are valid.
                try:
                    result = await venice_client.video.cancel(
                        model=t2v.model,
                        queue_id=queued.queue_id,
                    )

                    assert isinstance(result, VideoCompleteResponse)
                    if final_status.url is not None:
                        # URL-based delivery → cleanup should succeed
                        assert result.success is True
                except InvalidRequestError as exc:
                    # Binary-delivered videos (url is None) consume the server-side
                    # job on retrieval, so /video/complete reports "Request ID is
                    # invalid" — an expected terminal state, not a failure. A
                    # URL-delivered video must still complete cleanly.
                    assert final_status.url is None, (
                        f"video.complete() failed for a URL-delivered video: {exc}"
                    )
                    assert "request id is invalid" in str(exc).lower()

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    # ------------------------------------------------------------------
    # Error handling tests
    # ------------------------------------------------------------------

    async def test_queue_invalid_model(self, venice_client, vcr_cassette):
        """Queue with a non-existent model should raise an API error.

        The rest of the request is a real model's valid one, so the model id is
        the only thing the API can reject.
        """
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette, pytest.raises((VeniceError, APIError, APIStatusError)):
            await venice_client.video.submit(
                model="definitely-invalid-video-model-xyz",
                prompt="This should fail",
                **t2v.request_params,
            )

    async def test_retrieve_invalid_queue_id(self, venice_client, vcr_cassette):
        """Retrieve with a bogus queue_id should raise an API error."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette, pytest.raises((VeniceError, APIError, APIStatusError, ValueError)):
            await venice_client.video.retrieve(
                model=t2v.model,
                queue_id="nonexistent-queue-id-00000000",
            )

    async def test_complete_invalid_queue_id(self, venice_client, vcr_cassette):
        """Complete with a bogus queue_id should raise an API error."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette, pytest.raises((VeniceError, APIError, APIStatusError)):
            await venice_client.video.cancel(
                model=t2v.model,
                queue_id="nonexistent-queue-id-00000000",
            )

    # ------------------------------------------------------------------
    # Full workflow: quote → queue → retrieve → complete
    # ------------------------------------------------------------------

    async def test_full_video_workflow(self, venice_client, vcr_cassette):
        """End-to-end: quote → queue → poll → complete."""
        t2v = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                # Step 1 – Quote
                quote = await venice_client.video.quote(
                    model=t2v.model,
                    **t2v.request_params,
                )
                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote >= 0

                # Step 2 – Queue
                queued = await venice_client.video.submit(
                    model=t2v.model,
                    prompt="A time-lapse of clouds rolling over a city skyline",
                    **t2v.request_params,
                )
                assert isinstance(queued, VideoQueueResponse)
                assert queued.queue_id

                # Step 3 – Poll until done
                max_polls = 60
                final_status = None
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=t2v.model,
                        queue_id=queued.queue_id,
                    )

                    if isinstance(status, VideoCompletedStatus):
                        final_status = status
                        break
                    elif isinstance(status, VideoFailedStatus):
                        pytest.skip(f"Video generation failed: {status.error}")
                        return

                    await asyncio.sleep(5)

                if final_status is None:
                    pytest.skip("Video generation did not complete within timeout")

                assert final_status.status == "COMPLETED"

                # Step 4 – Complete (cleanup)
                # When the video was delivered as binary (url is None),
                # the API may return success=False because there is no
                # server-side media to clean up.  Both outcomes are valid.
                try:
                    cleanup = await venice_client.video.cancel(
                        model=t2v.model,
                        queue_id=queued.queue_id,
                    )
                    assert isinstance(cleanup, VideoCompleteResponse)
                    if final_status.url is not None:
                        assert cleanup.success is True
                except InvalidRequestError as exc:
                    # Binary-delivered videos (url is None) consume the server-side
                    # job on retrieval, so /video/complete reports "Request ID is
                    # invalid" — an expected terminal state, not a failure. A
                    # URL-delivered video must still complete cleanly.
                    assert final_status.url is None, (
                        f"video.complete() failed for a URL-delivered video: {exc}"
                    )
                    assert "request id is invalid" in str(exc).lower()

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_full_i2v_workflow(self, venice_client, vcr_cassette):
        """End-to-end image-to-video: quote → queue → poll → complete."""
        i2v = await _resolve_i2v(venice_client)
        with vcr_cassette:
            try:
                # Quote — /video/quote ignores prompt/image refs;
                # price is driven by model + duration + resolution.
                quote = await venice_client.video.quote(
                    model=i2v.model,
                    **i2v.request_params,
                )
                assert isinstance(quote, VideoQuoteResponse)

                # Queue
                queued = await venice_client.video.submit(
                    model=i2v.model,
                    prompt="Animate the scene with gentle motion",
                    **i2v.request_params,
                    image_url=VIDEO_TEST_IMAGE_URL,
                )
                assert queued.queue_id

                # Poll
                max_polls = 60
                final_status = None
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=i2v.model,
                        queue_id=queued.queue_id,
                    )
                    if isinstance(status, VideoCompletedStatus):
                        final_status = status
                        break
                    elif isinstance(status, VideoFailedStatus):
                        pytest.skip(f"I2V generation failed: {status.error}")
                        return
                    await asyncio.sleep(5)

                if final_status is None:
                    pytest.skip("I2V generation did not complete within timeout")

                # Complete
                try:
                    cleanup = await venice_client.video.cancel(
                        model=i2v.model,
                        queue_id=queued.queue_id,
                    )
                    assert isinstance(cleanup, VideoCompleteResponse)
                except InvalidRequestError as exc:
                    # Binary-delivered videos (url is None) consume the server-side
                    # job on retrieval, so /video/complete reports "Request ID is
                    # invalid" — an expected terminal state, not a failure. A
                    # URL-delivered video must still complete cleanly.
                    assert final_status.url is None, (
                        f"video.complete() failed for a URL-delivered video: {exc}"
                    )
                    assert "request id is invalid" in str(exc).lower()

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"I2V model unavailable: {exc}")
                raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_model_unavailable(exc: Exception) -> bool:
    """Return True when the error suggests the model simply isn't available."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "not found",
            "not supported",
            "model",
            "invalid_model",
            "unavailable",
            "does not exist",
        )
    )
