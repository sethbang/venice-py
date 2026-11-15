"""
Seedance 2.5 drafts, per-reference objects and the image-aware quote on the
video resource.

Covers what reaches the wire for ``draft``, ``draft_task_id``,
``image_references`` and ``video_references`` on ``/video/queue``; ``image_url``,
``draft`` and ``draft_task_id`` on ``/video/quote``; :meth:`Video.upgrade_draft`;
the draft fields on queue and retrieve responses; and the local checks that
reject a malformed request before it is sent.
"""

import inspect
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import TypeAdapter, ValidationError

from venice_ai.resources.video import Video
from venice_ai.types import VideoImageReference, VideoVideoReference
from venice_ai.types.api import (
    VideoCompletedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
    VideoQuoteResponse,
)
from venice_ai.types.api.models import ModelResponse, VideoModelConstraints, VideoModelSpec
from venice_ai.types.api.requests.video import VideoQuoteRequest, VideoTextToVideoRequest
from venice_ai.types.api.video import VideoRetrieveResponse

MODEL = "seedance-2-5-text-to-video"
DRAFT_ID = "123e4567-e89b-12d3-a456-426614174000"


def _video_client(*, durations: list[str] | None = None) -> Mock:
    """A client whose catalog lists ``durations`` for MODEL (or misses the model)."""
    client = Mock()
    client.post = AsyncMock(return_value=VideoQueueResponse(model=MODEL, queue_id="q-1"))
    client.models = Mock()
    if durations is None:
        client.models.get = AsyncMock(side_effect=LookupError("not in catalog"))
    else:
        spec = VideoModelSpec(
            name=MODEL,
            constraints=VideoModelConstraints(model_type="text-to-video", durations=durations),
        )
        client.models.get = AsyncMock(
            return_value=ModelResponse.model_validate(
                {
                    "id": MODEL,
                    "type": "video",
                    "object": "model",
                    "owned_by": "venice.ai",
                    "model_spec": spec.model_dump(),
                }
            )
        )
    return client


def _body(client: Mock) -> dict[str, Any]:
    body: dict[str, Any] = client.post.await_args.kwargs["json_data"]
    return body


class TestQueueWireBody:
    @pytest.mark.asyncio
    async def test_draft_is_forwarded(self) -> None:
        client = _video_client()
        await Video(client).submit(model=MODEL, prompt="p", duration_seconds=5, draft=True)
        assert client.post.await_args.args[0] == "video/queue"
        assert _body(client)["draft"] is True

    @pytest.mark.asyncio
    async def test_references_accept_models_and_dicts(self) -> None:
        client = _video_client()
        await Video(client).submit(
            model=MODEL,
            prompt="@Hero walks into @Forest",
            duration_seconds=5,
            image_references=[
                VideoImageReference(
                    image_url="https://example.com/hero.png",
                    type="subject",
                    ref_name="Hero",
                    audio_url="https://example.com/hero.mp3",
                ),
                {"image_url": "https://example.com/forest.png", "type": "background"},
            ],
            video_references=[{"video_url": "https://example.com/clip.mp4", "ref_name": "Walk"}],
        )
        body = _body(client)
        assert body["image_references"] == [
            {
                "image_url": "https://example.com/hero.png",
                "type": "subject",
                "ref_name": "Hero",
                "audio_url": "https://example.com/hero.mp3",
            },
            {"image_url": "https://example.com/forest.png", "type": "background"},
        ]
        assert body["video_references"] == [
            {"video_url": "https://example.com/clip.mp4", "ref_name": "Walk"}
        ]

    @pytest.mark.asyncio
    async def test_new_fields_absent_when_unset(self) -> None:
        client = _video_client()
        await Video(client).submit(model=MODEL, prompt="p", duration_seconds=5)
        body = _body(client)
        for key in ("draft", "draft_task_id", "image_references", "video_references"):
            assert key not in body

    @pytest.mark.asyncio
    async def test_run_forwards_draft_and_references(self) -> None:
        client = _video_client()
        job = await Video(client).run(
            model=MODEL,
            prompt="p",
            duration_seconds=5,
            draft=True,
            image_references=[{"image_url": "https://example.com/a.png"}],
            video_references=[{"video_url": "https://example.com/a.mp4"}],
        )
        body = _body(client)
        assert body["draft"] is True
        assert body["image_references"] == [{"image_url": "https://example.com/a.png"}]
        assert body["video_references"] == [{"video_url": "https://example.com/a.mp4"}]
        assert job.queue_id == "q-1"


class TestUpgradeDraft:
    """The API validates an upgrade like a new generation (live 400s, 2026-11-15).

    With only ``model`` and ``draft_task_id``, ``/video/queue`` named the missing
    fields: ``prompt``, ``duration`` and ``aspect_ratio`` on a text-to-video model,
    ``prompt`` and ``duration`` on an image-to-video one (no ``image_url``). With
    those added, both reached the draft lookup.
    """

    @pytest.mark.asyncio
    async def test_sends_the_fields_the_api_requires(self) -> None:
        client = _video_client(durations=["5s", "10s"])
        result = await Video(client).upgrade_draft(
            model=MODEL,
            draft_task_id=DRAFT_ID,
            prompt="A paper boat in the rain",
            duration_seconds=5,
            aspect_ratio="16:9",
            resolution="1080p",
        )
        assert client.post.await_args.args[0] == "video/queue"
        assert _body(client) == {
            "model": MODEL,
            "draft_task_id": DRAFT_ID,
            "prompt": "A paper boat in the rain",
            "duration": "5s",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
        }
        assert result.queue_id == "q-1"

    @pytest.mark.asyncio
    async def test_image_to_video_upgrade_sends_no_media(self) -> None:
        client = _video_client()
        await Video(client).upgrade_draft(
            model="seedance-2-5-image-to-video-basic",
            draft_task_id=DRAFT_ID,
            prompt="p",
            duration_seconds=5,
        )
        assert _body(client) == {
            "model": "seedance-2-5-image-to-video-basic",
            "draft_task_id": DRAFT_ID,
            "prompt": "p",
            "duration": "5s",
        }

    def test_media_inputs_are_not_parameters(self) -> None:
        """The upgrade follows the draft, so it takes no image or video input."""
        params = inspect.signature(Video.upgrade_draft).parameters
        for name in ("image_url", "video_url", "reference_image_urls"):
            assert name not in params

    @pytest.mark.asyncio
    async def test_consents_are_forwarded(self) -> None:
        client = _video_client()
        consents = {
            "seedance": {
                "confirmed_terms_and_privacy": True,
                "confirmed_legal_right": True,
                "confirmed_screening_acknowledged": True,
            }
        }
        await Video(client).upgrade_draft(
            model=MODEL,
            draft_task_id=DRAFT_ID,
            prompt="p",
            duration_seconds=5,
            aspect_ratio="16:9",
            consents=consents,
        )
        assert _body(client)["consents"] == consents

    def test_prompt_and_duration_are_required(self) -> None:
        params = inspect.signature(Video.upgrade_draft).parameters
        for name in ("model", "draft_task_id", "prompt", "duration_seconds"):
            assert params[name].default is inspect.Parameter.empty, name

    @pytest.mark.asyncio
    async def test_unsupported_duration_is_refused_before_sending(self) -> None:
        client = _video_client(durations=["5s", "10s"])
        with pytest.raises(ValueError, match="not supported"):
            await Video(client).upgrade_draft(
                model=MODEL, draft_task_id=DRAFT_ID, prompt="p", duration_seconds=7
            )
        client.post.assert_not_awaited()


class TestQuote:
    @pytest.mark.asyncio
    async def test_image_url_and_draft_fields_are_forwarded(self) -> None:
        client = _video_client()
        client.post = AsyncMock(return_value=VideoQuoteResponse(quote=0.25))
        await Video(client).quote(
            model=MODEL,
            duration_seconds=5,
            aspect_ratio="auto",
            image_url="data:image/png;base64,iVBORw0K",
            draft=True,
        )
        assert client.post.await_args.args[0] == "video/quote"
        body = _body(client)
        assert body["image_url"] == "data:image/png;base64,iVBORw0K"
        assert body["aspect_ratio"] == "auto"
        assert body["draft"] is True

    @pytest.mark.asyncio
    async def test_upgrade_quote_sends_draft_task_id_with_duration(self) -> None:
        client = _video_client(durations=["5s", "10s"])
        client.post = AsyncMock(return_value=VideoQuoteResponse(quote=1.5))
        await Video(client).quote(model=MODEL, draft_task_id=DRAFT_ID, duration_seconds=5)
        assert _body(client) == {"model": MODEL, "draft_task_id": DRAFT_ID, "duration": "5s"}

    @pytest.mark.asyncio
    async def test_upgrade_quote_without_duration_is_refused_locally(self) -> None:
        """The API answered 400 ``duration: Required`` for this (live, 2026-11-15)."""
        client = _video_client(durations=["5s", "10s"])
        with pytest.raises(ValueError, match="duration_seconds is required"):
            await Video(client).quote(model=MODEL, draft_task_id=DRAFT_ID)
        client.post.assert_not_awaited()

    def test_quote_rejects_a_non_url_image(self) -> None:
        with pytest.raises(ValidationError, match="URL must start with"):
            VideoQuoteRequest(model=MODEL, duration="5s", image_url="photo.png")


class TestLocalChecks:
    def test_draft_and_upgrade_cannot_be_combined_on_queue(self) -> None:
        with pytest.raises(ValidationError, match="cannot be combined"):
            VideoTextToVideoRequest(model=MODEL, draft=True, draft_task_id=DRAFT_ID)

    def test_draft_false_with_upgrade_is_also_refused(self) -> None:
        """Sending ``"draft": false`` beside ``draft_task_id`` still combines the two."""
        with pytest.raises(ValidationError, match="cannot be combined"):
            VideoTextToVideoRequest(model=MODEL, draft=False, draft_task_id=DRAFT_ID)

    def test_draft_and_upgrade_cannot_be_combined_on_quote(self) -> None:
        with pytest.raises(ValidationError, match="cannot be combined"):
            VideoQuoteRequest(model=MODEL, draft=True, draft_task_id=DRAFT_ID)

    @pytest.mark.asyncio
    async def test_combined_draft_fields_never_reach_the_api(self) -> None:
        client = _video_client()
        with pytest.raises(ValidationError):
            await Video(client).submit(
                model=MODEL, prompt="p", duration_seconds=5, draft=True, draft_task_id=DRAFT_ID
            )
        client.post.assert_not_awaited()

    def test_image_reference_type_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            VideoImageReference(image_url="https://example.com/a.png", type="foreground")  # type: ignore[arg-type]

    def test_reference_urls_must_be_urls(self) -> None:
        with pytest.raises(ValidationError, match="URL must start with"):
            VideoImageReference(image_url="a.png")
        with pytest.raises(ValidationError, match="URL must start with"):
            VideoVideoReference(video_url="https://example.com/a.mp4", audio_url="voice.mp3")

    def test_reference_objects_reject_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            VideoVideoReference.model_validate(
                {"video_url": "https://example.com/a.mp4", "role": "reference_video"}
            )

    @pytest.mark.parametrize(
        ("field", "limit", "ext"),
        [
            ("reference_image_urls", 30, "png"),
            ("reference_video_urls", 10, "mp4"),
            ("reference_audio_urls", 10, "mp3"),
        ],
    )
    def test_plain_reference_lists_take_the_spec_limits(
        self, field: str, limit: int, ext: str
    ) -> None:
        """The SDK used to cap these at 9, 3 and 3, refusing requests the API accepts."""
        urls = [f"https://example.com/{i}.{ext}" for i in range(limit)]
        request = VideoTextToVideoRequest.model_validate({"model": MODEL, field: urls})
        assert len(getattr(request, field)) == limit
        with pytest.raises(ValidationError):
            VideoTextToVideoRequest.model_validate(
                {"model": MODEL, field: [*urls, f"https://example.com/x.{ext}"]}
            )

    def test_reference_list_limits(self) -> None:
        images = [{"image_url": f"https://example.com/{i}.png"} for i in range(31)]
        with pytest.raises(ValidationError):
            VideoTextToVideoRequest.model_validate({"model": MODEL, "image_references": images})
        clips = [{"video_url": f"https://example.com/{i}.mp4"} for i in range(6)]
        with pytest.raises(ValidationError):
            VideoTextToVideoRequest.model_validate({"model": MODEL, "video_references": clips})


class TestDraftResponseFields:
    DRAFT_FIELDS = {
        "is_draft": True,
        "draft_task_id": DRAFT_ID,
        "can_upgrade_to_hd": True,
        "draft_expires_at": "2026-11-20T12:00:00Z",
    }

    def test_queue_response_types_the_draft_fields(self) -> None:
        resp = VideoQueueResponse.model_validate(
            {"model": MODEL, "queue_id": DRAFT_ID, **self.DRAFT_FIELDS}
        )
        assert resp.is_draft is True
        assert resp.draft_task_id == DRAFT_ID
        assert resp.can_upgrade_to_hd is True
        assert resp.draft_expires_at == "2026-11-20T12:00:00Z"

    @pytest.mark.parametrize(
        ("status", "cls"),
        [("PROCESSING", VideoProcessingStatus), ("COMPLETED", VideoCompletedStatus)],
    )
    def test_retrieve_statuses_type_the_draft_fields(self, status: str, cls: type) -> None:
        resp = TypeAdapter(VideoRetrieveResponse).validate_python(
            {"status": status, **self.DRAFT_FIELDS}
        )
        assert isinstance(resp, cls)
        assert resp.draft_task_id == DRAFT_ID
        assert resp.can_upgrade_to_hd is True

    def test_per_reference_audio_is_typed(self) -> None:
        constraints = VideoModelConstraints.model_validate(
            {"model_type": "reference-to-video", "per_reference_audio": True}
        )
        assert constraints.per_reference_audio is True
        assert "per_reference_audio" not in (constraints.model_extra or {})

    def test_non_draft_response_leaves_them_unset(self) -> None:
        resp = VideoQueueResponse.model_validate({"model": MODEL, "queue_id": "q-1"})
        assert resp.is_draft is None
        assert resp.draft_task_id is None
