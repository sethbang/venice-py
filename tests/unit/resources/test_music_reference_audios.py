"""
Tests for ``reference_audio_urls`` on ``client.music.submit`` / ``run``.

Covers the wire body, the local request checks (inline ``data:`` audio only,
no preset ``voice`` alongside), and the catalog preflight: a declared
``supports_reference_audios=False`` or a list longer than
``max_reference_audios`` fails before anything is sent, while an undeclared
capability or a catalog miss defers to the server.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from venice_ai.resources.music import Music
from venice_ai.types.api.models import ModelResponse, MusicModelSpec
from venice_ai.types.api.music import MusicQueueResponse
from venice_ai.types.api.requests.music import MusicQueueRequest

MODEL = "music-model"
CLIP = "data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0U="


def _client(
    *, supports: bool | None = None, limit: int | None = None, catalog_miss: bool = False
) -> Mock:
    client = Mock()
    client.post = AsyncMock(return_value=MusicQueueResponse(model=MODEL, queue_id="q-1"))
    client.models = Mock()
    if catalog_miss:
        client.models.get = AsyncMock(side_effect=LookupError("not in catalog"))
    else:
        spec = MusicModelSpec(
            name=MODEL, supports_reference_audios=supports, max_reference_audios=limit
        )
        client.models.get = AsyncMock(
            return_value=ModelResponse.model_validate(
                {
                    "id": MODEL,
                    "type": "music",
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


class TestWireBody:
    @pytest.mark.asyncio
    async def test_submit_forwards_clips_in_order(self) -> None:
        client = _client(supports=True, limit=3)
        await Music(client).submit(
            model=MODEL, prompt="Sing over @Audio1 and @Audio2", reference_audio_urls=[CLIP, CLIP]
        )
        assert client.post.await_args.args[0] == "audio/queue"
        assert _body(client)["reference_audio_urls"] == [CLIP, CLIP]

    @pytest.mark.asyncio
    async def test_run_forwards_clips(self) -> None:
        client = _client(supports=True)
        await Music(client).run(model=MODEL, prompt="p", reference_audio_urls=[CLIP])
        assert _body(client)["reference_audio_urls"] == [CLIP]

    @pytest.mark.asyncio
    async def test_absent_when_unset(self) -> None:
        client = _client()
        await Music(client).submit(model=MODEL, prompt="p")
        assert "reference_audio_urls" not in _body(client)

    @pytest.mark.asyncio
    async def test_describe_in_prompt_voice_is_allowed(self) -> None:
        client = _client(supports=True)
        await Music(client).submit(
            model=MODEL, prompt="p", voice="Describe in prompt", reference_audio_urls=[CLIP]
        )
        assert _body(client)["voice"] == "Describe in prompt"


class TestRequestChecks:
    def test_http_url_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="data: URL"):
            MusicQueueRequest(
                model=MODEL, prompt="p", reference_audio_urls=["https://example.com/a.mp3"]
            )

    def test_preset_voice_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="preset voice"):
            MusicQueueRequest(model=MODEL, prompt="p", voice="Aria", reference_audio_urls=[CLIP])

    def test_empty_list_is_dropped(self) -> None:
        """A model without the capability may reject the field's mere presence."""
        request = MusicQueueRequest(model=MODEL, prompt="p", reference_audio_urls=[])
        assert request.reference_audio_urls is None
        assert "reference_audio_urls" not in request.model_dump(exclude_none=True)


class TestPreflight:
    @pytest.mark.asyncio
    async def test_declared_unsupported_raises_before_sending(self) -> None:
        client = _client(supports=False)
        with pytest.raises(ValueError, match="not supported"):
            await Music(client).submit(model=MODEL, prompt="p", reference_audio_urls=[CLIP])
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exactly_the_limit_is_sent(self) -> None:
        client = _client(supports=True, limit=2)
        await Music(client).submit(model=MODEL, prompt="p", reference_audio_urls=[CLIP, CLIP])
        assert _body(client)["reference_audio_urls"] == [CLIP, CLIP]

    @pytest.mark.asyncio
    async def test_too_many_clips_raises_before_sending(self) -> None:
        client = _client(supports=True, limit=1)
        with pytest.raises(ValueError, match="at most 1"):
            await Music(client).submit(model=MODEL, prompt="p", reference_audio_urls=[CLIP, CLIP])
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_kwargs",
        [{"supports": None, "limit": None}, {"catalog_miss": True}],
        ids=["undeclared", "catalog-miss"],
    )
    async def test_unknown_capability_defers_to_server(self, client_kwargs: dict) -> None:
        client = _client(**client_kwargs)
        await Music(client).submit(model=MODEL, prompt="p", reference_audio_urls=[CLIP, CLIP])
        client.post.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("clips", [None, []], ids=["none", "empty"])
    async def test_no_preflight_without_clips(self, clips: list[str] | None) -> None:
        client = _client(supports=False)
        await Music(client).submit(model=MODEL, prompt="p", reference_audio_urls=clips)
        client.post.assert_awaited_once()
