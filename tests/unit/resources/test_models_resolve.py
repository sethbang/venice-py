"""Unit tests for Models.resolve() and convenience shortcuts."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.exceptions import NoMatchingModelError
from venice_ai.resources.models import Models


@pytest.fixture
def mock_client():
    """Create a mock Venice client."""
    client = AsyncMock()
    client.get = AsyncMock()
    return client


@pytest.fixture
def models_resource(mock_client):
    """Create a Models resource with a mock client."""
    return Models(mock_client)


# ============================================================================
# _get_selector / lazy init
# ============================================================================


class TestGetSelector:
    """Tests for lazy DynamicModelSelector initialization."""

    def test_selector_starts_none(self, models_resource):
        assert models_resource._selector is None

    def test_get_selector_creates_instance(self, models_resource):
        selector = models_resource._get_selector()
        assert selector is not None
        assert models_resource._selector is selector

    def test_get_selector_reuses_instance(self, models_resource):
        s1 = models_resource._get_selector()
        s2 = models_resource._get_selector()
        assert s1 is s2


# ============================================================================
# resolve() dispatch
# ============================================================================


class TestResolve:
    """Tests for Models.resolve() dispatching to selector methods."""

    @pytest.mark.asyncio
    async def test_resolve_chat(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_chat_model = AsyncMock(return_value="llama-3.3-70b")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="chat")
        assert result == "llama-3.3-70b"
        mock_selector.select_chat_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_chat_with_capabilities(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_chat_model = AsyncMock(return_value="gpt-model")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(
            type="chat",
            require_function_calling=True,
            require_vision=True,
            min_context_tokens=8000,
        )
        assert result == "gpt-model"
        call_kwargs = mock_selector.select_chat_model.call_args[1]
        assert call_kwargs["require_function_calling"] is True
        assert call_kwargs["require_vision"] is True
        assert call_kwargs["min_context_tokens"] == 8000

    @pytest.mark.asyncio
    async def test_resolve_embedding(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_embedding_model = AsyncMock(return_value="bge-m3")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="embedding")
        assert result == "bge-m3"
        mock_selector.select_embedding_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_image(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_image_model = AsyncMock(return_value="flux-schnell")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="image")
        assert result == "flux-schnell"

    @pytest.mark.asyncio
    async def test_resolve_video(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_video_model = AsyncMock(return_value="kling-video")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="video", video_type="text-to-video")
        assert result == "kling-video"
        call_kwargs = mock_selector.select_video_model.call_args[1]
        assert call_kwargs["model_type"] == "text-to-video"

    @pytest.mark.asyncio
    async def test_resolve_tts(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_audio_model = AsyncMock(return_value="tts-kokoro")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="tts")
        assert result == "tts-kokoro"

    @pytest.mark.asyncio
    async def test_resolve_asr(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_asr_model = AsyncMock(return_value="whisper")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="asr")
        assert result == "whisper"

    @pytest.mark.asyncio
    async def test_resolve_inpaint(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_inpaint_model = AsyncMock(return_value="inpaint-v1")
        models_resource._selector = mock_selector

        result = await models_resource.resolve(type="inpaint")
        assert result == "inpaint-v1"

    @pytest.mark.asyncio
    async def test_resolve_unknown_type_raises(self, models_resource):
        mock_selector = MagicMock()
        models_resource._selector = mock_selector

        with pytest.raises(ValueError, match="Unknown model type"):
            await models_resource.resolve(type="unknown")

    @pytest.mark.asyncio
    async def test_resolve_with_exclude_models(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_chat_model = AsyncMock(return_value="model-b")
        models_resource._selector = mock_selector

        await models_resource.resolve(type="chat", exclude_models=["model-a"])
        call_kwargs = mock_selector.select_chat_model.call_args[1]
        assert call_kwargs["exclude_models"] == {"model-a"}

    @pytest.mark.asyncio
    async def test_resolve_with_preferred_models(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_chat_model = AsyncMock(return_value="preferred-model")
        models_resource._selector = mock_selector

        await models_resource.resolve(type="chat", preferred_models=["preferred-model"])
        call_kwargs = mock_selector.select_chat_model.call_args[1]
        assert call_kwargs["preferred_models"] == ["preferred-model"]


# ============================================================================
# Convenience shortcuts
# ============================================================================


class TestConvenienceShortcuts:
    """Tests for resolve_chat, resolve_embedding, etc."""

    @pytest.mark.asyncio
    async def test_resolve_chat_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_chat_model = AsyncMock(return_value="llama")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_chat(require_function_calling=True)
        assert result == "llama"

    @pytest.mark.asyncio
    async def test_resolve_embedding_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_embedding_model = AsyncMock(return_value="bge-m3")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_embedding()
        assert result == "bge-m3"

    @pytest.mark.asyncio
    async def test_resolve_image_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_image_model = AsyncMock(return_value="flux")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_image()
        assert result == "flux"

    @pytest.mark.asyncio
    async def test_resolve_video_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_video_model = AsyncMock(return_value="kling")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_video(video_type="text-to-video")
        assert result == "kling"

    @pytest.mark.asyncio
    async def test_resolve_tts_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_audio_model = AsyncMock(return_value="kokoro")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_tts()
        assert result == "kokoro"

    @pytest.mark.asyncio
    async def test_resolve_asr_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_asr_model = AsyncMock(return_value="whisper")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_asr()
        assert result == "whisper"

    @pytest.mark.asyncio
    async def test_resolve_inpaint_shortcut(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_inpaint_model = AsyncMock(return_value="inpaint")
        models_resource._selector = mock_selector

        result = await models_resource.resolve_inpaint()
        assert result == "inpaint"


# ============================================================================
# resolve_video_upscale
# ============================================================================


class TestResolveVideoUpscale:
    """resolve_video_upscale must filter type=video for video-input upscalers."""

    @staticmethod
    def _video_model(model_id: str, *, model_type: str, video_input: bool) -> MagicMock:
        m = MagicMock()
        m.id = model_id
        m.model_spec.constraints.model_type = model_type
        m.model_spec.constraints.video_input = video_input
        return m

    @pytest.mark.asyncio
    async def test_picks_topaz_video_upscale_over_t2v_and_i2v(self, models_resource):
        listing = MagicMock()
        listing.data = [
            self._video_model(
                "veo3.1-text-to-video", model_type="text-to-video", video_input=False
            ),
            self._video_model(
                "sora-2-image-to-video", model_type="image-to-video", video_input=False
            ),
            self._video_model("topaz-video-upscale", model_type="video", video_input=True),
        ]
        models_resource.list = AsyncMock(return_value=listing)

        chosen = await models_resource.resolve_video_upscale()
        assert chosen == "topaz-video-upscale"

    @pytest.mark.asyncio
    async def test_raises_when_no_video_upscaler_available(self, models_resource):
        listing = MagicMock()
        listing.data = [
            self._video_model(
                "veo3.1-text-to-video", model_type="text-to-video", video_input=False
            ),
        ]
        models_resource.list = AsyncMock(return_value=listing)

        with pytest.raises(NoMatchingModelError, match="No video-upscaling model available"):
            await models_resource.resolve_video_upscale()

    @pytest.mark.asyncio
    async def test_preferred_models_win_when_present(self, models_resource):
        listing = MagicMock()
        listing.data = [
            self._video_model("topaz-video-upscale", model_type="video", video_input=True),
            self._video_model("future-video-upscale", model_type="video", video_input=True),
        ]
        models_resource.list = AsyncMock(return_value=listing)

        chosen = await models_resource.resolve_video_upscale(
            preferred_models=["future-video-upscale"]
        )
        assert chosen == "future-video-upscale"

    @pytest.mark.asyncio
    async def test_id_substring_fallback_when_constraints_missing(self, models_resource):
        # Defensive case: constraints missing/odd, but the id contains "upscale".
        m = MagicMock()
        m.id = "some-future-upscale"
        m.model_spec = None
        listing = MagicMock()
        listing.data = [m]
        models_resource.list = AsyncMock(return_value=listing)

        chosen = await models_resource.resolve_video_upscale()
        assert chosen == "some-future-upscale"


# ============================================================================
# resolve_cheapest_video
# ============================================================================


class TestResolveCheapestVideo:
    """Tests for Models.resolve_cheapest_video()."""

    @pytest.mark.asyncio
    async def test_resolve_cheapest_video(self, models_resource):
        mock_result = MagicMock()
        mock_result.model = "cheap-video"
        mock_result.price_usd = 0.01

        mock_selector = MagicMock()
        mock_selector.select_cheapest_video_model = AsyncMock(return_value=mock_result)
        models_resource._selector = mock_selector

        result = await models_resource.resolve_cheapest_video(duration="5s")
        assert result.model == "cheap-video"

        call_kwargs = mock_selector.select_cheapest_video_model.call_args[1]
        assert call_kwargs["duration"] == "5s"
        assert "prompt" not in call_kwargs

    @pytest.mark.asyncio
    async def test_resolve_cheapest_video_with_exclude(self, models_resource):
        mock_selector = MagicMock()
        mock_selector.select_cheapest_video_model = AsyncMock(return_value=MagicMock())
        models_resource._selector = mock_selector

        await models_resource.resolve_cheapest_video(exclude_models=["expensive-model"])

        call_kwargs = mock_selector.select_cheapest_video_model.call_args[1]
        assert call_kwargs["exclude_models"] == {"expensive-model"}
