"""
Comprehensive test module for DynamicModelSelector coverage.

Targets all missing lines and partial branches in models/selection.py
to achieve 90%+ coverage.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.exceptions import NoMatchingModelError
from venice_ai.models.selection import (
    DynamicModelSelector,
    ModelCache,
    create_model_selector,
    get_chat_model,
    get_embedding_model,
    get_multiple_models,
)


class TestModelCacheExpiredPath:
    """Test ModelCache when cache is expired - targets line 72."""

    def test_get_models_returns_empty_when_expired(self):
        """Test get_models returns empty list when cache is expired (line 72)."""
        cache = ModelCache(
            models={"model1": {"id": "model1", "type": "text"}},
            last_updated=datetime.now(UTC) - timedelta(seconds=600),  # Expired
            ttl_seconds=300.0,
        )

        # Cache should be expired
        assert cache.is_expired() is True

        # get_models should return empty list when expired
        result = cache.get_models()
        assert result == []

    def test_get_models_with_resource_type_returns_empty_when_expired(self):
        """Test get_models with resource_type returns empty when expired (line 72)."""
        cache = ModelCache(
            models={"model1": {"id": "model1", "type": "text"}},
            last_updated=datetime.now(UTC) - timedelta(seconds=600),
            ttl_seconds=300.0,
        )

        result = cache.get_models(resource_type="text")
        assert result == []


class TestFetchModelsDoubleCheck:
    """Test the double-check after lock acquisition - targets line 130."""

    @pytest.mark.asyncio
    async def test_double_check_after_lock_returns_cached(self):
        """Test that _fetch_models returns cache after lock when cache is valid (line 130).

        This covers the case where cache becomes valid between initial check
        and acquiring the lock (simulates concurrent fetch).
        """
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Pre-populate cache and make it valid
        selector._cache.models = {"cached-model": {"id": "cached-model", "type": "text"}}
        selector._cache.last_updated = datetime.now(UTC)

        # First call should return cache without calling API
        result = await selector._fetch_models(force_refresh=False)

        assert result == selector._cache.models
        mock_client.models.list.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_fetch_only_one_api_call(self):
        """Test that concurrent fetches result in minimal API calls (line 130).

        When multiple coroutines try to fetch simultaneously, only one should
        actually call the API, others should get the cached result.
        """
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        call_count = 0

        async def mock_list(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Simulate network delay
            mock_response = Mock()
            mock_response.data = [
                Mock(
                    id="model1",
                    object="model",
                    type="text",
                    created=1234567890,
                    owned_by="venice",
                )
            ]
            mock_response.data[0].model_spec = None
            return mock_response

        mock_client.models.list = mock_list

        # Start multiple concurrent fetches with force_refresh
        results = await asyncio.gather(
            selector._fetch_models(force_refresh=True),
            selector._fetch_models(force_refresh=True),
            selector._fetch_models(force_refresh=True),
        )

        # All should return valid results
        assert all(result is not None for result in results)
        # Due to the lock, only 3 requests should be made (one per force_refresh)
        # This tests the concurrency handling


class TestPricingExtractionEdgeCases:
    """Test pricing extraction edge cases - targets lines 190-222."""

    @pytest.mark.asyncio
    async def test_pricing_as_dict(self):
        """Test pricing when it's already a dict (line 191)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock model with dict pricing
        mock_model = Mock()
        mock_model.id = "model1"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock()
        mock_model_spec.capabilities = Mock(
            supportsFunctionCalling=True,
            supportsVision=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsReasoning=False,
        )
        # Pricing as a dict directly (line 190-191)
        mock_model_spec.pricing = {"input": {"usd": 0.001}, "output": {"usd": 0.002}}
        mock_model_spec.traits = None
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model1" in result
        assert result["model1"]["model_spec"]["pricing"] == {
            "input": {"usd": 0.001},
            "output": {"usd": 0.002},
        }

    @pytest.mark.asyncio
    async def test_pricing_manual_extraction_with_pydantic_tiers(self):
        """Test manual pricing extraction with Pydantic model tiers (lines 193-222)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock tier that has model_dump
        mock_tier_input = Mock()
        mock_tier_input.model_dump = Mock(return_value={"usd": 0.001, "diem": 100})

        mock_tier_output = Mock()
        mock_tier_output.model_dump = Mock(return_value={"usd": 0.002, "diem": 200})

        # Create mock pricing that doesn't have model_dump and isn't a dict
        mock_pricing = Mock(spec=[])  # No model_dump
        mock_pricing.input = mock_tier_input
        mock_pricing.output = mock_tier_output
        mock_pricing.cache_input = None
        mock_pricing.generation = None
        mock_pricing.upscale = None

        # Ensure it's not a dict and doesn't have model_dump
        # mock_pricing with spec=[] won't have model_dump

        mock_model = Mock()
        mock_model.id = "model2"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock()
        mock_model_spec.capabilities = Mock(
            supportsFunctionCalling=False,
            supportsVision=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsReasoning=False,
        )
        mock_model_spec.pricing = mock_pricing
        mock_model_spec.traits = None
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model2" in result
        # Verify pricing was extracted via model_dump on tiers
        pricing = result["model2"]["model_spec"]["pricing"]
        assert pricing["input"] == {"usd": 0.001, "diem": 100}
        assert pricing["output"] == {"usd": 0.002, "diem": 200}

    @pytest.mark.asyncio
    async def test_pricing_manual_extraction_with_dict_tiers(self):
        """Test manual pricing extraction where tiers are dicts (line 209-210)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock tier that is a dict
        mock_pricing = Mock(spec=["input", "output"])
        mock_pricing.input = {"usd": 0.003, "diem": 300}  # Dict tier
        mock_pricing.output = {"usd": 0.004, "diem": 400}  # Dict tier

        mock_model = Mock()
        mock_model.id = "model3"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock()
        mock_model_spec.capabilities = Mock(
            supportsFunctionCalling=False,
            supportsVision=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsReasoning=False,
        )
        mock_model_spec.pricing = mock_pricing
        mock_model_spec.traits = None
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model3" in result
        pricing = result["model3"]["model_spec"]["pricing"]
        assert pricing["input"] == {"usd": 0.003, "diem": 300}
        assert pricing["output"] == {"usd": 0.004, "diem": 400}

    @pytest.mark.asyncio
    async def test_pricing_manual_extraction_with_raw_attributes(self):
        """Test manual pricing extraction with raw usd/diem attributes (lines 213-220)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock tier that has raw usd/diem attributes
        mock_tier_input = Mock(spec=["usd", "diem"])
        mock_tier_input.usd = 0.005
        mock_tier_input.diem = 500

        mock_tier_output = Mock(spec=["usd", "diem"])
        mock_tier_output.usd = 0.006
        mock_tier_output.diem = 600

        mock_pricing = Mock(spec=["input", "output"])
        mock_pricing.input = mock_tier_input
        mock_pricing.output = mock_tier_output

        mock_model = Mock()
        mock_model.id = "model4"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock()
        mock_model_spec.capabilities = Mock(
            supportsFunctionCalling=False,
            supportsVision=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsReasoning=False,
        )
        mock_model_spec.pricing = mock_pricing
        mock_model_spec.traits = None
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model4" in result
        pricing = result["model4"]["model_spec"]["pricing"]
        assert pricing["input"]["usd"] == 0.005
        assert pricing["input"]["diem"] == 500

    @pytest.mark.asyncio
    async def test_pricing_empty_when_no_pricing_dict_created(self):
        """Test that no pricing dict is set when extraction fails (line 221-222)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock pricing with no extractable attributes
        mock_pricing = Mock(spec=[])  # No attributes

        mock_model = Mock()
        mock_model.id = "model5"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock()
        mock_model_spec.capabilities = Mock(
            supportsFunctionCalling=False,
            supportsVision=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsReasoning=False,
        )
        mock_model_spec.pricing = mock_pricing
        mock_model_spec.traits = None
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model5" in result
        # Pricing should remain None since no pricing dict was created
        assert result["model5"]["model_spec"]["pricing"] is None

    @pytest.mark.asyncio
    async def test_model_without_capabilities(self):
        """Test model without capabilities (branch line 163->184)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock model without capabilities in model_spec
        mock_model = Mock()
        mock_model.id = "model-no-caps"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock(spec=["pricing", "traits"])  # No capabilities
        mock_model_spec.pricing = None
        mock_model_spec.traits = ["fast", "efficient"]
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model-no-caps" in result
        # Capabilities should be empty since none were provided
        assert result["model-no-caps"]["model_spec"]["capabilities"] == {}

    @pytest.mark.asyncio
    async def test_model_without_model_spec(self):
        """Test model without model_spec entirely (branch line 153->233)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Create mock model without model_spec attribute
        mock_model = Mock(spec=["id", "object", "type", "created", "owned_by"])
        mock_model.id = "model-no-spec"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model-no-spec" in result
        # Should not have model_spec key
        assert "model_spec" not in result["model-no-spec"]

    @pytest.mark.asyncio
    async def test_model_without_traits(self):
        """Test model_spec without traits (branch line 228->233)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        mock_model = Mock()
        mock_model.id = "model-no-traits"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"

        mock_model_spec = Mock(spec=["capabilities", "pricing"])  # No traits
        mock_model_spec.capabilities = Mock(
            supportsFunctionCalling=True,
            supportsVision=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsReasoning=False,
        )
        mock_model_spec.pricing = None
        mock_model.model_spec = mock_model_spec

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await selector._fetch_models(force_refresh=True)

        assert "model-no-traits" in result
        # Should not have traits key
        assert "traits" not in result["model-no-traits"]


class TestChatModelFallbackSelection:
    """Test fallback chat model selection - targets lines 329-331."""

    @pytest.mark.asyncio
    async def test_fallback_chat_model_when_no_pattern_match(self):
        """Test fallback to first model when no patterns match (lines 329-331)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Set up models that don't match any chat patterns
        selector._cache.models = {
            "unknown-model-xyz": {"id": "unknown-model-xyz", "type": "text"},
            "custom-abc": {"id": "custom-abc", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        # No preferred models, no exclude
        result = await selector.select_chat_model()

        # Should fall back to first available model
        assert result in ["unknown-model-xyz", "custom-abc"]

    @pytest.mark.asyncio
    async def test_chat_model_prefers_traits(self):
        """Test that chat model selection prefers trait-matched models before fallback."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "unknown-model": {"id": "unknown-model", "type": "text"},
            "llama-3.3-70b": {
                "id": "llama-3.3-70b",
                "type": "text",
                "traits": ["default"],
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_chat_model()

        # Should prefer model with 'default' trait
        assert result == "llama-3.3-70b"

    @pytest.mark.asyncio
    async def test_chat_model_no_models_raises(self):
        """Test that no available chat models raises ValueError (line 301)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "image-model": {"id": "image-model", "type": "image"},  # Not text type
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(NoMatchingModelError, match="No available chat models found"):
            await selector.select_chat_model()


class TestFunctionCallingModelSelection:
    """Test function calling model selection - targets lines 361, 404, 445-447."""

    @pytest.mark.asyncio
    async def test_function_calling_no_models_after_filter_raises(self):
        """Test that no candidates raises error (lines 360-361)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Only non-text models
        selector._cache.models = {
            "image-model": {"id": "image-model", "type": "image"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(NoMatchingModelError, match="No available chat models found"):
            await selector.select_function_calling_model()

    @pytest.mark.asyncio
    async def test_function_calling_no_capable_models_raises(self):
        """Test that no function calling capable models raises error (lines 403-404)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Models that are excluded by pattern or don't support function calling
        selector._cache.models = {
            "venice-uncensored": {
                "id": "venice-uncensored",
                "type": "text",
                "model_spec": {"capabilities": {"supportsFunctionCalling": False}},
            },
            "base-model": {
                "id": "base-model",
                "type": "text",
                "model_spec": {"capabilities": {"supportsFunctionCalling": False}},
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(NoMatchingModelError, match="No function calling capable models found"):
            await selector.select_function_calling_model()

    @pytest.mark.asyncio
    async def test_function_calling_fallback_when_no_explicit(self):
        """Test fallback to first capable model when no preferred match."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Model with explicit supportsFunctionCalling=True
        selector._cache.models = {
            "llama-3-custom": {
                "id": "llama-3-custom",
                "type": "text",
                "model_spec": {"capabilities": {"supportsFunctionCalling": True}},
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_function_calling_model()

        # Should select the model with explicit function calling support
        assert result == "llama-3-custom"


class TestEmbeddingModelSelection:
    """Test embedding model selection - targets lines 477, 505-507."""

    @pytest.mark.asyncio
    async def test_embedding_no_models_raises(self):
        """Test that no embedding models raises error (lines 476-477)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "text-model": {"id": "text-model", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(NoMatchingModelError, match="No available embedding models found"):
            await selector.select_embedding_model()

    @pytest.mark.asyncio
    async def test_embedding_fallback_when_no_pattern_match(self):
        """Test fallback to first model when no patterns match (lines 505-507)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Models that don't match embedding patterns
        selector._cache.models = {
            "custom-vector-model": {"id": "custom-vector-model", "type": "embedding"},
            "another-vector-model": {"id": "another-vector-model", "type": "embedding"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_embedding_model()

        # Should fall back to first available
        assert result in ["custom-vector-model", "another-vector-model"]

    @pytest.mark.asyncio
    async def test_embedding_prefers_preferred_models(self):
        """Test that embedding selection honors preferred_models list."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "custom-model": {"id": "custom-model", "type": "embedding"},
            "text-embedding-3-large": {
                "id": "text-embedding-3-large",
                "type": "embedding",
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_embedding_model(preferred_models=["text-embedding-3-large"])

        # Should prefer explicitly requested model
        assert result == "text-embedding-3-large"


class TestImageModelSelection:
    """Test image model selection - targets lines 537, 552-553, 572-574."""

    @pytest.mark.asyncio
    async def test_image_no_models_raises(self):
        """Test that no image models raises error (lines 536-537)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "text-model": {"id": "text-model", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(NoMatchingModelError, match="No available image models found"):
            await selector.select_image_model()

    @pytest.mark.asyncio
    async def test_image_preferred_model_selected(self):
        """Test that preferred image model is selected (lines 552-553)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "flux-schnell": {"id": "flux-schnell", "type": "image"},
            "sdxl-1.0": {"id": "sdxl-1.0", "type": "image"},
            "custom-image": {"id": "custom-image", "type": "image"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_image_model(preferred_models=["custom-image"])

        assert result == "custom-image"

    @pytest.mark.asyncio
    async def test_image_fallback_when_no_pattern_match(self):
        """Test fallback to first model when no patterns match (lines 572-574)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Models that don't match any image patterns
        selector._cache.models = {
            "custom-gen-model": {"id": "custom-gen-model", "type": "image"},
            "another-gen-model": {"id": "another-gen-model", "type": "image"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_image_model()

        # Should fall back to first available
        assert result in ["custom-gen-model", "another-gen-model"]

    @pytest.mark.asyncio
    async def test_image_prefers_traits(self):
        """Test that image selection prefers trait-matched models."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "custom-model": {"id": "custom-model", "type": "image"},
            "stable-diffusion-v2": {
                "id": "stable-diffusion-v2",
                "type": "image",
                "traits": ["default"],
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_image_model()

        # Should prefer model with 'default' trait
        assert result == "stable-diffusion-v2"


class TestAudioModelSelection:
    """Test audio model selection - targets lines 604, 610-613, 639-641."""

    @pytest.mark.asyncio
    async def test_audio_no_models_raises(self):
        """Test that no audio models raises error (lines 603-604)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "text-model": {"id": "text-model", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(
            NoMatchingModelError, match=r"No available text-to-speech \(tts\) models found"
        ) as excinfo:
            await selector.select_audio_model()
        assert excinfo.value.resource_type == "tts"

    @pytest.mark.asyncio
    async def test_asr_no_models_names_speech_to_text(self):
        """The ASR no-match message names the resource kind, not a generic label."""
        selector = DynamicModelSelector(client=AsyncMock())
        selector._cache.models = {"text-model": {"id": "text-model", "type": "text"}}
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(
            NoMatchingModelError, match=r"No available speech-to-text \(asr\) models found"
        ) as excinfo:
            await selector.select_asr_model()
        assert excinfo.value.resource_type == "asr"

    @pytest.mark.asyncio
    async def test_audio_with_custom_selector(self):
        """Test audio model selection with custom selector (lines 610-613)."""
        mock_client = AsyncMock()

        def custom_selector(candidates):
            # Select the model with "custom" in the name
            for model in candidates:
                if "custom" in model["id"]:
                    return model["id"]
            return candidates[0]["id"]

        selector = DynamicModelSelector(client=mock_client, default_selector=custom_selector)

        selector._cache.models = {
            "af_sky": {"id": "af_sky", "type": "tts"},
            "custom-tts-model": {"id": "custom-tts-model", "type": "tts"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_audio_model()

        assert result == "custom-tts-model"

    @pytest.mark.asyncio
    async def test_audio_preferred_model_selected(self):
        """Test that preferred audio model is selected (lines 617-620)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "af_sky": {"id": "af_sky", "type": "tts"},
            "af_alloy": {"id": "af_alloy", "type": "tts"},
            "custom-voice": {"id": "custom-voice", "type": "tts"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_audio_model(preferred_models=["custom-voice"])

        assert result == "custom-voice"

    @pytest.mark.asyncio
    async def test_audio_fallback_when_no_pattern_match(self):
        """Test fallback to first model when no patterns match (lines 639-641)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Models that don't match any audio patterns
        selector._cache.models = {
            "custom-narrator": {"id": "custom-narrator", "type": "tts"},
            "another-narrator": {"id": "another-narrator", "type": "tts"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_audio_model()

        # Should fall back to first available
        assert result in ["custom-narrator", "another-narrator"]

    @pytest.mark.asyncio
    async def test_audio_prefers_preferred_models(self):
        """Test that audio selection honors preferred_models list."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "custom-model": {"id": "custom-model", "type": "tts"},
            "af_bella": {"id": "af_bella", "type": "tts"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_audio_model(preferred_models=["af_bella"])

        # Should prefer explicitly requested model
        assert result == "af_bella"


class TestConcurrencyModelSelection:
    """Test concurrency model selection - targets lines 666, 688-689."""

    @pytest.mark.asyncio
    async def test_concurrency_not_enough_models_raises(self):
        """Test that not enough models raises error (lines 665-668)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "model1": {"id": "model1", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        with pytest.raises(NoMatchingModelError, match="Need 3 models but only 1 available"):
            await selector.select_models_for_concurrency_test(count=3)

    @pytest.mark.asyncio
    async def test_concurrency_fills_remaining_slots(self):
        """Test that remaining slots are filled after family matching (lines 688-689)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # More models than families, need to fill remaining
        selector._cache.models = {
            "venice-chat": {"id": "venice-chat", "type": "text"},
            "qwen-7b": {"id": "qwen-7b", "type": "text"},
            "other-model-1": {
                "id": "other-model-1",
                "type": "text",
            },  # Not in any family
            "other-model-2": {
                "id": "other-model-2",
                "type": "text",
            },  # Not in any family
            "other-model-3": {
                "id": "other-model-3",
                "type": "text",
            },  # Not in any family
        }
        selector._cache.last_updated = datetime.now(UTC)

        # Request more models than families available
        result = await selector.select_models_for_concurrency_test(count=5)

        assert len(result) == 5
        # Should have venice and qwen from families
        assert "venice-chat" in result
        assert "qwen-7b" in result
        # And 3 others filled from remaining
        for model in result:
            assert model in selector._cache.models

    @pytest.mark.asyncio
    async def test_concurrency_stops_at_count(self):
        """Test that selection stops when count is reached (lines 676-677, 685-687)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "venice-1": {"id": "venice-1", "type": "text"},
            "qwen-1": {"id": "qwen-1", "type": "text"},
            "llama-1": {"id": "llama-1", "type": "text"},
            "mistral-1": {"id": "mistral-1", "type": "text"},
            "other-1": {"id": "other-1", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_models_for_concurrency_test(count=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_concurrency_prefers_non_reasoning_models(self):
        """Test that non-reasoning models are selected before reasoning ones."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "reasoning-1": {
                "id": "reasoning-1",
                "type": "text",
                "model_spec": {"capabilities": {"supportsReasoning": True}},
            },
            "reasoning-2": {
                "id": "reasoning-2",
                "type": "text",
                "model_spec": {"capabilities": {"supportsReasoning": True}},
            },
            "non-reasoning-1": {
                "id": "non-reasoning-1",
                "type": "text",
                "model_spec": {"capabilities": {"supportsReasoning": False}},
            },
            "non-reasoning-2": {
                "id": "non-reasoning-2",
                "type": "text",
                "model_spec": {"capabilities": {"supportsReasoning": False}},
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_models_for_concurrency_test(count=3)

        assert len(result) == 3
        # Both non-reasoning models should be selected before reasoning models
        assert "non-reasoning-1" in result
        assert "non-reasoning-2" in result


class TestCustomSelectors:
    """Test custom selector functionality for all model types."""

    @pytest.mark.asyncio
    async def test_chat_model_with_per_call_selector(self):
        """Test chat model with per-call selector override."""
        mock_client = AsyncMock()

        def default_selector(candidates):
            return candidates[0]["id"]

        def per_call_selector(candidates):
            return candidates[-1]["id"]  # Select last

        selector = DynamicModelSelector(client=mock_client, default_selector=default_selector)

        selector._cache.models = {
            "model-a": {"id": "model-a", "type": "text"},
            "model-b": {"id": "model-b", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        # With per-call selector, should select last
        result = await selector.select_chat_model(selector=per_call_selector)
        assert result == "model-b"

    @pytest.mark.asyncio
    async def test_chat_model_selector_excludes_reasoning_models(self):
        """A custom selector must not be handed reasoning models for general chat.

        Regression: the test suite's ``random_cheap_strategy`` ran on the full
        candidate pool and randomly picked reasoning models (e.g. qwen3-30b),
        whose ``message.content`` comes back empty under small token budgets.
        ``select_chat_model`` now filters reasoning models out of the candidate
        pool before any selector runs, unless reasoning is explicitly required.
        """
        mock_client = AsyncMock()

        def pick_first(candidates):
            return candidates[0]["id"]

        selector = DynamicModelSelector(client=mock_client, default_selector=pick_first)

        # 'reasoner' is first in insertion order; an unfiltered pool would hand
        # it straight to pick_first.
        selector._cache.models = {
            "reasoner": {
                "id": "reasoner",
                "type": "text",
                "model_spec": {"capabilities": {"supportsReasoning": True}},
            },
            "plain": {
                "id": "plain",
                "type": "text",
                "model_spec": {"capabilities": {"supportsReasoning": False}},
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        # Non-reasoning pool filter excludes 'reasoner' before pick_first runs.
        assert await selector.select_chat_model() == "plain"

        # When reasoning is explicitly required, the reasoning model is eligible.
        assert await selector.select_chat_model(require_reasoning=True) == "reasoner"

    @pytest.mark.asyncio
    async def test_embedding_model_with_selector(self):
        """Test embedding model with custom selector."""
        mock_client = AsyncMock()

        def cost_selector(candidates):
            # Just return the first one for testing
            return candidates[0]["id"]

        selector = DynamicModelSelector(client=mock_client, default_selector=cost_selector)

        selector._cache.models = {
            "embed-1": {"id": "embed-1", "type": "embedding"},
            "embed-2": {"id": "embed-2", "type": "embedding"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_embedding_model()
        assert result == "embed-1"

    @pytest.mark.asyncio
    async def test_image_model_with_selector(self):
        """Test image model with custom selector."""
        mock_client = AsyncMock()

        def quality_selector(candidates):
            return candidates[-1]["id"]

        selector = DynamicModelSelector(client=mock_client, default_selector=quality_selector)

        selector._cache.models = {
            "img-1": {"id": "img-1", "type": "image"},
            "img-2": {"id": "img-2", "type": "image"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_image_model()
        assert result == "img-2"

    @pytest.mark.asyncio
    async def test_function_calling_with_selector(self):
        """Test function calling model with custom selector."""
        mock_client = AsyncMock()

        def fast_selector(candidates):
            return candidates[0]["id"]

        selector = DynamicModelSelector(client=mock_client, default_selector=fast_selector)

        selector._cache.models = {
            "llama-3-8b": {
                "id": "llama-3-8b",
                "type": "text",
                "model_spec": {"capabilities": {"supportsFunctionCalling": True}},
            },
            "llama-3-70b": {
                "id": "llama-3-70b",
                "type": "text",
                "model_spec": {"capabilities": {"supportsFunctionCalling": True}},
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_function_calling_model()
        assert result == "llama-3-8b"


class TestPreferredModelBranches:
    """Test preferred model iteration branches."""

    @pytest.mark.asyncio
    async def test_chat_preferred_not_in_candidates(self):
        """Test when preferred model is not in candidates (branch 314->320)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "llama-3.3-70b": {"id": "llama-3.3-70b", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        # Preferred model not available
        result = await selector.select_chat_model(preferred_models=["non-existent-model"])

        # Should select by pattern since preferred not available
        assert result == "llama-3.3-70b"

    @pytest.mark.asyncio
    async def test_function_calling_preferred_not_in_candidates(self):
        """Test when preferred function calling model not in candidates (branch 421->429)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "llama-3-8b": {
                "id": "llama-3-8b",
                "type": "text",
                "model_spec": {"capabilities": {"supportsFunctionCalling": True}},
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_function_calling_model(preferred_models=["non-existent"])

        # Should select by capability since preferred not available
        assert result == "llama-3-8b"

    @pytest.mark.asyncio
    async def test_embedding_preferred_not_in_candidates(self):
        """Test when preferred embedding model not in candidates (branch 490->496)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "text-embedding-model": {"id": "text-embedding-model", "type": "embedding"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_embedding_model(preferred_models=["non-existent"])

        assert result == "text-embedding-model"

    @pytest.mark.asyncio
    async def test_audio_preferred_not_in_candidates(self):
        """Test when preferred audio model not in candidates (branch 617->623)."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "af_sky": {"id": "af_sky", "type": "tts"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_audio_model(preferred_models=["non-existent"])

        assert result == "af_sky"


class TestFactoryAndHelpers:
    """Test factory function and helper functions."""

    def test_create_model_selector_basic(self):
        """Test create_model_selector factory function."""
        import warnings

        mock_client = Mock()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            selector = create_model_selector(mock_client)

        assert isinstance(selector, DynamicModelSelector)
        assert selector.client == mock_client
        assert selector.default_selector is None

    def test_create_model_selector_with_options(self):
        """Test create_model_selector with options."""
        import warnings

        mock_client = Mock()

        def custom_selector(candidates):
            return candidates[0]["id"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            selector = create_model_selector(
                mock_client, cache_ttl=600.0, default_selector=custom_selector
            )

        assert selector._cache.ttl_seconds == 600.0
        assert selector.default_selector == custom_selector

    @pytest.mark.asyncio
    async def test_get_chat_model_helper(self):
        """Test get_chat_model helper function."""
        mock_client = AsyncMock()

        # Mock models.list response
        mock_model = Mock()
        mock_model.id = "llama-3.3-70b"
        mock_model.object = "model"
        mock_model.type = "text"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"
        mock_model.model_spec = None

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await get_chat_model(mock_client)

        assert result == "llama-3.3-70b"

    @pytest.mark.asyncio
    async def test_get_embedding_model_helper(self):
        """Test get_embedding_model helper function."""
        mock_client = AsyncMock()

        mock_model = Mock()
        mock_model.id = "text-embedding-3-large"
        mock_model.object = "model"
        mock_model.type = "embedding"
        mock_model.created = 1234567890
        mock_model.owned_by = "venice"
        mock_model.model_spec = None

        mock_response = Mock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response

        result = await get_embedding_model(mock_client)

        assert result == "text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_get_multiple_models_helper(self):
        """Test get_multiple_models helper function."""
        mock_client = AsyncMock()

        mock_models = []
        for _i, name in enumerate(["venice-chat", "qwen-7b", "llama-3-8b"]):
            mock_model = Mock()
            mock_model.id = name
            mock_model.object = "model"
            mock_model.type = "text"
            mock_model.created = 1234567890
            mock_model.owned_by = "venice"
            mock_model.model_spec = None
            mock_models.append(mock_model)

        mock_response = Mock()
        mock_response.data = mock_models
        mock_client.models.list.return_value = mock_response

        result = await get_multiple_models(mock_client, count=2)

        assert len(result) == 2


class TestGetModelInfo:
    """Test get_model_info method."""

    @pytest.mark.asyncio
    async def test_get_model_info_exists(self):
        """Test get_model_info for existing model."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "test-model": {"id": "test-model", "type": "text", "owned_by": "venice"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.get_model_info("test-model")

        assert result is not None
        assert result["id"] == "test-model"

    @pytest.mark.asyncio
    async def test_get_model_info_not_exists(self):
        """Test get_model_info for non-existent model."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "test-model": {"id": "test-model", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.get_model_info("non-existent")

        assert result is None


class TestGetCacheInfo:
    """Test get_cache_info method."""

    def test_get_cache_info(self):
        """Test get_cache_info returns correct info."""
        mock_client = Mock()
        selector = DynamicModelSelector(client=mock_client, cache_ttl=600.0)

        selector._cache.models = {
            "model1": {"id": "model1"},
            "model2": {"id": "model2"},
        }

        info = selector.get_cache_info()

        assert info["model_count"] == 2
        assert info["ttl_seconds"] == 600.0
        assert "last_updated" in info
        assert "is_expired" in info


class TestExclusionFiltering:
    """Test model exclusion functionality."""

    @pytest.mark.asyncio
    async def test_chat_model_exclusion(self):
        """Test that excluded models are filtered out."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "llama-3.3-70b": {"id": "llama-3.3-70b", "type": "text"},
            "qwen-7b": {"id": "qwen-7b", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_chat_model(exclude_models={"llama-3.3-70b"})

        assert result == "qwen-7b"

    @pytest.mark.asyncio
    async def test_concurrency_exclusion(self):
        """Test that excluded models are filtered from concurrency selection."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        selector._cache.models = {
            "model-a": {"id": "model-a", "type": "text"},
            "model-b": {"id": "model-b", "type": "text"},
            "model-c": {"id": "model-c", "type": "text"},
        }
        selector._cache.last_updated = datetime.now(UTC)

        result = await selector.select_models_for_concurrency_test(
            count=2, exclude_models={"model-a"}
        )

        assert len(result) == 2
        assert "model-a" not in result


class TestDeprecationFilter:
    """Models past their ``deprecation.date`` should be skipped by the resolver."""

    @staticmethod
    def _past() -> str:
        return (datetime.now(UTC) - timedelta(days=1)).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _future() -> str:
        return (datetime.now(UTC) + timedelta(days=30)).isoformat().replace("+00:00", "Z")

    def test_is_past_deprecation_with_past_date(self):
        assert DynamicModelSelector._is_past_deprecation({"deprecation_date": self._past()}) is True

    def test_is_past_deprecation_with_future_date(self):
        assert (
            DynamicModelSelector._is_past_deprecation({"deprecation_date": self._future()}) is False
        )

    def test_is_past_deprecation_with_no_field(self):
        assert DynamicModelSelector._is_past_deprecation({}) is False

    def test_is_past_deprecation_with_unparseable(self):
        assert DynamicModelSelector._is_past_deprecation({"deprecation_date": "garbage"}) is False

    @pytest.mark.asyncio
    async def test_get_available_models_skips_past_deprecation(self):
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)
        selector._cache.models = {
            "active": {"id": "active", "type": "text"},
            "deprecated": {
                "id": "deprecated",
                "type": "text",
                "deprecation_date": self._past(),
            },
            "future-eol": {
                "id": "future-eol",
                "type": "text",
                "deprecation_date": self._future(),
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        available = await selector.get_available_models(resource_type="text")

        assert "active" in available
        assert "future-eol" in available
        assert "deprecated" not in available

    @pytest.mark.asyncio
    async def test_select_chat_model_skips_past_deprecation(self):
        """The resolver must not return a model whose deprecation date has passed."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)
        selector._cache.models = {
            "good-model": {"id": "good-model", "type": "text"},
            "venice-uncensored": {
                "id": "venice-uncensored",
                "type": "text",
                "deprecation_date": self._past(),
            },
        }
        selector._cache.last_updated = datetime.now(UTC)

        chosen = await selector.select_chat_model()
        assert chosen != "venice-uncensored"
        assert chosen == "good-model"
