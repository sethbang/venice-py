"""
VCRpy-based integration tests for Model Selection functionality.

This module tests the DynamicModelSelector class and related utilities
through real API interactions recorded with VCRpy, expanding coverage
for cache management, model selection logic, error handling, and edge cases.
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.models.selection import (
    DynamicModelSelector,
    ModelCache,
    create_model_selector,
    get_chat_model,
    get_embedding_model,
    get_multiple_models,
)


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    # Use INTELLIGENT mode with MemoryBackend for VCR tests
    # This provides rate limit protection (prevents 429s) without Redis connection leaks
    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=False,  # Use MemoryBackend instead of Redis
    )
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is now provided by the root conftest.py


# ============================================================================
# Cache Functionality Tests
# ============================================================================


@pytest.mark.integration
async def test_model_cache_expiration(venice_client, vcr_cassette):
    """Test that cache expires correctly after TTL."""
    with vcr_cassette:
        # Create selector with very short TTL
        selector = DynamicModelSelector(venice_client, cache_ttl=0.1)

        # First fetch should hit API
        models1 = await selector.get_available_models()
        assert len(models1) > 0

        # Immediate re-fetch should use cache
        cache_info = selector.get_cache_info()
        assert not cache_info["is_expired"]

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Cache should now be expired
        cache_info = selector.get_cache_info()
        assert cache_info["is_expired"]


@pytest.mark.integration
async def test_model_cache_force_refresh(venice_client, vcr_cassette):
    """Test force refresh bypasses cache."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Initial fetch
        models1 = await selector.get_available_models()
        assert len(models1) > 0

        # Force refresh should fetch again
        models2 = await selector.get_available_models(force_refresh=True)
        assert len(models2) > 0


@pytest.mark.integration
async def test_model_cache_update(venice_client, vcr_cassette):
    """Test cache update functionality."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Fetch models to populate cache
        models = await selector.get_available_models()
        assert len(models) > 0

        # Check cache info
        cache_info = selector.get_cache_info()
        assert cache_info["model_count"] > 0
        assert cache_info["last_updated"] is not None
        assert cache_info["ttl_seconds"] == 300.0


@pytest.mark.integration
async def test_model_cache_empty_expiration():
    """Test that empty cache is always expired."""
    cache = ModelCache()
    assert cache.is_expired()  # Empty cache should be expired

    # Add models
    cache.update({"model1": {"id": "model1", "type": "text"}})
    assert not cache.is_expired()  # Non-empty, fresh cache


# ============================================================================
# Model Selection Tests
# ============================================================================


@pytest.mark.integration
async def test_select_chat_model_basic(venice_client, vcr_cassette):
    """Test basic chat model selection."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)
        model = await selector.select_chat_model()

        assert model is not None
        assert isinstance(model, str)
        assert len(model) > 0


@pytest.mark.integration
async def test_select_chat_model_with_preferences(venice_client, vcr_cassette):
    """Test chat model selection with preferred models."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Try with a preferred model that exists
        model = await selector.select_chat_model(preferred_models=["llama-3.3-70b", "qwen-2.5-72b"])

        assert model is not None
        # Should match one of the preferred models if available
        assert isinstance(model, str)


@pytest.mark.integration
async def test_select_chat_model_with_exclusions(venice_client, vcr_cassette):
    """Test chat model selection with exclusions."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Get all models first
        all_models = await selector.get_available_models(resource_type="text")

        # Exclude all but one
        excluded = set(all_models[1:]) if len(all_models) > 1 else set()

        model = await selector.select_chat_model(exclude_models=excluded)
        assert model is not None
        assert model not in excluded


@pytest.mark.integration
async def test_select_function_calling_model(venice_client, vcr_cassette):
    """Test function calling model selection."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        model = await selector.select_function_calling_model()

        assert model is not None
        assert isinstance(model, str)
        # Should not include "uncensored" or other non-function-calling patterns
        assert "uncensored" not in model.lower()


@pytest.mark.integration
async def test_select_chat_model_require_function_calling(venice_client, vcr_cassette):
    """Test chat model selection with function calling requirement."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Function calling is applied as a filter on the shared chat pool
        model = await selector.select_chat_model(require_function_calling=True)

        assert model is not None
        assert isinstance(model, str)


@pytest.mark.integration
async def test_select_function_calling_with_preferences(venice_client, vcr_cassette):
    """Test function calling model selection with preferences."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        model = await selector.select_function_calling_model(preferred_models=["llama-3.3-70b"])

        assert model is not None


@pytest.mark.integration
async def test_select_embedding_model(venice_client, vcr_cassette):
    """Test embedding model selection."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        model = await selector.select_embedding_model()

        assert model is not None
        assert isinstance(model, str)


@pytest.mark.integration
async def test_select_embedding_model_with_preferences(venice_client, vcr_cassette):
    """Test embedding model selection with preferences."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        model = await selector.select_embedding_model(preferred_models=["text-embedding-bge-m3"])

        assert model is not None


@pytest.mark.integration
async def test_select_image_model(venice_client, vcr_cassette):
    """Test image model selection."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        try:
            model = await selector.select_image_model()
            assert model is not None
            assert isinstance(model, str)
        except ValueError as e:
            # It's okay if no image models are available
            assert "No available image models" in str(e)


@pytest.mark.integration
async def test_select_image_model_with_preferences(venice_client, vcr_cassette):
    """Test image model selection with preferences."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        try:
            model = await selector.select_image_model(
                preferred_models=["flux-schnell", "stable-diffusion"]
            )
            assert model is not None
        except ValueError as e:
            # It's okay if no image models are available
            assert "No available image models" in str(e)


@pytest.mark.integration
async def test_select_audio_model(venice_client, vcr_cassette):
    """Test audio/TTS model selection."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        try:
            model = await selector.select_audio_model()
            assert model is not None
            assert isinstance(model, str)
        except ValueError as e:
            # It's okay if no audio models are available
            assert "No available text-to-speech (tts) models" in str(e)


@pytest.mark.integration
async def test_select_audio_model_with_preferences(venice_client, vcr_cassette):
    """Test audio model selection with preferences."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        try:
            model = await selector.select_audio_model(preferred_models=["tts-kokoro", "af_sky"])
            assert model is not None
        except ValueError as e:
            assert "No available text-to-speech (tts) models" in str(e)


@pytest.mark.integration
async def test_select_models_for_concurrency(venice_client, vcr_cassette):
    """Test selecting multiple models for concurrency testing."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        models = await selector.select_models_for_concurrency_test(count=2)

        assert len(models) == 2
        assert models[0] != models[1]  # Should be different models


@pytest.mark.integration
async def test_select_models_for_concurrency_with_exclusions(venice_client, vcr_cassette):
    """Test selecting multiple models with exclusions."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Get all models
        all_models = await selector.get_available_models()

        # Exclude some models
        excluded = set(all_models[:2]) if len(all_models) > 3 else set()

        models = await selector.select_models_for_concurrency_test(count=2, exclude_models=excluded)

        assert len(models) == 2
        for model in models:
            assert model not in excluded


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


@pytest.mark.integration
async def test_get_model_info(venice_client, vcr_cassette):
    """Test getting info for a specific model."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Get available models first
        models = await selector.get_available_models()
        if models:
            model_id = models[0]

            # Get model info
            info = await selector.get_model_info(model_id)

            assert info is not None
            assert "id" in info
            assert info["id"] == model_id


@pytest.mark.integration
async def test_get_model_info_nonexistent(venice_client, vcr_cassette):
    """Test getting info for a non-existent model."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Populate cache first
        await selector.get_available_models()

        # Try to get info for non-existent model
        info = await selector.get_model_info("nonexistent-model-xyz-123")

        assert info is None


@pytest.mark.integration
async def test_get_available_models_by_type(venice_client, vcr_cassette):
    """Test getting models filtered by resource type."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Get text models
        text_models = await selector.get_available_models(resource_type="text")
        assert len(text_models) > 0

        # Get embedding models
        embedding_models = await selector.get_available_models(resource_type="embedding")
        # May or may not have embedding models depending on API tier
        # Just verify the call succeeds
        assert isinstance(embedding_models, list)


# ============================================================================
# Utility Functions Tests
# ============================================================================


@pytest.mark.integration
async def test_create_model_selector_factory(venice_client, vcr_cassette):
    """Test the create_model_selector factory function."""
    import warnings

    with vcr_cassette, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        selector = create_model_selector(venice_client, cache_ttl=600.0)

        assert isinstance(selector, DynamicModelSelector)
        assert selector.client == venice_client

        # Verify it works
        models = await selector.get_available_models()
        assert len(models) > 0


@pytest.mark.integration
async def test_get_chat_model_utility(venice_client, vcr_cassette):
    """Test the get_chat_model utility function."""
    with vcr_cassette:
        model = await get_chat_model(venice_client)

        assert model is not None
        assert isinstance(model, str)


@pytest.mark.integration
async def test_get_chat_model_with_preferences(venice_client, vcr_cassette):
    """Test get_chat_model with preferred models."""
    with vcr_cassette:
        model = await get_chat_model(venice_client, preferred=["llama-3.3-70b", "qwen-2.5-72b"])

        assert model is not None


@pytest.mark.integration
async def test_get_embedding_model_utility(venice_client, vcr_cassette):
    """Test the get_embedding_model utility function."""
    with vcr_cassette:
        model = await get_embedding_model(venice_client)

        assert model is not None
        assert isinstance(model, str)


@pytest.mark.integration
async def test_get_embedding_model_with_preferences(venice_client, vcr_cassette):
    """Test get_embedding_model with preferred models."""
    with vcr_cassette:
        model = await get_embedding_model(venice_client, preferred=["text-embedding-bge-m3"])

        assert model is not None


@pytest.mark.integration
async def test_get_multiple_models_utility(venice_client, vcr_cassette):
    """Test the get_multiple_models utility function."""
    with vcr_cassette:
        models = await get_multiple_models(venice_client, count=2)

        assert len(models) == 2
        assert models[0] != models[1]


@pytest.mark.integration
async def test_get_multiple_models_different_counts(venice_client, vcr_cassette):
    """Test get_multiple_models with different counts."""
    with vcr_cassette:
        models = await get_multiple_models(venice_client, count=3)

        assert len(models) == 3
        # All models should be unique
        assert len(set(models)) == 3


# ============================================================================
# Model Spec and Capabilities Tests
# ============================================================================


@pytest.mark.integration
async def test_model_spec_capabilities_parsing(venice_client, vcr_cassette):
    """Test that model_spec capabilities are correctly parsed."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Fetch models to populate cache with model_spec data
        await selector.get_available_models()

        # Check that models with function calling capability are identified
        models_data = await selector._fetch_models()

        # Look for models with capabilities
        models_with_caps = [
            m
            for m in models_data.values()
            if "model_spec" in m and "capabilities" in m.get("model_spec", {})
        ]

        if models_with_caps:
            # Verify capability structure
            model_with_caps = models_with_caps[0]
            caps = model_with_caps["model_spec"]["capabilities"]

            # Should have boolean capability fields
            assert isinstance(caps.get("supportsFunctionCalling"), bool)


@pytest.mark.integration
async def test_cache_concurrent_access(venice_client, vcr_cassette):
    """Test cache with concurrent access."""
    with vcr_cassette:
        selector = DynamicModelSelector(venice_client)

        # Make multiple concurrent requests
        tasks = [
            selector.get_available_models(),
            selector.get_available_models(),
            selector.get_available_models(),
        ]

        results = await asyncio.gather(*tasks)

        # All should return the same data (from cache after first fetch)
        assert len(results) == 3
        for result in results:
            assert len(result) > 0
