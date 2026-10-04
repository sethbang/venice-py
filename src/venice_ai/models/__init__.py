"""
Venice AI Models Module
======================

This module provides comprehensive model discovery, selection, and capability analysis
for the Venice AI platform. It includes intelligent model selection algorithms with
caching, preference handling, and capability-based filtering.

Key Components:
    - Model listing and discovery via the Models resource
    - Intelligent model selection with DynamicModelSelector
    - Capability-based filtering and matching
    - Cached model information with TTL support

Quick Start:
    >>> from venice_ai import VeniceClient, create_model_selector
    >>>
    >>> async with VeniceClient(api_key="...") as client:
    ...     # Intelligent model selection
    ...     selector = create_model_selector(client)
    ...     model = await selector.select_chat_model(
    ...         preferred_models=["llama-3.3-70b"],
    ...         require_function_calling=True
    ...     )

For detailed selection capabilities, see the DynamicModelSelector class.
"""

from .selection import (
    CheapestMusicResult,
    CheapestVideoResult,
    DynamicModelSelector,
    ModelCache,
    VideoInputMode,
    cheapest_model_strategy,
    cheapest_selector,
    cheapest_video_params,
    create_model_selector,
    get_chat_model,
    get_cheapest_video_model,
    get_embedding_model,
    get_multiple_models,
    get_video_model,
    model_price,
    music_request_params,
    video_input_mode,
)

__all__ = [
    "ModelCache",
    "CheapestMusicResult",
    "CheapestVideoResult",
    "cheapest_model_strategy",
    "cheapest_selector",
    "cheapest_video_params",
    "model_price",
    "music_request_params",
    "video_input_mode",
    "VideoInputMode",
    "DynamicModelSelector",
    "create_model_selector",
    "get_chat_model",
    "get_embedding_model",
    "get_multiple_models",
    "get_video_model",
    "get_cheapest_video_model",
]
