"""
Model filtering and ID building helpers.

Exports:
    * :func:`get_filtered_models` — filter a list of models by capabilities
    * :func:`_apply_model_filters` — internal filter implementation
    * :func:`build_model_id` — construct a Venice model ID with feature suffixes
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types.api.models import ModelResponse as ModelDetails


def get_filtered_models(
    models: Sequence[ModelDetails],
    model_type: str | None = None,
    supports_vision: bool | None = None,
    supports_reasoning: bool | None = None,
    supports_function_calling: bool | None = None,
    supports_web_search: bool | None = None,
    supports_log_probs: bool | None = None,
    optimized_for_code: bool | None = None,
    quantization: str | None = None,
    is_beta: bool | None = None,
    has_trait: str | None = None,
) -> list[ModelDetails]:
    """
    Filters a list of models based on various capabilities.

    :param models: A list of model details objects to filter.
    :param model_type: Optional. Filter for model type.
    :param supports_vision: Optional. Filter by vision support.
    :param supports_reasoning: Optional. Filter by reasoning support.
    :param supports_function_calling: Optional. Filter by function calling support.
    :param supports_web_search: Optional. Filter by web search support.
    :param supports_log_probs: Optional. Filter by log probability support.
    :param optimized_for_code: Optional. Filter by code optimization.
    :param quantization: Optional. Filter by quantization type (e.g., "fp16", "fp8").
    :param is_beta: Optional. Filter by beta status.
    :param has_trait: Optional. Filter by a specific model trait.
    :return: A new list of model details objects that match the filters.
    """
    return _apply_model_filters(
        models,
        model_type=model_type,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        supports_function_calling=supports_function_calling,
        supports_web_search=supports_web_search,
        supports_log_probs=supports_log_probs,
        optimized_for_code=optimized_for_code,
        quantization=quantization,
        is_beta=is_beta,
        has_trait=has_trait,
    )


def _apply_model_filters(
    models: Sequence[ModelDetails],
    model_type: str | None = None,
    supports_vision: bool | None = None,
    supports_reasoning: bool | None = None,
    supports_function_calling: bool | None = None,
    supports_web_search: bool | None = None,
    supports_log_probs: bool | None = None,
    optimized_for_code: bool | None = None,
    quantization: str | None = None,
    is_beta: bool | None = None,
    has_trait: str | None = None,
) -> list[ModelDetails]:
    """Helper function to apply filters to a list of models."""
    filtered_list: list[ModelDetails] = []
    for model_data in models:
        # Access model_spec (generated types use model_spec, not specs)
        model_spec = model_data.model_spec

        # Handle both Pydantic model and dict formats
        if hasattr(model_spec, "model_dump"):
            model_spec_dict = model_spec.model_dump()
        elif isinstance(model_spec, dict):
            model_spec_dict = model_spec
        else:
            continue

        # Use capabilities from model_spec (camelCase only)
        capabilities_obj = model_spec_dict.get("capabilities", {})
        if isinstance(capabilities_obj, dict):
            capabilities = capabilities_obj
        elif hasattr(capabilities_obj, "model_dump"):
            capabilities = capabilities_obj.model_dump()
        else:
            capabilities = {}

        if not isinstance(capabilities, dict):
            continue

        passes_filter = True

        # Get type from model_data, not model_spec
        if model_type is not None and getattr(model_data, "type", None) != model_type:
            passes_filter = False
        if (
            passes_filter
            and supports_vision is not None
            and capabilities.get("supportsVision") != supports_vision
        ):
            passes_filter = False
        if (
            passes_filter
            and supports_reasoning is not None
            and capabilities.get("supportsReasoning") != supports_reasoning
        ):
            passes_filter = False
        if (
            passes_filter
            and supports_function_calling is not None
            and capabilities.get("supportsFunctionCalling") != supports_function_calling
        ):
            passes_filter = False
        if (
            passes_filter
            and supports_web_search is not None
            and capabilities.get("supportsWebSearch") != supports_web_search
        ):
            passes_filter = False
        if (
            passes_filter
            and supports_log_probs is not None
            and capabilities.get("supportsLogProbs") != supports_log_probs
        ):
            passes_filter = False
        if (
            passes_filter
            and optimized_for_code is not None
            and capabilities.get("optimizedForCode") != optimized_for_code
        ):
            passes_filter = False
        if (
            passes_filter
            and quantization is not None
            and capabilities.get("quantization") != quantization
        ):
            passes_filter = False
        if (
            passes_filter and is_beta is not None and model_spec_dict.get("beta", False) != is_beta
        ):  # Check beta from model_spec
            passes_filter = False
        if (
            passes_filter
            and has_trait is not None
            and has_trait not in model_spec_dict.get("traits", [])
        ):  # Check traits from model_spec
            passes_filter = False

        if passes_filter:
            filtered_list.append(model_data)

    return filtered_list


def build_model_id(model: str, **params: str | int | float | bool) -> str:
    """Build a Venice model ID with feature suffixes.

    The Venice API supports model feature suffixes in the format:
    ``model_id:param=value&param2=value2``

    Args:
        model: Base model ID (e.g., ``"llama-3.3-70b"``).
        **params: Feature parameters (e.g., ``reasoning_effort="high"``). A bool
            renders as lowercase ``true`` or ``false``.

    Returns:
        Model ID with feature suffix, or just the base model ID if no
        params are provided.

    Raises:
        ValueError: If *model* is empty or contains a ``':'`` character
            (i.e., it already has a suffix).

    Example:
        >>> build_model_id("llama-3.3-70b", reasoning_effort="high")
        'llama-3.3-70b:reasoning_effort=high'
        >>> build_model_id("llama-3.3-70b", reasoning_effort="high", max_completion_tokens=4096)
        'llama-3.3-70b:reasoning_effort=high&max_completion_tokens=4096'
        >>> build_model_id("llama-3.3-70b")
        'llama-3.3-70b'
    """
    if not model:
        raise ValueError("model must be a non-empty string")
    if ":" in model:
        raise ValueError(
            f"model already contains a suffix separator ':': {model!r}. "
            "Pass the base model ID without an existing suffix."
        )
    if not params:
        return model
    suffix = "&".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in params.items()
    )
    return f"{model}:{suffix}"
