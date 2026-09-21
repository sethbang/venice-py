"""
Dynamic Model Selection for Venice AI SDK.

This module provides intelligent model selection capabilities with caching,
preference handling, capability-based filtering, and custom selection strategies
for production use.

Classes:
    ModelCache: Cache for model information with TTL support
    DynamicModelSelector: Intelligent model selector with capability filtering

Functions:
    create_model_selector: Factory function to create a model selector
    get_chat_model: Quick helper to get a chat model
    get_embedding_model: Quick helper to get an embedding model
    get_video_model: Quick helper to get a video model
    get_cheapest_video_model: Quick helper to find the cheapest video model via quoting
    get_multiple_models: Quick helper to get multiple models for concurrency

Types:
    ModelSelectorType: Type alias for custom model selection functions
    CheapestVideoResult: Dataclass returned by select_cheapest_video_model

Example:
    >>> from venice_ai import VeniceClient, create_model_selector
    >>>
    >>> async with VeniceClient(api_key="...") as client:
    ...     selector = create_model_selector(client)
    ...     model = await selector.select_chat_model(
    ...         preferred_models=["llama-3.3-70b"],
    ...         require_function_calling=True
    ...     )

    # With custom selector for cost optimization:
    >>> def cheapest_model(candidates):
    ...     # Custom logic to pick cheapest model
    ...     return candidates[0]["id"]
    >>> selector = create_model_selector(client, default_selector=cheapest_model)
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for model selector functions
# Selectors receive a list of model dictionaries and return a selected model ID
ModelSelectorType = Callable[[list[dict[str, Any]]], str]


@dataclass
class CheapestVideoResult:
    """Result from cost-aware video model selection via the quote API.

    Attributes:
        model: The model ID with the lowest quoted price.
        quote_usd: The quoted cost in USD for the selected model.
        all_quotes: Mapping of every successfully quoted model ID to its
            USD price.  Useful for debugging or displaying alternatives.
    """

    model: str
    quote_usd: float
    all_quotes: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelCache:
    """Cache for model information with TTL support."""

    models: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: float = 300.0  # 5 minutes

    def is_expired(self) -> bool:
        """Check if cache has expired."""
        # Cache is always expired if empty, regardless of timestamp
        if not self.models:
            return True

        age = (datetime.now(UTC) - self.last_updated).total_seconds()
        return age > self.ttl_seconds

    def get_models(self, resource_type: str | None = None) -> list[str]:
        """Get list of model IDs, optionally filtered by resource type."""
        if self.is_expired():
            return []

        if resource_type:
            filtered = [
                model_id
                for model_id, model_info in self.models.items()
                if model_info.get("type") == resource_type
            ]
            return filtered

        return list(self.models.keys())

    def update(self, models: dict[str, Any]) -> None:
        """Update cache with new model data."""
        self.models = models
        self.last_updated = datetime.now(UTC)
        logger.info(f"Model cache updated with {len(models)} models")


# ---------------------------------------------------------------------------
# Private helpers for video constraint filtering (shared by select_video_model
# and select_cheapest_video_model).
# ---------------------------------------------------------------------------

_VIDEO_RESOLUTION_ORDER = [
    "360p",
    "480p",
    "540p",
    "580p",
    "720p",
    "1080p",
    "1440p",
    "2160p",
    "4k",
]


def _parse_duration_seconds(d: str) -> int:
    """Parse a duration string like ``'5s'`` to integer seconds."""
    try:
        return int(d.rstrip("s"))
    except (ValueError, AttributeError):
        logger.warning(
            "Unparseable duration string %r — defaulting to 0 seconds. "
            "Expected a format like '5s' or '10s'.",
            d,
        )
        return 0


def _meets_resolution(model_resolutions: list[str], min_res: str) -> bool:
    """Return ``True`` if any resolution in *model_resolutions* meets *min_res*."""
    if not model_resolutions:
        return False
    min_idx = _VIDEO_RESOLUTION_ORDER.index(min_res) if min_res in _VIDEO_RESOLUTION_ORDER else -1
    if min_idx == -1:
        return False
    return any(
        _VIDEO_RESOLUTION_ORDER.index(r) >= min_idx
        for r in model_resolutions
        if r in _VIDEO_RESOLUTION_ORDER
    )


def _meets_duration(model_durations: list[str], min_dur: str) -> bool:
    """Return ``True`` if any duration in *model_durations* meets *min_dur*."""
    if not model_durations:
        return False
    min_secs = _parse_duration_seconds(min_dur)
    return any(_parse_duration_seconds(d) >= min_secs for d in model_durations)


def _filter_video_candidates(
    candidates: list[str],
    models_data: dict[str, Any],
    *,
    model_type: str | None = None,
    require_audio: bool = False,
    min_resolution: str | None = None,
    min_duration: str | None = None,
    exclude_beta: bool = False,
) -> list[str]:
    """Filter video model candidates by constraint criteria.

    This is the single implementation of video-constraint filtering used by
    both :meth:`DynamicModelSelector.select_video_model` and
    :meth:`DynamicModelSelector.select_cheapest_video_model`.
    """
    filtered: list[str] = []
    for model_id in candidates:
        model_data = models_data.get(model_id, {})
        model_spec = model_data.get("model_spec", {})
        constraints = model_spec.get("constraints", {})

        # Filter by model_type
        if model_type and constraints.get("model_type") != model_type:
            continue

        # Filter by audio support
        if require_audio and not constraints.get("audio", False):
            continue

        # Filter by minimum resolution
        if min_resolution and not _meets_resolution(
            constraints.get("resolutions", []), min_resolution
        ):
            continue

        # Filter by minimum duration
        if min_duration and not _meets_duration(constraints.get("durations", []), min_duration):
            continue

        # Filter by beta status
        if exclude_beta and model_data.get("beta", False):
            continue

        filtered.append(model_id)
    return filtered


# ---------------------------------------------------------------------------
# Private helper for image-generation capability filtering
# (used by select_image_model).
# ---------------------------------------------------------------------------

# Substrings identifying ``type == "image"`` models that do NOT perform
# text-to-image *generation* and would 400 on an ``image.create(prompt=...)``
# call (e.g. ``bria-bg-remover``, a background remover).
#
# Why a denylist instead of a positive capability check: the models catalog
# exposes no clean positive "supports generation" signal for image models —
# ``model_spec.capabilities`` is empty ``{}`` for every image-type model, and
# ``traits`` / ``constraints.aspectRatios`` are inconsistent across genuine
# generators (e.g. ``venice-sd35`` and ``z-image-turbo`` are real generators
# that carry neither). Upscalers and inpainters are already excluded upstream
# because Venice types them separately (``upscale`` / ``inpaint``), not
# ``image``, so the only non-generators currently mis-typed as ``image`` are
# background removers. The denylist also covers ``upscal``/``enhance`` patterns
# defensively in case such models are ever surfaced under the ``image`` type.
_IMAGE_NON_GENERATOR_PATTERNS = (
    "bg-remover",
    "background",
    "remover",
    "removal",
    "upscal",
    "enhance",
)


def _is_image_generation_model(model_data: dict[str, Any]) -> bool:
    """Return ``True`` unless *model_data* is a known non-generative image model.

    Distinguishes text-to-image generators (``venice-sd35``, ``qwen-image``,
    ``flux-*`` …) from non-generators like ``bria-bg-remover`` that share the
    ``image`` resource type but reject ``image.create(prompt=...)`` requests.
    Matches defensively on both the model id and its human-readable name so a
    renamed background remover ("Background Remover") is still excluded.
    """
    haystack = " ".join(str(model_data.get(key, "")) for key in ("id", "name")).lower()
    return not any(pattern in haystack for pattern in _IMAGE_NON_GENERATOR_PATTERNS)


class DynamicModelSelector:
    """
    Dynamic model selector that fetches available models and provides
    intelligent selection for production and testing scenarios.

    Supports custom selection strategies via the default_selector parameter
    or per-call selector argument. Strategies receive full model dictionaries
    including pricing data for cost-aware selection.
    """

    def __init__(
        self,
        client: Any,
        cache_ttl: float = 300.0,
        default_selector: ModelSelectorType | None = None,
    ):
        """
        Initialize the model selector.

        Args:
            client: Venice AI client instance for API calls
            cache_ttl: Time-to-live for model cache in seconds
            default_selector: Optional custom selection function that receives
                a list of model dicts and returns the selected model ID.
                Used as fallback when no per-call selector is provided.
        """
        self.client = client
        self._cache = ModelCache(ttl_seconds=cache_ttl)
        self._fetch_lock = asyncio.Lock()
        self.default_selector = default_selector

    async def _fetch_models(self, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch models from API with caching."""
        if not force_refresh and not self._cache.is_expired():
            return self._cache.models

        async with self._fetch_lock:
            # Double-check after acquiring lock
            if not force_refresh and not self._cache.is_expired():
                return self._cache.models

            try:
                logger.info("Fetching available models from API...")
                # Use the models endpoint to get ALL available models
                response = await self.client.models.list(type="all")

                # Convert response to dict format
                models_dict = {}
                if hasattr(response, "data") and response.data:
                    for model in response.data:
                        model_id = model.id if hasattr(model, "id") else str(model)
                        model_data = {
                            "id": model_id,
                            "object": getattr(model, "object", "model"),
                            "type": getattr(model, "type", "unknown"),  # Store the actual type
                            "created": getattr(model, "created", time.time()),
                            "owned_by": getattr(model, "owned_by", "unknown"),
                        }

                        # Store model_spec information if available
                        if hasattr(model, "model_spec"):
                            model_spec = model.model_spec

                            # Build the model_spec dictionary with capabilities and pricing
                            model_spec_dict: dict[str, Any] = {
                                "capabilities": {},
                                "pricing": None,
                            }

                            # Extract capabilities if available
                            if hasattr(model_spec, "capabilities"):
                                capabilities = model_spec.capabilities
                                model_spec_dict["capabilities"] = {
                                    "supportsFunctionCalling": getattr(
                                        capabilities, "supportsFunctionCalling", False
                                    ),
                                    "supportsVision": getattr(
                                        capabilities, "supportsVision", False
                                    ),
                                    "supportsWebSearch": getattr(
                                        capabilities, "supportsWebSearch", False
                                    ),
                                    "optimizedForCode": getattr(
                                        capabilities, "optimizedForCode", False
                                    ),
                                    "supportsReasoning": getattr(
                                        capabilities, "supportsReasoning", False
                                    ),
                                    "supportsAudioInput": getattr(
                                        capabilities, "supportsAudioInput", False
                                    ),
                                    "supportsVideoInput": getattr(
                                        capabilities, "supportsVideoInput", False
                                    ),
                                    "supportsLogProbs": getattr(
                                        capabilities, "supportsLogProbs", False
                                    ),
                                    "supportsResponseSchema": getattr(
                                        capabilities, "supportsResponseSchema", False
                                    ),
                                    "quantization": getattr(
                                        capabilities, "quantization", "not-available"
                                    ),
                                }

                            # Extract pricing if available
                            if hasattr(model_spec, "pricing") and model_spec.pricing:
                                pricing = model_spec.pricing
                                # Convert pricing to dict, handling both Pydantic models
                                # and plain dicts
                                if hasattr(pricing, "model_dump"):
                                    model_spec_dict["pricing"] = pricing.model_dump()
                                elif isinstance(pricing, dict):
                                    model_spec_dict["pricing"] = pricing
                                else:
                                    # Manual extraction for edge cases
                                    pricing_dict = {}
                                    for attr in [
                                        "input",
                                        "output",
                                        "cache_input",
                                        "generation",
                                        "upscale",
                                    ]:
                                        if hasattr(pricing, attr):
                                            val = getattr(pricing, attr)
                                            if val is not None:
                                                if hasattr(val, "model_dump"):
                                                    pricing_dict[attr] = val.model_dump()
                                                elif isinstance(val, dict):
                                                    pricing_dict[attr] = val
                                                else:
                                                    # Extract usd/diem from tier
                                                    pricing_dict[attr] = {
                                                        "usd": getattr(val, "usd", None),
                                                        "diem": getattr(val, "diem", None),
                                                    }
                                    if pricing_dict:
                                        model_spec_dict["pricing"] = pricing_dict

                            # Extract constraints if available (for image, video, inpaint models)
                            if hasattr(model_spec, "constraints") and model_spec.constraints:
                                constraints = model_spec.constraints
                                if hasattr(constraints, "model_dump"):
                                    model_spec_dict["constraints"] = constraints.model_dump()
                                elif isinstance(constraints, dict):
                                    model_spec_dict["constraints"] = constraints
                                else:
                                    # Manual extraction for video/image constraints
                                    constraints_dict = {}
                                    for attr in [
                                        "model_type",
                                        "aspect_ratios",
                                        "resolutions",
                                        "durations",
                                        "audio",
                                        "audio_configurable",
                                        "video_input",
                                        "promptCharacterLimit",
                                        "steps",
                                        "widthHeightDivisor",
                                        "combineImages",
                                    ]:
                                        if hasattr(constraints, attr):
                                            val = getattr(constraints, attr)
                                            if val is not None:
                                                if hasattr(val, "model_dump"):
                                                    constraints_dict[attr] = val.model_dump()
                                                else:
                                                    constraints_dict[attr] = val
                                    if constraints_dict:
                                        model_spec_dict["constraints"] = constraints_dict

                            # Assign the complete model_spec dictionary
                            model_data["model_spec"] = model_spec_dict

                            # Extract additional metadata from model_spec
                            model_data["availableContextTokens"] = getattr(
                                model_spec, "availableContextTokens", None
                            )
                            model_data["beta"] = getattr(model_spec, "beta", False) or getattr(
                                model_spec, "betaModel", False
                            )
                            model_data["privacy"] = getattr(model_spec, "privacy", None)
                            model_data["model_sets"] = getattr(model_spec, "model_sets", []) or []
                            model_data["name"] = getattr(model_spec, "name", "")
                            model_data["description"] = getattr(model_spec, "description", "")
                            model_data["offline"] = getattr(model_spec, "offline", False)
                            deprecation = getattr(model_spec, "deprecation", None)
                            model_data["deprecation_date"] = (
                                getattr(deprecation, "date", None) if deprecation else None
                            )

                            # Extract traits if available
                            if hasattr(model_spec, "traits"):
                                model_data["traits"] = (
                                    list(model_spec.traits) if model_spec.traits else []
                                )

                        models_dict[model_id] = model_data

                self._cache.update(models_dict)
                logger.info(f"Successfully fetched {len(models_dict)} models")
                return models_dict

            except asyncio.CancelledError:
                raise  # Always re-raise for graceful shutdown
            except (ValueError, TypeError, AttributeError, OSError) as e:
                logger.exception(f"Failed to fetch models: {e}")
                if self._cache.models:
                    logger.warning("Using cached models despite fetch failure")
                    return self._cache.models
                raise

    async def select_by_trait(self, trait: str, resource_type: str | None = None) -> str | None:
        """
        Select the model assigned to a specific Venice trait.

        Venice assigns traits like "default", "fastest", "default_code",
        "default_reasoning", "default_vision", "function_calling_default",
        "most_intelligent", "most_uncensored" to indicate canonical model roles.

        Args:
            trait: The trait to search for (e.g., "default", "fastest", "default_code")
            resource_type: Optional filter by resource type (e.g., "text", "image")

        Returns:
            The model ID with the matching trait, or None if no model has that trait
        """
        models = await self._fetch_models()

        for model_id, model_data in models.items():
            # Filter by resource type if specified
            if resource_type and model_data.get("type") != resource_type:
                continue

            traits = model_data.get("traits", [])
            if trait in traits:
                logger.info(f"Found model '{model_id}' with trait '{trait}'")
                return model_id

        logger.debug(
            f"No model found with trait '{trait}'"
            + (f" and type '{resource_type}'" if resource_type else "")
        )
        return None

    def _get_trait_model(self, trait: str, resource_type: str | None = None) -> str | None:
        """
        Synchronous trait lookup against the already-populated cache.

        This is a non-async helper for use inside methods that have already
        called _fetch_models(). Returns None if cache is empty or no match.
        """
        if not self._cache.models:
            return None

        for model_id, model_data in self._cache.models.items():
            if resource_type and model_data.get("type") != resource_type:
                continue
            traits = model_data.get("traits", [])
            if trait in traits:
                return model_id

        return None

    async def _select_simple_model(
        self,
        resource_type: str,
        label: str,
        *,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """Generic model selection for simple resource types.

        Shared implementation for resource types that follow the same
        pattern: fetch available models, apply exclusions, run custom
        selector, try preferred list, then fall back to first candidate.
        """
        available = await self.get_available_models(resource_type=resource_type)
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise ValueError(f"No available {label} models found")

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            candidate_objects = [self._cache.models[mid] for mid in candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected {label} model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred {label} model: {preferred}")
                    return preferred

        # Fallback to first available model
        selected = candidates[0]
        logger.info(f"Selected fallback {label} model: {selected}")
        return selected

    async def _select_trait_model(
        self,
        trait_name: str,
        label: str,
        capability_kwarg: str,
        *,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """Select model by trait, falling back to chat model with capability.

        Shared implementation for text models that first check a Venice
        trait (e.g. ``default_code``) and, if that isn't available,
        delegate to :meth:`select_chat_model` with the appropriate
        capability requirement.
        """
        # Try trait-based selection first
        await self._fetch_models()
        trait_model = self._get_trait_model(trait_name, resource_type="text")

        exclude_models = exclude_models or set()
        available = await self.get_available_models(resource_type="text")

        if trait_model and trait_model in available and trait_model not in exclude_models:
            if preferred_models:
                for preferred in preferred_models:
                    if preferred in available and preferred not in exclude_models:
                        logger.info(f"Selected preferred {label} model: {preferred}")
                        return preferred

            logger.info(f"Selected {label} model via '{trait_name}' trait: {trait_model}")
            return trait_model

        # Fall back to capability-based selection
        return await self.select_chat_model(
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
            **{capability_kwarg: True},
        )

    @staticmethod
    def _is_past_deprecation(model_data: dict[str, Any]) -> bool:
        """Return True iff the model has a deprecation date that has already passed.

        Models with no ``deprecation_date``, an unparseable date, or a date in
        the future are considered active.
        """
        date_str = model_data.get("deprecation_date")
        if not date_str:
            return False
        try:
            # The API returns dates like "2026-04-15T00:00:00.000Z"; fromisoformat
            # accepts the trailing offset since Python 3.11 but not the literal "Z"
            # before 3.11, so normalize defensively.
            normalized = date_str.replace("Z", "+00:00")
            deprecation_at = datetime.fromisoformat(normalized)
            if deprecation_at.tzinfo is None:
                deprecation_at = deprecation_at.replace(tzinfo=UTC)
            return datetime.now(UTC) >= deprecation_at
        except (ValueError, TypeError):
            return False

    async def get_available_models(
        self, resource_type: str | None = None, force_refresh: bool = False
    ) -> list[str]:
        """
        Get list of available models, excluding offline and past-deprecation models.

        Args:
            resource_type: Filter by resource type ('text', 'image', 'video', etc.)
            force_refresh: Force refresh of model cache

        Returns:
            List of available model IDs (excludes offline models and models whose
            ``deprecation.date`` has passed). Callers who need a deprecated model
            should pass it directly to ``client.chat.completions.create(model=...)``
            rather than going through the resolver.
        """
        await self._fetch_models(force_refresh=force_refresh)
        all_models = self._cache.get_models(resource_type=resource_type)

        available = []
        for model_id in all_models:
            model_data = self._cache.models.get(model_id, {})
            if model_data.get("offline", False):
                continue
            if self._is_past_deprecation(model_data):
                logger.debug(
                    "Skipping deprecated model %s (deprecation_date=%s)",
                    model_id,
                    model_data.get("deprecation_date"),
                )
                continue
            available.append(model_id)
        return available

    def _is_reasoning_model(self, model_id: str) -> bool:
        """Return True if ``model_id`` advertises reasoning support.

        Reasoning models spend their token budget on internal thinking and
        frequently return an empty ``message.content`` under small
        ``max_completion_tokens`` limits. Callers that want guaranteed visible
        content (general chat, comparison/concurrency tests) use this to filter
        them out unless reasoning is explicitly required.
        """
        return bool(
            self._cache.models.get(model_id, {})
            .get("model_spec", {})
            .get("capabilities", {})
            .get("supportsReasoning", False)
        )

    async def select_chat_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        require_function_calling: bool = False,
        require_vision: bool = False,
        require_reasoning: bool = False,
        require_code_optimization: bool = False,
        require_response_schema: bool = False,
        min_context_tokens: int | None = None,
        require_private: bool = False,
        exclude_beta: bool = False,
        prefer_recommended: bool = False,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable chat completion model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            require_function_calling: If True, only select models that support function calling
            require_vision: If True, only select models that support vision (image input)
            require_reasoning: If True, only select models that support reasoning
                with thinking blocks
            require_code_optimization: If True, only select models optimized for
                code generation
            require_response_schema: If True, only select models that support
                structured output via response schema
            min_context_tokens: If set, only select models with at least this many
                context tokens
            require_private: If True, only select models with privacy="private"
                (no data stored by provider)
            exclude_beta: If True, exclude models marked as beta
            prefer_recommended: If True, prefer models in Venice's
                "venice_recommendations" model set
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable model found
        """
        if require_function_calling:
            return await self.select_function_calling_model(
                preferred_models=preferred_models,
                exclude_models=exclude_models,
                selector=selector,
            )

        available = await self.get_available_models(resource_type="text")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise ValueError("No available chat models found")

        # Apply capability-based filtering if any capability is required
        has_capability_filter = any(
            [
                require_vision,
                require_reasoning,
                require_code_optimization,
                require_response_schema,
                min_context_tokens is not None,
                require_private,
                exclude_beta,
            ]
        )

        if has_capability_filter:
            models_data = await self._fetch_models()
            filtered = []
            for model_id in candidates:
                model_data = models_data.get(model_id, {})
                model_spec = model_data.get("model_spec", {})
                capabilities = model_spec.get("capabilities", {})

                # Check vision support
                if require_vision and not capabilities.get("supportsVision", False):
                    continue

                # Check reasoning support
                if require_reasoning and not capabilities.get("supportsReasoning", False):
                    continue

                # Check code optimization
                if require_code_optimization and not capabilities.get("optimizedForCode", False):
                    continue

                # Check response schema support
                if require_response_schema and not capabilities.get(
                    "supportsResponseSchema", False
                ):
                    continue

                # Check minimum context tokens
                if min_context_tokens is not None:
                    available_context = model_data.get("availableContextTokens")
                    if available_context is None or available_context < min_context_tokens:
                        continue

                # Check privacy requirement
                if require_private and model_data.get("privacy") != "private":
                    continue

                # Check beta exclusion
                if exclude_beta and model_data.get("beta", False):
                    continue

                filtered.append(model_id)

            if not filtered:
                raise ValueError(
                    f"No chat models found matching requirements: "
                    f"vision={require_vision}, reasoning={require_reasoning}, "
                    f"code={require_code_optimization}, schema={require_response_schema}, "
                    f"min_context={min_context_tokens}, private={require_private}, "
                    f"exclude_beta={exclude_beta}"
                )
            candidates = filtered

        # Prefer non-reasoning models for general chat unless reasoning was
        # explicitly requested. Applied to the candidate pool itself so it
        # constrains every downstream path equally — a custom ``selector``,
        # ``preferred_models``, and trait-based selection. Without this, a
        # custom selector would return its pick before the trait-path
        # non-reasoning fallback can run, and
        # randomly picks a reasoning model whose content comes back empty under
        # small token budgets. Falls back to the full pool if every candidate
        # is a reasoning model.
        if not require_reasoning and len(candidates) > 1:
            non_reasoning = [m for m in candidates if not self._is_reasoning_model(m)]
            if non_reasoning:
                candidates = non_reasoning

        # Prefer Venice-recommended models if requested
        if prefer_recommended and len(candidates) > 1:
            recommended = []
            non_recommended = []
            models_data_for_ranking = await self._fetch_models()
            for mid in candidates:
                model_data = models_data_for_ranking.get(mid, {})
                model_sets = model_data.get("model_sets", [])
                if "venice_recommendations" in model_sets:
                    recommended.append(mid)
                else:
                    non_recommended.append(mid)
            if recommended:
                candidates = recommended + non_recommended
                logger.debug(
                    f"Reordered candidates to prefer {len(recommended)} Venice-recommended models"
                )

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            # Pass full model objects to the selector
            candidate_objects = [self._cache.models[mid] for mid in candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected chat model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred chat model: {preferred}")
                    return preferred

        # Try trait-based selection (Venice's canonical default)
        trait_model = self._get_trait_model("default", resource_type="text")
        if trait_model and trait_model in candidates:
            # If the caller didn't explicitly request reasoning, check whether
            # the "default" trait model is a reasoning model.  Reasoning models
            # consume thinking tokens from max_tokens, which can leave
            # message.content empty with standard token budgets.  For general
            # chat we prefer a non-reasoning model when one is available.
            if not require_reasoning:
                model_data = self._cache.models.get(trait_model, {})
                caps = model_data.get("model_spec", {}).get("capabilities", {})
                if caps.get("supportsReasoning", False):
                    non_reasoning = [
                        m
                        for m in candidates
                        if not self._cache.models.get(m, {})
                        .get("model_spec", {})
                        .get("capabilities", {})
                        .get("supportsReasoning", False)
                    ]
                    if non_reasoning:
                        # Prefer Venice-recommended models to avoid picking
                        # a niche or low-quality model from arbitrary API
                        # ordering.  Falls back to first non-reasoning model
                        # if none are in the recommended set.
                        recommended = [
                            m
                            for m in non_reasoning
                            if "venice_recommendations"
                            in self._cache.models.get(m, {}).get("model_sets", [])
                        ]
                        selected = recommended[0] if recommended else non_reasoning[0]
                        logger.info(
                            f"Selected non-reasoning chat model: {selected} "
                            f"(skipped reasoning default '{trait_model}')"
                        )
                        return selected
            logger.info(f"Selected chat model via 'default' trait: {trait_model}")
            return trait_model

        # Fallback to first available model
        selected = candidates[0]
        logger.info(f"Selected fallback chat model: {selected}")
        return selected

    async def select_function_calling_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a chat model that supports function calling/tools.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID that supports function calling

        Raises:
            ValueError: If no suitable function calling model found
        """
        available = await self.get_available_models(resource_type="text")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise ValueError("No available chat models found")

        # Get the full model data to check capabilities
        models_data = await self._fetch_models()

        # Filter models based on actual supportsFunctionCalling capability
        function_calling_candidates = []
        for model in candidates:
            # Check actual capabilities if available
            model_data = models_data.get(model, {})
            model_spec = model_data.get("model_spec", {})
            capabilities = model_spec.get("capabilities", {})

            # Use actual supportsFunctionCalling if available
            if capabilities.get("supportsFunctionCalling", False):
                function_calling_candidates.append(model)
                logger.debug(f"Model {model} supports function calling (API confirmed)")
            # Skip models without explicit function calling support
            # (pattern-based guessing is unreliable)

        if not function_calling_candidates:
            raise ValueError("No function calling capable models found")

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            candidate_objects = [self._cache.models[mid] for mid in function_calling_candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected function calling model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in function_calling_candidates:
                    logger.info(f"Selected preferred function calling model: {preferred}")
                    return preferred

        # Try trait-based selection (Venice's canonical function calling default)
        trait_model = self._get_trait_model("function_calling_default", resource_type="text")
        if trait_model and trait_model in function_calling_candidates:
            logger.info(f"Selected function calling model via trait: {trait_model}")
            return trait_model

        # Fallback to first candidate
        selected = function_calling_candidates[0]
        logger.info(f"Selected fallback function calling model: {selected}")
        return selected

    async def select_code_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a model optimized for code generation.

        Uses the 'default_code' trait as the primary selection, falling back
        to any model with optimizedForCode=True capability.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID optimized for code

        Raises:
            ValueError: If no suitable code model found
        """
        return await self._select_trait_model(
            "default_code",
            "code",
            "require_code_optimization",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_vision_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a model that supports vision (image input).

        Uses the 'default_vision' trait as the primary selection, falling back
        to any model with supportsVision=True capability.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID with vision support

        Raises:
            ValueError: If no suitable vision model found
        """
        return await self._select_trait_model(
            "default_vision",
            "vision",
            "require_vision",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_reasoning_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a model that supports reasoning with thinking blocks.

        Uses the 'default_reasoning' trait as the primary selection, falling back
        to any model with supportsReasoning=True capability.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID with reasoning support

        Raises:
            ValueError: If no suitable reasoning model found
        """
        return await self._select_trait_model(
            "default_reasoning",
            "reasoning",
            "require_reasoning",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_embedding_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable embedding model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable model found
        """
        return await self._select_simple_model(
            "embedding",
            "embedding",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_image_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable image generation model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable model found
        """
        available = await self.get_available_models(resource_type="image")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise ValueError("No available image models found")

        # Restrict to text-to-image *generation* models. The "image" resource
        # type also covers non-generative models such as background removers
        # (e.g. bria-bg-remover), which 400 on an image.create(prompt=...) call.
        # Filtering the candidate pool itself — before any selector runs —
        # constrains every downstream path equally (custom ``selector`` such as
        # random_cheap_strategy, ``preferred_models``, and trait selection), so a
        # cost-driven selector can never hand back a background remover for
        # generation. Falls back to the unfiltered pool if filtering would empty
        # it, so this never hard-fails on an unexpected catalog shape.
        generators = [
            m for m in candidates if _is_image_generation_model(self._cache.models.get(m, {}))
        ]
        if generators:
            candidates = generators

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            # Pass full model objects to the selector
            candidate_objects = [self._cache.models[mid] for mid in candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected image model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred image model: {preferred}")
                    return preferred

        # Try trait-based selection (Venice's canonical default image model)
        trait_model = self._get_trait_model("default", resource_type="image")
        if trait_model and trait_model in candidates:
            logger.info(f"Selected image model via 'default' trait: {trait_model}")
            return trait_model

        # Fallback to first available model
        selected = candidates[0]
        logger.info(f"Selected fallback image model: {selected}")
        return selected

    async def select_video_model(
        self,
        model_type: str | None = None,  # "text-to-video" or "image-to-video"
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        exclude_beta: bool = False,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable video generation model.

        Args:
            model_type: Filter by video model type ("text-to-video" or "image-to-video")
            require_audio: If True, only select models that support audio generation
            min_resolution: Minimum resolution (e.g., "720p", "1080p", "4k").
                Models must support this resolution or higher.
            min_duration: Minimum duration (e.g., "5s", "10s").
                Models must support at least this duration.
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            exclude_beta: If True, exclude models marked as beta
            selector: Optional custom selection function

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable video model found
        """
        available = await self.get_available_models(resource_type="video")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise ValueError("No available video models found")

        # Get full model data for constraint filtering
        models_data = await self._fetch_models()

        # Apply constraint-based filtering via shared helper
        filtered = _filter_video_candidates(
            candidates,
            models_data,
            model_type=model_type,
            require_audio=require_audio,
            min_resolution=min_resolution,
            min_duration=min_duration,
            exclude_beta=exclude_beta,
        )

        if not filtered:
            raise ValueError(
                f"No video models found matching criteria: "
                f"model_type={model_type}, require_audio={require_audio}, "
                f"min_resolution={min_resolution}, min_duration={min_duration}, "
                f"exclude_beta={exclude_beta}"
            )

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            candidate_objects = [self._cache.models[mid] for mid in filtered]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected video model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in filtered:
                    logger.info(f"Selected preferred video model: {preferred}")
                    return preferred

        # Fallback to first available model
        selected = filtered[0]
        logger.info(f"Selected video model: {selected}")
        return selected

    async def select_cheapest_video_model(
        self,
        *,
        duration: str = "5s",
        model_type: str | None = None,
        resolution: str | None = None,
        audio: bool | None = None,
        aspect_ratio: str | None = None,
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        exclude_models: set[str] | None = None,
        exclude_beta: bool = True,
    ) -> CheapestVideoResult:
        """
        Select the cheapest video model by quoting all viable candidates.

        This method first filters video models using the same constraint logic
        as :meth:`select_video_model` (model_type, audio support, resolution,
        duration), then concurrently calls the ``POST /video/quote`` endpoint
        for every candidate and returns the model with the lowest USD quote.

        .. note::

           Each call issues *N* quote API requests (one per candidate).
           Quotes are free (no generation occurs), but this is more expensive
           in terms of network round-trips than :meth:`select_video_model`.
           Consider caching the result across a test session.

        Args:
            duration: Video duration for the quote (e.g., ``"5s"``).
            model_type: Filter by ``"text-to-video"`` or ``"image-to-video"``.
            resolution: Resolution to quote (e.g., ``"720p"``).
                ``None`` uses each model's default (typically cheapest).
            audio: Whether to include audio in the quote.
                ``None`` omits the parameter (uses model default).
            aspect_ratio: Aspect ratio for the quote (e.g., ``"16:9"``).
            require_audio: Constraint filter — only consider models that support audio.
            min_resolution: Constraint filter — minimum supported resolution.
            min_duration: Constraint filter — minimum supported duration.
            exclude_models: Model IDs to exclude from consideration.
            exclude_beta: If ``True``, exclude beta models.

        Returns:
            A :class:`CheapestVideoResult` containing the cheapest model ID,
            its quoted USD price, and a dict of all successful quotes.

        Raises:
            ValueError: If no candidate models remain after filtering, or if
                every candidate fails to return a valid quote.

        Example:
            >>> selector = create_model_selector(client)
            >>> result = await selector.select_cheapest_video_model(
            ...     model_type="text-to-video",
            ...     duration="5s",
            ... )
            >>> print(f"Cheapest: {result.model} at ${result.quote_usd:.4f}")
        """
        # --- Step 1: Get constraint-filtered candidates -----------------
        available = await self.get_available_models(resource_type="video")
        exclude_models = exclude_models or set()

        candidates = [m for m in available if m not in exclude_models]
        if not candidates:
            raise ValueError("No available video models found")

        models_data = await self._fetch_models()

        # Apply constraint-based filtering via shared helper
        filtered = _filter_video_candidates(
            candidates,
            models_data,
            model_type=model_type,
            require_audio=require_audio,
            min_resolution=min_resolution,
            min_duration=min_duration,
            exclude_beta=exclude_beta,
        )

        if not filtered:
            raise ValueError(
                f"No video models found matching criteria: "
                f"model_type={model_type}, require_audio={require_audio}, "
                f"min_resolution={min_resolution}, min_duration={min_duration}, "
                f"exclude_beta={exclude_beta}"
            )

        # --- Step 2: Quote every candidate concurrently -----------------
        async def _quote_model(
            mid: str,
        ) -> tuple[str, float] | None:
            try:
                quote_resp = await self.client.video.quote(
                    model=mid,
                    duration_seconds=duration,
                    resolution=resolution,
                    audio=audio,
                    aspect_ratio=aspect_ratio,
                )
                cost: int | float = quote_resp.quote
                logger.debug(f"Video quote for {mid}: ${float(cost):.4f}")
                return (mid, float(cost))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"Quote failed for {mid}: {exc}")
                return None

        results = await asyncio.gather(
            *[_quote_model(mid) for mid in filtered],
            return_exceptions=False,
        )

        valid: list[tuple[str, float]] = [r for r in results if r is not None]

        if not valid:
            raise ValueError(
                f"All {len(filtered)} candidate video models failed to return "
                f"a valid quote. Candidates were: {filtered}"
            )

        # --- Step 3: Pick the cheapest ----------------------------------
        valid.sort(key=lambda pair: pair[1])
        cheapest_model, cheapest_price = valid[0]
        all_quotes = dict(valid)

        logger.info(
            f"Selected cheapest video model: {cheapest_model} "
            f"(${cheapest_price:.4f}) out of {len(valid)} quoted models"
        )

        return CheapestVideoResult(
            model=cheapest_model,
            quote_usd=cheapest_price,
            all_quotes=all_quotes,
        )

    async def select_audio_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable audio/speech generation model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable model found
        """
        return await self._select_simple_model(
            "tts",
            "audio",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_inpaint_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable inpaint (image editing) model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable inpaint model found
        """
        return await self._select_simple_model(
            "inpaint",
            "inpaint",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_asr_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable ASR (Automatic Speech Recognition) model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable ASR model found
        """
        return await self._select_simple_model(
            "asr",
            "ASR",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_music_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable music generation model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable music model found
        """
        return await self._select_simple_model(
            "music",
            "music",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_decision_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable decision ("System One") model.

        Note that decision models are currently beta-flagged; unlike the chat
        and video selectors this one does not offer an ``exclude_beta`` filter,
        because applying it would leave no candidates at all.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            ValueError: If no suitable decision model found
        """
        return await self._select_simple_model(
            "decision",
            "decision",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_models_for_concurrency_test(
        self,
        count: int = 2,
        resource_type: str = "text",
        exclude_models: set[str] | None = None,
    ) -> list[str]:
        """
        Select multiple models for concurrency testing or production use.

        Args:
            count: Number of models to select
            resource_type: Type of models to select (default "text"). Use "text"
                for chat models, "image" for image models, etc.
            exclude_models: Set of models to exclude from selection

        Returns:
            List of selected model IDs

        Raises:
            ValueError: If not enough models available
        """
        available = await self.get_available_models(resource_type=resource_type)
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if len(candidates) < count:
            raise ValueError(f"Need {count} models but only {len(candidates)} available")

        # Try to get diverse models for better testing
        selected: list[str] = []

        # Try to get diverse models using traits first. Reasoning models
        # consume their token budget on thinking and often return empty
        # message.content under standard token limits, which is undesirable for
        # concurrency/comparison tests — so skip reasoning trait models here
        # (e.g. the 'default'/'most_intelligent' traits, which now resolve to
        # reasoning models). They are still eligible via the reasoning fallback
        # below if too few non-reasoning models exist.
        diversity_traits = ["default", "fastest", "most_intelligent"]
        for trait in diversity_traits:
            if len(selected) >= count:
                break
            trait_model = self._get_trait_model(trait, resource_type=resource_type)
            if (
                trait_model
                and trait_model in candidates
                and trait_model not in selected
                and not self._is_reasoning_model(trait_model)
            ):
                selected.append(trait_model)

        # Fill remaining slots, preferring non-reasoning models.
        remaining = [m for m in candidates if m not in selected]
        non_reasoning = [m for m in remaining if not self._is_reasoning_model(m)]
        reasoning = [m for m in remaining if m not in non_reasoning]
        ordered_remaining = non_reasoning + reasoning

        for model in ordered_remaining:
            if len(selected) >= count:
                break
            selected.append(model)

        logger.info(f"Selected {len(selected)} models for concurrency test: {selected}")
        return selected[:count]

    async def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific model.

        Args:
            model_id: ID of the model to get info for

        Returns:
            Model information dict or None if not found
        """
        await self._fetch_models()
        return self._cache.models.get(model_id)

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "model_count": len(self._cache.models),
            "last_updated": self._cache.last_updated,
            "is_expired": self._cache.is_expired(),
            "ttl_seconds": self._cache.ttl_seconds,
        }


def create_model_selector(
    client: Any,
    cache_ttl: float = 300.0,
    default_selector: ModelSelectorType | None = None,
) -> DynamicModelSelector:
    """
    Factory function to create a model selector instance.

    .. deprecated::
        Use ``client.models.resolve()`` instead. This function will be removed
        in a future release.

    Args:
        client: Venice AI client instance
        cache_ttl: Cache TTL in seconds
        default_selector: Optional custom selection function that receives
            a list of model dicts and returns the selected model ID.

    Returns:
        DynamicModelSelector instance
    """
    import warnings

    warnings.warn(
        "create_model_selector() is deprecated, use client.models.resolve()",
        DeprecationWarning,
        stacklevel=2,
    )
    return DynamicModelSelector(client, cache_ttl=cache_ttl, default_selector=default_selector)


# Utility functions for common use cases
async def get_chat_model(client: Any, preferred: list[str] | None = None) -> str:
    """Quick helper to get a chat model for production or testing."""
    selector = DynamicModelSelector(client)
    return await selector.select_chat_model(preferred_models=preferred)


async def get_embedding_model(client: Any, preferred: list[str] | None = None) -> str:
    """Quick helper to get an embedding model for production or testing."""
    selector = DynamicModelSelector(client)
    return await selector.select_embedding_model(preferred_models=preferred)


async def get_multiple_models(client: Any, count: int = 2) -> list[str]:
    """Quick helper to get multiple models for concurrency testing or production use."""
    selector = DynamicModelSelector(client)
    return await selector.select_models_for_concurrency_test(count=count)


async def get_video_model(
    client: Any,
    model_type: str | None = None,
    preferred: list[str] | None = None,
) -> str:
    """Quick helper to get a video model."""
    selector = DynamicModelSelector(client)
    return await selector.select_video_model(model_type=model_type, preferred_models=preferred)


async def get_cheapest_video_model(
    client: Any,
    model_type: str | None = None,
    duration: str = "5s",
    **kwargs: Any,
) -> CheapestVideoResult:
    """Quick helper to find the cheapest video model for given parameters.

    Queries the ``POST /video/quote`` endpoint for every eligible model and
    returns a :class:`CheapestVideoResult` with the model that has the lowest
    cost.

    Args:
        client: Venice AI client instance.
        model_type: ``"text-to-video"`` or ``"image-to-video"``.
        duration: Video duration for the quote (default ``"5s"``).
        **kwargs: Forwarded to
            :meth:`DynamicModelSelector.select_cheapest_video_model`.

    Returns:
        A :class:`CheapestVideoResult` with the cheapest model, its USD
        price, and all successful quotes.

    Example:
        >>> result = await get_cheapest_video_model(client, model_type="text-to-video")
        >>> print(f"Use model {result.model} (${result.quote_usd:.4f})")
    """
    selector = DynamicModelSelector(client)
    return await selector.select_cheapest_video_model(
        model_type=model_type, duration=duration, **kwargs
    )
