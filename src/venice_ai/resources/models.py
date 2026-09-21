"""
Venice AI Models Resource Module.

This module provides comprehensive access to Venice AI's model ecosystem, enabling
developers to discover available AI models, understand their capabilities, and
efficiently select appropriate models for various tasks. The module supports model
discovery through multiple approaches including direct listing, semantic traits,
and compatibility mappings.

The Venice AI platform offers a diverse range of models optimized for different
use cases including text generation, image generation, embeddings, text-to-speech,
and image upscaling. This module provides the tools to navigate this ecosystem
and make informed model selection decisions.

Key Features:
    - Comprehensive model discovery and listing
    - Semantic trait-based model selection (e.g., "fastest", "best", "default")
    - Cross-platform compatibility mappings for migration from other AI services
    - Model capability and pricing information
    - Type-based filtering for specific model categories

Classes:
    Models: Asynchronous resource for model discovery and information retrieval
"""

from __future__ import annotations

import builtins
import time
from typing import TYPE_CHECKING, Literal

from .._resource import APIResource

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401
    from ..models.selection import CheapestVideoResult, DynamicModelSelector

from ..types.api import (
    ModelCompatibilityResponse,
    ModelsListResponse,
    ModelsQueryParams,
    ModelTraitsQueryParams,
    ModelTraitsResponse,
)
from ..types.api.capabilities import (
    Capabilities,
    ChatCapabilities,
    GenericCapabilities,
    ImageCapabilities,
    InpaintCapabilities,
    VideoCapabilities,
)
from ..types.api.models import (
    ImageModelConstraints,
    InpaintModelConstraints,
    ModelResponse,
    MusicModelSpec,
    VideoModelConstraints,
)

# Cache TTL for the listing reused by get() / get_capabilities() — short
# enough that catalog changes propagate quickly, long enough to absorb
# back-to-back lookups that would otherwise hammer /models.
_MODEL_LIST_CACHE_TTL_SECONDS = 30.0


# Public type alias for the ``Models.list(type=...)`` kwarg. Callers that
# fan out across multiple types (e.g. the CLI) should annotate their
# local variables with this alias so the call-site type-check stays tight.
type ModelListType = Literal[
    "text",
    "chat",
    "image",
    "embedding",
    "tts",
    "asr",
    "music",
    "upscale",
    "inpaint",
    "video",
    "decision",
    "all",
    "code",
]


class Models(APIResource["VeniceClient"]):
    """
    Asynchronous resource for comprehensive model discovery and capability analysis.

    This class provides a complete interface for exploring Venice AI's model ecosystem,
    including direct model listing, semantic trait-based discovery, and compatibility
    mappings for seamless migration from other AI platforms. All operations support
    flexible filtering and return detailed model metadata.

    The class enables developers to make informed model selection decisions by providing
    access to model capabilities, pricing information, performance characteristics,
    and compatibility details across the entire Venice AI model catalog.

    Key Capabilities:
        - List all available models with detailed metadata
        - Discover models by semantic traits (fastest, best, default, etc.)
        - Access compatibility mappings for external model migration
        - Filter models by type (text, image, embedding, TTS, upscale, decision)
        - Retrieve comprehensive model specifications and pricing

    Args:
        client: The Venice AI client instance for making authenticated API requests.

    Example:
        Model discovery and selection:

        .. code-block:: python

            async with VeniceClient() as client:
                # List all available models
                models = await client.models.list()

                # Find models by type
                text_models = await client.models.list(type="text")
                image_models = await client.models.list(type="image")

                # Use semantic traits for easy selection
                traits = await client.models.list_traits(type="text")
                fastest_model = traits.data["fastest"]
                best_model = traits.data["best"]

                # Check compatibility for migration
                compatibility = await client.models.list_compatibility()
                venice_equivalent = compatibility.data.get("gpt-4")
    """

    async def list(
        self,
        *,
        type: ModelListType | None = None,
    ) -> ModelsListResponse:
        """
        Lists available models asynchronously.

        Asynchronously retrieves a list of AI models available through the Venice API.
        Models can optionally be filtered by type to narrow down results to specific
        categories such as text generation, image generation, or embedding models.

        :param type: Filter for model type. Valid API values: ``"text"``,
            ``"image"``, ``"embedding"``, ``"tts"``, ``"asr"``, ``"music"``, ``"upscale"``,
            ``"inpaint"``, ``"video"``, ``"decision"``, ``"all"``, ``"code"``. The SDK also accepts
            ``"chat"`` as an alias for ``"text"`` to match the user-facing language used
            elsewhere (e.g. :py:meth:`resolve(type="chat") <resolve>`). If not provided,
            the SDK sends ``type="all"`` so the response is the union of every model
            type — the server's own default is ``text``-only, which surprises callers
            who expect "no filter" to mean "everything". Pass ``type="text"`` (or the
            ``"chat"`` alias) explicitly to recover the text-only listing.


        :return: A list of available models with their metadata, capabilities, and pricing information.


        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            List every model (text + image + embedding + tts + asr + music +
            upscale + inpaint + video + code)::

                models = await client.models.list()
                for model in models.data:
                    print(f"Model ID: {model.id}, Name: {model.name}")

            Filter models by type::

                chat_models = await client.models.list(type="chat")  # alias for "text"
                image_models = await client.models.list(type="image")
        """
        # Default to the union of every type. The server's own default is
        # ``text``-only with no filter — surprising for callers who expect
        # "no arg" to mean "everything". Sending ``type="all"`` matches the
        # docstring contract above and the spec's documented "use 'all' to
        # get all model types" hint.
        effective_type = type if type is not None else "all"

        # Create Pydantic query parameters model
        query_params = ModelsQueryParams(type=effective_type)

        # Convert to dictionary, excluding None values
        params = query_params.model_dump(exclude_none=True)

        result = await self._client.get(
            "models", params=params, cast_to=ModelsListResponse, force_direct=True
        )
        return result

    async def list_traits(
        self,
        *,
        type: str | None = None,
    ) -> ModelTraitsResponse:
        """
        Lists model traits and their associated model IDs asynchronously.

        Asynchronously retrieves a mapping of semantic trait names (e.g., "default",
        "fastest", "best") to their corresponding model IDs. Traits provide convenient
        shortcuts for selecting models based on desired characteristics rather than
        specific model identifiers, making it easier to choose appropriate models
        without needing to know exact model versions or IDs.

        :param type: Optional filter for model type. Only traits for models of the
            specified type will be returned. Valid values include ``"asr"``,
            ``"embedding"``, ``"image"``, ``"music"``, ``"text"``, ``"tts"``,
            ``"upscale"``, ``"inpaint"``, and ``"video"``.


        :return: A mapping of trait names to their corresponding model IDs.


        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            Get all model traits::

                traits = await client.models.list_traits()
                default_model = traits.data.get("default")
                fastest_model = traits.data.get("fastest")

            Get traits for specific model type::

                text_traits = await client.models.list_traits(type="text")
                print(f"Default text model: {text_traits.data['default']}")
        """
        # Create Pydantic query parameters model
        query_params = ModelTraitsQueryParams(type=type)

        # Convert to dictionary, excluding None values
        params = query_params.model_dump(exclude_none=True)

        result = await self._client.get(
            "models/traits",
            params=params,
            cast_to=ModelTraitsResponse,
            force_direct=True,
        )
        return result

    async def list_compatibility(
        self,
        *,
        type: str | None = None,
    ) -> ModelCompatibilityResponse:
        """
        Lists model compatibility mapping between external model names and Venice model IDs asynchronously.

        Asynchronously retrieves a mapping that allows applications to reference
        external model identifiers (e.g., from other AI platforms like OpenAI) and
        have them automatically mapped to equivalent Venice models. This compatibility
        layer facilitates smoother transitions when migrating applications from other
        AI platforms to Venice.

        :param type: Optional filter for model type. Only compatibility mappings for
            models of the specified type will be returned. Valid values include
            ``"asr"``, ``"embedding"``, ``"image"``, ``"music"``, ``"text"``,
            ``"tts"``, ``"upscale"``, ``"inpaint"``, ``"video"``, and ``"decision"``.
            Defaults to ``"text"`` per the API spec.


        :return: A mapping of external model names to their equivalent Venice model IDs.


        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            Get all compatibility mappings::

                compatibility = await client.models.list_compatibility()
                venice_model = compatibility.data.get("gpt-4")
                print(f"GPT-4 maps to Venice model: {venice_model}")

            Get compatibility for specific model type::

                text_compat = await client.models.list_compatibility(type="text")
                for external_name, venice_id in text_compat.data.items():
                    print(f"{external_name} -> {venice_id}")
        """
        # API spec defaults `type` to "text" — apply that default here so callers
        # who omit the param see the documented behaviour.
        query_params = ModelTraitsQueryParams(type=type)

        # Convert to dictionary, excluding None values
        params = query_params.model_dump(exclude_none=True)

        result = await self._client.get(
            "models/compatibility_mapping",
            params=params,
            cast_to=ModelCompatibilityResponse,
            force_direct=True,
        )
        return result

    # ------------------------------------------------------------------
    # Unified model resolution API
    # ------------------------------------------------------------------

    def __init__(self, client: VeniceClient) -> None:
        super().__init__(client)
        self._selector: DynamicModelSelector | None = None
        # 30-second TTL cache for the full model listing, shared by get()
        # and get_capabilities() so multiple lookups in a tight loop don't
        # each fetch /models from scratch.
        self._listing_cache: tuple[float, ModelsListResponse] | None = None

    def _get_selector(self) -> DynamicModelSelector:
        """Lazily initialize the underlying DynamicModelSelector."""
        if self._selector is None:
            from ..models.selection import DynamicModelSelector

            self._selector = DynamicModelSelector(self._client)
        return self._selector

    async def _cached_listing(self) -> ModelsListResponse:
        """Return a cached full-catalog :meth:`list` response, refreshing past TTL.

        Uses ``type="all"`` so :meth:`get` and :meth:`get_capabilities` can
        find any model id regardless of resource type. Per Venice API spec,
        ``type="all"`` returns the union of every model type.
        """
        now = time.monotonic()
        if self._listing_cache is not None:
            cached_at, listing = self._listing_cache
            if now - cached_at < _MODEL_LIST_CACHE_TTL_SECONDS:
                return listing
        listing = await self.list(type="all")
        self._listing_cache = (now, listing)
        return listing

    async def get(self, model_id: str) -> ModelResponse:
        """Fetch a single model entry by its id.

        Resolves against a 30-second TTL cache of :meth:`list` so back-to-back
        ``get()`` / :meth:`get_capabilities` calls don't each round-trip the
        full catalog. The Venice API has no per-model GET endpoint today;
        this method abstracts the list-and-filter pattern users would
        otherwise hand-roll.

        :param model_id: The id of the model to fetch (e.g.,
            ``"llama-3.3-70b"``).
        :raises ValueError: If no model with that id is found in the catalog.
        """
        listing = await self._cached_listing()
        for entry in listing.data:
            if entry.id == model_id:
                return entry
        raise ValueError(f"Model {model_id!r} not found in models.list()")

    async def get_capabilities(self, model_id: str) -> Capabilities:
        """Return a typed :class:`Capabilities` view of *model_id*.

        Polymorphic by model type — the result is one of
        :class:`ChatCapabilities`, :class:`ImageCapabilities`,
        :class:`VideoCapabilities`, :class:`InpaintCapabilities`, or
        :class:`GenericCapabilities`. Pattern-match on the result to access
        type-specific flags::

            caps = await client.models.get_capabilities(model_id)
            match caps:
                case ChatCapabilities(supports_function_calling=True):
                    ...
                case VideoCapabilities(supports_audio=True):
                    ...

        Eliminates the trial-and-error pattern of probing
        ``resolve_chat(require_function_calling=True)`` etc. Resolvers stay
        for ergonomic selection; this method exposes the same underlying
        flags for direct introspection.

        :param model_id: The id of the model to introspect.
        :raises ValueError: If no model with that id is in the catalog.
        """
        entry = await self.get(model_id)
        spec = entry.model_spec
        privacy = spec.privacy

        if entry.type == "text":
            # Narrow to TextModelSpec — ``availableContextTokens`` and
            # ``capabilities`` live on the text-specific subclass.
            # ``getattr`` keeps this resilient if a caller hands us a
            # base ModelSpec.
            ctx_tokens = getattr(spec, "availableContextTokens", None)
            context_window = int(ctx_tokens) if ctx_tokens is not None else None
            caps = getattr(spec, "capabilities", None)
            if caps is None:
                raise ValueError(
                    f"Model {model_id!r} is type='text' but has no capabilities payload."
                )
            return ChatCapabilities(
                context_window=context_window,
                supports_function_calling=caps.supportsFunctionCalling,
                supports_vision=caps.supportsVision,
                supports_reasoning=caps.supportsReasoning,
                supports_response_schema=caps.supportsResponseSchema,
                supports_web_search=caps.supportsWebSearch,
                supports_logprobs=caps.supportsLogProbs,
                supports_audio_input=caps.supportsAudioInput,
                supports_video_input=caps.supportsVideoInput,
                supports_multiple_images=caps.supportsMultipleImages,
                supports_reasoning_effort=caps.supportsReasoningEffort,
                supports_tee_attestation=caps.supportsTeeAttestation,
                supports_e2ee=caps.supportsE2EE,
                supports_x_search=caps.supportsXSearch,
                optimized_for_code=caps.optimizedForCode,
                quantization=caps.quantization,
                privacy=privacy,
            )

        if entry.type == "image":
            # ``spec`` is ImageModelSpec at runtime; getattr keeps mypy/pyright
            # happy without forcing an isinstance import here.
            constraints = getattr(spec, "constraints", None)
            supports_web_search = getattr(spec, "supportsWebSearch", None)
            if isinstance(constraints, ImageModelConstraints):
                return ImageCapabilities(
                    prompt_character_limit=int(constraints.promptCharacterLimit),
                    width_height_divisor=int(constraints.widthHeightDivisor),
                    supports_web_search=bool(supports_web_search),
                )
            return ImageCapabilities(supports_web_search=bool(supports_web_search))

        if entry.type == "video":
            constraints = getattr(spec, "constraints", None)
            if not isinstance(constraints, VideoModelConstraints):
                raise ValueError(
                    f"Model {model_id!r} is type='video' but has no video constraints."
                )
            return VideoCapabilities(
                model_type=constraints.model_type,
                supports_audio=constraints.audio,
                audio_configurable=constraints.audio_configurable,
                accepts_video_input=constraints.video_input,
                resolutions=builtins.list(constraints.resolutions),
                durations=builtins.list(constraints.durations),
                aspect_ratios=builtins.list(constraints.aspect_ratios),
            )

        if entry.type == "inpaint":
            constraints = getattr(spec, "constraints", None)
            if isinstance(constraints, InpaintModelConstraints):
                return InpaintCapabilities(
                    prompt_character_limit=int(constraints.promptCharacterLimit),
                    combine_images=constraints.combineImages,
                )
            return InpaintCapabilities()

        # Catch-all for embedding / tts / asr / music / upscale / decision,
        # plus any model type this release predates.
        return GenericCapabilities(type=entry.type, privacy=privacy)

    # NOTE: ``type`` deliberately shadows the builtin to give the public API a
    # short, ergonomic name. We use ``builtins.list`` etc. below where needed.
    # Renaming is a semver-major break — defer to v3.
    async def resolve(
        self,
        *,
        type: Literal[
            "chat", "embedding", "image", "video", "tts", "asr", "inpaint", "music", "decision"
        ] = "chat",
        # Chat capability filters
        require_function_calling: bool = False,
        require_vision: bool = False,
        require_reasoning: bool = False,
        require_code_optimization: bool = False,
        require_response_schema: bool = False,
        min_context_tokens: int | None = None,
        require_private: bool = False,
        exclude_beta: bool = True,
        # Video-specific
        video_type: Literal["text-to-video", "image-to-video"] | None = None,
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        # General
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Resolve a single model ID based on type and capability requirements.

        This is the unified entry point for model selection, replacing the
        multi-step ``create_model_selector()`` → ``selector.select_*()`` pattern.

        :param type: Model category to resolve. Defaults to ``"chat"``.
        :param require_function_calling: Only consider models with function calling support.
        :param require_vision: Only consider models with vision/image input support.
        :param require_reasoning: Only consider models with reasoning capabilities.
        :param require_code_optimization: Only consider code-optimized models.
        :param require_response_schema: Only consider models supporting structured output.
        :param min_context_tokens: Minimum context window size.
        :param require_private: Only consider privacy-first models.
        :param exclude_beta: Exclude beta models (default ``True``).
        :param video_type: Filter video models by ``"text-to-video"`` or ``"image-to-video"``.
        :param require_audio: Only consider video models with audio support.
        :param min_resolution: Minimum video resolution (e.g. ``"720p"``).
        :param min_duration: Minimum video duration (e.g. ``"5s"``).
        :param preferred_models: Preferred model IDs in priority order.
        :param exclude_models: Model IDs to exclude.
        :return: The resolved model ID string.
        :raises ValueError: If no model matches the given criteria.
        """
        selector = self._get_selector()
        exclude_set: set[str] | None = set(exclude_models) if exclude_models else None

        match type:
            case "chat":
                return await selector.select_chat_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    require_function_calling=require_function_calling,
                    require_vision=require_vision,
                    require_reasoning=require_reasoning,
                    require_code_optimization=require_code_optimization,
                    require_response_schema=require_response_schema,
                    min_context_tokens=min_context_tokens,
                    require_private=require_private,
                    exclude_beta=exclude_beta,
                )
            case "embedding":
                return await selector.select_embedding_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case "image":
                return await selector.select_image_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case "video":
                return await selector.select_video_model(
                    model_type=video_type,
                    require_audio=require_audio,
                    min_resolution=min_resolution,
                    min_duration=min_duration,
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    exclude_beta=exclude_beta,
                )
            case "tts":
                return await selector.select_audio_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case "asr":
                return await selector.select_asr_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case "inpaint":
                return await selector.select_inpaint_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case "music":
                return await selector.select_music_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case "decision":
                return await selector.select_decision_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                )
            case _:
                raise ValueError(f"Unknown model type: {type!r}")

    async def resolve_cheapest_video(
        self,
        *,
        duration: str = "5s",
        video_type: Literal["text-to-video", "image-to-video"] | None = None,
        resolution: str | None = None,
        audio: bool | None = None,
        aspect_ratio: str | None = None,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
    ) -> CheapestVideoResult:
        """Resolve the cheapest video model by quoting all candidates.

        Issues one ``POST /video/quote`` per candidate model and returns the
        model with the lowest USD quote.

        :param video_type: Filter by ``"text-to-video"`` or ``"image-to-video"``.
        :return: A :class:`CheapestVideoResult` with the cheapest model, price, and all quotes.
        """
        selector = self._get_selector()
        return await selector.select_cheapest_video_model(
            duration=duration,
            model_type=video_type,
            resolution=resolution,
            audio=audio,
            aspect_ratio=aspect_ratio,
            exclude_models=set(exclude_models) if exclude_models else None,
            exclude_beta=exclude_beta,
        )

    # ── Convenience shortcuts ──────────────────────────────────────────

    async def resolve_chat(
        self,
        *,
        require_function_calling: bool = False,
        require_vision: bool = False,
        require_reasoning: bool = False,
        require_code_optimization: bool = False,
        require_response_schema: bool = False,
        min_context_tokens: int | None = None,
        require_private: bool = False,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
    ) -> str:
        """Shortcut for ``resolve(type="chat", ...)``."""
        return await self.resolve(
            type="chat",
            require_function_calling=require_function_calling,
            require_vision=require_vision,
            require_reasoning=require_reasoning,
            require_code_optimization=require_code_optimization,
            require_response_schema=require_response_schema,
            min_context_tokens=min_context_tokens,
            require_private=require_private,
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            exclude_beta=exclude_beta,
        )

    async def resolve_embedding(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="embedding", ...)``."""
        return await self.resolve(
            type="embedding", preferred_models=preferred_models, exclude_models=exclude_models
        )

    async def resolve_image(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="image", ...)``."""
        return await self.resolve(
            type="image", preferred_models=preferred_models, exclude_models=exclude_models
        )

    async def resolve_video(
        self,
        *,
        video_type: Literal["text-to-video", "image-to-video"] | None = None,
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
    ) -> str:
        """Shortcut for ``resolve(type="video", ...)``.

        :param video_type: Optional filter — ``"text-to-video"`` or
            ``"image-to-video"``. Omit to consider any video model.
        :param require_audio: Only consider video models with audio support.
        :param min_resolution: Minimum video resolution (e.g. ``"720p"``).
        :param min_duration: Minimum video duration (e.g. ``"5s"``).
        :param preferred_models: Preferred model IDs in priority order.
        :param exclude_models: Model IDs to exclude.
        :param exclude_beta: Exclude beta models (default ``True``).
        """
        return await self.resolve(
            type="video",
            video_type=video_type,
            require_audio=require_audio,
            min_resolution=min_resolution,
            min_duration=min_duration,
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            exclude_beta=exclude_beta,
        )

    async def resolve_video_upscale(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Pick a video-upscaling model dynamically.

        ``models.list(type="upscale")`` returns the *image* upscaler, not video.
        Video upscalers are registered under ``type="video"`` with
        ``model_type="video"`` (rather than ``"text-to-video"`` /
        ``"image-to-video"``) and ``video_input=True``. This shortcut filters
        for that combination and returns the first match — typically
        ``topaz-video-upscale``.

        :param preferred_models: Preferred model IDs in priority order. The first
            preferred id present in the candidate set wins.
        :param exclude_models: Model IDs to exclude from selection.

        :return: Selected video-upscaling model ID.
        :raises ValueError: If no video-upscaling model is available.

        Example::

            async with VeniceClient() as client:
                model = await client.models.resolve_video_upscale()
                quote = await client.video.quote_upscale(
                    model=model, source_url=url, scale="2x"
                )
        """
        videos = await self.list(type="video")
        excluded = set(exclude_models or [])

        # Tier 1: explicit "upscale" in the model id (most reliable signal).
        # Tier 2: video-input models whose resolutions look like scaling factors
        # ("2x", "4x", ...). Topaz advertises this; transformation models like
        # ``wan-2-7-video-to-video`` advertise pixel resolutions instead.
        # Tier 3: any model_type="video" + video_input=True as a defensive fallback.
        tier1: builtins.list[str] = []
        tier2: builtins.list[str] = []
        tier3: builtins.list[str] = []

        for m in videos.data:
            if m.id in excluded:
                continue
            if "upscale" in m.id.lower():
                tier1.append(m.id)
                continue
            spec = getattr(m, "model_spec", None)
            constraints = getattr(spec, "constraints", None) if spec else None
            if not constraints:
                continue
            model_type = getattr(constraints, "model_type", None)
            video_input = getattr(constraints, "video_input", False)
            if model_type != "video" or not video_input:
                continue
            resolutions = list(getattr(constraints, "resolutions", []) or [])
            looks_like_scaling = any(
                isinstance(r, str)
                and r.lower().endswith("x")
                and r[:-1].replace(".", "", 1).isdigit()
                for r in resolutions
            )
            if looks_like_scaling:
                tier2.append(m.id)
            else:
                tier3.append(m.id)

        candidates = tier1 or tier2 or tier3

        if not candidates:
            raise ValueError(
                "No video-upscaling model available. Note that "
                "models.list(type='upscale') returns the IMAGE upscaler — "
                "video upscalers live under type='video'."
            )

        if preferred_models:
            for pref in preferred_models:
                if pref in candidates:
                    return pref
        return candidates[0]

    async def resolve_tts(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="tts", ...)``."""
        return await self.resolve(
            type="tts", preferred_models=preferred_models, exclude_models=exclude_models
        )

    async def resolve_asr(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="asr", ...)``."""
        return await self.resolve(
            type="asr", preferred_models=preferred_models, exclude_models=exclude_models
        )

    async def resolve_inpaint(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="inpaint", ...)``."""
        return await self.resolve(
            type="inpaint", preferred_models=preferred_models, exclude_models=exclude_models
        )

    async def resolve_music(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="music", ...)``."""
        return await self.resolve(
            type="music", preferred_models=preferred_models, exclude_models=exclude_models
        )

    async def resolve_voice_changer(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Pick a voice-changer model dynamically.

        Voice changing is **not** a distinct model type. Those models are
        registered under ``type="music"`` and are identified by
        ``voice_changer=True`` on their :class:`MusicModelSpec`, so
        ``resolve_music()`` may well hand back a music *generator* that the
        ``/audio/voice-changer/*`` endpoints reject. This shortcut filters on
        the capability flag instead.

        :param preferred_models: Preferred model IDs in priority order. The
            first preferred id present in the candidate set wins.
        :param exclude_models: Model IDs to exclude from selection.

        :return: Selected voice-changer model ID.
        :raises ValueError: If no voice-changer model is available to this
            account. The capability is not enabled everywhere, so this is an
            expected condition worth handling rather than a bug.

        Example::

            async with VeniceClient() as client:
                model = await client.models.resolve_voice_changer()
                job = await client.voice_changer.run(model=model, file="source.mp3")
        """
        music = await self.list(type="music")
        excluded = set(exclude_models or [])

        candidates = [
            entry.id
            for entry in music.data
            if entry.id not in excluded
            and isinstance(entry.model_spec, MusicModelSpec)
            and entry.model_spec.voice_changer
        ]

        if not candidates:
            raise ValueError(
                "No available voice-changer models found. Voice-changer models "
                "report type='music' with voice_changer=true; none in the current "
                "catalog does, so the capability is not enabled for this account."
            )

        for preferred in preferred_models or []:
            if preferred in candidates:
                return preferred
        return candidates[0]

    async def resolve_decision(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="decision", ...)``.

        Resolves a decision ("System One") model for
        :meth:`client.decisions.create() <venice_ai.resources.decisions.Decisions.create>`.
        Decision models are beta-flagged today, so unlike
        :meth:`resolve_chat` this shortcut does not filter them out.
        """
        return await self.resolve(
            type="decision", preferred_models=preferred_models, exclude_models=exclude_models
        )
