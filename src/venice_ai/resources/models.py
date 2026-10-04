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
from ..exceptions import NoMatchingModelError

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401
    from ..models.selection import (
        CheapestMusicResult,
        CheapestVideoResult,
        DynamicModelSelector,
        ModelSelectorType,
        VideoInputMode,
    )

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


# Ranking applied by ``resolve(prefer=...)``. ``None`` keeps the catalog's own
# ordering (Venice traits, then catalog order).
type ModelPreference = Literal["cheapest"]


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
                max_images=caps.maxImages,
                max_videos=caps.maxVideos,
                reasoning_effort_options=caps.reasoningEffortOptions,
                default_reasoning_effort=caps.defaultReasoningEffort,
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
        require_web_search: bool = False,
        require_reasoning_effort: str | None = None,
        require_prompt_caching: bool = False,
        require_multiple_images: bool = False,
        require_e2ee: bool = False,
        exclude_reasoning: bool = False,
        # Chat and image
        exclude_uncensored: bool = False,
        # Image-specific
        require_custom_size: bool = False,
        # Video-specific
        video_type: Literal["text-to-video", "image-to-video"] | None = None,
        input_mode: VideoInputMode | None = None,
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        require_duration: int | str | None = None,
        require_audio_configurable: bool = False,
        # Image / inpaint / video tiers
        require_quality: str | None = None,
        require_resolution: str | None = None,
        # Inpaint-specific
        require_combine_images: bool = False,
        require_uncensored: bool = False,
        # Music-specific
        exclude_non_music: bool = False,
        require_force_instrumental: bool = False,
        music_duration_seconds: int | None = None,
        # General
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        prefer: ModelPreference | None = None,
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
        :param require_web_search: Chat and image. Only consider models that support
            web search (``capabilities.supportsWebSearch`` for chat, the spec-level
            ``supportsWebSearch`` flag for image).
        :param require_reasoning_effort: Chat only. Only consider models whose
            ``reasoningEffortOptions`` include this value (e.g. ``"none"``).
        :param require_prompt_caching: Chat only. Only consider models whose
            catalog entry lists a cached-input price (``pricing.cache_input``).
            Venice exposes no prompt-caching capability flag; a listed price does
            not guarantee that a cache hit is served or reported.
        :param require_multiple_images: Chat only. Only consider models that accept
            several images in one request (``capabilities.supportsMultipleImages``).
        :param require_e2ee: Chat only. Only consider end-to-end encrypted models
            (``capabilities.supportsE2EE``), which need the client-side encryption flow.
        :param exclude_reasoning: Chat only. Only consider models that never reason
            (``capabilities.supportsReasoning`` is false), for callers that need
            visible ``content`` under a small token budget. Raises ``ValueError``
            with ``require_reasoning`` or ``require_reasoning_effort``. To keep
            reasoning-capable models and switch reasoning off per request, pass
            ``require_reasoning_effort="none"`` and send ``reasoning_effort="none"``.
        :param exclude_uncensored: Chat and image. Skip models Venice flags as
            ``uncensored``.
        :param require_custom_size: Image only. Only consider models sized by
            explicit ``width`` / ``height``: their constraints list no
            ``aspectRatios``. Models that list ``aspectRatios`` size by
            ``aspect_ratio`` (and ``resolution``) and reject or ignore pixel
            dimensions.
        :param video_type: Filter video models by ``"text-to-video"`` or
            ``"image-to-video"``. ``"image-to-video"`` means plain image-to-video
            models (one start image) unless ``input_mode`` says otherwise.
        :param input_mode: Video only. What an image-to-video model must take as
            input: ``"image"``, ``"reference"`` (reference-to-video),
            ``"first_last_frame"``, ``"transition"`` or ``"multi_angle"``. Venice
            types all of these ``image-to-video``; the SDK tells them apart by
            model id and name (see
            :func:`~venice_ai.models.selection.video_input_mode`). Implies
            ``video_type="image-to-video"``.
        :param require_audio: Only consider video models with audio support.
        :param min_resolution: Minimum video resolution (e.g. ``"720p"``).
        :param min_duration: Minimum video duration (e.g. ``"5s"``).
        :param require_duration: Video only. Exact duration the model must list
            (``5``, ``"5"`` or ``"5s"``); a model offering only longer clips is rejected.
        :param require_audio_configurable: Video only. Only consider models that
            accept an explicit ``audio=True/False`` (``audio_configurable``).
        :param require_quality: Image and inpaint. Quality tier the model must list
            in its ``qualities`` constraint (e.g. ``"high"``).
        :param require_resolution: Inpaint and video. Resolution tier the model
            must list in its ``resolutions`` constraint (e.g. ``"2K"``, ``"720p"``).
        :param require_combine_images: Inpaint only. Only consider models that can
            combine several input images (``combineImages``), as ``multi_edit`` needs.
        :param require_uncensored: Inpaint only. Only consider models Venice flags
            as ``uncensored``.
        :param exclude_non_music: Music only. Skip the text-to-speech, sound-effect
            and voice-changer models that Venice also types as ``music``.
        :param require_force_instrumental: Music only. Only consider models that
            accept ``force_instrumental`` (``supports_force_instrumental``).
        :param music_duration_seconds: Music only. The clip length wanted. Models
            that cannot make a clip this long are skipped, and with
            ``prefer="cheapest"`` each model is quoted at this length (see
            :meth:`resolve_cheapest_music`). ``None`` quotes each model at its
            shortest valid request.
        :param preferred_models: Preferred model IDs in priority order. With
            ``prefer="cheapest"`` the first preferred ID that passes the filters
            still wins over price.
        :param exclude_models: Model IDs to exclude.
        :param prefer: ``None`` (default) keeps the catalog's own ranking: Venice
            traits, then catalog order. For chat without ``require_reasoning``,
            models that do not reason are preferred over reasoning ones when
            any pass the filters (see
            :meth:`~venice_ai.models.selection.DynamicModelSelector.select_chat_model`).
            ``"cheapest"`` picks the lowest-priced model that passes every
            filter, in strict price order, by
            :func:`~venice_ai.models.selection.model_price`. Reasoning models
            compete on price like any other; pass ``exclude_reasoning=True`` to
            leave them out. Unpriced models sort last. Models at the same price
            are ordered by Venice's ``default`` trait first, then catalog order,
            then model ID. Chat is ranked on a blended 3:1 input:output token
            price; image and inpaint on one request at the model's default
            resolution and quality (or at ``require_quality`` when given).
            Music and video are ranked by free quote calls at each model's own
            valid request: music at ``music_duration_seconds`` (see
            :meth:`resolve_cheapest_music`), video at its cheapest settings (see
            :meth:`resolve_cheapest_video`). Music models whose catalog entry
            sets ``lyrics_required`` are skipped, since a request without lyrics
            would fail. Under ``"cheapest"``, beta models
            are skipped unless ``exclude_beta=False`` (decision models are exempt,
            since every one is beta), end-to-end encrypted chat models are
            skipped unless ``require_e2ee=True``, and music implies
            ``exclude_non_music=True``. Price order is only as accurate as the
            catalog: a model whose listed capability or price is wrong upstream
            can still be returned.
        :return: The resolved model ID string.
        :raises NoMatchingModelError: If no model matches the given criteria.
        :raises ModelQuotesUnavailableError: Video and music with
            ``prefer="cheapest"``: candidates match but every quote failed.
        :raises ValueError: For an unknown ``type`` or ``prefer`` value.
        """
        selector = self._get_selector()
        exclude_set: set[str] | None = set(exclude_models) if exclude_models else None
        strategy: ModelSelectorType | None = None
        if prefer is not None:
            if prefer != "cheapest":
                raise ValueError(f"Unknown prefer value {prefer!r}; expected 'cheapest' or None")
            if type == "music":
                return await self._resolve_cheapest_music_id(
                    duration_seconds=music_duration_seconds,
                    require_force_instrumental=require_force_instrumental,
                    preferred_models=preferred_models,
                    exclude_models=exclude_models,
                    exclude_beta=exclude_beta,
                )
            if type == "video":
                return await self._resolve_cheapest_video_id(
                    video_type=video_type,
                    input_mode=input_mode,
                    require_audio=require_audio,
                    min_resolution=min_resolution,
                    min_duration=min_duration,
                    require_duration=require_duration,
                    require_resolution=require_resolution,
                    require_audio_configurable=require_audio_configurable,
                    preferred_models=preferred_models,
                    exclude_models=exclude_models,
                    exclude_beta=exclude_beta,
                )
            from ..models.selection import cheapest_selector

            strategy = cheapest_selector(quality=require_quality, preferred_models=preferred_models)
            resource_type = "text" if type == "chat" else type
            skipped = await selector.cheapest_exclusions(
                resource_type,
                # Chat and video apply exclude_beta themselves.
                exclude_beta=exclude_beta and type not in ("chat", "video", "decision"),
                exclude_e2ee=type == "chat" and not require_e2ee,
            )
            if skipped:
                exclude_set = (exclude_set or set()) | skipped

        match type:
            case "chat":
                return await selector.select_chat_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                    require_function_calling=require_function_calling,
                    require_vision=require_vision,
                    require_reasoning=require_reasoning,
                    require_code_optimization=require_code_optimization,
                    require_response_schema=require_response_schema,
                    min_context_tokens=min_context_tokens,
                    require_private=require_private,
                    exclude_beta=exclude_beta,
                    require_web_search=require_web_search,
                    require_reasoning_effort=require_reasoning_effort,
                    require_prompt_caching=require_prompt_caching,
                    require_multiple_images=require_multiple_images,
                    require_e2ee=require_e2ee,
                    exclude_reasoning=exclude_reasoning,
                    exclude_uncensored=exclude_uncensored,
                )
            case "embedding":
                return await selector.select_embedding_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                )
            case "image":
                return await selector.select_image_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                    require_web_search=require_web_search,
                    require_quality=require_quality,
                    require_custom_size=require_custom_size,
                    exclude_uncensored=exclude_uncensored,
                )
            case "video":
                return await selector.select_video_model(
                    model_type=video_type,
                    input_mode=input_mode,
                    require_audio=require_audio,
                    min_resolution=min_resolution,
                    min_duration=min_duration,
                    require_duration=require_duration,
                    require_resolution=require_resolution,
                    require_audio_configurable=require_audio_configurable,
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                    exclude_beta=exclude_beta,
                )
            case "tts":
                return await selector.select_audio_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                )
            case "asr":
                return await selector.select_asr_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                )
            case "inpaint":
                return await selector.select_inpaint_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                    require_combine_images=require_combine_images,
                    require_quality=require_quality,
                    require_resolution=require_resolution,
                    require_uncensored=require_uncensored,
                )
            case "music":
                return await selector.select_music_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                    exclude_non_music=exclude_non_music,
                    require_force_instrumental=require_force_instrumental,
                    duration_seconds=music_duration_seconds,
                )
            case "decision":
                return await selector.select_decision_model(
                    preferred_models=preferred_models,
                    exclude_models=exclude_set,
                    selector=strategy,
                )
            case _:
                raise ValueError(f"Unknown model type: {type!r}")

    async def resolve_cheapest_video(
        self,
        *,
        duration: int | str | None = None,
        video_type: Literal["text-to-video", "image-to-video"] | None = None,
        resolution: str | None = None,
        audio: bool | None = False,
        aspect_ratio: str | None = None,
        require_audio: bool = False,
        require_audio_configurable: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
        max_concurrency: int = 8,
        input_mode: VideoInputMode | None = None,
    ) -> CheapestVideoResult:
        """Resolve the cheapest video model by quoting every candidate.

        Issues one free ``POST /video/quote`` per candidate (at most
        ``max_concurrency`` at once) and returns the lowest. This method adds no
        retries of its own; the client's retry policy still applies to each quote.
        Each model is quoted at its own cheapest valid request, as built by
        :func:`~venice_ai.models.selection.cheapest_video_params`: shortest
        listed duration, lowest listed resolution, 16:9 when listed, and
        ``audio=False`` where audio is configurable. Pass the result's
        ``request_params`` to ``client.video.submit`` to be billed that quote.

        :param duration: Pin the duration (``5``, ``"5"`` or ``"5s"``); models
            that do not list it are skipped. ``None`` uses each model's shortest.
        :param video_type: Filter by ``"text-to-video"`` or ``"image-to-video"``.
            ``None`` considers both (upscale / video-to-video models are skipped).
            ``"image-to-video"`` means plain image-to-video models; reference,
            transition, first/last-frame and multi-angle models are skipped
            unless ``input_mode`` asks for them.
        :param resolution: Pin the resolution tier; models that do not list it
            are skipped. ``None`` uses each model's lowest.
        :param audio: Sent to models with configurable audio. ``False`` (default)
            quotes the silent variant, ``True`` also requires audio support,
            ``None`` lets each model use its default.
        :param aspect_ratio: Aspect ratio to quote. ``None`` prefers 16:9.
        :param require_audio: Only consider models that support audio.
        :param require_audio_configurable: Only consider models that accept an
            explicit ``audio=True/False``.
        :param min_resolution: Only consider models offering at least this resolution.
        :param min_duration: Only consider models offering at least this duration.
        :param exclude_models: Model IDs to exclude.
        :param exclude_beta: Exclude beta models (default ``True``).
        :param max_concurrency: Maximum quote requests in flight at once.
        :param input_mode: What an image-to-video model must take as input; see
            :meth:`resolve`. Implies ``video_type="image-to-video"``.
        :return: A :class:`CheapestVideoResult` with the cheapest model, its
            quote and request parameters, every quote, and skipped candidates.
        :raises NoMatchingModelError: If no model passes the filters or can serve
            the requested settings. Nothing was quoted.
        :raises ModelQuotesUnavailableError: If candidates exist but every quote
            failed (a rate limit, a server error or a connection failure); ``failures``
            maps each model id to the exception its quote raised.
        """
        selector = self._get_selector()
        return await selector.select_cheapest_video_model(
            duration=duration,
            model_type=video_type,
            resolution=resolution,
            audio=audio,
            aspect_ratio=aspect_ratio,
            require_audio=require_audio,
            require_audio_configurable=require_audio_configurable,
            min_resolution=min_resolution,
            min_duration=min_duration,
            exclude_models=set(exclude_models) if exclude_models else None,
            exclude_beta=exclude_beta,
            max_concurrency=max_concurrency,
            input_mode=input_mode,
        )

    async def resolve_cheapest_music(
        self,
        *,
        duration_seconds: int | None = None,
        require_force_instrumental: bool = False,
        lyrics_supplied: bool = False,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
        max_concurrency: int = 8,
    ) -> CheapestMusicResult:
        """Resolve the cheapest music generator for a clip length by quoting each one.

        Issues one free ``POST /audio/quote`` per music generator (at most
        ``max_concurrency`` at once), each at the request
        :func:`~venice_ai.models.selection.music_request_params` builds for
        ``duration_seconds``: the smallest fixed option that covers it, the
        target raised to the model's minimum, or no duration for models that
        choose their own length. Models that cannot make a clip that long are
        skipped. Equal quotes go to the clip closest to the target length, then
        the ``default`` trait, catalog order and model id. Pass the result's
        ``request_params`` to ``client.music.submit`` to be billed the quote.

        :param duration_seconds: The clip length wanted. ``None`` quotes each
            model at its shortest valid request.
        :param require_force_instrumental: Only consider models that accept
            ``force_instrumental``.
        :param lyrics_supplied: Also consider models that require lyrics. Set it
            only when the submit call will pass ``lyrics_prompt``.
        :param exclude_models: Model IDs to exclude.
        :param exclude_beta: Exclude beta models (default ``True``).
        :param max_concurrency: Maximum quote requests in flight at once.
        :return: A :class:`~venice_ai.models.selection.CheapestMusicResult` with
            the model, its quote, request parameters and clip length, every
            quote, and skipped candidates.
        :raises NoMatchingModelError: If no generator passes the filters or can
            make a clip of ``duration_seconds``. Nothing was quoted.
        :raises ModelQuotesUnavailableError: If candidates exist but every quote
            failed (a rate limit, a server error or a connection failure); ``failures``
            maps each model id to the exception its quote raised.
        """
        selector = self._get_selector()
        return await selector.select_cheapest_music_model(
            duration_seconds=duration_seconds,
            require_force_instrumental=require_force_instrumental,
            lyrics_supplied=lyrics_supplied,
            exclude_models=set(exclude_models) if exclude_models else None,
            exclude_beta=exclude_beta,
            max_concurrency=max_concurrency,
        )

    async def _resolve_cheapest_music_id(
        self,
        *,
        duration_seconds: int | None,
        require_force_instrumental: bool,
        preferred_models: builtins.list[str] | None,
        exclude_models: builtins.list[str] | None,
        exclude_beta: bool,
    ) -> str:
        """``resolve(type="music", prefer="cheapest")``: rank generators by quotes."""
        selector = self._get_selector()
        if preferred_models:
            picked = await selector.select_music_model(
                preferred_models=preferred_models,
                exclude_models=set(exclude_models) if exclude_models else None,
                exclude_non_music=True,
                require_force_instrumental=require_force_instrumental,
                duration_seconds=duration_seconds,
            )
            if picked in preferred_models:
                return picked
        result = await self.resolve_cheapest_music(
            duration_seconds=duration_seconds,
            require_force_instrumental=require_force_instrumental,
            exclude_models=exclude_models,
            exclude_beta=exclude_beta,
        )
        return result.model

    async def _resolve_cheapest_video_id(
        self,
        *,
        video_type: Literal["text-to-video", "image-to-video"] | None,
        input_mode: VideoInputMode | None,
        require_audio: bool,
        min_resolution: str | None,
        min_duration: str | None,
        require_duration: int | str | None,
        require_resolution: str | None,
        require_audio_configurable: bool,
        preferred_models: builtins.list[str] | None,
        exclude_models: builtins.list[str] | None,
        exclude_beta: bool,
    ) -> str:
        """``resolve(type="video", prefer="cheapest")``: rank the filtered pool by quotes."""
        selector = self._get_selector()
        exclude_set = set(exclude_models) if exclude_models else None
        if preferred_models:
            # A preferred model that passes the filters wins without quoting.
            picked = await selector.select_video_model(
                model_type=video_type,
                input_mode=input_mode,
                require_audio=require_audio,
                min_resolution=min_resolution,
                min_duration=min_duration,
                require_duration=require_duration,
                require_resolution=require_resolution,
                require_audio_configurable=require_audio_configurable,
                preferred_models=preferred_models,
                exclude_models=exclude_set,
                exclude_beta=exclude_beta,
            )
            if picked in preferred_models:
                return picked
        result = await selector.select_cheapest_video_model(
            duration=require_duration,
            model_type=video_type,
            input_mode=input_mode,
            resolution=require_resolution,
            audio=require_audio,
            require_audio=require_audio,
            require_audio_configurable=require_audio_configurable,
            min_resolution=min_resolution,
            min_duration=min_duration,
            exclude_models=exclude_set,
            exclude_beta=exclude_beta,
        )
        return result.model

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
        require_web_search: bool = False,
        require_reasoning_effort: str | None = None,
        require_prompt_caching: bool = False,
        require_multiple_images: bool = False,
        require_e2ee: bool = False,
        exclude_reasoning: bool = False,
        exclude_uncensored: bool = False,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="chat", ...)``.

        :param require_web_search: Only consider models that support web search.
        :param require_reasoning_effort: Only consider models whose
            ``reasoningEffortOptions`` include this value. ``"none"`` selects
            reasoning models that can run with reasoning switched off; send
            ``reasoning_effort="none"`` with the request to do so.
        :param require_prompt_caching: Only consider models whose catalog entry
            lists a cached-input price (``pricing.cache_input``). Venice exposes
            no prompt-caching capability flag; a listed price does not guarantee
            that a cache hit is served or reported.
        :param require_multiple_images: Only consider models that accept several
            images in one request.
        :param require_e2ee: Only consider end-to-end encrypted models.
        :param exclude_reasoning: Only consider models that never reason; see
            :meth:`resolve`.
        :param exclude_uncensored: Skip models Venice flags as ``uncensored``.
        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            in strict price order instead of the catalog's default ranking;
            see :meth:`resolve`.
        """
        return await self.resolve(
            type="chat",
            exclude_reasoning=exclude_reasoning,
            exclude_uncensored=exclude_uncensored,
            require_web_search=require_web_search,
            require_reasoning_effort=require_reasoning_effort,
            require_prompt_caching=require_prompt_caching,
            require_multiple_images=require_multiple_images,
            require_e2ee=require_e2ee,
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
            prefer=prefer,
        )

    async def resolve_embedding(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="embedding", ...)``.

        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="embedding",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
        )

    async def resolve_image(
        self,
        *,
        require_web_search: bool = False,
        require_quality: str | None = None,
        require_custom_size: bool = False,
        exclude_uncensored: bool = False,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="image", ...)``.

        :param require_web_search: Only consider models that honor
            ``enable_web_search`` (spec-level ``supportsWebSearch``).
        :param require_quality: Quality tier the model must list in its
            ``qualities`` constraint (e.g. ``"high"``; case-insensitive).
        :param require_custom_size: Only consider models sized by explicit
            ``width`` / ``height`` (no ``aspectRatios`` constraint); see
            :meth:`resolve`.
        :param exclude_uncensored: Skip models Venice flags as ``uncensored``.
        :param preferred_models: Preferred model IDs in priority order.
        :param exclude_models: Model IDs to exclude.
        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="image",
            require_web_search=require_web_search,
            require_quality=require_quality,
            require_custom_size=require_custom_size,
            exclude_uncensored=exclude_uncensored,
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
        )

    async def resolve_video(
        self,
        *,
        video_type: Literal["text-to-video", "image-to-video"] | None = None,
        input_mode: VideoInputMode | None = None,
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        require_duration: int | str | None = None,
        require_resolution: str | None = None,
        require_audio_configurable: bool = False,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        exclude_beta: bool = True,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="video", ...)``.

        :param video_type: Optional filter — ``"text-to-video"`` or
            ``"image-to-video"`` (plain image-to-video unless ``input_mode`` says
            otherwise). Omit to consider any video model.
        :param input_mode: What an image-to-video model must take as input
            (``"reference"``, ``"transition"``, ...); see :meth:`resolve`.
        :param require_audio: Only consider video models with audio support.
        :param min_resolution: Minimum video resolution (e.g. ``"720p"``).
        :param min_duration: Minimum video duration (e.g. ``"5s"``).
        :param require_duration: Exact duration the model must list (``5``,
            ``"5"`` or ``"5s"``); a model offering only longer clips is rejected.
        :param require_resolution: Exact resolution tier the model must list
            (e.g. ``"720p"``; case-insensitive).
        :param require_audio_configurable: Only consider models that accept an
            explicit ``audio=True/False`` (``audio_configurable``).
        :param preferred_models: Preferred model IDs in priority order.
        :param exclude_models: Model IDs to exclude.
        :param exclude_beta: Exclude beta models (default ``True``).
        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="video",
            video_type=video_type,
            input_mode=input_mode,
            require_audio=require_audio,
            min_resolution=min_resolution,
            min_duration=min_duration,
            require_duration=require_duration,
            require_resolution=require_resolution,
            require_audio_configurable=require_audio_configurable,
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            exclude_beta=exclude_beta,
            prefer=prefer,
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
        :raises NoMatchingModelError: If no video-upscaling model is available.

        Example::

            async with VeniceClient() as client:
                model = await client.models.resolve_video_upscale()
                quote = await client.video.quote(
                    model=model,
                    video_url=url,
                    upscale_factor=2,
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
            raise NoMatchingModelError(
                "No video-upscaling model available. Note that "
                "models.list(type='upscale') returns the IMAGE upscaler — "
                "video upscalers live under type='video'.",
                resource_type="video_upscale",
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
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="tts", ...)``.

        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="tts",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
        )

    async def resolve_asr(
        self,
        *,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="asr", ...)``.

        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="asr",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
        )

    async def resolve_inpaint(
        self,
        *,
        require_combine_images: bool = False,
        require_quality: str | None = None,
        require_resolution: str | None = None,
        require_uncensored: bool = False,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="inpaint", ...)``.

        :param require_combine_images: Only consider models that can combine
            several input images (``combineImages``), as ``multi_edit`` needs.
        :param require_quality: Quality tier the model must list in its
            ``qualities`` constraint (e.g. ``"low"``; case-insensitive).
        :param require_resolution: Resolution tier the model must list in its
            ``resolutions`` constraint (e.g. ``"2K"``). Models without that
            constraint reject a ``resolution`` argument.
        :param require_uncensored: Only consider models Venice flags as
            ``uncensored``.
        :param preferred_models: Preferred model IDs in priority order.
        :param exclude_models: Model IDs to exclude.
        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="inpaint",
            require_combine_images=require_combine_images,
            require_quality=require_quality,
            require_resolution=require_resolution,
            require_uncensored=require_uncensored,
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
        )

    async def resolve_music(
        self,
        *,
        exclude_non_music: bool = False,
        require_force_instrumental: bool = False,
        duration_seconds: int | None = None,
        preferred_models: builtins.list[str] | None = None,
        exclude_models: builtins.list[str] | None = None,
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="music", ...)``.

        :param exclude_non_music: Skip the text-to-speech, sound-effect and
            voice-changer models that Venice also types as ``music``.
        :param require_force_instrumental: Only consider models that accept
            ``force_instrumental``.
        :param duration_seconds: The clip length wanted; models that cannot
            make it are skipped, and ``prefer="cheapest"`` quotes each model at
            it. Use :meth:`resolve_cheapest_music` to also get the request
            parameters and quote.
        :param preferred_models: Preferred model IDs in priority order.
        :param exclude_models: Model IDs to exclude.
        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="music",
            exclude_non_music=exclude_non_music,
            require_force_instrumental=require_force_instrumental,
            music_duration_seconds=duration_seconds,
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
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
        :raises NoMatchingModelError: If no voice-changer model is available to this
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
            raise NoMatchingModelError(
                "No available voice-changer models found. Voice-changer models "
                "report type='music' with voice_changer=true; none in the current "
                "catalog does, so the capability is not enabled for this account.",
                resource_type="voice_changer",
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
        prefer: ModelPreference | None = None,
    ) -> str:
        """Shortcut for ``resolve(type="decision", ...)``.

        Resolves a decision ("System One") model for
        :meth:`client.decisions.create() <venice_ai.resources.decisions.Decisions.create>`.
        Decision models are beta-flagged today, so unlike
        :meth:`resolve_chat` this shortcut does not filter them out.
        :param prefer: ``"cheapest"`` picks the lowest-priced matching model
            instead of the catalog's default ranking; see :meth:`resolve`.
        """
        return await self.resolve(
            type="decision",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            prefer=prefer,
        )
