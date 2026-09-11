"""
Models API response models for Venice AI.

This module contains Pydantic models for the models API endpoints,
including model listings, traits, compatibility, and specifications.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...core.models.common import VeniceBaseModel
from ..identifiers import ModelId


class ModelCapabilities(BaseModel):
    """Text model capabilities.

    All capability fields are validated against live API responses.
    API returns these fields: optimizedForCode, quantization, supportsAudioInput,
    supportsE2EE, supportsFunctionCalling, supportsLogProbs, supportsMultipleImages,
    supportsReasoning, supportsReasoningEffort, supportsResponseSchema,
    supportsTeeAttestation, supportsVideoInput, supportsVision, supportsWebSearch,
    supportsXSearch.

    ``extra='allow'`` so live capability keys the SDK doesn't yet model (e.g.
    ``maxImages`` on multi-image vision models, or any future ``supports*``
    bool) land on ``model_extra`` and survive ``model_dump()`` instead of being
    silently dropped — the outer ``ModelSpec`` allow-policy does not recurse
    into this nested object.
    """

    model_config = ConfigDict(extra="allow")

    optimizedForCode: bool = Field(..., description="Is the LLM optimized for coding?")
    quantization: str = Field(
        ...,
        description=(
            "The quantization type of the running model. Known live values: "
            "fp4, fp8, fp16, bf16, int4, int8, not-available. Typed as a plain "
            "str (not a Literal) so a new server-side quantization value never "
            "crashes the whole GET /models parse."
        ),
    )
    supportsFunctionCalling: bool = Field(
        ..., description="Does the LLM model support function calling?"
    )
    supportsReasoning: bool = Field(
        ..., description="Does the model support reasoning with <thinking> blocks?"
    )
    supportsResponseSchema: bool = Field(
        ..., description="Does the LLM model support response schema?"
    )
    supportsVision: bool = Field(..., description="Does the LLM support vision?")
    supportsWebSearch: bool = Field(..., description="Does the LLM model support web search?")
    supportsLogProbs: bool = Field(..., description="Does the LLM model support logprobs?")
    # ADDITIONS validated from API
    supportsAudioInput: bool = Field(
        default=False, description="Does the model support audio input?"
    )
    supportsVideoInput: bool = Field(
        default=False, description="Does the model support video input?"
    )
    supportsMultipleImages: bool = Field(
        default=False,
        description=(
            "Does the vision model preserve images across all messages in the "
            "conversation history? Single-image models only retain the last "
            "image-containing message."
        ),
    )
    supportsReasoningEffort: bool = Field(
        default=False,
        description=("Does the model accept the OpenAI-compatible reasoning_effort parameter?"),
    )
    reasoningEffortOptions: list[str] | None = Field(
        default=None,
        description=(
            "Supported reasoning_effort values for this model. Only present when "
            "supportsReasoningEffort is true. 'none' means reasoning can be disabled. "
            "Open str (not a Literal) so a new server-side tier never crashes parsing."
        ),
    )
    defaultReasoningEffort: str | None = Field(
        default=None,
        description=(
            "Default reasoning_effort value used when the request omits one. Only "
            "present when supportsReasoningEffort is true."
        ),
    )
    supportsTeeAttestation: bool = Field(
        default=False,
        description="Does the model run inside a TEE with remote attestation support?",
    )
    supportsE2EE: bool = Field(
        default=False,
        description=(
            "Does the model support end-to-end encrypted inference? Requires "
            "supportsTeeAttestation to also be true."
        ),
    )
    supportsStyleReferences: bool | None = Field(
        default=None,
        description="Whether the image model accepts ``style_references``.",
    )
    supportsStyleReferenceStrength: bool | None = Field(
        default=None,
        description=(
            "Whether per-reference ``strength`` is honoured; when false the "
            "strength is ignored rather than rejected."
        ),
    )
    supportsXSearch: bool = Field(
        default=False,
        description=(
            "Does the model support xAI's native search via the "
            "venice_parameters.enable_x_search flag?"
        ),
    )


class TemperatureConstraint(BaseModel):
    """Temperature constraint specification"""

    default: float = Field(..., description="Default temperature value for the model")


class TopPConstraint(BaseModel):
    """Top-p constraint specification"""

    default: float = Field(..., description="Default top_p value for the model")


class StepsConstraint(BaseModel):
    """Steps constraint for image models"""

    default: float = Field(..., description="Default steps value for the model")
    max: float = Field(..., description="Maximum supported steps value for the model")


class TextModelConstraints(BaseModel):
    """Constraints for text models.

    ``extra='allow'`` preserves any further live constraint keys not yet modeled
    on ``model_extra``.
    """

    model_config = ConfigDict(extra="allow")

    temperature: TemperatureConstraint = Field(
        ...,
        description="Temperature parameter constraints for controlling output randomness",
    )
    top_p: TopPConstraint = Field(
        ...,
        description="Top-p (nucleus sampling) parameter constraints for token selection",
    )
    frequency_penalty: TemperatureConstraint | None = Field(
        default=None,
        description="Frequency penalty parameter constraints for the model",
    )
    presence_penalty: TemperatureConstraint | None = Field(
        default=None,
        description="Presence penalty parameter constraints for the model",
    )
    repetition_penalty: TemperatureConstraint | None = Field(
        default=None,
        description="Repetition penalty parameter constraints for the model",
    )


class ImageModelConstraints(BaseModel):
    """Constraints for image models.

    ``extra='allow'`` preserves the documented per-model ``aspectRatios`` /
    ``resolutions`` / ``defaultResolution`` / ``defaultAspectRatio`` discovery
    keys (which the docs tell callers to read for the ``aspect_ratio`` feature)
    on ``model_extra`` instead of dropping them.
    """

    model_config = ConfigDict(extra="allow")

    promptCharacterLimit: float = Field(..., description="The maximum supported prompt length")
    steps: StepsConstraint = Field(
        ...,
        description="Inference steps constraints controlling image generation quality and speed",
    )
    widthHeightDivisor: float = Field(
        ..., description="Width and height must be divisible by this value"
    )
    defaultQuality: str | None = Field(
        default=None,
        description=(
            "Default quality for this model. Present only for quality-aware models. "
            "Open str (not a Literal) so a new server-side quality never crashes parsing."
        ),
    )
    qualities: list[str] | None = Field(
        default=None,
        description="Supported quality options (open str). Present only for quality-aware models.",
    )


class InpaintModelConstraints(BaseModel):
    """Constraints for inpaint models.

    Inpaint models have different constraints than standard image models,
    focusing on prompt limits and image combination capabilities.

    ``extra='allow'`` preserves live keys not yet modeled (e.g. ``aspectRatios``
    / ``resolutions`` / ``defaultResolution`` / ``singleImageAspectRatio``) on
    ``model_extra``.
    """

    model_config = ConfigDict(extra="allow")

    promptCharacterLimit: float = Field(..., description="The maximum supported prompt length")
    combineImages: bool = Field(
        default=False, description="Whether the model can combine multiple images"
    )


class VideoModelConstraints(BaseModel):
    """
    Constraints for video generation models.

    Defines the supported configurations for a video model including
    aspect ratios, resolutions, durations, and audio capabilities.

    Note: Empty lists indicate the value is determined by model/input.
    For example, image-to-video models often have empty aspect_ratios
    because they use the input image's aspect ratio.

    The swagger marks these fields required, but the SDK relaxes them to
    optional with safe defaults because live image-to-video models may omit
    aspect_ratios (and other fields) when they use the source image's
    dimensions.

    Attributes:
        model_type: Whether this is a text-to-video, image-to-video, or generic video model
        aspect_ratios: Supported aspect ratios (may be empty for I2V)
        resolutions: Supported resolutions (may be empty for some models)
        durations: Supported video durations
        audio: Whether the model supports audio generation
        audio_configurable: Whether audio can be toggled on/off
        video_input: Whether model accepts video input

    ``extra='allow'`` preserves live constraint keys not yet modeled (e.g.
    ``prompt_character_limit``, a real swagger field the class does not
    explicitly declare, and any other server-side keys observed on video
    models) on ``model_extra`` instead of dropping them.
    """

    model_config = ConfigDict(extra="allow")

    model_type: str = Field(
        ...,
        description=(
            "Whether model generates from text, image, or video input "
            "(e.g. 'text-to-video', 'image-to-video', 'video'). Open str so a new "
            "server-side model_type never crashes the /models parse."
        ),
    )
    aspect_ratios: list[str] = Field(
        default_factory=list,
        description="Supported aspect ratios (empty = uses input/model default)",
    )
    resolutions: list[str] = Field(
        default_factory=list,
        description="Supported resolutions (empty = uses model default)",
    )
    durations: list[str] = Field(
        default_factory=list, description="Supported durations e.g. ['5s', '10s']"
    )
    audio: bool = Field(default=False, description="Whether model supports audio generation")
    audio_configurable: bool = Field(
        default=False, description="Whether audio can be toggled via request parameter"
    )
    video_input: bool = Field(default=False, description="Whether model accepts video input")


class PricingTier(BaseModel):
    """Pricing tier for a currency.

    Subscript access (``tier["usd"]``) is supported alongside attribute
    access (``tier.usd``) so callers building rows from
    :meth:`pydantic.BaseModel.model_dump` — which renders this model as a
    plain ``dict`` — can transparently swap to instance access without
    rewriting the lookup.

    ``extra="allow"`` keeps a future server-side currency on ``model_extra``
    rather than dropping it, matching every enclosing pricing model.
    """

    model_config = ConfigDict(extra="allow")

    usd: float = Field(..., description="USD cost")
    diem: float = Field(..., description="Diem cost")

    def __getitem__(self, key: str) -> float:
        try:
            value: float = getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc
        return value


class ExtendedPricing(BaseModel):
    """Extended (long-context) pricing tier.

    Applies when input tokens exceed ``context_token_threshold``; the extended
    rates then apply to the entire request. Present only on models that publish
    long-context pricing. All fields are optional so a partial server payload
    never fails to parse.
    """

    model_config = ConfigDict(extra="allow")

    context_token_threshold: float | None = Field(
        default=None, description="Input token count above which extended pricing applies"
    )
    input: PricingTier | None = Field(default=None, description="Extended input token pricing")
    output: PricingTier | None = Field(default=None, description="Extended output token pricing")
    cache_input: PricingTier | None = Field(
        default=None, description="Extended cached input token pricing"
    )
    cache_write: PricingTier | None = Field(
        default=None, description="Extended cache write token pricing"
    )


class LLMModelPricing(BaseModel):
    """
    Token-based pricing for chat models.

    The cache_input field is present on select models that support
    prompt caching (e.g., claude-opus-45, grok-41-fast). Some providers (e.g.
    Anthropic) also charge for ``cache_write`` (cache creation) at a premium.

    ``extra="allow"`` keeps unknown live pricing keys on
    :attr:`pydantic.BaseModel.model_extra` (and in ``model_dump``) rather than
    silently dropping them.
    """

    model_config = ConfigDict(extra="allow")

    input: PricingTier = Field(..., description="Input token pricing")
    output: PricingTier = Field(..., description="Output token pricing")
    cache_input: PricingTier | None = Field(
        default=None, description="Cached/prompt-cached input token pricing"
    )
    cache_write: PricingTier | None = Field(
        default=None,
        description="Cache creation token pricing (e.g. Anthropic charges ~1.25x input)",
    )
    extended: ExtendedPricing | None = Field(
        default=None,
        description="Extended pricing for requests exceeding context_token_threshold",
    )


class UpscalePricing(BaseModel):
    """Upscaling pricing tiers.

    ``extra="allow"`` keeps an unknown upscale factor (say a future "8x") on
    ``model_extra`` rather than dropping it, matching the sibling pricing models.

    Note:
        Field aliases ("2x", "4x") are used because Python identifiers cannot
        start with numbers. Access via attribute names (x2, x4) in code, but
        JSON serialization uses the numeric aliases for API compatibility.
    """

    model_config = ConfigDict(extra="allow")

    x2: PricingTier = Field(..., alias="2x", description="2x upscale pricing")
    x4: PricingTier = Field(..., alias="4x", description="4x upscale pricing")


class ImageModelPricing(BaseModel):
    """Pricing for image generation and upscaling.

    ``extra="allow"`` keeps unknown live pricing keys (e.g. quality matrices)
    on :attr:`pydantic.BaseModel.model_extra` (and in ``model_dump``) rather
    than silently dropping them.
    """

    model_config = ConfigDict(extra="allow")

    generation: PricingTier | None = Field(
        default=None,
        description=(
            "Image generation pricing. Optional — swagger marks only `upscale` "
            "required, so upscale-only models omit it."
        ),
    )
    upscale: UpscalePricing = Field(..., description="Upscaling pricing")


class AudioModelPricing(BaseModel):
    """Pricing for audio models (TTS).

    ``extra="allow"`` keeps unknown live pricing keys on
    :attr:`pydantic.BaseModel.model_extra` (and in ``model_dump``) rather than
    silently dropping them.
    """

    model_config = ConfigDict(extra="allow")

    input: PricingTier = Field(..., description="Input character pricing")


class VideoResolutionPricing(BaseModel):
    """Resolution-based pricing for video models.

    Video models have different pricing tiers based on output resolution.
    Common resolutions include 1K (1024p) and 2K (2048p).

    Resolution-priced *image* models (e.g. ``gpt-image-2``,
    ``nano-banana-pro``) reuse this shape but add ``quality`` and/or
    ``upscale`` keys. ``extra="allow"`` keeps those siblings on
    :attr:`pydantic.BaseModel.model_extra` (and in ``model_dump``) instead of
    silently dropping them when the undiscriminated pricing union matches
    here on the required ``resolutions`` key alone.

    Note: because :data:`ModelPricing` is an
    *undiscriminated* union, any pricing payload carrying a ``resolutions``
    key — including inpaint models that surface one — resolves to this class
    rather than a more specific sibling. This isinstance-mislabel is an
    accepted low-severity tradeoff (converting the union to a discriminated
    union is out of scope); ``extra="allow"`` on every sibling means no keys
    are lost regardless of which member the union picks.
    """

    model_config = ConfigDict(extra="allow")

    resolutions: dict[str, PricingTier] = Field(
        ..., description="Pricing per resolution tier (e.g., '1K', '2K')"
    )


class InpaintModelPricing(BaseModel):
    """Pricing for inpaint models.

    Inpaint operations have flat per-operation pricing.

    ``extra="allow"`` keeps unknown live pricing keys on
    :attr:`pydantic.BaseModel.model_extra` (and in ``model_dump``) rather than
    silently dropping them.
    """

    model_config = ConfigDict(extra="allow")

    inpaint: PricingTier = Field(..., description="Per inpaint operation pricing")


class ASRModelPricing(BaseModel):
    """Pricing for ASR (Automatic Speech Recognition) models.

    ASR models are priced per second of audio processed.

    ``extra="allow"`` keeps unknown live pricing keys on
    :attr:`pydantic.BaseModel.model_extra` (and in ``model_dump``) rather than
    silently dropping them.
    """

    model_config = ConfigDict(extra="allow")

    per_audio_second: PricingTier = Field(..., description="Per audio second pricing")


class MusicModelPricing(BaseModel):
    """Pricing for music generation models.

    Music models support diverse pricing structures depending on the model:
    - Duration-based: tiers keyed by seconds (e.g., "60", "120") with per-tier pricing
    - Per-second: flat rate per audio second
    - Per-thousand-characters: for TTS-style music models
    - Generation: flat per-generation rate

    Each music model uses exactly one of these pricing structures.
    All fields are optional so that the union discriminator can fall through
    to this catch-all type after more specific pricing models fail to match.

    ``extra="allow"`` preserves any unknown live pricing keys in
    ``model_extra`` rather than silently dropping them, consistent with the
    sibling pricing models.
    """

    model_config = ConfigDict(extra="allow")

    durations: dict[str, Any] | None = Field(
        default=None, description="Duration-based pricing tiers keyed by seconds"
    )
    per_second: PricingTier | None = Field(default=None, description="Per-second audio pricing")
    per_thousand_characters: PricingTier | None = Field(
        default=None, description="Per-1000-characters pricing (TTS-style music models)"
    )
    generation: PricingTier | None = Field(default=None, description="Flat per-generation pricing")


# Union of all constraint types
ModelConstraints = (
    TextModelConstraints | ImageModelConstraints | VideoModelConstraints | InpaintModelConstraints
)

# Union of all pricing types
ModelPricing = (
    LLMModelPricing
    | ImageModelPricing
    | AudioModelPricing
    | VideoResolutionPricing
    | InpaintModelPricing
    | ASRModelPricing
    | MusicModelPricing
)


class ModelDeprecation(BaseModel):
    """Deprecation information for a model.

    Mirrors the swagger ``deprecation`` object. ``date`` is the legacy field
    (aligned with the ``x-venice-model-deprecation-date`` header); ``startsAt`` /
    ``removesAt`` are the preferred lifecycle instants for new integrations.
    """

    model_config = ConfigDict(extra="allow")

    autoRemap: bool = Field(
        default=False,
        description=(
            "When true, Venice may auto-remap requests for this model ID to "
            "replacementModelId instead of erroring."
        ),
    )
    date: str | None = Field(
        default=None,
        description="Legacy ISO 8601 deprecation sunset instant. Prefer startsAt / removesAt.",
    )
    removesAt: str | None = Field(
        default=None,
        description="ISO 8601 instant when this model ID is omitted from public GET /models.",
    )
    replacementModelId: str | None = Field(
        default=None,
        description="Suggested public API model ID to migrate to, when one exists.",
    )
    startsAt: str | None = Field(
        default=None,
        description="ISO 8601 instant when deprecation warnings should be considered active.",
    )


class ModelSpec(BaseModel):
    """Base model spec — fields common to all model types.

    Subclassed by per-type specs (:class:`TextModelSpec`, :class:`ImageModelSpec`,
    :class:`VideoModelSpec`, :class:`InpaintModelSpec`, :class:`MusicModelSpec`,
    :class:`TtsModelSpec`, :class:`AsrModelSpec`, :class:`EmbeddingModelSpec`,
    :class:`UpscaleModelSpec`). :class:`ModelResponse` dispatches to the right
    subclass via a ``model_validator`` based on the parent ``type`` field.

    ``extra='allow'`` so future Venice fields are preserved on
    :attr:`pydantic.BaseModel.model_extra` rather than silently dropped.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # Always present (5/9 model types) — verified against live ``/models`` for
    # all types as of 2026-04-28.
    name: str | None = Field(
        default=None, description="The name of the model (swagger does not require it)."
    )
    offline: bool = Field(default=False, description="Is this model presently offline?")
    privacy: Literal["private", "anonymized"] | None = Field(
        default=None,
        description=(
            "Data privacy level. 'private' = no data stored, 'anonymized' = "
            "provider stores anonymized data."
        ),
    )
    traits: list[str] = Field(default_factory=list, description="Model traits")
    uncensored: bool = Field(
        default=False,
        description=(
            "Whether Venice classifies this model as uncensored, meaning it applies "
            "minimal content-based filtering. Covers every modality. The API sends "
            "this field only when it is true and omits it otherwise, so an absent "
            "field is a definite 'not uncensored' — unlike the ``supports_*`` "
            "capability flags elsewhere in this module, where ``None`` means "
            "undeclared. Upstream providers may still apply their own filtering."
        ),
    )

    # Optional metadata that any type may grow.
    modelSource: str | None = Field(default=None, description="The source of the model")
    pricing: ModelPricing | None = Field(default=None, description="Pricing details for the model")
    beta: bool = Field(default=False, alias="betaModel", description="Is this model in beta?")
    deprecation: ModelDeprecation | None = Field(
        default=None, description="Deprecation information, if the model is deprecated"
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the model"
    )
    model_sets: list[str] | None = Field(
        default=None,
        description="Tags grouping the model into Venice content sets (e.g. 'high_resolution').",
    )


class TextModelSpec(ModelSpec):
    """Spec for text / chat models (``type='text'``).

    Adds the LLM-specific feature flags and parameter constraints that don't
    apply to other model families.
    """

    capabilities: ModelCapabilities | None = Field(
        default=None, description="Text model capabilities"
    )
    constraints: TextModelConstraints | None = Field(
        default=None, description="Text-model parameter constraints (temperature, top_p)"
    )
    availableContextTokens: float | None = Field(
        default=None, description="Context length supported by the model"
    )
    maxCompletionTokens: float | None = Field(
        default=None, description="Maximum completion tokens the model can produce"
    )


class ImageModelSpec(ModelSpec):
    """Spec for image-generation models (``type='image'``)."""

    constraints: ImageModelConstraints | None = Field(
        default=None, description="Image-model constraints (prompt limit, steps, dimensions)"
    )
    supportsWebSearch: bool | None = Field(
        default=None,
        description="Whether the image model supports web search for prompt enhancement.",
    )


class VideoModelSpec(ModelSpec):
    """Spec for video-generation models (``type='video'``)."""

    constraints: VideoModelConstraints | None = Field(
        default=None,
        description="Video-model constraints (aspect ratios, resolutions, durations, audio).",
    )


class InpaintModelSpec(ModelSpec):
    """Spec for inpaint models (``type='inpaint'``)."""

    constraints: InpaintModelConstraints | None = Field(
        default=None, description="Inpaint-model constraints (prompt limit, image combination)."
    )


class MusicModelSpec(ModelSpec):
    """Spec for music-generation models (``type='music'``).

    Music is the richest model type — Venice exposes ~20 capability fields
    on it. Field names are snake_case to match the wire.
    """

    voices: list[str] | None = Field(default=None, description="Available voices for music models.")
    default_voice: str | None = Field(
        default=None, description="Default voice when none is specified."
    )

    # Duration capability matrix — different music models use different shapes.
    # ``duration_options`` (enum) takes precedence over ``min_duration`` /
    # ``max_duration`` (range) when present.
    duration_options: list[int] | None = Field(
        default=None,
        description=(
            "Strict enum of supported durations in seconds. When present, the "
            "model rejects values not in the list (e.g. ace-step-15 only accepts "
            "[60, 90, 120, ...])."
        ),
    )
    min_duration: int | None = Field(
        default=None, description="Minimum supported duration in seconds."
    )
    max_duration: int | None = Field(
        default=None, description="Maximum supported duration in seconds."
    )
    default_duration: int | None = Field(
        default=None, description="Default duration when none is specified."
    )

    # Format capability.
    supported_formats: list[str] | None = Field(
        default=None, description="Supported output audio formats (e.g. ['mp3', 'wav'])."
    )
    default_format: str | None = Field(
        default=None, description="Default audio format when none is specified."
    )

    # Prompt limits.
    prompt_character_limit: int | None = Field(
        default=None, description="Maximum prompt length in characters."
    )
    lyrics_character_limit: int | None = Field(
        default=None, description="Maximum lyrics prompt length in characters."
    )
    min_prompt_length: int | None = Field(
        default=None, description="Minimum prompt length in characters."
    )

    # Lyrics support.
    supports_lyrics: bool | None = Field(
        default=None, description="Whether the model accepts a ``lyrics_prompt`` parameter."
    )
    lyrics_required: bool | None = Field(
        default=None,
        description="Whether ``lyrics_prompt`` is required (true for vocal-only models).",
    )
    supports_lyrics_optimizer: bool | None = Field(
        default=None,
        description="Whether the model supports the ``lyrics_optimizer`` flag.",
    )
    supports_force_instrumental: bool | None = Field(
        default=None,
        description="Whether the model supports the ``force_instrumental`` flag.",
    )
    supports_loop: bool | None = Field(
        default=None,
        description="Whether the model supports the ``loop`` flag.",
    )

    supports_language_code: bool | None = Field(
        default=None, description="Whether the model accepts a ``language_code`` parameter."
    )

    # Speed capability.
    supports_speed: bool | None = Field(
        default=None, description="Whether the model accepts a ``speed`` parameter."
    )
    min_speed: float | None = Field(default=None, description="Minimum supported speed multiplier.")
    max_speed: float | None = Field(default=None, description="Maximum supported speed multiplier.")
    default_speed: float | None = Field(default=None, description="Default speed multiplier.")


class TtsModelSpec(ModelSpec):
    """Spec for text-to-speech models (``type='tts'``)."""

    voices: list[str] | None = Field(default=None, description="Available voices for TTS models.")
    default_voice: str | None = Field(
        default=None, description="Default voice when none is specified."
    )


class AsrModelSpec(ModelSpec):
    """Spec for ASR (automatic speech recognition) models (``type='asr'``).

    No type-specific fields beyond the base today.
    """


class EmbeddingModelSpec(ModelSpec):
    """Spec for embedding models (``type='embedding'``)."""

    embeddingDimensions: int | None = Field(
        default=None,
        description=("Native/default number of dimensions in the output embedding vector."),
    )
    maxInputTokens: int | None = Field(
        default=None, description="Maximum input tokens accepted per input string."
    )
    supportsCustomDimensions: bool | None = Field(
        default=None,
        description=(
            "Whether the model supports reducing output dimensions via the "
            "``dimensions`` request parameter."
        ),
    )


class UpscaleModelSpec(ModelSpec):
    """Spec for image-upscale models (``type='upscale'``).

    No type-specific fields beyond the base today.
    """


# Dispatch map: model type string → typed spec class.
# Used by ``ModelResponse._coerce_spec_subclass`` to upgrade the raw
# ``model_spec`` dict into the right subclass during validation.
_SPEC_BY_TYPE: dict[str, type[ModelSpec]] = {
    "text": TextModelSpec,
    "image": ImageModelSpec,
    "video": VideoModelSpec,
    "inpaint": InpaintModelSpec,
    "music": MusicModelSpec,
    "tts": TtsModelSpec,
    "asr": AsrModelSpec,
    "embedding": EmbeddingModelSpec,
    "upscale": UpscaleModelSpec,
}


class ModelResponse(BaseModel):
    """Individual model information.

    The ``model_spec`` field's runtime type is the appropriate subclass of
    :class:`ModelSpec` (e.g. :class:`MusicModelSpec` for ``type='music'``);
    the static annotation stays as :class:`ModelSpec` so existing callers keep
    type-checking. Use ``isinstance(entry.model_spec, MusicModelSpec)`` to
    narrow.

    ``extra='allow'`` matches :class:`ModelSpec`'s policy so any future
    top-level Venice fields are preserved on
    :attr:`pydantic.BaseModel.model_extra` rather than silently dropped.
    ``context_length`` (added late 2025) is now a typed field rather than an
    extra; it surfaces the maximum context window in tokens as a top-level
    convenience field mirroring ``model_spec.availableContextTokens``.
    """

    model_config = ConfigDict(extra="allow")

    id: ModelId = Field(..., description="Model ID")
    object: Literal["model"] = Field(..., description="Object type")
    created: float | None = Field(default=None, description="Release date on Venice API")
    owned_by: str = Field(
        ...,
        description=(
            "Who runs the model. Typically ``venice.ai`` but kept as a string to "
            "tolerate any owner the API may report."
        ),
    )
    type: Literal[
        "embedding", "image", "text", "tts", "upscale", "inpaint", "video", "asr", "music"
    ] = Field(..., description="Model type")
    model_spec: ModelSpec = Field(..., description="Detailed model specifications")
    context_length: int | None = Field(
        default=None,
        description=(
            "Maximum context window in tokens. Mirrors "
            "``model_spec.availableContextTokens`` as a top-level convenience "
            "field (added late 2025). ``None`` for non-text models."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_spec_subclass(cls, data: Any) -> Any:
        """Promote ``model_spec`` to the right subclass based on ``type``.

        Runs *before* field validation, so the resulting dict still passes
        Pydantic's type check (every subclass is a ``ModelSpec``). If
        ``model_spec`` is already a model instance we leave it alone — the
        validator is only meant to handle raw API responses.
        """
        if isinstance(data, dict):
            spec = data.get("model_spec")
            if isinstance(spec, dict):
                type_value = data.get("type")
                spec_cls = (
                    _SPEC_BY_TYPE.get(type_value, ModelSpec)
                    if isinstance(type_value, str)
                    else ModelSpec
                )
                # Don't mutate the caller's dict.
                data = dict(data)
                data["model_spec"] = spec_cls.model_validate(spec)
        return data


class ModelsListResponse(VeniceBaseModel):
    """Models list endpoint response"""

    object: Literal["list"] = Field(..., description="Object type")
    type: str = Field(..., description="Type of models returned")
    data: list[ModelResponse] = Field(..., description="List of available models")


class ModelTraitsResponse(VeniceBaseModel):
    """Model traits endpoint response"""

    object: Literal["list"] = Field(..., description="Object type")
    type: str = Field(..., description="Type of models the traits apply to")
    data: dict[str, str] = Field(..., description="Key-value pairs of trait names to model IDs")


class ModelCompatibilityResponse(VeniceBaseModel):
    """Model compatibility mapping endpoint response"""

    object: Literal["list"] = Field(..., description="Object type")
    type: str = Field(..., description="Type of models the mappings apply to")
    data: dict[str, str] = Field(
        ..., description="Key-value pairs of external model names to Venice model IDs"
    )


__all__ = [
    "ModelCapabilities",
    "TemperatureConstraint",
    "TopPConstraint",
    "StepsConstraint",
    "TextModelConstraints",
    "ImageModelConstraints",
    "InpaintModelConstraints",
    "VideoModelConstraints",
    "PricingTier",
    "ExtendedPricing",
    "LLMModelPricing",
    "UpscalePricing",
    "ImageModelPricing",
    "AudioModelPricing",
    "VideoResolutionPricing",
    "InpaintModelPricing",
    "ASRModelPricing",
    "MusicModelPricing",
    "ModelConstraints",
    "ModelPricing",
    "ModelDeprecation",
    # Spec hierarchy (base + 9 per-type subclasses)
    "ModelSpec",
    "TextModelSpec",
    "ImageModelSpec",
    "VideoModelSpec",
    "InpaintModelSpec",
    "MusicModelSpec",
    "TtsModelSpec",
    "AsrModelSpec",
    "EmbeddingModelSpec",
    "UpscaleModelSpec",
    "ModelResponse",
    "ModelsListResponse",
    "ModelTraitsResponse",
    "ModelCompatibilityResponse",
]
