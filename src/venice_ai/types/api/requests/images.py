"""
Image generation, editing, and upscaling request models for Venice.ai API.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...identifiers import ModelId

# ============================================================================
# Image Generation Request Models
# ============================================================================


class StyleReference(BaseModel):
    """One style reference image guiding the aesthetic of a generated image."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        ...,
        description=(
            "Style reference as a base64 string (raw or data URI) or an "
            "http(s) URL. Must be under 8MB."
        ),
    )
    strength: float | None = Field(
        None,
        ge=0.1,
        le=1,
        description=(
            "How strongly the reference guides the output (0.1-1, default 0.5). "
            "Ignored by models where ``supportsStyleReferenceStrength`` is false."
        ),
    )


class ImageGenerationRequest(BaseModel):
    """Native Venice image generation request"""

    model: ModelId = Field(..., description="Image generation model to use")
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=7500,
        description=(
            "Image description. Per-model caps are exposed via "
            "``promptCharacterLimit`` on GET /models — the server enforces the "
            "model-specific limit; the 7500 ceiling here matches the endpoint spec."
        ),
    )

    # Image dimensions
    width: int | None = Field(1024, gt=0, le=1280, description="Image width in pixels")
    height: int | None = Field(1024, gt=0, le=1280, description="Image height in pixels")

    # Generation parameters
    cfg_scale: float | None = Field(
        None, gt=0, le=20, description="CFG scale (higher = more prompt adherence)"
    )
    steps: int | None = Field(
        8,
        gt=0,
        description=(
            "Number of inference steps. Per-model caps are enforced server-side; "
            "8 matches the documented endpoint default."
        ),
    )
    seed: int | None = Field(0, ge=-999999999, le=999999999, description="Random seed")

    # Style and enhancement
    style_preset: str | None = Field(None, description="Style preset to apply")
    lora_strength: int | None = Field(
        None, ge=0, le=100, description="LoRA strength (if model uses LoRAs)"
    )

    # Output control
    format: Literal["jpeg", "png", "webp"] | None = Field("webp", description="Output image format")
    variants: int | None = Field(
        None,
        ge=1,
        le=4,
        description="Number of images to generate (only when return_binary is false)",
    )
    return_binary: bool | None = Field(False, description="Return binary data instead of base64")
    safe_mode: bool | None = Field(True, description="Blur adult content")
    hide_watermark: bool | None = Field(False, description="Hide Venice watermark")
    embed_exif_metadata: bool | None = Field(False, description="Embed generation info in EXIF")

    # Resolution control
    resolution: str | None = Field(
        default=None, description="Output resolution: '1K', '2K', or '4K' for supported models"
    )
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Aspect ratio for the output image (e.g. '1:1', '16:9'). Supported by "
            "models advertising aspect-ratio control (e.g. Nano Banana); allowed "
            "values vary per model — inspect GET /models."
        ),
    )

    # Web-augmented generation
    enable_web_search: bool | None = Field(
        default=None,
        description=(
            "Allow the image model to incorporate recent web-search context. "
            "Supported by models with ``supportsWebSearch``."
        ),
    )

    # Quality control
    quality: Literal["low", "medium", "high"] | None = Field(
        None,
        description=(
            "Output quality for quality-aware models (e.g. GPT Image 2). Higher values can "
            "increase the request charge. See the model spec's `qualities` for supported values."
        ),
    )

    enhance_prompt: bool | None = Field(
        None,
        description=(
            "Rewrite the prompt before generation to add clarifying visual detail. "
            "Charges additional credits when a rewrite happens and adds up to ~30s "
            "before generation starts. The rewritten prompt comes back URL-encoded "
            "in the ``x-venice-enhanced-prompt`` response header, exposed as "
            "``ImageGenerationResponse.enhanced_prompt``."
        ),
    )
    disable_prompt_optimization_thinking: bool | None = Field(
        None,
        description=(
            "Skip the model's prompt-optimization thinking step for faster "
            "generation. Supported only by models that advertise it; ignored "
            "rather than rejected by the rest. Omit for the model default."
        ),
    )
    style_references: list[StyleReference] | None = Field(
        None,
        description=(
            "Style reference images guiding the aesthetic of the output. Only "
            "supported by models with ``supportsStyleReferences``."
        ),
    )


class SimpleImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request"""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description=(
            "Image description. The OpenAI-compatible endpoint caps prompts at "
            "1500 characters per the API spec."
        ),
    )

    # Basic parameters
    model: ModelId | None = Field("default", description="Model to use")
    n: int | None = Field(1, ge=1, le=1, description="Number of images (Venice supports 1)")
    size: (
        Literal[
            "auto",
            "256x256",
            "512x512",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "1792x1024",
            "1024x1792",
        ]
        | None
    ) = Field("auto", description="Image size")

    # Output control
    response_format: Literal["b64_json", "url"] | None = Field(
        "b64_json", description="Response format"
    )
    output_format: Literal["jpeg", "png", "webp"] | None = Field(
        "png", description="Output image format"
    )

    # OpenAI compatibility (not used by Venice)
    quality: Literal["auto", "high", "medium", "low", "hd", "standard"] | None = Field(
        "auto", description="Quality setting (compatibility only)"
    )
    style: Literal["vivid", "natural"] | None = Field(
        "natural", description="Style setting (compatibility only)"
    )
    background: Literal["transparent", "opaque", "auto"] | None = Field(
        "auto", description="Background setting (compatibility only)"
    )
    moderation: Literal["low", "auto"] | None = Field(
        "auto", description="Moderation level (auto = safe mode on)"
    )
    output_compression: int | None = Field(
        100, ge=0, le=100, description="Compression level (compatibility only)"
    )
    user: str | None = Field(None, description="User identifier (compatibility only)")


class ImageUpscaleRequest(BaseModel):
    """Image upscale request.

    ``POST /image/upscale`` takes exactly three fields. The older ``enhance``,
    ``enhancePrompt``, ``enhanceCreativity`` and ``replication`` fields were
    removed server-side and are no longer sent.
    """

    image: str | Any = Field(..., description="Image to upscale (file upload or base64 string)")

    scale: float | None = Field(
        2,
        ge=2,
        le=4,
        description=(
            "Scale factor for upscaling. Must be 2 or 4 — 1 is rejected. A scale "
            "of 4 on a large image is dynamically reduced to keep the result "
            "within the maximum size limit."
        ),
    )
    creativity: float | None = Field(
        None,
        description=(
            "How much detail and texture the upscaler adds; higher values add "
            "more, lower stay closer to the source. The server clamps this to "
            "0-0.02 (default 0.01) rather than rejecting out-of-range values, so "
            "the SDK forwards whatever you pass."
        ),
    )


class ImageEditRequest(BaseModel):
    """Image edit request"""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32768,
        description=(
            "Text directions for editing the image. Per-model caps are exposed "
            "via ``promptCharacterLimit`` on GET /models; 32768 matches the "
            "endpoint spec ceiling."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Edit model (e.g., 'flux-2-max-edit', 'gpt-image-1-5-edit')",
    )
    image: str | Any = Field(..., description="Image to edit (file, base64, or URL)")
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Aspect ratio for the output (e.g. '1:1', '16:9'). Omit to use the "
            "model's default; supported values vary per model."
        ),
    )
    safe_mode: bool | None = Field(
        default=None,
        description=(
            "Blur images classified as having adult content. Defaults to true "
            "server-side when omitted; pass ``False`` to disable."
        ),
    )
    resolution: str | None = Field(
        default=None,
        min_length=1,
        max_length=10,
        description=(
            "Resolution tier for the output image (e.g. '1K', '2K', '4K'). "
            "Supported values vary by model; defaults to '1K' server-side."
        ),
    )
    output_format: Literal["jpeg", "png", "webp"] | None = Field(
        default=None,
        description=(
            "Output format for the edited image. When omitted, the format is "
            "inferred from resolution (PNG for 1K edits, JPEG for 2K/4K edits)."
        ),
    )
    quality: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description=(
            "Output quality for quality-aware edit models (e.g. gpt-image-2-edit) "
            "per the edit docs. Model-dependent and sent only when set; omit for "
            "models that don't support it."
        ),
    )

    enhance_prompt: bool | None = Field(
        None,
        description=(
            "Rewrite the prompt before generation to add clarifying visual detail. "
            "Charges additional credits when a rewrite happens and adds up to ~30s "
            "before generation starts. The rewritten prompt comes back URL-encoded "
            "in the ``x-venice-enhanced-prompt`` response header, exposed as "
            "``ImageGenerationResponse.enhanced_prompt``."
        ),
    )
    disable_prompt_optimization_thinking: bool | None = Field(
        None,
        description=(
            "Skip the model's prompt-optimization thinking step for faster "
            "generation. Supported only by models that advertise it; ignored "
            "rather than rejected by the rest. Omit for the model default."
        ),
    )


class ImageBackgroundRemoveRequest(BaseModel):
    """Request model for background removal."""

    image: str | None = Field(default=None, description="Base64-encoded image data")
    image_url: str | None = Field(
        default=None, description="URL of the image to remove the background from"
    )


class ImageMultiEditRequest(BaseModel):
    """Request model for ``POST /image/multi-edit``.

    The endpoint accepts an ``images`` array of 1–3 base64 / URL inputs (the
    first is the base image; remaining ones are layered on top). ``modelId``
    selects the edit model. ``safe_mode`` blurs adult outputs (server default
    ``True``).
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32768,
        description="Edit instruction (per-model cap via ``promptCharacterLimit``).",
    )
    images: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description=(
            "1–3 layered images. Each is a base64 string or http(s) URL. The "
            "first image is treated as the base; remaining images are edit layers."
        ),
    )
    modelId: str | None = Field(
        default=None,
        description="Edit model ID (e.g. ``qwen-edit``, ``flux-2-max-edit``).",
    )
    safe_mode: bool | None = Field(
        default=None,
        description=(
            "Blur images classified as having adult content. Defaults to true "
            "server-side; pass ``False`` to disable."
        ),
    )
    resolution: str | None = Field(
        default=None,
        min_length=1,
        max_length=10,
        description=(
            "Resolution tier for the output image (e.g. '1K', '2K', '4K'). "
            "Supported values vary by model; defaults to '1K' server-side."
        ),
    )
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Aspect ratio for the output (e.g. '1:1', '16:9'). Omit to infer "
            "from the first input image; supported values vary per model."
        ),
    )
    output_format: Literal["jpeg", "png", "webp"] | None = Field(
        default=None,
        description=(
            "Output format for the edited image. When omitted, the format is "
            "inferred from resolution (PNG for 1K edits, JPEG for 2K/4K edits)."
        ),
    )
    quality: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description=(
            "Output quality for supported models (e.g. GPT Image 2). Higher "
            "values can increase the request charge."
        ),
    )

    # ============================================================================
    # Export All Models
    # ============================================================================

    enhance_prompt: bool | None = Field(
        None,
        description=(
            "Rewrite the prompt before generation to add clarifying visual detail. "
            "Charges additional credits when a rewrite happens and adds up to ~30s "
            "before generation starts. The rewritten prompt comes back URL-encoded "
            "in the ``x-venice-enhanced-prompt`` response header, exposed as "
            "``ImageGenerationResponse.enhanced_prompt``."
        ),
    )
    disable_prompt_optimization_thinking: bool | None = Field(
        None,
        description=(
            "Skip the model's prompt-optimization thinking step for faster "
            "generation. Supported only by models that advertise it; ignored "
            "rather than rejected by the rest. Omit for the model default."
        ),
    )


__all__ = [
    "ImageGenerationRequest",
    "SimpleImageGenerationRequest",
    "ImageUpscaleRequest",
    "ImageEditRequest",
    "ImageBackgroundRemoveRequest",
    "ImageMultiEditRequest",
]
