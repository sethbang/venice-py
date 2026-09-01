"""
This module defines common and shared Pydantic models that are used
throughout the Venice AI SDK. These models provide a consistent data
structure for common entities such as pagination, timestamps, and usage
information, ensuring type safety and clarity across different parts of the
application.

The module is split into focused submodules:

* ``base``    — ``VeniceBaseModel``, ``TimestampMixin``
* ``enums``   — ``ModelType``, ``APIKeyType``, ``Currency``, …
* ``headers`` — ``RateLimitInfo``, ``DeprecationInfo``, ``BalanceInfo``, …
* ``metrics`` — ``UsageInfo``, ``TimingInfo``, ``SchedulerMetrics``, ``CacheStats``

This file re-exports *everything* from the submodules so that the existing
import path ``from venice_ai.core.models.common import X`` continues to work
unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import (
    BaseModel,  # noqa: F401 — used by subclasses via wildcard
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

if TYPE_CHECKING:
    from venice_ai.types.api.common import WebSearchCitation

# ---------------------------------------------------------------------------
# Re-exports from focused submodules
# ---------------------------------------------------------------------------
from .base import TimestampMixin, VeniceBaseModel  # noqa: E402
from .enums import APIKeyType, Currency, FinishReason, MessageRole, ModelType  # noqa: E402
from .headers import (  # noqa: E402
    BalanceInfo,
    ContentSafetyInfo,
    DeprecationInfo,
    ModelInfo,
    PaginationInfo,
    RateLimitInfo,
)
from .metrics import CacheStats, SchedulerMetrics, TimingInfo, UsageInfo  # noqa: E402

# ============================================================================
# Pagination Parameters and Date Range (kept here — shared utility types)
# ============================================================================


class PaginationParams(VeniceBaseModel):
    """Generic pagination parameters for requests."""

    page: int | None = Field(1, ge=1, description="Page number")
    limit: int | None = Field(50, ge=1, le=500, description="Items per page")


class DateRangeParams(VeniceBaseModel):
    """Date range filtering parameters."""

    start_date: datetime | None = Field(None, description="Start date")
    end_date: datetime | None = Field(None, description="End date")

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: Any, info: ValidationInfo) -> Any:
        start_date = info.data.get("start_date")
        if start_date and v and v <= start_date:
            raise ValueError("end_date must be after start_date")
        return v


# ============================================================================
# Consumption and Pricing Models
# ============================================================================


class ConsumptionLimit(VeniceBaseModel):
    """Consumption limit specification for API keys and billing."""

    usd: float | None = Field(None, ge=0, description="USD limit")
    diem: float | None = Field(None, ge=0, description="Diem limit")


class Balances(VeniceBaseModel):
    """Account balances."""

    # swagger marks USD/DIEM optional with no additionalProperties:false; allow a
    # future server-side currency to land on model_extra instead of crashing.
    model_config = ConfigDict(extra="allow")

    # NOTE: USD/DIEM are kept as required (Field(...)) deliberately, tighter than
    # swagger (which declares no required array on the balances sub-object). Live
    # wire confirms both are always returned, so requiring them buys type-safety
    # without practical risk of a missing-key parse failure.
    USD: float = Field(..., description="USD balance")
    DIEM: float = Field(..., description="Diem balance")


# ============================================================================
# Content and Message Components
# ============================================================================


class TextContent(VeniceBaseModel):
    """Text content object for messages."""

    type: Literal["text"] = Field(..., description="Content type")
    text: str = Field(..., min_length=1, description="The text content")
    cache_control: dict[str, str] | None = Field(
        default=None,
        description="Cache control for prompt caching (e.g., {'type': 'ephemeral'})",
    )


class ImageUrl(VeniceBaseModel):
    """Image URL object."""

    url: str = Field(..., description="Image URL (data URL with base64 or public URL)")


class ImageContent(VeniceBaseModel):
    """Image content object for messages."""

    type: Literal["image_url"] = Field(..., description="Content type")
    image_url: ImageUrl = Field(..., description="Image URL information")
    cache_control: dict[str, str] | None = Field(
        default=None,
        description="Cache control for prompt caching (e.g., {'type': 'ephemeral'})",
    )


class FileObject(VeniceBaseModel):
    """File payload for a file content part."""

    file_data: str = Field(
        ...,
        description=(
            "File content as a data URL (e.g. ``data:application/pdf;base64,...``) "
            "or a publicly accessible URL. Supported: PDF, EPUB, DOCX, PPTX, "
            "XLSX/XLS, txt, md, csv, json, and most source-code files."
        ),
    )
    filename: str | None = Field(
        default=None, description="Optional filename, used for display purposes."
    )


class FileContent(VeniceBaseModel):
    """File content object for messages (extracted to text server-side)."""

    type: Literal["file"] = Field(..., description="Content type")
    file: FileObject = Field(..., description="File data and optional filename")
    cache_control: dict[str, str] | None = Field(
        default=None,
        description="Cache control for prompt caching (e.g., {'type': 'ephemeral'})",
    )


class AudioContent(VeniceBaseModel):
    """Audio content for multimodal messages."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: dict[str, str] = Field(
        ...,
        description="Audio data with 'data' (base64) and 'format' (e.g., 'wav', 'mp3') keys",
    )


class VideoContent(VeniceBaseModel):
    """Video content for multimodal messages."""

    type: Literal["video_url"] = "video_url"
    video_url: dict[str, str] = Field(
        ...,
        description="Video URL with 'url' key",
    )


MessageContentPart = Annotated[
    TextContent | ImageContent | AudioContent | VideoContent | FileContent,
    Field(discriminator="type"),
]
"""Discriminated union of multimodal message content parts.

Pydantic uses the ``type`` field as the discriminator, enabling fast O(1)
validation and unambiguous coercion of plain dicts ``{"type": "text", ...}``
or ``{"type": "image_url", ...}`` into the corresponding typed object.
"""


# TypedDict mirrors of each content part — purely for static-typing ergonomics
# at message-construction sites. Pydantic's discriminator validates the runtime
# shape regardless of whether the caller passes BaseModel instances or dicts.


class TextContentParam(TypedDict):
    """TypedDict shape of :class:`TextContent` for dict-form input."""

    type: Literal["text"]
    text: str
    cache_control: NotRequired[dict[str, str]]


class ImageUrlParam(TypedDict):
    """TypedDict shape of :class:`ImageUrl` for dict-form input."""

    url: str


class ImageContentParam(TypedDict):
    """TypedDict shape of :class:`ImageContent` for dict-form input."""

    type: Literal["image_url"]
    image_url: ImageUrlParam
    cache_control: NotRequired[dict[str, str]]


class AudioContentParam(TypedDict):
    """TypedDict shape of :class:`AudioContent` for dict-form input."""

    type: Literal["input_audio"]
    input_audio: dict[str, str]


class VideoContentParam(TypedDict):
    """TypedDict shape of :class:`VideoContent` for dict-form input."""

    type: Literal["video_url"]
    video_url: dict[str, str]


class FileObjectParam(TypedDict):
    """TypedDict shape of :class:`FileObject` for dict-form input."""

    file_data: str
    filename: NotRequired[str]


class FileContentParam(TypedDict):
    """TypedDict shape of :class:`FileContent` for dict-form input."""

    type: Literal["file"]
    file: FileObjectParam
    cache_control: NotRequired[dict[str, str]]


MessageContentPartParam = Annotated[
    MessageContentPart
    | TextContentParam
    | ImageContentParam
    | AudioContentParam
    | VideoContentParam
    | FileContentParam,
    Field(union_mode="left_to_right"),
]
"""Input alias accepted by message constructors.

Mirrors :data:`MessageContentPart` plus :class:`TypedDict` variants of each
content shape so callers can pass plain dicts without type-checker complaints.
At runtime Pydantic validates dicts via the same discriminated union — invalid
``type`` values still raise :class:`pydantic.ValidationError`.

``union_mode="left_to_right"`` keeps :data:`MessageContentPart` ahead of the
:class:`TypedDict` mirrors so a plain dict is coerced into the corresponding
typed content object instead of being left as a dict. The mirrors describe the
same shapes by construction, so smart mode — which scores union members rather
than honouring their order — has no stable reason to prefer one over the other.
"""


# ============================================================================
# Tool and Function Models
# ============================================================================


class ToolFunction(VeniceBaseModel):
    """Function definition for tool."""

    name: str = Field(..., description="Function name")
    description: str | None = None
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Function parameters schema - must remain Dict[str, Any] per JSON Schema spec",
    )
    strict: bool = False


class Tool(VeniceBaseModel):
    """Tool definition for chat completions.

    Two shapes are accepted to mirror the API spec:

    * ``type="function"`` with a ``function`` payload — the standard
      OpenAI-compatible function tool.
    * Built-in tools such as ``{"type": "web_search"}`` — Venice exposes a
      handful of server-side tools that take no schema; in that case
      ``function`` is omitted.
    """

    type: str = Field(
        ...,
        description=(
            "Tool type. ``function`` for user-defined function tools; built-in "
            "server-side tools use values like ``web_search``."
        ),
    )
    function: ToolFunction | None = Field(
        default=None,
        description="Function definition. Required when ``type`` is ``function``.",
    )
    id: str | None = None


class ToolChoiceFunction(VeniceBaseModel):
    """Tool choice function specification."""

    name: str = Field(..., description="Function name to call")


class SpecificToolChoice(VeniceBaseModel):
    """Specific tool choice."""

    type: str = Field(..., description="Tool choice type")
    function: ToolChoiceFunction = Field(..., description="Function to call")


class ToolChoice:
    """Typed factories for the ``tool_choice`` parameter on chat completions.

    The chat API accepts ``"none"``, ``"auto"``, or a function-pinned
    :class:`SpecificToolChoice`. Constructing the function-pinned form by hand
    requires nesting two Pydantic models or a raw dict; these factories wrap
    that shape so callers can write::

        await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool_from_function(my_fn)],
            tool_choice=ToolChoice.function("my_fn"),
        )

    The class is not instantiated; all members are classmethods that return
    plain values matching the chat-completion ``tool_choice`` signature.
    """

    @classmethod
    def auto(cls) -> Literal["auto"]:
        """Let the model decide whether to call a tool (default behavior)."""
        return "auto"

    @classmethod
    def none(cls) -> Literal["none"]:
        """Force a plain text response — no tool will be called."""
        return "none"

    @classmethod
    def function(cls, name: str) -> SpecificToolChoice:
        """Force the model to call the named function."""
        return SpecificToolChoice(
            type="function",
            function=ToolChoiceFunction(name=name),
        )


# ============================================================================
# Response Format Models
# ============================================================================


class JSONSchemaFormat(VeniceBaseModel):
    """JSON Schema response format."""

    type: Literal["json_schema"] = Field(..., description="Response format type")
    json_schema: dict[str, Any] = Field(
        ...,
        description="JSON schema specification - must remain Dict[str, Any] per OpenAI spec",
    )


class JSONObjectFormat(VeniceBaseModel):
    """JSON Object response format (deprecated)."""

    type: Literal["json_object"] = Field(..., description="Response format type")


class TextResponseFormat(VeniceBaseModel):
    """Plain-text response format (``type="text"``)."""

    type: Literal["text"] = Field(..., description="Response format type")


# ============================================================================
# Venice-Specific Parameters
# ============================================================================


class VeniceParameters(VeniceBaseModel):
    """Venice-specific parameters for requests."""

    model_config = ConfigDict(extra="allow")

    character_slug: str | None = Field(
        default=None, description="Character slug of a public Venice character"
    )
    strip_thinking_response: bool = Field(
        default=False, description="Strip <think></think> blocks from response"
    )
    disable_thinking: bool = Field(
        default=False, description="Disable thinking on reasoning models"
    )
    enable_web_search: Literal["auto", "off", "on"] = Field(
        default="off", description="Enable web search for this request"
    )
    enable_web_citations: bool = Field(
        default=False,
        description=(
            "When web search is enabled, request LLM to cite sources with "
            "^index^ or ^i,j^ superscript format (e.g., ^1^)"
        ),
    )
    include_search_results_in_stream: bool = Field(
        default=False, description="Include search results in stream as first chunk"
    )
    return_search_results_as_documents: bool | None = Field(
        default=None,
        description="Surface search results as OpenAI-compatible tool call",
    )
    include_venice_system_prompt: bool = Field(
        default=True, description="Include Venice system prompts with user prompts"
    )
    enable_web_scraping: bool | None = Field(
        default=None,
        description=(
            "Enable web scraping of URLs detected in the user message. "
            "Up to 5 URLs per request are detected and scraped. "
            "Scraped content augments responses. Only successfully scraped URLs are billed."
        ),
    )
    enable_e2ee: bool | None = Field(
        default=None,
        description=(
            "Request TEE (trusted-execution) routing for this request on "
            "'e2ee-*' models. Setting this to True engages the SDK's real "
            "client-side end-to-end encryption flow on "
            "``chat.completions.create`` (attestation verification, per-message "
            "ECDH/AES-256-GCM encryption, response decryption); the equivalent "
            "``create(e2ee=True)`` argument is the preferred entry point. See "
            "``client.tee`` and the create() docs for the attestation-trust "
            "limitation."
        ),
    )
    enable_x_search: bool | None = Field(
        default=None,
        description="Enable X (Twitter) search to augment responses with real-time posts",
    )


class VeniceParametersResponse(VeniceBaseModel):
    """Venice-specific parameters returned in response."""

    # The response venice_parameters object has no additionalProperties:false in
    # swagger and already grew a field live (enable_x_search); allow extras so a
    # new key doesn't ValidationError the whole chat response.
    model_config = ConfigDict(extra="allow")

    enable_web_search: Literal["auto", "off", "on"] = Field(
        ..., description="Web search setting used"
    )
    enable_web_scraping: bool = Field(..., description="Whether web scraping was enabled")
    enable_web_citations: bool = Field(..., description="Whether web citations were enabled")
    include_venice_system_prompt: bool = Field(
        ..., description="Whether Venice system prompt was included"
    )
    include_search_results_in_stream: bool = Field(
        ..., description="Whether search results were included in stream"
    )
    return_search_results_as_documents: bool = Field(
        ..., description="Whether search results were returned as documents"
    )
    web_search_citations: list[WebSearchCitation] = Field(
        default_factory=list, description="Citations from web search"
    )
    character_slug: str | None = Field(None, description="Character slug used")
    strip_thinking_response: bool = Field(..., description="Whether thinking response was stripped")
    disable_thinking: bool = Field(..., description="Whether thinking was disabled")
    enable_e2ee: bool = Field(False, description="Whether end-to-end encryption was enabled")
    enable_x_search: bool = Field(False, description="Whether X (Twitter) search was enabled")


# ============================================================================
# Stream Options
# ============================================================================


class StreamOptions(VeniceBaseModel):
    """Stream options for chat completions."""

    include_usage: bool = Field(
        ..., description="Whether to include usage information in the stream"
    )


# ============================================================================
# Generic Response Models
# ============================================================================


class SuccessResponse(VeniceBaseModel):
    """Generic success response."""

    success: bool = Field(..., description="Operation success status")


class ListResponse(VeniceBaseModel):
    """Generic list response."""

    object: Literal["list"] = Field(..., description="Object type")
    data: list[Any] = Field(..., description="List data")


# ============================================================================
# Structured Response Models
# ============================================================================


class HealthCheckResult(VeniceBaseModel):
    """Health check response structure."""

    healthy: bool = Field(..., description="Overall health status")
    status: str = Field(..., description="Status description")
    checks: dict[str, bool] | None = Field(None, description="Individual component checks")
    timestamp: str | None = Field(None, description="Check timestamp")
    uptime: float | None = Field(None, description="Service uptime in seconds")
    version: str | None = Field(None, description="Service version")


class RequestEcho(VeniceBaseModel):
    """Echo of original request parameters (for API responses)."""

    model_config = ConfigDict(extra="allow")  # Allow echoing arbitrary request parameters

    model: str | None = Field(None, description="Model used")
    prompt: str | None = Field(None, description="Original prompt")
    temperature: float | None = Field(None, description="Temperature setting")
    max_completion_tokens: int | None = Field(
        None,
        description=(
            "Echoed max completion tokens setting. On reasoning models this caps "
            "total completion tokens (visible output + reasoning)."
        ),
    )
    # Additional fields allowed via extra="allow"


class ValidationResult(VeniceBaseModel):
    """Message validation result."""

    valid: bool = Field(..., description="Whether validation passed")
    message_count: int = Field(..., description="Number of messages validated")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


# ============================================================================
# Deferred imports to resolve forward references (avoids circular import)
# ============================================================================
# At this point all classes in this module are defined, so the transitive
# import chain types.api -> types.api.api_keys -> core.models.common can
# safely resolve ``Balances`` and other names.

from venice_ai.types.api.common import (  # noqa: E402, F811, F401
    PromptTokensDetails,
    WebSearchCitation,
)

UsageInfo.model_rebuild()
VeniceParametersResponse.model_rebuild()

# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    # Base models (from base.py)
    "VeniceBaseModel",
    "TimestampMixin",
    # Enums (from enums.py)
    "ModelType",
    "APIKeyType",
    "Currency",
    "FinishReason",
    "MessageRole",
    # Header models (from headers.py)
    "PaginationInfo",
    "RateLimitInfo",
    "DeprecationInfo",
    "BalanceInfo",
    "ContentSafetyInfo",
    "ModelInfo",
    # Metrics (from metrics.py)
    "UsageInfo",
    "TimingInfo",
    "SchedulerMetrics",
    "CacheStats",
    # Pagination params (defined here)
    "PaginationParams",
    "DateRangeParams",
    # Consumption and pricing
    "ConsumptionLimit",
    "Balances",
    # Structured response models
    "HealthCheckResult",
    "RequestEcho",
    "ValidationResult",
    # Content components
    "TextContent",
    "ImageUrl",
    "ImageContent",
    "FileObject",
    "FileContent",
    "AudioContent",
    "VideoContent",
    "MessageContentPart",
    "MessageContentPartParam",
    "TextContentParam",
    "ImageUrlParam",
    "ImageContentParam",
    "FileObjectParam",
    "FileContentParam",
    "AudioContentParam",
    "VideoContentParam",
    # Tools and functions
    "ToolFunction",
    "Tool",
    "ToolChoiceFunction",
    "SpecificToolChoice",
    "ToolChoice",
    # Response formats
    "JSONSchemaFormat",
    "JSONObjectFormat",
    "TextResponseFormat",
    # Venice-specific
    "VeniceParameters",
    "VeniceParametersResponse",
    # Stream options
    "StreamOptions",
    # Generic responses
    "SuccessResponse",
    "ListResponse",
]
