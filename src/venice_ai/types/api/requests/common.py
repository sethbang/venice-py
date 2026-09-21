"""
Shared request components and utility models for Venice.ai API requests.

Most types are re-exported from the canonical definitions in
``venice_ai.core.models.common`` (which use ``VeniceBaseModel`` with strict
validation).  Only ``DateRangeParams`` is defined locally because the API
serialisation uses camelCase field names (``startDate`` / ``endDate``) whereas
the core model uses snake_case.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    field_validator,
)

# ---------------------------------------------------------------------------
# Re-exports from canonical location (core/models/common)
# ---------------------------------------------------------------------------
from venice_ai.core.models.common import (
    AudioContent,
    AudioContentParam,
    ConsumptionLimit,
    FileContent,
    FileContentParam,
    FileObject,
    FileObjectParam,
    ImageContent,
    ImageContentParam,
    ImageUrl,
    ImageUrlParam,
    JSONObjectFormat,
    JSONSchemaFormat,
    MessageContentPart,
    MessageContentPartParam,
    PaginationParams,
    SpecificToolChoice,
    StreamOptions,
    TextContent,
    TextContentParam,
    TextResponseFormat,
    Tool,
    ToolChoiceFunction,
    ToolFunction,
    VeniceParameters,
    VideoContent,
    VideoContentParam,
)

# ---------------------------------------------------------------------------
# Reasoning controls (shared by chat + completion requests)
# ---------------------------------------------------------------------------

ReasoningEffortLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""Effort tier for reasoning-capable models.

Mirrors the `/chat/completions` spec: higher levels allow more thinking tokens
but cost more. ``"max"`` unlocks the full ceiling on supported models.
"""

ReasoningSummary = Literal["auto", "concise", "detailed"]
"""Requested reasoning-summary verbosity."""


class ReasoningConfig(BaseModel):
    """Nested configuration for reasoning behavior on supported models.

    Corresponds to the ``reasoning`` object on ``/chat/completions``. The
    top-level ``reasoning_effort`` field takes precedence over
    ``reasoning.effort`` when both are provided.
    """

    effort: ReasoningEffortLevel | None = Field(default=None, description="Reasoning effort tier")
    summary: ReasoningSummary | None = Field(
        default=None, description="Requested reasoning summary style"
    )


# ============================================================================
# DateRangeParams — API-facing (camelCase) version
# ============================================================================


class DateRangeParams(BaseModel):
    """Date range filtering parameters"""

    startDate: datetime | None = Field(None, description="Start date")
    endDate: datetime | None = Field(None, description="End date")

    @field_validator("endDate")
    @classmethod
    def validate_date_range(cls, v: Any, info: ValidationInfo) -> Any:
        start_date = info.data.get("startDate")
        if start_date and v and v <= start_date:
            raise ValueError("endDate must be after startDate")
        return v


# ============================================================================
# Export All Models
# ============================================================================


#: Printable-ASCII range the API accepts for ``anon_user_id``.
ANON_USER_ID_PATTERN = r"^[\x20-\x7E]+$"


def _reject_delimiter(value: str) -> str:
    """``||`` is the delimiter Venice joins the id to its own user id with."""
    if "||" in value:
        raise ValueError("anon_user_id must not contain '||' — it is a reserved delimiter")
    return value


#: Optional identifier for the API customer's own end user.
#:
#: Venice combines it with the Venice user id when attributing a request to an
#: upstream provider. It is **not** the OpenAI-compatible ``user`` field, which
#: Venice discards — passing ``user`` does nothing, and the two are not aliases.
AnonUserId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=ANON_USER_ID_PATTERN),
    AfterValidator(_reject_delimiter),
]

#: Field definition shared by every endpoint that accepts ``anon_user_id``, so
#: the description and constraints cannot drift apart across the seven models.
ANON_USER_ID_DESCRIPTION = (
    "Optional identifier for the API customer's end user, combined with the "
    "Venice user id when attributing the request to upstream providers. "
    "Printable ASCII, 1-128 characters, and must not contain '||'. Distinct "
    "from the OpenAI-compatible `user` field, which Venice discards."
)


_ANON_USER_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(AnonUserId)


def validate_anon_user_id(value: str) -> str:
    """Validate an ``anon_user_id`` outside a request model.

    ``image.multi_edit()`` assembles its body as a plain dict rather than
    through :class:`ImageMultiEditRequest`, so it would otherwise skip the
    constraints every other endpoint enforces.
    """
    return _ANON_USER_ID_ADAPTER.validate_python(value)


__all__ = [
    # Content types
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
    # Stream and tool components
    "StreamOptions",
    "ToolFunction",
    "Tool",
    "ToolChoiceFunction",
    "SpecificToolChoice",
    # Response format components
    "JSONSchemaFormat",
    "JSONObjectFormat",
    "TextResponseFormat",
    # Venice-specific components
    "VeniceParameters",
    # Reasoning controls
    "ReasoningEffortLevel",
    "ReasoningSummary",
    "ReasoningConfig",
    # End-user attribution
    "AnonUserId",
    "ANON_USER_ID_PATTERN",
    "ANON_USER_ID_DESCRIPTION",
    "validate_anon_user_id",
    # Utility models
    "PaginationParams",
    "DateRangeParams",
    "ConsumptionLimit",
]
