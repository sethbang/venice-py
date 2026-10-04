"""
Common shared types for Venice AI API.

This module contains types that are shared across multiple API endpoints
and need to be in a neutral location to avoid circular imports.
"""

from pydantic import BaseModel, ConfigDict, Field


class WebSearchCitation(BaseModel):
    """Web search citation object"""

    title: str = Field(..., description="Title of the cited source")
    url: str = Field(..., description="URL of the cited source")
    content: str = Field(..., description="Excerpt from the cited source")
    date: str = Field(..., description="Date of the cited content")


class PromptTokensDetails(BaseModel):
    """Breakdown of tokens used in the prompt.

    Provides detailed information about different types of tokens consumed
    in the prompt for cost estimation and optimization. This breakdown helps
    understand how prompt caching and multimodal inputs affect token usage.

    Attributes:
        cached_tokens: Number of tokens retrieved from cache (cache read),
            reducing costs
        cache_creation_input_tokens: Number of prompt tokens written to cache
            (cache write); charged at a premium rate by some providers
        audio_tokens: Number of audio tokens present in the prompt (for audio input models)

    Note:
        All fields are optional as not all models or requests will populate
        these details. When present, they provide valuable insights for
        optimizing API usage and costs.
    """

    model_config = ConfigDict(extra="allow")

    cached_tokens: int | None = Field(
        default=None,
        description="Number of tokens retrieved from prompt cache",
        ge=0,
    )
    cache_creation_input_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of prompt tokens written to cache (cache write); charged at "
            "a premium rate by some providers."
        ),
    )
    audio_tokens: int | None = Field(
        default=None,
        description="Number of audio tokens in the prompt",
        ge=0,
    )


class CompletionTokensDetails(BaseModel):
    """Breakdown of tokens generated in the completion.

    Populated by reasoning models (e.g. ``openai-gpt-54-mini``, ``grok-4-20``)
    and multi-modal output models so callers can distinguish visible output
    tokens from internal reasoning, audio, or image tokens for cost and
    latency analysis.

    Attributes:
        reasoning_tokens: Internal reasoning/thinking tokens counted against
            the completion budget. Present on models advertising
            ``supportsReasoning``.
        audio_tokens: Audio tokens emitted by the completion (for audio output
            models).
        image_tokens: Image tokens emitted by the completion (for image-output
            models). Returned by the live API even though the public swagger
            schema does not yet list it.

    Note:
        All fields are optional — models that don't emit a given modality
        simply omit the corresponding field.
    """

    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int | None = Field(
        default=None,
        description="Number of reasoning/thinking tokens included in the completion",
        ge=0,
    )
    audio_tokens: int | None = Field(
        default=None,
        description="Number of audio tokens in the completion",
        ge=0,
    )
    image_tokens: int | None = Field(
        default=None,
        description="Number of image tokens in the completion",
        ge=0,
    )


class ErrorDetails(BaseModel):
    """Structured error details for validation and API errors.

    The Venice validation envelope is a Zod tree whose general-error list is
    keyed ``_errors`` on the wire (``{"_errors": [...], "<field>": {"_errors":
    [...]}}``); ``.errors`` populates from that documented key. ``extra="allow"``
    preserves the nested per-field subtrees, and ``populate_by_name=True`` keeps
    the ``errors=`` construction path working.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    errors: list[str] | None = Field(
        default_factory=list,
        alias="_errors",
        description="General errors (wire key is '_errors' per swagger example)",
    )


__all__ = [
    "WebSearchCitation",
    "PromptTokensDetails",
    "CompletionTokensDetails",
    "ErrorDetails",
]
