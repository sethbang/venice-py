"""
Chat completion request models for Venice.ai API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from ..chat import ChatCompletionResponse

from ...identifiers import ModelId
from .common import (
    ANON_USER_ID_DESCRIPTION,
    AnonUserId,
    AudioContent,
    FileContent,
    FileObject,
    ImageContent,
    ImageUrl,
    JSONObjectFormat,
    JSONSchemaFormat,
    MessageContentPartParam,
    ReasoningConfig,
    ReasoningEffortLevel,
    SpecificToolChoice,
    StreamOptions,
    TextContent,
    TextResponseFormat,
    Tool,
    VeniceParameters,
    VideoContent,
)

# ============================================================================
# Message Models
# ============================================================================


class UserMessage(BaseModel):
    """User message for chat completions"""

    role: Literal["user"] = "user"
    content: str | list[MessageContentPartParam] = Field(
        ..., description="Message content (text or mixed content)"
    )

    @classmethod
    def builder(cls) -> UserMessageBuilder:
        """Start a fluent builder for multimodal content.

        Replaces the verbose ``UserMessage(content=[TextContent(...), ImageContent(...)])``
        construction with a chained API::

            msg = (
                UserMessage.builder()
                .text("What's in this image?")
                .image("https://example.com/cat.jpg")
                .build()
            )
        """
        return UserMessageBuilder()


class UserMessageBuilder:
    """Incremental builder for multimodal :class:`UserMessage` content.

    Returned by :meth:`UserMessage.builder`. Each ``text``/``image``/``audio``/
    ``video``/``file`` call appends a content part and returns ``self`` for chaining.
    Call :meth:`build` to produce the final :class:`UserMessage`.
    """

    def __init__(self) -> None:
        self._parts: list[MessageContentPartParam] = []

    def text(self, text: str) -> Self:
        """Append a text content part."""
        self._parts.append(TextContent(type="text", text=text))
        return self

    def image(self, url: str) -> Self:
        """Append an image content part.

        :param url: A public image URL or a ``data:`` URI containing base64
            image bytes (e.g. ``data:image/png;base64,iVBORw0...``). Use
            :func:`venice_ai.detect_image_format` to build a data URI from
            raw bytes if needed.
        """
        self._parts.append(ImageContent(type="image_url", image_url=ImageUrl(url=url)))
        return self

    def audio(self, data: str, format: str) -> Self:
        """Append an audio content part.

        :param data: Base64-encoded audio bytes.
        :param format: Audio format (e.g. ``"wav"``, ``"mp3"``).
        """
        self._parts.append(
            AudioContent(type="input_audio", input_audio={"data": data, "format": format})
        )
        return self

    def video(self, url: str) -> Self:
        """Append a video content part."""
        self._parts.append(VideoContent(type="video_url", video_url={"url": url}))
        return self

    def file(self, file_data: str, filename: str | None = None) -> Self:
        """Append a file content part (extracted to text server-side).

        :param file_data: A ``data:`` URL with base64 file bytes (e.g.
            ``data:application/pdf;base64,...``) or a publicly accessible URL.
            Supported: PDF, EPUB, DOCX, PPTX, XLSX/XLS, txt, md, csv, json, and
            most source-code files.
        :param filename: Optional display filename (e.g. ``"report.pdf"``).
        """
        self._parts.append(
            FileContent(type="file", file=FileObject(file_data=file_data, filename=filename))
        )
        return self

    def build(self) -> UserMessage:
        """Return a :class:`UserMessage` with the accumulated content parts.

        :raises ValueError: If no parts have been added.
        """
        if not self._parts:
            raise ValueError("UserMessageBuilder needs at least one content part")
        return UserMessage(content=list(self._parts))


class AssistantMessage(BaseModel):
    """Assistant message for chat completions"""

    role: Literal["assistant"] = "assistant"
    content: str | list[TextContent] | None = None
    name: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[Any] | None = None
    tool_calls: list[Any] | None = None

    @classmethod
    def from_response(
        cls, response: ChatCompletionResponse, choice_index: int = 0
    ) -> AssistantMessage:
        """Create an AssistantMessage from a ChatCompletionResponse for multi-turn history."""
        choice = response.choices[choice_index]
        msg = choice.message
        return cls(
            content=msg.content,  # type: ignore[arg-type]  # response is str|list[TextContent|ImageContent] but assistant only emits str|list[TextContent]
            name=msg.name,
            reasoning_content=msg.reasoning_content,
            reasoning_details=msg.reasoning_details,
            tool_calls=msg.tool_calls,
        )


class ToolMessage(BaseModel):
    """Tool message for chat completions"""

    role: Literal["tool"] = "tool"
    content: str = Field(..., description="Tool response content")
    tool_call_id: str = Field(..., description="ID of the tool call")
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[Any] | None = None


class SystemMessage(BaseModel):
    """System message for chat completions"""

    role: Literal["system"] = "system"
    content: str | list[TextContent] = Field(..., description="System message content")
    name: str | None = None


class DeveloperMessage(BaseModel):
    """Developer message for chat completions.

    Used by models that distinguish between developer-issued instructions and
    system-level priming (notably OpenAI-compatible reasoning models). Accepted
    by Venice for models advertising developer-role support.
    """

    role: Literal["developer"] = "developer"
    content: str | list[TextContent] = Field(..., description="Developer message content")
    name: str | None = None


ChatMessageParam = (
    UserMessage
    | AssistantMessage
    | SystemMessage
    | ToolMessage
    | DeveloperMessage
    | Mapping[str, Any]
)
"""What ``messages=`` accepts: a typed message model or a plain mapping.

The models are the recommended form — they give completion, field
validation at construction, and self-documenting call sites. Mappings in
the OpenAI wire shape (``{"role": "user", "content": "hi"}``) are accepted
too, and are validated into the corresponding model before the request is
built, so a mistyped role or a missing field still raises rather than
reaching the API.

Note that this is deliberately wider than ``ChatCompletionRequest.messages``,
which stays model-only: that field is the validation boundary where mappings
are coerced, so widening it there would let raw mappings through unvalidated.
"""


class ChatCompletionRequest(BaseModel):
    """Complete chat completion request"""

    # ``extra="allow"`` preserves unmodeled forward-compat kwargs (e.g. new
    # OpenAI-compat or Venice params) so they reach the wire via
    # ``model_dump`` rather than being silently dropped before the request.
    model_config = ConfigDict(extra="allow")

    model: ModelId = Field(..., description="Model ID or trait to use")
    messages: list[
        UserMessage | AssistantMessage | ToolMessage | SystemMessage | DeveloperMessage
    ] = Field(
        ...,
        description="List of messages comprising the conversation",
        min_length=1,
    )

    # Sampling parameters
    frequency_penalty: float | None = Field(
        None, ge=-2.0, le=2.0, description="Frequency penalty for token repetition"
    )
    presence_penalty: float | None = Field(
        None, ge=-2.0, le=2.0, description="Presence penalty for new topics"
    )
    repetition_penalty: float | None = Field(
        None, ge=0, description="Repetition penalty (1.0 = no penalty)"
    )
    temperature: float | None = Field(None, ge=0, le=2, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0, le=1, description="Nucleus sampling probability mass")
    top_k: int | None = Field(None, ge=0, description="Top-k sampling limit")
    min_p: float | None = Field(None, ge=0, le=1, description="Minimum probability threshold")
    min_temp: float | None = Field(
        None, ge=0, le=2, description="Minimum temperature for dynamic scaling"
    )
    max_temp: float | None = Field(
        None, ge=0, le=2, description="Maximum temperature for dynamic scaling"
    )

    # Generation control
    max_completion_tokens: int | None = Field(
        None,
        description=(
            "Maximum completion tokens to generate. On reasoning-capable models "
            "this caps TOTAL completion tokens (visible output + reasoning), not "
            "just visible output. (max_tokens was accepted as an alias in v1 but "
            "is removed in v2; passing it raises TypeError.)"
        ),
    )
    n: int | None = Field(1, description="Number of completions to generate")
    seed: int | None = Field(None, gt=0, description="Random seed for reproducibility")
    stop: str | list[str] | None = Field(None, description="Stop sequences (up to 4)")
    stop_token_ids: list[int] | None = Field(None, description="Stop token IDs")

    # Advanced features
    stream: bool | None = Field(False, description="Stream partial progress")
    stream_options: StreamOptions | None = Field(None, description="Stream configuration")
    logprobs: bool | None = Field(None, description="Include log probabilities")
    top_logprobs: int | None = Field(None, ge=0, description="Number of top logprobs to return")

    # Tools and functions
    tools: list[Tool] | None = Field(None, description="Available tools")
    tool_choice: str | SpecificToolChoice | None = Field(None, description="Tool choice control")
    parallel_tool_calls: bool | None = Field(None, description="Enable parallel function calling")

    # Response format
    response_format: JSONSchemaFormat | JSONObjectFormat | TextResponseFormat | None = Field(
        None,
        description=(
            "Response format specification. One of ``json_schema``, ``json_object``, or ``text``."
        ),
    )

    # Reasoning control
    reasoning_effort: ReasoningEffortLevel | None = Field(
        default=None,
        description=(
            "Controls thinking depth on reasoning models. One of "
            "'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'. "
            "Takes precedence over reasoning.effort if both are provided."
        ),
    )
    reasoning: ReasoningConfig | None = Field(
        default=None,
        description="Nested reasoning configuration (effort + summary style)",
    )

    # Cache routing
    prompt_cache_key: str | None = Field(
        default=None,
        description="Routing hint to improve cache hit rates across multi-turn conversations",
    )
    prompt_cache_retention: Literal["default", "extended", "24h"] | None = Field(
        default=None,
        description="Cache retention tier. ``default`` uses the standard TTL; "
        "``extended`` or ``24h`` keep the prompt cached longer for repeated use.",
    )

    # Venice-specific
    venice_parameters: VeniceParameters | None = Field(
        None, description="Venice-specific parameters"
    )

    # OpenAI-compatible passthrough fields. The Venice API proxies these to
    # the underlying model without SDK-side interpretation.
    store: bool | None = Field(
        None,
        description="OpenAI-compat: whether to store the completion on the provider side.",
    )
    text: dict[str, Any] | None = Field(
        None,
        description='OpenAI-compat text configuration (e.g. ``{"verbosity": "low"}``).',
    )
    include: list[str] | None = Field(
        None,
        description="OpenAI-compat ``include`` specifier (response-enrichment opt-ins).",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="OpenAI-compat free-form metadata attached to the request.",
    )
    verbosity: Literal["low", "medium", "high", "auto"] | None = Field(
        None,
        description=(
            "Controls the verbosity of the text response (low/medium/high/auto). "
            "Distinct from the nested ``text.verbosity`` configuration."
        ),
    )
    fallbacks: list[dict[str, str]] | None = Field(
        None,
        description=(
            "Anthropic beta parameter for Claude Fable 5 server-side refusal "
            "fallback. Array of fallback model objects (max 10), e.g. "
            "``[{'model': 'claude-opus-4-8'}]``. Forwarded only for direct "
            "Anthropic routes; ignored for other providers."
        ),
    )

    # Compatibility
    user: str | None = Field(None, description="User identifier (compatibility, discarded)")

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("Stop sequences limited to 4 items")
        return v

    anon_user_id: AnonUserId | None = Field(
        default=None,
        description=ANON_USER_ID_DESCRIPTION,
    )


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    # Message models
    "UserMessage",
    "UserMessageBuilder",
    "AssistantMessage",
    "ToolMessage",
    "SystemMessage",
    "DeveloperMessage",
    # Message unions
    "ChatMessageParam",
    # Request model
    "ChatCompletionRequest",
]
