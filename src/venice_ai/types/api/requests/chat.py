"""
Chat completion request models for Venice.ai API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from ..chat import ChatCompletionResponse

from ...identifiers import ModelId
from .common import (
    ANON_USER_ID_DESCRIPTION,
    AnonUserId,
    AudioContent,
    CacheControlInput,
    FileContent,
    FileObject,
    ImageContent,
    ImageDetail,
    ImageUrl,
    JSONObjectFormat,
    JSONSchemaFormat,
    MessageContentPartParam,
    ReasoningConfig,
    ReasoningEffortLevel,
    SpecificToolChoice,
    StreamOptions,
    TextContent,
    TextContentParam,
    TextResponseFormat,
    Tool,
    VeniceParameters,
    VideoContent,
)

# ============================================================================
# Message Models
# ============================================================================

TextPartParam = Annotated[TextContent | TextContentParam, Field(union_mode="left_to_right")]
"""A text part of a system or developer message: a :class:`TextContent`, or its
dict form. A dict carrying keys TextContent does not declare is sent as given.
"""


class UserMessage(BaseModel):
    """User message for chat completions"""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

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

    def image(self, url: str, *, detail: ImageDetail | None = None) -> Self:
        """Append an image content part.

        :param url: A public image URL or a ``data:`` URI containing base64
            image bytes (e.g. ``data:image/png;base64,iVBORw0...``). Use
            :func:`venice_ai.detect_image_format` to build a data URI from
            raw bytes if needed.
        :param detail: Image detail level: ``"auto"``, ``"low"``, ``"high"``
            or ``"original"``. Explicit ``"high"`` may increase image-token
            usage and cost; ``"original"`` is kept on direct OpenAI routes and
            treated as ``"high"`` elsewhere. Omitted from the request when
            unset.
        """
        self._parts.append(
            ImageContent(type="image_url", image_url=ImageUrl(url=url, detail=detail))
        )
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

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    role: Literal["assistant"] = "assistant"
    content: (
        str
        | list[Annotated[TextContent | dict[str, Any], Field(union_mode="left_to_right")]]
        | None
    ) = Field(
        default=None,
        description=(
            "Text, or content parts. A text part with only the fields TextContent "
            "declares becomes TextContent; any other part, including a text part "
            "with extra keys, is sent as the dict given."
        ),
    )
    name: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[Any] | None = None
    tool_calls: list[Any] | None = None
    thought_signature: str | None = Field(
        default=None, description="Signature a model returned with this message, passed back"
    )
    cache_control: CacheControlInput | None = Field(
        default=None, description="Prompt-caching breakpoint at the end of this message"
    )

    @field_validator("content")
    @classmethod
    def _reject_malformed_text_parts(cls, v: Any) -> Any:
        """A text part that failed TextContent validation must not pass as a dict.

        Only the fields TextContent declares are checked, so a part that also
        carries keys the SDK does not model is sent as the dict it is.
        """
        if isinstance(v, list):
            for part in v:
                if isinstance(part, dict) and part.get("type") == "text":
                    TextContent.model_validate(
                        {k: val for k, val in part.items() if k in TextContent.model_fields}
                    )
        return v

    @classmethod
    def from_response(
        cls, response: ChatCompletionResponse, choice_index: int = 0
    ) -> AssistantMessage:
        """Create an AssistantMessage from a ChatCompletionResponse for multi-turn history.

        Fields the SDK does not model are carried over as well, so a value a
        model expects back on the next turn is not lost. The ``cache_control``
        the API echoes, on the message or on a content part, is not: where to
        place a cache breakpoint is the caller's choice.
        """
        choice = response.choices[choice_index]
        msg = choice.message
        content: str | list[TextContent | dict[str, Any]] | None
        if isinstance(msg.content, list):
            content = [_without_cache_control(part) for part in msg.content]
        else:
            content = msg.content
        extra = {k: v for k, v in (msg.model_extra or {}).items() if k != "cache_control"}
        return cls(
            content=content,
            name=msg.name,
            reasoning_content=msg.reasoning_content,
            reasoning_details=msg.reasoning_details,
            tool_calls=msg.tool_calls,
            thought_signature=msg.thought_signature,
            **extra,
        )


def _without_cache_control(part: Any) -> TextContent | dict[str, Any]:
    """A response content part ready for history, minus its echoed ``cache_control``.

    A text part stays a :class:`TextContent` unless it carries keys the SDK does
    not model; then it goes back as a dict, so those keys are sent too.
    """
    if isinstance(part, TextContent) and not part.model_extra:
        return part.model_copy(update={"cache_control": None}) if part.cache_control else part
    if not isinstance(part, dict):
        part = part.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in part.items() if k != "cache_control"}


class ToolMessage(BaseModel):
    """Tool message for chat completions"""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    role: Literal["tool"] = "tool"
    content: str = Field(..., description="Tool response content")
    tool_call_id: str = Field(..., description="ID of the tool call")
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[Any] | None = None
    cache_control: CacheControlInput | None = Field(
        default=None, description="Prompt-caching breakpoint at the end of this message"
    )


class SystemMessage(BaseModel):
    """System message for chat completions"""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    role: Literal["system"] = "system"
    content: str | list[TextPartParam] = Field(..., description="System message content")
    name: str | None = None


class DeveloperMessage(BaseModel):
    """Developer message for chat completions.

    Used by models that distinguish between developer-issued instructions and
    system-level priming (notably OpenAI-compatible reasoning models). Accepted
    by Venice for models advertising developer-role support.
    """

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    role: Literal["developer"] = "developer"
    content: str | list[TextPartParam] = Field(..., description="Developer message content")
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
reaching the API. A message-level key the models do not declare is sent as
given, so a response message passed back through
:meth:`AssistantMessage.from_response` or ``model_dump()`` keeps it.

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
        default=None, ge=-2.0, le=2.0, description="Frequency penalty for token repetition"
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty for new topics"
    )
    repetition_penalty: float | None = Field(
        default=None, ge=0, description="Repetition penalty (1.0 = no penalty)"
    )
    temperature: float | None = Field(default=None, ge=0, le=2, description="Sampling temperature")
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    top_k: int | None = Field(default=None, ge=0, description="Top-k sampling limit")
    min_p: float | None = Field(
        default=None, ge=0, le=1, description="Minimum probability threshold"
    )
    min_temp: float | None = Field(
        default=None, ge=0, le=2, description="Minimum temperature for dynamic scaling"
    )
    max_temp: float | None = Field(
        default=None, ge=0, le=2, description="Maximum temperature for dynamic scaling"
    )

    # Generation control
    max_completion_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum completion tokens to generate. On reasoning-capable models "
            "this caps TOTAL completion tokens (visible output + reasoning), not "
            "just visible output. (max_tokens was accepted as an alias in v1 but "
            "is removed in v2; passing it raises TypeError.)"
        ),
    )
    n: int | None = Field(default=1, description="Number of completions to generate")
    seed: int | None = Field(default=None, gt=0, description="Random seed for reproducibility")
    stop: str | list[str] | None = Field(default=None, description="Stop sequences (up to 4)")
    stop_token_ids: list[int] | None = Field(default=None, description="Stop token IDs")

    # Advanced features
    stream: bool | None = Field(default=False, description="Stream partial progress")
    stream_options: StreamOptions | None = Field(default=None, description="Stream configuration")
    logprobs: bool | None = Field(default=None, description="Include log probabilities")
    top_logprobs: int | None = Field(
        default=None, ge=0, description="Number of top logprobs to return"
    )

    # Tools and functions
    tools: list[Tool] | None = Field(default=None, description="Available tools")
    tool_choice: str | SpecificToolChoice | None = Field(
        default=None, description="Tool choice control"
    )
    parallel_tool_calls: bool | None = Field(
        default=None, description="Enable parallel function calling"
    )

    # Response format
    response_format: JSONSchemaFormat | JSONObjectFormat | TextResponseFormat | None = Field(
        default=None,
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
        default=None, description="Venice-specific parameters"
    )

    # OpenAI-compatible passthrough fields. The Venice API proxies these to
    # the underlying model without SDK-side interpretation.
    store: bool | None = Field(
        default=None,
        description="OpenAI-compat: whether to store the completion on the provider side.",
    )
    text: dict[str, Any] | None = Field(
        default=None,
        description='OpenAI-compat text configuration (e.g. ``{"verbosity": "low"}``).',
    )
    include: list[str] | None = Field(
        default=None,
        description="OpenAI-compat ``include`` specifier (response-enrichment opt-ins).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="OpenAI-compat free-form metadata attached to the request.",
    )
    verbosity: Literal["low", "medium", "high", "auto"] | None = Field(
        default=None,
        description=(
            "Controls the verbosity of the text response (low/medium/high/auto). "
            "Distinct from the nested ``text.verbosity`` configuration."
        ),
    )
    fallbacks: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Anthropic beta parameter for Claude Fable 5 server-side refusal "
            "fallback. Array of fallback model objects (max 10), e.g. "
            "``[{'model': 'claude-opus-4-8'}]``. Forwarded only for direct "
            "Anthropic routes; ignored for other providers."
        ),
    )

    # Compatibility
    user: str | None = Field(default=None, description="User identifier (compatibility, discarded)")

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
    "TextPartParam",
    # Request model
    "ChatCompletionRequest",
]
