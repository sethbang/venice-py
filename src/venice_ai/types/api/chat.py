"""
Chat completion models for Venice AI API.

This module contains comprehensive Pydantic models for chat completion responses,
including streaming, tool calling, and Venice-specific features.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...core.models.common import (
    ImageContent,
    TextContent,
    VeniceBaseModel,
    VeniceParametersResponse,
)
from ..identifiers import ModelId
from .common import CompletionTokensDetails, PromptTokensDetails, WebSearchCitation

if TYPE_CHECKING:
    from .models import LLMModelPricing

_T = TypeVar("_T", bound=BaseModel)

# ============================================================================
# Tool Call Models (Response DTOs)
# ============================================================================


class ToolCallFunction(BaseModel):
    """Function call details in a tool call response."""

    name: str = Field(..., description="The name of the function to call")
    arguments: str = Field(
        ..., description="The arguments to call the function with, as JSON string"
    )

    @property
    def arguments_dict(self) -> dict[str, Any]:
        """Return :attr:`arguments` parsed as a JSON object.

        Convenience for the common case of dispatching tool calls — saves a
        manual ``json.loads(call.function.arguments)`` at every call site.

        :raises json.JSONDecodeError: If :attr:`arguments` is not valid JSON.
        :raises TypeError: If the parsed JSON is not an object (e.g. a bare
            list or string), since tool arguments are always object-shaped.
        """
        parsed = json.loads(self.arguments)
        if not isinstance(parsed, dict):
            raise TypeError(
                f"Tool call arguments must decode to a JSON object, got {type(parsed).__name__}"
            )
        return parsed

    def parse_as(self, model: type[_T]) -> _T:
        """Parse :attr:`arguments` into a Pydantic model.

        Mirror of :meth:`ChatCompletionResponse.parse_as` for typed
        tool-call dispatch — saves manual
        ``model.model_validate_json(call.function.arguments)`` at every
        call site.

        :param model: A Pydantic ``BaseModel`` subclass to validate against.
        :return: A validated instance of *model*.
        :raises pydantic.ValidationError: If :attr:`arguments` is not valid
            JSON or doesn't satisfy the schema (Pydantic surfaces both as
            ``ValidationError`` from :meth:`BaseModel.model_validate_json`).
        """
        return model.model_validate_json(self.arguments)


class ToolCall(BaseModel):
    """Tool call made by the model in a chat completion response."""

    id: str = Field(..., description="The ID of the tool call")
    type: Literal["function"] = Field(..., description="The type of the tool call")
    function: ToolCallFunction = Field(..., description="The function that the model called")


# ============================================================================
# Chat Completions Response Models
# ============================================================================


class ChatMessage(BaseModel):
    """Base chat message model representing a message in a chat conversation.

    Can represent messages from users, assistants, system, or tools with
    optional multimodal content support (text and images).
    """

    model_config = ConfigDict(extra="allow")

    role: str
    """The role of the message sender.

    Valid values:
    * ``"system"``: System instructions or context
    * ``"user"``: User input message
    * ``"assistant"``: AI assistant response
    * ``"tool"``: Tool/function call result
    """

    content: str | list[TextContent | ImageContent] | None = None
    """The content of the message.

    Can be:
    * A string for plain text messages
    * A list of TextContent/ImageContent for multimodal input
    * ``None`` for messages that only contain tool calls
    """

    name: str | None = None
    """Optional name identifier for the message sender.

    Can be used to distinguish between multiple users or to name
    tool calls. Useful in multi-user conversations or for tracking
    specific participants.
    """

    reasoning_content: str | None = None
    """Content from reasoning/thinking models showing thought process.

    For models with reasoning capabilities (like DeepSeek R1), contains
    the model's internal reasoning steps before generating the final
    answer. Only present when using reasoning models.
    """

    tool_calls: list[ToolCall] | None = None
    """List of tool/function calls made by the assistant.

    Present when the model decides to call one or more functions/tools.
    Each tool call includes the function name and arguments. Client must
    execute these calls and return results via tool messages.
    """

    reasoning_details: list[Any] | None = None
    """Structured reasoning blocks emitted by reasoning-capable models.

    The API instructs clients to pass these back verbatim on subsequent
    turns to preserve thought signatures, so they must survive the round
    trip rather than being dropped on parse.
    """


class LogProbToken(BaseModel):
    """Log probability information for a token"""

    model_config = ConfigDict(extra="allow")

    token: str = Field(..., description="The token string")
    logprob: float = Field(..., description="The log probability of this token")
    bytes: list[int] | None = Field(None, description="Raw bytes of the token")
    top_logprobs: list["LogProbToken"] | None = Field(
        None, description="Top tokens considered with their log probabilities"
    )


KNOWN_FINISH_REASONS: Final[tuple[str, ...]] = (
    "stop",
    "length",
    "tool_calls",
    "content_filter",
)
"""``finish_reason`` values this SDK release knows about.

:attr:`ChatChoice.finish_reason` and
:attr:`~venice_ai.types.api.streaming.ChatCompletionChunkChoice.finish_reason`
are plain ``str`` so a reason the server adds later is preserved rather than
failing the whole response parse. Narrow against this tuple when branching on
a known reason.
"""


class ChatChoice(BaseModel):
    """Chat completion choice object representing a single completion result.

    When multiple completions are requested (n > 1), each choice represents
    one possible completion. Contains the generated message and metadata.
    """

    model_config = ConfigDict(extra="allow")

    index: int = Field(..., description="The index of the choice in the list")
    message: ChatMessage = Field(..., description="Assistant message response")
    finish_reason: str | None = Field(
        ...,
        description=(
            "The reason the completion finished, e.g. ``'stop'``, ``'length'``, "
            "``'tool_calls'`` or ``'content_filter'``. Known values are in "
            "KNOWN_FINISH_REASONS."
        ),
    )
    logprobs: LogProbToken | None = None
    """Log probability information for tokens in this completion.

    Only present if ``logprobs`` parameter was set to ``True`` in the
    request. Contains probability information for understanding model
    confidence in token selections.
    """

    stop_reason: str | None = Field(None, description="The reason the completion stopped")

    @field_validator("stop_reason", mode="before")
    @classmethod
    def _coerce_stop_reason(cls, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)


class ChatUsage(BaseModel):
    """Token usage information"""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(..., description="The number of tokens in the prompt")
    completion_tokens: int = Field(..., description="The number of tokens in the completion")
    total_tokens: int = Field(..., description="The total number of tokens used")
    prompt_tokens_details: PromptTokensDetails | None = Field(
        None, description="Breakdown of tokens used in the prompt"
    )
    completion_tokens_details: CompletionTokensDetails | None = Field(
        default=None,
        description=(
            "Breakdown of tokens generated in the completion. Populated by reasoning "
            "models (reasoning_tokens) and multi-modal output models (audio_tokens, "
            "image_tokens)."
        ),
    )
    cache_read_input_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Top-level count of prompt tokens served from the prompt cache. Mirrors "
            "``prompt_tokens_details.cached_tokens`` on models that emit it."
        ),
    )
    cache_creation_input_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Top-level count of prompt tokens written to the prompt cache (cache "
            "write), charged at a premium rate by some providers. Mirrors "
            "``prompt_tokens_details.cache_creation_input_tokens`` on models that emit it."
        ),
    )

    def __str__(self) -> str:
        """Concise human-readable summary, suitable for logs and notebooks.

        Includes cache and reasoning-token breakdowns when present::

            >>> str(response.usage)
            'prompt: 1234 / completion: 567 / total: 1801'

            >>> # with cache hits and reasoning tokens
            'prompt: 1234 (cache: 1100) / completion: 567 (reasoning: 200) / total: 1801'
        """
        parts: list[str] = []
        prompt = f"prompt: {self.prompt_tokens}"
        if self.cache_read_input_tokens:
            prompt += f" (cache: {self.cache_read_input_tokens})"
        parts.append(prompt)

        completion = f"completion: {self.completion_tokens}"
        details = self.completion_tokens_details
        if details is not None and getattr(details, "reasoning_tokens", None):
            completion += f" (reasoning: {details.reasoning_tokens})"
        parts.append(completion)

        parts.append(f"total: {self.total_tokens}")
        return " / ".join(parts)


# WebSearchCitation and VeniceParameters are imported from base and requests.common
# to avoid duplicate definitions


class ChatCost(BaseModel):
    """Request cost split by billing currency.

    Carried as the top-level ``cost`` field on a non-streaming chat completion
    response. ``diem``/``usd`` are left optional (rather than required) so a
    response assembled from stream chunks — which may not carry the cost block —
    never fails to parse.
    """

    model_config = ConfigDict(extra="allow")

    diem: float | None = Field(None, description="DIEM-denominated portion of the request cost")
    usd: float | None = Field(None, description="USD-denominated portion of the request cost")


class ChatCompletionResponse(VeniceBaseModel):
    """Complete chat completion response

    Note: This model allows extra fields for backward compatibility with
    API responses that may contain additional undocumented fields.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="The ID of the request")
    object: Literal["chat.completion"] = Field(..., description="The type of object returned")
    created: int = Field(..., description="Unix timestamp of request creation")
    model: ModelId = Field(..., description="The model id used for the request")
    choices: list[ChatChoice] = Field(
        default_factory=list,
        description=(
            "List of chat completion choices. Optional per the API spec — certain "
            "models may omit it under certain conditions — so it defaults to []."
        ),
    )
    usage: ChatUsage | None = Field(
        None,
        description=(
            "Token usage information. Always populated for non-streaming responses; "
            "may be ``None`` on assembled-from-stream responses when no usage chunk arrived."
        ),
    )
    prompt_logprobs: dict[str, Any] | None = Field(
        None, description="Log probability information for the prompt"
    )
    venice_parameters: VeniceParametersResponse | None = Field(
        None, description="Venice-specific parameters"
    )
    # Additional fields that may be present in responses
    service_tier: str | None = Field(None, description="Service tier used for the request")
    system_fingerprint: str | None = Field(
        None, description="System fingerprint for the completion"
    )
    kv_transfer_params: dict[str, Any] | None = Field(
        None, description="Key-value transfer parameters"
    )
    cost: ChatCost | None = Field(None, description="Request cost split by billing currency")

    # ── Content accessors ─────────────────────────────────────────────

    @property
    def text(self) -> str | None:
        """First choice's text content, normalized to a plain string.

        Replaces the boilerplate ``response.choices[0].message.content`` access:
        handles the ``str | list[TextContent | ImageContent] | None`` union by
        returning ``str`` content unchanged, joining the ``text`` of
        :class:`TextContent` parts in a multimodal list (image/audio/video
        parts are skipped), and returning ``None`` if the response has no
        choices, no content, or only non-text parts.
        """
        if not self.choices:
            return None
        content = self.choices[0].message.content
        if content is None:
            return None
        if isinstance(content, str):
            return content
        texts = [part.text for part in content if isinstance(part, TextContent)]
        return "".join(texts) if texts else None

    # ── Structured output helpers ─────────────────────────────────────

    @property
    def web_search_citations(self) -> list[WebSearchCitation]:
        """Web-search citations for this response.

        Sourced from ``venice_parameters.web_search_citations`` (where the API
        actually returns them); ``[]`` when web search was not used.
        """
        return self.venice_parameters.web_search_citations if self.venice_parameters else []

    @property
    def parsed(self) -> dict[str, Any] | list[Any] | None:
        """Parse the first choice's content as JSON.

        Use with structured output (``response_format``).

        :raises json.JSONDecodeError: If content is not valid JSON.
        :raises TypeError: If content is multimodal (list of content parts).
        :returns: Parsed JSON, or ``None`` if there are no choices or content is ``None``.
        """
        if not self.choices:
            return None
        content = self.choices[0].message.content
        if content is None:
            return None
        if not isinstance(content, str):
            raise TypeError("Cannot parse multimodal content as JSON")
        result: dict[str, Any] | list[Any] = json.loads(content)
        return result

    def parse_as(self, model: type[_T], choice_index: int = 0) -> _T:
        """Parse a choice's content into a Pydantic model.

        :param model: A Pydantic ``BaseModel`` subclass to validate against.
        :param choice_index: Which choice to parse (default ``0``).
        :raises ValueError: If the choice has no content or is multimodal.
        :raises json.JSONDecodeError: If content is not valid JSON.
        :raises pydantic.ValidationError: If JSON doesn't match *model*.
        """
        if choice_index >= len(self.choices):
            raise ValueError(f"Response has no choice at index {choice_index} (choices is empty)")
        content = self.choices[choice_index].message.content
        if content is None:
            raise ValueError(f"Choice {choice_index} has no content (may be a tool call response)")
        if not isinstance(content, str):
            raise ValueError(f"Choice {choice_index} has multimodal content, not a JSON string")
        return model.model_validate_json(content)

    def summary(self, *, pricing: "LLMModelPricing | None" = None) -> str:
        """One-line summary suitable for logs and notebook output.

        Format::

            model · prompt: 1234 / completion: 567 / total: 1801 · finish=stop

        With ``pricing`` supplied, an exact-cost segment is inserted::

            model · prompt: 1234 / completion: 567 / total: 1801 · $0.0034 · finish=stop

        Pricing is intentionally an explicit parameter rather than fetched
        from the registry — the registry lookup is async, and ``summary()``
        stays sync. Callers who want cost should fetch once and pass it::

            pricing = (await client.models.get(response.model)).model_spec.pricing
            print(response.summary(pricing=pricing))

        Missing usage / no choices / no pricing → that segment is omitted
        rather than raising. Always returns a non-empty string.

        :param pricing: Optional pricing for the model. If provided, includes
            an exact USD cost computed via ``calculate_completion_cost``.
        :return: Pipe-style summary line.
        """
        parts: list[str] = [self.model]
        if self.usage is not None:
            parts.append(str(self.usage))
        if pricing is not None:
            from ...costs import calculate_completion_cost

            cost = calculate_completion_cost(self, pricing)
            parts.append(f"${cost['usd']:.4f}")
        if self.choices and self.choices[0].finish_reason is not None:
            parts.append(f"finish={self.choices[0].finish_reason}")
        return " · ".join(parts)

    @property
    def thinking_blocks(self) -> list[str]:
        """Reasoning/thinking blocks from the first choice's message.

        Two server shapes are handled transparently:

        * **Separate field** (e.g. ``zai-org-glm-4.7``): the assistant
          message's ``reasoning_content`` is populated with the raw
          chain-of-thought, and ``content`` carries the user-facing
          answer alone. This branch returns ``[reasoning_content]``.

        * **Tagged inline** (e.g. ``venice-uncensored``): the assistant
          message's ``content`` carries ``<thinking>...</thinking>`` or
          ``<think>...</think>`` blocks. This branch returns each block's
          inner text, in document order, with surrounding tags stripped.

        :return: List of thinking-block strings, or ``[]`` if the response
            has no choices, no thinking content in either shape, or only
            multimodal non-text parts.
        """
        if not self.choices:
            return []

        msg = self.choices[0].message

        # Path A: dedicated reasoning_content field
        rc = msg.reasoning_content
        if rc is not None and rc.strip():
            return [rc]

        # Path B: regex-extract <think>/<thinking> tags from content.
        from ...helpers import extract_thinking_blocks

        content = msg.content
        if content is None:
            return []
        if isinstance(content, str):
            blocks, _ = extract_thinking_blocks(content)
            return blocks
        # Multimodal: only consider text parts.
        text_parts = [p.text for p in content if isinstance(p, TextContent)]
        if not text_parts:
            return []
        blocks, _ = extract_thinking_blocks("".join(text_parts))
        return blocks


# ============================================================================
# Parsed Response Wrapper (auto-validated structured output)
# ============================================================================


@dataclass(frozen=True)
class ParsedChatCompletion[T: BaseModel]:
    """Result of :meth:`client.chat.completions.parse` — paired raw + validated.

    ``parse`` is the auto-validating sibling of ``create``: callers pass a
    Pydantic ``BaseModel`` subclass as ``response_format`` and get back this
    wrapper containing both the original :class:`ChatCompletionResponse`
    (for usage / finish_reason / metadata) and the validated typed instance
    of their model.

    :param response: The raw chat completion response.
    :param parsed: The first choice's content validated against the
        Pydantic model the caller passed to ``parse``.
    """

    response: "ChatCompletionResponse"
    parsed: T

    @property
    def usage(self) -> "ChatUsage | None":
        """Convenience pass-through to ``response.usage``."""
        return self.response.usage

    @property
    def finish_reason(self) -> str | None:
        """Convenience pass-through to ``response.choices[0].finish_reason``."""
        if not self.response.choices:
            return None
        return self.response.choices[0].finish_reason


@dataclass(frozen=True)
class ToolLoopResult:
    """Result of :meth:`client.chat.completions.run_with_tools`.

    Pairs the final assistant response (the one that ended with
    ``finish_reason != "tool_calls"``) with the full message history
    produced during the tool-orchestration loop, so callers don't have to
    rebuild the timeline themselves.

    :param response: The terminal :class:`ChatCompletionResponse`.
    :param messages: Full history — the input messages followed by every
        assistant tool-call turn and tool-result message the loop produced.
        Suitable for feeding into a follow-up call.
    :param iterations: Number of model round trips before convergence
        (1 == "model didn't request any tools"; higher == that many
        tool-call → tool-result cycles plus the terminal response).
    """

    response: "ChatCompletionResponse"
    messages: list[Any]
    iterations: int

    @property
    def text(self) -> str | None:
        """Convenience pass-through to ``response.text``."""
        return self.response.text

    @property
    def usage(self) -> "ChatUsage | None":
        """Convenience pass-through to ``response.usage``."""
        return self.response.usage

    @property
    def finish_reason(self) -> str | None:
        """Convenience pass-through to ``response.choices[0].finish_reason``."""
        if not self.response.choices:
            return None
        return self.response.choices[0].finish_reason


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    "ToolCallFunction",
    "ToolCall",
    "ChatMessage",
    "LogProbToken",
    "ChatChoice",
    "ChatUsage",
    "ChatCost",
    "WebSearchCitation",
    "VeniceParametersResponse",
    "ChatCompletionResponse",
    "ParsedChatCompletion",
    "ToolLoopResult",
]
