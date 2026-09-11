"""
Streaming response models for Venice AI API.

This module contains specialized Pydantic models for streaming operations
across various Venice AI endpoints. Streaming models provide incremental
updates as responses are generated in real-time.

**Module Organization:**

* **Streaming Models**: Specialized models for real-time chat completion streaming
* **Protocol Definitions**: Type protocols for streaming model factories
* **Supporting Types**: Log probabilities referenced by streaming models

**Key Models:**

* **ChatCompletionChunk**: The main streaming response chunk model
* **ChatCompletionChunkChoice**: Individual choice within a streaming chunk
* **ChatCompletionChunkChoiceDelta**: Incremental content deltas in streaming
* **ChatCompletionChunkToolCall**: Tool call information in streaming chunks
* **ChunkModelFactory**: Protocol for streaming model instantiation

**Note:** Non-streaming response models (ChatCompletion, ChatCompletionChoice, etc.)
are in ``src/venice_ai/types/api/chat.py``. Request models are
in ``src/venice_ai/types/api/requests/``. This module focuses exclusively
on streaming-specific functionality.
"""

from typing import (
    Any,
    Literal,
    Protocol,
    TypeVar,
)

from pydantic import BaseModel, Field

# Import ChatUsage from the chat module so streaming chunks expose the same
# rich usage shape (detail + cache breakdowns, extra="allow") as non-streaming
# responses, rather than the narrower base ``UsageData`` which dropped them.
from .chat import ChatUsage

__all__ = [
    # Protocol
    "ChunkModelFactory",
    # Log probability types (used by streaming)
    "ChatCompletionTopLogprob",
    "ChatCompletionTokenLogprob",
    "ChatCompletionChoiceLogprobs",
    # Streaming models
    "ChatCompletionChunkToolCallFunction",
    "ChatCompletionChunkToolCall",
    "ChatCompletionChunkChoiceDelta",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunk",
]

# --- Protocol Definitions ---

_ChunkModelT = TypeVar("_ChunkModelT", covariant=True)


class ChunkModelFactory(Protocol[_ChunkModelT]):
    """
    A protocol for classes that can be instantiated from keyword arguments.
    Used to define the expected interface for `stream_cls` in chat completions,
    where the class's __init__ method should accept ``**data``.
    """

    def __init__(self, **data: Any) -> None: ...


# --- Logprobs Types (Used by Streaming) ---


class ChatCompletionTopLogprob(BaseModel):
    """
    Represents log probability information for alternative tokens at a specific position.

    Provides probability data for tokens that were considered but not selected
    at a particular position in the generation, useful for understanding model
    confidence and exploring alternative token choices.
    """

    token: str
    """The token string that was considered at this position.

    One of the top-N most probable tokens at this position according
    to the model's probability distribution.
    """

    logprob: float
    """The log probability of this token.

    Natural logarithm of the token's probability. More negative values
    indicate lower probability. Can be exponentiated to get actual probability.
    """

    bytes: list[int] | None = Field(default=None)
    """Raw byte representation of the token, if available.

    UTF-8 byte sequence for the token. Useful for low-level token analysis
    and handling of special or non-printable characters.
    """


class ChatCompletionTokenLogprob(BaseModel):
    """
    Contains comprehensive log probability information for a single token.

    Provides complete probability analysis for a generated token including
    the token itself, its probability, and alternative tokens that were
    considered at the same position.
    """

    token: str
    """The actual token that was selected and generated.

    The token string that appears in the final completion. This was
    the highest probability token chosen from the model's distribution.
    """

    logprob: float
    """The log probability of the selected token.

    Natural logarithm of the token's probability according to the model.
    Higher (less negative) values indicate higher confidence in the selection.
    """

    bytes: list[int] | None = Field(default=None)
    """Raw byte representation of the token, if available.

    UTF-8 byte sequence for the generated token. Useful for precise
    token analysis and handling of special characters.
    """

    top_logprobs: list[ChatCompletionTopLogprob] | None = Field(default=None)
    """Alternative tokens that were considered at this position.

    List of the top-N most probable tokens (including the selected one)
    with their probabilities. Only present if ``top_logprobs`` parameter
    was specified in the request. Useful for analyzing model uncertainty.
    """


class ChatCompletionChoiceLogprobs(BaseModel):
    """
    Aggregates log probability information for all tokens in a completion choice.
    """

    content: list[ChatCompletionTokenLogprob] | None = Field(
        default=None,
        description="List of token-level log probability information for each token in the completion",
    )


# --- Streaming Types ---


class ChatCompletionChunkToolCallFunction(BaseModel):
    """
    Represents function call details within a streaming chat completion chunk.
    Fields are optional as they arrive incrementally.
    """

    name: str | None = Field(
        default=None,
        description="Incremental function name as it streams in. May be partial until complete.",
    )
    arguments: str | None = Field(
        default=None,
        description="Incremental JSON-encoded function arguments as they stream in. Accumulate across chunks.",
    )


class ChatCompletionChunkToolCall(BaseModel):
    """
    Represents an incremental tool call within a streaming chat completion chunk.
    Fields are optional as they arrive incrementally.
    """

    id: str | None = Field(
        default=None,
        description="Unique identifier for this tool call, present once the tool call begins streaming",
    )
    type: Literal["function"] | None = Field(
        default=None,
        description="Type of tool being called, currently only 'function' is supported",
    )
    function: ChatCompletionChunkToolCallFunction | None = Field(
        default=None,
        description="Incremental function call details streaming in as chunks arrive",
    )
    index: int | None = Field(
        default=None,
        description="Index for parallel tool calls. Used when multiple tools are called simultaneously to distinguish between different concurrent tool calls in the stream.",
    )


class ChatCompletionChunkChoiceDelta(BaseModel):
    """
    Contains the incremental changes for a choice in a streaming chat completion.
    """

    role: Literal["system", "user", "assistant", "tool"] | None = Field(default=None)
    """The role for this delta.

    Typically only present in the first chunk of a streaming response to
    indicate the role of the message being streamed.
    """

    content: str | None = Field(default=None)
    """Incremental content for this chunk.

    Each chunk contains a small piece of the complete message. Content must
    be concatenated across chunks to build the full response.
    """

    reasoning_content: str | None = Field(
        default=None, description="Reasoning/thinking content from reasoning models"
    )
    """Reasoning/thinking content from reasoning models.

    Present when a reasoning model (e.g., DeepSeek-R1) emits thinking tokens
    during streaming. Should be accumulated across chunks separately from
    regular content.
    """

    reasoning_encrypted: bool | None = Field(default=None)
    """Marks this delta's ``reasoning_content`` as an opaque encrypted block.

    Reasoning models emit an encrypted reasoning item — a sentinel header and an
    opaque token — that the API expects back verbatim on subsequent turns. It
    arrives whole in a single delta, flagged with this field, interleaved with
    the plain-language summary deltas around it.

    The block is unterminated, so the server parses it from its header to the
    end of the field. Any summary text concatenated *after* it is absorbed into
    the token and the next turn is rejected with "encrypted content could not be
    decrypted or parsed". Accumulators must therefore keep flagged deltas apart
    from summary deltas and place the block last — see
    :meth:`~venice_ai.streaming.ChatStream.collect`.
    """

    tool_calls: list[ChatCompletionChunkToolCall] | None = Field(default=None)
    """Incremental tool call information.

    Present when the model is streaming a tool call. Each chunk may contain
    partial tool call data that must be accumulated across chunks.
    """


class ChatCompletionChunkChoice(BaseModel):
    """
    Represents a single choice within a streaming chat completion chunk.
    """

    index: int
    """Index of this choice.

    When ``n`` parameter is greater than 1, identifies which completion
    choice this chunk belongs to.
    """

    delta: ChatCompletionChunkChoiceDelta
    """The incremental changes in this chunk.

    Contains role, content, or tool call deltas that should be accumulated
    to build the complete message.
    """

    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    """The reason the model stopped generating (only in final chunk).

    * ``"stop"``: Model hit a natural stopping point
    * ``"length"``: Maximum token limit reached
    * ``"tool_calls"``: Model called a tool

    ``None`` for all chunks except the final one.
    """

    logprobs: ChatCompletionChoiceLogprobs | None = Field(default=None)
    """Log probability information for tokens in this chunk.

    Only present if ``logprobs`` parameter was enabled. Typically not included
    in streaming responses but supported for completeness.
    """


class ChatCompletionChunk(BaseModel):
    """
    Represents a single chunk in a streaming chat completion response.
    """

    id: str
    """Unique identifier for this chat completion stream.

    Same ID across all chunks in a single streaming response.
    Format: ``"chatcmpl-{random_string}"``
    """

    object: Literal["chat.completion.chunk"]
    """Object type identifier, always ``"chat.completion.chunk"`` for streaming chunks.

    Distinguishes streaming responses from non-streaming completions which
    use ``"chat.completion"``.
    """

    created: int
    """Unix timestamp (seconds since epoch) when the stream was created.

    Same timestamp across all chunks in a single streaming response.
    """

    model: str
    """The model ID being used for this streaming completion.

    Same across all chunks. May differ from requested model if a trait or
    compatibility mapping was used.
    """

    choices: list[ChatCompletionChunkChoice]
    """List of choice deltas for this chunk.

    Contains incremental content that should be accumulated across chunks
    to build the complete response.
    """

    usage: ChatUsage | None = Field(default=None)
    """Token usage statistics (only in final chunk if requested).

    Only present if ``stream_options.include_usage`` is ``True`` in the request.
    Provides final token counts in the last chunk of the stream. Uses the same
    :class:`~venice_ai.types.api.chat.ChatUsage` model as non-streaming
    responses, so detail breakdowns (``prompt_tokens_details``,
    ``completion_tokens_details``) and cache counters
    (``cache_read_input_tokens``, ``cache_creation_input_tokens``) carried on
    the final chunk survive parsing instead of being dropped.
    """

    system_fingerprint: str | None = Field(default=None)
    """System fingerprint for tracking backend configuration.

    Used for debugging and tracking which backend configuration generated
    the response.
    """

    @property
    def text(self) -> str:
        """Convenience accessor for ``choices[0].delta.content``.

        Returns the incremental content for this chunk's first choice as a
        string, or ``""`` if the chunk has no choices or the delta carries
        no content (typical for the first/last chunks that only carry
        ``role`` or ``finish_reason``). Mirrors
        :attr:`ChatCompletionResponse.text` so the same accumulation pattern
        works for both streaming and non-streaming responses::

            full = ""
            async for chunk in stream:
                full += chunk.text
        """
        if not self.choices:
            return ""
        return self.choices[0].delta.content or ""


# NOTE: This module contains only streaming-specific models.
# Non-streaming response models are in src/venice_ai/types/api/chat.py
# Request models are in src/venice_ai/types/api/requests/
# ChatUsage (from .chat) is used for both streaming chunks and non-streaming
# responses; UsageData in base.py is a legacy public export.
