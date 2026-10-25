"""
Chat type re-exports for Venice AI SDK.

This module provides convenient imports for chat-related types, consolidating
exports from both response and streaming modules for easy access.

The types are exported from:
- ``venice_ai.types.api.chat``: Non-streaming response types
- ``venice_ai.types.api.streaming``: Streaming chunk types
- ``venice_ai.types.api.base``: Common base types (UsageData)

Example usage:
    >>> from venice_ai.types.chat import ChatCompletionChunk, UsageData
    >>> # Process streaming chunks
    >>> async for chunk in stream:
    ...     if chunk.usage:
    ...         print(f"Total tokens: {chunk.usage.total_tokens}")
"""

# Re-export chat response types from api module
# Re-export common types from base
from .api.base import UsageData
from .api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageContentPart,
    ChatMessageImageContent,
    ChatMessageImageUrl,
    ChatMessageTextContent,
    ChatUsage,
    LogProbToken,
    ToolCall,
    ToolCallFunction,
)

# Re-export streaming types from api streaming module
from .api.streaming import (
    ChatCompletionChoiceLogprobs,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionChunkToolCall,
    ChatCompletionChunkToolCallFunction,
    ChatCompletionTokenLogprob,
    ChatCompletionTopLogprob,
    ChunkModelFactory,
)

__all__ = [
    # Response types (non-streaming)
    "ToolCallFunction",
    "ToolCall",
    "ChatMessage",
    "ChatMessageContentPart",
    "ChatMessageImageContent",
    "ChatMessageImageUrl",
    "ChatMessageTextContent",
    "LogProbToken",
    "ChatChoice",
    "ChatUsage",
    "ChatCompletionResponse",
    # Streaming types
    "ChunkModelFactory",
    "ChatCompletionTopLogprob",
    "ChatCompletionTokenLogprob",
    "ChatCompletionChoiceLogprobs",
    "ChatCompletionChunkToolCallFunction",
    "ChatCompletionChunkToolCall",
    "ChatCompletionChunkChoiceDelta",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunk",
    # Common types
    "UsageData",
]
