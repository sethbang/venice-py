"""
Unit tests for Venice AI chat type re-exports.

Tests that the chat module correctly re-exports types from canonical locations.
"""

# Import from the re-export module to cover import lines
from venice_ai.types.chat import (
    ChatChoice,
    ChatCompletionChoiceLogprobs,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionChunkToolCall,
    ChatCompletionChunkToolCallFunction,
    ChatCompletionResponse,
    ChatCompletionTokenLogprob,
    ChatCompletionTopLogprob,
    ChatMessage,
    ChatUsage,
    # Streaming types from generated.streaming (line 32-42)
    ChunkModelFactory,
    LogProbToken,
    ToolCall,
    # Response types from generated.chat (line 21-29)
    ToolCallFunction,
    # Common types from generated.base (line 45)
    UsageData,
)


class TestChatResponseReExports:
    """Test that chat response types are properly re-exported."""

    def test_tool_call_function_is_importable(self):
        """Test ToolCallFunction type is importable."""
        assert ToolCallFunction is not None
        assert hasattr(ToolCallFunction, "model_fields") or hasattr(ToolCallFunction, "__fields__")

    def test_tool_call_is_importable(self):
        """Test ToolCall type is importable."""
        assert ToolCall is not None
        assert hasattr(ToolCall, "model_fields") or hasattr(ToolCall, "__fields__")

    def test_chat_message_is_importable(self):
        """Test ChatMessage type is importable."""
        assert ChatMessage is not None
        assert hasattr(ChatMessage, "model_fields") or hasattr(ChatMessage, "__fields__")

    def test_log_prob_token_is_importable(self):
        """Test LogProbToken type is importable."""
        assert LogProbToken is not None
        assert hasattr(LogProbToken, "model_fields") or hasattr(LogProbToken, "__fields__")

    def test_chat_choice_is_importable(self):
        """Test ChatChoice type is importable."""
        assert ChatChoice is not None
        assert hasattr(ChatChoice, "model_fields") or hasattr(ChatChoice, "__fields__")

    def test_chat_usage_is_importable(self):
        """Test ChatUsage type is importable."""
        assert ChatUsage is not None
        assert hasattr(ChatUsage, "model_fields") or hasattr(ChatUsage, "__fields__")

    def test_chat_completion_response_is_importable(self):
        """Test ChatCompletionResponse type is importable."""
        assert ChatCompletionResponse is not None
        assert hasattr(ChatCompletionResponse, "model_fields") or hasattr(
            ChatCompletionResponse, "__fields__"
        )


class TestChatStreamingReExports:
    """Test that streaming types are properly re-exported."""

    def test_chunk_model_factory_is_importable(self):
        """Test ChunkModelFactory type is importable."""
        assert ChunkModelFactory is not None

    def test_chat_completion_top_logprob_is_importable(self):
        """Test ChatCompletionTopLogprob type is importable."""
        assert ChatCompletionTopLogprob is not None
        assert hasattr(ChatCompletionTopLogprob, "model_fields") or hasattr(
            ChatCompletionTopLogprob, "__fields__"
        )

    def test_chat_completion_token_logprob_is_importable(self):
        """Test ChatCompletionTokenLogprob type is importable."""
        assert ChatCompletionTokenLogprob is not None
        assert hasattr(ChatCompletionTokenLogprob, "model_fields") or hasattr(
            ChatCompletionTokenLogprob, "__fields__"
        )

    def test_chat_completion_choice_logprobs_is_importable(self):
        """Test ChatCompletionChoiceLogprobs type is importable."""
        assert ChatCompletionChoiceLogprobs is not None
        assert hasattr(ChatCompletionChoiceLogprobs, "model_fields") or hasattr(
            ChatCompletionChoiceLogprobs, "__fields__"
        )

    def test_chat_completion_chunk_tool_call_function_is_importable(self):
        """Test ChatCompletionChunkToolCallFunction type is importable."""
        assert ChatCompletionChunkToolCallFunction is not None
        assert hasattr(ChatCompletionChunkToolCallFunction, "model_fields") or hasattr(
            ChatCompletionChunkToolCallFunction, "__fields__"
        )

    def test_chat_completion_chunk_tool_call_is_importable(self):
        """Test ChatCompletionChunkToolCall type is importable."""
        assert ChatCompletionChunkToolCall is not None
        assert hasattr(ChatCompletionChunkToolCall, "model_fields") or hasattr(
            ChatCompletionChunkToolCall, "__fields__"
        )

    def test_chat_completion_chunk_choice_delta_is_importable(self):
        """Test ChatCompletionChunkChoiceDelta type is importable."""
        assert ChatCompletionChunkChoiceDelta is not None
        assert hasattr(ChatCompletionChunkChoiceDelta, "model_fields") or hasattr(
            ChatCompletionChunkChoiceDelta, "__fields__"
        )

    def test_chat_completion_chunk_choice_is_importable(self):
        """Test ChatCompletionChunkChoice type is importable."""
        assert ChatCompletionChunkChoice is not None
        assert hasattr(ChatCompletionChunkChoice, "model_fields") or hasattr(
            ChatCompletionChunkChoice, "__fields__"
        )

    def test_chat_completion_chunk_is_importable(self):
        """Test ChatCompletionChunk type is importable."""
        assert ChatCompletionChunk is not None
        assert hasattr(ChatCompletionChunk, "model_fields") or hasattr(
            ChatCompletionChunk, "__fields__"
        )


class TestChatCommonReExports:
    """Test that common types are properly re-exported."""

    def test_usage_data_is_importable(self):
        """Test UsageData type is importable."""
        assert UsageData is not None
        assert hasattr(UsageData, "model_fields") or hasattr(UsageData, "__fields__")


class TestChatModuleAllExports:
    """Test that __all__ contains expected exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from venice_ai.types import chat

        assert hasattr(chat, "__all__")
        expected = [
            # Response types
            "ToolCallFunction",
            "ToolCall",
            "ChatMessage",
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
        for item in expected:
            assert item in chat.__all__, f"{item} not in __all__"

    def test_all_exports_are_accessible(self):
        """Test that all items in __all__ are accessible as attributes."""
        from venice_ai.types import chat

        for name in chat.__all__:
            assert hasattr(chat, name), f"{name} in __all__ but not accessible"
            assert getattr(chat, name) is not None


class TestChatTypeFunctionality:
    """Test basic functionality of re-exported types."""

    def test_chat_completion_response_has_expected_fields(self):
        """Test ChatCompletionResponse has expected structure."""
        # Verify it has the expected fields for a chat completion
        fields = (
            ChatCompletionResponse.model_fields
            if hasattr(ChatCompletionResponse, "model_fields")
            else ChatCompletionResponse.__fields__
        )
        # Common expected fields in chat completions
        expected_field_names = {"id", "choices"}
        for field_name in expected_field_names:
            assert field_name in fields, f"Expected field '{field_name}' not found"

    def test_chat_completion_chunk_has_expected_fields(self):
        """Test ChatCompletionChunk has expected structure for streaming."""
        # Verify it has the expected fields for a streaming chunk
        fields = (
            ChatCompletionChunk.model_fields
            if hasattr(ChatCompletionChunk, "model_fields")
            else ChatCompletionChunk.__fields__
        )
        # Common expected fields in streaming chunks
        expected_field_names = {"id", "choices"}
        for field_name in expected_field_names:
            assert field_name in fields, f"Expected field '{field_name}' not found"


# ---------------------------------------------------------------------------
# The OpenAI-compatible Responses API (Alpha) types must be exported
# from ``venice_ai.types`` (the parent package), mirroring how the api
# sub-package already exports them. The primary request/response models must
# also be reachable from top-level ``venice_ai``.
# ---------------------------------------------------------------------------

# Every Responses model that api/__init__.py re-exports, plus ResponsesRequest.
_RESPONSES_TYPES = [
    "ResponsesError",
    "ResponsesFunctionCallOutput",
    "ResponsesMessageOutput",
    "ResponsesOutputItem",
    "ResponsesOutputText",
    "ResponsesReasoningOutput",
    "ResponsesRequest",
    "ResponsesResponse",
    "ResponsesStreamEvent",
    "ResponsesUsage",
    "ResponsesUsageInputDetails",
    "ResponsesUsageOutputDetails",
    "ResponsesWebSearchCallOutput",
]


class TestResponsesTypesExportedFromTypesPackage:
    """``from venice_ai.types import Responses*`` must succeed for the family."""

    def test_all_responses_types_importable_from_types_package(self):
        import venice_ai.types as types_pkg

        missing = [name for name in _RESPONSES_TYPES if not hasattr(types_pkg, name)]
        assert not missing, f"venice_ai.types is missing Responses exports: {missing}"

    def test_all_responses_types_in_types_package_all(self):
        from venice_ai import types as types_pkg

        absent = [name for name in _RESPONSES_TYPES if name not in types_pkg.__all__]
        assert not absent, f"Responses types missing from venice_ai.types.__all__: {absent}"

    def test_responses_types_importable_via_from_import(self):
        # Exercise the literal ``from venice_ai.types import X`` statement path.
        from venice_ai.types import (  # noqa: F401
            ResponsesError,
            ResponsesFunctionCallOutput,
            ResponsesMessageOutput,
            ResponsesOutputItem,
            ResponsesOutputText,
            ResponsesReasoningOutput,
            ResponsesRequest,
            ResponsesResponse,
            ResponsesStreamEvent,
            ResponsesUsage,
            ResponsesUsageInputDetails,
            ResponsesUsageOutputDetails,
            ResponsesWebSearchCallOutput,
        )


class TestResponsesPrimariesExportedFromTopLevel:
    """The primary Responses request/response models stay reachable top-level."""

    def test_primaries_importable_from_top_level(self):
        from venice_ai import ResponsesRequest, ResponsesResponse  # noqa: F401

    def test_primaries_in_top_level_all(self):
        import venice_ai

        assert "ResponsesRequest" in venice_ai.__all__
        assert "ResponsesResponse" in venice_ai.__all__


# ---------------------------------------------------------------------------
# ChatMessage / ChatChoice / ChatUsage / LogProbToken must allow and
# preserve unmodeled (forward-compat) fields, consistent with the repo's
# extra="allow" policy, rather than silently dropping them.
# ---------------------------------------------------------------------------


class TestChatResponseModelsPreserveExtraFields:
    """Bare response DTOs must not drop server-side additions."""

    def test_chat_message_preserves_extra_field(self):
        msg = ChatMessage.model_validate(
            {"role": "assistant", "content": "hi", "future_msg_field": 1}
        )
        assert msg.model_extra is not None
        assert msg.model_extra.get("future_msg_field") == 1
        assert msg.model_dump(exclude_none=True).get("future_msg_field") == 1

    def test_chat_choice_preserves_extra_field(self):
        choice = ChatChoice.model_validate(
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
                "future_choice_field": 2,
            }
        )
        assert choice.model_extra is not None
        assert choice.model_extra.get("future_choice_field") == 2
        assert choice.model_dump(exclude_none=True).get("future_choice_field") == 2

    def test_chat_usage_preserves_extra_field(self):
        usage = ChatUsage.model_validate(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "future_usage_field": 3,
            }
        )
        assert usage.model_extra is not None
        assert usage.model_extra.get("future_usage_field") == 3
        assert usage.model_dump(exclude_none=True).get("future_usage_field") == 3

    def test_log_prob_token_preserves_extra_field(self):
        token = LogProbToken.model_validate(
            {"token": "a", "logprob": -0.1, "future_logprob_field": 4}
        )
        assert token.model_extra is not None
        assert token.model_extra.get("future_logprob_field") == 4
        assert token.model_dump(exclude_none=True).get("future_logprob_field") == 4

    def test_usage_data_preserves_extra_field(self):
        # The shared base UsageData (non-streaming usage) must
        # also preserve unmodeled live keys rather than dropping them.
        from venice_ai.types.api.base import UsageData

        usage = UsageData.model_validate(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        )
        assert usage.model_extra is not None
        assert usage.model_extra.get("completion_tokens_details") == {"reasoning_tokens": 3}
        assert usage.model_dump(exclude_none=True).get("completion_tokens_details") == {
            "reasoning_tokens": 3
        }


def test_chat_message_param_admits_response_chat_message():
    import typing

    from venice_ai.types.api import ChatMessageParam

    assert ChatMessage in typing.get_args(ChatMessageParam)
